"""FastAPI backend for MNIST Handwritten Digit Classifier."""

import base64
import binascii
import io
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mnist_mpl_cache")
)

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
from pydantic import BaseModel
from scipy import ndimage


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

DEFAULT_MODEL_PATH = MODELS_DIR / "mnist_cnn.h5"
FALLBACK_MODEL_PATH = BASE_DIR / "mnist_cnn.h5"


# ==========================================
# MODEL MANAGEMENT
# ==========================================

_model_lock = threading.Lock()

model: Optional[tf.keras.Model] = None
loaded_model_path: Optional[Path] = None
model_load_attempted = False
model_load_error: Optional[str] = None


def resolve_model_path() -> Optional[Path]:

    if DEFAULT_MODEL_PATH.is_file():
        return DEFAULT_MODEL_PATH

    if FALLBACK_MODEL_PATH.is_file():
        return FALLBACK_MODEL_PATH

    if MODELS_DIR.exists():

        for extension in ("*.h5", "*.keras"):

            matches = sorted(MODELS_DIR.glob(extension))

            if matches:
                return matches[0]

    return None


def load_cnn_model() -> Optional[tf.keras.Model]:

    global model
    global loaded_model_path
    global model_load_attempted
    global model_load_error

    if model is not None:
        return model

    with _model_lock:

        if model is not None:
            return model

        if model_load_attempted:
            return None

        model_load_attempted = True

        try:

            path = resolve_model_path()

            if path is None:
                model_load_error = "Model file not found."
                print(model_load_error)
                return None

            print(f"Loading model: {path.name}")

            loaded = tf.keras.models.load_model(
                path,
                compile=False
            )

            # Warm up model

            loaded(
                np.zeros((1, 28, 28, 1), dtype=np.float32),
                training=False
            )

            model = loaded
            loaded_model_path = path
            model_load_error = None

            print("CNN model loaded successfully.")

        except Exception as exc:

            model_load_error = type(exc).__name__

            print(f"Model loading failed: {exc}")

            return None

    return model


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Handwritten Digit Classifier",
    description="MNIST CNN inference API using TensorFlow.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# STATIC FILES AND TEMPLATES
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


# ==========================================
# REQUEST MODEL
# ==========================================

class PredictRequest(BaseModel):
    image: str


# ==========================================
# IMAGE DECODING
# ==========================================

def decode_base64_image(data_url: str) -> Image.Image:

    raw = data_url.strip()

    if "," in raw:
        _, raw = raw.split(",", 1)

    try:

        image_bytes = base64.b64decode(
            raw,
            validate=True
        )

    except (binascii.Error, ValueError) as exc:

        raise HTTPException(
            status_code=400,
            detail="Invalid base64 image payload."
        ) from exc

    try:

        img = Image.open(io.BytesIO(image_bytes))
        img.load()

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail="Unable to decode image."
        ) from exc

    # Handle transparent backgrounds

    if img.mode in ("RGBA", "LA") or "transparency" in img.info:

        rgba = img.convert("RGBA")

        white_bg = Image.new(
            "RGBA",
            rgba.size,
            (255, 255, 255, 255)
        )

        img = Image.alpha_composite(
            white_bg,
            rgba
        ).convert("L")

    else:

        img = img.convert("L")

    return img


# ==========================================
# MNIST IMAGE PREPROCESSING
# ==========================================

def preprocess_mnist(img: Image.Image) -> Optional[np.ndarray]:

    gray = np.asarray(
        img,
        dtype=np.float32
    )

    inverted = 255.0 - gray

    if inverted.max() < 15.0:
        return None

    foreground_mask = inverted > 20.0

    if not np.any(foreground_mask):
        return None

    ys, xs = np.where(foreground_mask)

    crop = inverted[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    crop_h, crop_w = crop.shape

    scale = 20.0 / max(crop_h, crop_w)

    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    resized = Image.fromarray(
        crop.astype(np.uint8)
    ).resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    frame = np.zeros(
        (28, 28),
        dtype=np.float32
    )

    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    frame[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = np.asarray(
        resized,
        dtype=np.float32
    )

    # Center digit using center of mass

    cy, cx = ndimage.center_of_mass(frame)

    if not (np.isnan(cy) or np.isnan(cx)):

        frame = ndimage.shift(
            frame,
            shift=(
                14.0 - float(cy),
                14.0 - float(cx)
            ),
            mode="constant",
            cval=0.0
        )

    normalized = np.clip(
        frame / 255.0,
        0.0,
        1.0
    )

    return normalized.reshape(
        1, 28, 28, 1
    )


# ==========================================
# HOME ROUTE
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
async def health():

    active_model = load_cnn_model()

    return {
        "status": "ok" if active_model else "degraded",
        "model_loaded": active_model is not None,
        "model_file": (
            loaded_model_path.name
            if loaded_model_path
            else None
        ),
        "error": model_load_error
    }


# ==========================================
# PREDICTION API
# ==========================================

@app.post("/predict")
def predict(payload: PredictRequest):

    active_model = load_cnn_model()

    if active_model is None:

        raise HTTPException(
            status_code=503,
            detail="CNN model is unavailable."
        )

    start = time.perf_counter()

    img = decode_base64_image(payload.image)

    tensor = preprocess_mnist(img)

    if tensor is None:

        return JSONResponse({
            "blank": True,
            "predicted_digit": None,
            "confidence": 0.0,
            "probabilities": [0.0] * 10,
            "inference_ms": round(
                (time.perf_counter() - start) * 1000,
                2
            )
        })

    probs = active_model(
        tensor,
        training=False
    ).numpy()[0]

    predicted_digit = int(np.argmax(probs))

    confidence = float(probs[predicted_digit])

    return {
        "blank": False,
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": [
            float(p) for p in probs
        ],
        "inference_ms": round(
            (time.perf_counter() - start) * 1000,
            2
        )
    }


# ==========================================
# LOCAL DEVELOPMENT
# ==========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
