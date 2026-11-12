# Handwritten Digit Classifier (MNIST CNN)

Interactive handwritten digit recognition web application powered by a TensorFlow/Keras Convolutional Neural Network, FastAPI backend, and responsive HTML5 Canvas frontend.

## Project Structure

```text
mnist/
├── models/
│   └── mnist_cnn.h5       # Trained TensorFlow/Keras CNN weights
├── static/
│   ├── css/
│   │   └── style.css      # Dark glassmorphic & mobile-responsive styling
│   └── js/
│       └── app.js         # Smooth PointerEvents canvas + real-time inference client
├── templates/
│   └── index.html         # FastAPI Jinja2 UI template
├── .gitignore
├── LICENSE
├── README.md
├── app.py                 # FastAPI server & MNIST preprocessing pipeline
├── requirements.txt
└── train_mnist.py         # CNN training & evaluation script
```

## Features

- **Real-time & Manual Inference**: Toggle live predictions as you draw, or switch to manual mode and trigger inference with **Predict** (`Enter`).
- **Smooth HiDPI Drawing Pad**: Pointer-event stroke interpolation works seamlessly across mouse, stylus, and mobile touchscreens without accidental page scrolling.
- **MNIST-Aligned Preprocessing**: Automatically crops, preserves aspect ratio within a 20×20 box, centers by center-of-mass inside 28×28, and normalizes to `[0, 1]`.
- **Brush Size & Undo History**: Adjustable brush thickness (`8px`–`40px`) with live preview dot, stroke-by-stroke **Undo** (`Z`), and **Clear** (`C`).
- **Softmax Distribution**: Live confidence bar and per-digit probability distribution (`0`–`9`).

## Quick Start

1. Install dependencies (optional if already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. Run the FastAPI server:
   ```bash
   python app.py
   ```
   Or with Uvicorn:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. Open `http://127.0.0.1:8000` in your browser (or your machine's local IP on port `8000` from your phone).

## Running from VS Code

1. Open the `mnist` folder in VS Code (`File` → `Open Folder...`).
2. Press **`Ctrl + ~`** (or `Terminal` → `New Terminal`) to open the integrated terminal.
3. Run:
   ```bash
   python app.py
   ```
   *(Or open `app.py` and click the **▶ Run Python File** button, or press **`F5`** using the included `.vscode/launch.json`.)*
4. Once the terminal prints `Open the app at: http://127.0.0.1:8000`, `Ctrl + Click` that URL (do **not** use VS Code Live Server on `:3000`/`5500` alone without starting `app.py`, since Live Server does not run the Python model).

## API Endpoints

- `GET /` — Serves the interactive classifier interface.
- `GET /health` — Reports server and model readiness.
- `POST /predict` — Accepts `{"image": "<base64 data URL>"}` and returns the predicted digit, confidence, and 10-class softmax probabilities.
