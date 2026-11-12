"""Train a CNN on the MNIST dataset and export it to models/mnist_cnn.h5."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


MODEL_PATH = Path(__file__).resolve().parent / "models" / "mnist_cnn.h5"


def build_mnist_cnn() -> tf.keras.Model:
    """Build the 2-block Conv2D + Dense CNN matching models/mnist_cnn.h5."""
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(epochs: int = 8, batch_size: int = 128) -> None:
    """Load MNIST, train the CNN, evaluate test accuracy, and save weights."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    model = build_mnist_cnn()
    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc * 100:.2f}%")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
