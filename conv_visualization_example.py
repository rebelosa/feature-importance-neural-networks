"""Example of variance-based importance with a Conv2D model."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from neural_feature_importance.conv_callbacks import ConvVarianceImportanceKeras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_model() -> Sequential:
    """Return a minimal Conv2D model."""
    model = Sequential(
        [
            Conv2D(8, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def main() -> None:
    """Train model on MNIST and display a heatmap of importances."""
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train[..., None]
    y_train = to_categorical(y_train, 10)

    model = build_model()
    callback = ConvVarianceImportanceKeras()
    model.fit(x_train, y_train, epochs=2, batch_size=128, callbacks=[callback], verbose=0)

    scores = callback.feature_importances_
    if scores is None:
        logger.warning("No importance scores computed.")
        return

    weights = model.layers[0].get_weights()[0]
    heatmap = scores.reshape(weights.shape[0], weights.shape[1], weights.shape[2]).mean(axis=-1)
    plt.imshow(heatmap, cmap="hot")
    plt.colorbar()
    plt.title("Feature importance heatmap")
    plt.show()


if __name__ == "__main__":
    main()
