"""Example of variance-based importance with text classification."""

from __future__ import annotations

import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Conv1D, Dense, Embedding, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

from neural_feature_importance.conv_callbacks import ConvVarianceImportanceKeras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MAX_FEATURES = 5000
MAX_LEN = 400


def load_data() -> Tuple[tuple, tuple]:
    """Return padded IMDB data."""
    (x_train, y_train), _ = imdb.load_data(num_words=MAX_FEATURES)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN)
    return (x_train, y_train), _


def build_model() -> Sequential:
    """Return a small Conv1D model."""
    model = Sequential(
        [
            Embedding(MAX_FEATURES, 32, input_length=MAX_LEN, trainable=True),
            Conv1D(1, 5, activation="relu"),
            GlobalMaxPooling1D(),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.build((None, MAX_LEN))
    return model


def main() -> None:
    """Train the model and plot a heatmap of importances."""
    (x_train, y_train), _ = load_data()
    model = build_model()
    callback = ConvVarianceImportanceKeras()
    model.fit(x_train, y_train, epochs=2, batch_size=128, callbacks=[callback], verbose=0)

    scores = callback.feature_importances_
    if scores is None:
        logger.warning("No importance scores computed.")
        return

    weights = model.layers[1].get_weights()[0]
    heatmap = scores.reshape(weights.shape[0], weights.shape[1])
    plt.imshow(heatmap, aspect="auto", cmap="hot")
    plt.colorbar()
    plt.title("Conv1D feature importance")
    plt.show()



if __name__ == "__main__":
    main()
