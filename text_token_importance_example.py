"""Visualize per-token importances on a single text sample."""

from __future__ import annotations

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Conv1D, Dense, Embedding, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

from neural_feature_importance.embedding_callbacks import (
    EmbeddingVarianceImportanceKeras,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FEATURES = 5000
MAX_LEN = 400


def build_model() -> Sequential:
    """Return a small text classification model."""
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


def decode_review(tokens: List[int], index_word: dict[int, str]) -> str:
    """Return a readable string for the given token sequence."""
    words = [index_word.get(t, "?") for t in tokens if t]
    return " ".join(words)


def main() -> None:
    """Train a model and display token importances for one sample."""
    (x_train, y_train), _ = imdb.load_data(num_words=MAX_FEATURES)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN)
    model = build_model()
    callback = EmbeddingVarianceImportanceKeras()
    # At least two epochs are required so variance scores are meaningful
    model.fit(x_train, y_train, epochs=2, batch_size=128, callbacks=[callback], verbose=0)

    scores = callback.feature_importances_
    if scores is None:
        logger.warning("No importance scores computed.")
        return

    # Map scores back to tokens
    word_index = imdb.get_word_index()
    index_word = {v + 3: k for k, v in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"
    index_word[3] = "<UNUSED>"

    sample = x_train[0]
    token_scores = [scores[t] if t < len(scores) else 0.0 for t in sample]
    words = [index_word.get(t, "?") for t in sample]

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.bar(range(len(words)), token_scores, color="steelblue")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Importance")
    ax.set_title("Token importances for sample 0")
    plt.tight_layout()
    plt.show()

    logger.info("Sample text: %s", decode_review(sample.tolist(), index_word))


if __name__ == "__main__":
    main()

