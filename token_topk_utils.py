"""Utilities for analyzing top-k token importances.

These helpers build a small text classification model and provide functions to
decode token sequences and print tables of the most important tokens according
to variance-based scores.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable, List

import numpy as np
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Embedding,
    GlobalMaxPooling1D,
)
from tensorflow.keras.models import Sequential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FEATURES = 5000
MAX_LEN = 400


def build_model() -> Sequential:
    """Return a small text classification model.

    The model consists of an embedding layer followed by a single convolution
    and a global max pooling operation. It is intentionally tiny so it can be
    trained quickly on the IMDB dataset for demonstration purposes.
    """
    model = Sequential(
        [
            Embedding(MAX_FEATURES, 32, input_length=MAX_LEN, trainable=True),
            Conv1D(1, 5, activation="relu"),
            GlobalMaxPooling1D(),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.build((None, MAX_LEN))
    return model


def decode_review(tokens: Iterable[int], index_word: dict[int, str]) -> str:
    """Return a readable string for the given token sequence."""
    words = [index_word.get(t, "?") for t in tokens if t]
    return " ".join(words)


def summarize_top_tokens(
    tokens: Iterable[int],
    scores: np.ndarray,
    index_word: dict[int, str],
    k: int,
) -> str:
    """Return a table with the top ``k`` unique tokens and their counts.

    Parameters
    ----------
    tokens:
        Token ids representing a review.
    scores:
        Array of token importance scores obtained from the embedding callback.
    index_word:
        Mapping from token id to the corresponding word.
    k:
        Number of tokens to include in the table.
    """
    ignore = {"<PAD>", "<START>", "<UNK>", "<UNUSED>"}
    totals: dict[int, float] = {}
    counts: Counter[int] = Counter()
    for t in tokens:
        if index_word.get(t) in ignore:
            continue
        if t < len(scores):
            totals[t] = totals.get(t, 0.0) + float(scores[t])
        counts[t] += 1
    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:k]
    headers = ["Token", "Count", "Score"]
    rows: List[tuple[str, int, str]] = []
    for token_id, score in ordered:
        token = index_word.get(token_id, "?")
        if token in ignore:
            continue
        rows.append((token, counts[token_id], f"{score:.3f}"))
    table_lines = [" | ".join(headers)]
    table_lines.append("-|-".join("-" * len(h) for h in headers))
    for row in rows:
        table_lines.append(" | ".join(str(x) for x in row))
    return "\n".join(table_lines)


def analyze_samples(
    x: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    index_word: dict[int, str],
    num_samples: int = 5,
    k: int = 5,
) -> None:
    """Log original text and the most important tokens for several samples.

    Each selected sample is decoded to text, its label is printed, and a table
    of the top ``k`` tokens by importance score is logged.
    """
    for i in range(min(num_samples, len(x))):
        tokens = x[i].tolist()
        label = "positive" if y[i] == 1 else "negative"
        text = decode_review(tokens, index_word)
        table = summarize_top_tokens(tokens, scores, index_word, k)
        logger.info("Example %d - class: %s", i, label)
        logger.info("%s", text)
        logger.info("Top tokens:\n%s", table)

