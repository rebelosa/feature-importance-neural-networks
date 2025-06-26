"""Evaluate accuracy with combined convolutional filters."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from neural_feature_importance.conv_callbacks import ConvVarianceImportanceKeras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_model() -> Sequential:
    """Return a Conv2D model with ten filters."""
    model = Sequential(
        [
            Conv2D(10, (8, 8), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _threshold_filters(weights: np.ndarray, threshold: float) -> np.ndarray:
    """Return thresholded weights, keeping only values above the threshold."""
    mask = np.abs(weights) >= threshold
    return np.where(mask, weights, 0.0)


def compute_filter_scores(
    weights: np.ndarray, heatmap: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-filter scores and thresholded weights."""
    thr_weights = _threshold_filters(weights, threshold)
    scores = np.sum(np.abs(thr_weights) * heatmap[..., None], axis=(0, 1, 2))
    return scores.astype(float), thr_weights


def _accuracy_with_combined_filter(
    model: Sequential,
    x: np.ndarray,
    y: np.ndarray,
    top: Iterable[int],
    combine: Iterable[int],
) -> float:
    """Return accuracy using selected filters and one combined filter."""
    conv = model.layers[0]
    original = conv.get_weights()
    weights = original[0].copy()
    bias = original[1].copy()

    weights[..., :] = 0.0
    bias[:] = 0.0

    top = list(top)
    combine = list(combine)

    for idx in top:
        weights[..., idx] = original[0][..., idx]
        bias[idx] = original[1][idx]

    if combine:
        comb_w = np.mean(original[0][..., combine], axis=-1)
        comb_b = float(np.mean(original[1][combine]))
        target = combine[0]
        weights[..., target] = comb_w
        bias[target] = comb_b

    conv.set_weights([weights, bias])
    preds = np.argmax(model.predict(x, verbose=0), axis=1)
    acc = float(np.mean(preds == np.argmax(y, axis=1)))
    conv.set_weights(original)
    return acc


def main() -> None:
    """Train the model and evaluate accuracy with combined filters."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = build_model()
    callback = ConvVarianceImportanceKeras()
    model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=[callback], verbose=1)

    scores = callback.feature_importances_
    if scores is None:
        logger.warning("No importance scores computed.")
        return

    weights = model.layers[0].get_weights()[0]
    heatmap = scores.reshape(weights.shape[:3])
    filter_scores, _ = compute_filter_scores(weights, heatmap, threshold=0.0)
    order = np.argsort(filter_scores)[::-1]
    logger.info("Filter scores: %s", filter_scores.tolist())

    top5 = order[:5]
    rest = order[5:]
    acc = _accuracy_with_combined_filter(model, x_test, y_test, top5, rest)
    logger.info("Accuracy with combined filter setup: %.4f", acc)


if __name__ == "__main__":
    main()
