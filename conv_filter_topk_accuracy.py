"""Evaluate accuracy using top-k convolution filters."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from neural_feature_importance.conv_callbacks import ConvVarianceImportanceKeras
from conv_visualization_example import build_model, compute_filter_scores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _accuracy_with_filters(model, x, y, indices: Iterable[int]) -> float:
    """Return classification accuracy when only selected filters are active."""
    conv = model.layers[0]
    original = conv.get_weights()
    weights = original[0].copy()
    bias = original[1].copy()
    mask = np.zeros(weights.shape[-1], dtype=bool)
    mask[list(indices)] = True
    weights[..., ~mask] = 0.0
    bias[~mask] = 0.0
    conv.set_weights([weights, bias])
    preds = np.argmax(model.predict(x, verbose=0), axis=1)
    acc = float(np.mean(preds == np.argmax(y, axis=1)))
    conv.set_weights(original)
    return acc


def main() -> None:
    """Train a Conv2D model and evaluate accuracy with top filters."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = build_model()
    callback = ConvVarianceImportanceKeras()
    model.fit(x_train, y_train, epochs=2, batch_size=128, callbacks=[callback], verbose=0)

    scores = callback.feature_importances_
    if scores is None:
        logger.warning("No importance scores computed.")
        return

    weights = model.layers[0].get_weights()[0]
    heatmap = scores.reshape(weights.shape[:3])
    filter_scores, _ = compute_filter_scores(weights, heatmap, threshold=0.5)
    order = np.argsort(filter_scores)[::-1]
    logger.info("Filter scores: %s", filter_scores.tolist())

    results = {}
    for k in (2, 4, 6):
        acc = _accuracy_with_filters(model, x_test, y_test, order[:k])
        results[k] = acc
    logger.info("Accuracy with top filters: %s", results)


if __name__ == "__main__":
    main()
