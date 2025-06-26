"""Variance importance visualization on the digits dataset."""

from __future__ import annotations

import logging
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from neural_feature_importance.conv_callbacks import ConvVarianceImportanceKeras
from conv_visualization_example import compute_filter_scores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CMAP = mcolors.LinearSegmentedColormap.from_list("white_blue_black", ["white", "blue", "black"])
WEIGHT_CMAP = plt.cm.seismic


def build_model() -> Sequential:
    """Return a simple Conv2D model for the digits dataset."""
    model = Sequential(
        [
            Conv2D(8, (3, 3), activation="relu", input_shape=(8, 8, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _accuracy_with_filters(model: Sequential, x: np.ndarray, y: np.ndarray, indices: Iterable[int]) -> float:
    """Return accuracy when only selected filters are active."""
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
    """Train a Conv2D model on the digits dataset and plot importances."""
    digits = load_digits()
    x = digits.images[..., None] / 16.0
    y = to_categorical(digits.target, 10)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = build_model()
    callback = ConvVarianceImportanceKeras()
    model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=[callback], verbose=1)

    scores = callback.feature_importances_
    if scores is None:
        logger.warning("No importance scores computed.")
        return

    weights = model.layers[0].get_weights()[0]
    n_filters = weights.shape[-1]
    heatmap = scores.reshape(weights.shape[:3])
    filter_scores, _ = compute_filter_scores(weights, heatmap, threshold=0.0)
    order = np.argsort(filter_scores)[::-1]
    logger.info("Filter scores: %s", filter_scores.tolist())

    conv_model = Sequential([model.layers[0]])
    example_out = conv_model.predict(x_test[:1], verbose=0)[0]

    vmax = float(np.max(np.abs(weights)))
    fig, axes = plt.subplots(n_filters, 3, figsize=(9, 3 * n_filters))
    for row, idx in enumerate(order):
        ax_w = axes[row, 0]
        ax_i = axes[row, 1]
        ax_o = axes[row, 2]
        im_w = ax_w.imshow(weights[:, :, 0, idx], cmap=WEIGHT_CMAP, vmin=-vmax, vmax=vmax)
        ax_w.set_title(f"Filter {idx} weights")
        ax_w.axis("off")
        fig.colorbar(im_w, ax=ax_w)
        im_i = ax_i.imshow(heatmap[:, :, 0], cmap=CMAP, vmin=0.0, vmax=1.0)
        ax_i.set_title("Importance")
        ax_i.axis("off")
        fig.colorbar(im_i, ax=ax_i)
        im_o = ax_o.imshow(example_out[:, :, idx], cmap="gray", vmin=0.0, vmax=np.max(example_out))
        ax_o.set_title("Filter output")
        ax_o.axis("off")
        fig.colorbar(im_o, ax=ax_o)
    plt.tight_layout()
    plt.show()

    results = {}
    for k in (2, 4, 6):
        acc = _accuracy_with_filters(model, x_test, y_test, order[:k])
        results[k] = acc
    logger.info("Accuracy with top filters: %s", results)


if __name__ == "__main__":
    main()
