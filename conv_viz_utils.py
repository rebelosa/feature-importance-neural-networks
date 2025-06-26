"""Helper utilities for convolutional importance visualizations."""

from __future__ import annotations

import logging
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

logger = logging.getLogger(__name__)

CMAP = mcolors.LinearSegmentedColormap.from_list("white_blue_black", ["white", "blue", "black"])
WEIGHT_CMAP = plt.cm.seismic


def build_model(input_shape: tuple[int, int, int], kernel_size: tuple[int, int]) -> Sequential:
    """Return a simple Conv2D model for visualization experiments."""
    model = Sequential(
        [
            Conv2D(8, kernel_size, activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _threshold_filters(weights: np.ndarray, threshold: float) -> np.ndarray:
    """Return weights where values below the threshold are set to zero."""
    mask = np.abs(weights) >= threshold
    return np.where(mask, weights, 0.0)


def compute_filter_scores(
    weights: np.ndarray, heatmap: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-filter scores and thresholded weights."""
    thr_weights = _threshold_filters(weights, threshold)
    scores = np.sum(np.abs(thr_weights) * heatmap[..., None], axis=(0, 1, 2))
    return scores.astype(float), thr_weights


def rank_filters(weights: np.ndarray, heatmap: np.ndarray, threshold: float) -> np.ndarray:
    """Return filter indices sorted by descending importance."""
    scores, _ = compute_filter_scores(weights, heatmap, threshold)
    order = np.argsort(scores)[::-1]
    logger.info("Filter scores: %s", scores.tolist())
    return order


def accuracy_with_filters(
    model: Sequential,
    x: np.ndarray,
    y: np.ndarray,
    indices: Iterable[int],
) -> float:
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


def plot_filters(
    weights: np.ndarray,
    heatmap: np.ndarray,
    example_out: np.ndarray,
    order: Iterable[int],
) -> None:
    """Display filter weights, importances, and outputs."""
    n_filters = weights.shape[-1]
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
