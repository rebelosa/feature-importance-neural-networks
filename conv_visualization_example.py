"""Example of variance-based importance with a Conv2D model."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from neural_feature_importance.conv_callbacks import ConvVarianceImportanceKeras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Colormap for importance heatmaps
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "white_blue_black", ["white", "blue", "black"]
)
# Diverging colormap for weight visualization
WEIGHT_CMAP = plt.cm.seismic


def build_model() -> Sequential:
    """Return a minimal Conv2D model with multiple filters."""
    model = Sequential(
        [
            Conv2D(8, (8, 8), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def _threshold_filters(weights: np.ndarray, threshold: float) -> np.ndarray:
    """Return thresholded weights, keeping values above the absolute threshold."""
    thr = np.abs(weights) >= threshold
    return np.where(thr, weights, 0.0)


def compute_filter_scores(
    weights: np.ndarray, heatmap: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-filter scores and thresholded weights."""
    thr_weights = _threshold_filters(weights, threshold)
    scores = np.sum(np.abs(thr_weights) * heatmap[..., None], axis=(0, 1, 2))
    return scores.astype(float), thr_weights


def main() -> None:
    """Train model on MNIST and display a heatmap of importances."""
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train[..., None]
    y_train = to_categorical(y_train, 10)

    model = build_model()
    callback = ConvVarianceImportanceKeras()
    model.fit(
        x_train,
        y_train,
        epochs=2,
        batch_size=32,
        callbacks=[callback],
        verbose=0,
    )

    feat_scores = callback.feature_importances_
    max_var = callback.max_variance_
    if feat_scores is None or max_var is None:
        logger.warning("No importance scores computed.")
        return

    weights = model.layers[0].get_weights()[0]
    n_filters = weights.shape[-1]
    heatmap = feat_scores.reshape(weights.shape[:3])
    max_map = max_var.reshape(weights.shape[:3])
    threshold = 0.0
    filter_scores, thr_weights = compute_filter_scores(weights, heatmap, threshold)
    order = np.argsort(filter_scores)[::-1]
    logger.info("Filter scores: %s", filter_scores.tolist())

    vmax = float(np.max(np.abs(weights)))
    fig, axes = plt.subplots(n_filters, 4, figsize=(12, 3 * n_filters))
    for row, idx in enumerate(order):
        ax_w = axes[row, 0]
        ax_f = axes[row, 1]
        ax_i = axes[row, 2]
        ax_m = axes[row, 3]
        im_w = ax_w.imshow(
            weights[:, :, 0, idx], cmap=WEIGHT_CMAP, vmin=-vmax, vmax=vmax
        )
        ax_w.set_title(f"Filter {idx} weights")
        ax_w.axis("off")
        fig.colorbar(im_w, ax=ax_w)
        im_f = ax_f.imshow(
            thr_weights[:, :, 0, idx], cmap=WEIGHT_CMAP, vmin=-vmax, vmax=vmax
        )
        ax_f.set_title("Thresholded")
        ax_f.axis("off")
        fig.colorbar(im_f, ax=ax_f)
        im = ax_i.imshow(heatmap[:, :, 0], cmap=CMAP, vmin=0.0, vmax=1.0)
        ax_i.set_title("Importance")
        ax_i.axis("off")
        fig.colorbar(im, ax=ax_i)
        im2 = ax_m.imshow(max_map[:, :, 0], cmap=CMAP, vmin=0.0, vmax=1.0)
        ax_m.set_title("Max variance")
        ax_m.axis("off")
        fig.colorbar(im2, ax=ax_m)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
