"""Train a Conv2D model, visualize filter importances, and predict top classes."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Iterable

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

from neural_feature_importance.conv_callbacks import ConvVarianceImportanceKeras
from neural_feature_importance.predict_top_k import PredictTopKModel
from conv_visualization_example import compute_filter_scores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CMAP = mcolors.LinearSegmentedColormap.from_list(
    "white_blue_black", ["white", "blue", "black"]
)


def _accuracy_with_filters(model, x, y, indices: Iterable[int]) -> float:
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


def build_model() -> PredictTopKModel:
    """Return a Conv2D model with the `predict_top_k` method."""
    model = PredictTopKModel(
        [
            Conv2D(8, (8, 8), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    """Train the model, plot importances, and show predictions."""
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
    n_filters = weights.shape[-1]
    heatmap = scores.reshape(weights.shape[:3])
    threshold = 0.5
    filter_scores, thr_weights = compute_filter_scores(weights, heatmap, threshold)
    order = np.argsort(filter_scores)[::-1]

    fig, axes = plt.subplots(n_filters, 3, figsize=(9, 3 * n_filters))
    for row, idx in enumerate(order):
        ax_w = axes[row, 0]
        ax_f = axes[row, 1]
        ax_i = axes[row, 2]
        ax_w.imshow(weights[:, :, 0, idx], cmap="gray")
        ax_w.set_title(f"Filter {idx} weights")
        ax_w.axis("off")
        ax_f.imshow(thr_weights[:, :, 0, idx], cmap="gray_r")
        ax_f.set_title("Thresholded")
        ax_f.axis("off")
        im = ax_i.imshow(heatmap[:, :, 0], cmap=CMAP, vmin=0.0, vmax=1.0)
        ax_i.set_title("Importance")
        ax_i.axis("off")
        fig.colorbar(im, ax=ax_i)

    plt.tight_layout()
    plt.show()

    results = {}
    for k in (2, 4, 6):
        acc = _accuracy_with_filters(model, x_test, y_test, order[:k])
        results[k] = acc
    logger.info("Accuracy with top filters: %s", results)

    top_preds = model.predict_top_k(x_train[:5], k=3)
    logger.info("Top-3 predictions for first 5 samples: %s", top_preds.tolist())


if __name__ == "__main__":
    main()
