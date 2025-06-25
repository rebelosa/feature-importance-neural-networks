"""Train a Conv2D model, visualize filter importances, and predict top classes."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

from neural_feature_importance.conv_callbacks import ConvVarianceImportanceKeras
from neural_feature_importance.predict_top_k import PredictTopKModel
from conv_visualization_example import compute_filter_scores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_model() -> PredictTopKModel:
    """Return a Conv2D model with the `predict_top_k` method."""
    model = PredictTopKModel(
        [
            Conv2D(8, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    """Train the model, plot importances, and show predictions."""
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
    n_filters = weights.shape[-1]
    heatmap = scores.reshape(weights.shape[:3])
    threshold = 0.5
    filter_scores, masks = compute_filter_scores(weights, heatmap, threshold)
    order = np.argsort(filter_scores)[::-1]

    fig, axes = plt.subplots(n_filters, 3, figsize=(9, 3 * n_filters))
    for row, idx in enumerate(order):
        ax_w = axes[row, 0]
        ax_f = axes[row, 1]
        ax_i = axes[row, 2]
        ax_w.imshow(weights[:, :, 0, idx], cmap="gray")
        ax_w.set_title(f"Filter {idx} weights")
        ax_w.axis("off")
        ax_f.imshow(masks[:, :, 0, idx], cmap="gray_r")
        ax_f.set_title("Thresholded")
        ax_f.axis("off")
        im = ax_i.imshow(heatmap[:, :, 0], cmap="gray_r")
        ax_i.set_title("Importance")
        ax_i.axis("off")
        fig.colorbar(im, ax=ax_i)

    plt.tight_layout()
    plt.show()

    top_preds = model.predict_top_k(x_train[:5], k=3)
    logger.info("Top-3 predictions for first 5 samples: %s", top_preds.tolist())


if __name__ == "__main__":
    main()
