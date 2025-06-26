"""Visualize variance-based filter importances on several datasets."""

from __future__ import annotations

import logging
from typing import Callable, Tuple

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential

from neural_feature_importance.conv_callbacks import ConvVarianceImportanceKeras
from conv_viz_utils import build_model, rank_filters, plot_filters, accuracy_with_filters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DatasetLoader = Callable[
    [],
    Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        tuple[int, int, int],
        tuple[int, int],
    ],
]


def load_mnist() -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    tuple[int, int, int],
    tuple[int, int],
]:
    """Return MNIST data and model parameters."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test), (28, 28, 1), (8, 8)


def load_digits_data() -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    tuple[int, int, int],
    tuple[int, int],
]:
    """Return scikit-learn digits data and model parameters."""
    digits = load_digits()
    x = digits.images[..., None] / 16.0
    y = to_categorical(digits.target, 10)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    return (x_train, y_train), (x_test, y_test), (8, 8, 1), (3, 3)


DATASETS: dict[str, DatasetLoader] = {
    "mnist": load_mnist,
    "digits": load_digits_data,
}


def run_dataset(name: str, loader: DatasetLoader, threshold: float = 0.0) -> None:
    """Train a model on the given dataset and display filter importances."""
    (x_train, y_train), (x_test, y_test), input_shape, kernel_size = loader()

    model = build_model(input_shape, kernel_size)
    callback = ConvVarianceImportanceKeras()
    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        callbacks=[callback],
        verbose=1,
    )

    scores = callback.feature_importances_
    if scores is None:
        logger.warning("No importance scores computed for %s", name)
        return

    weights = model.layers[0].get_weights()[0]
    heatmap = scores.reshape(weights.shape[:3])
    order = rank_filters(weights, heatmap, threshold)

    conv_model = Sequential([model.layers[0]])
    example_out = conv_model.predict(x_test[:1], verbose=0)[0]
    plot_filters(weights, heatmap, example_out, order)

    results = {}
    for k in (2, 4, 6):
        acc = accuracy_with_filters(model, x_test, y_test, order[:k])
        results[k] = acc
    logger.info("Accuracy with top filters on %s: %s", name, results)


def main() -> None:
    """Run visualization on all configured datasets."""
    for name, loader in DATASETS.items():
        logger.info("Running visualization for %s", name)
        run_dataset(name, loader)


if __name__ == "__main__":
    main()
