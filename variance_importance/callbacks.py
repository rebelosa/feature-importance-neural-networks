"""Variance-based feature importance callback.

This module implements :class:`VarianceImportanceCallback`, a small utility
that can be plugged into a ``tf.keras`` training loop.  The callback tracks the
evolution of the first layer's weights using Welford's online algorithm and,
after training, derives a normalized importance score for each input feature.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer

logger = logging.getLogger(__name__)

class VarianceImportanceCallback(Callback):
    """Compute variance-based feature importance for ``tf.keras`` models.

    Parameters
    ----------
    None currently.
    """

    def __init__(self) -> None:
        super().__init__()

        self._n: int = 0
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._layer: Optional[Layer] = None
        self._last_weights: np.ndarray | None = None

        self.var_scores: np.ndarray | None = None

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        """Initialize running statistics at the start of training."""
        self._layer = self.model.layers[0]
        weights = self._layer.get_weights()
        if not weights:
            raise ValueError("First layer does not contain weights.")

        logger.info(
            "Tracking variance for layer '%s' with %d features",
            self._layer.name,
            weights[0].shape[0],
        )

        self._mean = weights[0].astype(np.float64)
        self._m2 = np.zeros_like(self._mean)
        self._n = 0

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Update running variance after each epoch."""
        if self._layer is None or self._mean is None or self._m2 is None:
            return

        current_weights = self._layer.get_weights()[0]

        self._n += 1
        delta = current_weights - self._mean
        self._mean += delta / self._n
        delta2 = current_weights - self._mean
        self._m2 += delta * delta2

        self._last_weights = current_weights

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """Finalize variance statistics and compute importance scores."""
        if self._last_weights is None or self._m2 is None:
            logger.warning(
                "VarianceImportanceCallback was not fully initialized; no scores computed"
            )
            return

        if self._n < 2:
            variance = np.full_like(self._m2, np.nan)
        else:
            variance = self._m2 / (self._n - 1)

        scores = np.sum(variance * np.abs(self._last_weights), axis=1)
        min_val = float(np.min(scores))
        max_val = float(np.max(scores))
        denom = max_val - min_val if max_val != min_val else 1.0
        self.var_scores = (scores - min_val) / denom

        top = np.argsort(self.var_scores)[-10:][::-1]
        logger.info("Most important variables: %s", top)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Normalized importance scores for each input feature."""
        return self.var_scores

    def get_config(self) -> dict[str, int]:
        """Return configuration for serialization."""
        return {}
