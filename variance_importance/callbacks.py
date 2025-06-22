"""Callback for computing variance-based feature importance.

This module provides :class:`VarianceImportanceCallback`, a Keras callback that
tracks the evolution of the first layer's weights during training. By measuring
the running variance of each weight and combining it with the final magnitude of
the weights, the callback derives a relative importance score for each input
feature.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from tensorflow.keras.callbacks import Callback

logger = logging.getLogger(__name__)

class VarianceImportanceCallback(Callback):
    """Compute variance-based feature importance for ``tf.keras`` models.

    Parameters
    ----------
    verbose : int, optional
        Verbosity mode. When ``> 0`` the callback reports basic information
        during training.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__()
        self.verbose = verbose
        self.n: int = 0
        self.M2: np.ndarray | float = 0.0
        self.diff: np.ndarray | None = None
        self.last_weights: np.ndarray | None = None
        self.var_scores: np.ndarray | None = None
        self._layer = None

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        """Initialize running statistics at the start of training."""
        if self.verbose:
            logger.info("VIANN version 1.0 (Welford + Mean) update per epoch")

        self._layer = self.model.layers[0]
        self.diff = self._layer.get_weights()[0].astype(np.float64)
        self.M2 = np.zeros_like(self.diff, dtype=np.float64)
        self.n = 0

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Update running variance after each epoch."""
        current_weights = self._layer.get_weights()[0]

        self.n += 1
        delta = current_weights - self.diff
        self.diff += delta / self.n
        delta2 = current_weights - self.diff
        self.M2 += delta * delta2

        self.last_weights = current_weights

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """Finalize variance statistics and compute importance scores."""
        if self.n < 2:
            s2 = np.full_like(self.M2, np.nan)
        else:
            s2 = self.M2 / (self.n - 1)

        scores = np.sum(s2 * np.abs(self.last_weights), axis=1)
        max_val = float(np.max(scores))
        min_val = float(np.min(scores))
        denom = max_val - min_val if max_val != min_val else 1.0
        self.var_scores = (scores - min_val) / denom

        if self.verbose:
            top = np.argsort(self.var_scores)[-10:][::-1]
            logger.info("Most important variables: %s", top)
