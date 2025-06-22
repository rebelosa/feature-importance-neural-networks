"""Variance-based feature importance utilities."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer

logger = logging.getLogger(__name__)


class VarianceImportanceBase:
    """Compute feature importance using Welford's algorithm."""

    def __init__(self) -> None:
        self._n = 0
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._last_weights: np.ndarray | None = None
        self.var_scores: np.ndarray | None = None

    def start(self, weights: np.ndarray) -> None:
        """Initialize statistics for the given weight matrix."""
        self._mean = weights.astype(np.float64)
        self._m2 = np.zeros_like(self._mean)
        self._n = 0

    def update(self, weights: np.ndarray) -> None:
        """Update running statistics with new weights."""
        if self._mean is None or self._m2 is None:
            return
        self._n += 1
        delta = weights - self._mean
        self._mean += delta / self._n
        delta2 = weights - self._mean
        self._m2 += delta * delta2
        self._last_weights = weights

    def finalize(self) -> None:
        """Finalize statistics and compute normalized scores."""
        if self._last_weights is None or self._m2 is None:
            logger.warning(
                "%s was not fully initialized; no scores computed", self.__class__.__name__
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


class VarianceImportanceKeras(Callback, VarianceImportanceBase):
    """Keras callback implementing variance-based feature importance."""

    def __init__(self) -> None:
        Callback.__init__(self)
        VarianceImportanceBase.__init__(self)
        self._layer: Optional[Layer] = None

    def on_train_begin(self, logs: Optional[dict] = None) -> None:  # type: ignore[override]
        self._layer = None
        for layer in self.model.layers:
            if layer.get_weights():
                self._layer = layer
                break
        if self._layer is None:
            raise ValueError("Model does not contain trainable weights.")
        weights = self._layer.get_weights()[0]
        logger.info(
            "Tracking variance for layer '%s' with %d features",
            self._layer.name,
            weights.shape[0],
        )
        self.start(weights)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:  # type: ignore[override]
        if self._layer is None:
            return
        weights = self._layer.get_weights()[0]
        self.update(weights)

    def on_train_end(self, logs: Optional[dict] = None) -> None:  # type: ignore[override]
        self.finalize()

    def get_config(self) -> dict[str, int]:
        """Return configuration for serialization."""
        return {}


class VarianceImportanceTorch(VarianceImportanceBase):
    """Track variance-based feature importance for PyTorch models."""

    def __init__(self, model: "nn.Module") -> None:
        from torch import nn  # Local import to avoid hard dependency

        super().__init__()
        self.model = model
        self._param: nn.Parameter | None = None

    def on_train_begin(self) -> None:
        from torch import nn

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                self._param = param
                weights = param.detach().cpu().numpy()
                logger.info(
                    "Tracking variance for parameter '%s' with %d features",
                    name,
                    weights.shape[1],
                )
                self.start(weights)
                break
        if self._param is None:
            raise ValueError("Model does not contain trainable parameters")

    def on_epoch_end(self) -> None:
        if self._param is None:
            return
        weights = self._param.detach().cpu().numpy()
        self.update(weights)

    def on_train_end(self) -> None:
        self.finalize()
