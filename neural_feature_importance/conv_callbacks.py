"""Callbacks with convolutional layer support."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .callbacks import VarianceImportanceKeras, VarianceImportanceTorch

logger = logging.getLogger(__name__)


def _flatten_weights(weights: np.ndarray, outputs_last: bool) -> np.ndarray:
    """Return a 2-D array with shape (features, outputs)."""
    if weights.ndim > 2:
        if outputs_last:
            return weights.reshape(-1, weights.shape[-1])
        return weights.reshape(weights.shape[0], -1).T
    return weights


class ConvVarianceImportanceKeras(VarianceImportanceKeras):
    """Keras callback supporting convolutional layers."""

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        self._layer = None
        for layer in self.model.layers:
            has_vars = bool(layer.trainable_weights)
            has_data = bool(layer.get_weights())
            if has_vars and has_data:
                self._layer = layer
                break
        if self._layer is None:
            raise ValueError("Model does not contain trainable weights.")
        weights = self._layer.get_weights()[0]
        weights = _flatten_weights(weights, outputs_last=True)
        logger.info(
            "Tracking variance for layer '%s' with %d features",
            self._layer.name,
            weights.shape[0],
        )
        self.start(weights)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if self._layer is None:
            return
        weights = self._layer.get_weights()[0]
        weights = _flatten_weights(weights, outputs_last=True)
        self.update(weights)


class ConvVarianceImportanceTorch(VarianceImportanceTorch):
    """PyTorch helper supporting convolutional layers."""

    def on_train_begin(self) -> None:
        from torch import nn

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                self._param = param
                weights = param.detach().cpu().numpy()
                weights = _flatten_weights(weights, outputs_last=False)
                logger.info(
                    "Tracking variance for parameter '%s' with %d features",
                    name,
                    weights.shape[0],
                )
                self.start(weights)
                break
        if self._param is None:
            raise ValueError("Model does not contain trainable parameters")

    def on_epoch_end(self) -> None:
        if self._param is None:
            return
        weights = self._param.detach().cpu().numpy()
        weights = _flatten_weights(weights, outputs_last=False)
        self.update(weights)
