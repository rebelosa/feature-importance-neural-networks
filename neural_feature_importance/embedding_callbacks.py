"""Callbacks that compute variance-based importance for embedding layers.

These callbacks extend :class:`~neural_feature_importance.callbacks.VarianceImportanceBase`
to operate on 2-D embedding matrices. The variance of each embedding vector is
accumulated over training and the resulting per-token scores are normalized
between 0 and 1.
"""

from __future__ import annotations

import logging

import numpy as np

from .callbacks import VarianceImportanceKeras, VarianceImportanceTorch

logger = logging.getLogger(__name__)


class EmbeddingVarianceImportanceKeras(VarianceImportanceKeras):
    """Variance-based importance callback for Keras embedding layers.

    During training this callback monitors the weights of the first trainable
    layer (expected to be an :class:`~tensorflow.keras.layers.Embedding`) and
    accumulates the running variance of each embedding vector. After training the
    variances are summed across the embedding dimension to yield a single score
    per token.
    """

    def finalize(self) -> None:  # type: ignore[override]
        if self._last_weights is None or self._m2 is None:
            logger.warning(
                "%s was not fully initialized; no scores computed",
                self.__class__.__name__,
            )
            return

        if self._n < 2:
            variance = np.full_like(self._m2, np.nan)
        else:
            variance = self._m2 / (self._n - 1)

        scores = np.sum(variance, axis=1)
        min_val = float(np.nanmin(scores))
        max_val = float(np.nanmax(scores))
        denom = max_val - min_val if max_val != min_val else 1.0
        self.var_scores = (scores - min_val) / denom

        top = np.argsort(self.var_scores)[-10:][::-1]
        logger.info("Most important tokens: %s", top)


class EmbeddingVarianceImportanceTorch(VarianceImportanceTorch):
    """Variance-based importance for PyTorch embedding layers.

    Parameters
    ----------
    model:
        Neural network containing an :class:`torch.nn.Embedding` layer whose
        weights will be monitored.
    """

    def finalize(self) -> None:  # type: ignore[override]
        if self._last_weights is None or self._m2 is None:
            logger.warning(
                "%s was not fully initialized; no scores computed",
                self.__class__.__name__,
            )
            return

        if self._n < 2:
            variance = np.full_like(self._m2, np.nan)
        else:
            variance = self._m2 / (self._n - 1)

        scores = np.sum(variance, axis=1)
        min_val = float(np.nanmin(scores))
        max_val = float(np.nanmax(scores))
        denom = max_val - min_val if max_val != min_val else 1.0
        self.var_scores = (scores - min_val) / denom

        top = np.argsort(self.var_scores)[-10:][::-1]
        logger.info("Most important tokens: %s", top)


