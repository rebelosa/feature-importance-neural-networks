"""Callback for variance importance on embedding layers."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .callbacks import VarianceImportanceKeras

logger = logging.getLogger(__name__)


class EmbeddingVarianceImportanceKeras(VarianceImportanceKeras):
    """Track variance importance for embedding layers.

    This callback computes the variance of each embedding weight over training
    epochs and derives a token-level importance score by summing variances along
    the embedding dimension.
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


