"""Model class with a convenience method for top-k predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from tensorflow.keras.models import Sequential


class PredictTopKModel(Sequential):
    """`Sequential` model extended with `predict_top_k`."""

    def predict_top_k(self, x: np.ndarray, k: int = 5) -> np.ndarray:
        """Return the indices of the top-``k`` predicted classes."""
        probs = self.predict(x, verbose=0)
        if probs.ndim == 1:
            probs = probs[:, None]
        top = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        return top
