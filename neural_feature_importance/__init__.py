"""Utilities for variance-based feature importance in neural networks."""

from importlib import metadata

from .callbacks import (
    VarianceImportanceBase,
    VarianceImportanceKeras,
    VarianceImportanceTorch,
)
from .utils import MetricThreshold

try:
    __version__ = metadata.version("neural-feature-importance")
except metadata.PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.dev0"

__all__ = [
    "VarianceImportanceBase",
    "VarianceImportanceKeras",
    "VarianceImportanceTorch",
    "MetricThreshold",
]
