"""Training utilities for keras models."""

from __future__ import annotations

import logging
from typing import Optional

from tensorflow.keras.callbacks import Callback

logger = logging.getLogger(__name__)


class MetricThreshold(Callback):
    """Stop training when a metric exceeds a given threshold.

    Parameters
    ----------
    monitor:
        Name of the metric to monitor (e.g. ``"val_accuracy"`` or ``"loss"``).
    threshold:
        Value that the metric must reach to trigger early stopping.
    min_epochs:
        Minimum number of epochs before stopping is allowed.
    """

    def __init__(self, monitor: str = "val_accuracy", threshold: float | None = None, min_epochs: int = 5) -> None:
        super().__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.min_epochs = min_epochs
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:  # type: ignore[override]
        logs = logs or {}
        metric = logs.get(self.monitor)
        if (
            metric is not None
            and self.threshold is not None
            and epoch + 1 >= self.min_epochs
            and metric >= self.threshold
        ):
            self.stopped_epoch = epoch + 1
            self.model.stop_training = True
            logger.info(
                "MetricThreshold: stopped at epoch %d with %s=%.4f (threshold %.4f)",
                self.stopped_epoch,
                self.monitor,
                metric,
                self.threshold,
            )

    def on_train_end(self, logs: Optional[dict] = None) -> None:  # type: ignore[override]
        if self.stopped_epoch:
            logger.info("Training stopped at epoch %d", self.stopped_epoch)
