"""Compare feature importance methods using the Iris dataset."""

from __future__ import annotations

import logging
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neural_feature_importance import (
    VarianceImportanceKeras,
    VarianceImportanceTorch,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_model(input_dim: int, num_classes: int) -> Sequential:
    """Return a simple feed-forward neural network."""
    model = Sequential(
        [
            Dense(16, input_dim=input_dim, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_torch_model(input_dim: int, num_classes: int) -> nn.Module:
    """Return a simple feed-forward PyTorch network."""
    return nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.ReLU(),
        nn.Linear(16, num_classes),
    )


def main() -> None:
    """Train models and compare feature importances."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    num_classes = len(np.unique(data.target))
    y_train_cat = to_categorical(y_train, num_classes)

    model = build_model(data.data.shape[1], num_classes)
    callback = VarianceImportanceKeras()
    model.fit(
        X_train,
        y_train_cat,
        epochs=50,
        batch_size=16,
        verbose=0,
        callbacks=[callback],
    )

    nn_scores = callback.var_scores
    logger.info("NN feature importances: %s", nn_scores)

    # Train a PyTorch model using the same data
    torch_model = build_torch_model(data.data.shape[1], num_classes)
    optimizer = torch.optim.Adam(torch_model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    tracker = VarianceImportanceTorch(torch_model)
    tracker.on_train_begin()
    for _ in range(50):
        torch_model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            out = torch_model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
        tracker.on_epoch_end()
    tracker.on_train_end()
    torch_scores = tracker.var_scores
    logger.info("PyTorch feature importances: %s", torch_scores)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_scores = rf.feature_importances_
    logger.info("Random forest importances: %s", rf_scores)

    if nn_scores is not None:
        corr = np.corrcoef(nn_scores, rf_scores)[0, 1]
        logger.info("Correlation between Keras and RF: %.3f", corr)

    if torch_scores is not None:
        corr_torch = np.corrcoef(torch_scores, rf_scores)[0, 1]
        logger.info("Correlation between PyTorch and RF: %.3f", corr_torch)


if __name__ == "__main__":
    main()
