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

from variance_importance import VarianceImportanceCallback

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


def main() -> None:
    """Train models and compare feature importances."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    num_classes = len(np.unique(data.target))
    y_train_cat = to_categorical(y_train, num_classes)

    model = build_model(data.data.shape[1], num_classes)
    callback = VarianceImportanceCallback()
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

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_scores = rf.feature_importances_
    logger.info("Random forest importances: %s", rf_scores)

    if nn_scores is not None:
        corr = np.corrcoef(nn_scores, rf_scores)[0, 1]
        logger.info("Correlation between methods: %.3f", corr)


if __name__ == "__main__":
    main()
