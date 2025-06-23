"""Replicate the notebook experiment using the reusable callbacks."""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

from neural_feature_importance import VarianceImportanceKeras
from neural_feature_importance.utils import MetricThreshold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def nn2(input_dim: int, output_dim: int, classification: bool) -> Sequential:
    """Return a small feed-forward network."""
    model = Sequential(
        [
            Dense(50, input_dim=input_dim, activation="relu", kernel_regularizer=l2(0.01)),
            Dense(100, activation="relu", kernel_regularizer=l2(0.01)),
            Dense(50, activation="relu", kernel_regularizer=l2(0.01)),
        ]
    )
    if classification:
        model.add(Dense(output_dim, activation="softmax"))
        model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    else:
        model.add(Dense(1))
        model.compile(optimizer="sgd", loss="mean_squared_error")
    return model


def run_experiment() -> None:
    """Train on several datasets and compare with random forest importances."""
    datasets_cfg: List[dict] = [
        {"name": "breastcancer", "classification": True, "data": datasets.load_breast_cancer()},
        {"name": "digits", "classification": True, "data": datasets.load_digits()},
        {"name": "iris", "classification": True, "data": datasets.load_iris()},
        {"name": "wine", "classification": True, "data": datasets.load_wine()},
        {"name": "boston", "classification": False, "data": datasets.load_boston()},
        {"name": "diabetes", "classification": False, "data": datasets.load_diabetes()},
    ]

    for cfg in datasets_cfg:
        ds = cfg["data"]
        X = scale(ds.data)
        if cfg["classification"]:
            enc = LabelEncoder()
            y_enc = enc.fit_transform(ds.target)
            Y = to_categorical(y_enc)
            output_size = Y.shape[1]
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X, ds.target)
            monitor = MetricThreshold(monitor="val_accuracy", threshold=0.95)
        else:
            Y = scale(ds.target)
            output_size = 1
            rf = RandomForestRegressor(n_estimators=100)
            rf.fit(X, ds.target)
            monitor = None

        model = nn2(X.shape[1], output_size, cfg["classification"])
        viann = VarianceImportanceKeras()
        callbacks = [viann]
        if monitor:
            callbacks.append(monitor)

        model.fit(
            X,
            Y,
            validation_split=0.05,
            epochs=100,
            batch_size=max(1, int(round(X.shape[0] / 7))),
            verbose=0,
            callbacks=callbacks,
        )

        nn_scores = viann.var_scores
        rf_scores = rf.feature_importances_
        corr = np.corrcoef(nn_scores, rf_scores)[0, 1]
        logger.info("%s correlation with RF: %.2f", cfg["name"], corr)


if __name__ == "__main__":
    run_experiment()
