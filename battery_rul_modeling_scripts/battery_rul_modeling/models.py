from __future__ import annotations

import inspect
from typing import Dict, Tuple

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import NumericFeatureCleaner


def make_pipeline(model_name: str, max_missing_frac: float, min_variance: float, random_seed: int) -> Pipeline:
    model_name = model_name.lower()
    cleaner = NumericFeatureCleaner(max_missing_frac=max_missing_frac, min_variance=min_variance)
    imputer = SimpleImputer(strategy="median")

    if model_name == "median":
        estimator = DummyRegressor(strategy="median")
        return Pipeline([("cleaner", cleaner), ("imputer", imputer), ("model", estimator)])

    if model_name == "elasticnet":
        estimator = ElasticNet(max_iter=20000, random_state=random_seed)
        return Pipeline([
            ("cleaner", cleaner), ("imputer", imputer), ("scaler", StandardScaler()), ("model", estimator)
        ])

    if model_name == "random_forest":
        estimator = RandomForestRegressor(random_state=random_seed, n_jobs=-1)
        return Pipeline([("cleaner", cleaner), ("imputer", imputer), ("model", estimator)])

    if model_name == "extra_trees":
        estimator = ExtraTreesRegressor(random_state=random_seed, n_jobs=-1)
        return Pipeline([("cleaner", cleaner), ("imputer", imputer), ("model", estimator)])

    if model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(random_state=random_seed, early_stopping=True)
        return Pipeline([("cleaner", cleaner), ("imputer", imputer), ("model", estimator)])

    if model_name == "mlp":
        estimator = MLPRegressor(
            random_state=random_seed,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        return Pipeline([
            ("cleaner", cleaner), ("imputer", imputer), ("scaler", StandardScaler()), ("model", estimator)
        ])

    raise ValueError(f"Modelo desconhecido: {model_name}")


def param_distributions(model_name: str, quick: bool = False) -> Dict[str, list]:
    model_name = model_name.lower()
    if model_name == "median":
        return {}
    if model_name == "elasticnet":
        return {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0] if not quick else [0.001, 0.01, 0.1],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9] if not quick else [0.3, 0.7],
        }
    if model_name == "random_forest":
        return {
            "model__n_estimators": [100, 300, 500] if not quick else [100],
            "model__max_depth": [5, 10, 20, None] if not quick else [10, None],
            "model__min_samples_leaf": [1, 2, 4, 8] if not quick else [1, 4],
            "model__max_features": ["sqrt", "log2", 0.5, 1.0] if not quick else ["sqrt", 0.5],
        }
    if model_name == "extra_trees":
        return {
            "model__n_estimators": [100, 300, 500] if not quick else [100],
            "model__max_depth": [5, 10, 20, None] if not quick else [10, None],
            "model__min_samples_leaf": [1, 2, 4, 8] if not quick else [1, 4],
            "model__max_features": ["sqrt", "log2", 0.5, 1.0] if not quick else ["sqrt", 0.5],
        }
    if model_name == "hist_gradient_boosting":
        return {
            "model__max_iter": [100, 300, 500] if not quick else [100],
            "model__learning_rate": [0.01, 0.05, 0.1] if not quick else [0.05, 0.1],
            "model__max_leaf_nodes": [15, 31, 63] if not quick else [31],
            "model__l2_regularization": [0.0, 0.001, 0.01, 0.1] if not quick else [0.0, 0.01],
        }
    if model_name == "mlp":
        return {
            "model__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)] if not quick else [(50,), (100,)],
            "model__activation": ["relu", "tanh"] if not quick else ["relu"],
            "model__alpha": [0.0001, 0.001, 0.01] if not quick else [0.001],
            "model__learning_rate_init": [0.001, 0.01] if not quick else [0.001],
        }
    raise ValueError(f"Modelo desconhecido: {model_name}")


def supports_sample_weight(pipeline: Pipeline) -> bool:
    estimator = pipeline.named_steps.get("model")
    if estimator is None:
        return False
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except Exception:
        return False


def dataset_balanced_sample_weights(dataset_values) -> np.ndarray:
    import pandas as pd

    s = pd.Series(dataset_values).astype(str)
    counts = s.value_counts()
    n_datasets = max(len(counts), 1)
    n_total = len(s)
    weights = s.map(lambda d: n_total / (n_datasets * counts.loc[d])).to_numpy(dtype=float)
    return weights
