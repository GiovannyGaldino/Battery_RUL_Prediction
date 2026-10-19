#!/usr/bin/env python3
"""
Final consolidated HPC experimental campaign for lithium-ion battery RUL prediction.

Inputs expected by default:
  ~/outputs/batteries_modeling_v3/modeling_ready/charge_X.parquet
  ~/outputs/batteries_modeling_v3/modeling_ready/charge_y.parquet
  ~/outputs/batteries_modeling_v3/modeling_ready/charge_id.parquet
  ~/outputs/batteries_modeling_v3/modeling_ready/discharge_X.parquet
  ~/outputs/batteries_modeling_v3/modeling_ready/discharge_y.parquet
  ~/outputs/batteries_modeling_v3/modeling_ready/discharge_id.parquet

Main outputs:
  ~/outputs/batteries_modeling_v3/modeling_results_final_campaign_v1/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Baseline model
# -----------------------------

class MedianRegressor(BaseEstimator, RegressorMixin):
    """Median RUL baseline compatible with scikit-learn."""

    def __init__(self):
        self.constant_ = None

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        values = values[mask]
        weights = weights[mask]
        if values.size == 0:
            return np.nan
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]
        cdf = np.cumsum(weights) / np.sum(weights)
        return float(values[np.searchsorted(cdf, 0.5)])

    def fit(self, X, y, sample_weight=None):
        y = to_1d(y)
        if sample_weight is None:
            self.constant_ = float(np.nanmedian(y))
        else:
            self.constant_ = self._weighted_median(y, sample_weight)
        return self

    def predict(self, X):
        if self.constant_ is None:
            raise RuntimeError("MedianRegressor must be fitted before predict().")
        return np.full(shape=(len(X),), fill_value=self.constant_, dtype=float)


# -----------------------------
# Helpers
# -----------------------------

def to_1d(x) -> np.ndarray:
    """Convert Series/DataFrame/array to finite-compatible 1D numpy array."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"Expected one target column, received shape={x.shape}")
        x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        return x.to_numpy().ravel()
    return np.asarray(x).ravel()


def safe_name(value) -> str:
    text = str(value)
    text = text.replace("/", "_").replace("\\", "_").replace(" ", "_")
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return text[:160]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_parquet_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_parquet(path)


def load_phase(input_root: Path, phase: str, target: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    x_path = input_root / f"{phase}_X.parquet"
    y_path = input_root / f"{phase}_y.parquet"
    id_path = input_root / f"{phase}_id.parquet"

    print(f"[LOAD] {phase}: {x_path}")
    X = read_parquet_any(x_path).reset_index(drop=True)
    y_df = read_parquet_any(y_path).reset_index(drop=True)
    meta = read_parquet_any(id_path).reset_index(drop=True)

    if target not in y_df.columns:
        if y_df.shape[1] == 1:
            print(f"[WARN] Target '{target}' not found. Using only y column: {y_df.columns[0]}")
            y = y_df.iloc[:, 0]
        else:
            raise ValueError(f"Target '{target}' not found in {y_path}. Columns: {list(y_df.columns)}")
    else:
        y = y_df[target]

    required_meta = ["dataset", "battery_id"]
    missing_meta = [c for c in required_meta if c not in meta.columns]
    if missing_meta:
        raise ValueError(f"Missing required metadata columns in {id_path}: {missing_meta}")

    if len(X) != len(y) or len(X) != len(meta):
        raise ValueError(f"Row mismatch for {phase}: X={len(X)}, y={len(y)}, meta={len(meta)}")

    # Valid RUL and traceability filter.
    y_numeric = pd.to_numeric(y, errors="coerce")
    valid = y_numeric.notna() & np.isfinite(y_numeric.to_numpy()) & (y_numeric > 0)
    valid &= meta["dataset"].notna() & meta["battery_id"].notna()

    X = X.loc[valid].reset_index(drop=True)
    y = y_numeric.loc[valid].reset_index(drop=True)
    meta = meta.loc[valid].reset_index(drop=True)

    # Keep only numeric features and remove obvious index/source identifiers from predictors.
    drop_exact = {
        "index", "source_row_id", "battery_id", "dataset", "phase",
        "cycle_number", "cycle_index", target
    }
    drop_contains = ["target_", "rul", "remaining_cycle", "eol", "cycle_life"]
    keep_cols = []
    for col in X.columns:
        low = str(col).lower()
        if low in drop_exact:
            continue
        if any(token in low for token in drop_contains):
            continue
        keep_cols.append(col)

    X = X[keep_cols].copy()
    # Coerce non-numeric columns defensively.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    print(f"[LOAD] {phase}: X={X.shape}, y={y.shape}, batteries={meta['battery_id'].nunique()}, datasets={meta['dataset'].nunique()}")
    return X, y, meta


def split_global_by_battery_stratified(
    meta: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Battery-level holdout split, stratified approximately within each dataset."""
    rng = np.random.default_rng(random_state)
    train_batteries = []
    test_batteries = []

    for dataset, part in meta.groupby("dataset"):
        batts = np.array(sorted(part["battery_id"].unique(), key=lambda x: str(x)), dtype=object)
        rng.shuffle(batts)
        if len(batts) < 2:
            train_batteries.extend(batts.tolist())
            continue
        n_test = max(1, int(round(len(batts) * test_size)))
        n_test = min(n_test, len(batts) - 1)
        test_batteries.extend(batts[:n_test].tolist())
        train_batteries.extend(batts[n_test:].tolist())

    train_set = set(train_batteries)
    test_set = set(test_batteries)
    train_mask = meta["battery_id"].isin(train_set).to_numpy()
    test_mask = meta["battery_id"].isin(test_set).to_numpy()

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Invalid split: train or test split is empty.")

    overlap = train_set.intersection(test_set)
    if overlap:
        raise RuntimeError(f"Battery leakage detected in split: {list(overlap)[:5]}")
    return train_mask, test_mask


def split_within_dataset_by_battery(
    meta: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    batteries = np.array(sorted(meta["battery_id"].unique(), key=lambda x: str(x)), dtype=object)
    rng = np.random.default_rng(random_state)
    rng.shuffle(batteries)

    if len(batteries) < 2:
        raise ValueError("At least two batteries are required for a battery-level split.")

    n_test = max(1, int(round(len(batteries) * test_size)))
    n_test = min(n_test, len(batteries) - 1)
    test_batts = set(batteries[:n_test])
    train_batts = set(batteries[n_test:])

    train_mask = meta["battery_id"].isin(train_batts).to_numpy()
    test_mask = meta["battery_id"].isin(test_batts).to_numpy()

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Invalid dataset split: train or test split is empty.")
    return train_mask, test_mask


def make_lodo_split(meta: pd.DataFrame, heldout_dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    test_mask = (meta["dataset"].astype(str) == str(heldout_dataset)).to_numpy()
    train_mask = ~test_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(f"Invalid LODO split for dataset={heldout_dataset}")
    return train_mask, test_mask


def compute_sample_weights(meta_train: pd.DataFrame, mode: str = "dataset_inverse") -> Optional[np.ndarray]:
    if mode == "none":
        return None
    if mode != "dataset_inverse":
        raise ValueError(f"Unsupported sample weighting mode: {mode}")
    counts = meta_train["dataset"].value_counts()
    weights = meta_train["dataset"].map(lambda d: 1.0 / float(counts.loc[d])).astype(float).to_numpy()
    # Normalize to mean 1 for numerical stability.
    weights = weights / np.nanmean(weights)
    return weights


LEGACY_FEATURE_GROUPS = False   # v3: definido por --legacy-feature-groups


def feature_group(col: str) -> str:
    c = col.lower()
    if c.startswith("qv_") or c.startswith("vq_") or c.startswith("eq_"):
        return "curves"
    if "qv_" in c or "vq_" in c or "eq_" in c or "curve" in c:
        return "curves"
    # v3: stage_gap ANTES das familias de sinal. Em v2 este teste vinha por
    # ultimo e nunca casava (time_gap_* caia em time_sampling, voltage_gap_jump_n
    # em voltage, ...), deixando o grupo vazio.
    if not LEGACY_FEATURE_GROUPS and (
        "stage" in c or "_gap_" in c or c.endswith("_gap") or "rest" in c or "jump" in c or "transition" in c
    ):
        return "stage_gap"
    if "capacity" in c or "capacityh" in c or "soh" in c:
        return "capacity"
    if "energy" in c:
        return "energy"
    if "temperature" in c or "temp" in c:
        return "temperature"
    if "voltage" in c:
        return "voltage"
    if "current" in c:
        return "current"
    if "power" in c:
        return "power"
    if "time" in c or "duration" in c or "sample" in c or "n_points" in c:
        return "time_sampling"
    if "stage" in c or "gap" in c or "rest" in c or "jump" in c or "transition" in c:
        return "stage_gap"
    return "other"


def has_resolution(col: str, p: int) -> bool:
    c = col.lower()
    p3 = f"p{p:03d}"
    p_plain = f"p{p}"
    return p3 in c or f"_{p3}" in c or f"_{p_plain}_" in c or c.endswith(f"_{p_plain}")


def select_experiment_features(all_cols: Sequence[str], experiment: str) -> List[str]:
    cols = list(all_cols)
    groups = {c: feature_group(c) for c in cols}

    if experiment == "all_features":
        selected = cols
    elif experiment == "no_curves":
        selected = [c for c in cols if groups[c] != "curves"]
    elif experiment == "curves_only":
        selected = [c for c in cols if groups[c] == "curves"]
    elif experiment == "global_stats_only":
        selected = [c for c in cols if groups[c] != "curves" and groups[c] != "stage_gap"]
    elif experiment == "capacity_only":
        selected = [c for c in cols if groups[c] == "capacity"]
    elif experiment == "energy_added":
        selected = [c for c in cols if groups[c] in {"capacity", "energy"}]
    elif experiment == "no_temperature":
        selected = [c for c in cols if groups[c] != "temperature"]
    elif experiment == "no_time_sampling":
        selected = [c for c in cols if groups[c] != "time_sampling"]
    elif experiment == "no_stage_gap":
        selected = [c for c in cols if groups[c] != "stage_gap"]
    elif experiment == "no_causal_trend":
        # v3: braco "removidas": sem as contrapartes causais de tendencia.
        # Comparar com all_features (causais) e com a campanha v2 (com vazamento).
        selected = [c for c in cols if not c.lower().startswith("causal_")]
    elif experiment.startswith("resolution_p"):
        try:
            p = int(experiment.replace("resolution_p", ""))
        except ValueError:
            raise ValueError(f"Invalid resolution experiment name: {experiment}")
        selected = [
            c for c in cols
            if groups[c] != "curves" or has_resolution(c, p)
        ]
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    if len(selected) == 0:
        raise ValueError(f"Experiment '{experiment}' selected zero features.")
    return selected


def expand_experiments(experiments: Sequence[str]) -> List[str]:
    if "all" not in experiments:
        return list(experiments)
    return [
        "all_features",
        "no_curves",
        "global_stats_only",
        "curves_only",
        "capacity_only",
        "energy_added",
        "no_temperature",
        "no_time_sampling",
        "no_stage_gap",
        "resolution_p003",
        "resolution_p009",
        "resolution_p015",
        "resolution_p030",
        "no_causal_trend",
    ]


def prepare_feature_columns(
    X_train: pd.DataFrame,
    selected_cols: Sequence[str],
    missing_threshold: float,
) -> List[str]:
    Xt = X_train.loc[:, selected_cols]
    miss = Xt.isna().mean(axis=0)
    keep = miss[miss <= missing_threshold].index.tolist()
    if not keep:
        raise ValueError("No features left after missingness filter.")

    nunique = Xt[keep].nunique(dropna=True)
    keep = nunique[nunique > 1].index.tolist()
    if not keep:
        raise ValueError("No features left after constant-feature filter.")
    return keep


def make_model_specs(random_state: int, model_n_jobs: int, elasticnet_cv: bool = False) -> Dict[str, Tuple[BaseEstimator, bool]]:
    """Return model_name -> (estimator, needs_scaling).

    v3: `random_state` aqui e a semente dos ESTIMADORES (--model-random-state),
    que pode diferir da semente do split. `elasticnet_cv=True` substitui o
    alpha fixo por ElasticNetCV ajustado dentro do conjunto de treino.
    """
    if elasticnet_cv:
        from sklearn.linear_model import ElasticNetCV
        from sklearn import __version__ as _skv
        # Trajetoria de alphas gerada pelo sklearn a partir dos dados (de alpha_max
        # ate eps*alpha_max, 20 pontos, warm start ao longo da trajetoria). A grade
        # fixa logspace(-4,0) da v3.0.0 ficava inteira abaixo de alpha_max quando o
        # alvo esta em milhares de ciclos: quase sem regularizacao, sem convergir e
        # mais de uma hora por ajuste. `n_jobs` paraleliza folds x l1_ratio (15 tarefas).
        _kw = dict(l1_ratio=[0.1, 0.5, 0.9], eps=1e-3, cv=5, max_iter=5000, tol=1e-3,
                   random_state=random_state, selection="random", n_jobs=model_n_jobs)
        _major, _minor = (int(x) for x in _skv.split(".")[:2])
        if (_major, _minor) >= (1, 7):
            _kw["alphas"] = 20          # inteiro = numero de pontos da trajetoria
        else:
            _kw["n_alphas"] = 20        # parametro antigo (removido nas versoes recentes)
        enet = ElasticNetCV(**_kw)
    else:
        enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=50000, tol=1e-3, random_state=random_state, selection="random")
    return {
        "median": (MedianRegressor(), False),
        "elasticnet": (enet, True),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                n_jobs=model_n_jobs,
                random_state=random_state,
            ),
            False,
        ),
        "extra_trees": (
            ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                n_jobs=model_n_jobs,
                random_state=random_state,
            ),
            False,
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=random_state,
            ),
            False,
        ),
        "mlp": (
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                early_stopping=True,
                max_iter=250,
                random_state=random_state,
            ),
            True,
        ),
    }


def build_pipeline(estimator: BaseEstimator, needs_scaling: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("regressor", estimator))
    return Pipeline(steps)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = to_1d(y_true).astype(float)
    y_pred = to_1d(y_pred).astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        return {
            "n_samples": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan,
            "sMAPE": np.nan, "nMAE": np.nan, "nRMSE": np.nan,
            "y_mean": np.nan, "y_std": np.nan, "y_min": np.nan, "y_max": np.nan,
            "MAE_clip0": np.nan, "RMSE_clip0": np.nan, "R2_clip0": np.nan,
            "nMAE_clip0": np.nan, "nRMSE_clip0": np.nan,
            "n_pred_negative": 0, "share_pred_negative": np.nan,
        }

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if y_true.size >= 2 and np.unique(y_true).size >= 2 else np.nan
    smape = 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    scale = float(np.nanmean(np.abs(y_true))) + 1e-8

    # v3: mesmas metricas com a previsao truncada no dominio admissivel (RUL >= 0).
    # As colunas brutas acima sao mantidas intactas; estas sao adicionais.
    y_clip = np.maximum(y_pred, 0.0)
    n_neg = int((y_pred < 0).sum())
    mae_c = mean_absolute_error(y_true, y_clip)
    rmse_c = math.sqrt(mean_squared_error(y_true, y_clip))
    r2_c = r2_score(y_true, y_clip) if y_true.size >= 2 and np.unique(y_true).size >= 2 else np.nan

    return {
        "MAE_clip0": float(mae_c),
        "RMSE_clip0": float(rmse_c),
        "R2_clip0": float(r2_c) if np.isfinite(r2_c) else np.nan,
        "nMAE_clip0": float(mae_c / scale),
        "nRMSE_clip0": float(rmse_c / scale),
        "n_pred_negative": n_neg,
        "share_pred_negative": float(n_neg / y_true.size),
        "n_samples": int(y_true.size),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "sMAPE": float(smape),
        "nMAE": float(mae / scale),
        "nRMSE": float(rmse / scale),
        "y_mean": float(np.nanmean(y_true)),
        "y_std": float(np.nanstd(y_true)),
        "y_min": float(np.nanmin(y_true)),
        "y_max": float(np.nanmax(y_true)),
    }


def add_prediction_context(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.copy()
    if "cycle_index" in pred_df.columns:
        cycle_idx = pd.to_numeric(pred_df["cycle_index"], errors="coerce")
        eol_est = cycle_idx + pred_df["y_true"]
        norm_rul = pred_df["y_true"] / (eol_est.replace(0, np.nan))
        pred_df["normalized_rul"] = norm_rul.clip(lower=0, upper=1)
        bins = [-np.inf, 1/3, 2/3, np.inf]
        labels = ["late_life", "middle_life", "early_life"]
        pred_df["rul_bin"] = pd.cut(pred_df["normalized_rul"], bins=bins, labels=labels)
    else:
        q1, q2 = pred_df["y_true"].quantile([1/3, 2/3])
        pred_df["normalized_rul"] = np.nan
        pred_df["rul_bin"] = pd.cut(
            pred_df["y_true"],
            bins=[-np.inf, q1, q2, np.inf],
            labels=["low_rul", "middle_rul", "high_rul"],
        )

    pred_df["residual"] = pred_df["y_pred"] - pred_df["y_true"]
    pred_df["abs_error"] = np.abs(pred_df["residual"])
    pred_df["squared_error"] = pred_df["residual"] ** 2
    return pred_df


def metrics_by_group(pred_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    existing_cols = [c for c in group_cols if c in pred_df.columns]
    if not existing_cols:
        return pd.DataFrame()
    for keys, g in pred_df.groupby(existing_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(existing_cols, keys))
        row.update(regression_metrics(g["y_true"], g["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def save_plots(pred_df: pd.DataFrame, out_dir: Path, max_points: int, title_prefix: str) -> None:
    ensure_dir(out_dir / "plots")
    if len(pred_df) == 0:
        return
    sample = pred_df.sample(n=min(max_points, len(pred_df)), random_state=42)

    plt.figure(figsize=(6, 6))
    plt.scatter(sample["y_true"], sample["y_pred"], s=4, alpha=0.25)
    low = float(np.nanmin([sample["y_true"].min(), sample["y_pred"].min()]))
    high = float(np.nanmax([sample["y_true"].max(), sample["y_pred"].max()]))
    plt.plot([low, high], [low, high], linewidth=1)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title(f"{title_prefix}: predicted vs true")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "predicted_vs_true.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(sample["residual"], bins=60)
    plt.xlabel("Residual (predicted - true)")
    plt.ylabel("Count")
    plt.title(f"{title_prefix}: residual distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "residual_histogram.png", dpi=200)
    plt.close()

    if "dataset" in pred_df.columns:
        ds_metrics = metrics_by_group(pred_df, ["dataset"])
        if not ds_metrics.empty:
            ds_metrics = ds_metrics.sort_values("MAE")
            plt.figure(figsize=(8, 4))
            plt.bar(ds_metrics["dataset"].astype(str), ds_metrics["MAE"])
            plt.ylabel("MAE")
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{title_prefix}: MAE by dataset")
            plt.tight_layout()
            plt.savefig(out_dir / "plots" / "mae_by_dataset.png", dpi=200)
            plt.close()

    if "rul_bin" in pred_df.columns:
        rb_metrics = metrics_by_group(pred_df, ["rul_bin"])
        if not rb_metrics.empty:
            plt.figure(figsize=(7, 4))
            plt.bar(rb_metrics["rul_bin"].astype(str), rb_metrics["MAE"])
            plt.ylabel("MAE")
            plt.xticks(rotation=20, ha="right")
            plt.title(f"{title_prefix}: MAE by RUL bin")
            plt.tight_layout()
            plt.savefig(out_dir / "plots" / "mae_by_rul_bin.png", dpi=200)
            plt.close()


def fit_with_optional_weights(pipe: Pipeline, X_train, y_train, sample_weight=None) -> Tuple[Pipeline, str]:
    y_train = to_1d(y_train)
    if sample_weight is None:
        pipe.fit(X_train, y_train)
        return pipe, "none"

    try:
        pipe.fit(X_train, y_train, regressor__sample_weight=sample_weight)
        return pipe, "used"
    except Exception as exc:
        warnings.warn(f"Model did not accept sample_weight. Fitting without weights. Reason: {exc}")
        pipe.fit(X_train, y_train)
        return pipe, "fallback_no_weights"


def run_group_cv(
    model_name: str,
    estimator: BaseEstimator,
    needs_scaling: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    sample_weight: Optional[np.ndarray],
    cv_folds: int,
    out_dir: Path,
    run_info: Dict[str, object],
) -> pd.DataFrame:
    unique_groups = pd.Series(groups).nunique()
    n_splits = min(cv_folds, unique_groups)
    if n_splits < 2:
        return pd.DataFrame()

    rows = []
    cv = GroupKFold(n_splits=n_splits)
    y_train_1d = to_1d(y_train)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train_1d, groups=groups), start=1):
        pipe = build_pipeline(clone(estimator), needs_scaling)
        sw = sample_weight[tr_idx] if sample_weight is not None else None
        pipe, sw_status = fit_with_optional_weights(pipe, X_train.iloc[tr_idx], y_train_1d[tr_idx], sw)
        pred = pipe.predict(X_train.iloc[va_idx])
        metrics = regression_metrics(y_train_1d[va_idx], pred)
        metrics.update(run_info)
        metrics.update({"fold": fold, "model": model_name, "sample_weight_status": sw_status})
        rows.append(metrics)

    cv_df = pd.DataFrame(rows)
    if not cv_df.empty:
        cv_df.to_csv(out_dir / "cv_metrics.csv", index=False)
    return cv_df


def train_evaluate_one_model(
    model_name: str,
    estimator: BaseEstimator,
    needs_scaling: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
    out_dir: Path,
    sample_weight: Optional[np.ndarray],
    run_info: Dict[str, str],
    make_plots: bool,
    plot_max_points: int,
    cv_folds: int,
    meta_train: pd.DataFrame,
    save_models: bool,
    save_predictions: bool,
) -> Dict[str, object]:
    ensure_dir(out_dir)
    pipe = build_pipeline(clone(estimator), needs_scaling)

    if cv_folds and cv_folds > 1:
        run_group_cv(
            model_name=model_name,
            estimator=estimator,
            needs_scaling=needs_scaling,
            X_train=X_train,
            y_train=y_train,
            groups=meta_train["battery_id"],
            sample_weight=sample_weight,
            cv_folds=cv_folds,
            out_dir=out_dir,
            run_info=run_info,
        )

    pipe, sample_weight_status = fit_with_optional_weights(pipe, X_train, y_train, sample_weight)
    y_pred = pipe.predict(X_test)
    y_true = to_1d(y_test)

    pred_meta_cols = [c for c in ["dataset", "battery_id", "cycle_number", "cycle_index", "phase"] if c in meta_test.columns]
    pred_df = meta_test[pred_meta_cols].reset_index(drop=True).copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = to_1d(y_pred)
    for k, v in run_info.items():
        pred_df[k] = v
    pred_df["model"] = model_name
    pred_df = add_prediction_context(pred_df)

    if save_predictions:
        pred_df.to_csv(out_dir / "predictions.csv", index=False)
    if save_models:
        joblib.dump(pipe, out_dir / "model.joblib")

    global_metrics = regression_metrics(pred_df["y_true"], pred_df["y_pred"])
    global_metrics.update(run_info)
    global_metrics["model"] = model_name
    global_metrics["sample_weight_status"] = sample_weight_status
    global_metrics["n_train_samples"] = int(len(X_train))
    global_metrics["n_test_samples"] = int(len(X_test))
    global_metrics["n_train_batteries"] = int(meta_train["battery_id"].nunique())
    global_metrics["n_test_batteries"] = int(meta_test["battery_id"].nunique())
    global_metrics["n_features"] = int(X_train.shape[1])

    metrics_global = pd.DataFrame([global_metrics])
    metrics_dataset = metrics_by_group(pred_df, ["dataset"])
    metrics_battery = metrics_by_group(pred_df, ["dataset", "battery_id"])
    metrics_rul_bin = metrics_by_group(pred_df, ["rul_bin"])

    for df_metrics in [metrics_dataset, metrics_battery, metrics_rul_bin]:
        if not df_metrics.empty:
            for k, v in run_info.items():
                df_metrics[k] = v
            df_metrics["model"] = model_name

    metrics_global.to_csv(out_dir / "metrics_global.csv", index=False)
    metrics_dataset.to_csv(out_dir / "metrics_by_dataset.csv", index=False)
    metrics_battery.to_csv(out_dir / "metrics_by_battery.csv", index=False)
    metrics_rul_bin.to_csv(out_dir / "metrics_by_rul_bin.csv", index=False)

    if make_plots:
        title = f"{run_info.get('phase')} | {run_info.get('mode')} | {run_info.get('scope')} | {run_info.get('experiment')} | {model_name}"
        save_plots(pred_df, out_dir, max_points=plot_max_points, title_prefix=title)

    print(
        f"[DONE] {run_info.get('phase')} | {run_info.get('mode')} | {run_info.get('scope')} | "
        f"{run_info.get('experiment')} | {model_name}: "
        f"MAE={global_metrics['MAE']:.3f}, RMSE={global_metrics['RMSE']:.3f}, R2={global_metrics['R2']:.3f}"
    )
    return {
        "global": metrics_global,
        "dataset": metrics_dataset,
        "battery": metrics_battery,
        "rul_bin": metrics_rul_bin,
    }


def run_training_case(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    phase: str,
    mode: str,
    scope: str,
    experiment: str,
    args,
    model_specs: Dict[str, Tuple[BaseEstimator, bool]],
) -> Dict[str, List[pd.DataFrame]]:
    selected_cols = select_experiment_features(X.columns, experiment)

    X_train_full = X.loc[train_mask, selected_cols].reset_index(drop=True)
    X_test_full = X.loc[test_mask, selected_cols].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)
    meta_train = meta.loc[train_mask].reset_index(drop=True)
    meta_test = meta.loc[test_mask].reset_index(drop=True)

    if len(X_train_full) == 0 or len(X_test_full) == 0:
        raise ValueError("Empty train/test matrix in training case.")

    kept_cols = prepare_feature_columns(X_train_full, X_train_full.columns, args.missing_threshold)
    X_train = X_train_full[kept_cols].replace([np.inf, -np.inf], np.nan)
    X_test = X_test_full[kept_cols].replace([np.inf, -np.inf], np.nan)

    sample_weight = compute_sample_weights(meta_train, args.sample_weighting)

    base_out = (
        args.output_root
        / phase
        / mode
        / safe_name(scope)
        / experiment
    )
    ensure_dir(base_out)
    split_summary = {
        "phase": phase,
        "mode": mode,
        "scope": scope,
        "experiment": experiment,
        "campaign_component": args.campaign_component,
        "random_state": int(args.random_state),
        "cv_folds": int(args.cv_folds),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "n_train_batteries": int(meta_train["battery_id"].nunique()),
        "n_test_batteries": int(meta_test["battery_id"].nunique()),
        "n_features_raw_experiment": int(len(selected_cols)),
        "n_features_after_filter": int(len(kept_cols)),
        "train_datasets": sorted(map(str, meta_train["dataset"].unique())),
        "test_datasets": sorted(map(str, meta_test["dataset"].unique())),
    }
    with open(base_out / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2, ensure_ascii=False)

    print(
        f"[CASE] phase={phase} mode={mode} scope={scope} experiment={experiment} "
        f"train={X_train.shape} test={X_test.shape} train_batt={meta_train['battery_id'].nunique()} "
        f"test_batt={meta_test['battery_id'].nunique()}"
    )

    run_info = {
        "campaign_component": args.campaign_component,
        "phase": phase,
        "mode": mode,
        "scope": str(scope),
        "experiment": experiment,
        "target": args.target,
        "random_state": int(args.random_state),
        "test_size": float(args.test_size),
        "cv_folds": int(args.cv_folds),
        "sample_weighting": str(args.sample_weighting),
        "missing_threshold": float(args.missing_threshold),
    }

    selected_model_specs = {
        k: v for k, v in model_specs.items()
        if k in args.models
    }

    tasks = []
    for model_name, (estimator, needs_scaling) in selected_model_specs.items():
        model_out = base_out / model_name
        if args.skip_existing and (model_out / "metrics_global.csv").exists():
            print(f"[SKIP] Existing result: {model_out}")
            continue
        tasks.append((model_name, estimator, needs_scaling, model_out))

    if args.dry_run:
        print(f"[DRY-RUN] Would train {len(tasks)} models for {phase}/{mode}/{scope}/{experiment}.")
        return {"global": [], "dataset": [], "battery": [], "rul_bin": []}

    if not tasks:
        return {"global": [], "dataset": [], "battery": [], "rul_bin": []}

    results = Parallel(n_jobs=args.n_jobs, prefer=args.parallel_backend)(
        delayed(train_evaluate_one_model)(
            model_name=model_name,
            estimator=estimator,
            needs_scaling=needs_scaling,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            meta_test=meta_test,
            out_dir=model_out,
            sample_weight=sample_weight,
            run_info=run_info,
            make_plots=args.make_plots,
            plot_max_points=args.plot_max_points,
            cv_folds=args.cv_folds,
            meta_train=meta_train,
            save_models=args.save_models,
            save_predictions=not args.no_save_predictions,
        )
        for model_name, estimator, needs_scaling, model_out in tasks
    )

    aggregated = {"global": [], "dataset": [], "battery": [], "rul_bin": []}
    for res in results:
        for key in aggregated:
            if key in res and isinstance(res[key], pd.DataFrame) and not res[key].empty:
                aggregated[key].append(res[key])
    return aggregated


def append_aggregate(store: Dict[str, List[pd.DataFrame]], case_res: Dict[str, List[pd.DataFrame]]) -> None:
    for key in store:
        store[key].extend(case_res.get(key, []))


def write_master_tables(store: Dict[str, List[pd.DataFrame]], output_root: Path) -> None:
    if os.environ.get("NO_MASTER_TABLES", "0") == "1":
        return
    for key, frames in store.items():
        if frames:
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.DataFrame()
        df.to_csv(output_root / f"all_metrics_{key}.csv", index=False)

    # Convenience macro tables from by-dataset metrics.
    dataset_path = output_root / "all_metrics_dataset.csv"
    if not dataset_path.exists():
        dataset_path = output_root / "all_metrics_dataset.csv"

    if store["dataset"]:
        ds = pd.concat(store["dataset"], ignore_index=True)
        group_cols = ["phase", "mode", "scope", "experiment", "model"]
        metric_cols = ["MAE", "RMSE", "nMAE", "nRMSE", "sMAPE", "R2"]
        available_metrics = [c for c in metric_cols if c in ds.columns]
        macro = ds.groupby(group_cols, dropna=False)[available_metrics].mean().reset_index()
        macro = macro.rename(columns={c: f"macro_dataset_{c}" for c in available_metrics})
        macro.to_csv(output_root / "all_metrics_macro_by_dataset.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final consolidated HPC RUL modeling campaign.")

    parser.add_argument("--input-root", type=Path, default=Path("~/outputs/batteries_modeling_v3/modeling_ready").expanduser())
    parser.add_argument("--output-root", type=Path, default=Path("~/outputs/batteries_modeling_v3/modeling_results_final_campaign_v1").expanduser())
    parser.add_argument("--target", default="target_rul80_steps_clipped")
    parser.add_argument("--campaign-component", default="full_matrix", help="Logical component of the final campaign, e.g., full_matrix, repeated_split, internal_cv.")
    parser.add_argument("--phases", nargs="+", default=["charge", "discharge"], choices=["charge", "discharge"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["median", "elasticnet", "random_forest", "extra_trees", "hist_gradient_boosting", "mlp"],
        choices=["median", "elasticnet", "random_forest", "extra_trees", "hist_gradient_boosting", "mlp"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["global", "per_dataset"],
        choices=["global", "per_dataset", "lodo"],
        help="global: mixed dataset model; per_dataset: one model per dataset; lodo: leave-one-dataset-out.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all_features"],
        help="Use 'all' for the full ablation list.",
    )
    parser.add_argument("--lodo-heldouts", nargs="*", default=None, help="Optional subset of heldout datasets for LODO mode.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42, help="Semente do SPLIT.")
    # ------------------------------------------------------------ v3
    parser.add_argument("--model-random-state", type=int, default=None,
                        help="Semente dos ESTIMADORES. Default: igual a --random-state (v2).")
    parser.add_argument("--elasticnet-cv", action="store_true",
                        help="ElasticNetCV ajustado dentro do treino em vez de alpha=0.01 fixo.")
    parser.add_argument("--exclude-datasets", nargs="*", default=None,
                        help="Dominios removidos de X/y/meta apos a carga (selecao nested source-only).")
    parser.add_argument("--legacy-feature-groups", action="store_true",
                        help="Reproduz o classificador de grupos de v2 (stage_gap vazio).")
    parser.add_argument("--missing-threshold", type=float, default=0.95)
    parser.add_argument("--sample-weighting", default="dataset_inverse", choices=["dataset_inverse", "none"])
    parser.add_argument("--min-batteries-per-dataset", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs across models within each training case.")
    parser.add_argument("--model-n-jobs", type=int, default=-1, help="n_jobs for RF/ExtraTrees. Use 1 if --n-jobs > 1 to avoid oversubscription.")
    parser.add_argument("--parallel-backend", default="threads", choices=["threads", "processes"])
    parser.add_argument("--cv-folds", type=int, default=0, help="Optional GroupKFold CV inside training set. 0 disables CV.")
    parser.add_argument("--make-plots", action="store_true", help="Generate PNG plots for each model/case.")
    parser.add_argument("--save-models", action="store_true", help="Persist fitted model.joblib files. Disabled by default to reduce disk usage in the final campaign.")
    parser.add_argument("--no-save-predictions", action="store_true", help="Do not save per-sample predictions.csv files. Metrics are still computed and saved.")
    parser.add_argument("--plot-max-points", type=int, default=20000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-rows-per-phase", type=int, default=None, help="Debug option: sample N rows per phase after filtering.")

    args = parser.parse_args()
    args.input_root = args.input_root.expanduser()
    args.output_root = args.output_root.expanduser()
    args.experiments = expand_experiments(args.experiments)
    if args.model_random_state is None:
        args.model_random_state = args.random_state
    global LEGACY_FEATURE_GROUPS
    LEGACY_FEATURE_GROUPS = bool(args.legacy_feature_groups)

    if args.parallel_backend == "processes":
        args.parallel_backend = "processes"
    else:
        args.parallel_backend = "threads"

    return args


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)

    with open(args.output_root / "run_config.json", "w", encoding="utf-8") as f:
        serializable = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        serializable["package"] = "campanha_v3"
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print("Battery RUL full modeling pipeline")
    print(f"Input root:   {args.input_root}")
    print(f"Output root:  {args.output_root}")
    print(f"Component:    {args.campaign_component}")
    print(f"Phases:       {args.phases}")
    print(f"Models:       {args.models}")
    print(f"Modes:        {args.modes}")
    print(f"Experiments:  {args.experiments}")
    print(f"n_jobs:       {args.n_jobs}")
    print(f"model_n_jobs: {args.model_n_jobs}")
    print(f"CV folds:     {args.cv_folds}")
    print(f"Save models:  {args.save_models}")
    print(f"Save preds:   {not args.no_save_predictions}")

    model_specs = make_model_specs(args.model_random_state, args.model_n_jobs, elasticnet_cv=args.elasticnet_cv)
    master_store = {"global": [], "dataset": [], "battery": [], "rul_bin": []}

    for phase in args.phases:
        print("\n" + "=" * 90)
        print(f"[PHASE] {phase}")
        X, y, meta = load_phase(args.input_root, phase, args.target)

        # v3: remove dominios inteiros (selecao nested: o alvo nao entra em nenhum treino)
        if args.exclude_datasets:
            keep = ~meta["dataset"].astype(str).isin([str(d) for d in args.exclude_datasets])
            n0 = len(X)
            X = X.loc[keep].reset_index(drop=True); y = y.loc[keep].reset_index(drop=True)
            meta = meta.loc[keep].reset_index(drop=True)
            print(f"[EXCLUDE] {phase}: removidos {args.exclude_datasets} -> {n0:,} -> {len(X):,} linhas")

        if args.max_rows_per_phase is not None and len(X) > args.max_rows_per_phase:
            idx = pd.Series(np.arange(len(X))).sample(args.max_rows_per_phase, random_state=args.random_state).sort_values().to_numpy()
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)
            meta = meta.iloc[idx].reset_index(drop=True)
            print(f"[DEBUG] Sampled phase to {len(X)} rows.")

        # Global mixed-dataset model.
        if "global" in args.modes:
            train_mask, test_mask = split_global_by_battery_stratified(meta, args.test_size, args.random_state)
            for experiment in args.experiments:
                try:
                    res = run_training_case(
                        X=X, y=y, meta=meta,
                        train_mask=train_mask,
                        test_mask=test_mask,
                        phase=phase,
                        mode="global",
                        scope="all_datasets",
                        experiment=experiment,
                        args=args,
                        model_specs=model_specs,
                    )
                    append_aggregate(master_store, res)
                    write_master_tables(master_store, args.output_root)
                except Exception as exc:
                    print(f"[ERROR] Failed global case phase={phase} experiment={experiment}: {exc}")

        # One independent model per dataset.
        if "per_dataset" in args.modes:
            for dataset in sorted(meta["dataset"].astype(str).unique()):
                ds_mask = (meta["dataset"].astype(str) == dataset).to_numpy()
                meta_ds = meta.loc[ds_mask].reset_index(drop=True)
                n_batt = meta_ds["battery_id"].nunique()
                if n_batt < args.min_batteries_per_dataset:
                    print(f"[SKIP] per_dataset {phase}/{dataset}: only {n_batt} batteries.")
                    continue

                X_ds = X.loc[ds_mask].reset_index(drop=True)
                y_ds = y.loc[ds_mask].reset_index(drop=True)

                try:
                    train_mask_ds, test_mask_ds = split_within_dataset_by_battery(meta_ds, args.test_size, args.random_state)
                except Exception as exc:
                    print(f"[SKIP] per_dataset {phase}/{dataset}: {exc}")
                    continue

                for experiment in args.experiments:
                    try:
                        res = run_training_case(
                            X=X_ds, y=y_ds, meta=meta_ds,
                            train_mask=train_mask_ds,
                            test_mask=test_mask_ds,
                            phase=phase,
                            mode="per_dataset",
                            scope=dataset,
                            experiment=experiment,
                            args=args,
                            model_specs=model_specs,
                        )
                        append_aggregate(master_store, res)
                        write_master_tables(master_store, args.output_root)
                    except Exception as exc:
                        print(f"[ERROR] Failed per_dataset case phase={phase} dataset={dataset} experiment={experiment}: {exc}")

        # Leave-one-dataset-out generalization.
        if "lodo" in args.modes:
            datasets = sorted(meta["dataset"].astype(str).unique())
            if getattr(args, "lodo_heldouts", None):
                requested = set(str(x).replace("heldout_", "") for x in args.lodo_heldouts)
                datasets = [d for d in datasets if str(d) in requested]
                print(f"[LODO-SUBSET] requested={sorted(requested)} selected={datasets}")
            for heldout in datasets:
                try:
                    train_mask, test_mask = make_lodo_split(meta, heldout)
                except Exception as exc:
                    print(f"[SKIP] LODO {phase}/{heldout}: {exc}")
                    continue

                for experiment in args.experiments:
                    try:
                        res = run_training_case(
                            X=X, y=y, meta=meta,
                            train_mask=train_mask,
                            test_mask=test_mask,
                            phase=phase,
                            mode="lodo",
                            scope=f"heldout_{heldout}",
                            experiment=experiment,
                            args=args,
                            model_specs=model_specs,
                        )
                        append_aggregate(master_store, res)
                        write_master_tables(master_store, args.output_root)
                    except Exception as exc:
                        print(f"[ERROR] Failed LODO case phase={phase} heldout={heldout} experiment={experiment}: {exc}")

    write_master_tables(master_store, args.output_root)
    print("\n[DONE] Full modeling pipeline finished.")
    print(f"[DONE] Master metrics saved in: {args.output_root}")


if __name__ == "__main__":
    main()
