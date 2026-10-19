from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_r2(y_true, y_pred) -> float:
    try:
        if len(y_true) < 2 or np.nanstd(y_true) == 0:
            return np.nan
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan


def smape(y_true, y_pred, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def filtered_mape(y_true, y_pred, min_true: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) >= min_true)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _normalizer_from_meta(y_true: np.ndarray, meta: Optional[pd.DataFrame]) -> float:
    candidates = []
    if meta is not None:
        for col in ["cycle_life", "target_eol80_cycle_index", "target_eol80_order_index", "target_eol70_cycle_index"]:
            if col in meta.columns:
                vals = pd.to_numeric(meta[col], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if vals.size:
                    candidates.append(float(np.nanmedian(vals)))
    if candidates:
        return max(float(np.nanmedian(candidates)), 1.0)
    finite_y = y_true[np.isfinite(y_true)]
    if finite_y.size:
        return max(float(np.nanmax(finite_y) - np.nanmin(finite_y)), 1.0)
    return 1.0


def regression_metrics(
    y_true,
    y_pred,
    meta: Optional[pd.DataFrame] = None,
    mape_min_true: float = 1.0,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {
            "n_samples": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan,
            "smape": np.nan, "mape_filtered": np.nan, "median_abs_error": np.nan,
            "mean_signed_error": np.nan, "nmae": np.nan, "nrmse": np.nan,
            "max_abs_error": np.nan,
        }
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    norm = _normalizer_from_meta(y_true, meta.loc[mask] if meta is not None and len(meta) == len(mask) else meta)
    return {
        "n_samples": int(y_true.size),
        "mae": mae,
        "rmse": rmse,
        "r2": safe_r2(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mape_filtered": filtered_mape(y_true, y_pred, min_true=mape_min_true),
        "median_abs_error": float(np.nanmedian(abs_err)),
        "mean_signed_error": float(np.nanmean(err)),
        "nmae": float(mae / norm),
        "nrmse": float(rmse / norm),
        "max_abs_error": float(np.nanmax(abs_err)),
    }


def add_error_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    out["smape_sample"] = 2.0 * out["abs_error"] / (out["y_true"].abs() + out["y_pred"].abs() + 1e-9) * 100.0
    return out


def make_rul_bins(df: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(df["y_true"], errors="coerce")
    absolute_bins = [-np.inf, 50, 100, 200, 500, np.inf]
    labels = ["0-50", "50-100", "100-200", "200-500", ">500"]
    try:
        return pd.cut(y, bins=absolute_bins, labels=labels, include_lowest=True).astype(str)
    except Exception:
        return pd.Series(["unknown"] * len(df), index=df.index)


def make_normalized_rul_bins(df: pd.DataFrame) -> pd.Series:
    if "cycle_life" in df.columns:
        denom = pd.to_numeric(df["cycle_life"], errors="coerce")
    elif "target_eol80_order_index" in df.columns:
        denom = pd.to_numeric(df["target_eol80_order_index"], errors="coerce")
    else:
        denom = pd.Series(np.nan, index=df.index)
    y = pd.to_numeric(df["y_true"], errors="coerce")
    norm = y / denom.replace(0, np.nan)
    bins = [-np.inf, 0.33, 0.66, np.inf]
    labels = ["late_0_0.33", "middle_0.33_0.66", "early_0.66_1"]
    return pd.cut(norm, bins=bins, labels=labels, include_lowest=True).astype(str).fillna("unknown")


def grouped_metrics(pred_df: pd.DataFrame, group_cols: Iterable[str], mape_min_true: float = 1.0) -> pd.DataFrame:
    rows = []
    group_cols = list(group_cols)
    for keys, g in pred_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(regression_metrics(g["y_true"], g["y_pred"], meta=g, mape_min_true=mape_min_true))
        if "battery_id" in g.columns:
            row["n_batteries"] = int(g["battery_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)
