"""
audit_extracted_features_v2.py

Audit and preliminary exploratory analysis for extracted hierarchical battery features.

Main outputs
------------
CSV reports:
- feature_files_index.csv
- columns_inventory.csv
- feature_families_inventory.csv
- dataset_summary.csv
- rows_by_dataset_level.csv
- batteries_per_dataset.csv
- cycles_per_battery.csv
- missing_values_report.csv
- outlier_summary.csv
- outlier_rows_sample.csv
- target_correlations.csv
- target_correlations_by_dataset.csv
- robustness_checks.csv
- constant_or_near_constant_features.csv
- redundant_feature_pairs.csv

Plots:
- rows_by_dataset.png
- batteries_by_dataset.png
- cycles_per_battery_boxplot.png
- missingness_top.png
- outlier_rate_top.png
- target_distribution.png
- target_correlations_top.png
- target_vs_top_features_*.png

Markdown:
- audit_report.md

Typical PowerShell usage
------------------------

& C:\\Users\\Giovanny\\anaconda3\\envs\\batteries\\python.exe ".\\scripts\\pipelines\\audit_extracted_features_v2.py" --features-root "C:\\Artigo Baterias de Litio\\Batteries\\data\\processed\\features_hierarchical_v2" --reports-root "C:\\Artigo Baterias de Litio\\Batteries\\reports\\feature_audit_v2" --target auto --correlation-feature-level cycle

If memory usage is high, use:
--sample-rows 200000

If you want to audit only a dataset:
--dataset hust
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_FEATURES_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/processed/features_hierarchical_v2")
DEFAULT_REPORTS_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/reports/feature_audit_v2")

ID_COLUMNS = {
    "dataset",
    "battery_id",
    "cycle",
    "phase",
    "feature_level",
    "segment_index",
    "source_file",
    "time_is_synthetic",
}

TARGET_PRIORITY = [
    "rul",
    "rul_last",
    "rul_mean",
    "soh",
    "soh_last",
    "soh_mean",
    "capacity_last",
    "capacity_mean",
    "capacity_delta",
    "integrated_capacity_abs_ah",
    "integrated_capacity_signed_ah",
]

STAT_SUFFIXES = [
    "_n_valid",
    "_first",
    "_last",
    "_min",
    "_max",
    "_mean",
    "_std",
    "_median",
    "_q05",
    "_q95",
    "_delta",
    "_range",
    "_boundary_jump_n",
    "_boundary_jump_mean",
    "_boundary_jump_abs_mean",
    "_boundary_jump_abs_max",
    "_boundary_jump_abs_sum",
]


@dataclass
class AuditCheck:
    check: str
    status: str
    metric: Any
    threshold: Any
    details: str


def now_id() -> str:
    return datetime.now().strftime("audit_%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def parse_list_argument(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    output: List[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                output.append(part)
    return list(dict.fromkeys(output))


def clean_filename(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value)).strip("_")
    return (value or "file")[:max_len]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def discover_feature_files(
    features_root: Path,
    datasets: Sequence[str],
) -> List[Path]:
    files = sorted(features_root.rglob("*.features.parquet"), key=lambda p: str(p).lower())

    if datasets:
        selected = []
        wanted = {d.lower() for d in datasets}
        for path in files:
            # Expected layout: root / dataset / parts / file.features.parquet
            parts = [part.lower() for part in path.parts]
            if any(dataset in parts for dataset in wanted):
                selected.append(path)
        files = selected

    return files


def infer_dataset_from_path(path: Path, features_root: Path) -> str:
    try:
        relative = path.relative_to(features_root)
        if len(relative.parts) >= 1:
            return relative.parts[0]
    except Exception:
        pass
    return "unknown"


def read_features(
    files: Sequence[Path],
    features_root: Path,
    sample_rows: Optional[int],
    add_source_file: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    index_records: List[Dict[str, Any]] = []

    rng = np.random.default_rng(42)

    for path in files:
        try:
            df = pd.read_parquet(path, engine="fastparquet")
        except Exception:
            # Fallback to pandas default engine if needed.
            df = pd.read_parquet(path)

        dataset_from_path = infer_dataset_from_path(path, features_root)

        if "dataset" not in df.columns:
            df["dataset"] = dataset_from_path

        if add_source_file:
            df["source_file"] = str(path)

        index_records.append(
            {
                "dataset_from_path": dataset_from_path,
                "file": str(path),
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 4),
                "has_done_marker": Path(str(path) + ".done").exists(),
            }
        )

        frames.append(df)

    file_index = pd.DataFrame(index_records)

    if not frames:
        return pd.DataFrame(), file_index

    data = pd.concat(frames, ignore_index=True, sort=False)

    del frames

    if sample_rows is not None and sample_rows > 0 and len(data) > sample_rows:
        sample_indices = rng.choice(data.index.to_numpy(), size=sample_rows, replace=False)
        data = data.loc[sample_indices].reset_index(drop=True)

    data = optimize_memory(data)
    return data, file_index


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in df.select_dtypes(include=["int64", "int32", "int16"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in ["dataset", "battery_id", "phase", "feature_level"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("unknown")

    return df


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = []
    for col in df.columns:
        if col in ID_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def get_id_columns_present(df: pd.DataFrame) -> List[str]:
    return [col for col in ["dataset", "battery_id", "cycle", "phase", "feature_level", "segment_index"] if col in df.columns]


def is_finite_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.isfinite(numeric.to_numpy(dtype=np.float64))


def resolve_target(df: pd.DataFrame, requested_target: str) -> Optional[str]:
    if requested_target and requested_target.lower() != "auto":
        return requested_target if requested_target in df.columns else None

    for target in TARGET_PRIORITY:
        if target in df.columns:
            valid = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum()
            if valid > 0:
                return target

    # More flexible fallback.
    candidates = []
    for col in df.columns:
        c = col.lower()
        if any(token in c for token in ["rul", "soh", "capacity"]):
            if pd.api.types.is_numeric_dtype(df[col]):
                valid = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum()
                candidates.append((valid, col))

    if candidates:
        candidates = sorted(candidates, reverse=True)
        return candidates[0][1]

    return None


def infer_feature_family(column: str) -> Tuple[str, str]:
    for suffix in sorted(STAT_SUFFIXES, key=len, reverse=True):
        if column.endswith(suffix):
            return column[: -len(suffix)], suffix.strip("_")
    return column, "raw_or_custom"


def make_columns_inventory(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_rate = missing_count / max(len(df), 1)

        if pd.api.types.is_numeric_dtype(s):
            numeric = pd.to_numeric(s, errors="coerce")
            inf_count = int(np.isinf(numeric.to_numpy(dtype=np.float64)).sum())
            finite_count = int(np.isfinite(numeric.to_numpy(dtype=np.float64)).sum())
            min_value = safe_float(np.nanmin(numeric.to_numpy(dtype=np.float64))) if finite_count else np.nan
            max_value = safe_float(np.nanmax(numeric.to_numpy(dtype=np.float64))) if finite_count else np.nan
            mean_value = safe_float(np.nanmean(numeric.to_numpy(dtype=np.float64))) if finite_count else np.nan
            std_value = safe_float(np.nanstd(numeric.to_numpy(dtype=np.float64))) if finite_count else np.nan
        else:
            inf_count = 0
            finite_count = np.nan
            min_value = np.nan
            max_value = np.nan
            mean_value = np.nan
            std_value = np.nan

        try:
            unique_count = int(s.nunique(dropna=True))
        except Exception:
            unique_count = np.nan

        records.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "non_missing_count": int(len(df) - missing_count),
                "unique_count": unique_count,
                "inf_count": inf_count,
                "finite_count": finite_count,
                "min": min_value,
                "max": max_value,
                "mean": mean_value,
                "std": std_value,
            }
        )

    return pd.DataFrame(records).sort_values(["missing_rate", "column"], ascending=[False, True])


def make_feature_families_inventory(columns: Sequence[str]) -> pd.DataFrame:
    records = []
    for col in columns:
        family, statistic = infer_feature_family(col)
        records.append(
            {
                "column": col,
                "family": family,
                "statistic": statistic,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    summary = (
        df.groupby("family", dropna=False)
        .agg(
            n_columns=("column", "count"),
            statistics=("statistic", lambda x: ", ".join(sorted(set(map(str, x))))),
            columns=("column", lambda x: ", ".join(sorted(map(str, x))[:20])),
        )
        .reset_index()
        .sort_values(["n_columns", "family"], ascending=[False, True])
    )
    return summary


def make_dataset_summaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "dataset" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    group_cols = ["dataset"]

    dataset_summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows=("dataset", "size"),
            batteries=("battery_id", "nunique") if "battery_id" in df.columns else ("dataset", "size"),
            cycles=("cycle", "nunique") if "cycle" in df.columns else ("dataset", "size"),
            feature_levels=("feature_level", "nunique") if "feature_level" in df.columns else ("dataset", "size"),
            phases=("phase", "nunique") if "phase" in df.columns else ("dataset", "size"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    if "feature_level" in df.columns:
        rows_by_level = (
            df.groupby(["dataset", "feature_level"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values(["dataset", "feature_level"])
        )
    else:
        rows_by_level = pd.DataFrame()

    if "battery_id" in df.columns:
        batteries_per_dataset = (
            df.groupby("dataset", dropna=False)["battery_id"]
            .nunique()
            .reset_index(name="n_batteries")
            .sort_values("n_batteries", ascending=False)
        )
    else:
        batteries_per_dataset = pd.DataFrame()

    if {"dataset", "battery_id", "cycle"}.issubset(df.columns):
        cycles_per_battery = (
            df.groupby(["dataset", "battery_id"], dropna=False)
            .agg(
                n_cycles=("cycle", "nunique"),
                min_cycle=("cycle", "min"),
                max_cycle=("cycle", "max"),
                rows=("cycle", "size"),
            )
            .reset_index()
            .sort_values(["dataset", "battery_id"])
        )
    else:
        cycles_per_battery = pd.DataFrame()

    return dataset_summary, rows_by_level, batteries_per_dataset, cycles_per_battery


def make_missing_report(columns_inventory: pd.DataFrame) -> pd.DataFrame:
    if columns_inventory.empty:
        return columns_inventory

    cols = [
        "column",
        "dtype",
        "missing_count",
        "missing_rate",
        "non_missing_count",
        "inf_count",
        "unique_count",
    ]
    cols = [col for col in cols if col in columns_inventory.columns]
    return columns_inventory[cols].sort_values(["missing_rate", "inf_count"], ascending=[False, False])


def compute_outlier_summary(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    iqr_multiplier: float,
    robust_z_threshold: float,
    min_valid: int,
) -> pd.DataFrame:
    records = []

    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = x.dropna()
        n_valid = int(len(valid))

        record = {
            "column": col,
            "n_valid": n_valid,
            "iqr_lower": np.nan,
            "iqr_upper": np.nan,
            "iqr_outliers": np.nan,
            "iqr_outlier_rate": np.nan,
            "median": np.nan,
            "mad": np.nan,
            "robust_z_outliers": np.nan,
            "robust_z_outlier_rate": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

        if n_valid < min_valid:
            records.append(record)
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        if pd.notna(iqr) and iqr > 0:
            iqr_mask = (x < lower) | (x > upper)
            iqr_count = int(iqr_mask.sum())
        else:
            iqr_count = 0

        median = valid.median()
        mad = (valid - median).abs().median()

        if pd.notna(mad) and mad > 0:
            robust_z = 0.6745 * (x - median) / mad
            rz_mask = robust_z.abs() > robust_z_threshold
            rz_count = int(rz_mask.sum())
        else:
            rz_count = 0

        record.update(
            {
                "iqr_lower": safe_float(lower),
                "iqr_upper": safe_float(upper),
                "iqr_outliers": iqr_count,
                "iqr_outlier_rate": iqr_count / max(n_valid, 1),
                "median": safe_float(median),
                "mad": safe_float(mad),
                "robust_z_outliers": rz_count,
                "robust_z_outlier_rate": rz_count / max(n_valid, 1),
                "min": safe_float(valid.min()),
                "max": safe_float(valid.max()),
            }
        )

        records.append(record)

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["iqr_outlier_rate", "robust_z_outlier_rate"], ascending=[False, False])
    return out


def sample_outlier_rows(
    df: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    top_columns: int,
    rows_per_column: int,
) -> pd.DataFrame:
    if outlier_summary.empty:
        return pd.DataFrame()

    id_cols = get_id_columns_present(df)
    records: List[Dict[str, Any]] = []

    candidates = outlier_summary.dropna(subset=["iqr_lower", "iqr_upper"]).head(top_columns)

    for _, row in candidates.iterrows():
        col = row["column"]
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce")
        mask = (x < row["iqr_lower"]) | (x > row["iqr_upper"])

        if not mask.any():
            continue

        sample = df.loc[mask, id_cols + [col]].head(rows_per_column).copy()

        for _, sample_row in sample.iterrows():
            rec = {
                "outlier_column": col,
                "value": sample_row[col],
                "iqr_lower": row["iqr_lower"],
                "iqr_upper": row["iqr_upper"],
                "method": "iqr",
            }
            for id_col in id_cols:
                rec[id_col] = sample_row[id_col]
            records.append(rec)

    return pd.DataFrame(records)


def filter_for_correlation(
    df: pd.DataFrame,
    feature_level: str,
) -> pd.DataFrame:
    if feature_level.lower() == "all":
        return df.copy()

    if "feature_level" not in df.columns:
        return df.copy()

    selected = df[df["feature_level"].astype(str).str.lower() == feature_level.lower()].copy()

    if selected.empty:
        return df.copy()

    return selected


def is_target_family_column(col: str, target: str) -> bool:
    target_base, _ = infer_feature_family(target)
    col_base, _ = infer_feature_family(col)
    return target_base == col_base


def compute_target_correlations(
    df: pd.DataFrame,
    target: Optional[str],
    feature_level: str,
    exclude_target_family: bool,
    min_valid_pairs: int,
) -> pd.DataFrame:
    if not target or target not in df.columns:
        return pd.DataFrame()

    corr_df = filter_for_correlation(df, feature_level)

    target_series = pd.to_numeric(corr_df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)

    numeric_cols = get_numeric_columns(corr_df)
    records: List[Dict[str, Any]] = []

    for col in numeric_cols:
        if col == target:
            continue

        if exclude_target_family and is_target_family_column(col, target):
            continue

        x = pd.to_numeric(corr_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = x.notna() & target_series.notna()
        n = int(valid.sum())

        if n < min_valid_pairs:
            continue

        if x[valid].nunique(dropna=True) < 2 or target_series[valid].nunique(dropna=True) < 2:
            continue

        pearson = x[valid].corr(target_series[valid], method="pearson")
        spearman = x[valid].corr(target_series[valid], method="spearman")

        records.append(
            {
                "feature": col,
                "target": target,
                "feature_level_filter": feature_level,
                "n_valid_pairs": n,
                "pearson": safe_float(pearson),
                "abs_pearson": abs(safe_float(pearson)) if pd.notna(pearson) else np.nan,
                "spearman": safe_float(spearman),
                "abs_spearman": abs(safe_float(spearman)) if pd.notna(spearman) else np.nan,
            }
        )

    result = pd.DataFrame(records)

    if not result.empty:
        result = result.sort_values(["abs_spearman", "abs_pearson"], ascending=[False, False])

    return result


def compute_target_correlations_by_dataset(
    df: pd.DataFrame,
    target: Optional[str],
    global_corr: pd.DataFrame,
    feature_level: str,
    top_n_features: int,
    min_valid_pairs: int,
) -> pd.DataFrame:
    if not target or target not in df.columns or "dataset" not in df.columns or global_corr.empty:
        return pd.DataFrame()

    top_features = global_corr["feature"].head(top_n_features).tolist()
    corr_df = filter_for_correlation(df, feature_level)

    records: List[Dict[str, Any]] = []

    for dataset, group in corr_df.groupby("dataset", dropna=False):
        y = pd.to_numeric(group[target], errors="coerce").replace([np.inf, -np.inf], np.nan)

        for feature in top_features:
            if feature not in group.columns:
                continue

            x = pd.to_numeric(group[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = x.notna() & y.notna()
            n = int(valid.sum())

            if n < min_valid_pairs:
                continue

            if x[valid].nunique(dropna=True) < 2 or y[valid].nunique(dropna=True) < 2:
                continue

            records.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "target": target,
                    "n_valid_pairs": n,
                    "pearson": safe_float(x[valid].corr(y[valid], method="pearson")),
                    "spearman": safe_float(x[valid].corr(y[valid], method="spearman")),
                }
            )

    result = pd.DataFrame(records)

    if not result.empty:
        result["abs_spearman"] = result["spearman"].abs()
        result = result.sort_values(["dataset", "abs_spearman"], ascending=[True, False])

    return result


def detect_constant_features(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    near_constant_threshold: float,
) -> pd.DataFrame:
    records = []

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = s.dropna()
        n_valid = int(len(valid))

        if n_valid == 0:
            records.append(
                {
                    "column": col,
                    "n_valid": 0,
                    "unique_count": 0,
                    "top_value_rate": np.nan,
                    "status": "all_missing_or_non_finite",
                }
            )
            continue

        unique_count = int(valid.nunique(dropna=True))
        value_counts = valid.value_counts(dropna=True)
        top_rate = float(value_counts.iloc[0] / n_valid) if len(value_counts) else np.nan

        if unique_count <= 1:
            status = "constant"
        elif top_rate >= near_constant_threshold:
            status = "near_constant"
        else:
            status = "ok"

        if status != "ok":
            records.append(
                {
                    "column": col,
                    "n_valid": n_valid,
                    "unique_count": unique_count,
                    "top_value_rate": top_rate,
                    "status": status,
                }
            )

    return pd.DataFrame(records).sort_values(["status", "top_value_rate"], ascending=[True, False]) if records else pd.DataFrame()


def compute_redundant_feature_pairs(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    max_columns: int,
    corr_threshold: float,
    min_valid_rate: float,
) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()

    candidates = []

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_rate = s.notna().mean()
        unique_count = s.nunique(dropna=True)

        if valid_rate >= min_valid_rate and unique_count >= 2:
            candidates.append((valid_rate, unique_count, col))

    candidates = sorted(candidates, reverse=True)
    selected_cols = [col for _, _, col in candidates[:max_columns]]

    if len(selected_cols) < 2:
        return pd.DataFrame()

    matrix = df[selected_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    corr = matrix.corr(method="spearman").abs()

    records = []
    for i, col_a in enumerate(selected_cols):
        for j in range(i + 1, len(selected_cols)):
            col_b = selected_cols[j]
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and value >= corr_threshold:
                records.append(
                    {
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "abs_spearman": safe_float(value),
                    }
                )

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("abs_spearman", ascending=False)

    return result


def make_robustness_checks(
    df: pd.DataFrame,
    columns_inventory: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    constants_df: pd.DataFrame,
    redundant_pairs: pd.DataFrame,
    cycles_per_battery: pd.DataFrame,
    target: Optional[str],
    target_corr: pd.DataFrame,
    duplicate_key_columns: Sequence[str],
    high_missing_threshold: float,
    high_outlier_threshold: float,
) -> pd.DataFrame:
    checks: List[AuditCheck] = []

    checks.append(
        AuditCheck(
            check="row_count",
            status="pass" if len(df) > 0 else "fail",
            metric=len(df),
            threshold="> 0",
            details="Total number of loaded feature rows.",
        )
    )

    existing_key_cols = [col for col in duplicate_key_columns if col in df.columns]
    if existing_key_cols:
        duplicate_count = int(df.duplicated(subset=existing_key_cols).sum())
        checks.append(
            AuditCheck(
                check="duplicate_feature_keys",
                status="pass" if duplicate_count == 0 else "warning",
                metric=duplicate_count,
                threshold=0,
                details=f"Duplicate rows using key columns: {existing_key_cols}",
            )
        )

    if not columns_inventory.empty:
        high_missing = columns_inventory[columns_inventory["missing_rate"] >= high_missing_threshold]
        all_missing = columns_inventory[columns_inventory["missing_rate"] >= 1.0]
        inf_total = int(columns_inventory["inf_count"].fillna(0).sum()) if "inf_count" in columns_inventory else 0

        checks.append(
            AuditCheck(
                check="high_missing_columns",
                status="pass" if len(high_missing) == 0 else "warning",
                metric=len(high_missing),
                threshold=f"< {high_missing_threshold}",
                details="Columns with missing-rate above threshold.",
            )
        )

        checks.append(
            AuditCheck(
                check="all_missing_columns",
                status="pass" if len(all_missing) == 0 else "fail",
                metric=len(all_missing),
                threshold=0,
                details="Columns that are entirely missing.",
            )
        )

        checks.append(
            AuditCheck(
                check="infinite_values",
                status="pass" if inf_total == 0 else "warning",
                metric=inf_total,
                threshold=0,
                details="Total number of infinite numeric values.",
            )
        )

    if "n_samples" in df.columns:
        invalid_samples = int((pd.to_numeric(df["n_samples"], errors="coerce") <= 0).sum())
        checks.append(
            AuditCheck(
                check="invalid_n_samples",
                status="pass" if invalid_samples == 0 else "fail",
                metric=invalid_samples,
                threshold=0,
                details="Rows with n_samples <= 0.",
            )
        )

    if "duration_s" in df.columns:
        duration = pd.to_numeric(df["duration_s"], errors="coerce")
        invalid_duration = int((duration < 0).sum())
        checks.append(
            AuditCheck(
                check="negative_duration",
                status="pass" if invalid_duration == 0 else "fail",
                metric=invalid_duration,
                threshold=0,
                details="Rows with negative duration_s.",
            )
        )

    if {"segment_count", "gap_count"}.issubset(df.columns):
        segment_count = pd.to_numeric(df["segment_count"], errors="coerce")
        gap_count = pd.to_numeric(df["gap_count"], errors="coerce")
        inconsistent = int(((segment_count - gap_count) != 1).fillna(False).sum())
        checks.append(
            AuditCheck(
                check="segment_gap_consistency",
                status="pass" if inconsistent == 0 else "warning",
                metric=inconsistent,
                threshold=0,
                details="Expected segment_count = gap_count + 1 for non-empty rows.",
            )
        )

    if "feature_level" in df.columns:
        levels = sorted(df["feature_level"].astype(str).dropna().unique().tolist())
        required = {"cycle", "phase", "phase_segment"}
        missing_levels = sorted(required - set(levels))
        checks.append(
            AuditCheck(
                check="feature_levels_present",
                status="pass" if not missing_levels else "warning",
                metric=", ".join(levels),
                threshold="cycle, phase, phase_segment",
                details=f"Missing expected levels: {missing_levels}",
            )
        )

    if target:
        target_valid = int(pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum()) if target in df.columns else 0
        checks.append(
            AuditCheck(
                check="target_available",
                status="pass" if target_valid > 0 else "fail",
                metric=f"{target}: {target_valid} valid values",
                threshold="> 0",
                details="Selected target variable availability.",
            )
        )

        checks.append(
            AuditCheck(
                check="target_correlations_available",
                status="pass" if not target_corr.empty else "warning",
                metric=len(target_corr),
                threshold="> 0",
                details="Number of features with valid target correlation.",
            )
        )
    else:
        checks.append(
            AuditCheck(
                check="target_detected",
                status="warning",
                metric="none",
                threshold="valid target column",
                details="No target variable was detected. Use --target explicitly if needed.",
            )
        )

    if not outlier_summary.empty and "iqr_outlier_rate" in outlier_summary:
        high_outliers = outlier_summary[outlier_summary["iqr_outlier_rate"] >= high_outlier_threshold]
        checks.append(
            AuditCheck(
                check="high_outlier_rate_features",
                status="pass" if len(high_outliers) == 0 else "warning",
                metric=len(high_outliers),
                threshold=f"< {high_outlier_threshold}",
                details="Features with IQR outlier rate above threshold.",
            )
        )

    if not constants_df.empty:
        constant_count = int((constants_df["status"] == "constant").sum()) if "status" in constants_df else 0
        near_constant_count = int((constants_df["status"] == "near_constant").sum()) if "status" in constants_df else 0
        checks.append(
            AuditCheck(
                check="constant_or_near_constant_features",
                status="pass" if constant_count == 0 else "warning",
                metric=f"constant={constant_count}, near_constant={near_constant_count}",
                threshold="constant=0",
                details="Constant features are usually not useful for modeling.",
            )
        )

    if not redundant_pairs.empty:
        checks.append(
            AuditCheck(
                check="highly_redundant_feature_pairs",
                status="warning",
                metric=len(redundant_pairs),
                threshold="0 preferred",
                details="Highly correlated feature pairs. Consider feature selection.",
            )
        )
    else:
        checks.append(
            AuditCheck(
                check="highly_redundant_feature_pairs",
                status="pass",
                metric=0,
                threshold="0 preferred",
                details="No highly redundant pairs found under the configured threshold.",
            )
        )

    if not cycles_per_battery.empty:
        low_cycle_batteries = int((cycles_per_battery["n_cycles"] < 2).sum())
        checks.append(
            AuditCheck(
                check="batteries_with_less_than_two_cycles",
                status="pass" if low_cycle_batteries == 0 else "warning",
                metric=low_cycle_batteries,
                threshold=0,
                details="Batteries with fewer than two cycles have limited modeling value.",
            )
        )

    return pd.DataFrame([asdict(check) for check in checks])


# =============================================================================
# Plotting
# =============================================================================

def save_bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
    top_n: Optional[int] = None,
    rotation: int = 45,
) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return

    plot_df = df[[x_col, y_col]].dropna().copy()

    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(y_col, ascending=False)

    if top_n:
        plot_df = plot_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df[x_col].astype(str), plot_df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=rotation)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_boxplot(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    output_path: Path,
) -> None:
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        return

    plot_df = df[[group_col, value_col]].dropna().copy()

    if plot_df.empty:
        return

    groups = []
    labels = []

    for label, group in plot_df.groupby(group_col, dropna=False):
        values = pd.to_numeric(group[value_col], errors="coerce").dropna()
        if not values.empty:
            groups.append(values.to_numpy())
            labels.append(str(label))

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(groups, labels=labels, showfliers=True)
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_histogram(
    series: pd.Series,
    title: str,
    output_path: Path,
    bins: int = 50,
) -> None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(series.name or "value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_correlation_plot(corr_df: pd.DataFrame, output_path: Path, top_n: int) -> None:
    if corr_df.empty or "feature" not in corr_df.columns or "spearman" not in corr_df.columns:
        return

    plot_df = corr_df.head(top_n).copy()
    plot_df = plot_df.sort_values("spearman", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"].astype(str), plot_df["spearman"])
    ax.set_title("Top target correlations by Spearman coefficient")
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_target_scatter_plots(
    df: pd.DataFrame,
    target: Optional[str],
    corr_df: pd.DataFrame,
    feature_level: str,
    output_dir: Path,
    top_n: int,
) -> None:
    if not target or target not in df.columns or corr_df.empty:
        return

    ensure_dir(output_dir)

    plot_df = filter_for_correlation(df, feature_level)

    y = pd.to_numeric(plot_df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)

    for feature in corr_df["feature"].head(top_n):
        if feature not in plot_df.columns:
            continue

        x = pd.to_numeric(plot_df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = x.notna() & y.notna()

        if valid.sum() < 5:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x[valid], y[valid], s=12, alpha=0.65)
        ax.set_title(f"{target} vs {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        fig.tight_layout()

        output_path = output_dir / f"target_vs_{clean_filename(feature)}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def make_plots(
    output_dir: Path,
    dataset_summary: pd.DataFrame,
    batteries_per_dataset: pd.DataFrame,
    cycles_per_battery: pd.DataFrame,
    missing_report: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    df: pd.DataFrame,
    target: Optional[str],
    target_corr: pd.DataFrame,
    correlation_feature_level: str,
    top_n: int,
) -> None:
    ensure_dir(output_dir)

    save_bar_plot(
        dataset_summary,
        "dataset",
        "rows",
        "Rows by dataset",
        output_dir / "rows_by_dataset.png",
        rotation=45,
    )

    save_bar_plot(
        batteries_per_dataset,
        "dataset",
        "n_batteries",
        "Batteries by dataset",
        output_dir / "batteries_by_dataset.png",
        rotation=45,
    )

    save_boxplot(
        cycles_per_battery,
        "dataset",
        "n_cycles",
        "Cycles per battery by dataset",
        output_dir / "cycles_per_battery_boxplot.png",
    )

    if not missing_report.empty and "missing_rate" in missing_report.columns:
        top_missing = missing_report[missing_report["missing_rate"] > 0].head(top_n)
        save_bar_plot(
            top_missing,
            "column",
            "missing_rate",
            "Top missing-value rates",
            output_dir / "missingness_top.png",
            top_n=top_n,
            rotation=75,
        )

    if not outlier_summary.empty and "iqr_outlier_rate" in outlier_summary.columns:
        top_outliers = outlier_summary.dropna(subset=["iqr_outlier_rate"]).head(top_n)
        save_bar_plot(
            top_outliers,
            "column",
            "iqr_outlier_rate",
            "Top IQR outlier rates",
            output_dir / "outlier_rate_top.png",
            top_n=top_n,
            rotation=75,
        )

    if target and target in df.columns:
        save_histogram(
            df[target],
            f"Target distribution: {target}",
            output_dir / "target_distribution.png",
        )

    save_correlation_plot(
        target_corr,
        output_dir / "target_correlations_top.png",
        top_n=top_n,
    )

    save_target_scatter_plots(
        df=df,
        target=target,
        corr_df=target_corr,
        feature_level=correlation_feature_level,
        output_dir=output_dir / "target_scatter",
        top_n=min(8, top_n),
    )


# =============================================================================
# Markdown report
# =============================================================================

def df_to_markdown_preview(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "_No data._"

    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)


def write_markdown_report(
    path: Path,
    args: argparse.Namespace,
    df: pd.DataFrame,
    file_index: pd.DataFrame,
    columns_inventory: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    batteries_per_dataset: pd.DataFrame,
    cycles_per_battery: pd.DataFrame,
    missing_report: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    target: Optional[str],
    target_corr: pd.DataFrame,
    robustness_checks: pd.DataFrame,
    constants_df: pd.DataFrame,
    redundant_pairs: pd.DataFrame,
) -> None:
    n_rows, n_cols = df.shape if not df.empty else (0, 0)

    lines = []
    lines.append("# Extracted Features Audit Report")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Features root: `{args.features_root}`")
    lines.append(f"- Reports root: `{args.reports_root}`")
    lines.append(f"- Loaded files: `{len(file_index)}`")
    lines.append(f"- Loaded rows: `{n_rows}`")
    lines.append(f"- Loaded columns: `{n_cols}`")
    lines.append(f"- Target: `{target if target else 'not detected'}`")
    lines.append(f"- Correlation feature level: `{args.correlation_feature_level}`")
    lines.append("")

    lines.append("## Dataset summary")
    lines.append("")
    lines.append(df_to_markdown_preview(dataset_summary, 20))
    lines.append("")

    lines.append("## Batteries per dataset")
    lines.append("")
    lines.append(df_to_markdown_preview(batteries_per_dataset, 20))
    lines.append("")

    lines.append("## Cycles per battery preview")
    lines.append("")
    lines.append(df_to_markdown_preview(cycles_per_battery, 20))
    lines.append("")

    lines.append("## Top missing values")
    lines.append("")
    missing_cols = ["column", "dtype", "missing_count", "missing_rate", "inf_count", "unique_count"]
    missing_cols = [c for c in missing_cols if c in missing_report.columns]
    lines.append(df_to_markdown_preview(missing_report[missing_cols] if not missing_report.empty else missing_report, 20))
    lines.append("")

    lines.append("## Top outlier rates")
    lines.append("")
    outlier_cols = ["column", "n_valid", "iqr_outliers", "iqr_outlier_rate", "robust_z_outliers", "robust_z_outlier_rate"]
    outlier_cols = [c for c in outlier_cols if c in outlier_summary.columns]
    lines.append(df_to_markdown_preview(outlier_summary[outlier_cols] if not outlier_summary.empty else outlier_summary, 20))
    lines.append("")

    lines.append("## Target correlation preview")
    lines.append("")
    if target:
        corr_cols = ["feature", "target", "n_valid_pairs", "pearson", "spearman", "abs_spearman"]
        corr_cols = [c for c in corr_cols if c in target_corr.columns]
        lines.append(df_to_markdown_preview(target_corr[corr_cols] if not target_corr.empty else target_corr, 20))
    else:
        lines.append("_No target was detected. Use --target explicitly._")
    lines.append("")

    lines.append("## Robustness checks")
    lines.append("")
    lines.append(df_to_markdown_preview(robustness_checks, 50))
    lines.append("")

    lines.append("## Constant or near-constant features")
    lines.append("")
    lines.append(df_to_markdown_preview(constants_df, 20))
    lines.append("")

    lines.append("## Highly redundant feature pairs")
    lines.append("")
    lines.append(df_to_markdown_preview(redundant_pairs, 20))
    lines.append("")

    lines.append("## Generated files")
    lines.append("")
    lines.append("The full CSV reports and plots were saved in the same audit directory.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit extracted hierarchical feature Parquet files."
    )

    parser.add_argument(
        "--features-root",
        type=Path,
        default=DEFAULT_FEATURES_ROOT,
        help=f"Root directory containing *.features.parquet files. Default: {DEFAULT_FEATURES_ROOT}",
    )

    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help=f"Root directory for audit reports. Default: {DEFAULT_REPORTS_ROOT}",
    )

    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset to audit. Can be repeated or comma-separated. If omitted, all datasets are audited.",
    )

    parser.add_argument(
        "--target",
        default="auto",
        help="Target variable for correlation study. Use 'auto' or an explicit column name, e.g. capacity_last.",
    )

    parser.add_argument(
        "--correlation-feature-level",
        default="cycle",
        choices=["cycle", "phase", "phase_segment", "all"],
        help="Feature level used for target correlation. Default: cycle.",
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row sample size after loading. Use this if memory is limited.",
    )

    parser.add_argument(
        "--workers-note",
        default="",
        help="Unused placeholder. This audit script is mostly vectorized and does not use multiprocessing by default.",
    )

    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection. Default: 1.5.",
    )

    parser.add_argument(
        "--robust-z-threshold",
        type=float,
        default=3.5,
        help="Robust z-score threshold using MAD. Default: 3.5.",
    )

    parser.add_argument(
        "--min-valid-outlier",
        type=int,
        default=20,
        help="Minimum valid observations to compute outlier statistics. Default: 20.",
    )

    parser.add_argument(
        "--min-valid-correlation",
        type=int,
        default=20,
        help="Minimum valid pairs for target correlation. Default: 20.",
    )

    parser.add_argument(
        "--high-missing-threshold",
        type=float,
        default=0.50,
        help="Missing-rate threshold used in robustness checks. Default: 0.50.",
    )

    parser.add_argument(
        "--high-outlier-threshold",
        type=float,
        default=0.10,
        help="Outlier-rate threshold used in robustness checks. Default: 0.10.",
    )

    parser.add_argument(
        "--near-constant-threshold",
        type=float,
        default=0.995,
        help="Top-value rate threshold for near-constant features. Default: 0.995.",
    )

    parser.add_argument(
        "--redundancy-corr-threshold",
        type=float,
        default=0.995,
        help="Absolute Spearman threshold for redundant feature pairs. Default: 0.995.",
    )

    parser.add_argument(
        "--max-redundancy-columns",
        type=int,
        default=100,
        help="Maximum number of numeric columns used in redundancy scan. Default: 100.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top items in plots and previews. Default: 30.",
    )

    parser.add_argument(
        "--no-source-file-column",
        action="store_true",
        help="Do not add source_file column to the loaded dataframe.",
    )

    return parser


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = build_arg_parser()
    args = parser.parse_args()

    features_root: Path = args.features_root
    reports_root: Path = args.reports_root

    if not features_root.exists():
        raise FileNotFoundError(f"Features root does not exist: {features_root}")

    run_dir = reports_root / now_id()
    plots_dir = run_dir / "plots"

    ensure_dir(run_dir)
    ensure_dir(plots_dir)

    selected_datasets = parse_list_argument(args.dataset)

    files = discover_feature_files(features_root, selected_datasets)

    if not files:
        raise FileNotFoundError(f"No *.features.parquet files found in {features_root}")

    print("Audit execution plan")
    print("-" * 60)
    print(f"Features root:        {features_root}")
    print(f"Reports directory:    {run_dir}")
    print(f"Datasets:             {', '.join(selected_datasets) if selected_datasets else 'all'}")
    print(f"Feature files found:  {len(files)}")
    print(f"Target:               {args.target}")
    print(f"Correlation level:    {args.correlation_feature_level}")

    df, file_index = read_features(
        files=files,
        features_root=features_root,
        sample_rows=args.sample_rows,
        add_source_file=not args.no_source_file_column,
    )

    print(f"Loaded shape:         {df.shape}")

    write_csv(file_index, run_dir / "feature_files_index.csv")

    numeric_cols = get_numeric_columns(df)
    target = resolve_target(df, args.target)

    columns_inventory = make_columns_inventory(df)
    feature_families = make_feature_families_inventory(numeric_cols)
    missing_report = make_missing_report(columns_inventory)

    dataset_summary, rows_by_level, batteries_per_dataset, cycles_per_battery = make_dataset_summaries(df)

    outlier_summary = compute_outlier_summary(
        df=df,
        numeric_cols=numeric_cols,
        iqr_multiplier=args.iqr_multiplier,
        robust_z_threshold=args.robust_z_threshold,
        min_valid=args.min_valid_outlier,
    )

    outlier_rows = sample_outlier_rows(
        df=df,
        outlier_summary=outlier_summary,
        top_columns=min(20, args.top_n),
        rows_per_column=20,
    )

    target_corr = compute_target_correlations(
        df=df,
        target=target,
        feature_level=args.correlation_feature_level,
        exclude_target_family=True,
        min_valid_pairs=args.min_valid_correlation,
    )

    target_corr_by_dataset = compute_target_correlations_by_dataset(
        df=df,
        target=target,
        global_corr=target_corr,
        feature_level=args.correlation_feature_level,
        top_n_features=min(args.top_n, 30),
        min_valid_pairs=args.min_valid_correlation,
    )

    constants_df = detect_constant_features(
        df=df,
        numeric_cols=numeric_cols,
        near_constant_threshold=args.near_constant_threshold,
    )

    redundant_pairs = compute_redundant_feature_pairs(
        df=filter_for_correlation(df, args.correlation_feature_level),
        numeric_cols=numeric_cols,
        max_columns=args.max_redundancy_columns,
        corr_threshold=args.redundancy_corr_threshold,
        min_valid_rate=0.50,
    )

    duplicate_key_columns = [
        "dataset",
        "battery_id",
        "cycle",
        "phase",
        "feature_level",
        "segment_index",
    ]

    robustness_checks = make_robustness_checks(
        df=df,
        columns_inventory=columns_inventory,
        outlier_summary=outlier_summary,
        constants_df=constants_df,
        redundant_pairs=redundant_pairs,
        cycles_per_battery=cycles_per_battery,
        target=target,
        target_corr=target_corr,
        duplicate_key_columns=duplicate_key_columns,
        high_missing_threshold=args.high_missing_threshold,
        high_outlier_threshold=args.high_outlier_threshold,
    )

    # Write CSV reports.
    write_csv(columns_inventory, run_dir / "columns_inventory.csv")
    write_csv(feature_families, run_dir / "feature_families_inventory.csv")
    write_csv(dataset_summary, run_dir / "dataset_summary.csv")
    write_csv(rows_by_level, run_dir / "rows_by_dataset_level.csv")
    write_csv(batteries_per_dataset, run_dir / "batteries_per_dataset.csv")
    write_csv(cycles_per_battery, run_dir / "cycles_per_battery.csv")
    write_csv(missing_report, run_dir / "missing_values_report.csv")
    write_csv(outlier_summary, run_dir / "outlier_summary.csv")
    write_csv(outlier_rows, run_dir / "outlier_rows_sample.csv")
    write_csv(target_corr, run_dir / "target_correlations.csv")
    write_csv(target_corr_by_dataset, run_dir / "target_correlations_by_dataset.csv")
    write_csv(constants_df, run_dir / "constant_or_near_constant_features.csv")
    write_csv(redundant_pairs, run_dir / "redundant_feature_pairs.csv")
    write_csv(robustness_checks, run_dir / "robustness_checks.csv")

    # Write metadata.
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "features_root": str(features_root),
        "reports_dir": str(run_dir),
        "n_files": len(files),
        "loaded_shape": list(df.shape),
        "target": target,
        "correlation_feature_level": args.correlation_feature_level,
        "numeric_columns": len(numeric_cols),
    }
    (run_dir / "audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Plots.
    make_plots(
        output_dir=plots_dir,
        dataset_summary=dataset_summary,
        batteries_per_dataset=batteries_per_dataset,
        cycles_per_battery=cycles_per_battery,
        missing_report=missing_report,
        outlier_summary=outlier_summary,
        df=df,
        target=target,
        target_corr=target_corr,
        correlation_feature_level=args.correlation_feature_level,
        top_n=args.top_n,
    )

    # Markdown report.
    write_markdown_report(
        path=run_dir / "audit_report.md",
        args=args,
        df=df,
        file_index=file_index,
        columns_inventory=columns_inventory,
        dataset_summary=dataset_summary,
        batteries_per_dataset=batteries_per_dataset,
        cycles_per_battery=cycles_per_battery,
        missing_report=missing_report,
        outlier_summary=outlier_summary,
        target=target,
        target_corr=target_corr,
        robustness_checks=robustness_checks,
        constants_df=constants_df,
        redundant_pairs=redundant_pairs,
    )

    print("\nAudit finished.")
    print("-" * 60)
    print(f"Reports saved to: {run_dir}")
    print(f"Plots saved to:   {plots_dir}")
    print(f"Target used:      {target if target else 'not detected'}")
    print(f"Rows loaded:      {len(df)}")
    print(f"Columns loaded:   {df.shape[1]}")
    print(f"Numeric features: {len(numeric_cols)}")

    if target is None:
        print("\nWARNING: No target variable was detected. Use --target explicitly if needed.")
        print("Examples: --target capacity_last, --target soh_last, --target rul_last")


if __name__ == "__main__":
    main()
