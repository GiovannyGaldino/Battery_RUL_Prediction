"""
inspect_features_v3_outputs_patched.py

Robust inspection script for features_hierarchical_v3.

This version avoids crashing when fastparquet fails to read nullable integer columns
with pd.NA/NAType. It reads feature files column-by-column when necessary and
skips only unreadable columns, writing them to parquet_read_failures.csv.

It also plots raw curves from:
    data/interim/charge
    data/interim/discharge

Outputs:
- inspection_report.md
- output_inventory.csv
- loaded_feature_files.csv
- parquet_read_failures.csv
- rows_by_dataset_phase_level.csv
- key_summary.csv
- feature_quality_summary.csv
- range_checks.csv
- target_summary.csv
- target_rows_sample.csv
- consistency_checks.csv
- sample_phase_features.csv
- sample_suspicious_phase_features.csv
- raw_curve_sample_metadata.csv
- curve_plot_index.csv
- capacity_life_plot_index.csv
- plots/raw_curves/*.png
- plots/capacity_life/*.png

Typical use:
& C:\\Users\\Giovanny\\anaconda3\\envs\\batteries\\python.exe ".\\src\\feature_extraction\\inspect_features_v3_outputs.py" --features-root "C:\\Artigo Baterias de Litio\\Batteries\\data\\processed\\features_hierarchical_v3" --interim-root "C:\\Artigo Baterias de Litio\\Batteries\\data\\interim" --reports-root "C:\\Artigo Baterias de Litio\\Batteries\\reports\\features_v3_inspection" --curves-per-dataset-phase 3
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_FEATURES_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/processed/features_hierarchical_v3")
DEFAULT_INTERIM_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/interim")
DEFAULT_REPORTS_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/reports/features_v3_inspection")

VECTOR_COLUMNS = ["time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah", "energy_wh", "power_w"]

KEY_COLUMNS = [
    "dataset", "battery_id", "phase_folder", "cycle_index", "cycle_number",
    "phase", "feature_level", "stage_index", "stage_total", "source_row_id",
]

TARGET_COLUMNS = [
    "target_capacity_ah",
    "target_initial_capacity_ah",
    "target_capacity_ratio_to_initial",
    "target_soh_recomputed",
    "target_soh_recomputed_clipped_0_1p2",
    "target_eol80_found",
    "target_rul80_steps_signed",
    "target_rul80_steps_clipped",
    "target_rul80_cycle_index_signed",
    "target_rul80_cycle_index_clipped",
    "target_eol70_found",
    "target_rul70_steps_signed",
    "target_rul70_steps_clipped",
    "target_rul70_cycle_index_signed",
    "target_rul70_cycle_index_clipped",
]

PHYSICAL_FEATURE_COLUMNS = [
    "time_n_valid", "time_vector_duration_s", "time_dt_median_s", "time_gap_count", "time_segment_count",
    "voltage_first", "voltage_last", "voltage_min", "voltage_max", "voltage_mean",
    "current_min", "current_max", "current_mean",
    "temperature_min", "temperature_max", "temperature_mean",
    "capacity_first", "capacity_last", "capacity_min", "capacity_max", "capacity_delta",
    "energy_first", "energy_last", "energy_min", "energy_max",
    "power_min", "power_max",
    "integral_current_signed_ah", "integral_current_abs_ah",
    "integral_energy_signed_wh_from_vi", "integral_energy_abs_wh_from_vi",
    "integral_power_signed_wh", "integral_power_abs_wh",
]

INPUT_AGG_COLUMNS = [
    "input_n_points",
    "input_voltage_min_v", "input_voltage_max_v", "input_voltage_start_v", "input_voltage_end_v",
    "input_current_min_a", "input_current_max_a", "input_current_mean_a",
    "input_temperature_min_c", "input_temperature_max_c", "input_temperature_mean_c",
    "input_duration_s", "input_phase_capacity_ah", "input_phase_energy_wh",
]

INSPECTION_COLUMNS = list(dict.fromkeys(KEY_COLUMNS + TARGET_COLUMNS + PHYSICAL_FEATURE_COLUMNS + INPUT_AGG_COLUMNS + [
    "source_parquet_file", "source_file_key",
]))

RANGE_RULES = {
    "voltage_min": (0.0, 10.0),
    "voltage_max": (0.0, 10.0),
    "voltage_first": (0.0, 10.0),
    "voltage_last": (0.0, 10.0),
    "current_min": (-500.0, 500.0),
    "current_max": (-500.0, 500.0),
    "current_mean": (-500.0, 500.0),
    "temperature_min": (-50.0, 120.0),
    "temperature_max": (-50.0, 120.0),
    "temperature_mean": (-50.0, 120.0),
    "capacity_min": (-1e-6, 100.0),
    "capacity_max": (-1e-6, 100.0),
    "capacity_last": (-1e-6, 100.0),
    "energy_min": (-1e-6, 1000.0),
    "energy_max": (-1e-6, 1000.0),
    "power_min": (-5000.0, 5000.0),
    "power_max": (-5000.0, 5000.0),
    "target_capacity_ah": (0.0, 100.0),
    "target_initial_capacity_ah": (0.0, 100.0),
    "target_soh_recomputed": (0.0, 1.5),
    "target_soh_recomputed_clipped_0_1p2": (0.0, 1.2),
    "target_rul80_steps_clipped": (0.0, 100000.0),
    "target_rul70_steps_clipped": (0.0, 100000.0),
}


def now_id() -> str:
    return datetime.now().strftime("inspection_%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def parse_list_arg(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(part.upper())
    return list(dict.fromkeys(out))


def safe_name(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value)).strip("_")
    return (value or "item")[:max_len]


def parquet_columns(path: Path) -> List[str]:
    try:
        import fastparquet
        return list(fastparquet.ParquetFile(str(path)).columns)
    except Exception:
        return []


def read_parquet_resilient(path: Path, preferred_columns: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a Parquet file while avoiding crashes caused by fastparquet + pd.NA
    in nullable integer columns.
    """
    failures: List[Dict[str, Any]] = []

    # First try full read.
    if preferred_columns is None:
        try:
            return pd.read_parquet(path, engine="fastparquet"), pd.DataFrame()
        except Exception as exc:
            failures.append({"file": str(path), "column": "__FULL_FILE__", "error": f"{type(exc).__name__}: {exc}"})

    available = parquet_columns(path)
    if not available:
        try:
            return pd.read_parquet(path), pd.DataFrame(failures)
        except Exception as exc:
            failures.append({"file": str(path), "column": "__PANDAS_DEFAULT__", "error": f"{type(exc).__name__}: {exc}"})
            return pd.DataFrame(), pd.DataFrame(failures)

    cols = available if preferred_columns is None else [c for c in preferred_columns if c in available]
    if not cols:
        return pd.DataFrame(), pd.DataFrame(failures)

    # Try selected columns.
    try:
        return pd.read_parquet(path, engine="fastparquet", columns=cols), pd.DataFrame(failures)
    except Exception as exc:
        failures.append({"file": str(path), "column": "__SELECTED_COLUMNS__", "error": f"{type(exc).__name__}: {exc}"})

    # Final fallback: column-by-column.
    frames: List[pd.DataFrame] = []
    n_rows: Optional[int] = None

    for col in cols:
        try:
            part = pd.read_parquet(path, engine="fastparquet", columns=[col])
            frames.append(part)
            n_rows = len(part) if n_rows is None else n_rows
        except Exception as exc:
            failures.append({"file": str(path), "column": col, "error": f"{type(exc).__name__}: {exc}"})

    if not frames:
        return pd.DataFrame(index=range(n_rows or 0)), pd.DataFrame(failures)

    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out, pd.DataFrame(failures)


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in ["dataset", "battery_id", "phase_folder", "phase", "feature_level"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("unknown")
    return df


def parse_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)

    if isinstance(value, np.ndarray):
        raw = value
    elif isinstance(value, (list, tuple, pd.Series)):
        raw = np.asarray(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return np.asarray([], dtype=np.float64)
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
            try:
                raw = np.asarray(ast.literal_eval(text))
            except Exception:
                raw = np.asarray(re.split(r"[,\s;]+", text.strip("[]()")))
        else:
            scalar = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
            if pd.notna(scalar):
                raw = np.asarray([scalar])
            else:
                return np.asarray([], dtype=np.float64)
    else:
        try:
            if pd.isna(value):
                return np.asarray([], dtype=np.float64)
        except Exception:
            pass
        raw = np.asarray([value])

    if raw.ndim > 1:
        raw = raw.reshape(-1)

    arr = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def align_vectors(vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    lengths = [len(v) for v in vectors.values() if isinstance(v, np.ndarray) and len(v) > 0]
    if not lengths:
        return vectors

    n = min(lengths)
    out = {}
    for key, value in vectors.items():
        if isinstance(value, np.ndarray) and len(value) >= n:
            out[key] = value[:n]
        else:
            out[key] = value

    if "time_s" in out and len(out["time_s"]) == n and n > 1:
        order = np.argsort(out["time_s"], kind="mergesort")
        if not np.all(order == np.arange(n)):
            for key, value in list(out.items()):
                if isinstance(value, np.ndarray) and len(value) == n:
                    out[key] = value[order]
    return out


def discover_feature_files(features_root: Path, datasets: Sequence[str], phase_folders: Sequence[str]) -> List[Path]:
    files = sorted(features_root.rglob("*.features.parquet"), key=lambda p: str(p).lower())
    wanted_datasets = {d.lower() for d in datasets}
    wanted_phases = {p.lower() for p in phase_folders}
    out = []
    for path in files:
        parts = [x.lower() for x in path.parts]
        if wanted_datasets and not any(d in parts for d in wanted_datasets):
            continue
        if wanted_phases and not any(p in parts for p in wanted_phases):
            continue
        out.append(path)
    return out


def infer_dataset_phase_from_feature_path(path: Path, features_root: Path) -> Tuple[str, str]:
    try:
        rel = path.relative_to(features_root)
        if len(rel.parts) >= 3:
            return rel.parts[1].upper(), rel.parts[0].lower()
    except Exception:
        pass
    return "UNKNOWN", "unknown"


def create_output_inventory(features_root: Path, files: Sequence[Path]) -> pd.DataFrame:
    records = []
    for path in files:
        dataset, phase = infer_dataset_phase_from_feature_path(path, features_root)
        done_path = Path(str(path) + ".done")
        done_info = {}
        if done_path.exists():
            try:
                done_info = json.loads(done_path.read_text(encoding="utf-8"))
            except Exception:
                done_info = {}
        records.append({
            "dataset": dataset,
            "phase_folder": phase,
            "feature_file": str(path),
            "done": done_path.exists(),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 4),
            "input_rows_from_done": done_info.get("input_rows"),
            "output_rows_from_done": done_info.get("output_rows"),
            "finished_at": done_info.get("finished_at"),
        })
    return pd.DataFrame(records).sort_values(["dataset", "phase_folder", "feature_file"]) if records else pd.DataFrame()


def load_feature_files(files: Sequence[Path], sample_rows: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    inventory_records = []
    failure_frames = []

    for path in files:
        df, failures = read_parquet_resilient(path, preferred_columns=None)

        # If a full read collapses, read only inspection columns.
        if df.empty:
            df, failures2 = read_parquet_resilient(path, preferred_columns=INSPECTION_COLUMNS)
            failures = pd.concat([failures, failures2], ignore_index=True, sort=False)

        if not failures.empty:
            failure_frames.append(failures)

        inventory_records.append({
            "file": str(path),
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 4),
            "has_done_marker": Path(str(path) + ".done").exists(),
            "read_failures": int(len(failures)),
        })

        if not df.empty:
            frames.append(df)

    inventory = pd.DataFrame(inventory_records)
    failures_all = pd.concat(failure_frames, ignore_index=True, sort=False) if failure_frames else pd.DataFrame()

    if not frames:
        return pd.DataFrame(), inventory, failures_all

    data = pd.concat(frames, ignore_index=True, sort=False)
    if sample_rows and len(data) > sample_rows:
        data = data.sample(n=sample_rows, random_state=42).reset_index(drop=True)
    return optimize_memory(data), inventory, failures_all


def create_rows_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["dataset", "phase_folder", "feature_level"] if c in df.columns]
    if df.empty or not cols:
        return pd.DataFrame()
    return df.groupby(cols, dropna=False).size().reset_index(name="rows").sort_values(cols)


def create_key_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["dataset", "phase_folder"] if c in df.columns]
    if df.empty or not cols:
        return pd.DataFrame()
    agg = {"rows": ("dataset", "size")}
    if "battery_id" in df.columns:
        agg["batteries"] = ("battery_id", "nunique")
    if "cycle_index" in df.columns:
        agg["cycles"] = ("cycle_index", "nunique")
    if "feature_level" in df.columns:
        agg["feature_levels"] = ("feature_level", "nunique")
    return df.groupby(cols, dropna=False).agg(**agg).reset_index().sort_values(cols)


def create_feature_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in df.columns:
        s = df[col]
        rec = {
            "column": col,
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()) if len(s) else np.nan,
            "unique_count": np.nan,
            "inf_count": 0,
            "finite_count": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
        }
        try:
            rec["unique_count"] = int(s.nunique(dropna=True))
        except Exception:
            pass
        if pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            arr = x.to_numpy(dtype=np.float64)
            finite = np.isfinite(arr)
            rec["inf_count"] = int(np.isinf(arr).sum())
            rec["finite_count"] = int(finite.sum())
            if finite.sum():
                vals = arr[finite]
                rec["min"] = safe_float(np.min(vals))
                rec["max"] = safe_float(np.max(vals))
                rec["mean"] = safe_float(np.mean(vals))
                rec["median"] = safe_float(np.median(vals))
                rec["std"] = safe_float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        records.append(rec)
    return pd.DataFrame(records).sort_values(["missing_rate", "inf_count"], ascending=[False, False])


def create_range_checks(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col, (low, high) in RANGE_RULES.items():
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = x.dropna()
        n = int(len(valid))
        if n == 0:
            records.append({"column": col, "expected_min": low, "expected_max": high, "n_valid": 0,
                            "below_min": np.nan, "above_max": np.nan, "below_rate": np.nan,
                            "above_rate": np.nan, "observed_min": np.nan, "observed_max": np.nan,
                            "observed_median": np.nan, "status": "no_valid_values"})
            continue
        below = int((valid < low).sum())
        above = int((valid > high).sum())
        records.append({
            "column": col,
            "expected_min": low,
            "expected_max": high,
            "n_valid": n,
            "below_min": below,
            "above_max": above,
            "below_rate": below / n,
            "above_rate": above / n,
            "observed_min": safe_float(valid.min()),
            "observed_max": safe_float(valid.max()),
            "observed_median": safe_float(valid.median()),
            "status": "ok" if below == 0 and above == 0 else "warning",
        })
    return pd.DataFrame(records).sort_values(["status", "above_rate", "below_rate"], ascending=[False, False, False]) if records else pd.DataFrame()


def create_target_report(df: pd.DataFrame, targets_path: Optional[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    if targets_path and targets_path.exists():
        target_df, failures = read_parquet_resilient(targets_path, preferred_columns=None)
        target_df["source_table"] = "targets_v3.parquet"
        frames.append(target_df)

    target_cols = [c for c in TARGET_COLUMNS if c in df.columns]
    key_cols = [c for c in ["dataset", "phase_folder", "battery_id", "cycle_index", "source_row_id"] if c in df.columns]
    if target_cols:
        phase_df = df[df["feature_level"].astype(str) == "phase"].copy() if "feature_level" in df.columns else df.copy()
        feat_target = phase_df[key_cols + target_cols].drop_duplicates()
        feat_target["source_table"] = "feature_phase_rows"
        frames.append(feat_target)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    records = []
    for col in TARGET_COLUMNS:
        if col not in combined.columns:
            continue
        if pd.api.types.is_bool_dtype(combined[col]):
            records.append({
                "target": col,
                "dtype": str(combined[col].dtype),
                "n_rows": len(combined),
                "missing_count": int(combined[col].isna().sum()),
                "missing_rate": float(combined[col].isna().mean()),
                "true_count": int((combined[col] == True).sum()),
                "false_count": int((combined[col] == False).sum()),
                "min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan,
            })
        elif pd.api.types.is_numeric_dtype(combined[col]):
            x = pd.to_numeric(combined[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = x.dropna()
            records.append({
                "target": col,
                "dtype": str(combined[col].dtype),
                "n_rows": len(combined),
                "missing_count": int(x.isna().sum()),
                "missing_rate": float(x.isna().mean()),
                "true_count": np.nan,
                "false_count": np.nan,
                "min": safe_float(valid.min()) if len(valid) else np.nan,
                "max": safe_float(valid.max()) if len(valid) else np.nan,
                "mean": safe_float(valid.mean()) if len(valid) else np.nan,
                "median": safe_float(valid.median()) if len(valid) else np.nan,
            })
    return pd.DataFrame(records), combined


def create_consistency_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = []
    phase_df = df[df["feature_level"].astype(str) == "phase"].copy() if "feature_level" in df.columns else df.copy()

    pairs = [
        ("voltage_min", "input_voltage_min_v", 1e-5),
        ("voltage_max", "input_voltage_max_v", 1e-5),
        ("voltage_first", "input_voltage_start_v", 1e-5),
        ("voltage_last", "input_voltage_end_v", 1e-5),
        ("current_min", "input_current_min_a", 1e-5),
        ("current_max", "input_current_max_a", 1e-5),
        ("current_mean", "input_current_mean_a", 1e-4),
        ("temperature_min", "input_temperature_min_c", 1e-5),
        ("temperature_max", "input_temperature_max_c", 1e-5),
        ("temperature_mean", "input_temperature_mean_c", 1e-4),
        ("time_vector_duration_s", "input_duration_s", 1e-3),
    ]

    for feature_col, input_col, tol in pairs:
        if feature_col not in phase_df.columns or input_col not in phase_df.columns:
            continue
        a = pd.to_numeric(phase_df[feature_col], errors="coerce")
        b = pd.to_numeric(phase_df[input_col], errors="coerce")
        valid = a.notna() & b.notna()
        n = int(valid.sum())
        if n == 0:
            checks.append({"check": f"{feature_col}_vs_{input_col}", "n_valid_pairs": 0,
                           "median_abs_error": np.nan, "p95_abs_error": np.nan, "max_abs_error": np.nan,
                           "fail_count": np.nan, "fail_rate": np.nan, "tolerance": tol, "status": "no_valid_pairs"})
            continue
        err = (a[valid] - b[valid]).abs()
        fail = err > tol
        checks.append({"check": f"{feature_col}_vs_{input_col}", "n_valid_pairs": n,
                       "median_abs_error": safe_float(err.median()),
                       "p95_abs_error": safe_float(err.quantile(0.95)),
                       "max_abs_error": safe_float(err.max()),
                       "fail_count": int(fail.sum()), "fail_rate": float(fail.mean()),
                       "tolerance": tol, "status": "ok" if int(fail.sum()) == 0 else "warning"})

    rel_pairs = [
        ("integral_current_abs_ah", "input_phase_capacity_ah", 0.05),
        ("integral_energy_abs_wh_from_vi", "input_phase_energy_wh", 0.25),
        ("integral_power_abs_wh", "input_phase_energy_wh", 0.25),
    ]

    for feature_col, input_col, rel_tol in rel_pairs:
        if feature_col not in phase_df.columns or input_col not in phase_df.columns:
            continue
        a = pd.to_numeric(phase_df[feature_col], errors="coerce")
        b = pd.to_numeric(phase_df[input_col], errors="coerce")
        valid = a.notna() & b.notna() & (b.abs() > 1e-12)
        n = int(valid.sum())
        if n == 0:
            continue
        rel_err = ((a[valid] - b[valid]).abs() / b[valid].abs())
        fail = rel_err > rel_tol
        checks.append({"check": f"{feature_col}_vs_{input_col}_relative", "n_valid_pairs": n,
                       "median_abs_error": safe_float(rel_err.median()),
                       "p95_abs_error": safe_float(rel_err.quantile(0.95)),
                       "max_abs_error": safe_float(rel_err.max()),
                       "fail_count": int(fail.sum()), "fail_rate": float(fail.mean()),
                       "tolerance": rel_tol, "status": "ok" if float(fail.mean()) <= 0.05 else "warning"})

    return pd.DataFrame(checks).sort_values(["status", "fail_rate"], ascending=[False, False]) if checks else pd.DataFrame()


def discover_interim_files(interim_root: Path, datasets: Sequence[str], phase_folders: Sequence[str]) -> List[Path]:
    wanted = {d.upper() for d in datasets}
    phases = [p.lower() for p in phase_folders] if phase_folders else ["charge", "discharge"]
    out = []
    for phase in phases:
        folder = interim_root / phase
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.parquet"), key=lambda p: str(p).lower()):
            dataset = re.sub(r"_(charge|discharge).*", "", path.stem, flags=re.IGNORECASE).upper()
            if wanted and dataset not in wanted:
                continue
            out.append(path)
    return out


def sample_curve_rows(interim_files: Sequence[Path], curves_per_dataset_phase: int,
                      selected_battery: Optional[str], selected_cycle: Optional[int]) -> pd.DataFrame:
    rows = []
    for path in interim_files:
        cols_avail = parquet_columns(path)
        wanted = [
            "dataset", "battery_id", "cycle_index", "cycle_number", "phase",
            "time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah", "energy_wh", "power_w",
            "phase_capacity_ah", "soh", "remaining_cycles", "remaining_cycles_norm",
        ]
        read_cols = [c for c in wanted if c in cols_avail] if cols_avail else wanted
        df, failures = read_parquet_resilient(path, preferred_columns=read_cols)
        if df.empty:
            continue

        if selected_battery and "battery_id" in df.columns:
            df = df[df["battery_id"].astype(str) == selected_battery]

        if selected_cycle is not None:
            if "cycle_index" in df.columns:
                df = df[pd.to_numeric(df["cycle_index"], errors="coerce") == selected_cycle]
            elif "cycle_number" in df.columns:
                df = df[pd.to_numeric(df["cycle_number"], errors="coerce") == selected_cycle]

        if df.empty:
            continue

        sample = df.head(curves_per_dataset_phase) if selected_battery or selected_cycle is not None else df.sample(n=min(curves_per_dataset_phase, len(df)), random_state=42)
        sample = sample.copy()
        sample["source_file"] = str(path)
        rows.append(sample)

    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def plot_single_curve(row: pd.Series, output_path: Path) -> bool:
    vectors = {col: parse_vector(row[col]) for col in VECTOR_COLUMNS if col in row.index}
    vectors = align_vectors(vectors)
    t = vectors.get("time_s", np.asarray([]))

    if len(t) == 0:
        return False

    dataset = str(row.get("dataset", "unknown"))
    battery = str(row.get("battery_id", "unknown"))
    cycle = row.get("cycle_index", row.get("cycle_number", "unknown"))
    phase = str(row.get("phase", "unknown"))

    series = [
        ("voltage_v", "Voltage [V]"),
        ("current_a", "Current [A]"),
        ("capacity_ah", "Capacity [Ah]"),
        ("temperature_c", "Temperature [C]"),
        ("energy_wh", "Energy [Wh]"),
        ("power_w", "Power [W]"),
    ]

    fig, axes = plt.subplots(len(series), 1, figsize=(10, 14), sharex=True)
    plotted = False
    for ax, (col, ylabel) in zip(axes, series):
        arr = vectors.get(col, np.asarray([]))
        n = min(len(t), len(arr))
        if n >= 2:
            ax.plot(t[:n], arr[:n], linewidth=1.2)
            plotted = True
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{dataset} | {battery} | cycle={cycle} | phase={phase}")
    fig.tight_layout()

    if plotted:
        ensure_dir(output_path.parent)
        fig.savefig(output_path, dpi=160)

    plt.close(fig)
    return plotted


def make_curve_plots(curve_rows: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    records = []
    if curve_rows.empty:
        return pd.DataFrame()

    ensure_dir(output_dir)

    for idx, row in curve_rows.iterrows():
        dataset = str(row.get("dataset", "unknown"))
        battery = str(row.get("battery_id", "unknown"))
        cycle = row.get("cycle_index", row.get("cycle_number", idx))
        phase = str(row.get("phase", "unknown"))
        filename = f"{idx:04d}_{safe_name(dataset)}_{safe_name(battery)}_cycle_{safe_name(str(cycle))}_{safe_name(phase)}.png"
        out = output_dir / filename

        vectors = {col: parse_vector(row[col]) for col in VECTOR_COLUMNS if col in row.index}
        lengths = {f"{col}_length": len(v) for col, v in vectors.items()}
        plotted = plot_single_curve(row, out)

        rec = {
            "plot_file": str(out) if plotted else "",
            "plotted": plotted,
            "dataset": dataset,
            "battery_id": battery,
            "cycle": cycle,
            "phase": phase,
            "source_file": row.get("source_file"),
            "phase_capacity_ah": row.get("phase_capacity_ah", np.nan),
            "input_soh": row.get("soh", np.nan),
            "remaining_cycles": row.get("remaining_cycles", np.nan),
            "remaining_cycles_norm": row.get("remaining_cycles_norm", np.nan),
        }
        rec.update(lengths)
        records.append(rec)

    return pd.DataFrame(records)


def plot_capacity_life(targets_df: pd.DataFrame, output_dir: Path, curves_per_dataset_phase: int) -> pd.DataFrame:
    if targets_df.empty:
        return pd.DataFrame()

    required = {"dataset", "phase_folder", "battery_id", "cycle_index", "target_capacity_ah", "target_soh_recomputed"}
    if not required.issubset(set(targets_df.columns)):
        return pd.DataFrame()

    ensure_dir(output_dir)
    records = []
    counter: Dict[Tuple[str, str], int] = {}

    for (dataset, phase, battery), group in targets_df.groupby(["dataset", "phase_folder", "battery_id"], dropna=False):
        key = (str(dataset), str(phase))
        counter.setdefault(key, 0)
        if counter[key] >= curves_per_dataset_phase:
            continue
        counter[key] += 1

        g = group.sort_values("cycle_index").copy()
        x = pd.to_numeric(g["cycle_index"], errors="coerce")
        cap = pd.to_numeric(g["target_capacity_ah"], errors="coerce")
        soh = pd.to_numeric(g["target_soh_recomputed"], errors="coerce")

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(x, cap, marker="o", markersize=2, linewidth=1.2)
        ax1.set_xlabel("cycle_index")
        ax1.set_ylabel("Capacity [Ah]")
        ax1.grid(True, alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(x, soh, linestyle="--", linewidth=1.2)
        ax2.set_ylabel("Recomputed SOH")

        if "target_eol80_cycle_index" in g.columns:
            eol80 = g["target_eol80_cycle_index"].dropna().unique()
            if len(eol80):
                ax1.axvline(float(eol80[0]), linestyle=":", linewidth=1.2)

        if "target_eol70_cycle_index" in g.columns:
            eol70 = g["target_eol70_cycle_index"].dropna().unique()
            if len(eol70):
                ax1.axvline(float(eol70[0]), linestyle="-.", linewidth=1.2)

        fig.suptitle(f"{dataset} | {phase} | {battery}: capacity and SOH")
        fig.tight_layout()

        output_path = output_dir / f"{safe_name(str(dataset))}_{safe_name(str(phase))}_{safe_name(str(battery))}_capacity_life.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

        records.append({
            "plot_file": str(output_path),
            "dataset": dataset,
            "phase_folder": phase,
            "battery_id": battery,
            "rows": len(g),
            "initial_capacity_ah": safe_float(g["target_initial_capacity_ah"].iloc[0]) if "target_initial_capacity_ah" in g.columns else np.nan,
            "min_soh": safe_float(soh.min()),
            "max_soh": safe_float(soh.max()),
            "last_soh": safe_float(soh.iloc[-1]) if len(soh) else np.nan,
        })

    return pd.DataFrame(records)


def make_sample_tables(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return

    phase_df = df[df["feature_level"].astype(str) == "phase"].copy() if "feature_level" in df.columns else df.copy()

    cols = [
        "dataset", "battery_id", "phase_folder", "cycle_index", "phase", "feature_level",
        "input_n_points", "time_n_valid", "voltage_min", "voltage_max", "voltage_first", "voltage_last",
        "current_min", "current_max", "current_mean",
        "capacity_first", "capacity_last", "capacity_delta",
        "input_phase_capacity_ah", "integral_current_abs_ah",
        "target_capacity_ah", "target_initial_capacity_ah", "target_soh_recomputed",
        "target_rul80_steps_clipped", "target_rul70_steps_clipped",
        "time_gap_count", "time_segment_count",
    ]
    cols = [c for c in cols if c in phase_df.columns]

    if cols:
        write_csv(phase_df[cols].head(200), output_dir / "sample_phase_features.csv")

    issue = pd.Series(False, index=phase_df.index)
    for col, (low, high) in RANGE_RULES.items():
        if col in phase_df.columns:
            x = pd.to_numeric(phase_df[col], errors="coerce")
            issue = issue | (x < low) | (x > high)

    if issue.any() and cols:
        write_csv(phase_df.loc[issue, cols].head(500), output_dir / "sample_suspicious_phase_features.csv")


def df_to_markdown(df: pd.DataFrame, n: int = 20) -> str:
    if df is None or df.empty:
        return "_No data._"
    try:
        return df.head(n).to_markdown(index=False)
    except Exception:
        return df.head(n).to_string(index=False)


def write_report(path: Path, inventory: pd.DataFrame, loaded_files: pd.DataFrame,
                 read_failures: pd.DataFrame, rows_summary: pd.DataFrame,
                 key_summary: pd.DataFrame, quality: pd.DataFrame,
                 ranges: pd.DataFrame, target_summary: pd.DataFrame,
                 consistency: pd.DataFrame, curve_plots: pd.DataFrame,
                 capacity_plots: pd.DataFrame) -> None:
    lines = []
    lines.append("# Features v3 Inspection Report")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    lines.append("## Output inventory")
    lines.append("")
    lines.append(df_to_markdown(inventory, 60))
    lines.append("")

    lines.append("## Loaded feature files")
    lines.append("")
    lines.append(df_to_markdown(loaded_files, 60))
    lines.append("")

    lines.append("## Parquet read failures / skipped columns")
    lines.append("")
    lines.append(df_to_markdown(read_failures, 40))
    lines.append("")

    lines.append("## Rows by dataset / phase / feature level")
    lines.append("")
    lines.append(df_to_markdown(rows_summary, 80))
    lines.append("")

    lines.append("## Key summary")
    lines.append("")
    lines.append(df_to_markdown(key_summary, 60))
    lines.append("")

    lines.append("## Top missing or infinite columns")
    lines.append("")
    cols = ["column", "dtype", "missing_count", "missing_rate", "inf_count", "finite_count", "min", "max", "mean", "median"]
    cols = [c for c in cols if c in quality.columns]
    lines.append(df_to_markdown(quality.sort_values(["missing_rate", "inf_count"], ascending=[False, False])[cols], 50))
    lines.append("")

    lines.append("## Physical and target range checks")
    lines.append("")
    lines.append(df_to_markdown(ranges, 80))
    lines.append("")

    lines.append("## Target summary")
    lines.append("")
    lines.append(df_to_markdown(target_summary, 80))
    lines.append("")

    lines.append("## Consistency checks against input aggregate columns")
    lines.append("")
    lines.append(df_to_markdown(consistency, 80))
    lines.append("")

    lines.append("## Raw curve plot files")
    lines.append("")
    lines.append(df_to_markdown(curve_plots, 80))
    lines.append("")

    lines.append("## Capacity life plot files")
    lines.append("")
    lines.append(df_to_markdown(capacity_plots, 80))
    lines.append("")

    lines.append("## How to inspect")
    lines.append("")
    lines.append("1. Open `sample_phase_features.csv` to check real rows of extracted features.")
    lines.append("2. Open `range_checks.csv` to see voltage/current/temperature/capacity/SOH/RUL warnings.")
    lines.append("3. Open `consistency_checks.csv` to compare extracted vector features against input aggregate columns.")
    lines.append("4. Open PNGs in `plots/raw_curves` to visually inspect charge/discharge curves.")
    lines.append("5. Open PNGs in `plots/capacity_life` to inspect capacity degradation and SOH targets.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect generated v3 feature files and plot raw charge/discharge curves.")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--phase-folder", action="append", default=None)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--curves-per-dataset-phase", type=int, default=3)
    parser.add_argument("--selected-battery", default=None)
    parser.add_argument("--selected-cycle", type=int, default=None)
    parser.add_argument("--skip-curve-plots", action="store_true")
    parser.add_argument("--skip-capacity-plots", action="store_true")
    args = parser.parse_args()

    datasets = parse_list_arg(args.dataset)
    phase_folders = [p.lower() for p in parse_list_arg(args.phase_folder)]

    run_dir = args.reports_root / now_id()
    plots_dir = run_dir / "plots"
    ensure_dir(run_dir)
    ensure_dir(plots_dir)

    if not args.features_root.exists():
        raise FileNotFoundError(f"Features root not found: {args.features_root}")

    feature_files = discover_feature_files(args.features_root, datasets, phase_folders)
    if not feature_files:
        raise FileNotFoundError(f"No *.features.parquet files found in {args.features_root}")

    print("Features v3 inspection")
    print("-" * 60)
    print(f"Features root: {args.features_root}")
    print(f"Interim root:   {args.interim_root}")
    print(f"Reports dir:    {run_dir}")
    print(f"Feature files:  {len(feature_files)}")
    print(f"Datasets:       {', '.join(datasets) if datasets else 'all'}")
    print(f"Phase folders:  {', '.join(phase_folders) if phase_folders else 'all'}")

    inventory = create_output_inventory(args.features_root, feature_files)
    write_csv(inventory, run_dir / "output_inventory.csv")

    df, loaded_files, read_failures = load_feature_files(feature_files, args.sample_rows)
    write_csv(loaded_files, run_dir / "loaded_feature_files.csv")
    write_csv(read_failures, run_dir / "parquet_read_failures.csv")

    rows_summary = create_rows_summary(df)
    key_summary = create_key_summary(df)
    quality = create_feature_quality_report(df)
    ranges = create_range_checks(df)

    targets_path = args.features_root / "_targets" / "targets_v3.parquet"
    target_summary, targets_combined = create_target_report(df, targets_path if targets_path.exists() else None)
    consistency = create_consistency_checks(df)

    write_csv(rows_summary, run_dir / "rows_by_dataset_phase_level.csv")
    write_csv(key_summary, run_dir / "key_summary.csv")
    write_csv(quality, run_dir / "feature_quality_summary.csv")
    write_csv(ranges, run_dir / "range_checks.csv")
    write_csv(target_summary, run_dir / "target_summary.csv")
    write_csv(targets_combined.head(100000), run_dir / "target_rows_sample.csv")
    write_csv(consistency, run_dir / "consistency_checks.csv")

    make_sample_tables(df, run_dir)

    curve_plots = pd.DataFrame()
    if not args.skip_curve_plots:
        interim_files = discover_interim_files(args.interim_root, datasets, phase_folders)
        curve_rows = sample_curve_rows(interim_files, args.curves_per_dataset_phase, args.selected_battery, args.selected_cycle)
        meta = curve_rows.drop(columns=[c for c in VECTOR_COLUMNS if c in curve_rows.columns], errors="ignore")
        write_csv(meta, run_dir / "raw_curve_sample_metadata.csv")
        curve_plots = make_curve_plots(curve_rows, plots_dir / "raw_curves")
        write_csv(curve_plots, run_dir / "curve_plot_index.csv")

    capacity_plots = pd.DataFrame()
    if not args.skip_capacity_plots and targets_path.exists():
        targets_df, target_read_failures = read_parquet_resilient(targets_path, preferred_columns=None)
        if datasets and "dataset" in targets_df.columns:
            targets_df = targets_df[targets_df["dataset"].astype(str).str.upper().isin(datasets)]
        if phase_folders and "phase_folder" in targets_df.columns:
            targets_df = targets_df[targets_df["phase_folder"].astype(str).str.lower().isin(phase_folders)]
        capacity_plots = plot_capacity_life(targets_df, plots_dir / "capacity_life", args.curves_per_dataset_phase)
        write_csv(capacity_plots, run_dir / "capacity_life_plot_index.csv")

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "features_root": str(args.features_root),
        "interim_root": str(args.interim_root),
        "reports_dir": str(run_dir),
        "feature_files": len(feature_files),
        "loaded_shape": list(df.shape),
        "datasets": datasets,
        "phase_folders": phase_folders,
        "sample_rows": args.sample_rows,
        "curves_per_dataset_phase": args.curves_per_dataset_phase,
    }
    (run_dir / "inspection_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(
        path=run_dir / "inspection_report.md",
        inventory=inventory,
        loaded_files=loaded_files,
        read_failures=read_failures,
        rows_summary=rows_summary,
        key_summary=key_summary,
        quality=quality,
        ranges=ranges,
        target_summary=target_summary,
        consistency=consistency,
        curve_plots=curve_plots,
        capacity_plots=capacity_plots,
    )

    print("\nInspection finished.")
    print("-" * 60)
    print(f"Reports saved to: {run_dir}")
    print(f"Plots saved to:   {plots_dir}")
    print(f"Loaded feature shape: {df.shape}")
    print(f"Parquet read failures/skipped columns: {len(read_failures)}")

    if not ranges.empty:
        print(f"Range warnings: {len(ranges[ranges['status'] != 'ok'])}")

    if not quality.empty:
        print(f"Columns with >50% missing: {len(quality[quality['missing_rate'] > 0.50])}")


if __name__ == "__main__":
    main()
