# -*- coding: utf-8 -*-
"""
Shared schema, path, and utility functions for the XJTU battery-cycle adapter.

The clean-cycle outputs follow the project-wide standard schema used for the
charge and discharge branches. Keep this module in the same folder as the XJTU
adapter scripts.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

DATASET_NAME = "XJTU"
DATASET_NAME_LOWER = "xjtu"
ADAPTER_VERSION = "xjtu_adapter_v1.3"
CLEANING_VERSION = "xjtu_cleaning_v1.3"

STANDARD_CYCLE_COLUMNS = [
    # Identification
    "dataset", "source_file", "source_path", "battery_id", "battery_index",
    "cell_id_original", "cycle_index", "cycle_number", "source_cycle_id",
    "cycle_pair_id", "phase",

    # Lifetime and target
    "cycle_life", "remaining_cycles", "remaining_cycles_norm", "eol_criterion",
    "eol_capacity_ah", "initial_capacity_ah", "nominal_capacity_ah", "soh",

    # Vector signals
    "time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah",
    "energy_wh", "power_w",

    # Scalar values derived from signals
    "n_points", "duration_s", "sample_period_median_s", "voltage_min_v",
    "voltage_max_v", "voltage_start_v", "voltage_end_v", "current_min_a",
    "current_max_a", "current_mean_a", "temperature_min_c",
    "temperature_max_c", "temperature_mean_c", "phase_capacity_ah",
    "phase_energy_wh",

    # Signal source metadata
    "voltage_source", "current_source", "temperature_source", "capacity_source",
    "energy_source",

    # Experimental protocol
    "chemistry", "form_factor", "manufacturer", "nominal_voltage_v",
    "ambient_temperature_c", "charge_protocol", "discharge_protocol",
    "c_rate_charge", "c_rate_discharge", "dod", "policy_readable", "barcode",

    # Quality and traceability
    "is_valid_cycle", "is_model_usable", "exclude_reason", "cleaning_warnings",
    "n_removed_points", "trim_start_index", "trim_end_index", "has_voltage",
    "has_current", "has_temperature", "has_capacity", "has_energy",
    "adapter_version", "cleaning_version", "created_at",
]

VECTOR_COLUMNS = [
    "time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah",
    "energy_wh", "power_w",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_project_root(start: Path | None = None) -> Path:
    """Find the Batteries project root from the current script location."""
    if start is None:
        start = Path(__file__).resolve()

    candidates = [start]
    candidates.extend(start.parents)

    for candidate in candidates:
        if candidate.name.lower() == "batteries":
            return candidate
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate

    try:
        return Path(__file__).resolve().parents[3]
    except Exception:
        return Path.cwd()


PROJECT_ROOT = find_project_root()


def default_raw_dir() -> Path:
    return PROJECT_ROOT / "data" / "raw" / "XJTU"


def default_extracted_dir() -> Path:
    return PROJECT_ROOT / "data" / "interim" / DATASET_NAME_LOWER / "extracted"


def default_clean_dir() -> Path:
    return PROJECT_ROOT / "data" / "interim" / DATASET_NAME_LOWER / "clean"


def default_report_dir() -> Path:
    return PROJECT_ROOT / "data" / "reports" / DATASET_NAME_LOWER


def extracted_file(phase: str, extracted_dir: Path | None = None) -> Path:
    out_dir = extracted_dir if extracted_dir is not None else default_extracted_dir()
    return out_dir / f"{DATASET_NAME_LOWER}_{phase}_extracted_cycles.parquet"


def clean_file(phase: str, clean_dir: Path | None = None) -> Path:
    out_dir = clean_dir if clean_dir is not None else default_clean_dir()
    return out_dir / f"{DATASET_NAME_LOWER}_{phase}_clean_cycles.parquet"


def rejected_file(phase: str, clean_dir: Path | None = None) -> Path:
    out_dir = clean_dir if clean_dir is not None else default_clean_dir()
    return out_dir / f"{DATASET_NAME_LOWER}_{phase}_rejected_cycles.csv"


def empty_standard_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)


def ensure_standard_columns(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Return a dataframe with all standard columns in the official order."""
    out = df.copy()
    for col in STANDARD_CYCLE_COLUMNS:
        if col not in out.columns:
            default = default_value_for_column(col)
            if isinstance(default, list) and len(out) > 0:
                out[col] = [list(default) for _ in range(len(out))]
            else:
                out[col] = default

    if strict:
        extra = [col for col in out.columns if col not in STANDARD_CYCLE_COLUMNS]
        if extra:
            raise ValueError(f"Unexpected columns found: {extra}")

    return out[STANDARD_CYCLE_COLUMNS]


def validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in STANDARD_CYCLE_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in STANDARD_CYCLE_COLUMNS]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected columns found: {extra}")


def default_value_for_column(col: str) -> Any:
    if col in VECTOR_COLUMNS:
        return []
    if col == "cleaning_warnings":
        return []
    if col in {"is_valid_cycle", "is_model_usable", "has_voltage", "has_current", "has_temperature", "has_capacity", "has_energy"}:
        return False
    if col in {"dataset"}:
        return DATASET_NAME
    if col in {"phase", "source_file", "source_path", "battery_id", "cell_id_original", "source_cycle_id", "cycle_pair_id", "eol_criterion", "exclude_reason", "voltage_source", "current_source", "temperature_source", "capacity_source", "energy_source", "chemistry", "form_factor", "manufacturer", "charge_protocol", "discharge_protocol", "policy_readable", "barcode", "adapter_version", "cleaning_version", "created_at"}:
        return ""
    if col in {"battery_index", "cycle_index", "n_points", "n_removed_points", "trim_start_index", "trim_end_index"}:
        return 0
    return np.nan


def as_1d_float(value: Any) -> np.ndarray:
    """Convert MATLAB, pandas, list, tuple, or stringified-list values to 1D float arrays."""
    if value is None:
        return np.array([], dtype=float)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.array([], dtype=float)
        try:
            import ast
            value = ast.literal_eval(text)
        except Exception:
            return np.array([], dtype=float)

    try:
        arr = np.asarray(value, dtype=float).squeeze()
        arr = np.atleast_1d(arr).astype(float)
        return arr
    except Exception:
        return np.array([], dtype=float)


def as_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        arr = np.asarray(value).squeeze()
        if arr.dtype.kind in {"U", "S", "O"}:
            if arr.size == 1:
                return str(arr.item())
            return " ".join(str(x) for x in arr.ravel())
        if arr.dtype.kind in {"i", "u"} and arr.size > 1:
            chars = []
            for x in arr.ravel():
                ix = int(x)
                if ix > 0:
                    chars.append(chr(ix))
            if chars:
                return "".join(chars)
        if arr.size == 1:
            return str(arr.item())
        return str(arr.tolist())
    except Exception:
        return str(value)


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(np.asarray(value).squeeze())
        return out if np.isfinite(out) else default
    except Exception:
        return default


def safe_min(arr: Iterable[float] | np.ndarray) -> float:
    x = as_1d_float(arr)
    x = x[np.isfinite(x)]
    return float(np.min(x)) if x.size else np.nan


def safe_max(arr: Iterable[float] | np.ndarray) -> float:
    x = as_1d_float(arr)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if x.size else np.nan


def safe_mean(arr: Iterable[float] | np.ndarray) -> float:
    x = as_1d_float(arr)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def safe_median_diff(time_s: np.ndarray) -> float:
    t = as_1d_float(time_s)
    if t.size < 2:
        return np.nan
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if dt.size else np.nan


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = as_1d_float(y)
    x = as_1d_float(x)
    n = min(len(y), len(x))
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([0.0], dtype=float)
    y = y[:n]
    x = x[:n]
    dx = np.diff(x)
    dx = np.where(np.isfinite(dx), dx, 0.0)
    dx = np.maximum(dx, 0.0)
    increments = 0.5 * (y[1:] + y[:-1]) * dx
    return np.concatenate([[0.0], np.cumsum(increments)]).astype(float)


def cumulative_capacity_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    """Capacity in Ah, accumulated from |I| over local phase time."""
    current = np.abs(as_1d_float(current_a))
    time = as_1d_float(time_s)
    return cumulative_trapezoid(current, time) / 3600.0


def cumulative_energy_wh(time_s: np.ndarray, voltage_v: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    """Signed accumulated energy in Wh, computed from P = V * I."""
    time = as_1d_float(time_s)
    voltage = as_1d_float(voltage_v)
    current = as_1d_float(current_a)
    n = min(len(time), len(voltage), len(current))
    if n == 0:
        return np.array([], dtype=float)
    power = voltage[:n] * current[:n]
    return cumulative_trapezoid(power, time[:n]) / 3600.0


def compute_power(voltage_v: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    voltage = as_1d_float(voltage_v)
    current = as_1d_float(current_a)
    n = min(len(voltage), len(current))
    if n == 0:
        return np.array([], dtype=float)
    return (voltage[:n] * current[:n]).astype(float)


def vector_has_finite(arr: Any) -> bool:
    x = as_1d_float(arr)
    return bool(x.size and np.any(np.isfinite(x)))


def all_vector_lengths(row: pd.Series | dict[str, Any]) -> list[int]:
    lengths = []
    for col in VECTOR_COLUMNS:
        value = row[col] if isinstance(row, pd.Series) and col in row.index else row.get(col, [])
        lengths.append(len(as_1d_float(value)))
    return lengths


def vector_lengths_equal(row: pd.Series | dict[str, Any]) -> bool:
    lengths = [length for length in all_vector_lengths(row) if length > 0]
    return bool(lengths and len(set(lengths)) == 1)


def to_builtin(obj: Any) -> Any:
    """Convert numpy/pandas values to JSON-compatible Python builtins."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_builtin(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, indent=2, ensure_ascii=False)
