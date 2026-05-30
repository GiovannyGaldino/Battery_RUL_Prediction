#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared constants and utility functions for the HUST lithium-ion battery adapter.

Version: 1.0.0
This module is intentionally lightweight and contains no dataset execution logic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET_NAME = "HUST"
ADAPTER_VERSION = "hust_adapter_v1.0.0"
CLEANING_VERSION = "hust_cleaning_v1.0.0"
DEFAULT_BASE_DIR = Path(
    r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries"
)
DEFAULT_TEMPERATURE_C = 30.0

# Conservative cleaning criteria. These are structural criteria, not aggressive
# physics-based filtering rules. The HUST dataset may contain diverse profiles.
MIN_POINTS = 20
MIN_DURATION_S = 1.0
MIN_VOLTAGE_SPAN_V = 0.03
MIN_CAPACITY_AH = 1e-6
MIN_CELL_VOLTAGE_V = 0.0
MAX_CELL_VOLTAGE_V = 5.5

RAW_REQUIRED_COLUMNS = {
    "Status",
    "Current (mA)",
    "Voltage (V)",
    "Capacity (mAh)",
    "Time (s)",
}

TEMPERATURE_CANDIDATES = [
    "Temperature (C)",
    "Temperature(C)",
    "Temperature (°C)",
    "Temperature(°C)",
    "Temp (C)",
    "Temp(C)",
    "Temp (°C)",
    "Temp(°C)",
    "Temperature",
    "Temp",
]

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

EXTRACTED_COLUMNS = [
    "dataset", "source_file", "source_path", "battery_id", "battery_index",
    "cell_id_original", "cycle_number", "source_cycle_id", "cycle_pair_id",
    "phase", "remaining_cycles_raw", "cycle_life_raw", "summary_capacity_ah",
    "time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah",
    "temperature_source", "capacity_source", "status_values", "n_points_raw",
    "extract_warnings", "adapter_version", "created_at",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_tqdm():
    """Return tqdm if installed; otherwise return a no-op wrapper."""
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm
    except Exception:
        def _tqdm(iterable=None, **_: Any):
            return iterable if iterable is not None else []
        return _tqdm


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def dump_json(data: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)


def resolve_base_paths(args: Any) -> dict[str, Path]:
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else DEFAULT_BASE_DIR
    raw_dir = Path(args.raw_dir).expanduser().resolve() if getattr(args, "raw_dir", None) else base_dir / "data" / "raw" / "HUST"
    interim_dir = Path(args.interim_dir).expanduser().resolve() if getattr(args, "interim_dir", None) else base_dir / "data" / "interim" / "hust"
    reports_dir = Path(args.reports_dir).expanduser().resolve() if getattr(args, "reports_dir", None) else base_dir / "data" / "reports" / "hust"
    return {
        "base_dir": base_dir,
        "raw_dir": raw_dir,
        "interim_dir": interim_dir,
        "extracted_dir": interim_dir / "extracted",
        "clean_dir": interim_dir / "clean",
        "reports_dir": reports_dir,
    }


def empty_extracted_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=EXTRACTED_COLUMNS)


def empty_clean_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df.loc[:, columns].copy()


def write_parquet(df: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)


def read_parquet_if_exists(file_path: Path, expected_columns: list[str] | None = None) -> pd.DataFrame:
    if not file_path.exists():
        if expected_columns is None:
            return pd.DataFrame()
        return pd.DataFrame(columns=expected_columns)
    return pd.read_parquet(file_path)


def safe_numeric_array(values: Any) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    try:
        arr = np.asarray(values, dtype=float).squeeze()
    except Exception:
        return np.array([], dtype=float)
    return np.atleast_1d(arr).astype(float)


def finite_or_nan_vector(values: Any, length: int | None = None) -> np.ndarray:
    arr = safe_numeric_array(values)
    if length is not None:
        if arr.size == 0:
            return np.full(length, np.nan, dtype=float)
        arr = arr[:length]
        if arr.size < length:
            arr = np.concatenate([arr, np.full(length - arr.size, np.nan)])
    return arr


def listify(arr: Any) -> list[float]:
    out = safe_numeric_array(arr)
    return [float(x) if np.isfinite(x) else float("nan") for x in out]


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return x if np.isfinite(x) else default


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def get_dict_value_flexible(d: Any, key: Any) -> Any:
    if not isinstance(d, dict):
        return None
    candidates = [key, str(key)]
    try:
        candidates.extend([int(float(key)), float(key), str(int(float(key)))])
    except Exception:
        pass
    for candidate in candidates:
        if candidate in d:
            return d[candidate]
    return None


def estimate_cycle_life_from_rul(rul_dict: Any) -> float:
    if not isinstance(rul_dict, dict):
        return np.nan
    values: list[float] = []
    for cycle_key, rul_value in rul_dict.items():
        cycle_number = safe_float(cycle_key)
        remaining = safe_float(rul_value)
        if np.isfinite(cycle_number) and np.isfinite(remaining):
            values.append(cycle_number + remaining)
    return float(max(values)) if values else np.nan


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def describe_raw_dir(raw_dir: Path, max_items: int = 80) -> str:
    if not raw_dir.exists():
        return f"Raw directory does not exist: {raw_dir}"
    children = sorted(raw_dir.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    lines = [f"No .pkl/.pickle/.pk files were found under: {raw_dir}"]
    if children:
        lines.append("Directory content:")
        for item in children[:max_items]:
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            lines.append(f"  {prefix} {item.name}")
        if len(children) > max_items:
            lines.append(f"  ... {len(children) - max_items} additional items omitted")
    return "\n".join(lines)
