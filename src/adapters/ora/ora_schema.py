# -*- coding: utf-8 -*-
"""
Shared constants and utility functions for the Oxford/ORA battery adapter.

This module intentionally contains only English identifiers, comments, and
messages. It centralizes the official clean-cycle schema so extraction,
cleaning, auditing, and future merge scripts use the same column order.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import pandas as pd

DATASET_NAME = "OXFORD"
DATASET_FOLDER = "ora"
ADAPTER_VERSION = "ora_adapter_v3"
CLEANING_VERSION = "ora_cleaning_v3"

NOMINAL_CAPACITY_AH = 0.740
EOL_FRACTION = 0.80
EOL_CAPACITY_AH = NOMINAL_CAPACITY_AH * EOL_FRACTION
AMBIENT_TEMPERATURE_C = 40.0
C1_CURRENT_A = 0.740
OCV_CURRENT_A = 0.040

PRIMARY_CHARGE_PHASE = "C1ch"
PRIMARY_DISCHARGE_PHASE = "C1dc"
OPTIONAL_CHARGE_PHASE = "OCVch"
OPTIONAL_DISCHARGE_PHASE = "OCVdc"
SIGNAL_KEYS = ["t", "v", "q", "T"]

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

FLOAT_COLUMNS = [
    "cycle_life", "remaining_cycles", "remaining_cycles_norm", "eol_capacity_ah",
    "initial_capacity_ah", "nominal_capacity_ah", "soh", "duration_s",
    "sample_period_median_s", "voltage_min_v", "voltage_max_v",
    "voltage_start_v", "voltage_end_v", "current_min_a", "current_max_a",
    "current_mean_a", "temperature_min_c", "temperature_max_c",
    "temperature_mean_c", "phase_capacity_ah", "phase_energy_wh",
    "nominal_voltage_v", "ambient_temperature_c", "c_rate_charge",
    "c_rate_discharge", "dod",
]

INT_COLUMNS = [
    "battery_index", "cycle_index", "n_points", "n_removed_points",
    "trim_start_index", "trim_end_index",
]

BOOL_COLUMNS = [
    "is_valid_cycle", "is_model_usable", "has_voltage", "has_current",
    "has_temperature", "has_capacity", "has_energy",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)


def empty_standard_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)


def validate_standard_schema(df: pd.DataFrame, *, allow_extra: bool = False) -> None:
    missing = [column for column in STANDARD_CYCLE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not allow_extra:
        extra = [column for column in df.columns if column not in STANDARD_CYCLE_COLUMNS]
        if extra:
            raise ValueError(f"Unexpected columns found: {extra}")


def reorder_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in STANDARD_CYCLE_COLUMNS:
        if column not in df.columns:
            df[column] = default_value_for_column(column)
    return df[STANDARD_CYCLE_COLUMNS].copy()


def default_value_for_column(column: str) -> Any:
    if column in VECTOR_COLUMNS or column == "cleaning_warnings":
        return []
    if column in FLOAT_COLUMNS:
        return np.nan
    if column in INT_COLUMNS:
        return 0
    if column in BOOL_COLUMNS:
        return False
    return ""


def safe_array(value: Any, *, keep_nan: bool = True) -> np.ndarray:
    """Convert a scalar/list/array value into a one-dimensional float array."""
    try:
        array = np.asarray(value, dtype=float).squeeze()
        array = np.atleast_1d(array).astype(float)
    except Exception:
        return np.array([], dtype=float)

    if not keep_nan:
        array = array[np.isfinite(array)]
    return array


def finite_array(value: Any) -> np.ndarray:
    return safe_array(value, keep_nan=False)


def to_float_list(value: Any, decimals: int | None = None) -> list[float]:
    array = safe_array(value, keep_nan=True)
    if decimals is not None and array.size > 0:
        array = np.round(array, decimals=decimals)
    return array.astype(float).tolist()


def align_vectors(vectors: dict[str, Any], *, allow_nan_keys: Iterable[str] = ()) -> dict[str, np.ndarray]:
    """
    Align vectors by the shortest length and remove rows with invalid values.

    Keys listed in allow_nan_keys are allowed to contain NaN values and therefore
    do not participate in the finite-value mask.
    """
    arrays = {key: safe_array(value, keep_nan=True) for key, value in vectors.items()}
    if not arrays:
        return {}

    min_len = min(len(array) for array in arrays.values())
    if min_len <= 0:
        return {key: np.array([], dtype=float) for key in arrays}

    aligned = {key: array[:min_len].astype(float) for key, array in arrays.items()}
    allow_nan = set(allow_nan_keys)
    mask = np.ones(min_len, dtype=bool)
    for key, array in aligned.items():
        if key in allow_nan:
            mask &= ~np.isinf(array)
        else:
            mask &= np.isfinite(array)

    return {key: array[mask] for key, array in aligned.items()}


def normalize_time_seconds(time_s: Any) -> np.ndarray:
    """
    Convert Oxford time vectors to local seconds.

    The Oxford dataset stores characterization time in fractional days in the
    MAT file. Previous adapter versions only shifted the origin, which produced
    durations around 0.03 instead of approximately 3000 s. This function is also
    defensive for already-extracted files: if the duration is already clearly in
    seconds, it is not rescaled.
    """
    time = finite_array(time_s)
    if time.size == 0:
        return time

    time = time - time[0]
    duration = float(time[-1] - time[0]) if time.size >= 2 else 0.0

    # Oxford raw time is in days. A full C1 phase appears as ~0.03-0.04 days.
    # If the vector already spans hundreds or thousands of units, treat it as
    # seconds and do not rescale.
    if 0.0 < duration < 2.0:
        time = time * 86400.0

    return time.astype(float)


def normalize_capacity_to_ah(capacity: Any) -> np.ndarray:
    """
    Normalize a source capacity vector to non-negative accumulated Ah.

    Oxford q vectors can appear with offsets and phase-dependent sign. This
    routine removes the initial offset, flips the vector if the end is negative,
    converts mAh to Ah when needed, and enforces weak monotonicity.
    """
    q = finite_array(capacity)
    if q.size == 0:
        return q

    q = q - q[0]
    tail = q[-min(20, q.size):]
    if tail.size > 0 and np.nanmedian(tail) < 0:
        q = -q

    q = q - np.nanmin(q)

    if np.nanmax(np.abs(q)) > 5.0:
        q = q / 1000.0

    q = np.maximum.accumulate(q)
    return q.astype(float)


def phase_to_current(phase_name: str, n_points: int) -> np.ndarray:
    phase = phase_name.lower()
    if phase == PRIMARY_CHARGE_PHASE.lower():
        current = C1_CURRENT_A
    elif phase == PRIMARY_DISCHARGE_PHASE.lower():
        current = -C1_CURRENT_A
    elif phase == OPTIONAL_CHARGE_PHASE.lower():
        current = OCV_CURRENT_A
    elif phase == OPTIONAL_DISCHARGE_PHASE.lower():
        current = -OCV_CURRENT_A
    else:
        current = np.nan
    return np.full(int(n_points), current, dtype=float)


def cumulative_trapezoid(y: Sequence[float], x: Sequence[float]) -> np.ndarray:
    y_arr = safe_array(y, keep_nan=False)
    x_arr = safe_array(x, keep_nan=False)
    n = min(y_arr.size, x_arr.size)
    if n <= 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float)
    y_arr = y_arr[:n]
    x_arr = x_arr[:n]
    dx = np.diff(x_arr)
    dx = np.where(dx > 0, dx, 0.0)
    increments = 0.5 * (y_arr[1:] + y_arr[:-1]) * dx
    return np.concatenate([[0.0], np.cumsum(increments)])


def integrate_capacity_from_current(current_a: Sequence[float], time_s: Sequence[float]) -> np.ndarray:
    return cumulative_trapezoid(np.abs(current_a), time_s) / 3600.0


def integrate_energy_from_power(power_w: Sequence[float], time_s: Sequence[float]) -> np.ndarray:
    return cumulative_trapezoid(np.abs(power_w), time_s) / 3600.0


def vector_stats(array_like: Any) -> dict[str, float]:
    array = finite_array(array_like)
    if array.size == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "start": np.nan,
            "end": np.nan,
        }
    return {
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
        "start": float(array[0]),
        "end": float(array[-1]),
    }


def median_sample_period(time_s: Sequence[float]) -> float:
    time = finite_array(time_s)
    if time.size < 2:
        return np.nan
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan
    return float(np.nanmedian(dt))


def infer_default_base_dir() -> Path:
    return Path.cwd()


def make_relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except Exception:
        return str(path)


def can_interpolate_qv_curve(voltage_v: Sequence[float], capacity_ah: Sequence[float], *, min_voltage_range: float = 0.05, min_unique_voltage_points: int = 10) -> bool:
    voltage = finite_array(voltage_v)
    capacity = finite_array(capacity_ah)
    n = min(voltage.size, capacity.size)
    if n < 20:
        return False
    voltage = voltage[:n]
    capacity = capacity[:n]
    if not np.isfinite(voltage).all() or not np.isfinite(capacity).all():
        return False
    if float(np.nanmax(voltage) - np.nanmin(voltage)) < min_voltage_range:
        return False
    unique_voltage = np.unique(np.round(voltage, decimals=5))
    return unique_voltage.size >= min_unique_voltage_points


def ensure_python_scalars(row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned
