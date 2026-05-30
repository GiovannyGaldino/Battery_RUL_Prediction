"""
Clean and standardize CALCE charge and discharge cycle files.

Inputs:
    data/interim/calce/extracted/calce_charge_extracted_cycles.parquet
    data/interim/calce/extracted/calce_discharge_extracted_cycles.parquet

Outputs:
    data/interim/calce/clean/calce_charge_clean_cycles.parquet
    data/interim/calce/clean/calce_discharge_clean_cycles.parquet
    data/interim/calce/clean/calce_charge_rejected_cycles.csv
    data/interim/calce/clean/calce_discharge_rejected_cycles.csv
    data/interim/calce/clean/calce_cleaning_report.json

The clean parquet files always use the official STANDARD_CYCLE_COLUMNS order.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET = "CALCE"
DATASET_DIRNAME = "calce"
CLEANING_VERSION = "calce_cleaner_v6.0.0"

STANDARD_CYCLE_COLUMNS = [
    "dataset", "source_file", "source_path", "battery_id", "battery_index",
    "cell_id_original", "cycle_index", "cycle_number", "source_cycle_id",
    "cycle_pair_id", "phase",
    "cycle_life", "remaining_cycles", "remaining_cycles_norm", "eol_criterion",
    "eol_capacity_ah", "initial_capacity_ah", "nominal_capacity_ah", "soh",
    "time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah",
    "energy_wh", "power_w",
    "n_points", "duration_s", "sample_period_median_s", "voltage_min_v",
    "voltage_max_v", "voltage_start_v", "voltage_end_v", "current_min_a",
    "current_max_a", "current_mean_a", "temperature_min_c",
    "temperature_max_c", "temperature_mean_c", "phase_capacity_ah",
    "phase_energy_wh",
    "voltage_source", "current_source", "temperature_source", "capacity_source",
    "energy_source",
    "chemistry", "form_factor", "manufacturer", "nominal_voltage_v",
    "ambient_temperature_c", "charge_protocol", "discharge_protocol",
    "c_rate_charge", "c_rate_discharge", "dod", "policy_readable", "barcode",
    "is_valid_cycle", "is_model_usable", "exclude_reason", "cleaning_warnings",
    "n_removed_points", "trim_start_index", "trim_end_index", "has_voltage",
    "has_current", "has_temperature", "has_capacity", "has_energy",
    "adapter_version", "cleaning_version", "created_at",
]

MIN_N_POINTS = 30
EOL_FRACTION = 0.80
EOL_SMOOTHING_WINDOW = 5
CAPACITY_OUTLIER_ROLLING_WINDOW = 9
CAPACITY_OUTLIER_MAD_SCALE = 3.5
MIN_ABS_CAPACITY_OUTLIER_AH = 0.06
DEFAULT_TEMPERATURE_C = 24.0

GLOBAL_VOLTAGE_MIN = 2.45
GLOBAL_VOLTAGE_MAX = 4.35
MIN_VOLTAGE_SPAN = 0.20
DISCHARGE_START_VOLTAGE_MIN = 3.50
DISCHARGE_START_VOLTAGE_MAX = 4.35
DISCHARGE_END_VOLTAGE_MIN = 2.45
DISCHARGE_END_VOLTAGE_MAX = 3.35
DISCHARGE_MIN_VOLTAGE_SPAN = 0.70
CHARGE_START_VOLTAGE_MIN = 2.40
CHARGE_START_VOLTAGE_MAX = 3.80
CHARGE_END_VOLTAGE_MIN = 3.90
CHARGE_END_VOLTAGE_MAX = 4.35
CHARGE_MIN_VOLTAGE_SPAN = 0.50
MIN_CAPACITY_FACTOR = 0.50
MAX_CAPACITY_FACTOR = 1.35
CAPACITY_MEASURED_TO_INTEGRATED_MIN = 0.70
CAPACITY_MEASURED_TO_INTEGRATED_MAX = 1.30


def resolve_project_root(project_root_arg: str | None) -> Path:
    if project_root_arg:
        return Path(project_root_arg).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() or (candidate / "src").exists():
            return candidate
    return current


def as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, np.ndarray):
        return pd.to_numeric(pd.Series(value.tolist()), errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    if isinstance(value, (list, tuple)):
        return pd.to_numeric(pd.Series(list(value)), errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    if isinstance(value, float) and math.isnan(value):
        return np.array([], dtype=float)
    return pd.to_numeric(pd.Series([value]), errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def to_list(values: np.ndarray) -> list[float]:
    return [float(x) if np.isfinite(x) else math.nan for x in np.asarray(values, dtype=float)]


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def safe_stat(values: np.ndarray, name: str) -> float:
    finite = finite_values(values)
    if finite.size == 0:
        return math.nan
    if name == "min":
        return float(np.min(finite))
    if name == "max":
        return float(np.max(finite))
    if name == "mean":
        return float(np.mean(finite))
    if name == "first":
        return float(finite[0])
    if name == "last":
        return float(finite[-1])
    raise ValueError(name)


def sample_period_median(time_s: np.ndarray) -> float:
    finite = finite_values(time_s)
    if finite.size < 2:
        return math.nan
    diffs = np.diff(finite)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size else math.nan


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) == 0:
        return np.array([], dtype=float)
    out = np.zeros(len(y), dtype=float)
    if len(y) == 1:
        return out
    dt = np.diff(x)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)
    avg = (y[1:] + y[:-1]) / 2.0
    avg = np.where(np.isfinite(avg), avg, 0.0)
    out[1:] = np.cumsum(avg * dt)
    return out


def integrate_capacity_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    return cumulative_trapezoid(np.abs(current_a), time_s) / 3600.0


def integrate_energy_wh(time_s: np.ndarray, power_w: np.ndarray) -> np.ndarray:
    return cumulative_trapezoid(power_w, time_s) / 3600.0


def normalize_measured_accumulated(values: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.full(arr.size, np.nan), False
    finite = arr[np.isfinite(arr)]
    normalized = arr - finite[0]
    finite_norm = normalized[np.isfinite(normalized)]
    if finite_norm.size and np.nanmax(finite_norm) <= 0:
        normalized = finite[0] - arr
    finite_norm = normalized[np.isfinite(normalized)]
    if finite_norm.size:
        normalized = normalized - np.nanmin(finite_norm)
    diffs = np.diff(normalized[np.isfinite(normalized)])
    monotonic = bool(diffs.size == 0 or np.nanmin(diffs) >= -1e-6)
    return normalized, monotonic


def align_vectors(*vectors: np.ndarray) -> tuple[list[np.ndarray], int]:
    lengths = [len(v) for v in vectors if len(v) > 0]
    if not lengths:
        return [np.array([], dtype=float) for _ in vectors], 0
    n = min(lengths)
    return [np.asarray(v, dtype=float)[:n] if len(v) else np.full(n, np.nan) for v in vectors], n


def clean_time_and_duplicates(time_s: np.ndarray, vectors: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], int, int, int]:
    n_original = len(time_s)
    if n_original == 0:
        return time_s, vectors, 0, -1, 0
    valid = np.isfinite(time_s)
    for vector in vectors:
        valid &= len(vector) == n_original
    if not valid.any():
        return np.array([], dtype=float), [np.array([], dtype=float) for _ in vectors], 0, -1, n_original
    keep_idx = np.where(valid)[0]
    time_s = time_s[keep_idx]
    vectors = [v[keep_idx] for v in vectors]
    order = np.argsort(time_s)
    time_s = time_s[order]
    vectors = [v[order] for v in vectors]
    _, unique_idx = np.unique(time_s, return_index=True)
    unique_idx = np.sort(unique_idx)
    time_s = time_s[unique_idx]
    vectors = [v[unique_idx] for v in vectors]
    increasing = np.concatenate([[True], np.diff(time_s) > 0])
    time_s = time_s[increasing]
    vectors = [v[increasing] for v in vectors]
    if len(time_s):
        time_s = time_s - time_s[0]
    start_idx = int(keep_idx[order][unique_idx][increasing][0]) if len(time_s) else 0
    end_idx = int(keep_idx[order][unique_idx][increasing][-1]) if len(time_s) else -1
    removed = n_original - len(time_s)
    return time_s, vectors, start_idx, end_idx, removed


def trim_nonphysical_voltage_tail(phase: str, voltage_v: np.ndarray) -> int:
    if len(voltage_v) < MIN_N_POINTS:
        return len(voltage_v)
    finite = np.isfinite(voltage_v)
    if not finite.any():
        return len(voltage_v)
    if phase == "discharge":
        idx = int(np.nanargmin(voltage_v))
        rebound = safe_stat(voltage_v[idx:], "last") - safe_stat(voltage_v[idx:], "min") if idx < len(voltage_v) else 0.0
        if idx < len(voltage_v) - 1 and np.isfinite(rebound) and rebound > 0.03:
            return max(idx + 1, MIN_N_POINTS)
    if phase == "charge":
        idx = int(np.nanargmax(voltage_v))
        drop = safe_stat(voltage_v[idx:], "max") - safe_stat(voltage_v[idx:], "last") if idx < len(voltage_v) else 0.0
        if idx < len(voltage_v) - 1 and np.isfinite(drop) and drop > 0.03:
            return max(idx + 1, MIN_N_POINTS)
    return len(voltage_v)


def time_is_increasing(time_s: np.ndarray) -> bool:
    return len(time_s) >= 2 and np.isfinite(time_s).all() and bool(np.all(np.diff(time_s) > 0))


def has_finite(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).any())


def can_interpolate_qv_curve(voltage_v: np.ndarray, capacity_ah: np.ndarray) -> bool:
    finite = np.isfinite(voltage_v) & np.isfinite(capacity_ah)
    if finite.sum() < 10:
        return False
    return bool((np.nanmax(voltage_v[finite]) - np.nanmin(voltage_v[finite])) > 0.05 and (np.nanmax(capacity_ah[finite]) - np.nanmin(capacity_ah[finite])) > 0)


def nominal_capacity_from_row(row: pd.Series) -> float:
    nominal = pd.to_numeric(pd.Series([row.get("nominal_capacity_ah", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(nominal):
        return float(nominal)
    battery_id = str(row.get("battery_id", "")).upper()
    if "CS2" in battery_id:
        return 1.10
    if "CX2" in battery_id:
        return 1.35
    return math.nan


def physical_window_checks(phase: str, voltage_v: np.ndarray, phase_capacity_ah: float, nominal_capacity_ah: float) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    unusable: list[str] = []
    v_min = safe_stat(voltage_v, "min")
    v_max = safe_stat(voltage_v, "max")
    v_start = safe_stat(voltage_v, "first")
    v_end = safe_stat(voltage_v, "last")
    v_span = v_max - v_min if np.isfinite(v_max) and np.isfinite(v_min) else math.nan

    if not (np.isfinite(v_min) and np.isfinite(v_max) and GLOBAL_VOLTAGE_MIN <= v_min and v_max <= GLOBAL_VOLTAGE_MAX):
        unusable.append("voltage_out_of_global_range")
    if not (np.isfinite(v_span) and v_span >= MIN_VOLTAGE_SPAN):
        unusable.append("voltage_span_too_small")

    if phase == "discharge":
        if not (np.isfinite(v_start) and DISCHARGE_START_VOLTAGE_MIN <= v_start <= DISCHARGE_START_VOLTAGE_MAX):
            unusable.append("discharge_start_voltage_not_near_full_charge")
        if not (np.isfinite(v_end) and DISCHARGE_END_VOLTAGE_MIN <= v_end <= DISCHARGE_END_VOLTAGE_MAX):
            unusable.append("discharge_end_voltage_not_near_cutoff")
        if not (np.isfinite(v_span) and v_span >= DISCHARGE_MIN_VOLTAGE_SPAN):
            unusable.append("discharge_voltage_span_too_small")
    elif phase == "charge":
        if not (np.isfinite(v_start) and CHARGE_START_VOLTAGE_MIN <= v_start <= CHARGE_START_VOLTAGE_MAX):
            warnings.append("charge_start_voltage_outside_typical_range")
        if not (np.isfinite(v_end) and CHARGE_END_VOLTAGE_MIN <= v_end <= CHARGE_END_VOLTAGE_MAX):
            unusable.append("charge_end_voltage_not_near_full_charge")
        if not (np.isfinite(v_span) and v_span >= CHARGE_MIN_VOLTAGE_SPAN):
            unusable.append("charge_voltage_span_too_small")

    if np.isfinite(nominal_capacity_ah):
        q_min = MIN_CAPACITY_FACTOR * nominal_capacity_ah
        q_max = MAX_CAPACITY_FACTOR * nominal_capacity_ah
        if not (np.isfinite(phase_capacity_ah) and q_min <= phase_capacity_ah <= q_max):
            unusable.append("capacity_out_of_plausible_range")
    return warnings, unusable


def clean_single_row(row: pd.Series, phase: str) -> dict[str, Any]:
    warnings: list[str] = []
    reasons: list[str] = []
    base = {col: row[col] if col in row.index else math.nan for col in STANDARD_CYCLE_COLUMNS}
    time_s = as_float_array(row.get("time_s"))
    voltage_v = as_float_array(row.get("voltage_v"))
    current_a = as_float_array(row.get("current_a"))
    temperature_c = as_float_array(row.get("temperature_c"))
    capacity_raw = as_float_array(row.get("capacity_ah"))
    energy_raw = as_float_array(row.get("energy_wh"))

    [time_s, voltage_v, current_a, temperature_c, capacity_raw, energy_raw], _ = align_vectors(
        time_s, voltage_v, current_a, temperature_c, capacity_raw, energy_raw
    )
    time_s, [voltage_v, current_a, temperature_c, capacity_raw, energy_raw], trim_start, trim_end, removed = clean_time_and_duplicates(
        time_s, [voltage_v, current_a, temperature_c, capacity_raw, energy_raw]
    )

    if len(time_s) >= MIN_N_POINTS:
        trim_end_exclusive = trim_nonphysical_voltage_tail(phase, voltage_v)
        if trim_end_exclusive < len(time_s):
            warnings.append("trimmed_nonphysical_voltage_tail")
            removed += len(time_s) - trim_end_exclusive
            time_s = time_s[:trim_end_exclusive]
            voltage_v = voltage_v[:trim_end_exclusive]
            current_a = current_a[:trim_end_exclusive]
            temperature_c = temperature_c[:trim_end_exclusive]
            capacity_raw = capacity_raw[:trim_end_exclusive]
            energy_raw = energy_raw[:trim_end_exclusive]
            trim_end = trim_start + trim_end_exclusive - 1

    if len(time_s):
        time_s = time_s - time_s[0]

    if phase == "charge" and np.nanmean(current_a) < 0:
        current_a = -current_a
        warnings.append("charge_current_sign_flipped")
    if phase == "discharge" and np.nanmean(current_a) > 0:
        current_a = -current_a
        warnings.append("discharge_current_sign_flipped")

    temp_source = str(row.get("temperature_source", "unknown_nan"))
    if not has_finite(temperature_c):
        ambient = pd.to_numeric(pd.Series([row.get("ambient_temperature_c", np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(ambient):
            temperature_c = np.full(len(time_s), float(ambient), dtype=float)
            temp_source = "nominal_constant"
            warnings.append("temperature_filled_with_nominal_constant")
        else:
            temperature_c = np.full(len(time_s), DEFAULT_TEMPERATURE_C, dtype=float)
            temp_source = "literature_constant"
            warnings.append("temperature_filled_with_literature_constant")

    measured_capacity, measured_capacity_monotonic = normalize_measured_accumulated(capacity_raw)
    integrated_capacity = integrate_capacity_ah(time_s, current_a) if has_finite(current_a) and len(time_s) else np.full(len(time_s), np.nan, dtype=float)
    measured_final = safe_stat(measured_capacity, "last") if has_finite(measured_capacity) else math.nan
    integrated_final = safe_stat(integrated_capacity, "last") if has_finite(integrated_capacity) else math.nan
    measured_to_integrated_ratio = measured_final / integrated_final if np.isfinite(measured_final) and np.isfinite(integrated_final) and integrated_final > 0 else math.nan
    capacity_scale_suspect = bool(
        np.isfinite(measured_to_integrated_ratio)
        and (measured_to_integrated_ratio < CAPACITY_MEASURED_TO_INTEGRATED_MIN or measured_to_integrated_ratio > CAPACITY_MEASURED_TO_INTEGRATED_MAX)
    )
    if has_finite(measured_capacity) and measured_capacity_monotonic and safe_stat(measured_capacity, "last") > 0 and not capacity_scale_suspect:
        capacity_ah = measured_capacity
        capacity_source = str(row.get("capacity_source", "measured"))
    elif has_finite(integrated_capacity) and safe_stat(integrated_capacity, "last") > 0:
        capacity_ah = integrated_capacity
        capacity_source = "integrated_current"
        if capacity_scale_suspect:
            warnings.append(f"capacity_reconstructed_from_current_due_to_measured_to_integrated_ratio_{measured_to_integrated_ratio:.3g}")
        elif has_finite(capacity_raw):
            warnings.append("capacity_reconstructed_from_current_due_to_nonmonotonic_or_invalid_measured_capacity")
        else:
            warnings.append("capacity_reconstructed_from_current")
    else:
        capacity_ah = np.full(len(time_s), np.nan, dtype=float)
        capacity_source = "missing"
        reasons.append("missing_current_and_capacity")

    power_w = voltage_v * current_a if has_finite(voltage_v) and has_finite(current_a) else np.full(len(time_s), np.nan, dtype=float)
    if has_finite(power_w):
        energy_wh = integrate_energy_wh(time_s, power_w)
        energy_source = "integrated_power"
    else:
        measured_energy, measured_energy_monotonic = normalize_measured_accumulated(energy_raw)
        if has_finite(measured_energy) and measured_energy_monotonic:
            energy_wh = measured_energy
            energy_source = str(row.get("energy_source", "measured"))
        else:
            energy_wh = np.full(len(time_s), np.nan, dtype=float)
            energy_source = "missing"

    n_points = int(len(time_s))
    duration_s = safe_stat(time_s, "last") - safe_stat(time_s, "first") if n_points else math.nan
    phase_capacity_ah = safe_stat(capacity_ah, "last")
    phase_energy_wh = safe_stat(energy_wh, "last")
    nominal_capacity_ah = nominal_capacity_from_row(row)

    structural_valid = (
        n_points >= MIN_N_POINTS
        and time_is_increasing(time_s)
        and has_finite(voltage_v)
        and has_finite(capacity_ah)
        and np.isfinite(phase_capacity_ah)
        and phase_capacity_ah > 0
        and (safe_stat(voltage_v, "max") > safe_stat(voltage_v, "min"))
        and np.isfinite(duration_s)
        and duration_s > 0
    )

    if n_points < MIN_N_POINTS:
        reasons.append("too_few_points")
    if not time_is_increasing(time_s):
        reasons.append("time_not_strictly_increasing")
    if not has_finite(voltage_v):
        reasons.append("missing_voltage")
    if not has_finite(capacity_ah) or not np.isfinite(phase_capacity_ah) or phase_capacity_ah <= 0:
        reasons.append("missing_or_nonpositive_capacity")
    if not np.isfinite(duration_s) or duration_s <= 0:
        reasons.append("nonpositive_duration")

    physical_warnings, physical_unusable = physical_window_checks(phase, voltage_v, phase_capacity_ah, nominal_capacity_ah)
    warnings.extend(physical_warnings)
    model_unusable_reasons = list(dict.fromkeys(physical_unusable))

    base.update({
        "time_s": to_list(time_s),
        "voltage_v": to_list(voltage_v),
        "current_a": to_list(current_a),
        "temperature_c": to_list(temperature_c),
        "capacity_ah": to_list(capacity_ah),
        "energy_wh": to_list(energy_wh),
        "power_w": to_list(power_w),
        "n_points": n_points,
        "duration_s": float(duration_s) if np.isfinite(duration_s) else math.nan,
        "sample_period_median_s": sample_period_median(time_s),
        "voltage_min_v": safe_stat(voltage_v, "min"),
        "voltage_max_v": safe_stat(voltage_v, "max"),
        "voltage_start_v": safe_stat(voltage_v, "first"),
        "voltage_end_v": safe_stat(voltage_v, "last"),
        "current_min_a": safe_stat(current_a, "min"),
        "current_max_a": safe_stat(current_a, "max"),
        "current_mean_a": safe_stat(current_a, "mean"),
        "temperature_min_c": safe_stat(temperature_c, "min"),
        "temperature_max_c": safe_stat(temperature_c, "max"),
        "temperature_mean_c": safe_stat(temperature_c, "mean"),
        "phase_capacity_ah": phase_capacity_ah,
        "phase_energy_wh": phase_energy_wh,
        "voltage_source": "measured" if has_finite(voltage_v) else "missing",
        "current_source": "measured" if has_finite(current_a) else "missing",
        "temperature_source": temp_source if temp_source in {"measured", "nominal_constant", "literature_constant", "unknown_nan"} else "literature_constant",
        "capacity_source": capacity_source,
        "energy_source": energy_source,
        "nominal_capacity_ah": nominal_capacity_ah,
        "is_valid_cycle": bool(structural_valid),
        "is_model_usable": False,
        "exclude_reason": "accepted" if structural_valid else "|".join(dict.fromkeys(reasons)),
        "cleaning_warnings": sorted(set([str(w) for w in warnings])),
        "n_removed_points": int(removed),
        "trim_start_index": int(trim_start),
        "trim_end_index": int(trim_end),
        "has_voltage": bool(has_finite(voltage_v)),
        "has_current": bool(has_finite(current_a)),
        "has_temperature": bool(str(row.get("temperature_source", "")) == "measured" and has_finite(temperature_c)),
        "has_capacity": bool(has_finite(capacity_ah)),
        "has_energy": bool(has_finite(energy_wh)),
        "cleaning_version": CLEANING_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "_model_unusable_reasons": model_unusable_reasons,
    })
    return base


def robust_mad(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return math.nan
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)))


def add_capacity_outlier_reasons(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["battery_id", "cycle_index"]).copy()
    df["_capacity_outlier"] = False
    for _, idx in df.groupby("battery_id").groups.items():
        g = df.loc[idx].sort_values("cycle_index")
        q = pd.to_numeric(g["phase_capacity_ah"], errors="coerce")
        smooth = q.rolling(window=CAPACITY_OUTLIER_ROLLING_WINDOW, min_periods=1, center=True).median()
        residual = q - smooth
        mad = robust_mad(residual)
        threshold = MIN_ABS_CAPACITY_OUTLIER_AH if not np.isfinite(mad) or mad == 0 else max(MIN_ABS_CAPACITY_OUTLIER_AH, CAPACITY_OUTLIER_MAD_SCALE * 1.4826 * mad)
        outlier = residual.abs() > threshold
        df.loc[g.index, "_capacity_outlier"] = outlier.fillna(False).to_numpy(dtype=bool)
    return df


def assign_lifetime_targets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frames: list[pd.DataFrame] = []
    for battery_id, g in df.groupby("battery_id", sort=True):
        # Preserve extracted row order. CALCE source files are discovered in
        # chronological filename order; sorting by source cycle numbers is unsafe
        # because many converted files reset their internal cycle counter.
        g = g.copy().reset_index(drop=True)
        g["cycle_index"] = np.arange(1, len(g) + 1, dtype=int)
        q = pd.to_numeric(g["phase_capacity_ah"], errors="coerce")
        nominal = pd.to_numeric(g["nominal_capacity_ah"], errors="coerce")
        if nominal.notna().any():
            nominal_value = float(nominal.dropna().iloc[0])
            plausible_mask = (q > 0.25 * nominal_value) & (q < 1.25 * nominal_value)
            valid_q = q[(q.notna()) & plausible_mask]
            if valid_q.empty:
                valid_q = q[(q.notna()) & (q > 0)]
        else:
            valid_q = q[(q.notna()) & (q > 0)]
        if valid_q.empty:
            initial_capacity = math.nan
            eol_capacity = math.nan
            cycle_life = math.nan
            eol_observed = False
        else:
            initial_capacity = float(valid_q.head(5).median())
            eol_capacity = float(EOL_FRACTION * initial_capacity)
            q_smooth = q.rolling(window=EOL_SMOOTHING_WINDOW, min_periods=1).median()
            below = g[(q_smooth <= eol_capacity) & q_smooth.notna()]
            if below.empty:
                cycle_life = math.nan
                eol_observed = False
            else:
                cycle_life = float(below["cycle_index"].iloc[0])
                eol_observed = True
        g["initial_capacity_ah"] = initial_capacity
        g["eol_capacity_ah"] = eol_capacity
        g["cycle_life"] = cycle_life
        g["remaining_cycles"] = g["cycle_life"] - g["cycle_index"]
        g["remaining_cycles_norm"] = g["remaining_cycles"] / g["cycle_life"]
        g["soh"] = g["phase_capacity_ah"] / initial_capacity if np.isfinite(initial_capacity) and initial_capacity > 0 else math.nan
        g["eol_criterion"] = "80pct_initial_capacity" if eol_observed else "80pct_initial_capacity_not_observed"
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def finalize_model_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    model_flags: list[bool] = []
    exclude_reasons: list[str] = []
    for _, row in df.iterrows():
        reasons: list[str] = []
        if not bool(row.get("is_valid_cycle", False)):
            reasons.extend(str(row.get("exclude_reason", "invalid_cycle")).split("|"))
        model_unusable = row.get("_model_unusable_reasons", [])
        if isinstance(model_unusable, list):
            reasons.extend(model_unusable)
        if bool(row.get("_capacity_outlier", False)):
            reasons.append("capacity_outlier")
        if not np.isfinite(float(row.get("cycle_life", math.nan))):
            reasons.append("cycle_life_unknown")
        if not np.isfinite(float(row.get("remaining_cycles", math.nan))) or float(row.get("remaining_cycles", math.nan)) <= 0:
            reasons.append("nonpositive_or_missing_remaining_cycles")
        if not can_interpolate_qv_curve(as_float_array(row.get("voltage_v")), as_float_array(row.get("capacity_ah"))):
            reasons.append("qv_curve_not_interpolable")
        reasons = [r for r in dict.fromkeys([str(r) for r in reasons if str(r) and str(r) != "accepted"])]
        usable = len(reasons) == 0 and str(row.get("phase")) in {"charge", "discharge"}
        model_flags.append(bool(usable))
        exclude_reasons.append("accepted" if usable else "|".join(reasons))
    df["is_model_usable"] = model_flags
    df["exclude_reason"] = exclude_reasons
    return df


def validate_and_order_schema(df: pd.DataFrame) -> pd.DataFrame:
    for column in STANDARD_CYCLE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.Series(dtype="object")
    return df[STANDARD_CYCLE_COLUMNS]


def empty_clean_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)


def clean_phase_file(input_file: Path, output_file: Path, rejected_file: Path, phase: str) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not input_file.exists():
        df_out = empty_clean_frame()
        df_out.to_parquet(output_file, index=False)
        pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS).to_csv(rejected_file, index=False, encoding="utf-8-sig")
        return {"phase": phase, "input_exists": False, "n_input": 0, "n_output": 0, "n_model_usable": 0, "n_rejected": 0}

    df = pd.read_parquet(input_file)
    if df.empty:
        df_out = empty_clean_frame()
    else:
        records = [clean_single_row(row, phase) for _, row in df.iterrows()]
        df_out = pd.DataFrame(records)
        df_out = add_capacity_outlier_reasons(df_out)
        df_out = assign_lifetime_targets(df_out)
        df_out = finalize_model_flags(df_out)
        df_out = df_out.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)
        df_out = validate_and_order_schema(df_out)

    df_out.to_parquet(output_file, index=False)
    rejected_df = df_out[~df_out["is_model_usable"].fillna(False).astype(bool)].copy() if not df_out.empty else pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)
    rejected_df.to_csv(rejected_file, index=False, encoding="utf-8-sig")
    return {
        "phase": phase,
        "input_exists": True,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "rejected_file": str(rejected_file),
        "n_input": int(len(df)),
        "n_output": int(len(df_out)),
        "n_valid": int(df_out["is_valid_cycle"].fillna(False).astype(bool).sum()) if not df_out.empty else 0,
        "n_model_usable": int(df_out["is_model_usable"].fillna(False).astype(bool).sum()) if not df_out.empty else 0,
        "n_rejected": int(len(rejected_df)),
        "exclude_reason_counts": rejected_df["exclude_reason"].value_counts(dropna=False).to_dict() if not rejected_df.empty else {},
    }



def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean CALCE charge and discharge cycle files.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--extracted-dir", default=None)
    parser.add_argument("--clean-dir", default=None)
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    extracted_dir = Path(args.extracted_dir).expanduser().resolve() if args.extracted_dir else project_root / "data" / "interim" / DATASET_DIRNAME / "extracted"
    clean_dir = Path(args.clean_dir).expanduser().resolve() if args.clean_dir else project_root / "data" / "interim" / DATASET_DIRNAME / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for phase in ["charge", "discharge"]:
        summaries.append(clean_phase_file(
            input_file=extracted_dir / f"calce_{phase}_extracted_cycles.parquet",
            output_file=clean_dir / f"calce_{phase}_clean_cycles.parquet",
            rejected_file=clean_dir / f"calce_{phase}_rejected_cycles.csv",
            phase=phase,
        ))

    report = {
        "dataset": DATASET,
        "cleaning_version": CLEANING_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "extracted_dir": str(extracted_dir),
        "clean_dir": str(clean_dir),
        "parameters": {
            "min_n_points": MIN_N_POINTS,
            "eol_fraction": EOL_FRACTION,
            "eol_smoothing_window": EOL_SMOOTHING_WINDOW,
            "capacity_outlier_rolling_window": CAPACITY_OUTLIER_ROLLING_WINDOW,
            "capacity_measured_to_integrated_min": CAPACITY_MEASURED_TO_INTEGRATED_MIN,
            "capacity_measured_to_integrated_max": CAPACITY_MEASURED_TO_INTEGRATED_MAX,
            "capacity_outlier_mad_scale": CAPACITY_OUTLIER_MAD_SCALE,
            "min_abs_capacity_outlier_ah": MIN_ABS_CAPACITY_OUTLIER_AH,
            "global_voltage_min": GLOBAL_VOLTAGE_MIN,
            "global_voltage_max": GLOBAL_VOLTAGE_MAX,
            "min_capacity_factor": MIN_CAPACITY_FACTOR,
            "max_capacity_factor": MAX_CAPACITY_FACTOR,
        },
        "phase_summaries": summaries,
    }
    report_path = clean_dir / "calce_cleaning_report.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(json_safe(report), fp, indent=2, ensure_ascii=False)

    for summary in summaries:
        print(f"[OK] {summary['phase']} clean cycles: {summary.get('output_file')} ({summary['n_output']} rows; {summary['n_model_usable']} model-usable)")
    print(f"[OK] Cleaning report: {report_path}")


if __name__ == "__main__":
    main()
