"""
Clean NASA Random Walk (NRW) extracted charge and discharge cycles.

The cleaner reads phase-specific extracted parquet files and writes clean files
that follow the official cycle schema in the exact standard column order.

Default inputs:
    data/interim/nrw/extracted/nrw_charge_extracted_cycles.parquet
    data/interim/nrw/extracted/nrw_discharge_extracted_cycles.parquet

Default outputs:
    data/interim/nrw/clean/nrw_charge_clean_cycles.parquet
    data/interim/nrw/clean/nrw_discharge_clean_cycles.parquet
    data/interim/nrw/clean/nrw_charge_rejected_cycles.csv
    data/interim/nrw/clean/nrw_discharge_rejected_cycles.csv

Run:
    python src/adapters/nrw/clean_nrw_cycles.py
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable


DATASET_CODE = "NRW"
DATASET_FOLDER = "nrw"
CLEANING_VERSION = "nrw_cleaner_v1.0"

DEFAULT_BASE_DIR = Path(
    r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries"
)

NOMINAL_CAPACITY_AH = 2.1
NOMINAL_VOLTAGE_V = 3.7
MIN_VALID_POINTS = 20
MIN_QV_VOLTAGE_RANGE = 0.10
DEFAULT_EOL_RATIO = 0.70
DEFAULT_EOL_CONSECUTIVE_CYCLES = 2

TEMP_MIN_VALID_C = -10.0
TEMP_MAX_VALID_C = 80.0
VOLTAGE_MIN_PHYSICAL_V = 1.5
VOLTAGE_MAX_PHYSICAL_V = 4.5
MAX_REASONABLE_PHASE_CAPACITY_AH = 3.5

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, np.ndarray):
        arr = value
    else:
        try:
            arr = np.asarray(value, dtype=object)
        except Exception:
            return np.array([], dtype=float)
    try:
        arr = np.asarray(arr, dtype=float).squeeze()
        return np.atleast_1d(arr).astype(float).ravel()
    except Exception:
        return np.array([], dtype=float)


def safe_warning_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    if isinstance(value, tuple):
        return [str(v) for v in value if str(v)]
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.ravel().tolist() if str(v)]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def nominal_temperature_for_battery(battery_index: int) -> float:
    return 40.0 if int(battery_index) >= 21 else 25.0


def finite_nanmin(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.min(finite)) if finite.size else math.nan


def finite_nanmax(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else math.nan


def finite_nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else math.nan


def robust_initial_capacity(capacities: Iterable[float], n_initial: int = 3) -> float:
    values = np.asarray(list(capacities), dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return math.nan
    return float(np.median(values[: min(n_initial, values.size)]))


def cumulative_trapezoid_integral(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(time_s) == 0:
        return np.array([], dtype=float)
    if len(time_s) == 1:
        return np.array([0.0], dtype=float)
    dt = np.diff(time_s)
    mid_values = 0.5 * (values[1:] + values[:-1])
    increments = mid_values * dt
    increments = np.where(np.isfinite(increments), increments, 0.0)
    return np.concatenate([[0.0], np.cumsum(increments)])


def calculate_capacity_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    capacity = cumulative_trapezoid_integral(time_s, np.abs(current_a)) / 3600.0
    capacity = np.maximum.accumulate(np.where(np.isfinite(capacity), capacity, 0.0))
    if capacity.size:
        capacity[0] = 0.0
    return capacity


def calculate_energy_wh(time_s: np.ndarray, voltage_v: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    abs_power = np.abs(voltage_v * current_a)
    energy = cumulative_trapezoid_integral(time_s, abs_power) / 3600.0
    energy = np.maximum.accumulate(np.where(np.isfinite(energy), energy, 0.0))
    if energy.size:
        energy[0] = 0.0
    return energy


def enforce_current_sign(current_a: np.ndarray, phase: str) -> np.ndarray:
    if phase == "charge":
        return np.abs(current_a.astype(float))
    if phase == "discharge":
        return -np.abs(current_a.astype(float))
    raise ValueError(f"Unsupported phase: {phase}")


def align_core_vectors(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int, int, int] | None:
    """Align vectors, remove bad time/voltage/current samples, sort, and reset time."""
    warnings = safe_warning_list(row.get("cleaning_warnings", []))
    time = safe_float_array(row.get("time_s"))
    voltage = safe_float_array(row.get("voltage_v"))
    current = safe_float_array(row.get("current_a"))
    temperature = safe_float_array(row.get("temperature_c"))

    n_original = max(len(time), len(voltage), len(current), len(temperature))
    n = min(len(time), len(voltage), len(current))
    if n == 0:
        return None

    time = time[:n]
    voltage = voltage[:n]
    current = current[:n]

    if len(temperature) >= n:
        temperature = temperature[:n]
    else:
        warnings.append("temperature_rebuilt_to_match_core_vectors")
        temperature = np.full(n, np.nan, dtype=float)

    mask = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current)
    removed_nonfinite = int((~mask).sum())
    if removed_nonfinite:
        warnings.append(f"removed_nonfinite_core_samples={removed_nonfinite}")

    time = time[mask]
    voltage = voltage[mask]
    current = current[mask]
    temperature = temperature[mask]

    if len(time) == 0:
        return None

    order = np.argsort(time)
    time = time[order]
    voltage = voltage[order]
    current = current[order]
    temperature = temperature[order]

    _, unique_idx = np.unique(time, return_index=True)
    if len(unique_idx) != len(time):
        warnings.append(f"removed_duplicate_time_samples={len(time) - len(unique_idx)}")
    unique_idx = np.sort(unique_idx)
    time = time[unique_idx]
    voltage = voltage[unique_idx]
    current = current[unique_idx]
    temperature = temperature[unique_idx]

    if len(time) == 0:
        return None

    time = time - time[0]
    n_after = len(time)
    n_removed = int(max(n_original - n_after, 0))
    return time, voltage, current, temperature, warnings, n_removed, 0, max(n_after - 1, 0)


def clean_temperature(
    temperature_c: np.ndarray,
    n_points: int,
    battery_index: int,
    warnings: list[str],
) -> tuple[np.ndarray, str, bool, list[str]]:
    temp = np.asarray(temperature_c, dtype=float)
    if len(temp) != n_points:
        temp = np.resize(temp, n_points) if len(temp) else np.full(n_points, np.nan)
        warnings.append("temperature_length_adjusted")

    valid = np.isfinite(temp) & (temp >= TEMP_MIN_VALID_C) & (temp <= TEMP_MAX_VALID_C)

    if valid.any():
        replacement = float(np.median(temp[valid]))
        n_invalid = int((~valid).sum())
        if n_invalid:
            temp = temp.copy()
            temp[~valid] = replacement
            warnings.append(f"temperature_invalid_values_replaced={n_invalid}")
        return temp, "measured", True, warnings

    nominal = nominal_temperature_for_battery(battery_index)
    warnings.append(f"temperature_replaced_by_nominal_constant={nominal}")
    return np.full(n_points, nominal, dtype=float), "nominal_constant", False, warnings


def trim_nonphysical_edges(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    temperature_c: np.ndarray,
    phase: str,
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int, list[str]]:
    """
    Remove obvious post-cutoff rebounds or artificial tails.

    For discharge, points after the minimum voltage are removed when the tail
    rebounds more than 20 mV. For charge, points after the maximum voltage are
    removed only when the final tail drops more than 50 mV, preserving normal CV
    plateaus.
    """
    n_original = len(time_s)
    start_idx = 0
    end_idx = n_original - 1

    if n_original < MIN_VALID_POINTS:
        return time_s, voltage_v, current_a, temperature_c, 0, start_idx, end_idx, warnings

    if phase == "discharge":
        min_idx = int(np.nanargmin(voltage_v))
        if min_idx < n_original - 2:
            tail_rebound = float(np.nanmax(voltage_v[min_idx:]) - voltage_v[min_idx])
            if tail_rebound > 0.020:
                end_idx = min_idx
                warnings.append(f"trimmed_discharge_post_cutoff_rebound={tail_rebound:.4f}V")
    elif phase == "charge":
        max_idx = int(np.nanargmax(voltage_v))
        if max_idx < n_original - 2:
            tail_drop = float(voltage_v[max_idx] - np.nanmin(voltage_v[max_idx:]))
            if tail_drop > 0.050:
                end_idx = max_idx
                warnings.append(f"trimmed_charge_post_cutoff_drop={tail_drop:.4f}V")

    if end_idx < start_idx:
        end_idx = n_original - 1

    sl = slice(start_idx, end_idx + 1)
    removed = int(n_original - (end_idx - start_idx + 1))

    time = time_s[sl]
    voltage = voltage_v[sl]
    current = current_a[sl]
    temperature = temperature_c[sl]

    if len(time):
        time = time - time[0]

    return time, voltage, current, temperature, removed, start_idx, end_idx, warnings


def can_interpolate_qv_curve(voltage_v: np.ndarray, capacity_ah: np.ndarray) -> bool:
    voltage = np.asarray(voltage_v, dtype=float)
    capacity = np.asarray(capacity_ah, dtype=float)
    finite = np.isfinite(voltage) & np.isfinite(capacity)
    if finite.sum() < MIN_VALID_POINTS:
        return False
    voltage = voltage[finite]
    capacity = capacity[finite]
    if float(np.nanmax(voltage) - np.nanmin(voltage)) < MIN_QV_VOLTAGE_RANGE:
        return False
    if float(np.nanmax(capacity) - np.nanmin(capacity)) <= 0:
        return False
    if np.unique(np.round(voltage, 4)).size < 5:
        return False
    return True


def infer_battery_eol(
    discharge_df: pd.DataFrame,
    eol_ratio: float,
    consecutive_cycles: int,
    eol_policy: str,
) -> dict[str, dict[str, Any]]:
    """Infer battery-level EOL from clean reference discharge capacity."""
    eol_by_battery: dict[str, dict[str, Any]] = {}

    if discharge_df.empty:
        return eol_by_battery

    for battery_id, group in discharge_df.groupby("battery_id"):
        g = group.sort_values("cycle_number").copy()
        capacities = pd.to_numeric(g["phase_capacity_ah"], errors="coerce").to_numpy(dtype=float)
        cycles = pd.to_numeric(g["cycle_number"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(capacities) & (capacities > 0) & np.isfinite(cycles)

        if valid.sum() == 0:
            eol_by_battery[battery_id] = {
                "cycle_life": math.nan,
                "initial_capacity_ah_eol_source": math.nan,
                "eol_capacity_ah": math.nan,
                "eol_status": "no_valid_discharge_capacity",
                "target_is_censored": True,
            }
            continue

        cap = capacities[valid]
        cyc = cycles[valid]
        initial = robust_initial_capacity(cap)
        threshold = eol_ratio * initial if np.isfinite(initial) else math.nan

        cycle_life = math.nan
        eol_status = "invalid_initial_capacity"
        target_is_censored = True

        if np.isfinite(initial) and initial > 0:
            below = cap <= threshold
            for idx in range(0, len(below) - consecutive_cycles + 1):
                if bool(np.all(below[idx : idx + consecutive_cycles])):
                    cycle_life = float(cyc[idx])
                    eol_status = "eol_reached"
                    target_is_censored = False
                    break

            if not np.isfinite(cycle_life):
                if eol_policy == "last_observed":
                    cycle_life = float(np.nanmax(cyc))
                    eol_status = "eol_not_reached_last_observed_used"
                    target_is_censored = False
                else:
                    cycle_life = float(np.nanmax(cyc))
                    eol_status = "eol_not_reached_censored"
                    target_is_censored = True

        eol_by_battery[battery_id] = {
            "cycle_life": cycle_life,
            "initial_capacity_ah_eol_source": initial,
            "eol_capacity_ah": threshold,
            "eol_status": eol_status,
            "target_is_censored": target_is_censored,
        }

    return eol_by_battery


def clean_one_row(row: pd.Series) -> dict[str, Any]:
    base = {col: row[col] if col in row.index else np.nan for col in STANDARD_CYCLE_COLUMNS}
    phase = str(base.get("phase", "")).strip().lower()
    battery_index = int(base.get("battery_index", -1)) if pd.notna(base.get("battery_index", np.nan)) else -1
    warnings = safe_warning_list(base.get("cleaning_warnings", []))
    invalid_reasons: list[str] = []

    if phase not in {"charge", "discharge"}:
        phase = "unknown"
        invalid_reasons.append("invalid_phase")

    aligned = align_core_vectors(row)
    if aligned is None:
        time_s = voltage_v = current_a = temperature_c = np.array([], dtype=float)
        warnings.append("missing_or_unalignable_core_vectors")
        invalid_reasons.append("missing_core_vectors")
        n_removed_points = 0
        trim_start_index = 0
        trim_end_index = 0
    else:
        time_s, voltage_v, current_a, temperature_c, warnings, n_removed_points, trim_start_index, trim_end_index = aligned
        if phase in {"charge", "discharge"}:
            current_a = enforce_current_sign(current_a, phase)

        temperature_c, temperature_source, has_temperature, warnings = clean_temperature(
            temperature_c=temperature_c,
            n_points=len(time_s),
            battery_index=battery_index,
            warnings=warnings,
        )

        time_s, voltage_v, current_a, temperature_c, removed_trim, trim_start_index, trim_end_index, warnings = trim_nonphysical_edges(
            time_s=time_s,
            voltage_v=voltage_v,
            current_a=current_a,
            temperature_c=temperature_c,
            phase=phase,
            warnings=warnings,
        )
        n_removed_points += removed_trim
    
    if aligned is None:
        temperature_source = "unknown_nan"
        has_temperature = False

    n_points = int(len(time_s))
    capacity_ah = calculate_capacity_ah(time_s, current_a) if n_points else np.array([], dtype=float)
    energy_wh = calculate_energy_wh(time_s, voltage_v, current_a) if n_points else np.array([], dtype=float)
    power_w = voltage_v * current_a if n_points else np.array([], dtype=float)

    duration_s = float(time_s[-1] - time_s[0]) if n_points > 1 else math.nan
    sample_period = float(np.nanmedian(np.diff(time_s))) if n_points > 1 else math.nan
    phase_capacity = float(capacity_ah[-1]) if capacity_ah.size else math.nan
    phase_energy = float(energy_wh[-1]) if energy_wh.size else math.nan

    has_voltage = bool(n_points and np.isfinite(voltage_v).any())
    has_current = bool(n_points and np.isfinite(current_a).any())
    has_capacity = bool(np.isfinite(phase_capacity) and phase_capacity > 0)
    has_energy = bool(np.isfinite(phase_energy) and phase_energy > 0)

    if n_points < MIN_VALID_POINTS:
        invalid_reasons.append("too_few_points")
    if not has_voltage:
        invalid_reasons.append("missing_voltage")
    if not has_current and not has_capacity:
        invalid_reasons.append("missing_current_and_capacity")
    if n_points > 1 and not bool(np.all(np.diff(time_s) > 0)):
        invalid_reasons.append("time_not_strictly_increasing")
    if not np.isfinite(duration_s) or duration_s <= 0:
        invalid_reasons.append("invalid_duration")
    if not has_capacity:
        invalid_reasons.append("missing_or_zero_capacity")
    if has_voltage and np.isfinite(finite_nanmax(voltage_v)) and np.isfinite(finite_nanmin(voltage_v)):
        if finite_nanmax(voltage_v) <= finite_nanmin(voltage_v):
            invalid_reasons.append("voltage_no_range")
        if finite_nanmin(voltage_v) < VOLTAGE_MIN_PHYSICAL_V or finite_nanmax(voltage_v) > VOLTAGE_MAX_PHYSICAL_V:
            invalid_reasons.append("voltage_outside_physical_range")
    if np.isfinite(phase_capacity) and phase_capacity > MAX_REASONABLE_PHASE_CAPACITY_AH:
        invalid_reasons.append("phase_capacity_unreasonably_high")

    is_valid_cycle = len(invalid_reasons) == 0

    base.update({
        "phase": phase,
        "time_s": time_s.astype(float).tolist(),
        "voltage_v": voltage_v.astype(float).tolist(),
        "current_a": current_a.astype(float).tolist(),
        "temperature_c": temperature_c.astype(float).tolist(),
        "capacity_ah": capacity_ah.astype(float).tolist(),
        "energy_wh": energy_wh.astype(float).tolist(),
        "power_w": power_w.astype(float).tolist(),
        "n_points": n_points,
        "duration_s": duration_s,
        "sample_period_median_s": sample_period,
        "voltage_min_v": finite_nanmin(voltage_v),
        "voltage_max_v": finite_nanmax(voltage_v),
        "voltage_start_v": float(voltage_v[0]) if n_points else math.nan,
        "voltage_end_v": float(voltage_v[-1]) if n_points else math.nan,
        "current_min_a": finite_nanmin(current_a),
        "current_max_a": finite_nanmax(current_a),
        "current_mean_a": finite_nanmean(current_a),
        "temperature_min_c": finite_nanmin(temperature_c),
        "temperature_max_c": finite_nanmax(temperature_c),
        "temperature_mean_c": finite_nanmean(temperature_c),
        "phase_capacity_ah": phase_capacity,
        "phase_energy_wh": phase_energy,
        "voltage_source": "measured" if has_voltage else "missing",
        "current_source": "measured" if has_current else "missing",
        "temperature_source": temperature_source,
        "capacity_source": "integrated_current" if has_current else "missing",
        "energy_source": "integrated_power" if has_current and has_voltage else "missing",
        "chemistry": "LCO" if pd.isna(base.get("chemistry", np.nan)) or not str(base.get("chemistry", "")).strip() else base.get("chemistry"),
        "form_factor": "18650" if pd.isna(base.get("form_factor", np.nan)) or not str(base.get("form_factor", "")).strip() else base.get("form_factor"),
        "manufacturer": "unknown" if pd.isna(base.get("manufacturer", np.nan)) or not str(base.get("manufacturer", "")).strip() else base.get("manufacturer"),
        "nominal_voltage_v": NOMINAL_VOLTAGE_V,
        "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "ambient_temperature_c": nominal_temperature_for_battery(battery_index),
        "is_valid_cycle": bool(is_valid_cycle),
        "is_model_usable": False,
        "exclude_reason": ";".join(dict.fromkeys(invalid_reasons)) if invalid_reasons else "",
        "cleaning_warnings": list(dict.fromkeys(warnings)),
        "n_removed_points": int(n_removed_points),
        "trim_start_index": int(trim_start_index),
        "trim_end_index": int(trim_end_index),
        "has_voltage": has_voltage,
        "has_current": has_current,
        "has_temperature": bool(has_temperature),
        "has_capacity": has_capacity,
        "has_energy": has_energy,
        "cleaning_version": CLEANING_VERSION,
        "created_at": utc_now_iso(),
    })
    return base


def empty_standard_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)


def reorder_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in STANDARD_CYCLE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[STANDARD_CYCLE_COLUMNS].copy()


def assign_targets_and_flags(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    *,
    eol_ratio: float,
    eol_consecutive_cycles: int,
    eol_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign battery-level cycle_life and model-usability flags to both branches."""
    eol_info = infer_battery_eol(discharge_df, eol_ratio, eol_consecutive_cycles, eol_policy)
    eol_rows = []

    phase_initials: dict[tuple[str, str], float] = {}
    for phase, df in [("charge", charge_df), ("discharge", discharge_df)]:
        if df.empty:
            continue
        for battery_id, group in df.groupby("battery_id"):
            valid_caps = group.sort_values("cycle_number")["phase_capacity_ah"].to_numpy(dtype=float)
            phase_initials[(battery_id, phase)] = robust_initial_capacity(valid_caps)

    output_frames = []
    for phase, df in [("charge", charge_df), ("discharge", discharge_df)]:
        if df.empty:
            output_frames.append(df)
            continue

        df = df.copy()
        for idx, row in df.iterrows():
            battery_id = str(row["battery_id"])
            info = eol_info.get(battery_id, {})
            cycle_life = float(info.get("cycle_life", math.nan))
            eol_capacity_ah = float(info.get("eol_capacity_ah", math.nan))
            eol_status = str(info.get("eol_status", "missing_eol_info"))
            target_is_censored = bool(info.get("target_is_censored", True))
            initial_capacity = phase_initials.get((battery_id, phase), math.nan)

            cycle_number = float(row["cycle_number"]) if pd.notna(row["cycle_number"]) else math.nan
            remaining = cycle_life - cycle_number if np.isfinite(cycle_life) and np.isfinite(cycle_number) else math.nan
            remaining_norm = remaining / cycle_life if np.isfinite(remaining) and np.isfinite(cycle_life) and cycle_life > 0 else math.nan
            phase_capacity = float(row["phase_capacity_ah"]) if pd.notna(row["phase_capacity_ah"]) else math.nan
            soh = phase_capacity / initial_capacity if np.isfinite(phase_capacity) and np.isfinite(initial_capacity) and initial_capacity > 0 else math.nan

            reasons = []
            if isinstance(row.get("exclude_reason", ""), str) and row.get("exclude_reason", ""):
                reasons.extend(str(row["exclude_reason"]).split(";"))

            qv_ok = can_interpolate_qv_curve(
                safe_float_array(row["voltage_v"]),
                safe_float_array(row["capacity_ah"]),
            )

            if target_is_censored:
                reasons.append(eol_status)
            if not np.isfinite(cycle_life):
                reasons.append("missing_cycle_life")
            if not np.isfinite(remaining) or remaining <= 0:
                reasons.append("non_positive_remaining_cycles")
            if not qv_ok:
                reasons.append("insufficient_voltage_capacity_range_for_qv")

            reasons = [reason for reason in dict.fromkeys(reasons) if reason]
            is_valid = bool(row["is_valid_cycle"])
            is_model_usable = bool(is_valid and len(reasons) == 0)

            warnings = safe_warning_list(row.get("cleaning_warnings", []))
            if eol_status and eol_status != "eol_reached":
                warnings.append(eol_status)

            df.at[idx, "cycle_life"] = cycle_life
            df.at[idx, "remaining_cycles"] = remaining
            df.at[idx, "remaining_cycles_norm"] = remaining_norm
            df.at[idx, "eol_criterion"] = f"{int(round(eol_ratio * 100))}pct_initial_reference_discharge_capacity"
            df.at[idx, "eol_capacity_ah"] = eol_capacity_ah
            df.at[idx, "initial_capacity_ah"] = initial_capacity
            df.at[idx, "soh"] = soh
            df.at[idx, "is_model_usable"] = is_model_usable
            df.at[idx, "exclude_reason"] = ";".join(reasons) if reasons else ""
            df.at[idx, "cleaning_warnings"] = list(dict.fromkeys(warnings))

        # Reassign sequential clean cycle index within battery and phase.
        df = df.sort_values(["battery_index", "battery_id", "cycle_number", "source_cycle_id"]).reset_index(drop=True)
        df["cycle_index"] = df.groupby("battery_id").cumcount() + 1
        output_frames.append(df)

    for battery_id, info in eol_info.items():
        eol_rows.append({"battery_id": battery_id, **info})

    return output_frames[0], output_frames[1], pd.DataFrame(eol_rows)


def clean_phase_file(input_file: Path, phase: str) -> pd.DataFrame:
    if not input_file.exists():
        return empty_standard_frame()

    df_raw = pd.read_parquet(input_file)
    if df_raw.empty:
        return empty_standard_frame()

    if "phase" in df_raw.columns:
        df_raw = df_raw[df_raw["phase"].astype(str).str.lower() == phase].copy()

    rows = []
    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc=f"Cleaning {phase}", unit="cycle"):
        rows.append(clean_one_row(row))

    return reorder_standard_columns(pd.DataFrame(rows)) if rows else empty_standard_frame()


def write_rejected_csv(df: pd.DataFrame, output_file: Path) -> None:
    if df.empty:
        pd.DataFrame(columns=[
            "dataset", "battery_id", "cycle_index", "cycle_number", "phase",
            "source_file", "source_cycle_id", "is_valid_cycle", "is_model_usable",
            "exclude_reason", "cleaning_warnings", "n_points", "phase_capacity_ah",
            "voltage_min_v", "voltage_max_v", "remaining_cycles",
        ]).to_csv(output_file, index=False, encoding="utf-8-sig")
        return

    rejected = df[(~df["is_valid_cycle"].astype(bool)) | (~df["is_model_usable"].astype(bool))].copy()
    keep_cols = [
        "dataset", "battery_id", "cycle_index", "cycle_number", "phase",
        "source_file", "source_cycle_id", "is_valid_cycle", "is_model_usable",
        "exclude_reason", "cleaning_warnings", "n_points", "phase_capacity_ah",
        "voltage_min_v", "voltage_max_v", "remaining_cycles",
    ]
    rejected[keep_cols].to_csv(output_file, index=False, encoding="utf-8-sig")


def summarize_cleaning(charge_df: pd.DataFrame, discharge_df: pd.DataFrame, eol_df: pd.DataFrame) -> dict[str, Any]:
    def phase_summary(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {"rows": 0, "batteries": 0, "valid": 0, "model_usable": 0, "rejected": 0}
        return {
            "rows": int(len(df)),
            "batteries": int(df["battery_id"].nunique()),
            "valid": int(df["is_valid_cycle"].sum()),
            "model_usable": int(df["is_model_usable"].sum()),
            "rejected": int(((~df["is_valid_cycle"].astype(bool)) | (~df["is_model_usable"].astype(bool))).sum()),
            "exclude_reasons": (
                df.loc[df["exclude_reason"].astype(str) != "", "exclude_reason"]
                .str.get_dummies(sep=";")
                .sum()
                .sort_values(ascending=False)
                .astype(int)
                .to_dict()
            ),
        }

    return {
        "dataset": DATASET_CODE,
        "cleaning_version": CLEANING_VERSION,
        "created_at": utc_now_iso(),
        "charge": phase_summary(charge_df),
        "discharge": phase_summary(discharge_df),
        "eol_by_battery": eol_df.to_dict(orient="records") if not eol_df.empty else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean NASA Random Walk extracted cycles.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Project root directory containing data/ and src/.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Input directory. Defaults to <base-dir>/data/interim/nrw/extracted.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <base-dir>/data/interim/nrw/clean.")
    parser.add_argument("--reports-dir", type=Path, default=None, help="Report directory. Defaults to <base-dir>/data/reports/nrw.")
    parser.add_argument("--eol-ratio", type=float, default=DEFAULT_EOL_RATIO, help="EOL threshold ratio applied to initial reference discharge capacity. Default: 0.70.")
    parser.add_argument("--eol-consecutive-cycles", type=int, default=DEFAULT_EOL_CONSECUTIVE_CYCLES, help="Consecutive cycles below threshold required to declare EOL. Default: 2.")
    parser.add_argument("--eol-policy", choices=["strict", "last_observed"], default="strict", help="strict marks batteries that do not reach EOL as censored/non-usable; last_observed uses the last observed cycle as cycle_life.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    input_dir = args.input_dir or (base_dir / "data" / "interim" / DATASET_FOLDER / "extracted")
    output_dir = args.output_dir or (base_dir / "data" / "interim" / DATASET_FOLDER / "clean")
    reports_dir = args.reports_dir or (base_dir / "data" / "reports" / DATASET_FOLDER)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    charge_in = input_dir / f"{DATASET_FOLDER}_charge_extracted_cycles.parquet"
    discharge_in = input_dir / f"{DATASET_FOLDER}_discharge_extracted_cycles.parquet"

    charge_out = output_dir / f"{DATASET_FOLDER}_charge_clean_cycles.parquet"
    discharge_out = output_dir / f"{DATASET_FOLDER}_discharge_clean_cycles.parquet"
    charge_rejected = output_dir / f"{DATASET_FOLDER}_charge_rejected_cycles.csv"
    discharge_rejected = output_dir / f"{DATASET_FOLDER}_discharge_rejected_cycles.csv"
    report_out = reports_dir / "cleaning_report.json"
    eol_out = reports_dir / "cleaning_eol_summary.csv"

    print(f"Cleaning charge input:    {charge_in}")
    print(f"Cleaning discharge input: {discharge_in}")

    charge_df = clean_phase_file(charge_in, "charge")
    discharge_df = clean_phase_file(discharge_in, "discharge")

    charge_df, discharge_df, eol_df = assign_targets_and_flags(
        charge_df,
        discharge_df,
        eol_ratio=args.eol_ratio,
        eol_consecutive_cycles=args.eol_consecutive_cycles,
        eol_policy=args.eol_policy,
    )

    charge_df = reorder_standard_columns(charge_df)
    discharge_df = reorder_standard_columns(discharge_df)

    charge_df.to_parquet(charge_out, index=False)
    discharge_df.to_parquet(discharge_out, index=False)
    write_rejected_csv(charge_df, charge_rejected)
    write_rejected_csv(discharge_df, discharge_rejected)

    if not eol_df.empty:
        eol_df.to_csv(eol_out, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(eol_out, index=False, encoding="utf-8-sig")

    report_payload = summarize_cleaning(charge_df, discharge_df, eol_df)
    report_payload.update({
        "eol_ratio": args.eol_ratio,
        "eol_consecutive_cycles": args.eol_consecutive_cycles,
        "eol_policy": args.eol_policy,
        "charge_output": str(charge_out),
        "discharge_output": str(discharge_out),
        "charge_rejected": str(charge_rejected),
        "discharge_rejected": str(discharge_rejected),
    })
    with report_out.open("w", encoding="utf-8") as fp:
        json.dump(report_payload, fp, indent=2, ensure_ascii=False)

    print("\nCleaning completed.")
    print(f"Charge clean cycles:    {len(charge_df):6d} -> {charge_out}")
    print(f"Discharge clean cycles: {len(discharge_df):6d} -> {discharge_out}")
    print(f"Charge rejected CSV:    {charge_rejected}")
    print(f"Discharge rejected CSV: {discharge_rejected}")
    print(f"Report:                 {report_out}")


if __name__ == "__main__":
    main()
