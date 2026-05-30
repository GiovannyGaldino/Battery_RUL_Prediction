r"""
MATR/Severson dataset adapter - cleaning step (version 2.0).

This script reads phase-specific extracted MATR cycles and writes clean cycle
files that follow the official project schema for charge and discharge branches.

Inputs:
    data/interim/matr/extracted/matr_charge_extracted_cycles.parquet
    data/interim/matr/extracted/matr_discharge_extracted_cycles.parquet

Outputs:
    data/interim/matr/clean/matr_charge_clean_cycles.parquet
    data/interim/matr/clean/matr_discharge_clean_cycles.parquet
    data/interim/matr/clean/matr_charge_rejected_cycles.csv
    data/interim/matr/clean/matr_discharge_rejected_cycles.csv
    data/interim/matr/clean/matr_cleaning_report.json
    data/interim/matr/clean/matr_charge_cycle_anomalies.csv
    data/interim/matr/clean/matr_discharge_cycle_anomalies.csv
    data/interim/matr/clean/matr_cycle_anomalies_all.csv
    data/interim/matr/clean/matr_cleaning_anomaly_report.json

Main conventions:
    - current_a > 0 for charge and current_a < 0 for discharge.
    - capacity_ah is non-negative and accumulated within each phase.
    - power_w = voltage_v * current_a.
    - energy_wh is accumulated from abs(power_w) to keep phase energy non-negative.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable | None = None, **kwargs):
        return iterable if iterable is not None else (lambda x: x)


DATASET = "MATR"
CLEANING_VERSION = "matr_adapter_v2_1_clean_memory_safe_contiguous_split_2026-05-26"

DEFAULT_BASE_DIR = Path.cwd()

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_base_dir(base_dir_arg: str | None) -> Path:
    if base_dir_arg:
        return Path(base_dir_arg).expanduser().resolve()
    if DEFAULT_BASE_DIR.exists():
        return DEFAULT_BASE_DIR
    return Path.cwd().resolve()


def safe_vector(x: Any) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float).squeeze()
        return np.atleast_1d(arr)
    except Exception:
        return np.array([], dtype=float)


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except Exception:
        return np.nan


def finite_min(x: np.ndarray) -> float:
    values = safe_vector(x)
    values = values[np.isfinite(values)]
    return float(np.min(values)) if values.size else np.nan


def finite_max(x: np.ndarray) -> float:
    values = safe_vector(x)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else np.nan


def finite_mean(x: np.ndarray) -> float:
    values = safe_vector(x)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def all_vector_lengths_equal(row: dict[str, Any]) -> bool:
    lengths = [len(row[col]) for col in ["time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah", "energy_wh", "power_w"]]
    return len(set(lengths)) == 1 and lengths[0] > 0


def time_is_strictly_increasing(time_s: np.ndarray) -> bool:
    time_s = safe_vector(time_s)
    if time_s.size < 2 or not np.all(np.isfinite(time_s)):
        return False
    return bool(np.all(np.diff(time_s) > 0))


def integrate_capacity_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    time_s = safe_vector(time_s)
    current_a = safe_vector(current_a)
    n = min(len(time_s), len(current_a))
    if n < 2:
        return np.array([], dtype=float)
    time_s = time_s[:n]
    current_a = current_a[:n]
    dt = np.diff(time_s, prepend=time_s[0])
    dt[~np.isfinite(dt)] = 0.0
    dt[dt < 0] = 0.0
    capacity = np.cumsum(np.abs(current_a) * dt) / 3600.0
    capacity = capacity - capacity[0]
    return capacity


def integrate_energy_wh(time_s: np.ndarray, power_w: np.ndarray) -> np.ndarray:
    time_s = safe_vector(time_s)
    power_w = safe_vector(power_w)
    n = min(len(time_s), len(power_w))
    if n < 2:
        return np.full(n, np.nan, dtype=float)
    time_s = time_s[:n]
    power_w = power_w[:n]
    dt = np.diff(time_s, prepend=time_s[0])
    dt[~np.isfinite(dt)] = 0.0
    dt[dt < 0] = 0.0
    energy = np.cumsum(np.abs(power_w) * dt) / 3600.0
    energy = energy - energy[0]
    return energy


def normalize_capacity(capacity_ah: np.ndarray, force_monotonic: bool) -> np.ndarray:
    capacity_ah = safe_vector(capacity_ah).astype(float)
    if capacity_ah.size == 0:
        return capacity_ah
    finite = np.isfinite(capacity_ah)
    if not finite.any():
        return capacity_ah
    capacity_ah = capacity_ah - np.nanmin(capacity_ah[finite])
    capacity_ah[capacity_ah < 0] = 0.0
    if force_monotonic:
        tmp = capacity_ah.copy()
        tmp[~np.isfinite(tmp)] = np.nan
        if np.isfinite(tmp).any():
            # interpolate NaNs before accumulating, then restore finite result
            s = pd.Series(tmp).interpolate(limit_direction="both")
            capacity_ah = np.maximum.accumulate(s.to_numpy(dtype=float))
    if capacity_ah.size:
        capacity_ah = capacity_ah - capacity_ah[0]
    return capacity_ah


def has_usable_capacity(capacity_ah: np.ndarray) -> bool:
    values = safe_vector(capacity_ah)
    finite = values[np.isfinite(values)]
    return finite.size >= 2 and float(np.nanmax(finite) - np.nanmin(finite)) > 1e-9


def capacity_is_monotonic(capacity_ah: np.ndarray) -> bool:
    values = safe_vector(capacity_ah)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return False
    return bool(np.all(np.diff(values) >= -1e-10))


def can_interpolate_qv_curve(voltage_v: np.ndarray, capacity_ah: np.ndarray, min_voltage_span: float, min_unique_voltage: int) -> bool:
    voltage_v = safe_vector(voltage_v)
    capacity_ah = safe_vector(capacity_ah)
    n = min(len(voltage_v), len(capacity_ah))
    if n < 20:
        return False
    voltage_v = voltage_v[:n]
    capacity_ah = capacity_ah[:n]
    mask = np.isfinite(voltage_v) & np.isfinite(capacity_ah)
    if mask.sum() < 20:
        return False
    voltage_v = voltage_v[mask]
    if float(np.nanmax(voltage_v) - np.nanmin(voltage_v)) < min_voltage_span:
        return False
    return len(np.unique(np.round(voltage_v, 5))) >= min_unique_voltage


def align_and_clean_basic(row: pd.Series) -> tuple[dict[str, np.ndarray] | None, list[str], str]:
    warnings: list[str] = []
    time_s = safe_vector(row.get("time_s", []))
    voltage_v = safe_vector(row.get("voltage_v", []))
    current_a = safe_vector(row.get("current_a", []))
    temperature_c = safe_vector(row.get("temperature_c", []))
    capacity_ah = safe_vector(row.get("capacity_ah", []))

    if time_s.size == 0 or voltage_v.size == 0:
        return None, warnings, "missing_time_or_voltage"

    # Current may be missing only if capacity exists, but MATR normally provides current.
    if current_a.size == 0 and capacity_ah.size == 0:
        return None, warnings, "missing_current_and_capacity"

    n = min(len(time_s), len(voltage_v))
    if current_a.size > 0:
        n = min(n, len(current_a))
    if capacity_ah.size > 0:
        n = min(n, len(capacity_ah))

    if n <= 0:
        return None, warnings, "empty_aligned_vectors"

    time_s = time_s[:n]
    voltage_v = voltage_v[:n]
    if current_a.size > 0:
        current_a = current_a[:n]
    else:
        current_a = np.full(n, np.nan, dtype=float)
    if capacity_ah.size > 0:
        capacity_ah = capacity_ah[:n]
    else:
        capacity_ah = np.full(n, np.nan, dtype=float)

    if temperature_c.size >= n:
        temperature_c = temperature_c[:n]
    else:
        temperature_c = np.full(n, np.nan, dtype=float)

    mask = np.isfinite(time_s) & np.isfinite(voltage_v)
    if np.isfinite(current_a).any():
        mask &= np.isfinite(current_a)
    if not mask.any():
        return None, warnings, "no_finite_required_samples"

    removed = int(len(time_s) - mask.sum())
    if removed > 0:
        warnings.append(f"removed_nonfinite_required_samples:{removed}")

    time_s = time_s[mask]
    voltage_v = voltage_v[mask]
    current_a = current_a[mask]
    temperature_c = temperature_c[mask]
    capacity_ah = capacity_ah[mask]

    order = np.argsort(time_s)
    time_s = time_s[order]
    voltage_v = voltage_v[order]
    current_a = current_a[order]
    temperature_c = temperature_c[order]
    capacity_ah = capacity_ah[order]

    _, unique_idx = np.unique(time_s, return_index=True)
    unique_idx = np.sort(unique_idx)
    if len(unique_idx) < len(time_s):
        warnings.append(f"removed_duplicated_time_samples:{len(time_s) - len(unique_idx)}")
    time_s = time_s[unique_idx]
    voltage_v = voltage_v[unique_idx]
    current_a = current_a[unique_idx]
    temperature_c = temperature_c[unique_idx]
    capacity_ah = capacity_ah[unique_idx]

    if len(time_s) < 2:
        return None, warnings, "insufficient_samples_after_basic_cleaning"

    time_s = time_s - time_s[0]

    return {
        "time_s": time_s,
        "voltage_v": voltage_v,
        "current_a": current_a,
        "temperature_c": temperature_c,
        "capacity_ah": capacity_ah,
    }, warnings, ""


def apply_voltage_window(
    vectors: dict[str, np.ndarray],
    *,
    phase: str,
    voltage_min: float,
    voltage_max: float,
    truncate_after_cutoff: bool,
) -> tuple[dict[str, np.ndarray], list[str], int, int, int]:
    warnings: list[str] = []
    voltage_v = vectors["voltage_v"]
    n_before = len(voltage_v)
    raw_indices = np.arange(n_before)

    keep = np.ones(n_before, dtype=bool)

    if truncate_after_cutoff:
        if phase == "discharge":
            low_positions = np.where(voltage_v < voltage_min)[0]
            if low_positions.size > 0:
                first_low = int(low_positions[0])
                keep[first_low:] = False
                warnings.append("truncated_after_first_low_voltage")
        elif phase == "charge":
            high_positions = np.where(voltage_v > voltage_max)[0]
            if high_positions.size > 0:
                first_high = int(high_positions[0])
                keep[first_high:] = False
                warnings.append("truncated_after_first_high_voltage")

    keep &= voltage_v >= voltage_min
    keep &= voltage_v <= voltage_max

    cleaned = {key: value[keep] for key, value in vectors.items()}
    kept_indices = raw_indices[keep]
    n_removed = int(n_before - len(kept_indices))
    trim_start = int(kept_indices[0]) if kept_indices.size else -1
    trim_end = int(kept_indices[-1]) if kept_indices.size else -1
    if n_removed > 0:
        warnings.append(f"voltage_window_removed_points:{n_removed}")
    return cleaned, warnings, n_removed, trim_start, trim_end


def apply_phase_current_convention(current_a: np.ndarray, phase: str) -> np.ndarray:
    current_a = safe_vector(current_a).astype(float)
    if not np.isfinite(current_a).any():
        return current_a
    if phase == "charge":
        return np.abs(current_a)
    if phase == "discharge":
        return -np.abs(current_a)
    raise ValueError(f"Unsupported phase: {phase}")


def build_rejection(row: pd.Series, phase: str, reason: str, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "source_file": row.get("source_file", ""),
        "source_path": row.get("source_path", ""),
        "battery_id": row.get("battery_id", ""),
        "battery_index": row.get("battery_index", np.nan),
        "cell_id_original": row.get("cell_id_original", ""),
        "cycle_number": row.get("cycle_number", np.nan),
        "source_cycle_id": row.get("source_cycle_id", ""),
        "cycle_pair_id": row.get("cycle_pair_id", ""),
        "phase": phase,
        "exclude_reason": reason,
        "cleaning_warnings": warnings or [],
    }


def clean_one_cycle(
    row: pd.Series,
    *,
    phase: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    warnings: list[str] = []
    vectors, basic_warnings, reason = align_and_clean_basic(row)
    warnings.extend(basic_warnings)
    if vectors is None:
        return None, build_rejection(row, phase, reason, warnings)

    original_n = len(vectors["time_s"])

    if not args.disable_voltage_window:
        if phase == "charge":
            vmin, vmax = args.charge_v_min, args.charge_v_max
        else:
            vmin, vmax = args.discharge_v_min, args.discharge_v_max
        vectors, window_warnings, n_removed_window, trim_start, trim_end = apply_voltage_window(
            vectors,
            phase=phase,
            voltage_min=vmin,
            voltage_max=vmax,
            truncate_after_cutoff=args.truncate_after_cutoff,
        )
        warnings.extend(window_warnings)
    else:
        n_removed_window = 0
        trim_start = 0
        trim_end = original_n - 1

    if len(vectors["time_s"]) < args.min_points_after_cleaning:
        return None, build_rejection(row, phase, "insufficient_points_after_cleaning", warnings)

    # Reset time origin after trimming and re-check monotonicity.
    vectors["time_s"] = vectors["time_s"] - vectors["time_s"][0]
    if not time_is_strictly_increasing(vectors["time_s"]):
        return None, build_rejection(row, phase, "invalid_or_non_increasing_time", warnings)

    current_source = "measured" if np.isfinite(vectors["current_a"]).any() else "missing"
    vectors["current_a"] = apply_phase_current_convention(vectors["current_a"], phase)

    capacity_source = str(row.get("capacity_source_raw", "measured"))
    if has_usable_capacity(vectors["capacity_ah"]):
        capacity_ah = normalize_capacity(vectors["capacity_ah"], force_monotonic=args.force_capacity_monotonic)
        if not capacity_is_monotonic(capacity_ah):
            capacity_ah = integrate_capacity_ah(vectors["time_s"], vectors["current_a"])
            capacity_source = "integrated_current"
            warnings.append("capacity_recomputed_because_non_monotonic")
    elif current_source == "measured":
        capacity_ah = integrate_capacity_ah(vectors["time_s"], vectors["current_a"])
        capacity_source = "integrated_current"
        warnings.append("capacity_reconstructed_from_current")
    else:
        return None, build_rejection(row, phase, "missing_current_and_capacity", warnings)

    if capacity_ah.size != len(vectors["time_s"]) or not has_usable_capacity(capacity_ah):
        return None, build_rejection(row, phase, "invalid_capacity_after_cleaning", warnings)

    vectors["capacity_ah"] = capacity_ah

    # Temperature policy.
    temp = vectors["temperature_c"]
    if np.isfinite(temp).any():
        temperature_source = "measured"
        has_temperature = True
        # Fill isolated NaNs in measured temperature to keep vector length stable.
        if not np.all(np.isfinite(temp)):
            temp = pd.Series(temp).interpolate(limit_direction="both").to_numpy(dtype=float)
            warnings.append("temperature_interpolated")
    elif np.isfinite(args.fill_missing_temperature_c):
        temp = np.full(len(vectors["time_s"]), float(args.fill_missing_temperature_c), dtype=float)
        temperature_source = "nominal_constant"
        has_temperature = False
        warnings.append("temperature_filled_with_nominal_constant")
    else:
        temp = np.full(len(vectors["time_s"]), np.nan, dtype=float)
        temperature_source = "unknown_nan"
        has_temperature = False
    vectors["temperature_c"] = temp

    power_w = vectors["voltage_v"] * vectors["current_a"]
    energy_wh = integrate_energy_wh(vectors["time_s"], power_w)
    energy_source = "integrated_power" if np.isfinite(energy_wh).any() else "missing"

    n_points = int(len(vectors["time_s"]))
    duration_s = float(vectors["time_s"][-1] - vectors["time_s"][0]) if n_points else np.nan
    dt = np.diff(vectors["time_s"])
    sample_period_median_s = float(np.nanmedian(dt)) if dt.size else np.nan

    phase_capacity_ah = float(capacity_ah[-1]) if capacity_ah.size else np.nan
    phase_energy_wh = float(energy_wh[-1]) if energy_wh.size and np.isfinite(energy_wh[-1]) else np.nan

    cycle_life = finite_or_nan(row.get("cycle_life", np.nan))
    cycle_number = finite_or_nan(row.get("cycle_number", np.nan))

    # Initial capacity, SOH, and target are filled later after grouping by battery and phase.
    clean_row: dict[str, Any] = {
        "dataset": DATASET,
        "source_file": str(row.get("source_file", "")),
        "source_path": str(row.get("source_path", "")),
        "battery_id": str(row.get("battery_id", "")),
        "battery_index": int(row.get("battery_index", -1)) if pd.notna(row.get("battery_index", np.nan)) else -1,
        "cell_id_original": str(row.get("cell_id_original", "")),
        "cycle_index": -1,
        "cycle_number": cycle_number,
        "source_cycle_id": str(row.get("source_cycle_id", "")),
        "cycle_pair_id": str(row.get("cycle_pair_id", "")),
        "phase": phase,
        "cycle_life": cycle_life,
        "remaining_cycles": np.nan,
        "remaining_cycles_norm": np.nan,
        "eol_criterion": "provided_by_dataset_80pct_initial_capacity",
        "eol_capacity_ah": np.nan,
        "initial_capacity_ah": np.nan,
        "nominal_capacity_ah": args.nominal_capacity_ah,
        "soh": np.nan,
        "time_s": vectors["time_s"].astype(float).tolist(),
        "voltage_v": vectors["voltage_v"].astype(float).tolist(),
        "current_a": vectors["current_a"].astype(float).tolist(),
        "temperature_c": vectors["temperature_c"].astype(float).tolist(),
        "capacity_ah": vectors["capacity_ah"].astype(float).tolist(),
        "energy_wh": energy_wh.astype(float).tolist(),
        "power_w": power_w.astype(float).tolist(),
        "n_points": n_points,
        "duration_s": duration_s,
        "sample_period_median_s": sample_period_median_s,
        "voltage_min_v": finite_min(vectors["voltage_v"]),
        "voltage_max_v": finite_max(vectors["voltage_v"]),
        "voltage_start_v": float(vectors["voltage_v"][0]),
        "voltage_end_v": float(vectors["voltage_v"][-1]),
        "current_min_a": finite_min(vectors["current_a"]),
        "current_max_a": finite_max(vectors["current_a"]),
        "current_mean_a": finite_mean(vectors["current_a"]),
        "temperature_min_c": finite_min(vectors["temperature_c"]),
        "temperature_max_c": finite_max(vectors["temperature_c"]),
        "temperature_mean_c": finite_mean(vectors["temperature_c"]),
        "phase_capacity_ah": phase_capacity_ah,
        "phase_energy_wh": phase_energy_wh,
        "voltage_source": "measured",
        "current_source": current_source,
        "temperature_source": temperature_source,
        "capacity_source": capacity_source,
        "energy_source": energy_source,
        "chemistry": "LFP",
        "form_factor": "cylindrical",
        "manufacturer": "A123 Systems",
        "nominal_voltage_v": args.nominal_voltage_v,
        "ambient_temperature_c": args.ambient_temperature_c,
        "charge_protocol": "Fast-charge policy from MATR/Severson batch metadata; CC-CV completion step",
        "discharge_protocol": "Standardized discharge protocol from MATR/Severson experiments",
        "c_rate_charge": np.nan,
        "c_rate_discharge": 4.0,
        "dod": np.nan,
        "policy_readable": str(row.get("policy_readable", "")),
        "barcode": str(row.get("barcode", "")),
        "is_valid_cycle": False,
        "is_model_usable": False,
        "exclude_reason": "",
        "cleaning_warnings": warnings,
        "n_removed_points": int(original_n - n_points + n_removed_window),
        "trim_start_index": int(trim_start),
        "trim_end_index": int(trim_end),
        "has_voltage": True,
        "has_current": current_source == "measured",
        "has_temperature": bool(has_temperature),
        "has_capacity": True,
        "has_energy": energy_source != "missing",
        "adapter_version": str(row.get("adapter_version", "")),
        "cleaning_version": CLEANING_VERSION,
        "created_at": utc_now_iso(),
    }

    is_valid_cycle = (
        n_points >= args.min_points_after_cleaning
        and duration_s > 0
        and time_is_strictly_increasing(np.asarray(clean_row["time_s"], dtype=float))
        and np.isfinite(clean_row["voltage_min_v"])
        and np.isfinite(clean_row["voltage_max_v"])
        and clean_row["voltage_max_v"] > clean_row["voltage_min_v"]
        and np.isfinite(phase_capacity_ah)
        and phase_capacity_ah > 0
        and all_vector_lengths_equal(clean_row)
    )

    clean_row["is_valid_cycle"] = bool(is_valid_cycle)
    if not is_valid_cycle:
        return None, build_rejection(row, phase, "failed_minimum_validity_criteria", warnings)

    return clean_row, None



def robust_initial_capacity(values: np.ndarray, args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    """Estimate the initial phase capacity using a robust early-cycle median.

    The previous version used the first valid capacity. In MATR, some early charge
    or discharge segments can be clipped by the voltage window or affected by
    policy-dependent behavior. This function therefore uses the first N positive
    capacities, removes IQR outliers, and returns the median of the retained set.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > float(args.initial_capacity_min_abs_ah))]

    info: dict[str, Any] = {
        "method": "median_first_n_iqr_filtered",
        "n_available_positive": int(values.size),
        "n_requested": int(args.initial_capacity_n_cycles),
        "n_used_before_iqr": 0,
        "n_used_after_iqr": 0,
        "iqr_low": np.nan,
        "iqr_high": np.nan,
        "fallback_used": False,
    }

    if values.size == 0:
        return np.nan, info

    early = values[: int(args.initial_capacity_n_cycles)]
    info["n_used_before_iqr"] = int(early.size)

    if early.size >= int(args.initial_capacity_min_cycles_for_iqr):
        q1 = float(np.nanquantile(early, 0.25))
        q3 = float(np.nanquantile(early, 0.75))
        iqr = q3 - q1
        low = q1 - float(args.initial_capacity_iqr_multiplier) * iqr
        high = q3 + float(args.initial_capacity_iqr_multiplier) * iqr
        retained = early[(early >= low) & (early <= high)]
        info["iqr_low"] = float(low)
        info["iqr_high"] = float(high)
        if retained.size >= max(1, int(args.initial_capacity_min_cycles_after_iqr)):
            early = retained
        else:
            info["fallback_used"] = True
    else:
        info["fallback_used"] = True

    info["n_used_after_iqr"] = int(early.size)
    return float(np.nanmedian(early)) if early.size else np.nan, info


def append_warning(existing: Any, warning: str) -> list[str]:
    """Append a warning to a list-like cleaning_warnings field."""
    if isinstance(existing, list):
        warnings = list(existing)
    elif isinstance(existing, str) and existing.strip():
        warnings = [existing]
    else:
        warnings = []
    if warning not in warnings:
        warnings.append(warning)
    return warnings


def add_cleaning_anomaly(
    rows: list[dict[str, Any]],
    row: pd.Series,
    *,
    anomaly_type: str,
    anomaly_group: str,
    severity: str,
    metric_name: str,
    metric_value: Any,
    threshold_low: Any = np.nan,
    threshold_high: Any = np.nan,
    affects_modeling: bool = False,
    description: str = "",
) -> None:
    """Store one anomaly record for article/report tables without changing the clean schema."""
    rows.append(
        {
            "dataset": DATASET,
            "phase": row.get("phase", ""),
            "battery_id": row.get("battery_id", ""),
            "battery_index": row.get("battery_index", np.nan),
            "cell_id_original": row.get("cell_id_original", ""),
            "cycle_index": row.get("cycle_index", np.nan),
            "cycle_number": row.get("cycle_number", np.nan),
            "source_cycle_id": row.get("source_cycle_id", ""),
            "cycle_pair_id": row.get("cycle_pair_id", ""),
            "anomaly_type": anomaly_type,
            "anomaly_group": anomaly_group,
            "severity": severity,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "threshold_low": threshold_low,
            "threshold_high": threshold_high,
            "affects_modeling": bool(affects_modeling),
            "is_valid_cycle": row.get("is_valid_cycle", np.nan),
            "is_model_usable": row.get("is_model_usable", np.nan),
            "exclude_reason": row.get("exclude_reason", ""),
            "description": description,
            "created_at": utc_now_iso(),
        }
    )


def build_cleaning_anomalies(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Build a detailed anomaly table after targets and SOH have been finalized.

    The returned table is intentionally stored outside the clean parquet files so
    that the clean schema remains exactly equal to STANDARD_CYCLE_COLUMNS.
    """
    anomaly_rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()

    iterator = tqdm(df.iterrows(), total=len(df), desc="Building cleaning anomaly table", unit="cycle")
    for _, row in iterator:
        phase_capacity = finite_or_nan(row.get("phase_capacity_ah", np.nan))
        initial_capacity = finite_or_nan(row.get("initial_capacity_ah", np.nan))
        soh = finite_or_nan(row.get("soh", np.nan))
        remaining_cycles = finite_or_nan(row.get("remaining_cycles", np.nan))
        cycle_life = finite_or_nan(row.get("cycle_life", np.nan))
        n_points = finite_or_nan(row.get("n_points", np.nan))
        duration_s = finite_or_nan(row.get("duration_s", np.nan))
        voltage_span = finite_or_nan(row.get("voltage_max_v", np.nan)) - finite_or_nan(row.get("voltage_min_v", np.nan))
        n_removed = finite_or_nan(row.get("n_removed_points", np.nan))

        if not np.isfinite(cycle_life):
            add_cleaning_anomaly(
                anomaly_rows, row,
                anomaly_type="missing_cycle_life",
                anomaly_group="target",
                severity="major",
                metric_name="cycle_life",
                metric_value=cycle_life,
                affects_modeling=True,
                description="Cycle life is unavailable; supervised RUL target cannot be computed.",
            )
        if np.isfinite(remaining_cycles) and remaining_cycles <= 0:
            add_cleaning_anomaly(
                anomaly_rows, row,
                anomaly_type="non_positive_remaining_cycles",
                anomaly_group="target",
                severity="major",
                metric_name="remaining_cycles",
                metric_value=remaining_cycles,
                threshold_low=0,
                affects_modeling=True,
                description="Cycle is at or beyond the EOL target and must not be used for positive-RUL training.",
            )
        if not np.isfinite(initial_capacity) or initial_capacity <= 0:
            add_cleaning_anomaly(
                anomaly_rows, row,
                anomaly_type="invalid_initial_capacity",
                anomaly_group="capacity",
                severity="critical",
                metric_name="initial_capacity_ah",
                metric_value=initial_capacity,
                affects_modeling=True,
                description="Robust initial capacity could not be estimated for this battery/phase.",
            )
        if np.isfinite(phase_capacity) and np.isfinite(initial_capacity) and initial_capacity > 0:
            low_capacity = float(args.phase_capacity_low_ratio_warning) * initial_capacity
            high_capacity = float(args.phase_capacity_high_ratio_warning) * initial_capacity
            if phase_capacity < low_capacity:
                add_cleaning_anomaly(
                    anomaly_rows, row,
                    anomaly_type="low_phase_capacity_relative_to_initial",
                    anomaly_group="capacity",
                    severity="minor",
                    metric_name="phase_capacity_ah",
                    metric_value=phase_capacity,
                    threshold_low=low_capacity,
                    threshold_high=high_capacity,
                    affects_modeling=False,
                    description="Phase capacity is unusually low relative to the robust initial capacity.",
                )
            if phase_capacity > high_capacity:
                add_cleaning_anomaly(
                    anomaly_rows, row,
                    anomaly_type="high_phase_capacity_relative_to_initial",
                    anomaly_group="capacity",
                    severity="minor",
                    metric_name="phase_capacity_ah",
                    metric_value=phase_capacity,
                    threshold_low=low_capacity,
                    threshold_high=high_capacity,
                    affects_modeling=False,
                    description="Phase capacity is unusually high relative to the robust initial capacity.",
                )
        if np.isfinite(soh):
            if soh <= float(args.min_soh_model_usable) or soh > float(args.max_soh_model_usable):
                add_cleaning_anomaly(
                    anomaly_rows, row,
                    anomaly_type="abnormal_soh_model_exclusion",
                    anomaly_group="soh",
                    severity="major",
                    metric_name="soh",
                    metric_value=soh,
                    threshold_low=args.min_soh_model_usable,
                    threshold_high=args.max_soh_model_usable,
                    affects_modeling=True,
                    description="SOH is outside the model-usable interval defined for this cleaning run.",
                )
            elif soh < float(args.min_soh_warning) or soh > float(args.max_soh_warning):
                add_cleaning_anomaly(
                    anomaly_rows, row,
                    anomaly_type="abnormal_soh_warning",
                    anomaly_group="soh",
                    severity="minor",
                    metric_name="soh",
                    metric_value=soh,
                    threshold_low=args.min_soh_warning,
                    threshold_high=args.max_soh_warning,
                    affects_modeling=False,
                    description="SOH is outside the warning interval but still inside the model-usable interval.",
                )
        if np.isfinite(voltage_span) and voltage_span < float(args.min_voltage_span_for_interpolation):
            add_cleaning_anomaly(
                anomaly_rows, row,
                anomaly_type="insufficient_qv_interpolation_range",
                anomaly_group="voltage",
                severity="major",
                metric_name="voltage_span_v",
                metric_value=voltage_span,
                threshold_low=args.min_voltage_span_for_interpolation,
                affects_modeling=True,
                description="Voltage range is insufficient for reliable Q(V) interpolation.",
            )
        if np.isfinite(n_points) and n_points < int(args.min_points_after_cleaning):
            add_cleaning_anomaly(
                anomaly_rows, row,
                anomaly_type="few_points_after_cleaning",
                anomaly_group="signal_structure",
                severity="major",
                metric_name="n_points",
                metric_value=n_points,
                threshold_low=args.min_points_after_cleaning,
                affects_modeling=True,
                description="Cycle has too few samples after cleaning.",
            )
        if np.isfinite(duration_s) and duration_s <= 0:
            add_cleaning_anomaly(
                anomaly_rows, row,
                anomaly_type="non_positive_duration",
                anomaly_group="signal_structure",
                severity="critical",
                metric_name="duration_s",
                metric_value=duration_s,
                threshold_low=0,
                affects_modeling=True,
                description="Cycle duration is not physically valid.",
            )
        if np.isfinite(n_removed) and n_removed > 0:
            add_cleaning_anomaly(
                anomaly_rows, row,
                anomaly_type="trimmed_samples",
                anomaly_group="cleaning",
                severity="info",
                metric_name="n_removed_points",
                metric_value=n_removed,
                threshold_low=0,
                affects_modeling=False,
                description="Samples were removed by the cleaning routine, usually due to voltage-window trimming or duplicate/non-finite samples.",
            )
        reason = str(row.get("exclude_reason", ""))
        if reason:
            add_cleaning_anomaly(
                anomaly_rows, row,
                anomaly_type=reason,
                anomaly_group="model_usability",
                severity="major",
                metric_name="exclude_reason",
                metric_value=reason,
                affects_modeling=True,
                description="Final model-usability exclusion reason assigned by the cleaning routine.",
            )

    return pd.DataFrame(anomaly_rows)


def anomaly_summary(anomaly_df: pd.DataFrame) -> dict[str, Any]:
    if anomaly_df.empty:
        return {"total_anomaly_records": 0, "by_type": {}, "by_group": {}, "by_phase": {}}
    return {
        "total_anomaly_records": int(len(anomaly_df)),
        "affected_cycles": int(
            anomaly_df[["phase", "battery_id", "cycle_number", "source_cycle_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "by_type": {str(k): int(v) for k, v in anomaly_df["anomaly_type"].value_counts(dropna=False).to_dict().items()},
        "by_group": {str(k): int(v) for k, v in anomaly_df["anomaly_group"].value_counts(dropna=False).to_dict().items()},
        "by_severity": {str(k): int(v) for k, v in anomaly_df["severity"].value_counts(dropna=False).to_dict().items()},
        "by_phase": {str(k): int(v) for k, v in anomaly_df["phase"].value_counts(dropna=False).to_dict().items()},
        "model_affecting_records": int(anomaly_df["affects_modeling"].astype(bool).sum()),
    }

def finalize_targets_and_flags(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)

    df = df.sort_values(["battery_id", "phase", "cycle_number", "source_cycle_id"]).reset_index(drop=True)
    df["cycle_index"] = df.groupby(["battery_id", "phase"]).cumcount() + 1

    initial_capacity_records: list[dict[str, Any]] = []

    for (battery_id, phase), idx in tqdm(
        df.groupby(["battery_id", "phase"]).groups.items(),
        total=len(df.groupby(["battery_id", "phase"]).groups),
        desc="Estimating robust initial capacity",
        unit="battery-phase",
    ):
        idx_list = list(idx)
        sub = df.loc[idx_list].sort_values(["cycle_number", "cycle_index"])
        capacities = sub["phase_capacity_ah"].to_numpy(dtype=float)
        initial_capacity, info = robust_initial_capacity(capacities, args)
        eol_capacity = 0.8 * initial_capacity if np.isfinite(initial_capacity) else np.nan

        df.loc[idx_list, "initial_capacity_ah"] = initial_capacity
        df.loc[idx_list, "eol_capacity_ah"] = eol_capacity
        if np.isfinite(initial_capacity) and initial_capacity > 0:
            df.loc[idx_list, "soh"] = df.loc[idx_list, "phase_capacity_ah"] / initial_capacity
        else:
            df.loc[idx_list, "soh"] = np.nan

        initial_capacity_records.append(
            {
                "battery_id": battery_id,
                "phase": phase,
                "initial_capacity_ah": initial_capacity,
                "eol_capacity_ah": eol_capacity,
                **info,
            }
        )

    # Store robust-initial-capacity metadata in attrs for the caller to save.
    df.attrs["initial_capacity_records"] = initial_capacity_records

    df["remaining_cycles"] = df["cycle_life"] - df["cycle_number"]
    df.loc[~np.isfinite(df["remaining_cycles"]), "remaining_cycles"] = np.nan
    df["remaining_cycles_norm"] = df["remaining_cycles"] / df["cycle_life"]
    df.loc[~np.isfinite(df["remaining_cycles_norm"]), "remaining_cycles_norm"] = np.nan

    usable_flags = []
    exclude_reasons = []
    updated_warnings = []

    iterator = tqdm(df.iterrows(), total=len(df), desc="Assigning model-usability flags", unit="cycle")
    for _, row in iterator:
        voltage = np.asarray(row["voltage_v"], dtype=float)
        capacity = np.asarray(row["capacity_ah"], dtype=float)
        qv_ok = can_interpolate_qv_curve(
            voltage,
            capacity,
            min_voltage_span=args.min_voltage_span_for_interpolation,
            min_unique_voltage=args.min_unique_voltage_for_interpolation,
        )
        soh = finite_or_nan(row.get("soh", np.nan))
        soh_ok = np.isfinite(soh) and (float(args.min_soh_model_usable) < soh <= float(args.max_soh_model_usable))
        model_usable = (
            bool(row["is_valid_cycle"])
            and np.isfinite(row["cycle_life"])
            and np.isfinite(row["remaining_cycles"])
            and row["remaining_cycles"] > 0
            and np.isfinite(row["phase_capacity_ah"])
            and qv_ok
            and soh_ok
            and row["phase"] in ["charge", "discharge"]
        )
        usable_flags.append(bool(model_usable))

        warnings = row.get("cleaning_warnings", [])
        if not np.isfinite(soh):
            warnings = append_warning(warnings, "soh_not_finite")
        elif soh <= float(args.min_soh_warning) or soh > float(args.max_soh_warning):
            warnings = append_warning(warnings, f"soh_warning_outside_{args.min_soh_warning}_{args.max_soh_warning}")
        if np.isfinite(soh) and (soh <= float(args.min_soh_model_usable) or soh > float(args.max_soh_model_usable)):
            warnings = append_warning(warnings, f"soh_model_exclusion_outside_{args.min_soh_model_usable}_{args.max_soh_model_usable}")
        updated_warnings.append(warnings)

        if model_usable:
            exclude_reasons.append("")
        elif not np.isfinite(row["cycle_life"]):
            exclude_reasons.append("missing_cycle_life")
        elif not np.isfinite(row["remaining_cycles"]) or row["remaining_cycles"] <= 0:
            exclude_reasons.append("non_positive_remaining_cycles")
        elif not np.isfinite(soh):
            exclude_reasons.append("invalid_soh")
        elif not soh_ok:
            exclude_reasons.append("abnormal_soh")
        elif not qv_ok:
            exclude_reasons.append("insufficient_qv_interpolation_range")
        else:
            exclude_reasons.append("not_model_usable")

    df["is_model_usable"] = usable_flags
    df["exclude_reason"] = exclude_reasons
    df["cleaning_warnings"] = updated_warnings

    # Keep official column order and no extra columns in clean parquet.
    for col in STANDARD_CYCLE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    out = df[STANDARD_CYCLE_COLUMNS].reset_index(drop=True)
    out.attrs["initial_capacity_records"] = initial_capacity_records
    return out


def remove_path_if_exists(path: Path) -> None:
    """Remove a file or directory if it already exists."""
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def write_parquet_memory_safe(df: pd.DataFrame, output_file: Path, args: argparse.Namespace, description: str = "") -> None:
    """
    Write a dataframe with vector/object columns using small fastparquet row groups.

    The MATR clean tables contain many rows whose signal columns are Python lists.
    With fastparquet, writing the whole dataframe as one row group may allocate a
    very large bytes object and raise MemoryError. Small row groups reduce the
    peak memory required during serialization while preserving a single parquet
    output file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_name(output_file.name + ".tmp")

    base_size = int(getattr(args, "parquet_row_group_size", 500) or 500)
    candidate_sizes = [base_size, 250, 100, 50, 25, 10]
    candidate_sizes = [x for i, x in enumerate(candidate_sizes) if x > 0 and x not in candidate_sizes[:i]]
    compression = getattr(args, "parquet_compression", "SNAPPY")

    last_error: Exception | None = None
    for row_group_size in candidate_sizes:
        remove_path_if_exists(tmp_file)
        try:
            print(
                f"Writing {description or output_file.name} with fastparquet "
                f"row_group_offsets={row_group_size}, compression={compression} ..."
            )
            df.to_parquet(
                tmp_file,
                index=False,
                engine="fastparquet",
                compression=compression,
                row_group_offsets=int(row_group_size),
            )
            remove_path_if_exists(output_file)
            tmp_file.replace(output_file)
            return
        except MemoryError as exc:
            last_error = exc
            print(
                f"[WARN] MemoryError while writing {output_file.name} "
                f"with row_group_offsets={row_group_size}. Retrying with smaller row groups."
            )
            remove_path_if_exists(tmp_file)
            gc.collect()
        except TypeError as exc:
            # Some older pandas/fastparquet combinations may reject row_group_offsets
            # through pandas. In that case, fail clearly instead of silently writing a
            # huge one-row-group file.
            last_error = exc
            remove_path_if_exists(tmp_file)
            raise TypeError(
                "Your pandas/fastparquet combination did not accept row_group_offsets. "
                "Update fastparquet or use the previous command with a smaller output phase. "
                f"Original error: {exc}"
            ) from exc

    raise MemoryError(
        f"Unable to write {output_file} even with row_group_offsets={candidate_sizes[-1]}. "
        "Close other applications and rerun with --parquet-row-group-size 10, or split the phase output."
    ) from last_error


def make_phase_report(
    phase: str,
    df_clean: pd.DataFrame,
    df_rejected: pd.DataFrame,
    df_anomalies: pd.DataFrame,
    df_initial_capacity: pd.DataFrame,
) -> dict[str, Any]:
    """Build the compact report for one phase without retaining the large clean dataframe."""
    phase_report: dict[str, Any] = {
        "clean_rows": int(len(df_clean)),
        "model_usable_rows": int(df_clean["is_model_usable"].sum()) if not df_clean.empty else 0,
        "valid_rows": int(df_clean["is_valid_cycle"].sum()) if not df_clean.empty else 0,
        "rejected_rows": int(len(df_rejected)),
        "unique_batteries": int(df_clean["battery_id"].nunique()) if not df_clean.empty else 0,
        "exclude_reason_counts_clean": {},
        "exclude_reason_counts_rejected": {},
        "anomaly_summary": anomaly_summary(df_anomalies),
        "initial_capacity_estimation": {
            "rows": int(len(df_initial_capacity)),
            "method": "median_first_n_iqr_filtered",
        },
    }
    if not df_clean.empty and "exclude_reason" in df_clean.columns:
        phase_report["exclude_reason_counts_clean"] = {
            str(k): int(v) for k, v in df_clean["exclude_reason"].value_counts(dropna=False).to_dict().items()
        }
    if not df_rejected.empty and "exclude_reason" in df_rejected.columns:
        phase_report["exclude_reason_counts_rejected"] = {
            str(k): int(v) for k, v in df_rejected["exclude_reason"].value_counts(dropna=False).to_dict().items()
        }
    return phase_report


def clean_phase(phase: str, input_file: Path, output_file: Path, rejected_file: Path, anomaly_file: Path, initial_capacity_file: Path, args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not input_file.exists():
        empty_clean = pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)
        write_parquet_memory_safe(empty_clean, output_file, args, description=f"{phase} clean empty parquet")
        rejected = pd.DataFrame([{"phase": phase, "exclude_reason": f"input_file_not_found: {input_file}"}])
        anomalies = pd.DataFrame([{
            "dataset": DATASET,
            "phase": phase,
            "anomaly_type": "input_file_not_found",
            "anomaly_group": "io",
            "severity": "critical",
            "metric_name": "input_file",
            "metric_value": str(input_file),
            "affects_modeling": True,
            "description": "Extracted input file was not found.",
            "created_at": utc_now_iso(),
        }])
        initial_capacity_df = pd.DataFrame()
        rejected.to_csv(rejected_file, index=False, encoding="utf-8-sig")
        anomalies.to_csv(anomaly_file, index=False, encoding="utf-8-sig")
        initial_capacity_df.to_csv(initial_capacity_file, index=False, encoding="utf-8-sig")
        phase_report = make_phase_report(phase, empty_clean, rejected, anomalies, initial_capacity_df)
        return phase_report, anomalies, initial_capacity_df

    df_in = pd.read_parquet(input_file)
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    iterator = tqdm(
        df_in.iterrows(),
        total=len(df_in),
        desc=f"Cleaning {phase}",
        unit="cycle",
    )

    for _, row in iterator:
        clean_row, rejection = clean_one_cycle(row, phase=phase, args=args)
        if clean_row is not None:
            accepted_rows.append(clean_row)
        if rejection is not None:
            rejected_rows.append(rejection)

    df_clean = pd.DataFrame(accepted_rows)
    df_clean = finalize_targets_and_flags(df_clean, args)
    initial_capacity_df = pd.DataFrame(df_clean.attrs.get("initial_capacity_records", []))
    anomalies_df = build_cleaning_anomalies(df_clean, args)
    df_rejected = pd.DataFrame(rejected_rows)
    if df_rejected.empty:
        df_rejected = pd.DataFrame(columns=[
            "dataset", "source_file", "source_path", "battery_id", "battery_index",
            "cell_id_original", "cycle_number", "source_cycle_id", "cycle_pair_id",
            "phase", "exclude_reason", "cleaning_warnings",
        ])
    # Rejected-at-cleaning cycles are also part of the anomaly evidence.
    if not df_rejected.empty:
        rejected_anomalies = []
        for _, rejected_row in df_rejected.iterrows():
            rejected_anomalies.append(
                {
                    "dataset": DATASET,
                    "phase": phase,
                    "battery_id": rejected_row.get("battery_id", ""),
                    "battery_index": rejected_row.get("battery_index", np.nan),
                    "cell_id_original": rejected_row.get("cell_id_original", ""),
                    "cycle_index": np.nan,
                    "cycle_number": rejected_row.get("cycle_number", np.nan),
                    "source_cycle_id": rejected_row.get("source_cycle_id", ""),
                    "cycle_pair_id": rejected_row.get("cycle_pair_id", ""),
                    "anomaly_type": str(rejected_row.get("exclude_reason", "rejected_during_cleaning")),
                    "anomaly_group": "hard_rejection",
                    "severity": "critical",
                    "metric_name": "exclude_reason",
                    "metric_value": str(rejected_row.get("exclude_reason", "")),
                    "threshold_low": np.nan,
                    "threshold_high": np.nan,
                    "affects_modeling": True,
                    "is_valid_cycle": False,
                    "is_model_usable": False,
                    "exclude_reason": str(rejected_row.get("exclude_reason", "")),
                    "description": "Cycle was rejected before being written to the clean parquet file.",
                    "created_at": utc_now_iso(),
                }
            )
        anomalies_df = pd.concat([anomalies_df, pd.DataFrame(rejected_anomalies)], ignore_index=True)

    # Write the large clean dataframe only after rejection anomalies have been appended
    # and after the phase summary has been computed. The row-grouped writer avoids
    # fastparquet MemoryError on vector-heavy columns.
    phase_report = make_phase_report(phase, df_clean, df_rejected, anomalies_df, initial_capacity_df)
    write_parquet_memory_safe(df_clean, output_file, args, description=f"{phase} clean parquet")

    df_rejected.to_csv(rejected_file, index=False, encoding="utf-8-sig")
    anomalies_df.to_csv(anomaly_file, index=False, encoding="utf-8-sig")
    initial_capacity_df.to_csv(initial_capacity_file, index=False, encoding="utf-8-sig")

    # Release the vector-heavy objects before the next phase.
    del df_in, accepted_rows, rejected_rows, df_clean, df_rejected
    gc.collect()
    return phase_report, anomalies_df, initial_capacity_df


def build_report(phase_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dataset": DATASET,
        "cleaning_version": CLEANING_VERSION,
        "created_at": utc_now_iso(),
        "phases": phase_reports,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean extracted MATR charge/discharge cycles using the official schema.")
    parser.add_argument("--base-dir", type=str, default=None, help="Project root directory. Defaults to the known Batteries path or current directory.")
    parser.add_argument("--input-dir", type=str, default=None, help="Extracted cycle directory. Defaults to <base-dir>/data/interim/matr/extracted.")
    parser.add_argument("--output-dir", type=str, default=None, help="Clean output directory. Defaults to <base-dir>/data/interim/matr/clean.")

    parser.add_argument("--min-points-after-cleaning", type=int, default=20)
    # Version 2.0 keeps the complete contiguous charge/discharge segments by default.
    # Use --enable-voltage-window only if you intentionally want the older voltage-window trimming.
    parser.add_argument("--disable-voltage-window", dest="disable_voltage_window", action="store_true", default=True)
    parser.add_argument("--enable-voltage-window", dest="disable_voltage_window", action="store_false")
    parser.add_argument("--charge-v-min", type=float, default=2.10)
    parser.add_argument("--charge-v-max", type=float, default=3.55)
    parser.add_argument("--discharge-v-min", type=float, default=2.10)
    parser.add_argument("--discharge-v-max", type=float, default=3.55)
    parser.add_argument("--truncate-after-cutoff", action="store_true", default=True)
    parser.add_argument("--no-truncate-after-cutoff", dest="truncate_after_cutoff", action="store_false")
    parser.add_argument("--force-capacity-monotonic", action="store_true", default=True)
    parser.add_argument("--no-force-capacity-monotonic", dest="force_capacity_monotonic", action="store_false")

    parser.add_argument("--fill-missing-temperature-c", type=float, default=30.0)
    parser.add_argument("--nominal-capacity-ah", type=float, default=1.1)
    parser.add_argument("--nominal-voltage-v", type=float, default=3.3)
    parser.add_argument("--ambient-temperature-c", type=float, default=30.0)
    parser.add_argument("--min-voltage-span-for-interpolation", type=float, default=0.20)
    parser.add_argument("--min-unique-voltage-for-interpolation", type=int, default=20)

    # Robust initial capacity estimation.
    parser.add_argument("--initial-capacity-n-cycles", type=int, default=10)
    parser.add_argument("--initial-capacity-min-abs-ah", type=float, default=0.05)
    parser.add_argument("--initial-capacity-iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--initial-capacity-min-cycles-for-iqr", type=int, default=4)
    parser.add_argument("--initial-capacity-min-cycles-after-iqr", type=int, default=2)

    # SOH and capacity anomaly thresholds. The model-usable interval is stricter;
    # the warning interval is used only to document article/report anomalies.
    parser.add_argument("--min-soh-model-usable", type=float, default=0.05)
    parser.add_argument("--max-soh-model-usable", type=float, default=1.20)
    parser.add_argument("--min-soh-warning", type=float, default=0.50)
    parser.add_argument("--max-soh-warning", type=float, default=1.10)
    parser.add_argument("--phase-capacity-low-ratio-warning", type=float, default=0.50)
    parser.add_argument("--phase-capacity-high-ratio-warning", type=float, default=1.20)

    # fastparquet memory control. Small row groups are important because the clean
    # MATR table stores long vectors in object/list columns.
    parser.add_argument("--parquet-row-group-size", type=int, default=500)
    parser.add_argument("--parquet-compression", type=str, default="SNAPPY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else base_dir / "data" / "interim" / "matr" / "extracted"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else base_dir / "data" / "interim" / "matr" / "clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("MATR cleaning - version 2.1 memory-safe contiguous split compatible")
    print("=" * 88)
    print(f"Base directory  : {base_dir}")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")

    phase_reports: dict[str, dict[str, Any]] = {}
    anomaly_tables: list[pd.DataFrame] = []
    initial_capacity_tables: list[pd.DataFrame] = []
    for phase in ["charge", "discharge"]:
        input_file = input_dir / f"matr_{phase}_extracted_cycles.parquet"
        output_file = output_dir / f"matr_{phase}_clean_cycles.parquet"
        rejected_file = output_dir / f"matr_{phase}_rejected_cycles.csv"
        anomaly_file = output_dir / f"matr_{phase}_cycle_anomalies.csv"
        initial_capacity_file = output_dir / f"matr_{phase}_initial_capacity_estimates.csv"
        phase_report, df_anomalies, df_initial_capacity = clean_phase(
            phase,
            input_file,
            output_file,
            rejected_file,
            anomaly_file,
            initial_capacity_file,
            args,
        )
        phase_reports[phase] = phase_report
        anomaly_tables.append(df_anomalies)
        initial_capacity_tables.append(df_initial_capacity)
        print(
            f"[{phase}] clean rows: {phase_report['clean_rows']:,} | "
            f"model usable: {phase_report['model_usable_rows']:,} | "
            f"rejected: {phase_report['rejected_rows']:,} | "
            f"anomaly records: {phase_report['anomaly_summary']['total_anomaly_records']:,}"
        )
        print(f"[{phase}] clean parquet       : {output_file}")
        print(f"[{phase}] rejected csv        : {rejected_file}")
        print(f"[{phase}] anomaly csv         : {anomaly_file}")
        print(f"[{phase}] initial-capacity csv: {initial_capacity_file}")

    report = build_report(phase_reports)
    report_file = output_dir / "matr_cleaning_report.json"
    with report_file.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    all_anomalies = pd.concat(anomaly_tables, ignore_index=True) if anomaly_tables else pd.DataFrame()
    all_anomalies_file = output_dir / "matr_cycle_anomalies_all.csv"
    all_anomalies.to_csv(all_anomalies_file, index=False, encoding="utf-8-sig")

    all_initial_capacity = pd.concat(initial_capacity_tables, ignore_index=True) if initial_capacity_tables else pd.DataFrame()
    all_initial_capacity_file = output_dir / "matr_initial_capacity_estimates_all.csv"
    all_initial_capacity.to_csv(all_initial_capacity_file, index=False, encoding="utf-8-sig")

    anomaly_report_file = output_dir / "matr_cleaning_anomaly_report.json"
    with anomaly_report_file.open("w", encoding="utf-8") as fp:
        json.dump(anomaly_summary(all_anomalies), fp, indent=2, ensure_ascii=False)

    print(f"Cleaning report         : {report_file}")
    print(f"Cleaning anomaly report : {anomaly_report_file}")
    print(f"All cycle anomalies     : {all_anomalies_file}")
    print(f"All initial capacities  : {all_initial_capacity_file}")
    print("Done.")


if __name__ == "__main__":
    main()
