# -*- coding: utf-8 -*-
"""
Clean Oxford/ORA extracted charge and discharge cycles.

Inputs:
    data/interim/ora/extracted/ora_charge_extracted_cycles.parquet
    data/interim/ora/extracted/ora_discharge_extracted_cycles.parquet

Outputs:
    data/interim/ora/clean/ora_charge_clean_cycles.parquet
    data/interim/ora/clean/ora_discharge_clean_cycles.parquet
    data/interim/ora/clean/ora_charge_rejected_cycles.csv
    data/interim/ora/clean/ora_discharge_rejected_cycles.csv
    data/reports/ora/cleaning_report.json

The clean files follow the project STANDARD_CYCLE_COLUMNS order and keep charge
and discharge as independent modeling branches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

try:
    from .ora_schema import (
        AMBIENT_TEMPERATURE_C,
        CLEANING_VERSION,
        DATASET_FOLDER,
        DATASET_NAME,
        EOL_CAPACITY_AH,
        NOMINAL_CAPACITY_AH,
        STANDARD_CYCLE_COLUMNS,
        align_vectors,
        can_interpolate_qv_curve,
        empty_standard_dataframe,
        ensure_dir,
        ensure_python_scalars,
        infer_default_base_dir,
        integrate_capacity_from_current,
        integrate_energy_from_power,
        median_sample_period,
        normalize_capacity_to_ah,
        normalize_time_seconds,
        now_iso,
        reorder_standard_columns,
        safe_array,
        to_float_list,
        validate_standard_schema,
        vector_stats,
        write_json,
    )
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from ora_schema import (  # type: ignore
        AMBIENT_TEMPERATURE_C,
        CLEANING_VERSION,
        DATASET_FOLDER,
        DATASET_NAME,
        EOL_CAPACITY_AH,
        NOMINAL_CAPACITY_AH,
        STANDARD_CYCLE_COLUMNS,
        align_vectors,
        can_interpolate_qv_curve,
        empty_standard_dataframe,
        ensure_dir,
        ensure_python_scalars,
        infer_default_base_dir,
        integrate_capacity_from_current,
        integrate_energy_from_power,
        median_sample_period,
        normalize_capacity_to_ah,
        normalize_time_seconds,
        now_iso,
        reorder_standard_columns,
        safe_array,
        to_float_list,
        validate_standard_schema,
        vector_stats,
        write_json,
    )


def sort_and_deduplicate_time(
    vectors: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int, list[str]]:
    warnings: list[str] = []
    time = safe_array(vectors.get("time_s", []), keep_nan=True)
    if time.size == 0:
        return vectors, 0, ["empty_time_vector"]

    order = np.argsort(time, kind="stable")
    sorted_vectors = {key: safe_array(value, keep_nan=True)[order] for key, value in vectors.items()}
    sorted_time = sorted_vectors["time_s"]

    keep = np.ones(sorted_time.size, dtype=bool)
    keep[1:] = np.diff(sorted_time) > 0
    removed = int(np.sum(~keep))
    if removed > 0:
        warnings.append("removed_non_increasing_or_duplicate_time_samples")

    deduped = {key: value[keep] for key, value in sorted_vectors.items()}
    if deduped["time_s"].size > 0:
        deduped["time_s"] = deduped["time_s"] - deduped["time_s"][0]

    return deduped, removed, warnings


def apply_voltage_trim(
    vectors: dict[str, np.ndarray],
    *,
    voltage_min: float,
    voltage_max: float,
) -> tuple[dict[str, np.ndarray], int, int, int, list[str]]:
    warnings: list[str] = []
    voltage = safe_array(vectors.get("voltage_v", []), keep_nan=True)
    if voltage.size == 0:
        return vectors, 0, 0, 0, ["empty_voltage_vector"]

    valid_voltage = np.isfinite(voltage) & (voltage >= voltage_min) & (voltage <= voltage_max)
    if not np.any(valid_voltage):
        return vectors, int(voltage.size), 0, int(max(voltage.size - 1, 0)), ["all_points_outside_voltage_window"]

    indices = np.flatnonzero(valid_voltage)
    start = int(indices[0])
    end = int(indices[-1])
    mask = np.zeros(voltage.size, dtype=bool)
    mask[start:end + 1] = True
    mask &= valid_voltage
    removed = int(np.sum(~mask))

    if removed > 0:
        warnings.append(f"trimmed_points_outside_voltage_window:{voltage_min:.3f}_{voltage_max:.3f}")

    trimmed = {key: safe_array(value, keep_nan=True)[mask] for key, value in vectors.items()}
    if trimmed.get("time_s", np.array([])).size > 0:
        trimmed["time_s"] = trimmed["time_s"] - trimmed["time_s"][0]
    return trimmed, removed, start, end, warnings


def enforce_current_sign(current_a: np.ndarray, phase: str) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    current = safe_array(current_a, keep_nan=True)
    if current.size == 0 or not np.isfinite(current).any():
        reconstructed = 0.740 if phase == "charge" else -0.740
        warnings.append("current_reconstructed_in_cleaning")
        return np.full(max(current.size, 1), reconstructed, dtype=float), warnings

    mean_current = float(np.nanmean(current))
    if phase == "charge" and mean_current < 0:
        current = -current
        warnings.append("charge_current_sign_flipped")
    elif phase == "discharge" and mean_current > 0:
        current = -current
        warnings.append("discharge_current_sign_flipped")
    return current, warnings




def sanitize_temperature_vector(
    temperature_c: np.ndarray,
    *,
    n: int,
    temperature_source: str,
    has_temperature: bool,
) -> tuple[np.ndarray, str, bool, list[str]]:
    """Return a cleaned Oxford temperature vector.

    Oxford characterization cycles are nominally performed around 40 °C. Some
    malformed vectors can appear near zero due to missing or corrupted records;
    those vectors must not be treated as measured thermal information because
    they would distort temperature features.
    """
    warnings: list[str] = []
    temperature = safe_array(temperature_c, keep_nan=True)

    if temperature.size != n or not np.isfinite(temperature).any():
        warnings.append("temperature_filled_with_nominal_40C")
        return np.full(n, AMBIENT_TEMPERATURE_C, dtype=float), "nominal_constant", False, warnings

    if temperature.size > n:
        temperature = temperature[:n]
    elif temperature.size < n:
        fill_value = float(np.nanmedian(temperature[np.isfinite(temperature)]))
        temperature = np.pad(temperature, (0, n - temperature.size), constant_values=fill_value)
        warnings.append("temperature_vector_padded_with_median")

    finite = temperature[np.isfinite(temperature)]
    if finite.size == 0:
        warnings.append("temperature_filled_with_nominal_40C")
        return np.full(n, AMBIENT_TEMPERATURE_C, dtype=float), "nominal_constant", False, warnings

    median_temp = float(np.nanmedian(finite))
    outlier_mask = (~np.isfinite(temperature)) | (temperature < 15.0) | (temperature > 70.0)
    outlier_ratio = float(np.mean(outlier_mask)) if temperature.size else 1.0

    if median_temp < 15.0 or median_temp > 70.0 or outlier_ratio > 0.20:
        warnings.append("temperature_replaced_with_nominal_40C_due_to_implausible_values")
        return np.full(n, AMBIENT_TEMPERATURE_C, dtype=float), "nominal_constant", False, warnings

    if np.any(outlier_mask):
        temperature = np.where(outlier_mask, median_temp, temperature)
        warnings.append("temperature_outliers_replaced_with_cycle_median")

    if np.isnan(temperature).any():
        temperature = np.where(np.isfinite(temperature), temperature, median_temp)
        warnings.append("temperature_nan_values_filled_with_median")

    return temperature, temperature_source or "measured", bool(has_temperature), warnings

def clean_single_cycle(
    row: pd.Series,
    *,
    phase: str,
    voltage_min: float,
    voltage_max: float,
    round_vectors: int | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    warnings = list(row.get("cleaning_warnings", []) or [])
    source_id = row.get("source_cycle_id", "")

    raw_vectors = {
        "time_s": safe_array(row.get("time_s", []), keep_nan=True),
        "voltage_v": safe_array(row.get("voltage_v", []), keep_nan=True),
        "current_a": safe_array(row.get("current_a", []), keep_nan=True),
        "temperature_c": safe_array(row.get("temperature_c", []), keep_nan=True),
        "capacity_ah": safe_array(row.get("capacity_ah", []), keep_nan=True),
    }

    # Keep temperature NaNs if the source temperature is unknown, but require the
    # structural electrical vectors to be finite.
    aligned = align_vectors(raw_vectors, allow_nan_keys={"temperature_c"})
    if not aligned:
        rejected = build_rejected_row(row, phase, "empty_or_unalignable_vectors")
        return None, rejected

    n0 = min(len(value) for value in aligned.values())
    if n0 <= 0:
        rejected = build_rejected_row(row, phase, "empty_aligned_vectors")
        return None, rejected

    aligned = {key: value[:n0] for key, value in aligned.items()}
    aligned, removed_time, time_warnings = sort_and_deduplicate_time(aligned)
    warnings.extend(time_warnings)

    if aligned["time_s"].size < 2:
        rejected = build_rejected_row(row, phase, "too_few_time_points_after_deduplication")
        return None, rejected

    aligned, removed_voltage, trim_start, trim_end, voltage_warnings = apply_voltage_trim(
        aligned,
        voltage_min=voltage_min,
        voltage_max=voltage_max,
    )
    warnings.extend(voltage_warnings)

    n_removed_points = int(removed_time + removed_voltage)
    n = min(len(value) for value in aligned.values())
    if n <= 0:
        rejected = build_rejected_row(row, phase, "no_points_after_voltage_trim")
        return None, rejected

    time_s = normalize_time_seconds(aligned["time_s"][:n])
    voltage_v = safe_array(aligned["voltage_v"][:n], keep_nan=True)
    current_a, current_warnings = enforce_current_sign(aligned["current_a"][:n], phase)
    warnings.extend(current_warnings)
    current_a = current_a[:n] if current_a.size >= n else np.full(n, np.nanmean(current_a), dtype=float)

    temperature_source = str(row.get("temperature_source", "unknown_nan"))
    has_temperature = bool(row.get("has_temperature", False))
    temperature_c, temperature_source, has_temperature, temperature_warnings = sanitize_temperature_vector(
        aligned["temperature_c"][:n],
        n=n,
        temperature_source=temperature_source,
        has_temperature=has_temperature,
    )
    warnings.extend(temperature_warnings)

    capacity_ah = normalize_capacity_to_ah(aligned["capacity_ah"][:n])
    capacity_source = str(row.get("capacity_source", "measured"))
    if capacity_ah.size != n or not np.isfinite(capacity_ah).any() or float(np.nanmax(capacity_ah)) <= 0:
        capacity_ah = integrate_capacity_from_current(current_a, time_s)
        capacity_source = "integrated_current"
        warnings.append("capacity_reconstructed_from_current_in_cleaning")
    else:
        # Re-run normalization after trimming and reset exactly to zero.
        capacity_ah = normalize_capacity_to_ah(capacity_ah)

    if capacity_ah.size != n:
        n2 = min(n, capacity_ah.size)
        time_s = time_s[:n2]
        voltage_v = voltage_v[:n2]
        current_a = current_a[:n2]
        temperature_c = temperature_c[:n2]
        capacity_ah = capacity_ah[:n2]
        n = n2
        warnings.append("vectors_realigned_after_capacity_reconstruction")

    power_w = voltage_v * current_a
    energy_wh = integrate_energy_from_power(power_w, time_s)

    voltage_stats = vector_stats(voltage_v)
    current_stats = vector_stats(current_a)
    temperature_stats = vector_stats(temperature_c)

    duration_s = float(time_s[-1] - time_s[0]) if n >= 2 else np.nan
    phase_capacity_ah = float(capacity_ah[-1]) if capacity_ah.size else np.nan
    phase_energy_wh = float(energy_wh[-1]) if energy_wh.size else np.nan

    is_valid_cycle = evaluate_structural_validity(time_s, voltage_v, current_a, capacity_ah, duration_s, phase_capacity_ah)
    exclude_reason = "" if is_valid_cycle else structural_exclude_reason(time_s, voltage_v, current_a, capacity_ah, duration_s, phase_capacity_ah)

    cleaned = {column: row.get(column, "") for column in STANDARD_CYCLE_COLUMNS}
    cleaned.update({
        "dataset": str(row.get("dataset", DATASET_NAME)) or DATASET_NAME,
        "source_file": str(row.get("source_file", "")),
        "source_path": str(row.get("source_path", "")),
        "battery_id": str(row.get("battery_id", "")),
        "battery_index": int(row.get("battery_index", 0) or 0),
        "cell_id_original": str(row.get("cell_id_original", "")),
        "cycle_index": int(row.get("cycle_index", 0) or 0),
        "cycle_number": float(row.get("cycle_number", np.nan)),
        "source_cycle_id": str(source_id),
        "cycle_pair_id": str(row.get("cycle_pair_id", "")),
        "phase": phase,
        "cycle_life": np.nan,
        "remaining_cycles": np.nan,
        "remaining_cycles_norm": np.nan,
        "eol_criterion": "80pct_nominal_capacity",
        "eol_capacity_ah": EOL_CAPACITY_AH,
        "initial_capacity_ah": np.nan,
        "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "soh": np.nan,
        "time_s": to_float_list(time_s, decimals=round_vectors),
        "voltage_v": to_float_list(voltage_v, decimals=round_vectors),
        "current_a": to_float_list(current_a, decimals=round_vectors),
        "temperature_c": to_float_list(temperature_c, decimals=round_vectors),
        "capacity_ah": to_float_list(capacity_ah, decimals=round_vectors),
        "energy_wh": to_float_list(energy_wh, decimals=round_vectors),
        "power_w": to_float_list(power_w, decimals=round_vectors),
        "n_points": int(n),
        "duration_s": duration_s,
        "sample_period_median_s": median_sample_period(time_s),
        "voltage_min_v": voltage_stats["min"],
        "voltage_max_v": voltage_stats["max"],
        "voltage_start_v": voltage_stats["start"],
        "voltage_end_v": voltage_stats["end"],
        "current_min_a": current_stats["min"],
        "current_max_a": current_stats["max"],
        "current_mean_a": current_stats["mean"],
        "temperature_min_c": temperature_stats["min"],
        "temperature_max_c": temperature_stats["max"],
        "temperature_mean_c": temperature_stats["mean"],
        "phase_capacity_ah": phase_capacity_ah,
        "phase_energy_wh": phase_energy_wh,
        "voltage_source": str(row.get("voltage_source", "measured")) or "measured",
        "current_source": str(row.get("current_source", "reconstructed")) or "reconstructed",
        "temperature_source": temperature_source,
        "capacity_source": capacity_source,
        "energy_source": "integrated_power",
        "chemistry": str(row.get("chemistry", "LCO")) or "LCO",
        "form_factor": str(row.get("form_factor", "pouch")) or "pouch",
        "manufacturer": str(row.get("manufacturer", "Kokam")) or "Kokam",
        "nominal_voltage_v": float(row.get("nominal_voltage_v", np.nan)) if pd.notna(row.get("nominal_voltage_v", np.nan)) else np.nan,
        "ambient_temperature_c": AMBIENT_TEMPERATURE_C,
        "charge_protocol": str(row.get("charge_protocol", "C1ch characterization at +0.740 A")),
        "discharge_protocol": str(row.get("discharge_protocol", "C1dc characterization at -0.740 A")),
        "c_rate_charge": 1.0 if phase == "charge" else np.nan,
        "c_rate_discharge": 1.0 if phase == "discharge" else np.nan,
        "dod": float(row.get("dod", np.nan)) if pd.notna(row.get("dod", np.nan)) else np.nan,
        "policy_readable": str(row.get("policy_readable", "")),
        "barcode": str(row.get("barcode", "")),
        "is_valid_cycle": bool(is_valid_cycle),
        "is_model_usable": False,
        "exclude_reason": exclude_reason,
        "cleaning_warnings": warnings,
        "n_removed_points": int(n_removed_points),
        "trim_start_index": int(trim_start),
        "trim_end_index": int(trim_end),
        "has_voltage": bool(np.isfinite(voltage_v).any()),
        "has_current": bool(np.isfinite(current_a).any()),
        "has_temperature": bool(has_temperature),
        "has_capacity": bool(np.isfinite(capacity_ah).any()),
        "has_energy": bool(np.isfinite(energy_wh).any()),
        "adapter_version": str(row.get("adapter_version", "ora_adapter_v1")),
        "cleaning_version": CLEANING_VERSION,
        "created_at": now_iso(),
    })

    return ensure_python_scalars(cleaned), None


def evaluate_structural_validity(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    capacity_ah: np.ndarray,
    duration_s: float,
    phase_capacity_ah: float,
) -> bool:
    return bool(
        len(time_s) >= 20
        and np.all(np.isfinite(time_s))
        and np.all(np.diff(time_s) > 0)
        and np.isfinite(voltage_v).any()
        and np.nanmax(voltage_v) > np.nanmin(voltage_v)
        and np.isfinite(current_a).any()
        and np.isfinite(capacity_ah).any()
        and np.isfinite(phase_capacity_ah)
        and phase_capacity_ah > 0
        and np.isfinite(duration_s)
        and duration_s > 0
    )


def structural_exclude_reason(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    capacity_ah: np.ndarray,
    duration_s: float,
    phase_capacity_ah: float,
) -> str:
    if len(time_s) < 20:
        return "too_few_points"
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0):
        return "invalid_time_vector"
    if not np.isfinite(voltage_v).any():
        return "missing_voltage"
    if np.nanmax(voltage_v) <= np.nanmin(voltage_v):
        return "invalid_voltage_range"
    if not np.isfinite(current_a).any() and not np.isfinite(capacity_ah).any():
        return "missing_current_and_capacity"
    if not np.isfinite(phase_capacity_ah) or phase_capacity_ah <= 0:
        return "invalid_phase_capacity"
    if not np.isfinite(duration_s) or duration_s <= 0:
        return "invalid_duration"
    return "invalid_cycle"


def build_rejected_row(row: pd.Series, phase: str, reason: str) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", DATASET_NAME),
        "battery_id": row.get("battery_id", ""),
        "battery_index": row.get("battery_index", np.nan),
        "cell_id_original": row.get("cell_id_original", ""),
        "cycle_index": row.get("cycle_index", np.nan),
        "cycle_number": row.get("cycle_number", np.nan),
        "source_cycle_id": row.get("source_cycle_id", ""),
        "cycle_pair_id": row.get("cycle_pair_id", ""),
        "phase": phase,
        "exclude_reason": reason,
    }


def estimate_cycle_life_for_group(group: pd.DataFrame) -> dict[str, Any]:
    ordered = group.sort_values("cycle_number")
    x = pd.to_numeric(ordered["cycle_number"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(ordered["phase_capacity_ah"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return {"cycle_life": np.nan, "method": "no_valid_capacity", "eol_observed": False}

    below = y <= EOL_CAPACITY_AH
    if not np.any(below):
        return {"cycle_life": np.nan, "method": "not_reached", "eol_observed": False}

    first_idx = int(np.argmax(below))
    if first_idx == 0:
        return {"cycle_life": float(x[first_idx]), "method": "first_point_below_threshold", "eol_observed": True}

    x0, x1 = float(x[first_idx - 1]), float(x[first_idx])
    y0, y1 = float(y[first_idx - 1]), float(y[first_idx])
    if y1 == y0:
        cycle_life = x1
        method = "first_below_flat_segment"
    else:
        fraction = float(np.clip((EOL_CAPACITY_AH - y0) / (y1 - y0), 0.0, 1.0))
        cycle_life = x0 + fraction * (x1 - x0)
        method = "linear_interpolation_between_characterizations"
    return {"cycle_life": float(cycle_life), "method": method, "eol_observed": True}


def assign_targets_and_flags(df: pd.DataFrame, phase: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return reorder_standard_columns(df), pd.DataFrame(), pd.DataFrame()

    df = df.sort_values(["battery_id", "cycle_number", "cycle_index"]).copy()
    eol_rows: list[dict[str, Any]] = []

    for battery_id, index in tqdm(df.groupby("battery_id").groups.items(), desc=f"{phase} EOL", unit="battery"):
        idx = list(index)
        group = df.loc[idx].sort_values("cycle_number")

        valid_capacity = pd.to_numeric(group["phase_capacity_ah"], errors="coerce")
        finite_capacity = valid_capacity[np.isfinite(valid_capacity)]
        initial_capacity = float(finite_capacity.iloc[0]) if len(finite_capacity) else np.nan
        eol = estimate_cycle_life_for_group(group[group["is_valid_cycle"] == True])

        df.loc[idx, "initial_capacity_ah"] = initial_capacity
        if np.isfinite(initial_capacity) and initial_capacity > 0:
            df.loc[idx, "soh"] = pd.to_numeric(df.loc[idx, "phase_capacity_ah"], errors="coerce") / initial_capacity
        else:
            df.loc[idx, "soh"] = np.nan

        df.loc[idx, "cycle_life"] = eol["cycle_life"]
        df.loc[idx, "eol_criterion"] = "80pct_nominal_capacity"
        df.loc[idx, "eol_capacity_ah"] = EOL_CAPACITY_AH
        if np.isfinite(eol["cycle_life"]):
            df.loc[idx, "remaining_cycles"] = eol["cycle_life"] - pd.to_numeric(df.loc[idx, "cycle_number"], errors="coerce")
            df.loc[idx, "remaining_cycles_norm"] = df.loc[idx, "remaining_cycles"] / eol["cycle_life"]
        else:
            df.loc[idx, "remaining_cycles"] = np.nan
            df.loc[idx, "remaining_cycles_norm"] = np.nan

        eol_rows.append({
            "phase": phase,
            "battery_id": battery_id,
            "initial_capacity_ah": initial_capacity,
            "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
            "eol_capacity_ah": EOL_CAPACITY_AH,
            "cycle_life": eol["cycle_life"],
            "eol_method": eol["method"],
            "eol_observed": eol["eol_observed"],
            "n_cycles": int(len(group)),
        })

    for row_index in tqdm(df.index, desc=f"{phase} model flags", unit="cycle"):
        row = df.loc[row_index]
        reason = str(row.get("exclude_reason", ""))
        valid = bool(row.get("is_valid_cycle", False))
        remaining = row.get("remaining_cycles", np.nan)
        cycle_life = row.get("cycle_life", np.nan)
        voltage = row.get("voltage_v", [])
        capacity = row.get("capacity_ah", [])

        usable = bool(
            valid
            and np.isfinite(cycle_life)
            and np.isfinite(remaining)
            and float(remaining) > 0
            and can_interpolate_qv_curve(voltage, capacity)
            and row.get("phase") in ["charge", "discharge"]
        )
        df.loc[row_index, "is_model_usable"] = usable

        if usable:
            df.loc[row_index, "exclude_reason"] = ""
        else:
            if not valid:
                df.loc[row_index, "exclude_reason"] = reason or "invalid_cycle"
            elif not np.isfinite(cycle_life):
                df.loc[row_index, "exclude_reason"] = "eol_not_observed"
            elif not np.isfinite(remaining) or float(remaining) <= 0:
                df.loc[row_index, "exclude_reason"] = "non_positive_remaining_cycles"
            elif not can_interpolate_qv_curve(voltage, capacity):
                df.loc[row_index, "exclude_reason"] = "insufficient_voltage_range_for_qv_interpolation"
            else:
                df.loc[row_index, "exclude_reason"] = "not_model_usable"

    rejected = df[df["is_model_usable"] != True][[
        "dataset", "battery_id", "battery_index", "cell_id_original", "cycle_index",
        "cycle_number", "source_cycle_id", "cycle_pair_id", "phase", "exclude_reason",
        "is_valid_cycle", "is_model_usable", "n_points", "phase_capacity_ah",
        "voltage_min_v", "voltage_max_v", "remaining_cycles",
    ]].copy()

    eol_df = pd.DataFrame(eol_rows)
    return reorder_standard_columns(df), rejected, eol_df




def build_rejected_cycles_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "dataset", "battery_id", "battery_index", "cell_id_original", "cycle_index",
            "cycle_number", "source_cycle_id", "cycle_pair_id", "phase", "exclude_reason",
            "is_valid_cycle", "is_model_usable", "n_points", "phase_capacity_ah",
            "voltage_min_v", "voltage_max_v", "remaining_cycles",
        ])
    return df[df["is_model_usable"] != True][[
        "dataset", "battery_id", "battery_index", "cell_id_original", "cycle_index",
        "cycle_number", "source_cycle_id", "cycle_pair_id", "phase", "exclude_reason",
        "is_valid_cycle", "is_model_usable", "n_points", "phase_capacity_ah",
        "voltage_min_v", "voltage_max_v", "remaining_cycles",
    ]].copy()


def build_phase_report(input_file: Path, phase: str, clean_df: pd.DataFrame, rejected_df: pd.DataFrame) -> dict[str, Any]:
    reason_counts = rejected_df["exclude_reason"].value_counts(dropna=False).to_dict() if not rejected_df.empty else {}
    return {
        "phase": phase,
        "input_file": str(input_file),
        "n_input_rows": int(len(pd.read_parquet(input_file))) if input_file.exists() else 0,
        "n_clean_rows": int(len(clean_df)),
        "n_valid_rows": int(clean_df["is_valid_cycle"].sum()) if not clean_df.empty else 0,
        "n_model_usable_rows": int(clean_df["is_model_usable"].sum()) if not clean_df.empty else 0,
        "n_rejected_rows": int(len(rejected_df)),
        "rejection_reason_counts": reason_counts,
    }


def recompute_model_flags(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for row_index in tqdm(df.index, desc=f"{phase} common-target model flags", unit="cycle"):
        row = df.loc[row_index]
        reason = str(row.get("exclude_reason", ""))
        valid = bool(row.get("is_valid_cycle", False))
        remaining = row.get("remaining_cycles", np.nan)
        cycle_life = row.get("cycle_life", np.nan)
        voltage = row.get("voltage_v", [])
        capacity = row.get("capacity_ah", [])

        usable = bool(
            valid
            and np.isfinite(cycle_life)
            and np.isfinite(remaining)
            and float(remaining) > 0
            and can_interpolate_qv_curve(voltage, capacity)
            and row.get("phase") in ["charge", "discharge"]
        )
        df.loc[row_index, "is_model_usable"] = usable

        if usable:
            df.loc[row_index, "exclude_reason"] = ""
        else:
            if not valid:
                df.loc[row_index, "exclude_reason"] = reason or "invalid_cycle"
            elif not np.isfinite(cycle_life):
                df.loc[row_index, "exclude_reason"] = "eol_not_observed"
            elif not np.isfinite(remaining) or float(remaining) <= 0:
                df.loc[row_index, "exclude_reason"] = "non_positive_remaining_cycles"
            elif not can_interpolate_qv_curve(voltage, capacity):
                df.loc[row_index, "exclude_reason"] = "insufficient_voltage_range_for_qv_interpolation"
            else:
                df.loc[row_index, "exclude_reason"] = "not_model_usable"
    return df


def apply_discharge_eol_targets(phase_frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Use discharge capacity to define battery-level RUL targets for both branches.

    RUL is a battery-level target, not a phase-specific target. Therefore, the
    clean charge branch keeps charge-specific capacity and SOH, but receives the
    same cycle_life and remaining_cycles as the discharge branch for the same
    battery.
    """
    if "discharge" not in phase_frames or phase_frames["discharge"].empty:
        return phase_frames, pd.DataFrame()

    discharge_df = phase_frames["discharge"]
    eol_reference: dict[str, dict[str, Any]] = {}
    for battery_id, group in tqdm(discharge_df.groupby("battery_id"), desc="discharge EOL reference", unit="battery"):
        eol_reference[str(battery_id)] = estimate_cycle_life_for_group(group[group["is_valid_cycle"] == True])

    updated: dict[str, pd.DataFrame] = {}
    eol_rows: list[dict[str, Any]] = []

    for phase, df in phase_frames.items():
        if df.empty:
            updated[phase] = df
            continue
        df = df.sort_values(["battery_id", "cycle_number", "cycle_index"]).copy()
        for battery_id, index in tqdm(df.groupby("battery_id").groups.items(), desc=f"{phase} common-target assignment", unit="battery"):
            idx = list(index)
            group = df.loc[idx].sort_values("cycle_number")
            valid_capacity = pd.to_numeric(group["phase_capacity_ah"], errors="coerce")
            finite_capacity = valid_capacity[np.isfinite(valid_capacity)]
            initial_capacity = float(finite_capacity.iloc[0]) if len(finite_capacity) else np.nan

            ref = eol_reference.get(str(battery_id), {"cycle_life": np.nan, "method": "missing_discharge_reference", "eol_observed": False})
            cycle_life = ref.get("cycle_life", np.nan)

            df.loc[idx, "initial_capacity_ah"] = initial_capacity
            if np.isfinite(initial_capacity) and initial_capacity > 0:
                df.loc[idx, "soh"] = pd.to_numeric(df.loc[idx, "phase_capacity_ah"], errors="coerce") / initial_capacity
            else:
                df.loc[idx, "soh"] = np.nan

            df.loc[idx, "cycle_life"] = cycle_life
            df.loc[idx, "eol_criterion"] = "80pct_nominal_capacity_from_discharge_branch"
            df.loc[idx, "eol_capacity_ah"] = EOL_CAPACITY_AH
            if np.isfinite(cycle_life):
                df.loc[idx, "remaining_cycles"] = cycle_life - pd.to_numeric(df.loc[idx, "cycle_number"], errors="coerce")
                df.loc[idx, "remaining_cycles_norm"] = df.loc[idx, "remaining_cycles"] / cycle_life
            else:
                df.loc[idx, "remaining_cycles"] = np.nan
                df.loc[idx, "remaining_cycles_norm"] = np.nan

            eol_rows.append({
                "phase": phase,
                "battery_id": battery_id,
                "initial_capacity_ah": initial_capacity,
                "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
                "eol_capacity_ah": EOL_CAPACITY_AH,
                "cycle_life": cycle_life,
                "eol_method": ref.get("method", "unknown"),
                "eol_observed": ref.get("eol_observed", False),
                "target_source_phase": "discharge",
                "n_cycles": int(len(group)),
            })
        df = recompute_model_flags(df, phase)
        updated[phase] = reorder_standard_columns(df)

    return updated, pd.DataFrame(eol_rows)

def clean_phase_file(
    *,
    input_file: Path,
    phase: str,
    voltage_min: float,
    voltage_max: float,
    round_vectors: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not input_file.exists():
        raise FileNotFoundError(f"Extracted {phase} file not found: {input_file}")

    df_in = pd.read_parquet(input_file)
    if df_in.empty:
        empty_df = empty_standard_dataframe()
        return empty_df, pd.DataFrame(), pd.DataFrame(), {
            "phase": phase,
            "input_file": str(input_file),
            "n_input_rows": 0,
            "n_clean_rows": 0,
            "n_model_usable_rows": 0,
            "n_rejected_rows": 0,
        }

    if "phase" in df_in.columns:
        df_in = df_in[df_in["phase"] == phase].copy()

    clean_rows: list[dict[str, Any]] = []
    immediate_rejected: list[dict[str, Any]] = []

    for _, row in tqdm(df_in.iterrows(), total=len(df_in), desc=f"clean {phase}", unit="cycle"):
        cleaned, rejected = clean_single_cycle(
            row,
            phase=phase,
            voltage_min=voltage_min,
            voltage_max=voltage_max,
            round_vectors=round_vectors,
        )
        if cleaned is not None:
            clean_rows.append(cleaned)
        if rejected is not None:
            immediate_rejected.append(rejected)

    clean_df = reorder_standard_columns(pd.DataFrame(clean_rows)) if clean_rows else empty_standard_dataframe()
    clean_df, rejected_df, eol_df = assign_targets_and_flags(clean_df, phase)

    if immediate_rejected:
        rejected_df = pd.concat([pd.DataFrame(immediate_rejected), rejected_df], ignore_index=True, sort=False)

    reason_counts = rejected_df["exclude_reason"].value_counts(dropna=False).to_dict() if not rejected_df.empty else {}
    report = {
        "phase": phase,
        "input_file": str(input_file),
        "n_input_rows": int(len(df_in)),
        "n_clean_rows": int(len(clean_df)),
        "n_valid_rows": int(clean_df["is_valid_cycle"].sum()) if not clean_df.empty else 0,
        "n_model_usable_rows": int(clean_df["is_model_usable"].sum()) if not clean_df.empty else 0,
        "n_rejected_rows": int(len(rejected_df)),
        "rejection_reason_counts": reason_counts,
    }
    return clean_df, rejected_df, eol_df, report


def parse_args() -> argparse.Namespace:
    default_base_dir = infer_default_base_dir()
    parser = argparse.ArgumentParser(description="Clean Oxford/ORA extracted charge and discharge cycles.")
    parser.add_argument("--base-dir", type=str, default=str(default_base_dir), help="Project root directory.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(default_base_dir / "data" / "interim" / DATASET_FOLDER / "extracted"),
        help="Directory containing extracted ORA parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(default_base_dir / "data" / "interim" / DATASET_FOLDER / "clean"),
        help="Directory where clean ORA parquet files will be saved.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=str(default_base_dir / "data" / "reports" / DATASET_FOLDER),
        help="Directory where cleaning reports will be saved.",
    )
    parser.add_argument("--voltage-min", type=float, default=2.70, help="Minimum plausible voltage retained during cleaning.")
    parser.add_argument("--voltage-max", type=float, default=4.25, help="Maximum plausible voltage retained during cleaning.")
    parser.add_argument("--round-vectors", type=int, default=None, help="Optional decimal places for vector rounding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    ensure_dir(out_dir)
    ensure_dir(report_dir)

    phase_files = {
        "charge": input_dir / "ora_charge_extracted_cycles.parquet",
        "discharge": input_dir / "ora_discharge_extracted_cycles.parquet",
    }

    all_reports: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "cleaning_version": CLEANING_VERSION,
        "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "eol_capacity_ah": EOL_CAPACITY_AH,
        "eol_target_policy": "discharge_capacity_threshold_applied_to_charge_and_discharge_branches",
        "voltage_window": [args.voltage_min, args.voltage_max],
        "phases": {},
    }

    phase_clean_frames: dict[str, pd.DataFrame] = {}
    immediate_rejections: dict[str, pd.DataFrame] = {}

    for phase, input_file in phase_files.items():
        clean_df, rejected_df, _eol_df, _report = clean_phase_file(
            input_file=input_file,
            phase=phase,
            voltage_min=args.voltage_min,
            voltage_max=args.voltage_max,
            round_vectors=args.round_vectors,
        )
        phase_clean_frames[phase] = clean_df
        # Keep only structural rejections from the first pass. Target-based
        # rejections are recomputed after the common discharge EOL is assigned.
        if not rejected_df.empty and "is_valid_cycle" in rejected_df.columns:
            immediate_rejections[phase] = rejected_df[rejected_df["is_valid_cycle"] != True].copy()
        else:
            immediate_rejections[phase] = rejected_df.copy() if not rejected_df.empty else pd.DataFrame()

    phase_clean_frames, common_eol_df = apply_discharge_eol_targets(phase_clean_frames)

    for phase, clean_df in phase_clean_frames.items():
        input_file = phase_files[phase]
        validate_standard_schema(clean_df)

        rejected_df = build_rejected_cycles_dataframe(clean_df)
        if phase in immediate_rejections and not immediate_rejections[phase].empty:
            rejected_df = pd.concat([immediate_rejections[phase], rejected_df], ignore_index=True, sort=False)

        report = build_phase_report(input_file, phase, clean_df, rejected_df)

        clean_file = out_dir / f"ora_{phase}_clean_cycles.parquet"
        rejected_file = out_dir / f"ora_{phase}_rejected_cycles.csv"
        clean_df.to_parquet(clean_file, index=False)
        rejected_df.to_csv(rejected_file, index=False, encoding="utf-8-sig")

        all_reports["phases"][phase] = report | {
            "clean_file": str(clean_file),
            "rejected_file": str(rejected_file),
        }

        print(f"\nCleaned {phase} rows: {len(clean_df)} | model-usable: {int(clean_df['is_model_usable'].sum()) if not clean_df.empty else 0}")
        print(f"Clean file: {clean_file}")
        print(f"Rejected file: {rejected_file}")

    if not common_eol_df.empty:
        common_eol_df.to_csv(report_dir / "cleaning_eol_summary.csv", index=False, encoding="utf-8-sig")

    write_json(report_dir / "cleaning_report.json", all_reports)
    print(f"\nCleaning report: {report_dir / 'cleaning_report.json'}")
    print("Oxford/ORA cleaning completed.")


if __name__ == "__main__":
    main()
