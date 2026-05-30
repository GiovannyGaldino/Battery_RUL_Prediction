#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean and standardize extracted HUST charge/discharge cycles.

Input files:
    data/interim/hust/extracted/hust_charge_extracted_cycles.parquet
    data/interim/hust/extracted/hust_discharge_extracted_cycles.parquet

Output files:
    data/interim/hust/clean/hust_charge_clean_cycles.parquet
    data/interim/hust/clean/hust_discharge_clean_cycles.parquet
    data/interim/hust/clean/hust_charge_rejected_cycles.csv
    data/interim/hust/clean/hust_discharge_rejected_cycles.csv
    data/reports/hust/cleaning_report.json

Example:
    python src/adapters/hust/clean_hust_cycles.py \
        --base-dir "D:/USUÁRIOS/Marcellus/OneDrive/Doutorado/BATERIAS - Ressubmissão/Batteries" \
        --temperature-c 30
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .hust_schema import (
        ADAPTER_VERSION,
        CLEANING_VERSION,
        DATASET_NAME,
        DEFAULT_BASE_DIR,
        DEFAULT_TEMPERATURE_C,
        EXTRACTED_COLUMNS,
        MAX_CELL_VOLTAGE_V,
        MIN_CAPACITY_AH,
        MIN_CELL_VOLTAGE_V,
        MIN_DURATION_S,
        MIN_POINTS,
        MIN_VOLTAGE_SPAN_V,
        STANDARD_CYCLE_COLUMNS,
        dump_json,
        empty_clean_dataframe,
        ensure_columns,
        get_tqdm,
        read_parquet_if_exists,
        resolve_base_paths,
        safe_float,
        safe_int,
        safe_numeric_array,
        utc_now_iso,
        write_parquet,
    )
except ImportError:
    from hust_schema import (  # type: ignore
        ADAPTER_VERSION,
        CLEANING_VERSION,
        DATASET_NAME,
        DEFAULT_BASE_DIR,
        DEFAULT_TEMPERATURE_C,
        EXTRACTED_COLUMNS,
        MAX_CELL_VOLTAGE_V,
        MIN_CAPACITY_AH,
        MIN_CELL_VOLTAGE_V,
        MIN_DURATION_S,
        MIN_POINTS,
        MIN_VOLTAGE_SPAN_V,
        STANDARD_CYCLE_COLUMNS,
        dump_json,
        empty_clean_dataframe,
        ensure_columns,
        get_tqdm,
        read_parquet_if_exists,
        resolve_base_paths,
        safe_float,
        safe_int,
        safe_numeric_array,
        utc_now_iso,
        write_parquet,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean extracted HUST charge/discharge cycles into the standard schema.")
    parser.add_argument("--base-dir", type=str, default=str(DEFAULT_BASE_DIR), help="Project root directory.")
    parser.add_argument("--interim-dir", type=str, default=None, help="Dataset interim directory. Default: <base>/data/interim/hust")
    parser.add_argument("--reports-dir", type=str, default=None, help="Dataset reports directory. Default: <base>/data/reports/hust")
    parser.add_argument("--temperature-c", type=float, default=DEFAULT_TEMPERATURE_C, help="Fallback nominal temperature when extracted temperature is missing.")
    parser.add_argument("--min-points", type=int, default=MIN_POINTS, help="Minimum number of valid samples per accepted cycle.")
    parser.add_argument("--min-voltage-span-v", type=float, default=MIN_VOLTAGE_SPAN_V, help="Minimum voltage span per valid phase.")
    return parser.parse_args()


def cumulative_trapezoid_abs_current(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    if len(time_s) < 2 or len(current_a) < 2:
        return np.array([], dtype=float)
    n = min(len(time_s), len(current_a))
    t = np.asarray(time_s[:n], dtype=float)
    i_abs = np.abs(np.asarray(current_a[:n], dtype=float))
    dt = np.diff(t)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)
    i_mid = 0.5 * (i_abs[1:] + i_abs[:-1])
    dq = i_mid * dt / 3600.0
    return np.concatenate([[0.0], np.cumsum(dq)])


def cumulative_trapezoid_power(time_s: np.ndarray, power_w: np.ndarray) -> np.ndarray:
    if len(time_s) < 2 or len(power_w) < 2:
        return np.array([], dtype=float)
    n = min(len(time_s), len(power_w))
    t = np.asarray(time_s[:n], dtype=float)
    p_abs = np.abs(np.asarray(power_w[:n], dtype=float))
    dt = np.diff(t)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)
    p_mid = 0.5 * (p_abs[1:] + p_abs[:-1])
    de = p_mid * dt / 3600.0
    return np.concatenate([[0.0], np.cumsum(de)])


def normalize_capacity(capacity_ah: np.ndarray) -> np.ndarray:
    cap = np.asarray(capacity_ah, dtype=float).copy()
    if cap.size == 0:
        return cap
    if np.isfinite(cap).any():
        first_idx = np.where(np.isfinite(cap))[0]
        if len(first_idx):
            cap = cap - cap[first_idx[0]]
    diffs = np.diff(cap)
    finite_diffs = diffs[np.isfinite(diffs)]
    if finite_diffs.size and np.nanmedian(finite_diffs) < 0:
        cap = -cap
    if np.isfinite(cap).any():
        min_cap = np.nanmin(cap)
        if np.isfinite(min_cap) and min_cap < 0:
            cap = cap - min_cap
    return cap


def align_and_clean_vectors(row: pd.Series, phase: str, fallback_temperature_c: float) -> tuple[dict[str, Any] | None, list[str], int, int, int, str]:
    warnings: list[str] = []
    time_s = safe_numeric_array(row.get("time_s"))
    voltage_v = safe_numeric_array(row.get("voltage_v"))
    current_a = safe_numeric_array(row.get("current_a"))
    capacity_ah = safe_numeric_array(row.get("capacity_ah"))
    temperature_c = safe_numeric_array(row.get("temperature_c"))

    lengths = [len(time_s), len(voltage_v), len(current_a), len(capacity_ah)]
    if len(temperature_c) > 0:
        lengths.append(len(temperature_c))
    n0 = min(lengths) if lengths else 0
    if n0 <= 0:
        return None, warnings, 0, 0, 0, "empty_vectors"

    time_s = time_s[:n0]
    voltage_v = voltage_v[:n0]
    current_a = current_a[:n0]
    capacity_ah = capacity_ah[:n0]
    if len(temperature_c) >= n0:
        temperature_c = temperature_c[:n0]
        temperature_source = str(row.get("temperature_source", "measured"))
    else:
        temperature_c = np.full(n0, float(fallback_temperature_c), dtype=float)
        temperature_source = "nominal_constant"
        warnings.append("temperature_replaced_by_nominal_constant")

    original_n = n0
    mask = np.isfinite(time_s) & np.isfinite(voltage_v)

    if np.isfinite(current_a).any():
        mask &= np.isfinite(current_a)
    else:
        warnings.append("current_missing_or_all_nan")

    if np.isfinite(capacity_ah).any():
        mask &= np.isfinite(capacity_ah)
    else:
        warnings.append("capacity_missing_or_all_nan")

    if np.isfinite(temperature_c).any():
        mask &= np.isfinite(temperature_c)

    if not mask.any():
        return None, warnings, original_n, 0, 0, "no_finite_signal_overlap"

    kept_indices = np.where(mask)[0]
    trim_start_index = int(kept_indices[0])
    trim_end_index = int(kept_indices[-1])

    time_s = time_s[mask]
    voltage_v = voltage_v[mask]
    current_a = current_a[mask]
    capacity_ah = capacity_ah[mask]
    temperature_c = temperature_c[mask]

    order = np.argsort(time_s, kind="mergesort")
    time_s = time_s[order]
    voltage_v = voltage_v[order]
    current_a = current_a[order]
    capacity_ah = capacity_ah[order]
    temperature_c = temperature_c[order]

    _, unique_indices = np.unique(time_s, return_index=True)
    unique_indices = np.sort(unique_indices)
    if len(unique_indices) < len(time_s):
        warnings.append("duplicated_time_samples_removed")

    time_s = time_s[unique_indices]
    voltage_v = voltage_v[unique_indices]
    current_a = current_a[unique_indices]
    capacity_ah = capacity_ah[unique_indices]
    temperature_c = temperature_c[unique_indices]

    if len(time_s) < 2:
        return None, warnings, original_n, original_n - len(time_s), trim_start_index, "too_few_time_samples"

    time_s = time_s - time_s[0]

    # Enforce the standard current sign convention.
    if phase == "charge":
        current_a = np.abs(current_a)
    elif phase == "discharge":
        current_a = -np.abs(current_a)
    else:
        return None, warnings, original_n, original_n - len(time_s), trim_start_index, "invalid_phase"

    capacity_source = str(row.get("capacity_source", "measured"))
    capacity_ah = normalize_capacity(capacity_ah)
    cap_span = np.nanmax(capacity_ah) - np.nanmin(capacity_ah) if capacity_ah.size else np.nan
    if (not np.isfinite(cap_span)) or cap_span < MIN_CAPACITY_AH:
        integrated_capacity = cumulative_trapezoid_abs_current(time_s, current_a)
        if integrated_capacity.size == len(time_s) and np.nanmax(integrated_capacity) > MIN_CAPACITY_AH:
            capacity_ah = integrated_capacity
            capacity_source = "integrated_current"
            warnings.append("capacity_reconstructed_from_current")
        else:
            return None, warnings, original_n, original_n - len(time_s), trim_start_index, "missing_current_and_capacity"

    power_w = voltage_v * current_a
    energy_wh = cumulative_trapezoid_power(time_s, power_w)
    if energy_wh.size != len(time_s):
        energy_wh = np.full(len(time_s), np.nan, dtype=float)
        energy_source = "missing"
    else:
        energy_source = "integrated_power"

    n_removed = int(original_n - len(time_s))
    return {
        "time_s": time_s,
        "voltage_v": voltage_v,
        "current_a": current_a,
        "temperature_c": temperature_c,
        "capacity_ah": capacity_ah,
        "power_w": power_w,
        "energy_wh": energy_wh,
        "capacity_source": capacity_source,
        "energy_source": energy_source,
        "temperature_source": temperature_source,
    }, warnings, original_n, n_removed, trim_start_index, "accepted"


def can_interpolate_qv_curve(voltage_v: np.ndarray, capacity_ah: np.ndarray) -> bool:
    if len(voltage_v) < MIN_POINTS or len(capacity_ah) < MIN_POINTS:
        return False
    if not np.isfinite(voltage_v).any() or not np.isfinite(capacity_ah).any():
        return False
    return float(np.nanmax(voltage_v) - np.nanmin(voltage_v)) >= MIN_VOLTAGE_SPAN_V


def classify_validity(vectors: dict[str, Any], phase: str, args: argparse.Namespace) -> tuple[bool, str]:
    time_s = vectors["time_s"]
    voltage_v = vectors["voltage_v"]
    current_a = vectors["current_a"]
    capacity_ah = vectors["capacity_ah"]

    if len(time_s) < args.min_points:
        return False, "few_points"
    if not np.all(np.diff(time_s) > 0):
        return False, "time_not_strictly_increasing"
    duration_s = float(time_s[-1] - time_s[0])
    if not np.isfinite(duration_s) or duration_s < MIN_DURATION_S:
        return False, "duration_too_short"
    if np.nanmin(voltage_v) < MIN_CELL_VOLTAGE_V or np.nanmax(voltage_v) > MAX_CELL_VOLTAGE_V:
        return False, "voltage_out_of_sanity_range"
    if float(np.nanmax(voltage_v) - np.nanmin(voltage_v)) < args.min_voltage_span_v:
        return False, "insufficient_voltage_span"
    if not np.isfinite(capacity_ah).any() or float(np.nanmax(capacity_ah)) <= MIN_CAPACITY_AH:
        return False, "non_positive_capacity"
    if phase == "charge" and np.nanmean(current_a) <= 0:
        return False, "invalid_charge_current_sign"
    if phase == "discharge" and np.nanmean(current_a) >= 0:
        return False, "invalid_discharge_current_sign"
    return True, ""


def vector_stats(vectors: dict[str, Any]) -> dict[str, float | int]:
    t = vectors["time_s"]
    v = vectors["voltage_v"]
    i = vectors["current_a"]
    temp = vectors["temperature_c"]
    cap = vectors["capacity_ah"]
    e = vectors["energy_wh"]
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return {
        "n_points": int(len(t)),
        "duration_s": float(t[-1] - t[0]) if len(t) else np.nan,
        "sample_period_median_s": float(np.nanmedian(dt)) if len(dt) else np.nan,
        "voltage_min_v": float(np.nanmin(v)) if len(v) else np.nan,
        "voltage_max_v": float(np.nanmax(v)) if len(v) else np.nan,
        "voltage_start_v": float(v[0]) if len(v) else np.nan,
        "voltage_end_v": float(v[-1]) if len(v) else np.nan,
        "current_min_a": float(np.nanmin(i)) if len(i) else np.nan,
        "current_max_a": float(np.nanmax(i)) if len(i) else np.nan,
        "current_mean_a": float(np.nanmean(i)) if len(i) else np.nan,
        "temperature_min_c": float(np.nanmin(temp)) if len(temp) and np.isfinite(temp).any() else np.nan,
        "temperature_max_c": float(np.nanmax(temp)) if len(temp) and np.isfinite(temp).any() else np.nan,
        "temperature_mean_c": float(np.nanmean(temp)) if len(temp) and np.isfinite(temp).any() else np.nan,
        "phase_capacity_ah": float(np.nanmax(cap)) if len(cap) else np.nan,
        "phase_energy_wh": float(np.nanmax(e)) if len(e) and np.isfinite(e).any() else np.nan,
    }


def build_clean_row(
    row: pd.Series,
    phase: str,
    vectors: dict[str, Any],
    is_valid_cycle: bool,
    exclude_reason: str,
    warnings: list[str],
    n_removed_points: int,
    trim_start_index: int,
    trim_end_index: int,
) -> dict[str, Any]:
    stats = vector_stats(vectors)
    cycle_life = safe_float(row.get("cycle_life_raw"))
    cycle_number = safe_float(row.get("cycle_number"))
    remaining_raw = safe_float(row.get("remaining_cycles_raw"))
    if not np.isfinite(cycle_life) and np.isfinite(cycle_number) and np.isfinite(remaining_raw):
        cycle_life = cycle_number + remaining_raw

    record = {
        "dataset": DATASET_NAME,
        "source_file": str(row.get("source_file", "")),
        "source_path": str(row.get("source_path", "")),
        "battery_id": str(row.get("battery_id", "")),
        "battery_index": safe_int(row.get("battery_index"), default=-1),
        "cell_id_original": str(row.get("cell_id_original", "")),
        "cycle_index": -1,  # assigned after target calculation
        "cycle_number": cycle_number,
        "source_cycle_id": str(row.get("source_cycle_id", "")),
        "cycle_pair_id": str(row.get("cycle_pair_id", "")),
        "phase": phase,
        "cycle_life": cycle_life,
        "remaining_cycles": np.nan,
        "remaining_cycles_norm": np.nan,
        "eol_criterion": "80pct_initial_capacity_or_provided_by_dataset",
        "eol_capacity_ah": np.nan,
        "initial_capacity_ah": np.nan,
        "nominal_capacity_ah": np.nan,
        "soh": np.nan,
        "time_s": vectors["time_s"].astype(float).tolist(),
        "voltage_v": vectors["voltage_v"].astype(float).tolist(),
        "current_a": vectors["current_a"].astype(float).tolist(),
        "temperature_c": vectors["temperature_c"].astype(float).tolist(),
        "capacity_ah": vectors["capacity_ah"].astype(float).tolist(),
        "energy_wh": vectors["energy_wh"].astype(float).tolist(),
        "power_w": vectors["power_w"].astype(float).tolist(),
        "voltage_source": "measured",
        "current_source": "measured",
        "temperature_source": vectors["temperature_source"],
        "capacity_source": vectors["capacity_source"],
        "energy_source": vectors["energy_source"],
        "chemistry": "unknown",
        "form_factor": "unknown",
        "manufacturer": "unknown",
        "nominal_voltage_v": np.nan,
        "ambient_temperature_c": np.nan,
        "charge_protocol": "HUST protocol; extracted from Status == charge when available",
        "discharge_protocol": "HUST protocol; extracted from Status == discharge when available",
        "c_rate_charge": np.nan,
        "c_rate_discharge": np.nan,
        "dod": np.nan,
        "policy_readable": "HUST aging dataset; charge/discharge phases extracted from Status labels",
        "barcode": str(row.get("cell_id_original", "")),
        "is_valid_cycle": bool(is_valid_cycle),
        "is_model_usable": False,
        "exclude_reason": exclude_reason,
        "cleaning_warnings": warnings,
        "n_removed_points": int(n_removed_points),
        "trim_start_index": int(trim_start_index),
        "trim_end_index": int(trim_end_index),
        "has_voltage": True,
        "has_current": True,
        "has_temperature": vectors["temperature_source"] == "measured",
        "has_capacity": True,
        "has_energy": vectors["energy_source"] != "missing",
        "adapter_version": ADAPTER_VERSION,
        "cleaning_version": CLEANING_VERSION,
        "created_at": utc_now_iso(),
    }
    record.update(stats)
    return record


def assign_lifetime_targets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return ensure_columns(df, STANDARD_CYCLE_COLUMNS)

    df = df.sort_values(["battery_id", "phase", "cycle_number"]).reset_index(drop=True)
    df["cycle_index"] = df.groupby(["battery_id", "phase"]).cumcount() + 1

    for (battery_id, phase), idx in df.groupby(["battery_id", "phase"]).groups.items():
        idx_list = list(idx)
        valid_mask = df.loc[idx_list, "is_valid_cycle"].astype(bool)
        valid_idx = [i for i, ok in zip(idx_list, valid_mask) if bool(ok)]
        if not valid_idx:
            continue

        capacities = pd.to_numeric(df.loc[valid_idx, "phase_capacity_ah"], errors="coerce")
        finite_capacity = capacities[np.isfinite(capacities)]
        initial_capacity = float(finite_capacity.iloc[0]) if not finite_capacity.empty else np.nan
        eol_capacity = 0.8 * initial_capacity if np.isfinite(initial_capacity) else np.nan

        # Prefer cycle_life supplied through HUST RUL metadata. If absent, infer
        # from the first capacity crossing below 80% of the initial phase capacity.
        provided_life = pd.to_numeric(df.loc[valid_idx, "cycle_life"], errors="coerce")
        provided_life = provided_life[np.isfinite(provided_life)]
        if not provided_life.empty:
            cycle_life = float(provided_life.max())
            eol_criterion = "provided_by_dataset"
        elif np.isfinite(eol_capacity):
            crossed = capacities[capacities <= eol_capacity]
            if not crossed.empty:
                first_cross_idx = crossed.index[0]
                cycle_life = float(df.loc[first_cross_idx, "cycle_number"])
                eol_criterion = "80pct_initial_capacity"
            else:
                cycle_life = np.nan
                eol_criterion = "80pct_initial_capacity_not_reached"
        else:
            cycle_life = np.nan
            eol_criterion = "unknown"

        for i in idx_list:
            phase_capacity = safe_float(df.at[i, "phase_capacity_ah"])
            cycle_number = safe_float(df.at[i, "cycle_number"])
            df.at[i, "initial_capacity_ah"] = initial_capacity
            df.at[i, "eol_capacity_ah"] = eol_capacity
            df.at[i, "eol_criterion"] = eol_criterion
            if np.isfinite(initial_capacity) and initial_capacity > 0 and np.isfinite(phase_capacity):
                df.at[i, "soh"] = phase_capacity / initial_capacity
            if np.isfinite(cycle_life):
                df.at[i, "cycle_life"] = cycle_life
            if np.isfinite(cycle_life) and np.isfinite(cycle_number):
                remaining = cycle_life - cycle_number
                df.at[i, "remaining_cycles"] = remaining
                df.at[i, "remaining_cycles_norm"] = remaining / cycle_life if cycle_life > 0 else np.nan

    # Model usability is computed after target assignment.
    for i, row in df.iterrows():
        is_valid = bool(row.get("is_valid_cycle", False))
        remaining = safe_float(row.get("remaining_cycles"))
        cycle_life = safe_float(row.get("cycle_life"))
        voltage_v = safe_numeric_array(row.get("voltage_v"))
        capacity_ah = safe_numeric_array(row.get("capacity_ah"))
        model_usable = (
            is_valid
            and np.isfinite(cycle_life)
            and np.isfinite(remaining)
            and remaining > 0
            and can_interpolate_qv_curve(voltage_v, capacity_ah)
        )
        df.at[i, "is_model_usable"] = bool(model_usable)
        if not model_usable and is_valid and not str(row.get("exclude_reason", "")):
            if not np.isfinite(cycle_life):
                df.at[i, "exclude_reason"] = "missing_cycle_life"
            elif not np.isfinite(remaining) or remaining <= 0:
                df.at[i, "exclude_reason"] = "non_positive_remaining_cycles"
            elif not can_interpolate_qv_curve(voltage_v, capacity_ah):
                df.at[i, "exclude_reason"] = "cannot_interpolate_qv_curve"
            else:
                df.at[i, "exclude_reason"] = "not_model_usable"
    return ensure_columns(df, STANDARD_CYCLE_COLUMNS)


def clean_phase(df_extracted: pd.DataFrame, phase: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons = Counter()
    warnings_counter = Counter()

    tqdm = get_tqdm()
    iterator = tqdm(df_extracted.iterrows(), total=len(df_extracted), desc=f"clean:{phase}", unit="cycle")
    for _, row in iterator:
        vectors, warnings, original_n, n_removed, trim_start, status = align_and_clean_vectors(row, phase, args.temperature_c)
        if vectors is None:
            reason = status
            reasons[reason] += 1
            rejected.append({
                "dataset": DATASET_NAME,
                "battery_id": row.get("battery_id", ""),
                "source_file": row.get("source_file", ""),
                "source_cycle_id": row.get("source_cycle_id", ""),
                "cycle_number": row.get("cycle_number", np.nan),
                "phase": phase,
                "exclude_reason": reason,
                "n_points_raw": original_n,
                "cleaning_warnings": ";".join(warnings),
            })
            continue

        is_valid, reason = classify_validity(vectors, phase, args)
        if not is_valid:
            reasons[reason] += 1
        for warning in warnings:
            warnings_counter[warning] += 1

        trim_end = trim_start + int(len(vectors["time_s"])) - 1 if len(vectors["time_s"]) else trim_start
        record = build_clean_row(
            row=row,
            phase=phase,
            vectors=vectors,
            is_valid_cycle=is_valid,
            exclude_reason="" if is_valid else reason,
            warnings=warnings,
            n_removed_points=n_removed,
            trim_start_index=trim_start,
            trim_end_index=trim_end,
        )
        records.append(record)

        if not is_valid:
            rejected.append({
                "dataset": DATASET_NAME,
                "battery_id": record["battery_id"],
                "source_file": record["source_file"],
                "source_cycle_id": record["source_cycle_id"],
                "cycle_number": record["cycle_number"],
                "phase": phase,
                "exclude_reason": reason,
                "n_points_raw": original_n,
                "n_points_clean": record["n_points"],
                "cleaning_warnings": ";".join(warnings),
            })

    df_clean = assign_lifetime_targets(pd.DataFrame(records)) if records else empty_clean_dataframe()

    # Add model-usability rejections to the rejected table without duplicating
    # physically invalid cycles already captured above.
    if not df_clean.empty:
        not_usable = df_clean[(df_clean["is_valid_cycle"] == True) & (df_clean["is_model_usable"] == False)]
        for _, row in not_usable.iterrows():
            reason = str(row.get("exclude_reason", "not_model_usable")) or "not_model_usable"
            reasons[reason] += 1
            rejected.append({
                "dataset": DATASET_NAME,
                "battery_id": row.get("battery_id", ""),
                "source_file": row.get("source_file", ""),
                "source_cycle_id": row.get("source_cycle_id", ""),
                "cycle_number": row.get("cycle_number", np.nan),
                "phase": phase,
                "exclude_reason": reason,
                "n_points_clean": row.get("n_points", np.nan),
                "cleaning_warnings": ";".join(row.get("cleaning_warnings", []) if isinstance(row.get("cleaning_warnings", []), list) else []),
            })

    df_rejected = pd.DataFrame(rejected)
    report = {
        "phase": phase,
        "input_rows": int(len(df_extracted)),
        "clean_rows": int(len(df_clean)),
        "valid_cycles": int(df_clean["is_valid_cycle"].sum()) if not df_clean.empty else 0,
        "model_usable_cycles": int(df_clean["is_model_usable"].sum()) if not df_clean.empty else 0,
        "rejected_rows": int(len(df_rejected)),
        "exclude_reasons": dict(reasons),
        "cleaning_warnings": dict(warnings_counter),
    }
    return df_clean, df_rejected, report


def main() -> None:
    args = parse_args()
    paths = resolve_base_paths(args)
    extracted_dir = paths["extracted_dir"]
    clean_dir = paths["clean_dir"]
    reports_dir = paths["reports_dir"]
    clean_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    phase_reports: dict[str, Any] = {}
    for phase in ("charge", "discharge"):
        input_file = extracted_dir / f"hust_{phase}_extracted_cycles.parquet"
        df_extracted = read_parquet_if_exists(input_file, EXTRACTED_COLUMNS)
        if df_extracted.empty:
            df_clean = empty_clean_dataframe()
            df_rejected = pd.DataFrame(columns=["dataset", "battery_id", "source_file", "source_cycle_id", "cycle_number", "phase", "exclude_reason"])
            report = {
                "phase": phase,
                "input_file": str(input_file),
                "input_rows": 0,
                "clean_rows": 0,
                "valid_cycles": 0,
                "model_usable_cycles": 0,
                "rejected_rows": 0,
                "exclude_reasons": {"empty_extracted_file": 0},
            }
        else:
            df_clean, df_rejected, report = clean_phase(df_extracted, phase, args)
            report["input_file"] = str(input_file)

        clean_output = clean_dir / f"hust_{phase}_clean_cycles.parquet"
        rejected_output = clean_dir / f"hust_{phase}_rejected_cycles.csv"
        write_parquet(ensure_columns(df_clean, STANDARD_CYCLE_COLUMNS), clean_output)
        df_rejected.to_csv(rejected_output, index=False, encoding="utf-8")
        report["clean_output_file"] = str(clean_output)
        report["rejected_output_file"] = str(rejected_output)
        phase_reports[phase] = report

    full_report = {
        "dataset": DATASET_NAME,
        "adapter_version": ADAPTER_VERSION,
        "cleaning_version": CLEANING_VERSION,
        "phases": phase_reports,
        "created_at": utc_now_iso(),
    }
    dump_json(full_report, reports_dir / "cleaning_report.json")

    print("\n[OK] HUST cleaning completed.")
    for phase, report in phase_reports.items():
        print(
            f"[OK] {phase}: clean_rows={report['clean_rows']}, "
            f"valid={report['valid_cycles']}, usable={report['model_usable_cycles']}"
        )
    print(f"[OK] Report: {reports_dir / 'cleaning_report.json'}")


if __name__ == "__main__":
    main()
