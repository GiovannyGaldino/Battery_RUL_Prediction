# -*- coding: utf-8 -*-
"""
RWTH Aachen / ISEA battery dataset adapter - cleaning stage.

This script reads the phase-specific extracted files produced by
extract_rwth_cycles.py and writes clean charge/discharge parquet files using the
standard project schema.

Run
---
python src/adapters/rwth/clean_rwth_cycles.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DATASET = "RWTH"
CLEANING_VERSION = "rwth_clean_v1.4"

DEFAULT_BASE_DIR = Path(
    r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries"
)
DEFAULT_EXTRACTED_DIR = DEFAULT_BASE_DIR / "data" / "interim" / "rwth" / "extracted"
DEFAULT_CLEAN_DIR = DEFAULT_BASE_DIR / "data" / "interim" / "rwth" / "clean"
DEFAULT_REPORT_DIR = DEFAULT_BASE_DIR / "data" / "reports" / "rwth" / "cleaning"

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

VECTOR_COLUMNS = ["time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah", "energy_wh", "power_w"]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return np.array([], dtype=float)
        try:
            arr = np.asarray(ast.literal_eval(text))
        except Exception:
            text = text.replace(";", ",").replace("[", "").replace("]", "")
            parts = [part.strip() for part in text.split(",") if part.strip()]
            values = []
            for part in parts:
                try:
                    values.append(float(part))
                except ValueError:
                    continue
            arr = np.asarray(values)
    else:
        try:
            arr = np.asarray(value)
        except Exception:
            return np.array([], dtype=float)
    try:
        return np.atleast_1d(np.asarray(arr, dtype=float).squeeze())
    except Exception:
        return np.array([], dtype=float)


def robust_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def integrate_abs_current_to_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    n = min(len(time_s), len(current_a))
    if n < 2:
        return np.array([], dtype=float)
    time = np.asarray(time_s[:n], dtype=float)
    current = np.asarray(current_a[:n], dtype=float)
    dt = np.diff(time)
    current_mid = 0.5 * (np.abs(current[1:]) + np.abs(current[:-1]))
    dq = current_mid * dt / 3600.0
    dq = np.where(np.isfinite(dq) & (dq >= 0), dq, 0.0)
    return np.concatenate([[0.0], np.cumsum(dq)])


def integrate_abs_power_to_wh(time_s: np.ndarray, power_w: np.ndarray) -> np.ndarray:
    n = min(len(time_s), len(power_w))
    if n < 2:
        return np.array([], dtype=float)
    time = np.asarray(time_s[:n], dtype=float)
    power = np.asarray(power_w[:n], dtype=float)
    dt = np.diff(time)
    power_mid = 0.5 * (np.abs(power[1:]) + np.abs(power[:-1]))
    de = power_mid * dt / 3600.0
    de = np.where(np.isfinite(de) & (de >= 0), de, 0.0)
    return np.concatenate([[0.0], np.cumsum(de)])


def make_nan_vector(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=float)


def align_and_sort_vectors(row: pd.Series) -> tuple[dict[str, np.ndarray], list[str]]:
    warnings: list[str] = []
    vectors = {
        "time_s": safe_array(row.get("time_s")),
        "voltage_v": safe_array(row.get("voltage_v")),
        "current_a": safe_array(row.get("current_a")),
        "temperature_c": safe_array(row.get("temperature_c")),
        "capacity_ah": safe_array(row.get("capacity_ah")),
    }
    lengths = [len(arr) for arr in vectors.values() if len(arr) > 0]
    if not lengths:
        return {key: np.array([], dtype=float) for key in vectors}, ["empty_vectors"]
    n = min(lengths)
    if len(set(lengths)) > 1:
        warnings.append("vector_lengths_trimmed_to_common_minimum")

    for key in vectors:
        arr = vectors[key]
        if len(arr) == 0:
            vectors[key] = make_nan_vector(n)
        else:
            vectors[key] = arr[:n].astype(float)

    mask = np.isfinite(vectors["time_s"]) & np.isfinite(vectors["voltage_v"])
    if np.isfinite(vectors["current_a"]).any():
        mask &= np.isfinite(vectors["current_a"])
    if np.isfinite(vectors["capacity_ah"]).any():
        mask &= np.isfinite(vectors["capacity_ah"])

    for key in vectors:
        vectors[key] = vectors[key][mask]

    if len(vectors["time_s"]) == 0:
        return vectors, warnings + ["no_valid_time_voltage_samples"]

    order = np.argsort(vectors["time_s"])
    for key in vectors:
        vectors[key] = vectors[key][order]

    # Keep the first sample of each duplicated time stamp.
    time = vectors["time_s"]
    unique_mask = np.ones(len(time), dtype=bool)
    unique_mask[1:] = np.diff(time) > 0
    if unique_mask.sum() < len(unique_mask):
        warnings.append("duplicated_or_non_increasing_time_samples_removed")
    for key in vectors:
        vectors[key] = vectors[key][unique_mask]

    if len(vectors["time_s"]):
        vectors["time_s"] = vectors["time_s"] - vectors["time_s"][0]

    return vectors, warnings


def trim_edges(vectors: dict[str, np.ndarray], phase: str) -> tuple[dict[str, np.ndarray], int, int, list[str]]:
    warnings: list[str] = []
    n = len(vectors["time_s"])
    if n == 0:
        return vectors, 0, -1, ["empty_after_alignment"]

    start_idx = 0
    end_idx = n - 1
    voltage = vectors["voltage_v"]

    # Discharge files sometimes include post-cutoff voltage rebound. Keep the segment up to the first minimum
    # if the rebound occurs near the end.
    if phase == "discharge" and np.isfinite(voltage).any() and n >= 10:
        min_idx = int(np.nanargmin(voltage))
        if min_idx >= int(0.6 * n) and min_idx < n - 1:
            rebound = float(np.nanmax(voltage[min_idx:]) - voltage[min_idx])
            if np.isfinite(rebound) and rebound >= 0.02:
                end_idx = min_idx
                warnings.append("post_cutoff_voltage_rebound_trimmed")

    # Charge segments may contain a short pre-charge voltage dip before voltage rises.
    if phase == "charge" and np.isfinite(voltage).any() and n >= 10:
        min_idx = int(np.nanargmin(voltage))
        if min_idx <= int(0.25 * n) and min_idx > 0:
            rise_after_min = float(np.nanmax(voltage[min_idx:]) - voltage[min_idx])
            if np.isfinite(rise_after_min) and rise_after_min >= 0.05:
                start_idx = min_idx
                warnings.append("pre_charge_voltage_dip_trimmed")

    if end_idx < start_idx:
        end_idx = start_idx
    trimmed = {key: value[start_idx : end_idx + 1] for key, value in vectors.items()}
    if len(trimmed["time_s"]):
        trimmed["time_s"] = trimmed["time_s"] - trimmed["time_s"][0]
    return trimmed, start_idx, end_idx, warnings


def standardize_current(current: np.ndarray, phase: str) -> tuple[np.ndarray, str, list[str]]:
    warnings: list[str] = []
    if current.size == 0 or not np.isfinite(current).any():
        return current, "missing", ["current_missing"]
    if phase == "charge":
        out = np.abs(current)
        if np.nanmean(current) < 0:
            warnings.append("charge_current_sign_corrected")
    elif phase == "discharge":
        out = -np.abs(current)
        if np.nanmean(current) > 0:
            warnings.append("discharge_current_sign_corrected")
    else:
        out = current
    return out, "measured", warnings


def standardize_capacity(
    capacity: np.ndarray,
    time_s: np.ndarray,
    current_a: np.ndarray,
    source: Any,
) -> tuple[np.ndarray, str, list[str]]:
    warnings: list[str] = []
    cap = np.asarray(capacity, dtype=float)
    has_capacity = cap.size > 0 and np.isfinite(cap).sum() >= max(2, int(0.5 * len(cap)))
    if has_capacity:
        cap = cap - np.nanmin(cap)
        cap = np.where(np.isfinite(cap), cap, np.nan)
        qmax = float(np.nanmax(cap)) if np.isfinite(cap).any() else np.nan
        if np.isfinite(qmax) and qmax > 0:
            cap = np.maximum.accumulate(np.nan_to_num(cap, nan=0.0))
            src = str(source) if isinstance(source, str) and source else "measured"
            if src.startswith("measured"):
                src = "measured"
            elif src == "integrated_current":
                src = "integrated_current"
            else:
                src = "measured"
            return cap, src, warnings
    if current_a.size > 0 and np.isfinite(current_a).any():
        reconstructed = integrate_abs_current_to_ah(time_s, current_a)
        if len(reconstructed) == len(time_s):
            warnings.append("capacity_reconstructed_from_current")
            return reconstructed, "integrated_current", warnings
    return make_nan_vector(len(time_s)), "missing", ["missing_current_and_capacity"]


def scalar_stats(array: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(array, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.nanmin(finite)), float(np.nanmax(finite)), float(np.nanmean(finite))


def can_interpolate_qv_curve(voltage_v: np.ndarray, capacity_ah: np.ndarray, min_voltage_span: float) -> bool:
    voltage = np.asarray(voltage_v, dtype=float)
    capacity = np.asarray(capacity_ah, dtype=float)
    mask = np.isfinite(voltage) & np.isfinite(capacity)
    if mask.sum() < 5:
        return False
    voltage = voltage[mask]
    if float(np.nanmax(voltage) - np.nanmin(voltage)) < min_voltage_span:
        return False
    return np.unique(np.round(voltage, 4)).size >= 4


def time_is_strictly_increasing(time_s: np.ndarray) -> bool:
    time = np.asarray(time_s, dtype=float)
    if len(time) < 2 or not np.isfinite(time).all():
        return False
    return bool(np.all(np.diff(time) > 0))


def clean_one_row(row: pd.Series, min_points: int, min_voltage_span: float) -> dict[str, Any]:
    phase = str(row.get("phase", "")).lower().strip()
    warnings: list[str] = []

    vectors, align_warnings = align_and_sort_vectors(row)
    warnings.extend(align_warnings)
    aligned_n = len(vectors["time_s"])

    vectors, trim_start, trim_end, trim_warnings = trim_edges(vectors, phase)
    warnings.extend(trim_warnings)
    n_removed = max(0, aligned_n - len(vectors["time_s"]))

    time_s = vectors["time_s"]
    voltage_v = vectors["voltage_v"]
    current_a, current_source, current_warnings = standardize_current(vectors["current_a"], phase)
    warnings.extend(current_warnings)
    temperature_c = vectors["temperature_c"]
    temperature_source = str(row.get("temperature_source", "unknown_nan") or "unknown_nan")
    if temperature_c.size == 0 or not np.isfinite(temperature_c).any():
        temperature_c = make_nan_vector(len(time_s))
        temperature_source = "unknown_nan"
    elif temperature_source not in {"measured", "nominal_constant", "literature_constant", "unknown_nan"}:
        temperature_source = "measured"

    capacity_ah, capacity_source, cap_warnings = standardize_capacity(
        vectors["capacity_ah"], time_s, current_a, row.get("capacity_source", "")
    )
    warnings.extend(cap_warnings)

    n = min(len(time_s), len(voltage_v), len(current_a), len(temperature_c), len(capacity_ah))
    time_s, voltage_v, current_a = time_s[:n], voltage_v[:n], current_a[:n]
    temperature_c, capacity_ah = temperature_c[:n], capacity_ah[:n]

    if n > 0 and len(time_s):
        time_s = time_s - time_s[0]

    power_w = voltage_v * current_a if n else np.array([], dtype=float)
    if n and np.isfinite(power_w).any() and np.isfinite(time_s).all():
        energy_wh = integrate_abs_power_to_wh(time_s, power_w)
        energy_source = "integrated_power"
    else:
        energy_wh = make_nan_vector(n)
        energy_source = "missing"

    voltage_min_v, voltage_max_v, _ = scalar_stats(voltage_v)
    current_min_a, current_max_a, current_mean_a = scalar_stats(current_a)
    temp_min_c, temp_max_c, temp_mean_c = scalar_stats(temperature_c)

    duration_s = float(time_s[-1] - time_s[0]) if n >= 2 else np.nan
    sample_period_median_s = float(np.nanmedian(np.diff(time_s))) if n >= 2 else np.nan
    phase_capacity_ah = float(capacity_ah[-1]) if n and np.isfinite(capacity_ah[-1]) else np.nan
    phase_energy_wh = float(energy_wh[-1]) if n and np.isfinite(energy_wh[-1]) else np.nan

    has_voltage = bool(n > 0 and np.isfinite(voltage_v).any())
    has_current = bool(current_source != "missing" and np.isfinite(current_a).any())
    has_temperature = bool(temperature_source == "measured" and np.isfinite(temperature_c).any())
    has_capacity = bool(capacity_source != "missing" and np.isfinite(capacity_ah).any())
    has_energy = bool(energy_source != "missing" and np.isfinite(energy_wh).any())

    validity_failures: list[str] = []
    if n < min_points:
        validity_failures.append("too_few_points")
    if not time_is_strictly_increasing(time_s):
        validity_failures.append("invalid_time_vector")
    if not has_voltage:
        validity_failures.append("missing_voltage")
    if not has_capacity:
        validity_failures.append("missing_current_and_capacity")
    if not np.isfinite(phase_capacity_ah) or phase_capacity_ah <= 0:
        validity_failures.append("non_positive_phase_capacity")
    if not np.isfinite(voltage_min_v) or not np.isfinite(voltage_max_v) or voltage_max_v <= voltage_min_v:
        validity_failures.append("invalid_voltage_range")
    if not np.isfinite(duration_s) or duration_s <= 0:
        validity_failures.append("non_positive_duration")
    if phase not in {"charge", "discharge"}:
        validity_failures.append("invalid_phase")

    is_valid_cycle = len(validity_failures) == 0
    interpolate_ok = can_interpolate_qv_curve(voltage_v, capacity_ah, min_voltage_span)

    if is_valid_cycle and not interpolate_ok:
        warnings.append("insufficient_voltage_range_for_qv_interpolation")

    out = {
        "dataset": DATASET,
        "source_file": str(row.get("source_file", "")),
        "source_path": str(row.get("source_path", "")),
        "battery_id": str(row.get("battery_id", "")),
        "battery_index": int(robust_float(row.get("battery_index", -1), -1)),
        "cell_id_original": str(row.get("cell_id_original", "unknown")),
        "cycle_index": -1,
        "cycle_number": np.nan,
        "source_cycle_id": str(row.get("source_cycle_id", "")),
        "cycle_pair_id": str(row.get("cycle_pair_id", "")),
        "phase": phase,
        "cycle_life": np.nan,
        "remaining_cycles": np.nan,
        "remaining_cycles_norm": np.nan,
        "eol_criterion": "unknown_not_reached",
        "eol_capacity_ah": np.nan,
        "initial_capacity_ah": np.nan,
        "nominal_capacity_ah": robust_float(row.get("nominal_capacity_ah", np.nan)),
        "soh": np.nan,
        "time_s": time_s.tolist(),
        "voltage_v": voltage_v.tolist(),
        "current_a": current_a.tolist(),
        "temperature_c": temperature_c.tolist(),
        "capacity_ah": capacity_ah.tolist(),
        "energy_wh": energy_wh.tolist(),
        "power_w": power_w.tolist(),
        "n_points": int(n),
        "duration_s": duration_s,
        "sample_period_median_s": sample_period_median_s,
        "voltage_min_v": voltage_min_v,
        "voltage_max_v": voltage_max_v,
        "voltage_start_v": float(voltage_v[0]) if n else np.nan,
        "voltage_end_v": float(voltage_v[-1]) if n else np.nan,
        "current_min_a": current_min_a,
        "current_max_a": current_max_a,
        "current_mean_a": current_mean_a,
        "temperature_min_c": temp_min_c,
        "temperature_max_c": temp_max_c,
        "temperature_mean_c": temp_mean_c,
        "phase_capacity_ah": phase_capacity_ah,
        "phase_energy_wh": phase_energy_wh,
        "voltage_source": "measured" if has_voltage else "missing",
        "current_source": current_source,
        "temperature_source": temperature_source,
        "capacity_source": capacity_source,
        "energy_source": energy_source,
        "chemistry": str(row.get("chemistry", "NMC/graphite") or "NMC/graphite"),
        "form_factor": str(row.get("form_factor", "18650") or "18650"),
        "manufacturer": str(row.get("manufacturer", "Sanyo/Panasonic") or "Sanyo/Panasonic"),
        "nominal_voltage_v": robust_float(row.get("nominal_voltage_v", np.nan)),
        "ambient_temperature_c": robust_float(row.get("ambient_temperature_c", np.nan)),
        "charge_protocol": str(row.get("charge_protocol", "RWTH source protocol") or "RWTH source protocol"),
        "discharge_protocol": str(row.get("discharge_protocol", "RWTH source protocol") or "RWTH source protocol"),
        "c_rate_charge": robust_float(row.get("c_rate_charge", np.nan)),
        "c_rate_discharge": robust_float(row.get("c_rate_discharge", np.nan)),
        "dod": robust_float(row.get("dod", np.nan)),
        "policy_readable": str(row.get("policy_readable", "") or ""),
        "barcode": str(row.get("barcode", "") or ""),
        "is_valid_cycle": bool(is_valid_cycle),
        "is_model_usable": False,  # assigned after lifetime metadata is computed
        "exclude_reason": ";".join(validity_failures),
        "cleaning_warnings": sorted(set(warnings)),
        "n_removed_points": int(n_removed),
        "trim_start_index": int(trim_start),
        "trim_end_index": int(trim_end),
        "has_voltage": has_voltage,
        "has_current": has_current,
        "has_temperature": has_temperature,
        "has_capacity": has_capacity,
        "has_energy": has_energy,
        "_source_timestamp": str(row.get("source_timestamp", "") or ""),
        "_source_cycle_sort": parse_numeric_token(row.get("source_cycle_id", "")),
        "adapter_version": str(row.get("adapter_version", "rwth_extract_unknown") or "rwth_extract_unknown"),
        "cleaning_version": CLEANING_VERSION,
        "created_at": now_iso(),
    }
    return out


def parse_numeric_token(value: Any) -> float:
    """Return the first numeric token from a source identifier for stable sorting."""
    text = str(value or "")
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return np.inf
    try:
        return float(match.group(1))
    except Exception:
        return np.inf


def parse_source_sort_timestamp(row: pd.Series) -> float:
    """Return a chronological sort key from RWTH metadata.

    The first cleaning version sorted by source_file text. For RWTH, this is unsafe
    because labels such as BOL, EOL, and ZYK are alphabetically ordered differently
    from the experimental chronology. This function prioritizes source_timestamp
    stored by extraction and falls back to timestamps embedded in the ZIP name.
    """
    candidates = [
        str(row.get("_source_timestamp", "") or ""),
        str(row.get("source_timestamp", "") or ""),
        str(row.get("source_file", "") or ""),
        str(row.get("source_path", "") or ""),
    ]
    for text in candidates:
        if not text:
            continue
        ts = pd.to_datetime(text, errors="coerce")
        if not pd.isna(ts):
            try:
                return float(ts.value)
            except Exception:
                pass
        match = re.search(r"(20\d{2}-\d{2}-\d{2}\s+\d{6})", text)
        if match:
            ts = pd.to_datetime(match.group(1), format="%Y-%m-%d %H%M%S", errors="coerce")
            if not pd.isna(ts):
                return float(ts.value)
    return np.inf


def parse_source_ts_number(row: pd.Series) -> float:
    text = " ".join([
        str(row.get("source_file", "") or ""),
        str(row.get("source_path", "") or ""),
    ])
    match = re.search(r"TS(\d+)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return np.inf


def classify_role_from_policy(policy: Any) -> str:
    text = str(policy).lower()
    if "full" in text:
        return "full"
    if "partial" in text:
        return "partial"
    return "unknown"


def assign_cycle_order_and_targets(
    df: pd.DataFrame,
    eol_ratio: float,
    min_voltage_span: float,
    target_mode: str = "end_of_sequence",
    min_capacity_eol_cycle_life: int = 100,
) -> pd.DataFrame:
    """Assign clean cycle order, initial capacity, SOH, and target columns.

    RWTH contains both full reference tests and many partial aging segments.
    The partial segments have different depth of discharge and therefore their
    phase_capacity_ah values are not directly comparable to full-cycle capacity.

    For that reason, the default target mode is ``end_of_sequence``. In this mode,
    ``cycle_life`` is the last observed chronological segment for each battery and
    phase. This yields a consistent remaining-observed-cycle target for feature
    extraction and prevents the discharge branch from receiving artificial
    cycle_life values close to 8--10.

    The optional ``capacity_eol`` mode keeps the stricter 80%-of-initial-full-
    reference-capacity criterion, but it is usually not appropriate for RWTH
    discharge segments because the discharge branch is dominated by partial
    variable-DoD segments. The ``hybrid`` mode uses capacity_eol only when the
    inferred cycle life is plausible; otherwise it falls back to end_of_sequence.
    """
    if df.empty:
        return df
    df = df.copy()
    df["_source_datetime_sort"] = df.apply(parse_source_sort_timestamp, axis=1)
    df["_source_ts_sort"] = df.apply(parse_source_ts_number, axis=1)
    if "_source_cycle_sort" not in df.columns:
        df["_source_cycle_sort"] = df["source_cycle_id"].apply(parse_numeric_token)
    sort_cols = [
        "battery_id", "phase", "_source_datetime_sort", "_source_ts_sort",
        "_source_cycle_sort", "source_file", "source_cycle_id",
    ]
    df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    groups: list[pd.DataFrame] = []

    for (battery_id, phase), group in df.groupby(["battery_id", "phase"], sort=True):
        group = group.copy().reset_index(drop=True)
        group["cycle_index"] = np.arange(1, len(group) + 1, dtype=int)
        group["cycle_number"] = group["cycle_index"].astype(float)
        group["role_class"] = group["policy_readable"].apply(classify_role_from_policy)

        # Initial capacity is role-specific. This prevents partial aging throughput
        # from being interpreted as degraded full-cell capacity.
        group["initial_capacity_ah"] = np.nan
        for role_class, role_group in group.groupby("role_class", sort=False):
            valid_cap = role_group.loc[
                role_group["is_valid_cycle"] & np.isfinite(role_group["phase_capacity_ah"]),
                "phase_capacity_ah",
            ]
            if len(valid_cap):
                initial = float(np.nanmedian(valid_cap.head(min(3, len(valid_cap))).to_numpy(dtype=float)))
                group.loc[group["role_class"] == role_class, "initial_capacity_ah"] = initial

        group["soh"] = np.where(
            np.isfinite(group["initial_capacity_ah"]) & (group["initial_capacity_ah"] > 0),
            group["phase_capacity_ah"] / group["initial_capacity_ah"],
            np.nan,
        )

        # Cycle life is inferred only from full reference cycles.
        full_ref = group[
            (group["role_class"] == "full")
            & group["is_valid_cycle"]
            & np.isfinite(group["phase_capacity_ah"])
        ].copy()
        if not full_ref.empty:
            initial_full = float(np.nanmedian(full_ref["phase_capacity_ah"].head(min(3, len(full_ref))).to_numpy(dtype=float)))
            eol_capacity = eol_ratio * initial_full if np.isfinite(initial_full) else np.nan
        else:
            initial_full = np.nan
            eol_capacity = np.nan

        observed_cycle_life = float(group["cycle_number"].max()) if len(group) else np.nan

        inferred_capacity_cycle_life = np.nan
        if np.isfinite(eol_capacity) and not full_ref.empty:
            below = full_ref[full_ref["phase_capacity_ah"] <= eol_capacity]
            if not below.empty:
                inferred_capacity_cycle_life = float(below.iloc[0]["cycle_number"])

        use_capacity_eol = False
        if target_mode == "capacity_eol":
            use_capacity_eol = np.isfinite(inferred_capacity_cycle_life)
        elif target_mode == "hybrid":
            use_capacity_eol = (
                np.isfinite(inferred_capacity_cycle_life)
                and inferred_capacity_cycle_life >= float(min_capacity_eol_cycle_life)
                and inferred_capacity_cycle_life <= observed_cycle_life
            )
        elif target_mode == "end_of_sequence":
            use_capacity_eol = False
        else:
            raise ValueError(f"Unsupported target_mode: {target_mode}")

        if use_capacity_eol:
            cycle_life = float(inferred_capacity_cycle_life)
            group["eol_capacity_ah"] = eol_capacity
            group["eol_criterion"] = f"{int(round(eol_ratio * 100))}pct_initial_full_reference_capacity"
        else:
            cycle_life = observed_cycle_life
            group["eol_capacity_ah"] = np.nan
            if target_mode == "capacity_eol" and not np.isfinite(inferred_capacity_cycle_life):
                group["eol_criterion"] = "unknown_capacity_eol_not_reached_or_unavailable"
                cycle_life = np.nan
            elif target_mode == "hybrid" and np.isfinite(inferred_capacity_cycle_life):
                group["eol_criterion"] = "end_of_available_rwth_sequence_fallback_from_implausible_capacity_eol"
            else:
                group["eol_criterion"] = "end_of_available_rwth_sequence"

        if np.isfinite(cycle_life):
            group["cycle_life"] = cycle_life
            group["remaining_cycles"] = group["cycle_life"] - group["cycle_number"]
            group.loc[group["remaining_cycles"] < 0, "remaining_cycles"] = np.nan
            group["remaining_cycles_norm"] = group["remaining_cycles"] / group["cycle_life"]
        else:
            group["cycle_life"] = np.nan
            group["remaining_cycles"] = np.nan
            group["remaining_cycles_norm"] = np.nan

        model_flags = []
        reasons = []
        for _, row in group.iterrows():
            reason_parts = [part for part in str(row.get("exclude_reason", "")).split(";") if part]
            interpolation_ok = can_interpolate_qv_curve(
                safe_array(row["voltage_v"]), safe_array(row["capacity_ah"]), min_voltage_span
            )
            if bool(row["is_valid_cycle"]) and not interpolation_ok:
                reason_parts.append("insufficient_voltage_range_for_interpolation")
            if bool(row["is_valid_cycle"]) and not np.isfinite(row.get("cycle_life", np.nan)):
                reason_parts.append("unknown_cycle_life")
            if bool(row["is_valid_cycle"]) and np.isfinite(row.get("cycle_life", np.nan)):
                remaining = robust_float(row.get("remaining_cycles", np.nan))
                if not np.isfinite(remaining) or remaining <= 0:
                    reason_parts.append("non_positive_remaining_cycles")
            is_usable = (
                bool(row["is_valid_cycle"])
                and interpolation_ok
                and np.isfinite(row.get("cycle_life", np.nan))
                and np.isfinite(row.get("remaining_cycles", np.nan))
                and row.get("remaining_cycles", np.nan) > 0
                and row.get("phase") in {"charge", "discharge"}
            )
            model_flags.append(bool(is_usable))
            reasons.append(";".join(sorted(set(reason_parts))))
        group["is_model_usable"] = model_flags
        group["exclude_reason"] = reasons
        drop_cols = [
            "role_class", "_source_timestamp", "_source_cycle_sort",
            "_source_datetime_sort", "_source_ts_sort",
        ]
        group = group.drop(columns=[col for col in drop_cols if col in group.columns])
        groups.append(group)

    return pd.concat(groups, ignore_index=True)


def empty_clean_frame() -> pd.DataFrame:
    df = pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)
    return df


def validate_clean_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in STANDARD_CYCLE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[STANDARD_CYCLE_COLUMNS]


def read_extracted_file(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def write_rejected_cycles(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        pd.DataFrame(columns=[
            "dataset", "battery_id", "phase", "cycle_index", "cycle_number", "source_file",
            "source_cycle_id", "n_points", "phase_capacity_ah", "is_valid_cycle",
            "is_model_usable", "exclude_reason", "cleaning_warnings",
        ]).to_csv(path, index=False, encoding="utf-8-sig")
        return
    rejected = df[~df["is_model_usable"].astype(bool)].copy()
    cols = [
        "dataset", "battery_id", "phase", "cycle_index", "cycle_number", "source_file",
        "source_cycle_id", "n_points", "phase_capacity_ah", "is_valid_cycle",
        "is_model_usable", "exclude_reason", "cleaning_warnings",
    ]
    rejected[cols].to_csv(path, index=False, encoding="utf-8-sig")


def clean_phase(phase: str, extracted_dir: Path, clean_dir: Path, report_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    input_file = extracted_dir / f"rwth_{phase}_extracted_cycles.parquet"
    output_file = clean_dir / f"rwth_{phase}_clean_cycles.parquet"
    rejected_file = clean_dir / f"rwth_{phase}_rejected_cycles.csv"

    raw = read_extracted_file(input_file)
    if raw.empty:
        clean = empty_clean_frame()
        clean.to_parquet(output_file, index=False)
        write_rejected_cycles(clean, rejected_file)
        return clean

    rows = []
    iterator = raw.iterrows()
    for _, row in tqdm(iterator, total=len(raw), desc=f"Cleaning RWTH {phase} cycles", unit="cycle"):
        rows.append(clean_one_row(row, min_points=args.min_points, min_voltage_span=args.min_voltage_span))
    clean = pd.DataFrame(rows)
    clean = assign_cycle_order_and_targets(
        clean,
        eol_ratio=args.eol_ratio,
        min_voltage_span=args.min_voltage_span,
        target_mode=args.target_mode,
        min_capacity_eol_cycle_life=args.min_capacity_eol_cycle_life,
    )
    clean = validate_clean_schema(clean)
    clean.to_parquet(output_file, index=False)
    write_rejected_cycles(clean, rejected_file)

    summary = {
        "dataset": DATASET,
        "phase": phase,
        "created_at": now_iso(),
        "input_file": str(input_file),
        "output_file": str(output_file),
        "rejected_file": str(rejected_file),
        "n_extracted_rows": int(len(raw)),
        "n_clean_rows": int(len(clean)),
        "n_valid_cycles": int(clean["is_valid_cycle"].sum()) if not clean.empty else 0,
        "n_model_usable_cycles": int(clean["is_model_usable"].sum()) if not clean.empty else 0,
        "target_mode": args.target_mode,
        "exclude_reason_counts": clean["exclude_reason"].fillna("").value_counts().to_dict() if not clean.empty else {},
    }
    (report_dir / f"rwth_{phase}_cleaning_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return clean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean RWTH extracted charge/discharge cycles using the official schema.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--extracted-dir", type=Path, default=None)
    parser.add_argument("--clean-dir", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--min-voltage-span", type=float, default=0.05)
    parser.add_argument("--eol-ratio", type=float, default=0.80)
    parser.add_argument(
        "--target-mode",
        choices=["end_of_sequence", "capacity_eol", "hybrid"],
        default="end_of_sequence",
        help=(
            "Target assignment mode. For RWTH, end_of_sequence is the recommended default because "
            "the dataset contains many partial variable-DoD segments whose capacities are not "
            "directly comparable to full-cycle EOL capacity. Use capacity_eol only for strict "
            "80%% full-reference-capacity targets; use hybrid to fall back when capacity EOL is implausible."
        ),
    )
    parser.add_argument(
        "--min-capacity-eol-cycle-life",
        type=int,
        default=100,
        help="Minimum plausible cycle life for accepting capacity-based EOL in hybrid mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extracted_dir = args.extracted_dir or args.base_dir / "data" / "interim" / "rwth" / "extracted"
    clean_dir = ensure_dir(args.clean_dir or args.base_dir / "data" / "interim" / "rwth" / "clean")
    report_dir = ensure_dir(args.report_dir or args.base_dir / "data" / "reports" / "rwth" / "cleaning")

    charge = clean_phase("charge", extracted_dir, clean_dir, report_dir, args)
    discharge = clean_phase("discharge", extracted_dir, clean_dir, report_dir, args)

    global_summary = {
        "dataset": DATASET,
        "cleaning_version": CLEANING_VERSION,
        "created_at": now_iso(),
        "clean_dir": str(clean_dir),
        "target_mode": args.target_mode,
        "n_charge_cycles": int(len(charge)),
        "n_discharge_cycles": int(len(discharge)),
        "n_charge_valid": int(charge["is_valid_cycle"].sum()) if not charge.empty else 0,
        "n_discharge_valid": int(discharge["is_valid_cycle"].sum()) if not discharge.empty else 0,
        "n_charge_model_usable": int(charge["is_model_usable"].sum()) if not charge.empty else 0,
        "n_discharge_model_usable": int(discharge["is_model_usable"].sum()) if not discharge.empty else 0,
    }
    summary_file = report_dir / "rwth_cleaning_summary.json"
    summary_file.write_text(json.dumps(global_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nRWTH cleaning completed.")
    print(f"Clean directory: {clean_dir}")
    print(f"Report: {summary_file}")
    print(f"Charge cycles: {global_summary['n_charge_cycles']} | valid: {global_summary['n_charge_valid']} | usable: {global_summary['n_charge_model_usable']}")
    print(f"Discharge cycles: {global_summary['n_discharge_cycles']} | valid: {global_summary['n_discharge_valid']} | usable: {global_summary['n_discharge_model_usable']}")


if __name__ == "__main__":
    main()
