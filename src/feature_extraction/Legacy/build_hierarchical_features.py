"""
Hierarchical feature extraction for lithium-ion battery RUL prediction.

This script reads standardized clean cycle files from data/interim, extracts
hierarchical scalar features from charge/discharge phase rows, and saves the
outputs in block-based parquet parts to reduce memory pressure and support
resume mode.

Methodological convention:
    One input row = one observed charge/discharge phase or phase segment.
    The script does not reconstruct unobserved full electrochemical cycles.
    Rest/relaxation features are only extracted from low-current regions or
    time gaps already preserved in the input vectors.

Default project layout expected by this script:
    C:/Artigo Baterias de Litio/Batteries/data/interim
    C:/Artigo Baterias de Litio/Batteries/data/processed/feature_extraction
    C:/Artigo Baterias de Litio/Batteries/data/reports/feature_extraction

Recommended location in the project:
    src/feature_extraction/build_hierarchical_features.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Constants and defaults
# =============================================================================

DEFAULT_DATA_ROOT = Path(r"C:\Artigo Baterias de Litio\Batteries\data")
DEFAULT_PARQUET_ENGINE = "fastparquet"

LEVEL_ORDER = [
    "H1_global",
    "H2_curves",
    "H3_stage_structure",
    "H4_individual_events",
    "H5_temporal_degradation",
    "H6_full",
]

LEVEL_SHORT_TO_NAME = {
    "H1": "H1_global",
    "H2": "H2_curves",
    "H3": "H3_stage_structure",
    "H4": "H4_individual_events",
    "H5": "H5_temporal_degradation",
    "H6": "H6_full",
    "H1_GLOBAL": "H1_global",
    "H2_CURVES": "H2_curves",
    "H3_STAGE_STRUCTURE": "H3_stage_structure",
    "H4_INDIVIDUAL_EVENTS": "H4_individual_events",
    "H5_TEMPORAL_DEGRADATION": "H5_temporal_degradation",
    "H6_FULL": "H6_full",
    "ALL": "H6_full",
}

H0_COLUMNS = [
    "dataset",
    "battery_id",
    "phase",
    "cycle_number",
    "cycle_index",
    "cycle_life",
    "remaining_cycles",
    "remaining_cycles_norm",
    "soh",
    "source_file",
    "source_path",
    "source_cycle_id",
    "cycle_pair_id",
    "is_valid_cycle",
    "is_model_usable",
    "exclude_reason",
]

MINIMUM_REQUIRED_COLUMNS = [
    "dataset",
    "battery_id",
    "cycle_number",
    "cycle_index",
    "cycle_life",
    "remaining_cycles",
    "phase",
    "time_s",
    "voltage_v",
    "current_a",
    "temperature_c",
    "capacity_ah",
]

VECTOR_COLUMNS = [
    "time_s",
    "voltage_v",
    "current_a",
    "temperature_c",
    "capacity_ah",
    "energy_wh",
    "power_w",
]

SCALAR_METADATA_PASSTHROUGH = [
    "battery_index",
    "cell_id_original",
    "nominal_capacity_ah",
    "initial_capacity_ah",
    "nominal_voltage_v",
    "ambient_temperature_c",
    "chemistry",
    "form_factor",
    "manufacturer",
    "charge_protocol",
    "discharge_protocol",
    "c_rate_charge",
    "c_rate_discharge",
    "dod",
    "policy_readable",
    "barcode",
    "voltage_source",
    "current_source",
    "temperature_source",
    "capacity_source",
    "energy_source",
]


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class ExtractionConfig:
    data_root: Path
    processed_root: Path
    reports_root: Path
    parquet_engine: str
    resolutions: List[int]
    feature_level: str
    batch_size: int
    resume: bool
    overwrite: bool
    merge: bool
    save_intermediate_levels: bool
    keep_raw_vectors: bool
    dataset_filter: str
    battery_filter: str
    phase: str
    minimum_points: int
    minimum_voltage_span_v: float
    minimum_capacity_span_ah: float
    rest_c_rate_fraction: float
    minimum_rest_current_a: float
    min_rest_duration_s: float
    min_active_duration_s: float
    gap_factor: float
    minimum_gap_duration_s: float
    q_gap_abs_threshold_ah: float
    q_gap_fraction_threshold: float
    v_gap_threshold_v: float
    max_active_segments: int
    max_observed_rest_segments: int
    max_gap_events: int


# =============================================================================
# General utilities
# =============================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_name(value: Any) -> str:
    text = str(value) if value is not None else "unknown"
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def finite_or_nan(value: Any) -> float:
    try:
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(value, dtype=float).ravel()
            valid = arr[np.isfinite(arr)]
            return float(valid[-1]) if valid.size else np.nan
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def safe_divide(a: Any, b: Any) -> float:
    a_f = finite_or_nan(a)
    b_f = finite_or_nan(b)
    if not np.isfinite(a_f) or not np.isfinite(b_f) or abs(b_f) < 1e-12:
        return np.nan
    return float(a_f / b_f)


def as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, pd.Series):
        arr = value.to_numpy()
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    else:
        try:
            arr = np.asarray(value)
            if arr.ndim == 0:
                return np.array([], dtype=float)
        except Exception:
            return np.array([], dtype=float)
    try:
        arr = arr.astype(float, copy=False)
    except Exception:
        return np.array([], dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr.astype(float, copy=False)


def finite_fraction(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).sum() / arr.size)


def last_finite(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    valid = arr[np.isfinite(arr)]
    return float(valid[-1]) if valid.size else np.nan


def first_finite(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    valid = arr[np.isfinite(arr)]
    return float(valid[0]) if valid.size else np.nan


def numeric_stats(prefix: str, arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {
            f"{prefix}_start": np.nan,
            f"{prefix}_end": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_q1": np.nan,
            f"{prefix}_q3": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_delta": np.nan,
            f"{prefix}_abs_delta": np.nan,
        }
    q1 = float(np.nanpercentile(valid, 25))
    q3 = float(np.nanpercentile(valid, 75))
    start = first_finite(arr)
    end = last_finite(arr)
    min_v = float(np.nanmin(valid))
    max_v = float(np.nanmax(valid))
    delta = end - start if np.isfinite(start) and np.isfinite(end) else np.nan
    return {
        f"{prefix}_start": start,
        f"{prefix}_end": end,
        f"{prefix}_min": min_v,
        f"{prefix}_max": max_v,
        f"{prefix}_mean": float(np.nanmean(valid)),
        f"{prefix}_median": float(np.nanmedian(valid)),
        f"{prefix}_std": float(np.nanstd(valid)),
        f"{prefix}_q1": q1,
        f"{prefix}_q3": q3,
        f"{prefix}_iqr": float(q3 - q1),
        f"{prefix}_range": float(max_v - min_v),
        f"{prefix}_delta": float(delta) if np.isfinite(delta) else np.nan,
        f"{prefix}_abs_delta": float(abs(delta)) if np.isfinite(delta) else np.nan,
    }


def slope_stats(prefix: str, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return {
            f"{prefix}_slope_mean": np.nan,
            f"{prefix}_slope_std": np.nan,
            f"{prefix}_slope_min": np.nan,
            f"{prefix}_slope_max": np.nan,
            f"{prefix}_slope_max_abs": np.nan,
        }
    dx = np.diff(x)
    dy = np.diff(y)
    valid = np.isfinite(dx) & np.isfinite(dy) & (np.abs(dx) > 1e-12)
    if valid.sum() == 0:
        slopes = np.array([], dtype=float)
    else:
        slopes = dy[valid] / dx[valid]
        slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return {
            f"{prefix}_slope_mean": np.nan,
            f"{prefix}_slope_std": np.nan,
            f"{prefix}_slope_min": np.nan,
            f"{prefix}_slope_max": np.nan,
            f"{prefix}_slope_max_abs": np.nan,
        }
    return {
        f"{prefix}_slope_mean": float(np.nanmean(slopes)),
        f"{prefix}_slope_std": float(np.nanstd(slopes)),
        f"{prefix}_slope_min": float(np.nanmin(slopes)),
        f"{prefix}_slope_max": float(np.nanmax(slopes)),
        f"{prefix}_slope_max_abs": float(np.nanmax(np.abs(slopes))),
    }


def trapz_custom(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(y) & np.isfinite(x)
    y = y[valid]
    x = x[valid]
    if y.size < 2:
        return np.nan
    dx = np.diff(x)
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * dx))


def coefficient_of_variation(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    mean = float(np.nanmean(arr))
    if not np.isfinite(mean) or abs(mean) < 1e-12:
        return np.nan
    return float(np.nanstd(arr) / abs(mean))


def linear_slope(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 2 or np.nanmax(x_arr) == np.nanmin(x_arr):
        return np.nan
    try:
        slope, _ = np.polyfit(x_arr, y_arr, 1)
        return float(slope)
    except Exception:
        return np.nan


def normalized_euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        return np.nan
    valid = np.isfinite(a_arr) & np.isfinite(b_arr)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.nanmean((a_arr[valid] - b_arr[valid]) ** 2)))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# =============================================================================
# Vector alignment and reconstruction
# =============================================================================

def integrate_capacity_from_current(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    current_a = np.asarray(current_a, dtype=float)
    if time_s.size < 2 or current_a.size != time_s.size:
        return np.full(time_s.size, np.nan, dtype=float)
    dt = np.diff(time_s)
    i_abs = np.abs(current_a)
    dq = (i_abs[1:] + i_abs[:-1]) * 0.5 * dt / 3600.0
    dq[~np.isfinite(dq)] = 0.0
    dq[dq < 0] = 0.0
    return np.concatenate([[0.0], np.cumsum(dq)])


def integrate_energy_from_power(time_s: np.ndarray, power_w: np.ndarray) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    power_w = np.asarray(power_w, dtype=float)
    if time_s.size < 2 or power_w.size != time_s.size:
        return np.full(time_s.size, np.nan, dtype=float)
    dt = np.diff(time_s)
    p_abs = np.abs(power_w)
    de = (p_abs[1:] + p_abs[:-1]) * 0.5 * dt / 3600.0
    de[~np.isfinite(de)] = 0.0
    de[de < 0] = 0.0
    return np.concatenate([[0.0], np.cumsum(de)])


def get_cycle_arrays(row: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_s = as_float_array(row.get("time_s"))
    voltage_v = as_float_array(row.get("voltage_v"))
    current_a = as_float_array(row.get("current_a"))
    temperature_c = as_float_array(row.get("temperature_c"))
    capacity_ah = as_float_array(row.get("capacity_ah"))
    energy_wh = as_float_array(row.get("energy_wh"))
    power_w = as_float_array(row.get("power_w"))

    lengths = [
        len(x)
        for x in [time_s, voltage_v, current_a, temperature_c, capacity_ah, energy_wh, power_w]
        if len(x) > 0
    ]
    if not lengths:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty, empty, empty

    n = min(lengths)
    if n <= 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty, empty, empty

    def align(arr: np.ndarray, fill: float = np.nan) -> np.ndarray:
        if len(arr) >= n:
            return arr[:n].astype(float, copy=False)
        return np.full(n, fill, dtype=float)

    time_s = align(time_s)
    voltage_v = align(voltage_v)
    current_a = align(current_a)
    temperature_c = align(temperature_c)

    if len(capacity_ah) >= n and np.isfinite(capacity_ah[:n]).sum() >= 2:
        capacity_ah = capacity_ah[:n].astype(float, copy=False)
    else:
        capacity_ah = integrate_capacity_from_current(time_s, current_a)

    if len(power_w) >= n and np.isfinite(power_w[:n]).sum() >= 2:
        power_w = power_w[:n].astype(float, copy=False)
    else:
        power_w = voltage_v * current_a

    if len(energy_wh) >= n and np.isfinite(energy_wh[:n]).sum() >= 2:
        energy_wh = energy_wh[:n].astype(float, copy=False)
    else:
        energy_wh = integrate_energy_from_power(time_s, power_w)

    return time_s, voltage_v, current_a, temperature_c, capacity_ah, energy_wh, power_w


# =============================================================================
# Input discovery
# =============================================================================

def normalize_filter(value: Optional[str]) -> str:
    if value is None:
        return "all"
    value = str(value).strip()
    return value if value else "all"


def filter_matches(value: Any, filter_text: str) -> bool:
    filter_text = normalize_filter(filter_text)
    if filter_text.lower() == "all":
        return True
    options = [x.strip().lower() for x in filter_text.split(",") if x.strip()]
    value_text = str(value).lower()
    return any(opt in value_text for opt in options)


def identify_dataset_phase_from_path(path: Path) -> Tuple[str, str]:
    name = path.name.lower()
    phase = "unknown"
    if "charge" in name:
        phase = "charge"
    elif "discharge" in name:
        phase = "discharge"

    dataset = path.stem
    m = re.match(r"(.+?)_(charge|discharge)_clean_cycles", path.stem, flags=re.IGNORECASE)
    if m:
        dataset = m.group(1)
    elif path.parent.name.lower() == "clean" and path.parent.parent.name:
        dataset = path.parent.parent.name
    elif path.parent.name.lower() in {"charge", "discharge"}:
        dataset = "global"
    return dataset.lower(), phase


def find_input_files(data_root: Path, phase: str, dataset_filter: str = "all") -> List[Path]:
    interim_root = data_root / "interim"
    phases = ["charge", "discharge"] if phase.lower() == "all" else [phase.lower()]
    selected: List[Path] = []

    for ph in phases:
        per_dataset_files = sorted(interim_root.glob(f"*/clean/*_{ph}_clean_cycles.parquet"))
        per_dataset_files = [
            p for p in per_dataset_files
            if p.parent.parent.name.lower() not in {"charge", "discharge"}
            and filter_matches(str(p), dataset_filter)
        ]
        if per_dataset_files:
            selected.extend(per_dataset_files)
            continue

        global_file = interim_root / ph / f"{ph}_all_clean_cycles.parquet"
        if global_file.exists():
            selected.append(global_file)

    return sorted(dict.fromkeys(selected))


def validate_minimum_columns(df: pd.DataFrame, file_path: Path) -> None:
    missing = [col for col in MINIMUM_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_path}: {missing}")


# =============================================================================
# H0 metadata
# =============================================================================

def extract_h0_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in H0_COLUMNS:
        out[col] = row.get(col, np.nan)
    for col in SCALAR_METADATA_PASSTHROUGH:
        if col in row:
            value = row.get(col)
            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                continue
            out[col] = value
    return out


# =============================================================================
# H1 global features
# =============================================================================

def extract_global_features(row: Dict[str, Any], cfg: ExtractionConfig) -> Dict[str, Any]:
    time_s, voltage_v, current_a, temperature_c, capacity_ah, energy_wh, power_w = get_cycle_arrays(row)
    phase = str(row.get("phase", "")).lower().strip()

    feats: Dict[str, Any] = {}
    n = int(len(time_s))
    feats["n_points"] = n

    if n >= 2:
        dt = np.diff(time_s)
        dt_valid = dt[np.isfinite(dt)]
        feats["duration_s"] = float(time_s[-1] - time_s[0]) if np.isfinite(time_s[-1] - time_s[0]) else np.nan
        if dt_valid.size:
            feats["sample_period_median_s"] = float(np.nanmedian(dt_valid))
            feats["sample_period_mean_s"] = float(np.nanmean(dt_valid))
            feats["sample_period_std_s"] = float(np.nanstd(dt_valid))
            feats["sample_period_min_s"] = float(np.nanmin(dt_valid))
            feats["sample_period_max_s"] = float(np.nanmax(dt_valid))
            feats["sampling_irregularity_ratio"] = safe_divide(feats["sample_period_max_s"], feats["sample_period_median_s"])
        else:
            feats.update({
                "sample_period_median_s": np.nan,
                "sample_period_mean_s": np.nan,
                "sample_period_std_s": np.nan,
                "sample_period_min_s": np.nan,
                "sample_period_max_s": np.nan,
                "sampling_irregularity_ratio": np.nan,
            })
    else:
        feats.update({
            "duration_s": np.nan,
            "sample_period_median_s": np.nan,
            "sample_period_mean_s": np.nan,
            "sample_period_std_s": np.nan,
            "sample_period_min_s": np.nan,
            "sample_period_max_s": np.nan,
            "sampling_irregularity_ratio": np.nan,
        })

    # Voltage
    vstats = numeric_stats("voltage_v", voltage_v)
    feats.update({
        "voltage_start_v": vstats.pop("voltage_v_start"),
        "voltage_end_v": vstats.pop("voltage_v_end"),
        "voltage_min_v": vstats.pop("voltage_v_min"),
        "voltage_max_v": vstats.pop("voltage_v_max"),
        "voltage_mean_v": vstats.pop("voltage_v_mean"),
        "voltage_median_v": vstats.pop("voltage_v_median"),
        "voltage_std_v": vstats.pop("voltage_v_std"),
        "voltage_q1_v": vstats.pop("voltage_v_q1"),
        "voltage_q3_v": vstats.pop("voltage_v_q3"),
        "voltage_iqr_v": vstats.pop("voltage_v_iqr"),
        "voltage_range_v": vstats.pop("voltage_v_range"),
        "voltage_delta_v": vstats.pop("voltage_v_delta"),
        "voltage_abs_delta_v": vstats.pop("voltage_v_abs_delta"),
    })
    vslope = slope_stats("voltage", time_s, voltage_v)
    feats["voltage_slope_mean_v_per_s"] = vslope["voltage_slope_mean"]
    feats["voltage_slope_std_v_per_s"] = vslope["voltage_slope_std"]
    feats["voltage_slope_max_abs_v_per_s"] = vslope["voltage_slope_max_abs"]

    # Current
    istats = numeric_stats("current_a", current_a)
    feats.update({
        "current_start_a": istats.pop("current_a_start"),
        "current_end_a": istats.pop("current_a_end"),
        "current_min_a": istats.pop("current_a_min"),
        "current_max_a": istats.pop("current_a_max"),
        "current_mean_a": istats.pop("current_a_mean"),
        "current_median_a": istats.pop("current_a_median"),
        "current_std_a": istats.pop("current_a_std"),
        "current_q1_a": istats.pop("current_a_q1"),
        "current_q3_a": istats.pop("current_a_q3"),
        "current_iqr_a": istats.pop("current_a_iqr"),
        "current_range_a": istats.pop("current_a_range"),
    })
    current_valid = current_a[np.isfinite(current_a)]
    if current_valid.size:
        feats["current_abs_mean_a"] = float(np.nanmean(np.abs(current_valid)))
        feats["current_abs_median_a"] = float(np.nanmedian(np.abs(current_valid)))
        feats["current_rms_a"] = float(np.sqrt(np.nanmean(current_valid ** 2)))
    else:
        feats["current_abs_mean_a"] = np.nan
        feats["current_abs_median_a"] = np.nan
        feats["current_rms_a"] = np.nan
    feats["current_variation_index"] = safe_divide(feats["current_std_a"], abs(feats["current_mean_a"]))
    if current_valid.size:
        if phase == "charge":
            feats["current_sign_consistency"] = float((current_valid > 0).sum() / current_valid.size)
        elif phase == "discharge":
            feats["current_sign_consistency"] = float((current_valid < 0).sum() / current_valid.size)
        else:
            feats["current_sign_consistency"] = np.nan
    else:
        feats["current_sign_consistency"] = np.nan

    # Temperature
    tstats = numeric_stats("temperature_c", temperature_c)
    feats.update({
        "temperature_start_c": tstats.pop("temperature_c_start"),
        "temperature_end_c": tstats.pop("temperature_c_end"),
        "temperature_min_c": tstats.pop("temperature_c_min"),
        "temperature_max_c": tstats.pop("temperature_c_max"),
        "temperature_mean_c": tstats.pop("temperature_c_mean"),
        "temperature_median_c": tstats.pop("temperature_c_median"),
        "temperature_std_c": tstats.pop("temperature_c_std"),
        "temperature_q1_c": tstats.pop("temperature_c_q1"),
        "temperature_q3_c": tstats.pop("temperature_c_q3"),
        "temperature_iqr_c": tstats.pop("temperature_c_iqr"),
        "temperature_range_c": tstats.pop("temperature_c_range"),
        "temperature_delta_c": tstats.pop("temperature_c_delta"),
    })
    tslope = slope_stats("temperature", time_s, temperature_c)
    feats["temperature_slope_mean_c_per_s"] = tslope["temperature_slope_mean"]
    feats["temperature_slope_std_c_per_s"] = tslope["temperature_slope_std"]

    # Capacity
    qstats = numeric_stats("capacity_ah", capacity_ah)
    feats.update({
        "capacity_start_ah": qstats.pop("capacity_ah_start"),
        "capacity_end_ah": qstats.pop("capacity_ah_end"),
        "phase_capacity_ah": last_finite(capacity_ah),
        "capacity_min_ah": qstats.pop("capacity_ah_min"),
        "capacity_max_ah": qstats.pop("capacity_ah_max"),
        "capacity_range_ah": qstats.pop("capacity_ah_range"),
        "capacity_delta_ah": qstats.pop("capacity_ah_delta"),
    })
    qslope = slope_stats("capacity", time_s, capacity_ah)
    feats["capacity_slope_mean_ah_per_s"] = qslope["capacity_slope_mean"]
    feats["capacity_slope_std_ah_per_s"] = qslope["capacity_slope_std"]
    if capacity_ah.size >= 2:
        dq = np.diff(capacity_ah)
        finite_dq = dq[np.isfinite(dq)]
        if finite_dq.size:
            feats["capacity_monotonicity_ratio"] = float((finite_dq >= -1e-12).sum() / finite_dq.size)
            feats["capacity_negative_step_count"] = int((finite_dq < -1e-12).sum())
            feats["capacity_zero_step_fraction"] = float((np.abs(finite_dq) <= 1e-12).sum() / finite_dq.size)
        else:
            feats["capacity_monotonicity_ratio"] = np.nan
            feats["capacity_negative_step_count"] = 0
            feats["capacity_zero_step_fraction"] = np.nan
    else:
        feats["capacity_monotonicity_ratio"] = np.nan
        feats["capacity_negative_step_count"] = 0
        feats["capacity_zero_step_fraction"] = np.nan

    # Energy and power
    feats["phase_energy_wh"] = last_finite(energy_wh)
    feats["energy_start_wh"] = first_finite(energy_wh)
    feats["energy_end_wh"] = last_finite(energy_wh)
    feats["energy_delta_wh"] = (
        feats["energy_end_wh"] - feats["energy_start_wh"]
        if np.isfinite(feats["energy_end_wh"]) and np.isfinite(feats["energy_start_wh"])
        else np.nan
    )
    feats["energy_per_capacity_wh_per_ah"] = safe_divide(feats["phase_energy_wh"], feats["phase_capacity_ah"])

    pstats = numeric_stats("power_w", power_w)
    feats.update({
        "power_start_w": pstats.pop("power_w_start"),
        "power_end_w": pstats.pop("power_w_end"),
        "power_min_w": pstats.pop("power_w_min"),
        "power_max_w": pstats.pop("power_w_max"),
        "power_mean_w": pstats.pop("power_w_mean"),
        "power_median_w": pstats.pop("power_w_median"),
        "power_std_w": pstats.pop("power_w_std"),
    })
    p_valid = power_w[np.isfinite(power_w)]
    if p_valid.size:
        feats["power_abs_mean_w"] = float(np.nanmean(np.abs(p_valid)))
        feats["power_abs_max_w"] = float(np.nanmax(np.abs(p_valid)))
        feats["power_rms_w"] = float(np.sqrt(np.nanmean(p_valid ** 2)))
    else:
        feats["power_abs_mean_w"] = np.nan
        feats["power_abs_max_w"] = np.nan
        feats["power_rms_w"] = np.nan

    # Quality flags
    feats["finite_time_fraction"] = finite_fraction(time_s)
    feats["finite_voltage_fraction"] = finite_fraction(voltage_v)
    feats["finite_current_fraction"] = finite_fraction(current_a)
    feats["finite_temperature_fraction"] = finite_fraction(temperature_c)
    feats["finite_capacity_fraction"] = finite_fraction(capacity_ah)
    feats["finite_energy_fraction"] = finite_fraction(energy_wh)
    feats["finite_power_fraction"] = finite_fraction(power_w)

    feats["has_voltage"] = bool(np.isfinite(voltage_v).any())
    feats["has_current"] = bool(np.isfinite(current_a).any())
    feats["has_temperature"] = bool(np.isfinite(temperature_c).any())
    feats["has_capacity"] = bool(np.isfinite(capacity_ah).any())
    feats["has_energy"] = bool(np.isfinite(energy_wh).any())
    feats["has_power"] = bool(np.isfinite(power_w).any())

    feats["is_short_segment"] = int(n < cfg.minimum_points)
    feats["is_low_voltage_span"] = int(
        np.isfinite(feats["voltage_range_v"]) and feats["voltage_range_v"] < cfg.minimum_voltage_span_v
    )
    feats["is_low_capacity_span"] = int(
        np.isfinite(feats["capacity_range_ah"]) and feats["capacity_range_ah"] < cfg.minimum_capacity_span_ah
    )
    feats["is_irregular_sampling"] = int(
        np.isfinite(feats["sampling_irregularity_ratio"]) and feats["sampling_irregularity_ratio"] > 10
    )

    if cfg.keep_raw_vectors:
        feats["time_s"] = time_s.tolist()
        feats["voltage_v"] = voltage_v.tolist()
        feats["current_a"] = current_a.tolist()
        feats["temperature_c"] = temperature_c.tolist()
        feats["capacity_ah"] = capacity_ah.tolist()
        feats["energy_wh"] = energy_wh.tolist()
        feats["power_w"] = power_w.tolist()

    return feats


# =============================================================================
# H2 curve features: V(Q) and Q(V)
# =============================================================================

def interpolate_linear_y_by_x(
    x: np.ndarray,
    y: np.ndarray,
    p: int,
    min_x_window: float,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return np.full(p, np.nan), np.full(p, np.nan), False

    df = pd.DataFrame({"x": x, "y": y})
    df = df.groupby("x", as_index=False)["y"].mean().sort_values("x")
    x_unique = df["x"].to_numpy(dtype=float)
    y_unique = df["y"].to_numpy(dtype=float)
    if x_unique.size < 2:
        return np.full(p, np.nan), np.full(p, np.nan), False

    x_min = float(np.nanmin(x_unique))
    x_max = float(np.nanmax(x_unique))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or abs(x_max - x_min) < min_x_window:
        return np.full(p, np.nan), np.full(p, np.nan), False

    x_samples = np.linspace(x_min, x_max, p)
    y_samples = np.interp(x_samples, x_unique, y_unique)
    return x_samples, y_samples, True


def curve_shape_features(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    ok: bool,
    prefix: str,
    p: int,
) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    p_tag = f"p{p:03d}"
    for i in range(p):
        value = y_samples[i] if i < len(y_samples) else np.nan
        feats[f"{prefix}_{p_tag}_{i + 1:03d}"] = float(value) if np.isfinite(value) else np.nan
    feats[f"{prefix}_valid_{p_tag}"] = int(bool(ok))

    if not ok:
        feats[f"{prefix}_area_{p_tag}"] = np.nan
        feats[f"{prefix}_slope_mean_{p_tag}"] = np.nan
        feats[f"{prefix}_slope_std_{p_tag}"] = np.nan
        feats[f"{prefix}_slope_min_{p_tag}"] = np.nan
        feats[f"{prefix}_slope_max_{p_tag}"] = np.nan
        feats[f"{prefix}_slope_max_abs_{p_tag}"] = np.nan
        feats[f"{prefix}_inflection_count_{p_tag}"] = np.nan
        feats[f"{prefix}_range_{p_tag}"] = np.nan
        return feats

    feats[f"{prefix}_area_{p_tag}"] = trapz_custom(y_samples, x_samples)
    feats[f"{prefix}_range_{p_tag}"] = float(np.nanmax(y_samples) - np.nanmin(y_samples))

    slopes_info = slope_stats(prefix, x_samples, y_samples)
    feats[f"{prefix}_slope_mean_{p_tag}"] = slopes_info[f"{prefix}_slope_mean"]
    feats[f"{prefix}_slope_std_{p_tag}"] = slopes_info[f"{prefix}_slope_std"]
    feats[f"{prefix}_slope_min_{p_tag}"] = slopes_info[f"{prefix}_slope_min"]
    feats[f"{prefix}_slope_max_{p_tag}"] = slopes_info[f"{prefix}_slope_max"]
    feats[f"{prefix}_slope_max_abs_{p_tag}"] = slopes_info[f"{prefix}_slope_max_abs"]

    if p >= 5 and np.isfinite(x_samples).all() and np.isfinite(y_samples).all():
        try:
            d1 = np.gradient(y_samples, x_samples)
            d2 = np.gradient(d1, x_samples)
            signs = np.sign(d2[np.isfinite(d2)])
            if signs.size >= 2:
                feats[f"{prefix}_inflection_count_{p_tag}"] = int(np.sum(np.diff(signs) != 0))
            else:
                feats[f"{prefix}_inflection_count_{p_tag}"] = 0
        except Exception:
            feats[f"{prefix}_inflection_count_{p_tag}"] = np.nan
    else:
        feats[f"{prefix}_inflection_count_{p_tag}"] = np.nan

    return feats


def extract_curve_features(row: Dict[str, Any], h1: Dict[str, Any], cfg: ExtractionConfig) -> Dict[str, Any]:
    time_s, voltage_v, current_a, temperature_c, capacity_ah, energy_wh, power_w = get_cycle_arrays(row)
    feats: Dict[str, Any] = {}

    enough_points = h1.get("n_points", 0) >= cfg.minimum_points
    enough_voltage = finite_or_nan(h1.get("voltage_range_v")) >= cfg.minimum_voltage_span_v
    enough_capacity = finite_or_nan(h1.get("capacity_range_ah")) >= cfg.minimum_capacity_span_ah
    enough_finite = min(finite_fraction(voltage_v), finite_fraction(capacity_ah)) >= 0.5
    base_ok = bool(enough_points and enough_voltage and enough_capacity and enough_finite)

    for p in cfg.resolutions:
        if base_ok:
            q_samples, v_samples, vq_ok = interpolate_linear_y_by_x(
                x=capacity_ah,
                y=voltage_v,
                p=p,
                min_x_window=cfg.minimum_capacity_span_ah,
            )
            v_samples_x, q_samples_y, qv_ok = interpolate_linear_y_by_x(
                x=voltage_v,
                y=capacity_ah,
                p=p,
                min_x_window=cfg.minimum_voltage_span_v,
            )
        else:
            q_samples = np.full(p, np.nan)
            v_samples = np.full(p, np.nan)
            v_samples_x = np.full(p, np.nan)
            q_samples_y = np.full(p, np.nan)
            vq_ok = False
            qv_ok = False

        feats.update(curve_shape_features(q_samples, v_samples, vq_ok, "VQ", p))
        feats.update(curve_shape_features(v_samples_x, q_samples_y, qv_ok, "QV", p))

    return feats


# =============================================================================
# H3 segmentation and time-gap features
# =============================================================================

def get_one_c_current_a(row: Dict[str, Any], current_a: np.ndarray) -> float:
    nominal = finite_or_nan(row.get("nominal_capacity_ah"))
    initial = finite_or_nan(row.get("initial_capacity_ah"))
    if np.isfinite(nominal) and nominal > 0:
        return float(nominal)
    if np.isfinite(initial) and initial > 0:
        return float(initial)
    abs_current = np.abs(current_a[np.isfinite(current_a)])
    if abs_current.size == 0:
        return np.nan
    p95 = float(np.nanpercentile(abs_current, 95))
    return p95 if np.isfinite(p95) and p95 > 0 else np.nan


def rest_current_threshold(row: Dict[str, Any], current_a: np.ndarray, cfg: ExtractionConfig) -> float:
    one_c = get_one_c_current_a(row, current_a)
    if np.isfinite(one_c) and one_c > 0:
        return float(max(cfg.minimum_rest_current_a, cfg.rest_c_rate_fraction * one_c))
    return float(cfg.minimum_rest_current_a)


def find_segments_from_mask(mask: np.ndarray, min_points: int = 2) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            end = i - 1
            if end - start + 1 >= min_points:
                segments.append((start, end))
            start = None
    if start is not None:
        end = len(mask) - 1
        if end - start + 1 >= min_points:
            segments.append((start, end))
    return segments


def filter_segments_by_duration(segments: List[Tuple[int, int]], time_s: np.ndarray, min_duration_s: float) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s, e in segments:
        if s < 0 or e >= len(time_s):
            continue
        duration = time_s[e] - time_s[s]
        if np.isfinite(duration) and duration >= min_duration_s:
            out.append((s, e))
    return out


def segment_duration(time_s: np.ndarray, seg: Tuple[int, int]) -> float:
    s, e = seg
    if len(time_s) == 0 or s < 0 or e >= len(time_s):
        return np.nan
    value = time_s[e] - time_s[s]
    return float(value) if np.isfinite(value) else np.nan


def segment_capacity_delta(capacity_ah: np.ndarray, seg: Tuple[int, int]) -> float:
    s, e = seg
    if len(capacity_ah) == 0 or s < 0 or e >= len(capacity_ah):
        return np.nan
    value = capacity_ah[e] - capacity_ah[s]
    return float(value) if np.isfinite(value) else np.nan


def detect_stage_context(row: Dict[str, Any], cfg: ExtractionConfig) -> Dict[str, Any]:
    time_s, voltage_v, current_a, temperature_c, capacity_ah, energy_wh, power_w = get_cycle_arrays(row)
    n = len(time_s)
    phase = str(row.get("phase", "")).lower().strip()
    context: Dict[str, Any] = {
        "active_segments": [],
        "low_current_segments": [],
        "observed_rest_segments": [],
        "internal_low_current_segments": [],
        "terminal_low_current_segments": [],
        "gap_events": [],
        "large_gap_events": [],
        "possible_relaxation_gap_events": [],
        "rest_current_threshold_a": np.nan,
        "one_c_current_a": np.nan,
    }
    if n < 2:
        return context

    valid = np.isfinite(time_s) & np.isfinite(voltage_v) & np.isfinite(current_a) & np.isfinite(capacity_ah)
    i_rest = rest_current_threshold(row, current_a, cfg)
    one_c = get_one_c_current_a(row, current_a)
    context["rest_current_threshold_a"] = float(i_rest)
    context["one_c_current_a"] = float(one_c) if np.isfinite(one_c) else np.nan

    low_current_mask = valid & (np.abs(current_a) <= i_rest)
    if phase == "charge":
        active_mask = valid & (current_a > i_rest)
    elif phase == "discharge":
        active_mask = valid & (current_a < -i_rest)
    else:
        active_mask = valid & (np.abs(current_a) > i_rest)

    active_raw = find_segments_from_mask(active_mask, min_points=2)
    low_raw = find_segments_from_mask(low_current_mask, min_points=2)
    active_segments = filter_segments_by_duration(active_raw, time_s, cfg.min_active_duration_s)
    low_segments = filter_segments_by_duration(low_raw, time_s, cfg.min_rest_duration_s)

    observed_rest_segments: List[Tuple[int, int]] = []
    internal_low: List[Tuple[int, int]] = []
    terminal_low: List[Tuple[int, int]] = []
    for s, e in low_segments:
        has_active_before = any(ae < s for _, ae in active_segments)
        has_active_after = any(as_ > e for as_, _ in active_segments)
        is_internal = has_active_before and has_active_after
        is_terminal = has_active_before and not has_active_after
        if is_internal:
            internal_low.append((s, e))
            observed_rest_segments.append((s, e))
        elif is_terminal:
            terminal_low.append((s, e))
            observed_rest_segments.append((s, e))

    # Gap events are point transitions i -> i+1
    gap_events: List[int] = []
    large_gap_events: List[int] = []
    possible_relaxation_gap_events: List[int] = []
    if n >= 2:
        dt = np.diff(time_s)
        dv = np.diff(voltage_v)
        dq = np.diff(capacity_ah)
        median_dt = np.nanmedian(dt[np.isfinite(dt)]) if np.isfinite(dt).any() else np.nan
        phase_capacity = last_finite(capacity_ah)
        gap_threshold = max(
            cfg.minimum_gap_duration_s,
            cfg.gap_factor * median_dt if np.isfinite(median_dt) else cfg.minimum_gap_duration_s,
        )
        for i, dt_i in enumerate(dt):
            if not np.isfinite(dt_i) or dt_i <= 0:
                continue
            gap_events.append(i)
            if dt_i > gap_threshold:
                large_gap_events.append(i)
                dq_thr = max(
                    cfg.q_gap_abs_threshold_ah,
                    cfg.q_gap_fraction_threshold * phase_capacity if np.isfinite(phase_capacity) else cfg.q_gap_abs_threshold_ah,
                )
                if (
                    np.isfinite(dq[i])
                    and np.isfinite(dv[i])
                    and abs(dq[i]) <= dq_thr
                    and abs(dv[i]) >= cfg.v_gap_threshold_v
                ):
                    possible_relaxation_gap_events.append(i)

    context.update({
        "active_segments": active_segments,
        "low_current_segments": low_segments,
        "observed_rest_segments": observed_rest_segments,
        "internal_low_current_segments": internal_low,
        "terminal_low_current_segments": terminal_low,
        "gap_events": gap_events,
        "large_gap_events": large_gap_events,
        "possible_relaxation_gap_events": possible_relaxation_gap_events,
    })
    return context


def aggregate_duration_features(prefix: str, segments: List[Tuple[int, int]], time_s: np.ndarray, total_duration_s: float) -> Dict[str, Any]:
    durations = np.asarray([segment_duration(time_s, seg) for seg in segments], dtype=float)
    durations = durations[np.isfinite(durations)]
    if durations.size == 0:
        return {
            f"{prefix}_total_duration_s": 0.0,
            f"{prefix}_mean_duration_s": np.nan,
            f"{prefix}_std_duration_s": np.nan,
            f"{prefix}_min_duration_s": np.nan,
            f"{prefix}_max_duration_s": np.nan,
            f"{prefix}_duration_fraction": 0.0,
        }
    return {
        f"{prefix}_total_duration_s": float(np.nansum(durations)),
        f"{prefix}_mean_duration_s": float(np.nanmean(durations)),
        f"{prefix}_std_duration_s": float(np.nanstd(durations)),
        f"{prefix}_min_duration_s": float(np.nanmin(durations)),
        f"{prefix}_max_duration_s": float(np.nanmax(durations)),
        f"{prefix}_duration_fraction": safe_divide(float(np.nansum(durations)), total_duration_s),
    }


def extract_stage_structure_features(row: Dict[str, Any], h1: Dict[str, Any], cfg: ExtractionConfig, context: Dict[str, Any]) -> Dict[str, Any]:
    time_s, voltage_v, current_a, temperature_c, capacity_ah, energy_wh, power_w = get_cycle_arrays(row)
    active_segments: List[Tuple[int, int]] = context["active_segments"]
    low_segments: List[Tuple[int, int]] = context["low_current_segments"]
    rest_segments: List[Tuple[int, int]] = context["observed_rest_segments"]
    internal_low: List[Tuple[int, int]] = context["internal_low_current_segments"]
    terminal_low: List[Tuple[int, int]] = context["terminal_low_current_segments"]
    large_gaps: List[int] = context["large_gap_events"]
    possible_relax_gaps: List[int] = context["possible_relaxation_gap_events"]
    total_duration = finite_or_nan(h1.get("duration_s"))

    feats: Dict[str, Any] = {
        "rest_current_threshold_a": context.get("rest_current_threshold_a", np.nan),
        "one_c_current_a": context.get("one_c_current_a", np.nan),
        "active_segment_count": int(len(active_segments)),
        "low_current_segment_count": int(len(low_segments)),
        "observed_rest_segment_count": int(len(rest_segments)),
        "internal_low_current_segment_count": int(len(internal_low)),
        "terminal_low_current_segment_count": int(len(terminal_low)),
        "stage_count": int(len(active_segments)),
        "has_multistage": int(len(active_segments) > 1),
        "has_low_current_region": int(len(low_segments) > 0),
        "has_internal_rest": int(len(internal_low) > 0),
        "has_terminal_low_current": int(len(terminal_low) > 0),
    }
    feats.update(aggregate_duration_features("active", active_segments, time_s, total_duration))
    feats.update(aggregate_duration_features("observed_rest", rest_segments, time_s, total_duration))

    # Transitions between consecutive active stages
    trans_dv: List[float] = []
    trans_dq: List[float] = []
    trans_dt: List[float] = []
    trans_di: List[float] = []
    current_means: List[float] = []
    durations: List[float] = []
    capacities: List[float] = []
    voltage_spans: List[float] = []
    power_means: List[float] = []

    for seg in active_segments:
        s, e = seg
        cur = current_a[s:e + 1]
        volt = voltage_v[s:e + 1]
        pow_ = power_w[s:e + 1]
        current_means.append(float(np.nanmean(cur)) if np.isfinite(cur).any() else np.nan)
        durations.append(segment_duration(time_s, seg))
        capacities.append(segment_capacity_delta(capacity_ah, seg))
        if np.isfinite(volt).any():
            voltage_spans.append(float(np.nanmax(volt) - np.nanmin(volt)))
        else:
            voltage_spans.append(np.nan)
        power_means.append(float(np.nanmean(pow_)) if np.isfinite(pow_).any() else np.nan)

    for idx in range(len(active_segments) - 1):
        s1, e1 = active_segments[idx]
        s2, e2 = active_segments[idx + 1]
        trans_dv.append(float(voltage_v[s2] - voltage_v[e1]) if np.isfinite(voltage_v[s2]) and np.isfinite(voltage_v[e1]) else np.nan)
        trans_dq.append(float(capacity_ah[s2] - capacity_ah[e1]) if np.isfinite(capacity_ah[s2]) and np.isfinite(capacity_ah[e1]) else np.nan)
        trans_dt.append(float(time_s[s2] - time_s[e1]) if np.isfinite(time_s[s2]) and np.isfinite(time_s[e1]) else np.nan)
        if np.isfinite(current_means[idx]) and np.isfinite(current_means[idx + 1]):
            trans_di.append(float(current_means[idx + 1] - current_means[idx]))
        else:
            trans_di.append(np.nan)

    def agg(prefix: str, values: List[float]) -> Dict[str, Any]:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {
                f"{prefix}_mean": np.nan,
                f"{prefix}_std": np.nan,
                f"{prefix}_max": np.nan,
                f"{prefix}_min": np.nan,
                f"{prefix}_mean_abs": np.nan,
                f"{prefix}_max_abs": np.nan,
                f"{prefix}_sum_abs": 0.0,
                f"{prefix}_first": np.nan,
                f"{prefix}_last": np.nan,
            }
        return {
            f"{prefix}_mean": float(np.nanmean(arr)),
            f"{prefix}_std": float(np.nanstd(arr)),
            f"{prefix}_max": float(np.nanmax(arr)),
            f"{prefix}_min": float(np.nanmin(arr)),
            f"{prefix}_mean_abs": float(np.nanmean(np.abs(arr))),
            f"{prefix}_max_abs": float(np.nanmax(np.abs(arr))),
            f"{prefix}_sum_abs": float(np.nansum(np.abs(arr))),
            f"{prefix}_first": float(arr[0]),
            f"{prefix}_last": float(arr[-1]),
        }

    feats["stage_transition_count"] = int(max(0, len(active_segments) - 1))
    dv_agg = agg("stage_transition_delta_v", trans_dv)
    dq_agg = agg("stage_transition_delta_q", trans_dq)
    dt_agg = agg("stage_transition_delta_t_s", trans_dt)
    feats.update({
        "stage_transition_mean_delta_v": dv_agg["stage_transition_delta_v_mean"],
        "stage_transition_std_delta_v": dv_agg["stage_transition_delta_v_std"],
        "stage_transition_max_delta_v": dv_agg["stage_transition_delta_v_max"],
        "stage_transition_min_delta_v": dv_agg["stage_transition_delta_v_min"],
        "stage_transition_mean_abs_delta_v": dv_agg["stage_transition_delta_v_mean_abs"],
        "stage_transition_max_abs_delta_v": dv_agg["stage_transition_delta_v_max_abs"],
        "stage_transition_sum_abs_delta_v": dv_agg["stage_transition_delta_v_sum_abs"],
        "stage_transition_first_delta_v": dv_agg["stage_transition_delta_v_first"],
        "stage_transition_last_delta_v": dv_agg["stage_transition_delta_v_last"],
        "stage_transition_mean_delta_q": dq_agg["stage_transition_delta_q_mean"],
        "stage_transition_max_abs_delta_q": dq_agg["stage_transition_delta_q_max_abs"],
        "stage_transition_sum_abs_delta_q": dq_agg["stage_transition_delta_q_sum_abs"],
        "stage_transition_mean_delta_t_s": dt_agg["stage_transition_delta_t_s_mean"],
        "stage_transition_max_delta_t_s": dt_agg["stage_transition_delta_t_s_max"],
        "stage_transition_sum_delta_t_s": float(np.nansum(np.asarray(trans_dt, dtype=float))) if trans_dt else 0.0,
    })
    cur_means = np.asarray(current_means, dtype=float)
    cur_means = cur_means[np.isfinite(cur_means)]
    feats["stage_transition_current_mean_range_a"] = float(np.nanmax(cur_means) - np.nanmin(cur_means)) if cur_means.size else np.nan
    feats["stage_transition_current_mean_std_a"] = float(np.nanstd(cur_means)) if cur_means.size else np.nan

    feats["stage_duration_cv"] = coefficient_of_variation(durations)
    feats["stage_capacity_cv"] = coefficient_of_variation(capacities)
    feats["stage_voltage_span_cv"] = coefficient_of_variation(voltage_spans)
    feats["stage_current_mean_cv"] = coefficient_of_variation(current_means)
    feats["stage_power_mean_cv"] = coefficient_of_variation(power_means)

    # Time gap features
    dt = np.diff(time_s) if len(time_s) >= 2 else np.array([], dtype=float)
    dv = np.diff(voltage_v) if len(voltage_v) >= 2 else np.array([], dtype=float)
    dq = np.diff(capacity_ah) if len(capacity_ah) >= 2 else np.array([], dtype=float)
    di = np.diff(current_a) if len(current_a) >= 2 else np.array([], dtype=float)

    large_dt = np.asarray([dt[i] for i in large_gaps if i < len(dt)], dtype=float)
    large_dv = np.asarray([dv[i] for i in large_gaps if i < len(dv)], dtype=float)
    large_dq = np.asarray([dq[i] for i in large_gaps if i < len(dq)], dtype=float)
    large_di = np.asarray([di[i] for i in large_gaps if i < len(di)], dtype=float)

    feats["time_gap_count"] = int(len(dt[np.isfinite(dt)]))
    feats["large_time_gap_count"] = int(len(large_gaps))
    feats["possible_relaxation_gap_count"] = int(len(possible_relax_gaps))
    feats["has_time_gap"] = int(len(dt[np.isfinite(dt)]) > 0)
    feats["has_large_time_gap"] = int(len(large_gaps) > 0)
    feats["has_possible_relaxation_gap"] = int(len(possible_relax_gaps) > 0)

    if large_dt.size:
        feats["time_gap_total_duration_s"] = float(np.nansum(large_dt))
        feats["time_gap_mean_duration_s"] = float(np.nanmean(large_dt))
        feats["time_gap_std_duration_s"] = float(np.nanstd(large_dt))
        feats["time_gap_min_duration_s"] = float(np.nanmin(large_dt))
        feats["time_gap_max_duration_s"] = float(np.nanmax(large_dt))
        feats["time_gap_duration_fraction"] = safe_divide(feats["time_gap_total_duration_s"], total_duration)
    else:
        feats["time_gap_total_duration_s"] = 0.0
        feats["time_gap_mean_duration_s"] = np.nan
        feats["time_gap_std_duration_s"] = np.nan
        feats["time_gap_min_duration_s"] = np.nan
        feats["time_gap_max_duration_s"] = np.nan
        feats["time_gap_duration_fraction"] = 0.0

    def basic_gap(prefix: str, arr: np.ndarray) -> None:
        valid = arr[np.isfinite(arr)]
        if valid.size:
            feats[f"{prefix}_mean"] = float(np.nanmean(valid))
            feats[f"{prefix}_std"] = float(np.nanstd(valid))
            feats[f"{prefix}_min"] = float(np.nanmin(valid))
            feats[f"{prefix}_max"] = float(np.nanmax(valid))
            feats[f"{prefix}_abs_mean"] = float(np.nanmean(np.abs(valid)))
            feats[f"{prefix}_abs_max"] = float(np.nanmax(np.abs(valid)))
            feats[f"{prefix}_abs_sum"] = float(np.nansum(np.abs(valid)))
        else:
            feats[f"{prefix}_mean"] = np.nan
            feats[f"{prefix}_std"] = np.nan
            feats[f"{prefix}_min"] = np.nan
            feats[f"{prefix}_max"] = np.nan
            feats[f"{prefix}_abs_mean"] = np.nan
            feats[f"{prefix}_abs_max"] = np.nan
            feats[f"{prefix}_abs_sum"] = 0.0

    basic_gap("gap_delta_v", large_dv)
    basic_gap("gap_delta_q", large_dq)
    if large_dt.size and large_dv.size:
        rates = large_dv / large_dt[:len(large_dv)]
        rates = rates[np.isfinite(rates)]
        feats["gap_voltage_rate_mean_v_per_s"] = float(np.nanmean(rates)) if rates.size else np.nan
        feats["gap_voltage_rate_max_abs_v_per_s"] = float(np.nanmax(np.abs(rates))) if rates.size else np.nan
    else:
        feats["gap_voltage_rate_mean_v_per_s"] = np.nan
        feats["gap_voltage_rate_max_abs_v_per_s"] = np.nan
    feats["gap_delta_q_fraction_of_phase_capacity"] = safe_divide(feats["gap_delta_q_abs_sum"], h1.get("phase_capacity_ah"))

    before_currents = np.asarray([current_a[i] for i in large_gaps if i < len(current_a)], dtype=float)
    after_currents = np.asarray([current_a[i + 1] for i in large_gaps if i + 1 < len(current_a)], dtype=float)
    current_jumps = large_di
    feats["gap_current_before_mean_a"] = float(np.nanmean(before_currents)) if np.isfinite(before_currents).any() else np.nan
    feats["gap_current_after_mean_a"] = float(np.nanmean(after_currents)) if np.isfinite(after_currents).any() else np.nan
    feats["gap_current_jump_mean_a"] = float(np.nanmean(current_jumps)) if np.isfinite(current_jumps).any() else np.nan
    feats["gap_current_jump_max_abs_a"] = float(np.nanmax(np.abs(current_jumps))) if np.isfinite(current_jumps).any() else np.nan

    return feats


# =============================================================================
# H4 individual event features
# =============================================================================

def empty_active_segment_features(k: int) -> Dict[str, Any]:
    prefix = f"seg{k}"
    keys = [
        "available", "start_time_s", "end_time_s", "duration_s",
        "capacity_start_ah", "capacity_end_ah", "capacity_delta_ah", "capacity_fraction",
        "voltage_start_v", "voltage_end_v", "voltage_min_v", "voltage_max_v",
        "voltage_mean_v", "voltage_delta_v", "voltage_range_v",
        "current_mean_a", "current_std_a", "current_abs_mean_a",
        "temperature_mean_c", "power_mean_w", "energy_wh",
    ]
    return {f"{prefix}_{key}": (0 if key == "available" else np.nan) for key in keys}


def active_segment_features(k: int, seg: Tuple[int, int], arrays: Tuple[np.ndarray, ...], phase_capacity: float) -> Dict[str, Any]:
    time_s, voltage_v, current_a, temperature_c, capacity_ah, energy_wh, power_w = arrays
    s, e = seg
    prefix = f"seg{k}"
    volt = voltage_v[s:e + 1]
    cur = current_a[s:e + 1]
    temp = temperature_c[s:e + 1]
    pow_ = power_w[s:e + 1]
    cap_delta = segment_capacity_delta(capacity_ah, seg)
    energy_delta = energy_wh[e] - energy_wh[s] if e < len(energy_wh) and np.isfinite(energy_wh[e]) and np.isfinite(energy_wh[s]) else np.nan
    return {
        f"{prefix}_available": 1,
        f"{prefix}_start_time_s": float(time_s[s]) if np.isfinite(time_s[s]) else np.nan,
        f"{prefix}_end_time_s": float(time_s[e]) if np.isfinite(time_s[e]) else np.nan,
        f"{prefix}_duration_s": segment_duration(time_s, seg),
        f"{prefix}_capacity_start_ah": float(capacity_ah[s]) if np.isfinite(capacity_ah[s]) else np.nan,
        f"{prefix}_capacity_end_ah": float(capacity_ah[e]) if np.isfinite(capacity_ah[e]) else np.nan,
        f"{prefix}_capacity_delta_ah": cap_delta,
        f"{prefix}_capacity_fraction": safe_divide(cap_delta, phase_capacity),
        f"{prefix}_voltage_start_v": float(voltage_v[s]) if np.isfinite(voltage_v[s]) else np.nan,
        f"{prefix}_voltage_end_v": float(voltage_v[e]) if np.isfinite(voltage_v[e]) else np.nan,
        f"{prefix}_voltage_min_v": float(np.nanmin(volt)) if np.isfinite(volt).any() else np.nan,
        f"{prefix}_voltage_max_v": float(np.nanmax(volt)) if np.isfinite(volt).any() else np.nan,
        f"{prefix}_voltage_mean_v": float(np.nanmean(volt)) if np.isfinite(volt).any() else np.nan,
        f"{prefix}_voltage_delta_v": float(voltage_v[e] - voltage_v[s]) if np.isfinite(voltage_v[e]) and np.isfinite(voltage_v[s]) else np.nan,
        f"{prefix}_voltage_range_v": float(np.nanmax(volt) - np.nanmin(volt)) if np.isfinite(volt).any() else np.nan,
        f"{prefix}_current_mean_a": float(np.nanmean(cur)) if np.isfinite(cur).any() else np.nan,
        f"{prefix}_current_std_a": float(np.nanstd(cur)) if np.isfinite(cur).any() else np.nan,
        f"{prefix}_current_abs_mean_a": float(np.nanmean(np.abs(cur))) if np.isfinite(cur).any() else np.nan,
        f"{prefix}_temperature_mean_c": float(np.nanmean(temp)) if np.isfinite(temp).any() else np.nan,
        f"{prefix}_power_mean_w": float(np.nanmean(pow_)) if np.isfinite(pow_).any() else np.nan,
        f"{prefix}_energy_wh": float(energy_delta) if np.isfinite(energy_delta) else np.nan,
    }


def empty_rest_segment_features(k: int) -> Dict[str, Any]:
    prefix = f"observed_rest{k}"
    keys = [
        "available", "start_time_s", "end_time_s", "duration_s",
        "voltage_start_v", "voltage_end_v", "delta_v", "abs_delta_v",
        "rate_v_per_s", "capacity_change_ah", "current_abs_mean_a",
    ]
    return {f"{prefix}_{key}": (0 if key == "available" else np.nan) for key in keys}


def rest_segment_features(k: int, seg: Tuple[int, int], arrays: Tuple[np.ndarray, ...]) -> Dict[str, Any]:
    time_s, voltage_v, current_a, temperature_c, capacity_ah, energy_wh, power_w = arrays
    s, e = seg
    prefix = f"observed_rest{k}"
    dt = time_s[e] - time_s[s]
    dv = voltage_v[e] - voltage_v[s]
    dq = capacity_ah[e] - capacity_ah[s]
    cur = current_a[s:e + 1]
    return {
        f"{prefix}_available": 1,
        f"{prefix}_start_time_s": float(time_s[s]) if np.isfinite(time_s[s]) else np.nan,
        f"{prefix}_end_time_s": float(time_s[e]) if np.isfinite(time_s[e]) else np.nan,
        f"{prefix}_duration_s": float(dt) if np.isfinite(dt) else np.nan,
        f"{prefix}_voltage_start_v": float(voltage_v[s]) if np.isfinite(voltage_v[s]) else np.nan,
        f"{prefix}_voltage_end_v": float(voltage_v[e]) if np.isfinite(voltage_v[e]) else np.nan,
        f"{prefix}_delta_v": float(dv) if np.isfinite(dv) else np.nan,
        f"{prefix}_abs_delta_v": float(abs(dv)) if np.isfinite(dv) else np.nan,
        f"{prefix}_rate_v_per_s": float(dv / dt) if np.isfinite(dv) and np.isfinite(dt) and dt > 0 else np.nan,
        f"{prefix}_capacity_change_ah": float(dq) if np.isfinite(dq) else np.nan,
        f"{prefix}_current_abs_mean_a": float(np.nanmean(np.abs(cur))) if np.isfinite(cur).any() else np.nan,
    }


def empty_gap_features(k: int) -> Dict[str, Any]:
    prefix = f"gap{k}"
    keys = [
        "available", "index_before", "time_before_s", "time_after_s", "duration_s",
        "delta_v", "abs_delta_v", "delta_q", "abs_delta_q", "delta_i",
        "abs_delta_i", "voltage_rate_v_per_s", "possible_relaxation_gap",
    ]
    return {f"{prefix}_{key}": (0 if key in {"available", "possible_relaxation_gap"} else np.nan) for key in keys}


def gap_event_features(k: int, idx: int, possible_indices: List[int], arrays: Tuple[np.ndarray, ...]) -> Dict[str, Any]:
    time_s, voltage_v, current_a, temperature_c, capacity_ah, energy_wh, power_w = arrays
    prefix = f"gap{k}"
    if idx + 1 >= len(time_s):
        return empty_gap_features(k)
    dt = time_s[idx + 1] - time_s[idx]
    dv = voltage_v[idx + 1] - voltage_v[idx]
    dq = capacity_ah[idx + 1] - capacity_ah[idx]
    di = current_a[idx + 1] - current_a[idx]
    return {
        f"{prefix}_available": 1,
        f"{prefix}_index_before": int(idx),
        f"{prefix}_time_before_s": float(time_s[idx]) if np.isfinite(time_s[idx]) else np.nan,
        f"{prefix}_time_after_s": float(time_s[idx + 1]) if np.isfinite(time_s[idx + 1]) else np.nan,
        f"{prefix}_duration_s": float(dt) if np.isfinite(dt) else np.nan,
        f"{prefix}_delta_v": float(dv) if np.isfinite(dv) else np.nan,
        f"{prefix}_abs_delta_v": float(abs(dv)) if np.isfinite(dv) else np.nan,
        f"{prefix}_delta_q": float(dq) if np.isfinite(dq) else np.nan,
        f"{prefix}_abs_delta_q": float(abs(dq)) if np.isfinite(dq) else np.nan,
        f"{prefix}_delta_i": float(di) if np.isfinite(di) else np.nan,
        f"{prefix}_abs_delta_i": float(abs(di)) if np.isfinite(di) else np.nan,
        f"{prefix}_voltage_rate_v_per_s": float(dv / dt) if np.isfinite(dv) and np.isfinite(dt) and dt > 0 else np.nan,
        f"{prefix}_possible_relaxation_gap": int(idx in possible_indices),
    }


def extract_individual_event_features(row: Dict[str, Any], h1: Dict[str, Any], cfg: ExtractionConfig, context: Dict[str, Any]) -> Dict[str, Any]:
    arrays = get_cycle_arrays(row)
    phase_capacity = finite_or_nan(h1.get("phase_capacity_ah"))
    active_segments: List[Tuple[int, int]] = context["active_segments"]
    rest_segments: List[Tuple[int, int]] = context["observed_rest_segments"]
    large_gaps: List[int] = context["large_gap_events"]
    possible_gaps: List[int] = context["possible_relaxation_gap_events"]
    feats: Dict[str, Any] = {}

    for k in range(1, cfg.max_active_segments + 1):
        if k - 1 < len(active_segments):
            feats.update(active_segment_features(k, active_segments[k - 1], arrays, phase_capacity))
        else:
            feats.update(empty_active_segment_features(k))
    extra_active = active_segments[cfg.max_active_segments:]
    feats["seg_extra_count"] = int(len(extra_active))
    feats["seg_extra_total_duration_s"] = float(np.nansum([segment_duration(arrays[0], s) for s in extra_active])) if extra_active else 0.0
    feats["seg_extra_total_capacity_ah"] = float(np.nansum([segment_capacity_delta(arrays[4], s) for s in extra_active])) if extra_active else 0.0

    for k in range(1, cfg.max_observed_rest_segments + 1):
        if k - 1 < len(rest_segments):
            feats.update(rest_segment_features(k, rest_segments[k - 1], arrays))
        else:
            feats.update(empty_rest_segment_features(k))
    extra_rest = rest_segments[cfg.max_observed_rest_segments:]
    feats["observed_rest_extra_count"] = int(len(extra_rest))
    feats["observed_rest_extra_total_duration_s"] = float(np.nansum([segment_duration(arrays[0], s) for s in extra_rest])) if extra_rest else 0.0

    for k in range(1, cfg.max_gap_events + 1):
        if k - 1 < len(large_gaps):
            feats.update(gap_event_features(k, large_gaps[k - 1], possible_gaps, arrays))
        else:
            feats.update(empty_gap_features(k))
    extra_gaps = large_gaps[cfg.max_gap_events:]
    time_s = arrays[0]
    feats["gap_extra_count"] = int(len(extra_gaps))
    if extra_gaps:
        durations = []
        for idx in extra_gaps:
            if idx + 1 < len(time_s):
                durations.append(time_s[idx + 1] - time_s[idx])
        feats["gap_extra_total_duration_s"] = float(np.nansum(durations)) if durations else 0.0
    else:
        feats["gap_extra_total_duration_s"] = 0.0

    return feats


# =============================================================================
# H5 temporal degradation features
# =============================================================================

def sample_vector_from_row(row: Dict[str, Any], prefix: str, p: int) -> np.ndarray:
    cols = [f"{prefix}_p{p:03d}_{i + 1:03d}" for i in range(p)]
    return np.asarray([row.get(col, np.nan) for col in cols], dtype=float)


def extract_temporal_degradation_features(row: Dict[str, Any], current_features: Dict[str, Any], history: List[Dict[str, Any]], cfg: ExtractionConfig) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    idx = len(history)
    feats["row_order_in_battery_phase"] = int(idx + 1)

    phase_capacity = finite_or_nan(current_features.get("phase_capacity_ah"))
    initial_capacity_row = finite_or_nan(row.get("initial_capacity_ah"))
    if np.isfinite(initial_capacity_row) and initial_capacity_row > 0:
        initial_capacity = initial_capacity_row
    elif history:
        initial_capacity = finite_or_nan(history[0].get("phase_capacity_ah"))
    else:
        initial_capacity = phase_capacity

    feats["capacity_ratio_to_initial"] = safe_divide(phase_capacity, initial_capacity)
    feats["capacity_loss_ah"] = initial_capacity - phase_capacity if np.isfinite(initial_capacity) and np.isfinite(phase_capacity) else np.nan
    feats["capacity_loss_pct"] = safe_divide(feats["capacity_loss_ah"], initial_capacity)

    def prev(offset: int, key: str) -> float:
        if len(history) >= offset:
            return finite_or_nan(history[-offset].get(key))
        return np.nan

    feats["capacity_diff_prev1"] = phase_capacity - prev(1, "phase_capacity_ah") if np.isfinite(phase_capacity) and np.isfinite(prev(1, "phase_capacity_ah")) else np.nan
    feats["capacity_diff_prev5"] = phase_capacity - prev(5, "phase_capacity_ah") if np.isfinite(phase_capacity) and np.isfinite(prev(5, "phase_capacity_ah")) else np.nan
    feats["duration_diff_prev1_s"] = finite_or_nan(current_features.get("duration_s")) - prev(1, "duration_s") if np.isfinite(finite_or_nan(current_features.get("duration_s"))) and np.isfinite(prev(1, "duration_s")) else np.nan
    feats["voltage_end_diff_prev1_v"] = finite_or_nan(current_features.get("voltage_end_v")) - prev(1, "voltage_end_v") if np.isfinite(finite_or_nan(current_features.get("voltage_end_v"))) and np.isfinite(prev(1, "voltage_end_v")) else np.nan
    feats["current_abs_mean_diff_prev1_a"] = finite_or_nan(current_features.get("current_abs_mean_a")) - prev(1, "current_abs_mean_a") if np.isfinite(finite_or_nan(current_features.get("current_abs_mean_a"))) and np.isfinite(prev(1, "current_abs_mean_a")) else np.nan

    recent = history[-4:] + [current_features]
    recent_cycles = [finite_or_nan(x.get("cycle_number")) for x in recent]
    recent_caps = [finite_or_nan(x.get("phase_capacity_ah")) for x in recent]
    recent_durations = [finite_or_nan(x.get("duration_s")) for x in recent]
    feats["capacity_slope_last5"] = linear_slope(recent_cycles, recent_caps)
    feats["duration_slope_last5"] = linear_slope(recent_cycles, recent_durations)

    # Curve distances to initial and previous cycle.
    first = history[0] if history else current_features
    previous = history[-1] if history else None
    for p in cfg.resolutions:
        for prefix in ["VQ", "QV"]:
            cur_vec = sample_vector_from_row(current_features, prefix, p)
            first_vec = sample_vector_from_row(first, prefix, p)
            feats[f"{prefix}_distance_to_initial_p{p:03d}"] = normalized_euclidean_distance(cur_vec, first_vec)
            if previous is not None:
                prev_vec = sample_vector_from_row(previous, prefix, p)
                feats[f"{prefix}_distance_to_prev1_p{p:03d}"] = normalized_euclidean_distance(cur_vec, prev_vec)
            else:
                feats[f"{prefix}_distance_to_prev1_p{p:03d}"] = np.nan

    # Phase-specific useful aliases.
    phase = str(row.get("phase", "")).lower().strip()
    if phase == "charge":
        feats["charge_capacity_ah"] = phase_capacity
        feats["charge_capacity_ratio_to_initial"] = feats["capacity_ratio_to_initial"]
        feats["charge_capacity_loss_ah"] = feats["capacity_loss_ah"]
    elif phase == "discharge":
        feats["discharge_capacity_ah"] = phase_capacity
        feats["discharge_capacity_ratio_to_initial"] = feats["capacity_ratio_to_initial"]
        feats["discharge_capacity_loss_ah"] = feats["capacity_loss_ah"]

    return feats


# =============================================================================
# Per-cycle extraction
# =============================================================================

def extract_cycle_features(row: Dict[str, Any], cfg: ExtractionConfig, history: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    h0 = extract_h0_metadata(row)
    h1 = extract_global_features(row, cfg)
    h2 = extract_curve_features(row, h1, cfg)
    context = detect_stage_context(row, cfg)
    h3 = extract_stage_structure_features(row, h1, cfg, context)
    h4 = extract_individual_event_features(row, h1, cfg, context)

    # H5 uses all currently available scalar features as current_features.
    current_for_h5: Dict[str, Any] = {}
    current_for_h5.update(h0)
    current_for_h5.update(h1)
    current_for_h5.update(h2)
    current_for_h5.update(h3)
    current_for_h5.update(h4)
    h5 = extract_temporal_degradation_features(row, current_for_h5, history, cfg)

    h6 = {}
    h6.update(h0)
    h6.update(h1)
    h6.update(h2)
    h6.update(h3)
    h6.update(h4)
    h6.update(h5)

    # Update history after H5 is computed. The caller appends h6 to history.
    return {
        "H0_metadata": h0,
        "H1_global": h1,
        "H2_curves": h2,
        "H3_stage_structure": h3,
        "H4_individual_events": h4,
        "H5_temporal_degradation": h5,
        "H6_full": h6,
    }


# =============================================================================
# Output selection and validation
# =============================================================================

def max_level_name(feature_level: str) -> str:
    key = feature_level.strip().upper()
    if key not in LEVEL_SHORT_TO_NAME:
        raise ValueError(f"Invalid feature level: {feature_level}")
    return LEVEL_SHORT_TO_NAME[key]


def levels_to_save(cfg: ExtractionConfig) -> List[str]:
    max_level = max_level_name(cfg.feature_level)
    if max_level == "H6_full":
        if cfg.save_intermediate_levels:
            return LEVEL_ORDER.copy()
        return ["H6_full"]
    max_idx = LEVEL_ORDER.index(max_level)
    if cfg.save_intermediate_levels:
        return LEVEL_ORDER[: max_idx + 1]
    return [max_level]


def combine_level_features(level: str, extracted: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    h0 = extracted["H0_metadata"]
    if level == "H6_full":
        return dict(extracted["H6_full"])
    out = dict(h0)
    out.update(extracted[level])
    return out


def remove_vector_columns(df: pd.DataFrame, keep_raw_vectors: bool) -> pd.DataFrame:
    if keep_raw_vectors:
        return df
    drop_cols = [c for c in VECTOR_COLUMNS if c in df.columns]
    if drop_cols:
        return df.drop(columns=drop_cols)
    return df


def validate_feature_block(df: pd.DataFrame, level: str) -> None:
    if df.empty:
        raise ValueError(f"Refusing to save empty feature block for {level}")
    required = ["dataset", "battery_id", "phase", "cycle_number", "remaining_cycles"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Feature block for {level} is missing required columns: {missing}")

    for col in df.columns:
        if col in VECTOR_COLUMNS:
            continue
        if df[col].map(lambda x: isinstance(x, (list, tuple, np.ndarray, pd.Series, dict))).any():
            raise ValueError(f"Column {col} in {level} contains non-scalar values. Use --keep-raw-vectors only for raw vectors.")


def output_parts_dir(cfg: ExtractionConfig, level: str, phase: str, dataset: str, battery_id: str) -> Path:
    return (
        cfg.processed_root
        / level
        / safe_name(phase.lower())
        / safe_name(dataset.lower())
        / safe_name(battery_id)
        / "parts"
    )


def progress_file_for(cfg: ExtractionConfig, phase: str, dataset: str, battery_id: str) -> Path:
    return (
        cfg.processed_root
        / "progress"
        / safe_name(phase.lower())
        / safe_name(dataset.lower())
        / safe_name(battery_id)
        / "progress.json"
    )


def part_file_for(parts_dir: Path, block_id: int, start: int, end: int) -> Path:
    return parts_dir / f"part_{block_id:06d}_{start}_{end}.parquet"


def save_block_parts(
    rows_by_level: Dict[str, List[Dict[str, Any]]],
    cfg: ExtractionConfig,
    phase: str,
    dataset: str,
    battery_id: str,
    block_id: int,
    start: int,
    end: int,
) -> Dict[str, str]:
    saved: Dict[str, str] = {}
    for level, rows in rows_by_level.items():
        if not rows:
            continue
        out_dir = output_parts_dir(cfg, level, phase, dataset, battery_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        part_file = part_file_for(out_dir, block_id, start, end)
        tmp_file = part_file.with_suffix(part_file.suffix + ".tmp")
        df_block = pd.DataFrame(rows)
        df_block = remove_vector_columns(df_block, cfg.keep_raw_vectors)
        validate_feature_block(df_block, level)
        df_block.to_parquet(tmp_file, index=False, engine=cfg.parquet_engine)
        os.replace(tmp_file, part_file)
        saved[level] = str(part_file)
        del df_block
    return saved


def reference_part_exists(cfg: ExtractionConfig, phase: str, dataset: str, battery_id: str, block_id: int, start: int, end: int) -> bool:
    ref_level = max_level_name(cfg.feature_level)
    part = part_file_for(output_parts_dir(cfg, ref_level, phase, dataset, battery_id), block_id, start, end)
    return part.exists() and part.stat().st_size > 0


def load_history_from_existing_part(cfg: ExtractionConfig, phase: str, dataset: str, battery_id: str, block_id: int, start: int, end: int) -> List[Dict[str, Any]]:
    ref_level = "H6_full" if "H6_full" in levels_to_save(cfg) else max_level_name(cfg.feature_level)
    part = part_file_for(output_parts_dir(cfg, ref_level, phase, dataset, battery_id), block_id, start, end)
    if not part.exists():
        return []
    try:
        df = pd.read_parquet(part, engine=cfg.parquet_engine)
        return df.to_dict(orient="records")
    except Exception:
        return []


# =============================================================================
# Processing
# =============================================================================

def prepare_output_roots(cfg: ExtractionConfig) -> None:
    cfg.processed_root.mkdir(parents=True, exist_ok=True)
    cfg.reports_root.mkdir(parents=True, exist_ok=True)
    if cfg.overwrite and cfg.processed_root.exists():
        # Safety guard: remove only the feature_extraction folder, never data/interim/raw.
        shutil.rmtree(cfg.processed_root)
        cfg.processed_root.mkdir(parents=True, exist_ok=True)


def process_group(
    group: pd.DataFrame,
    cfg: ExtractionConfig,
    dataset: str,
    battery_id: str,
    phase: str,
) -> Dict[str, Any]:
    levels = levels_to_save(cfg)
    group = group.sort_values(["cycle_number", "cycle_index"], na_position="last").reset_index(drop=True)
    records = group.to_dict(orient="records")
    n_cycles = len(records)
    total_blocks = int(math.ceil(n_cycles / cfg.batch_size)) if n_cycles else 0
    history: List[Dict[str, Any]] = []

    report: Dict[str, Any] = {
        "dataset": dataset,
        "battery_id": battery_id,
        "phase": phase,
        "n_cycles": int(n_cycles),
        "n_blocks": int(total_blocks),
        "n_blocks_processed": 0,
        "n_blocks_skipped": 0,
        "n_cycles_processed": 0,
        "n_cycles_skipped": 0,
        "levels_saved": levels,
        "errors": [],
    }

    desc = f"{dataset} | {phase} | {battery_id}"
    with tqdm(total=total_blocks, desc=desc, leave=False) as pbar:
        for block_id, start in enumerate(range(0, n_cycles, cfg.batch_size), start=1):
            block = records[start:start + cfg.batch_size]
            end = start + len(block) - 1

            if cfg.resume and reference_part_exists(cfg, phase, dataset, battery_id, block_id, start, end):
                existing_history = load_history_from_existing_part(cfg, phase, dataset, battery_id, block_id, start, end)
                if len(existing_history) == len(block):
                    history.extend(existing_history)
                    report["n_blocks_skipped"] += 1
                    report["n_cycles_skipped"] += len(block)
                    pbar.update(1)
                    continue

            rows_by_level: Dict[str, List[Dict[str, Any]]] = {level: [] for level in levels}
            try:
                for row in block:
                    extracted = extract_cycle_features(row, cfg, history)
                    h6_row = dict(extracted["H6_full"])
                    for level in levels:
                        rows_by_level[level].append(combine_level_features(level, extracted))
                    history.append(h6_row)

                save_block_parts(rows_by_level, cfg, phase, dataset, battery_id, block_id, start, end)
                report["n_blocks_processed"] += 1
                report["n_cycles_processed"] += len(block)

                progress = {
                    "dataset": dataset,
                    "phase": phase,
                    "battery_id": battery_id,
                    "last_completed_part": block_id,
                    "last_completed_start_index": start,
                    "last_completed_end_index": end,
                    "last_completed_cycle_number": finite_or_nan(block[-1].get("cycle_number")) if block else np.nan,
                    "updated_at": now_iso(),
                }
                write_json(progress_file_for(cfg, phase, dataset, battery_id), progress)
            except Exception as exc:
                report["errors"].append({
                    "block_id": block_id,
                    "start": start,
                    "end": end,
                    "error": repr(exc),
                })
                raise
            finally:
                del rows_by_level
                gc.collect()
                pbar.update(1)

    return report


def process_file(file_path: Path, cfg: ExtractionConfig) -> Dict[str, Any]:
    dataset_hint, phase_hint = identify_dataset_phase_from_path(file_path)
    print(f"\nReading: {file_path}")
    df = pd.read_parquet(file_path, engine=cfg.parquet_engine)
    validate_minimum_columns(df, file_path)

    if cfg.dataset_filter.lower() != "all":
        df = df[df["dataset"].astype(str).str.lower().map(lambda x: filter_matches(x, cfg.dataset_filter))]
    if cfg.battery_filter.lower() != "all":
        df = df[df["battery_id"].astype(str).str.lower().map(lambda x: filter_matches(x, cfg.battery_filter))]
    if cfg.phase.lower() != "all":
        df = df[df["phase"].astype(str).str.lower() == cfg.phase.lower()]

    file_report: Dict[str, Any] = {
        "file": str(file_path),
        "dataset_hint": dataset_hint,
        "phase_hint": phase_hint,
        "n_rows_after_filter": int(len(df)),
        "groups": [],
        "started_at": now_iso(),
        "finished_at": None,
    }

    if df.empty:
        file_report["finished_at"] = now_iso()
        return file_report

    group_cols = ["dataset", "battery_id", "phase"]
    total_groups = int(df.groupby(group_cols, dropna=False).ngroups)
    print(f"Groups to process: {total_groups}")

    for (dataset, battery_id, phase), group in tqdm(
        df.groupby(group_cols, dropna=False),
        total=total_groups,
        desc="Groups",
    ):
        dataset_s = str(dataset).lower()
        battery_s = str(battery_id)
        phase_s = str(phase).lower()
        group_report = process_group(group, cfg, dataset_s, battery_s, phase_s)
        file_report["groups"].append(group_report)
        gc.collect()

    file_report["finished_at"] = now_iso()
    return file_report


# =============================================================================
# Merging
# =============================================================================

def append_parquet_streaming(part_files: List[Path], out_file: Path, engine: str) -> Dict[str, Any]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    report = {
        "output_file": str(out_file),
        "n_parts": len(part_files),
        "n_rows": 0,
        "method": "streaming_fastparquet_append" if engine == "fastparquet" else "concat_fallback",
    }

    if not part_files:
        return report

    if engine == "fastparquet":
        try:
            import fastparquet  # type: ignore
            append = False
            for pf in tqdm(part_files, desc=f"Merging {out_file.name}", leave=False):
                df = pd.read_parquet(pf, engine=engine)
                report["n_rows"] += int(len(df))
                fastparquet.write(str(out_file), df, compression="SNAPPY", append=append)
                append = True
                del df
                gc.collect()
            return report
        except Exception as exc:
            report["method"] = f"concat_fallback_after_error: {repr(exc)}"

    # Fallback. Features are scalar and smaller than raw vectors, but this can be memory-heavy.
    frames = []
    for pf in tqdm(part_files, desc=f"Merging {out_file.name}", leave=False):
        df = pd.read_parquet(pf, engine=engine)
        report["n_rows"] += int(len(df))
        frames.append(df)
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged.to_parquet(out_file, index=False, engine=engine)
        del merged, frames
    gc.collect()
    return report


def merge_dataset_outputs(cfg: ExtractionConfig) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for level in levels_to_save(cfg):
        level_root = cfg.processed_root / level
        if not level_root.exists():
            continue
        for phase_dir in sorted([p for p in level_root.iterdir() if p.is_dir()]):
            for dataset_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()]):
                part_files = sorted(dataset_dir.glob("*/parts/part_*.parquet"))
                if not part_files:
                    continue
                out_file = dataset_dir / f"{dataset_dir.name}_{phase_dir.name}_{level}.parquet"
                rep = append_parquet_streaming(part_files, out_file, cfg.parquet_engine)
                rep.update({"level": level, "phase": phase_dir.name, "dataset": dataset_dir.name, "scope": "dataset"})
                reports.append(rep)
    return reports


def merge_global_outputs(cfg: ExtractionConfig) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    merged_root = cfg.processed_root / "merged"
    for level in levels_to_save(cfg):
        level_root = cfg.processed_root / level
        if not level_root.exists():
            continue
        for phase_dir in sorted([p for p in level_root.iterdir() if p.is_dir()]):
            part_files = sorted(phase_dir.glob("*/*/parts/part_*.parquet"))
            if not part_files:
                continue
            out_file = merged_root / f"{phase_dir.name}_{level}_all.parquet"
            rep = append_parquet_streaming(part_files, out_file, cfg.parquet_engine)
            rep.update({"level": level, "phase": phase_dir.name, "scope": "global"})
            reports.append(rep)
    return reports


# =============================================================================
# CLI and main
# =============================================================================

def parse_resolutions(values: Optional[List[str]]) -> List[int]:
    if not values:
        return [3, 9, 15, 30]
    out: List[int] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            p = int(part)
            if p <= 1:
                raise ValueError("Curve resolutions must be greater than 1.")
            out.append(p)
    return sorted(dict.fromkeys(out))


def build_config(args: argparse.Namespace) -> ExtractionConfig:
    data_root = Path(args.data_root)
    processed_root = Path(args.processed_root) if args.processed_root else data_root / "processed" / "feature_extraction"
    reports_root = Path(args.reports_root) if args.reports_root else data_root / "reports" / "feature_extraction"
    return ExtractionConfig(
        data_root=data_root,
        processed_root=processed_root,
        reports_root=reports_root,
        parquet_engine=args.parquet_engine,
        resolutions=parse_resolutions(args.resolutions),
        feature_level=max_level_name(args.feature_level),
        batch_size=int(args.batch_size),
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        merge=bool(args.merge),
        save_intermediate_levels=not bool(args.only_selected_level),
        keep_raw_vectors=bool(args.keep_raw_vectors),
        dataset_filter=normalize_filter(args.dataset_filter),
        battery_filter=normalize_filter(args.battery_filter),
        phase=args.phase.lower(),
        minimum_points=int(args.minimum_points),
        minimum_voltage_span_v=float(args.minimum_voltage_span_v),
        minimum_capacity_span_ah=float(args.minimum_capacity_span_ah),
        rest_c_rate_fraction=float(args.rest_c_rate_fraction),
        minimum_rest_current_a=float(args.minimum_rest_current_a),
        min_rest_duration_s=float(args.min_rest_duration_s),
        min_active_duration_s=float(args.min_active_duration_s),
        gap_factor=float(args.gap_factor),
        minimum_gap_duration_s=float(args.minimum_gap_duration_s),
        q_gap_abs_threshold_ah=float(args.q_gap_abs_threshold_ah),
        q_gap_fraction_threshold=float(args.q_gap_fraction_threshold),
        v_gap_threshold_v=float(args.v_gap_threshold_v),
        max_active_segments=int(args.max_active_segments),
        max_observed_rest_segments=int(args.max_observed_rest_segments),
        max_gap_events=int(args.max_gap_events),
    )


def run_pipeline(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    prepare_output_roots(cfg)

    input_files = find_input_files(cfg.data_root, cfg.phase, cfg.dataset_filter)
    if not input_files:
        raise FileNotFoundError(
            f"No input parquet files found under {cfg.data_root / 'interim'} for phase={cfg.phase}, "
            f"dataset_filter={cfg.dataset_filter}."
        )

    full_report: Dict[str, Any] = {
        "script": "build_hierarchical_features.py",
        "version": "v2_hierarchical_block_resume",
        "started_at": now_iso(),
        "finished_at": None,
        "data_root": str(cfg.data_root),
        "processed_root": str(cfg.processed_root),
        "reports_root": str(cfg.reports_root),
        "parquet_engine": cfg.parquet_engine,
        "phase": cfg.phase,
        "dataset_filter": cfg.dataset_filter,
        "battery_filter": cfg.battery_filter,
        "feature_level": cfg.feature_level,
        "levels_saved": levels_to_save(cfg),
        "resolutions": cfg.resolutions,
        "batch_size": cfg.batch_size,
        "resume": cfg.resume,
        "merge": cfg.merge,
        "input_files": [str(p) for p in input_files],
        "files": [],
        "merge_reports": [],
    }

    print("\nInput files:")
    for path in input_files:
        print(f"  - {path}")

    for file_path in input_files:
        file_report = process_file(file_path, cfg)
        full_report["files"].append(file_report)
        report_path = cfg.reports_root / "hierarchical_feature_extraction_report.partial.json"
        write_json(report_path, full_report)

    if cfg.merge:
        print("\nMerging dataset-level outputs...")
        full_report["merge_reports"].extend(merge_dataset_outputs(cfg))
        print("\nMerging global outputs...")
        full_report["merge_reports"].extend(merge_global_outputs(cfg))

    full_report["finished_at"] = now_iso()
    report_path = cfg.reports_root / "hierarchical_feature_extraction_report.json"
    write_json(report_path, full_report)

    print(f"\nDone. Report saved to: {report_path}")
    print(f"Features saved under: {cfg.processed_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hierarchical charge/discharge features for lithium-ion battery RUL prediction."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Root data folder containing interim/, processed/ and reports/. Default: C:/Artigo Baterias de Litio/Batteries/data",
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default=None,
        help="Output folder for processed features. Default: <data-root>/processed/feature_extraction",
    )
    parser.add_argument(
        "--reports-root",
        type=str,
        default=None,
        help="Output folder for reports. Default: <data-root>/reports/feature_extraction",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "charge", "discharge"],
        help="Phase to process.",
    )
    parser.add_argument(
        "--dataset-filter",
        type=str,
        default="all",
        help="Dataset filter. Use all, or comma-separated substrings such as calce,hust,matr.",
    )
    parser.add_argument(
        "--battery-filter",
        type=str,
        default="all",
        help="Battery filter. Use all, or a substring/comma-separated list of battery ids.",
    )
    parser.add_argument(
        "--feature-level",
        type=str,
        default="H6",
        choices=["H1", "H2", "H3", "H4", "H5", "H6", "all"],
        help="Maximum feature level to compute. H6 computes full H0-H5 feature union.",
    )
    parser.add_argument(
        "--only-selected-level",
        action="store_true",
        help="Save only the selected feature level. By default all levels up to feature-level are saved.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="*",
        default=["3", "9", "15", "30"],
        help="Curve resolutions for V(Q) and Q(V). Example: --resolutions 3 9 15 30",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Cycles per parquet part.")
    parser.add_argument("--resume", action="store_true", help="Skip valid existing part files and continue.")
    parser.add_argument("--overwrite", action="store_true", help="Delete only the feature_extraction processed output before running.")
    parser.add_argument("--merge", action="store_true", help="Create dataset-level and global merged parquet files from parts.")
    parser.add_argument("--keep-raw-vectors", action="store_true", help="Keep raw vector columns in outputs. Not recommended for modeling files.")
    parser.add_argument(
        "--parquet-engine",
        type=str,
        default=DEFAULT_PARQUET_ENGINE,
        choices=["fastparquet", "pyarrow"],
        help="Parquet engine. Default is fastparquet to avoid pyarrow dependency issues.",
    )

    # Validity and curve thresholds.
    parser.add_argument("--minimum-points", type=int, default=20)
    parser.add_argument("--minimum-voltage-span-v", type=float, default=0.05)
    parser.add_argument("--minimum-capacity-span-ah", type=float, default=0.01)

    # Stage/rest thresholds.
    parser.add_argument("--rest-c-rate-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-rest-current-a", type=float, default=0.002)
    parser.add_argument("--min-rest-duration-s", type=float, default=2.0)
    parser.add_argument("--min-active-duration-s", type=float, default=1.0)

    # Gap thresholds.
    parser.add_argument("--gap-factor", type=float, default=5.0)
    parser.add_argument("--minimum-gap-duration-s", type=float, default=10.0)
    parser.add_argument("--q-gap-abs-threshold-ah", type=float, default=0.005)
    parser.add_argument("--q-gap-fraction-threshold", type=float, default=0.01)
    parser.add_argument("--v-gap-threshold-v", type=float, default=0.01)

    # Individual event limits.
    parser.add_argument("--max-active-segments", type=int, default=5)
    parser.add_argument("--max-observed-rest-segments", type=int, default=4)
    parser.add_argument("--max-gap-events", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_pipeline(parse_args())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Re-run with --resume to continue.", file=sys.stderr)
        raise
 