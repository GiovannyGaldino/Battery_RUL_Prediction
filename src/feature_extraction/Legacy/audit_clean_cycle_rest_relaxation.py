"""
Audit clean cycle vectors before hierarchical feature extraction.

Purpose
-------
This script audits *_clean_cycles.parquet files, especially MATR charge cycles,
to determine whether rest/relaxation periods are still present after dataset
adaptation and cleaning.

It checks, per phase-specific cycle:
    - vector integrity;
    - time monotonicity;
    - large time gaps inside the cycle;
    - low-current points using I_rest = rest_c_rate_fraction * 1C;
    - near-constant capacity intervals;
    - voltage drops/recoveries;
    - possible observed rest segments;
    - possible rest gaps between active segments;
    - possible multistage charge/discharge behavior.

Recommended location
--------------------
src/feature_extraction/audit_clean_cycle_rest_relaxation.py

Example commands
----------------
# Audit MATR charge clean cycles
python src/feature_extraction/audit_clean_cycle_rest_relaxation.py ^
  --input-file "D:\\USUÁRIOS\\Marcellus\\OneDrive\\Doutorado\\BATERIAS - Ressubmissão\\Batteries\\data\\interim\\charge\\matr_charge_clean_cycles.parquet" ^
  --output-dir "D:\\USUÁRIOS\\Marcellus\\OneDrive\\Doutorado\\BATERIAS - Ressubmissão\\Batteries\\data\\reports\\feature_extraction\\clean_cycle_audit" ^
  --dataset-filter matr ^
  --phase charge ^
  --rest-c-rate-fraction 0.01

# Try a more tolerant threshold
python src/feature_extraction/audit_clean_cycle_rest_relaxation.py ^
  --input-file "D:\\USUÁRIOS\\Marcellus\\OneDrive\\Doutorado\\BATERIAS - Ressubmissão\\Batteries\\data\\interim\\charge\\matr_charge_clean_cycles.parquet" ^
  --output-dir "D:\\USUÁRIOS\\Marcellus\\OneDrive\\Doutorado\\BATERIAS - Ressubmissão\\Batteries\\data\\reports\\feature_extraction\\clean_cycle_audit" ^
  --dataset-filter matr ^
  --phase charge ^
  --rest-c-rate-fraction 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


PARQUET_ENGINE = "fastparquet"


# =============================================================================
# General utilities
# =============================================================================

def safe_name(value: Any) -> str:
    text = str(value)
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip("_")


def finite_or_nan(value: Any) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def safe_divide(a: float, b: float) -> float:
    if b is None or not np.isfinite(b) or abs(b) < 1e-12:
        return np.nan
    if a is None or not np.isfinite(a):
        return np.nan
    return float(a / b)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if pd.isna(obj):
        return None
    return obj


def as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)

    if isinstance(value, np.ndarray):
        arr = value.astype(float, copy=False)
    elif isinstance(value, (list, tuple, pd.Series)):
        arr = np.asarray(value, dtype=float)
    else:
        try:
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                return np.array([], dtype=float)
        except Exception:
            return np.array([], dtype=float)

    if arr.ndim != 1:
        arr = arr.ravel()

    return arr.astype(float, copy=False)


def get_vector(row: Dict[str, Any], col: str) -> np.ndarray:
    return as_float_array(row.get(col))


def align_vectors(vectors: List[np.ndarray]) -> List[np.ndarray]:
    lengths = [len(v) for v in vectors if len(v) > 0]
    if not lengths:
        return [np.array([], dtype=float) for _ in vectors]

    n = min(lengths)

    out = []
    for v in vectors:
        if len(v) >= n:
            out.append(v[:n].astype(float, copy=False))
        else:
            out.append(np.full(n, np.nan, dtype=float))

    return out


def get_one_c_current_a(row: Dict[str, Any], current_a: np.ndarray) -> float:
    nominal_capacity = finite_or_nan(row.get("nominal_capacity_ah"))
    initial_capacity = finite_or_nan(row.get("initial_capacity_ah"))

    if np.isfinite(nominal_capacity) and nominal_capacity > 0:
        return float(nominal_capacity)

    if np.isfinite(initial_capacity) and initial_capacity > 0:
        return float(initial_capacity)

    current_abs = np.abs(current_a[np.isfinite(current_a)])

    if current_abs.size == 0:
        return np.nan

    i_ref = float(np.nanpercentile(current_abs, 95))

    if np.isfinite(i_ref) and i_ref > 0:
        return i_ref

    return np.nan


def rest_current_threshold_a(
    row: Dict[str, Any],
    current_a: np.ndarray,
    rest_c_rate_fraction: float,
    minimum_floor_a: float,
) -> float:
    one_c = get_one_c_current_a(row, current_a)

    if np.isfinite(one_c) and one_c > 0:
        return float(max(minimum_floor_a, rest_c_rate_fraction * one_c))

    return float(minimum_floor_a)


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


# =============================================================================
# Per-cycle audit
# =============================================================================

def audit_cycle(
    row: Dict[str, Any],
    rest_c_rate_fraction: float,
    minimum_floor_a: float,
    min_rest_duration_s: float,
    min_active_duration_s: float,
    time_gap_factor: float,
    min_gap_duration_s: float,
    dq_low_fraction: float,
    dv_event_threshold_v: float,
) -> Dict[str, Any]:
    dataset = row.get("dataset", None)
    battery_id = row.get("battery_id", None)
    phase = str(row.get("phase", "")).lower().strip()
    cycle_number = row.get("cycle_number", np.nan)
    cycle_index = row.get("cycle_index", np.nan)

    time_s = get_vector(row, "time_s")
    voltage_v = get_vector(row, "voltage_v")
    current_a = get_vector(row, "current_a")
    capacity_ah = get_vector(row, "capacity_ah")
    temperature_c = get_vector(row, "temperature_c")

    time_s, voltage_v, current_a, capacity_ah, temperature_c = align_vectors([
        time_s,
        voltage_v,
        current_a,
        capacity_ah,
        temperature_c,
    ])

    n = len(time_s)

    out: Dict[str, Any] = {
        "dataset": dataset,
        "battery_id": battery_id,
        "phase": phase,
        "cycle_number": cycle_number,
        "cycle_index": cycle_index,
        "n_points_vector": int(n),
        "audit_status": "ok",
    }

    if n < 2:
        out.update(empty_cycle_audit_fields())
        out["audit_status"] = "too_few_points"
        return out

    valid = (
        np.isfinite(time_s)
        & np.isfinite(voltage_v)
        & np.isfinite(current_a)
        & np.isfinite(capacity_ah)
    )

    duration_s = time_s[-1] - time_s[0] if np.isfinite(time_s[-1]) and np.isfinite(time_s[0]) else np.nan
    dt = np.diff(time_s)
    dv = np.diff(voltage_v)
    dq = np.diff(capacity_ah)
    di = np.diff(current_a)

    dt_valid = dt[np.isfinite(dt)]
    positive_dt = dt_valid[dt_valid > 0]

    median_dt = float(np.nanmedian(positive_dt)) if positive_dt.size else np.nan
    max_dt = float(np.nanmax(positive_dt)) if positive_dt.size else np.nan
    min_dt = float(np.nanmin(positive_dt)) if positive_dt.size else np.nan
    time_is_strictly_increasing = bool(np.all(dt_valid > 0)) if dt_valid.size else False

    gap_threshold = np.nan
    if np.isfinite(median_dt):
        gap_threshold = max(float(min_gap_duration_s), float(time_gap_factor * median_dt))

    large_gap_mask = np.isfinite(dt) & np.isfinite(gap_threshold) & (dt >= gap_threshold)
    large_gap_indices = np.where(large_gap_mask)[0]

    one_c_a = get_one_c_current_a(row, current_a)
    i_rest = rest_current_threshold_a(
        row=row,
        current_a=current_a,
        rest_c_rate_fraction=rest_c_rate_fraction,
        minimum_floor_a=minimum_floor_a,
    )

    current_abs = np.abs(current_a)
    low_current_mask = valid & np.isfinite(current_abs) & (current_abs <= i_rest)

    if phase == "charge":
        active_mask = valid & np.isfinite(current_a) & (current_a > i_rest)
    elif phase == "discharge":
        active_mask = valid & np.isfinite(current_a) & (current_a < -i_rest)
    else:
        active_mask = valid & np.isfinite(current_a) & (np.abs(current_a) > i_rest)

    raw_active_segments = find_segments_from_mask(active_mask, min_points=2)
    active_segments = []
    for s, e in raw_active_segments:
        seg_duration = time_s[e] - time_s[s]
        if np.isfinite(seg_duration) and seg_duration >= min_active_duration_s:
            active_segments.append((s, e))

    raw_low_current_segments = find_segments_from_mask(low_current_mask, min_points=2)
    observed_rest_segments: List[Tuple[int, int]] = []
    terminal_low_current_segments: List[Tuple[int, int]] = []
    internal_low_current_segments: List[Tuple[int, int]] = []

    for s, e in raw_low_current_segments:
        seg_duration = time_s[e] - time_s[s]

        if not np.isfinite(seg_duration) or seg_duration < min_rest_duration_s:
            continue

        has_active_before = any(ae < s for _, ae in active_segments)
        has_active_after = any(as_ > e for as_, _ in active_segments)

        is_internal = has_active_before and has_active_after
        is_terminal = has_active_before and not has_active_after

        if is_internal:
            observed_rest_segments.append((s, e))
            internal_low_current_segments.append((s, e))
        elif phase == "discharge" and is_terminal:
            observed_rest_segments.append((s, e))
            terminal_low_current_segments.append((s, e))
        elif phase == "charge" and is_terminal:
            terminal_low_current_segments.append((s, e))

    # Capacity-change diagnostics.
    q_window = float(np.nanmax(capacity_ah[valid]) - np.nanmin(capacity_ah[valid])) if valid.any() else np.nan
    v_window = float(np.nanmax(voltage_v[valid]) - np.nanmin(voltage_v[valid])) if valid.any() else np.nan

    dq_abs = np.abs(dq)
    dq_threshold = np.nan

    if np.isfinite(q_window) and q_window > 0:
        dq_threshold = max(1e-10, dq_low_fraction * q_window)
    else:
        finite_dq = dq_abs[np.isfinite(dq_abs)]
        dq_threshold = max(1e-10, float(np.nanpercentile(finite_dq, 10))) if finite_dq.size else 1e-10

    near_constant_capacity_step_mask = np.isfinite(dq_abs) & (dq_abs <= dq_threshold)

    # Voltage-direction diagnostics.
    voltage_drop_step_mask = np.isfinite(dv) & (dv <= -abs(dv_event_threshold_v))
    voltage_recovery_step_mask = np.isfinite(dv) & (dv >= abs(dv_event_threshold_v))

    # Gap-rest diagnostics between active segments.
    inter_active_gap_segments: List[Dict[str, Any]] = []

    for k in range(len(active_segments) - 1):
        s1, e1 = active_segments[k]
        s2, e2 = active_segments[k + 1]

        if s2 <= e1:
            continue

        gap_dt = time_s[s2] - time_s[e1]
        gap_dv = voltage_v[s2] - voltage_v[e1]
        gap_dq = capacity_ah[s2] - capacity_ah[e1]

        if np.isfinite(gap_dt) and gap_dt >= min_gap_duration_s:
            inter_active_gap_segments.append({
                "gap_index": k + 1,
                "start_index": int(e1),
                "end_index": int(s2),
                "duration_s": float(gap_dt),
                "delta_v": float(gap_dv) if np.isfinite(gap_dv) else np.nan,
                "abs_delta_v": float(abs(gap_dv)) if np.isfinite(gap_dv) else np.nan,
                "delta_q_ah": float(gap_dq) if np.isfinite(gap_dq) else np.nan,
            })

    # Large time gaps anywhere, regardless of active segmentation.
    large_time_gap_events: List[Dict[str, Any]] = []
    for idx in large_gap_indices[:20]:
        i = int(idx)
        large_time_gap_events.append({
            "index_before_gap": i,
            "index_after_gap": i + 1,
            "duration_s": float(time_s[i + 1] - time_s[i]),
            "delta_v": float(voltage_v[i + 1] - voltage_v[i]) if np.isfinite(voltage_v[i + 1] - voltage_v[i]) else np.nan,
            "delta_q_ah": float(capacity_ah[i + 1] - capacity_ah[i]) if np.isfinite(capacity_ah[i + 1] - capacity_ah[i]) else np.nan,
            "current_before_a": float(current_a[i]) if np.isfinite(current_a[i]) else np.nan,
            "current_after_a": float(current_a[i + 1]) if np.isfinite(current_a[i + 1]) else np.nan,
        })

    observed_rest_durations = []
    observed_rest_delta_v = []
    observed_rest_delta_q = []

    for s, e in observed_rest_segments:
        observed_rest_durations.append(time_s[e] - time_s[s])
        observed_rest_delta_v.append(voltage_v[e] - voltage_v[s])
        observed_rest_delta_q.append(capacity_ah[e] - capacity_ah[s])

    gap_rest_durations = [g["duration_s"] for g in inter_active_gap_segments]
    gap_rest_delta_v = [g["delta_v"] for g in inter_active_gap_segments]
    gap_rest_delta_q = [g["delta_q_ah"] for g in inter_active_gap_segments]

    out.update({
        "duration_s_vector": float(duration_s) if np.isfinite(duration_s) else np.nan,
        "valid_common_points": int(valid.sum()),
        "valid_common_fraction": safe_divide(float(valid.sum()), float(n)),
        "time_is_strictly_increasing": int(time_is_strictly_increasing),
        "median_dt_s": median_dt,
        "min_dt_s": min_dt,
        "max_dt_s": max_dt,
        "max_dt_to_median_dt_ratio": safe_divide(max_dt, median_dt),
        "gap_threshold_s": gap_threshold,
        "large_time_gap_count": int(len(large_gap_indices)),
        "has_large_time_gap": int(len(large_gap_indices) > 0),
        "one_c_current_a": float(one_c_a) if np.isfinite(one_c_a) else np.nan,
        "rest_current_threshold_a": float(i_rest) if np.isfinite(i_rest) else np.nan,
        "rest_c_rate_fraction": float(rest_c_rate_fraction),
        "current_abs_min_a": float(np.nanmin(current_abs[np.isfinite(current_abs)])) if np.isfinite(current_abs).any() else np.nan,
        "current_abs_mean_a": float(np.nanmean(current_abs[np.isfinite(current_abs)])) if np.isfinite(current_abs).any() else np.nan,
        "current_abs_median_a": float(np.nanmedian(current_abs[np.isfinite(current_abs)])) if np.isfinite(current_abs).any() else np.nan,
        "current_abs_max_a": float(np.nanmax(current_abs[np.isfinite(current_abs)])) if np.isfinite(current_abs).any() else np.nan,
        "low_current_point_count": int(low_current_mask.sum()),
        "low_current_point_fraction": safe_divide(float(low_current_mask.sum()), float(valid.sum())),
        "raw_low_current_segment_count": int(len(raw_low_current_segments)),
        "internal_low_current_segment_count": int(len(internal_low_current_segments)),
        "terminal_low_current_segment_count": int(len(terminal_low_current_segments)),
        "active_segment_count": int(len(active_segments)),
        "has_multiple_active_segments": int(len(active_segments) > 1),
        "observed_rest_segment_count": int(len(observed_rest_segments)),
        "has_observed_rest": int(len(observed_rest_segments) > 0),
        "observed_rest_total_duration_s": float(np.nansum(observed_rest_durations)) if observed_rest_durations else 0.0,
        "observed_rest_mean_duration_s": float(np.nanmean(observed_rest_durations)) if observed_rest_durations else np.nan,
        "observed_rest_max_duration_s": float(np.nanmax(observed_rest_durations)) if observed_rest_durations else np.nan,
        "observed_rest_mean_delta_v": float(np.nanmean(observed_rest_delta_v)) if observed_rest_delta_v else np.nan,
        "observed_rest_max_abs_delta_v": float(np.nanmax(np.abs(observed_rest_delta_v))) if observed_rest_delta_v else np.nan,
        "observed_rest_total_abs_delta_v": float(np.nansum(np.abs(observed_rest_delta_v))) if observed_rest_delta_v else 0.0,
        "observed_rest_mean_delta_q_ah": float(np.nanmean(observed_rest_delta_q)) if observed_rest_delta_q else np.nan,
        "gap_rest_segment_count": int(len(inter_active_gap_segments)),
        "has_gap_rest": int(len(inter_active_gap_segments) > 0),
        "gap_rest_total_duration_s": float(np.nansum(gap_rest_durations)) if gap_rest_durations else 0.0,
        "gap_rest_mean_duration_s": float(np.nanmean(gap_rest_durations)) if gap_rest_durations else np.nan,
        "gap_rest_max_duration_s": float(np.nanmax(gap_rest_durations)) if gap_rest_durations else np.nan,
        "gap_rest_mean_delta_v": float(np.nanmean(gap_rest_delta_v)) if gap_rest_delta_v else np.nan,
        "gap_rest_max_abs_delta_v": float(np.nanmax(np.abs(gap_rest_delta_v))) if gap_rest_delta_v else np.nan,
        "gap_rest_total_abs_delta_v": float(np.nansum(np.abs(gap_rest_delta_v))) if gap_rest_delta_v else 0.0,
        "gap_rest_mean_delta_q_ah": float(np.nanmean(gap_rest_delta_q)) if gap_rest_delta_q else np.nan,
        "capacity_window_ah_vector": q_window,
        "voltage_window_v_vector": v_window,
        "dq_low_threshold_ah": dq_threshold,
        "near_constant_capacity_step_count": int(near_constant_capacity_step_mask.sum()),
        "near_constant_capacity_step_fraction": safe_divide(float(near_constant_capacity_step_mask.sum()), float(len(dq))),
        "voltage_drop_step_count": int(voltage_drop_step_mask.sum()),
        "voltage_recovery_step_count": int(voltage_recovery_step_mask.sum()),
        "possible_rest_by_current": int(len(observed_rest_segments) > 0),
        "possible_rest_by_time_gap": int(len(inter_active_gap_segments) > 0 or len(large_gap_indices) > 0),
        "possible_multistage_operation": int(len(active_segments) > 1),
        "large_time_gap_events_preview": large_time_gap_events,
        "gap_rest_events_preview": inter_active_gap_segments[:20],
    })

    # Preserve first four observed rests as flat fields.
    for k in range(1, 5):
        if k <= len(observed_rest_segments):
            s, e = observed_rest_segments[k - 1]
            dt_seg = time_s[e] - time_s[s]
            dv_seg = voltage_v[e] - voltage_v[s]
            dq_seg = capacity_ah[e] - capacity_ah[s]
            out.update({
                f"observed_rest{k}_available": 1,
                f"observed_rest{k}_duration_s": float(dt_seg) if np.isfinite(dt_seg) else np.nan,
                f"observed_rest{k}_delta_v": float(dv_seg) if np.isfinite(dv_seg) else np.nan,
                f"observed_rest{k}_abs_delta_v": float(abs(dv_seg)) if np.isfinite(dv_seg) else np.nan,
                f"observed_rest{k}_delta_q_ah": float(dq_seg) if np.isfinite(dq_seg) else np.nan,
            })
        else:
            out.update({
                f"observed_rest{k}_available": 0,
                f"observed_rest{k}_duration_s": np.nan,
                f"observed_rest{k}_delta_v": np.nan,
                f"observed_rest{k}_abs_delta_v": np.nan,
                f"observed_rest{k}_delta_q_ah": np.nan,
            })

    # Preserve first four gap rests as flat fields.
    for k in range(1, 5):
        if k <= len(inter_active_gap_segments):
            g = inter_active_gap_segments[k - 1]
            out.update({
                f"gap_rest{k}_available": 1,
                f"gap_rest{k}_duration_s": g["duration_s"],
                f"gap_rest{k}_delta_v": g["delta_v"],
                f"gap_rest{k}_abs_delta_v": g["abs_delta_v"],
                f"gap_rest{k}_delta_q_ah": g["delta_q_ah"],
            })
        else:
            out.update({
                f"gap_rest{k}_available": 0,
                f"gap_rest{k}_duration_s": np.nan,
                f"gap_rest{k}_delta_v": np.nan,
                f"gap_rest{k}_abs_delta_v": np.nan,
                f"gap_rest{k}_delta_q_ah": np.nan,
            })

    return out


def empty_cycle_audit_fields() -> Dict[str, Any]:
    fields = {
        "duration_s_vector": np.nan,
        "valid_common_points": 0,
        "valid_common_fraction": np.nan,
        "time_is_strictly_increasing": 0,
        "median_dt_s": np.nan,
        "min_dt_s": np.nan,
        "max_dt_s": np.nan,
        "max_dt_to_median_dt_ratio": np.nan,
        "gap_threshold_s": np.nan,
        "large_time_gap_count": 0,
        "has_large_time_gap": 0,
        "one_c_current_a": np.nan,
        "rest_current_threshold_a": np.nan,
        "rest_c_rate_fraction": np.nan,
        "current_abs_min_a": np.nan,
        "current_abs_mean_a": np.nan,
        "current_abs_median_a": np.nan,
        "current_abs_max_a": np.nan,
        "low_current_point_count": 0,
        "low_current_point_fraction": np.nan,
        "raw_low_current_segment_count": 0,
        "internal_low_current_segment_count": 0,
        "terminal_low_current_segment_count": 0,
        "active_segment_count": 0,
        "has_multiple_active_segments": 0,
        "observed_rest_segment_count": 0,
        "has_observed_rest": 0,
        "observed_rest_total_duration_s": 0.0,
        "observed_rest_mean_duration_s": np.nan,
        "observed_rest_max_duration_s": np.nan,
        "observed_rest_mean_delta_v": np.nan,
        "observed_rest_max_abs_delta_v": np.nan,
        "observed_rest_total_abs_delta_v": 0.0,
        "observed_rest_mean_delta_q_ah": np.nan,
        "gap_rest_segment_count": 0,
        "has_gap_rest": 0,
        "gap_rest_total_duration_s": 0.0,
        "gap_rest_mean_duration_s": np.nan,
        "gap_rest_max_duration_s": np.nan,
        "gap_rest_mean_delta_v": np.nan,
        "gap_rest_max_abs_delta_v": np.nan,
        "gap_rest_total_abs_delta_v": 0.0,
        "gap_rest_mean_delta_q_ah": np.nan,
        "capacity_window_ah_vector": np.nan,
        "voltage_window_v_vector": np.nan,
        "dq_low_threshold_ah": np.nan,
        "near_constant_capacity_step_count": 0,
        "near_constant_capacity_step_fraction": np.nan,
        "voltage_drop_step_count": 0,
        "voltage_recovery_step_count": 0,
        "possible_rest_by_current": 0,
        "possible_rest_by_time_gap": 0,
        "possible_multistage_operation": 0,
        "large_time_gap_events_preview": [],
        "gap_rest_events_preview": [],
    }

    for k in range(1, 5):
        fields.update({
            f"observed_rest{k}_available": 0,
            f"observed_rest{k}_duration_s": np.nan,
            f"observed_rest{k}_delta_v": np.nan,
            f"observed_rest{k}_abs_delta_v": np.nan,
            f"observed_rest{k}_delta_q_ah": np.nan,
            f"gap_rest{k}_available": 0,
            f"gap_rest{k}_duration_s": np.nan,
            f"gap_rest{k}_delta_v": np.nan,
            f"gap_rest{k}_abs_delta_v": np.nan,
            f"gap_rest{k}_delta_q_ah": np.nan,
        })

    return fields


# =============================================================================
# Summary and reporting
# =============================================================================

def summarize_audit(df_audit: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "n_cycles": int(len(df_audit)),
    }

    if len(df_audit) == 0:
        return report

    if "dataset" in df_audit.columns:
        report["datasets"] = sorted(df_audit["dataset"].astype(str).dropna().unique().tolist())

    if "battery_id" in df_audit.columns:
        report["n_batteries"] = int(df_audit["battery_id"].astype(str).nunique())
        report["cycles_by_battery_top20"] = (
            df_audit.groupby("battery_id").size().sort_values(ascending=False).head(20).astype(int).to_dict()
        )

    binary_cols = [
        "time_is_strictly_increasing",
        "has_large_time_gap",
        "has_multiple_active_segments",
        "has_observed_rest",
        "has_gap_rest",
        "possible_rest_by_current",
        "possible_rest_by_time_gap",
        "possible_multistage_operation",
    ]

    for col in binary_cols:
        if col in df_audit.columns:
            values = pd.to_numeric(df_audit[col], errors="coerce")
            report[col] = {
                "count_1": int((values == 1).sum()),
                "count_0": int((values == 0).sum()),
                "rate_1": float((values == 1).mean()) if len(values) else None,
            }

    numeric_cols = [
        "n_points_vector",
        "duration_s_vector",
        "median_dt_s",
        "max_dt_s",
        "max_dt_to_median_dt_ratio",
        "large_time_gap_count",
        "one_c_current_a",
        "rest_current_threshold_a",
        "low_current_point_fraction",
        "raw_low_current_segment_count",
        "internal_low_current_segment_count",
        "terminal_low_current_segment_count",
        "active_segment_count",
        "observed_rest_segment_count",
        "observed_rest_total_duration_s",
        "observed_rest_mean_delta_v",
        "observed_rest_max_abs_delta_v",
        "gap_rest_segment_count",
        "gap_rest_total_duration_s",
        "gap_rest_mean_delta_v",
        "gap_rest_max_abs_delta_v",
        "capacity_window_ah_vector",
        "voltage_window_v_vector",
        "near_constant_capacity_step_fraction",
        "voltage_drop_step_count",
        "voltage_recovery_step_count",
    ]

    report["numeric_summary"] = {}

    for col in numeric_cols:
        if col in df_audit.columns:
            values = pd.to_numeric(df_audit[col], errors="coerce")
            report["numeric_summary"][col] = {
                "non_nan_count": int(values.notna().sum()),
                "nan_count": int(values.isna().sum()),
                "min": float(values.min()) if values.notna().any() else None,
                "mean": float(values.mean()) if values.notna().any() else None,
                "median": float(values.median()) if values.notna().any() else None,
                "max": float(values.max()) if values.notna().any() else None,
            }

    # Interpretive conclusion flags.
    n_observed_rest = int((pd.to_numeric(df_audit.get("has_observed_rest", pd.Series([], dtype=float)), errors="coerce") == 1).sum())
    n_gap_rest = int((pd.to_numeric(df_audit.get("has_gap_rest", pd.Series([], dtype=float)), errors="coerce") == 1).sum())
    n_multi_active = int((pd.to_numeric(df_audit.get("has_multiple_active_segments", pd.Series([], dtype=float)), errors="coerce") == 1).sum())
    n_large_gap = int((pd.to_numeric(df_audit.get("has_large_time_gap", pd.Series([], dtype=float)), errors="coerce") == 1).sum())
    n_low_current = int((pd.to_numeric(df_audit.get("low_current_point_count", pd.Series([], dtype=float)), errors="coerce").fillna(0) > 0).sum())

    interpretation = []

    if n_observed_rest > 0:
        interpretation.append(
            "Observed low-current rest/relaxation points are present in the clean vectors. "
            "Feature extraction should be able to compute observed rest features."
        )
    else:
        interpretation.append(
            "No observed low-current rest/relaxation segments were detected with the selected threshold."
        )

    if n_gap_rest > 0 or n_large_gap > 0:
        interpretation.append(
            "Time-gap or inter-active-gap rest candidates are present. "
            "The feature pipeline should include gap_rest/inter_segment_relaxation features."
        )

    if n_multi_active > 0 and n_observed_rest == 0 and n_gap_rest == 0 and n_large_gap == 0:
        interpretation.append(
            "Multiple active segments exist, but neither low-current rest nor time-gap rest was detected. "
            "This suggests that the original cleaning/adaptation may have removed intermediate rest samples "
            "or segmented the cycle before preserving the relaxation interval."
        )

    if n_low_current == 0:
        interpretation.append(
            "No low-current points were found. If visual plots show rest periods, the rest information was likely removed "
            "or not represented by current samples in the clean file."
        )

    report["interpretation"] = interpretation

    return report


def save_outputs(
    df_audit: pd.DataFrame,
    summary: Dict[str, Any],
    output_dir: Path,
    output_prefix: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{output_prefix}_cycle_level_audit.csv"
    parquet_path = output_dir / f"{output_prefix}_cycle_level_audit.parquet"
    json_path = output_dir / f"{output_prefix}_summary.json"

    preview_cols_to_drop = [
        "large_time_gap_events_preview",
        "gap_rest_events_preview",
    ]

    df_csv = df_audit.drop(columns=[c for c in preview_cols_to_drop if c in df_audit.columns], errors="ignore")

    df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df_audit.to_parquet(parquet_path, index=False, engine=PARQUET_ENGINE)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2, ensure_ascii=False)

    return {
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "json": str(json_path),
    }


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit clean battery cycle files for rest/relaxation preservation."
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default=r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\interim\charge\matr_charge_clean_cycles.parquet",
        help="Clean cycle parquet file to audit.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\reports\feature_extraction\clean_cycle_audit",
        help="Directory where audit files will be saved.",
    )

    parser.add_argument(
        "--dataset-filter",
        type=str,
        default=None,
        help="Optional dataset filter, e.g. matr, calce, hust.",
    )

    parser.add_argument(
        "--battery-filter",
        type=str,
        default=None,
        help="Optional substring filter for battery_id.",
    )

    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        choices=["charge", "discharge"],
        help="Optional phase filter.",
    )

    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional maximum number of cycles to audit for quick tests.",
    )

    parser.add_argument(
        "--rest-c-rate-fraction",
        type=float,
        default=0.01,
        help="Low-current threshold as fraction of 1C. Default: 0.01.",
    )

    parser.add_argument(
        "--minimum-floor-a",
        type=float,
        default=0.002,
        help="Minimum low-current threshold in amperes. Default: 0.002 A.",
    )

    parser.add_argument(
        "--min-rest-duration-s",
        type=float,
        default=2.0,
        help="Minimum duration for observed low-current rest segment.",
    )

    parser.add_argument(
        "--min-active-duration-s",
        type=float,
        default=1.0,
        help="Minimum duration for active segment.",
    )

    parser.add_argument(
        "--time-gap-factor",
        type=float,
        default=10.0,
        help="Large gap threshold multiplier: max(min_gap_duration_s, factor * median_dt).",
    )

    parser.add_argument(
        "--min-gap-duration-s",
        type=float,
        default=5.0,
        help="Minimum absolute duration in seconds for a time gap.",
    )

    parser.add_argument(
        "--dq-low-fraction",
        type=float,
        default=0.001,
        help="Threshold for near-constant capacity steps as fraction of cycle capacity window.",
    )

    parser.add_argument(
        "--dv-event-threshold-v",
        type=float,
        default=0.001,
        help="Voltage event threshold in volts for drop/recovery step counts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Reading clean cycle file:\n{input_file}")

    df = pd.read_parquet(input_file, engine=PARQUET_ENGINE)

    if args.dataset_filter:
        key = args.dataset_filter.lower()
        if "dataset" in df.columns:
            df = df[df["dataset"].astype(str).str.lower().str.contains(key, na=False)]
        else:
            print("[WARN] dataset column not found. Dataset filter ignored.")

    if args.battery_filter:
        key = args.battery_filter.lower()
        if "battery_id" in df.columns:
            df = df[df["battery_id"].astype(str).str.lower().str.contains(key, na=False)]
        else:
            print("[WARN] battery_id column not found. Battery filter ignored.")

    if args.phase:
        if "phase" in df.columns:
            df = df[df["phase"].astype(str).str.lower().eq(args.phase.lower())]
        else:
            print("[WARN] phase column not found. Phase filter ignored.")

    if args.max_cycles is not None and args.max_cycles > 0:
        df = df.head(args.max_cycles)

    print(f"Cycles selected for audit: {len(df)}")

    audit_rows: List[Dict[str, Any]] = []

    for row in tqdm(df.to_dict(orient="records"), desc="Auditing clean cycles"):
        audit_rows.append(audit_cycle(
            row=row,
            rest_c_rate_fraction=args.rest_c_rate_fraction,
            minimum_floor_a=args.minimum_floor_a,
            min_rest_duration_s=args.min_rest_duration_s,
            min_active_duration_s=args.min_active_duration_s,
            time_gap_factor=args.time_gap_factor,
            min_gap_duration_s=args.min_gap_duration_s,
            dq_low_fraction=args.dq_low_fraction,
            dv_event_threshold_v=args.dv_event_threshold_v,
        ))

    df_audit = pd.DataFrame(audit_rows)
    summary = summarize_audit(df_audit)

    input_stem = safe_name(input_file.stem)
    filters = []
    if args.dataset_filter:
        filters.append(safe_name(args.dataset_filter.lower()))
    if args.phase:
        filters.append(safe_name(args.phase.lower()))
    if args.battery_filter:
        filters.append(safe_name(args.battery_filter.lower()))

    filter_suffix = "_".join(filters) if filters else "all"
    threshold_suffix = f"rest{args.rest_c_rate_fraction:g}C".replace(".", "p")
    output_prefix = f"{input_stem}_{filter_suffix}_{threshold_suffix}"

    paths = save_outputs(
        df_audit=df_audit,
        summary=summary,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )

    print("\nAudit completed.")
    print(f"CSV:     {paths['csv']}")
    print(f"Parquet: {paths['parquet']}")
    print(f"JSON:    {paths['json']}")

    print("\nKey summary:")
    for key in [
        "has_observed_rest",
        "has_gap_rest",
        "has_large_time_gap",
        "has_multiple_active_segments",
        "possible_rest_by_current",
        "possible_rest_by_time_gap",
        "possible_multistage_operation",
    ]:
        if key in summary:
            item = summary[key]
            print(f"  {key}: count_1={item.get('count_1')}, rate_1={item.get('rate_1')}")

    print("\nInterpretation:")
    for msg in summary.get("interpretation", []):
        print(f"  - {msg}")


if __name__ == "__main__":
    main()
