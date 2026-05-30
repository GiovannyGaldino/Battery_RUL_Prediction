r"""
MATR/Severson dataset adapter - extraction step (version 2.0).

This version reads the raw MATR MATLAB v7.3 files and extracts contiguous
phase-specific cycle segments from each complete electrochemical cycle.

Segmentation rule implemented for this project revision
-------------------------------------------------------
The raw MATR cycle vector contains the complete charge + discharge record.
For each cycle:

    charge segment    = samples from the beginning of the raw cycle up to,
                        but not including, the first discharge sample.
    discharge segment = samples starting at the first point where current
                        becomes lower than -0.5 A by default.

This replaces the previous mask-based segmentation that selected all samples
with I > epsilon for charge and all samples with I < -epsilon for discharge.
The previous approach could discard low-current/rest samples and could create
non-contiguous segments. This script preserves the contiguous morphology of the
cycle before and after the discharge transition.

Default raw-data path:
    <base_dir>/data/raw/MATR/*.mat

Outputs:
    data/interim/matr/extracted/matr_charge_extracted_cycles.parquet
    data/interim/matr/extracted/matr_discharge_extracted_cycles.parquet
    data/interim/matr/extracted/matr_extraction_report.csv
    data/interim/matr/extracted/matr_extraction_report.json

Current convention in extracted rows:
    charge    -> current_a is positive/non-negative by absolute-value convention
    discharge -> current_a is negative/non-positive by negative absolute-value convention

Capacity convention:
    capacity_ah is non-negative and starts from zero inside the extracted phase.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable | None = None, **kwargs):
        return iterable if iterable is not None else (lambda x: x)


DATASET = "MATR"
ADAPTER_VERSION = "matr_adapter_v2_contiguous_current_split_2026-05-26"

DEFAULT_BASE_DIR = Path(
    r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries"
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_base_dir(base_dir_arg: str | None) -> Path:
    if base_dir_arg:
        return Path(base_dir_arg).expanduser().resolve()
    if DEFAULT_BASE_DIR.exists():
        return DEFAULT_BASE_DIR
    return Path.cwd().resolve()


def relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except Exception:
        return str(path)


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        value = float(np.asarray(value).squeeze())
        return value if math.isfinite(value) else default
    except Exception:
        return default


def decode_uint16_string(arr: Any) -> str:
    """Decode MATLAB uint16 string arrays."""
    try:
        flat = np.asarray(arr).squeeze().astype(np.uint16).flatten()
        return "".join(chr(int(x)) for x in flat if int(x) > 0)
    except Exception:
        try:
            return str(arr)
        except Exception:
            return ""


def read_mat_ref_array(handle: h5py.File, ref: Any) -> np.ndarray:
    """Read a numeric array stored behind an HDF5 MATLAB reference."""
    try:
        obj = handle[ref]
        arr = np.asarray(obj[()]).squeeze()
        return np.atleast_1d(arr).astype(float)
    except Exception:
        return np.array([], dtype=float)


def read_mat_ref_scalar(handle: h5py.File, ref: Any) -> float:
    try:
        arr = read_mat_ref_array(handle, ref)
        if arr.size == 0:
            return np.nan
        return finite_float(arr[0])
    except Exception:
        return np.nan


def read_mat_ref_string(handle: h5py.File, ref: Any) -> str:
    try:
        obj = handle[ref]
        arr = np.asarray(obj[()]).squeeze()
        return decode_uint16_string(arr)
    except Exception:
        return ""


def get_reference(dataset: h5py.Dataset, index: int) -> Any:
    """Return a reference from a MATLAB cell-array-like HDF5 dataset."""
    candidates: list[Any] = []
    try:
        candidates.append(dataset[index, 0])
    except Exception:
        pass
    try:
        candidates.append(dataset[0, index])
    except Exception:
        pass
    try:
        candidates.append(dataset[index])
    except Exception:
        pass

    if not candidates:
        raise IndexError(f"Could not read reference at index {index}.")
    return candidates[0]


def read_cycle_vector(
    handle: h5py.File,
    cycle_group: h5py.Group,
    key: str,
    cycle_index: int,
) -> np.ndarray:
    if key not in cycle_group:
        return np.array([], dtype=float)
    try:
        ref = get_reference(cycle_group[key], cycle_index)
        return read_mat_ref_array(handle, ref)
    except Exception:
        return np.array([], dtype=float)


def read_summary_array(summary_group: h5py.Group, key: str) -> np.ndarray:
    if key not in summary_group:
        return np.array([], dtype=float)
    try:
        return np.atleast_1d(np.asarray(summary_group[key][()]).squeeze()).astype(float)
    except Exception:
        return np.array([], dtype=float)


def summary_value(summary_array: np.ndarray, index: int, default: float = np.nan) -> float:
    try:
        if summary_array.size == 0 or index >= len(summary_array):
            return default
        return finite_float(summary_array[index], default=default)
    except Exception:
        return default


def safe_vector(x: Any) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float).squeeze()
        return np.atleast_1d(arr)
    except Exception:
        return np.array([], dtype=float)


def integrate_capacity_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    """Compute accumulated non-negative phase capacity from current."""
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
    cap = np.cumsum(np.abs(current_a) * dt) / 3600.0
    if cap.size:
        cap = cap - cap[0]
    return cap


def has_usable_capacity_vector(capacity_ah: np.ndarray) -> bool:
    capacity_ah = safe_vector(capacity_ah)
    finite = capacity_ah[np.isfinite(capacity_ah)]
    return finite.size >= 2 and float(np.nanmax(finite) - np.nanmin(finite)) > 1e-10


def normalize_capacity_vector(capacity_ah: np.ndarray) -> np.ndarray:
    capacity_ah = safe_vector(capacity_ah).astype(float)
    if capacity_ah.size == 0:
        return capacity_ah
    finite = np.isfinite(capacity_ah)
    if not finite.any():
        return capacity_ah
    offset = np.nanmin(capacity_ah[finite])
    capacity_ah = capacity_ah - offset
    capacity_ah[capacity_ah < 0] = 0.0
    if capacity_ah.size:
        capacity_ah = capacity_ah - capacity_ah[0]
    return capacity_ah


def align_complete_cycle(
    *,
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    temperature_c: np.ndarray,
    q_charge_ah: np.ndarray,
    q_discharge_ah: np.ndarray,
) -> tuple[dict[str, np.ndarray] | None, list[str], str]:
    """Align complete-cycle vectors while preserving raw contiguous order."""
    warnings: list[str] = []

    time_s = safe_vector(time_s)
    voltage_v = safe_vector(voltage_v)
    current_a = safe_vector(current_a)
    temperature_c = safe_vector(temperature_c)
    q_charge_ah = safe_vector(q_charge_ah)
    q_discharge_ah = safe_vector(q_discharge_ah)

    if time_s.size == 0 or voltage_v.size == 0 or current_a.size == 0:
        return None, warnings, "missing_required_vectors"

    n = min(len(time_s), len(voltage_v), len(current_a))
    if n < 2:
        return None, warnings, "insufficient_required_vector_length"

    time_s = time_s[:n]
    voltage_v = voltage_v[:n]
    current_a = current_a[:n]
    raw_indices = np.arange(n, dtype=int)

    if temperature_c.size >= n:
        temperature_c = temperature_c[:n]
    else:
        temperature_c = np.full(n, np.nan, dtype=float)
        warnings.append("temperature_vector_missing_or_short")

    if q_charge_ah.size >= n:
        q_charge_ah = q_charge_ah[:n]
    else:
        q_charge_ah = np.full(n, np.nan, dtype=float)
        warnings.append("q_charge_vector_missing_or_short")

    if q_discharge_ah.size >= n:
        q_discharge_ah = q_discharge_ah[:n]
    else:
        q_discharge_ah = np.full(n, np.nan, dtype=float)
        warnings.append("q_discharge_vector_missing_or_short")

    mask = np.isfinite(time_s) & np.isfinite(voltage_v) & np.isfinite(current_a)
    removed = int((~mask).sum())
    if removed:
        warnings.append(f"removed_nonfinite_required_samples:{removed}")

    time_s = time_s[mask]
    voltage_v = voltage_v[mask]
    current_a = current_a[mask]
    temperature_c = temperature_c[mask]
    q_charge_ah = q_charge_ah[mask]
    q_discharge_ah = q_discharge_ah[mask]
    raw_indices = raw_indices[mask]

    if time_s.size < 2:
        return None, warnings, "insufficient_samples_after_finite_filter"

    if np.any(np.diff(time_s) < 0):
        order = np.argsort(time_s)
        time_s = time_s[order]
        voltage_v = voltage_v[order]
        current_a = current_a[order]
        temperature_c = temperature_c[order]
        q_charge_ah = q_charge_ah[order]
        q_discharge_ah = q_discharge_ah[order]
        raw_indices = raw_indices[order]
        warnings.append("cycle_sorted_by_time")

    _, unique_idx = np.unique(time_s, return_index=True)
    unique_idx = np.sort(unique_idx)
    if len(unique_idx) < len(time_s):
        warnings.append(f"removed_duplicated_time_samples:{len(time_s) - len(unique_idx)}")

    time_s = time_s[unique_idx]
    voltage_v = voltage_v[unique_idx]
    current_a = current_a[unique_idx]
    temperature_c = temperature_c[unique_idx]
    q_charge_ah = q_charge_ah[unique_idx]
    q_discharge_ah = q_discharge_ah[unique_idx]
    raw_indices = raw_indices[unique_idx]

    if time_s.size < 2:
        return None, warnings, "insufficient_samples_after_time_deduplication"

    return {
        "time_s": time_s,
        "voltage_v": voltage_v,
        "current_a": current_a,
        "temperature_c": temperature_c,
        "q_charge_ah": q_charge_ah,
        "q_discharge_ah": q_discharge_ah,
        "raw_indices": raw_indices,
    }, warnings, ""


# ---------------------------------------------------------------------------
# Contiguous current-threshold segmentation
# ---------------------------------------------------------------------------

def find_discharge_start_index(
    current_a: np.ndarray,
    *,
    discharge_current_threshold_a: float,
    confirm_points: int,
) -> int | None:
    """Return first index where discharge starts according to I < -threshold."""
    current_a = safe_vector(current_a)
    threshold = -abs(float(discharge_current_threshold_a))
    below = np.isfinite(current_a) & (current_a < threshold)
    if not below.any():
        return None

    confirm_points = max(1, int(confirm_points))
    if confirm_points <= 1:
        return int(np.where(below)[0][0])

    # Optional robustness: require N consecutive points below the threshold.
    idx = np.where(below)[0]
    for start in idx:
        end = int(start) + confirm_points
        if end <= len(below) and bool(np.all(below[int(start):end])):
            return int(start)

    # Fallback: no confirmed run found; preserve the user's threshold rule.
    return int(idx[0])


def slice_segment(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    return safe_vector(arr)[int(start):int(end)]


def extract_contiguous_phase_segment(
    *,
    aligned: dict[str, np.ndarray],
    phase: str,
    discharge_start_idx: int | None,
    min_points: int,
    discharge_current_threshold_a: float,
) -> tuple[dict[str, Any] | None, str, list[str]]:
    warnings: list[str] = []
    n_total = len(aligned["time_s"])

    if discharge_start_idx is None:
        if phase == "discharge":
            return None, "missing_discharge_current_threshold_crossing", warnings
        start_idx = 0
        end_idx = n_total
        warnings.append("no_discharge_threshold_found_charge_kept_until_cycle_end")
    elif phase == "charge":
        start_idx = 0
        end_idx = int(discharge_start_idx)
    elif phase == "discharge":
        start_idx = int(discharge_start_idx)
        end_idx = n_total
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    if end_idx <= start_idx:
        return None, f"empty_{phase}_segment_after_current_split", warnings

    time_s = slice_segment(aligned["time_s"], start_idx, end_idx)
    voltage_v = slice_segment(aligned["voltage_v"], start_idx, end_idx)
    current_a = slice_segment(aligned["current_a"], start_idx, end_idx)
    temperature_c = slice_segment(aligned["temperature_c"], start_idx, end_idx)
    q_charge_ah = slice_segment(aligned["q_charge_ah"], start_idx, end_idx)
    q_discharge_ah = slice_segment(aligned["q_discharge_ah"], start_idx, end_idx)
    raw_indices = slice_segment(aligned["raw_indices"], start_idx, end_idx).astype(int)

    if len(time_s) < int(min_points):
        return None, f"insufficient_{phase}_points_after_current_split", warnings

    # Reset local phase time.
    time_s = time_s - time_s[0]

    if phase == "charge":
        current_a = np.abs(current_a)
        raw_capacity = q_charge_ah
        capacity_label = "QCharge"
    else:
        current_a = -np.abs(current_a)
        raw_capacity = q_discharge_ah
        capacity_label = "QDischarge"

    capacity_source = "measured"
    if has_usable_capacity_vector(raw_capacity):
        capacity_ah = normalize_capacity_vector(raw_capacity)
    else:
        capacity_ah = integrate_capacity_ah(time_s, current_a)
        capacity_source = "integrated_current"
        warnings.append("capacity_reconstructed_from_current")

    if capacity_ah.size != time_s.size:
        capacity_ah = integrate_capacity_ah(time_s, current_a)
        capacity_source = "integrated_current"
        warnings.append("capacity_length_mismatch_reconstructed_from_current")

    if capacity_ah.size == 0 or not np.isfinite(capacity_ah).any():
        return None, "missing_capacity_after_extraction", warnings

    if np.nanmax(capacity_ah) <= 0:
        warnings.append("zero_or_near_zero_phase_capacity")

    segment = {
        "phase": phase,
        "time_s": time_s.astype(float).tolist(),
        "voltage_v": voltage_v.astype(float).tolist(),
        "current_a": current_a.astype(float).tolist(),
        "temperature_c": temperature_c.astype(float).tolist(),
        "capacity_ah": capacity_ah.astype(float).tolist(),
        "capacity_label": capacity_label,
        "capacity_source_raw": capacity_source,
        "n_points": int(len(time_s)),
        "trim_start_index_raw": int(raw_indices[0]) if raw_indices.size else -1,
        "trim_end_index_raw": int(raw_indices[-1]) if raw_indices.size else -1,
        "split_start_index_aligned": int(start_idx),
        "split_end_index_aligned_exclusive": int(end_idx),
        "split_discharge_start_index_aligned": int(discharge_start_idx) if discharge_start_idx is not None else -1,
        "discharge_current_threshold_a": -abs(float(discharge_current_threshold_a)),
        "split_rule": "charge_start_to_before_first_I_lt_minus_0p5A__discharge_from_first_I_lt_minus_0p5A_to_end",
        "extraction_warnings": warnings,
    }
    return segment, "accepted", warnings


def extract_one_file(
    mat_path: Path,
    *,
    base_dir: Path,
    min_points: int,
    discharge_current_threshold_a: float,
    confirm_discharge_points: int,
    start_cycle_index: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    charge_rows: list[dict[str, Any]] = []
    discharge_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []

    with h5py.File(mat_path, "r") as handle:
        if "batch" not in handle:
            raise KeyError(f"The file does not contain a 'batch' group: {mat_path}")

        batch = handle["batch"]
        if "cycles" not in batch or "summary" not in batch:
            raise KeyError("The 'batch' group must contain 'cycles' and 'summary'.")

        n_batteries = batch["cycles"].shape[0]
        source_path = relative_path(mat_path, base_dir)

        battery_iterator = tqdm(
            range(n_batteries),
            desc=f"Batteries in {mat_path.name}",
            unit="battery",
            leave=False,
        )

        for battery_index in battery_iterator:
            battery_id = f"MATR_{mat_path.stem}_b{battery_index:03d}"
            try:
                cycles_ref = get_reference(batch["cycles"], battery_index)
                summary_ref = get_reference(batch["summary"], battery_index)
                cycles_group = handle[cycles_ref]
                summary_group = handle[summary_ref]

                cycle_life = np.nan
                policy_readable = ""
                barcode = ""

                if "cycle_life" in batch:
                    cycle_life = read_mat_ref_scalar(
                        handle,
                        get_reference(batch["cycle_life"], battery_index),
                    )
                if "policy_readable" in batch:
                    policy_readable = read_mat_ref_string(
                        handle,
                        get_reference(batch["policy_readable"], battery_index),
                    )
                if "barcode" in batch:
                    barcode = read_mat_ref_string(
                        handle,
                        get_reference(batch["barcode"], battery_index),
                    )

                summary_cycle = read_summary_array(summary_group, "cycle")
                summary_qd = read_summary_array(summary_group, "QDischarge")
                summary_qc = read_summary_array(summary_group, "QCharge")
                summary_tmin = read_summary_array(summary_group, "Tmin")
                summary_tavg = read_summary_array(summary_group, "Tavg")
                summary_tmax = read_summary_array(summary_group, "Tmax")
                summary_chargetime = read_summary_array(summary_group, "chargetime")

                if "V" not in cycles_group:
                    raise KeyError("Field 'V' is missing in the cycles group.")
                n_cycles = cycles_group["V"].shape[0]
                cell_id_original = barcode if barcode else f"battery_{battery_index:03d}"

                cycle_start = max(0, int(start_cycle_index))
                cycle_iterator = tqdm(
                    range(cycle_start, n_cycles),
                    desc=f"Cycles {battery_id}",
                    unit="cycle",
                    leave=False,
                )

                for source_cycle_index in cycle_iterator:
                    cycle_number = summary_value(
                        summary_cycle,
                        source_cycle_index,
                        default=float(source_cycle_index),
                    )
                    cycle_pair_id = f"{battery_id}_cycle_{int(source_cycle_index):05d}"
                    source_cycle_id = (
                        f"{mat_path.stem}:battery_{battery_index:03d}:"
                        f"cycle_{source_cycle_index:05d}"
                    )

                    try:
                        aligned, align_warnings, align_reason = align_complete_cycle(
                            time_s=read_cycle_vector(handle, cycles_group, "t", source_cycle_index),
                            voltage_v=read_cycle_vector(handle, cycles_group, "V", source_cycle_index),
                            current_a=read_cycle_vector(handle, cycles_group, "I", source_cycle_index),
                            temperature_c=read_cycle_vector(handle, cycles_group, "T", source_cycle_index),
                            q_charge_ah=read_cycle_vector(handle, cycles_group, "Qc", source_cycle_index),
                            q_discharge_ah=read_cycle_vector(handle, cycles_group, "Qd", source_cycle_index),
                        )

                        if aligned is None:
                            for phase in ["charge", "discharge"]:
                                report_rows.append(
                                    {
                                        "dataset": DATASET,
                                        "source_file": mat_path.name,
                                        "source_path": source_path,
                                        "battery_index": battery_index,
                                        "battery_id": battery_id,
                                        "source_cycle_index": source_cycle_index,
                                        "cycle_number": cycle_number,
                                        "phase": phase,
                                        "status": "rejected",
                                        "reason": align_reason,
                                        "warnings": align_warnings,
                                    }
                                )
                            continue

                        discharge_start_idx = find_discharge_start_index(
                            aligned["current_a"],
                            discharge_current_threshold_a=discharge_current_threshold_a,
                            confirm_points=confirm_discharge_points,
                        )

                        for phase in ["charge", "discharge"]:
                            segment, status, warnings = extract_contiguous_phase_segment(
                                aligned=aligned,
                                phase=phase,
                                discharge_start_idx=discharge_start_idx,
                                min_points=min_points,
                                discharge_current_threshold_a=discharge_current_threshold_a,
                            )
                            warnings = [*align_warnings, *warnings]

                            if segment is None:
                                report_rows.append(
                                    {
                                        "dataset": DATASET,
                                        "source_file": mat_path.name,
                                        "source_path": source_path,
                                        "battery_index": battery_index,
                                        "battery_id": battery_id,
                                        "source_cycle_index": source_cycle_index,
                                        "cycle_number": cycle_number,
                                        "phase": phase,
                                        "status": "rejected",
                                        "reason": status,
                                        "warnings": warnings,
                                        "discharge_start_index_aligned": int(discharge_start_idx) if discharge_start_idx is not None else -1,
                                    }
                                )
                                continue

                            row = {
                                "dataset": DATASET,
                                "source_file": mat_path.name,
                                "source_path": source_path,
                                "battery_id": battery_id,
                                "battery_index": int(battery_index),
                                "cell_id_original": cell_id_original,
                                "source_cycle_index": int(source_cycle_index),
                                "cycle_number": float(cycle_number),
                                "source_cycle_id": source_cycle_id,
                                "cycle_pair_id": cycle_pair_id,
                                "cycle_life": float(cycle_life),
                                "summary_QDischarge": summary_value(summary_qd, source_cycle_index),
                                "summary_QCharge": summary_value(summary_qc, source_cycle_index),
                                "summary_Tmin": summary_value(summary_tmin, source_cycle_index),
                                "summary_Tavg": summary_value(summary_tavg, source_cycle_index),
                                "summary_Tmax": summary_value(summary_tmax, source_cycle_index),
                                "summary_chargetime_s": summary_value(summary_chargetime, source_cycle_index),
                                "policy_readable": policy_readable,
                                "barcode": barcode,
                                "adapter_version": ADAPTER_VERSION,
                                "created_at": utc_now_iso(),
                                **segment,
                            }

                            if phase == "charge":
                                charge_rows.append(row)
                            else:
                                discharge_rows.append(row)

                            report_rows.append(
                                {
                                    "dataset": DATASET,
                                    "source_file": mat_path.name,
                                    "source_path": source_path,
                                    "battery_index": battery_index,
                                    "battery_id": battery_id,
                                    "source_cycle_index": source_cycle_index,
                                    "cycle_number": cycle_number,
                                    "phase": phase,
                                    "status": "accepted",
                                    "reason": "",
                                    "warnings": warnings,
                                    "n_points": segment["n_points"],
                                    "discharge_start_index_aligned": int(discharge_start_idx) if discharge_start_idx is not None else -1,
                                    "trim_start_index_raw": segment["trim_start_index_raw"],
                                    "trim_end_index_raw": segment["trim_end_index_raw"],
                                }
                            )

                    except Exception as exc:
                        for phase in ["charge", "discharge"]:
                            report_rows.append(
                                {
                                    "dataset": DATASET,
                                    "source_file": mat_path.name,
                                    "source_path": source_path,
                                    "battery_index": battery_index,
                                    "battery_id": battery_id,
                                    "source_cycle_index": source_cycle_index,
                                    "cycle_number": cycle_number,
                                    "phase": phase,
                                    "status": "rejected",
                                    "reason": f"cycle_read_error: {exc}",
                                    "warnings": [],
                                }
                            )

            except Exception as exc:
                report_rows.append(
                    {
                        "dataset": DATASET,
                        "source_file": mat_path.name,
                        "source_path": source_path,
                        "battery_index": battery_index,
                        "battery_id": battery_id,
                        "source_cycle_index": np.nan,
                        "cycle_number": np.nan,
                        "phase": "unknown",
                        "status": "battery_error",
                        "reason": str(exc),
                        "warnings": [],
                    }
                )

    return (
        {
            "charge": pd.DataFrame(charge_rows),
            "discharge": pd.DataFrame(discharge_rows),
        },
        pd.DataFrame(report_rows),
    )


def summarize_extraction(report_df: pd.DataFrame, phase_dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset": DATASET,
        "adapter_version": ADAPTER_VERSION,
        "created_at": utc_now_iso(),
        "segmentation_rule": {
            "charge": "from cycle start to the sample before first current < -0.5 A by default",
            "discharge": "from first current < -0.5 A by default to cycle end",
        },
        "rows_by_phase": {},
        "status_counts": {},
        "unique_batteries_by_phase": {},
        "n_points_by_phase": {},
    }
    for phase, df in phase_dfs.items():
        summary["rows_by_phase"][phase] = int(len(df))
        summary["unique_batteries_by_phase"][phase] = int(df["battery_id"].nunique()) if not df.empty else 0
        if not df.empty and "n_points" in df.columns:
            summary["n_points_by_phase"][phase] = {
                "min": int(df["n_points"].min()),
                "median": float(df["n_points"].median()),
                "max": int(df["n_points"].max()),
            }
    if not report_df.empty:
        status_counts = report_df.groupby(["phase", "status"]).size().reset_index(name="count")
        summary["status_counts"] = {
            f"{row.phase}:{row.status}": int(row["count"])
            for _, row in status_counts.iterrows()
        }
        reason_counts = report_df.loc[report_df["status"] != "accepted", "reason"].value_counts(dropna=False)
        summary["rejection_reason_counts"] = {str(k): int(v) for k, v in reason_counts.to_dict().items()}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MATR charge/discharge cycles from raw .mat files using contiguous current-threshold segmentation."
    )
    parser.add_argument("--base-dir", type=str, default=None, help="Project root directory. Defaults to the known Batteries path or current directory.")
    parser.add_argument("--raw-dir", type=str, default=None, help="Directory containing MATR .mat files. Defaults to <base-dir>/data/raw/MATR.")
    parser.add_argument("--output-dir", type=str, default=None, help="Extraction output directory. Defaults to <base-dir>/data/interim/matr/extracted.")
    parser.add_argument("--min-points", type=int, default=20, help="Minimum points required to keep an extracted phase segment.")
    parser.add_argument(
        "--discharge-current-threshold-a",
        type=float,
        default=0.5,
        help="Absolute current threshold used to detect discharge start. Discharge starts at first I < -threshold. Default: 0.5 A.",
    )
    parser.add_argument(
        "--confirm-discharge-points",
        type=int,
        default=1,
        help="Optional number of consecutive samples below -threshold required to confirm discharge start. Default 1 follows the exact requested rule.",
    )
    parser.add_argument(
        "--start-cycle-index",
        type=int,
        default=1,
        help="First source cycle index to process. Default 1 preserves the previous MATR adapter behavior.",
    )
    parser.add_argument("--file-pattern", type=str, default="*.mat", help="Raw file glob pattern.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    raw_dir = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else base_dir / "data" / "raw" / "MATR"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else base_dir / "data" / "interim" / "matr" / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"MATR raw-data directory not found: {raw_dir}")

    mat_files = sorted(raw_dir.glob(args.file_pattern))
    if not mat_files:
        raise FileNotFoundError(f"No MATR .mat files found in: {raw_dir} using pattern {args.file_pattern!r}")

    print("=" * 88)
    print("MATR extraction - version 2.0 contiguous current split")
    print("=" * 88)
    print(f"Base directory : {base_dir}")
    print(f"Raw directory  : {raw_dir}")
    print(f"Output directory: {output_dir}")
    print(f"MAT files found: {len(mat_files)}")
    print(f"Discharge rule : first current < {-abs(args.discharge_current_threshold_a):.6g} A")
    print(f"Confirm points : {args.confirm_discharge_points}")
    print(f"Start cycle idx: {args.start_cycle_index}")

    collected: dict[str, list[pd.DataFrame]] = {"charge": [], "discharge": []}
    reports: list[pd.DataFrame] = []

    for mat_path in tqdm(mat_files, desc="MATR files", unit="file"):
        try:
            phase_dfs, report_df = extract_one_file(
                mat_path,
                base_dir=base_dir,
                min_points=args.min_points,
                discharge_current_threshold_a=args.discharge_current_threshold_a,
                confirm_discharge_points=args.confirm_discharge_points,
                start_cycle_index=args.start_cycle_index,
            )
            for phase in ["charge", "discharge"]:
                if not phase_dfs[phase].empty:
                    collected[phase].append(phase_dfs[phase])
            reports.append(report_df)
        except Exception as exc:
            reports.append(
                pd.DataFrame(
                    [
                        {
                            "dataset": DATASET,
                            "source_file": mat_path.name,
                            "source_path": relative_path(mat_path, base_dir),
                            "battery_index": np.nan,
                            "battery_id": "",
                            "source_cycle_index": np.nan,
                            "cycle_number": np.nan,
                            "phase": "unknown",
                            "status": "file_error",
                            "reason": str(exc),
                            "warnings": [],
                        }
                    ]
                )
            )

    final_dfs: dict[str, pd.DataFrame] = {}
    for phase in ["charge", "discharge"]:
        if collected[phase]:
            df = pd.concat(collected[phase], ignore_index=True)
            df = df.sort_values(["battery_id", "source_cycle_index"]).reset_index(drop=True)
        else:
            df = pd.DataFrame()
        final_dfs[phase] = df
        out_file = output_dir / f"matr_{phase}_extracted_cycles.parquet"
        df.to_parquet(out_file, index=False)
        print(f"[{phase}] extracted rows: {len(df):,} -> {out_file}")

    report_df = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    report_csv = output_dir / "matr_extraction_report.csv"
    report_df.to_csv(report_csv, index=False, encoding="utf-8-sig")

    summary = summarize_extraction(report_df, final_dfs)
    summary_json = output_dir / "matr_extraction_report.json"
    with summary_json.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print(f"Extraction report CSV : {report_csv}")
    print(f"Extraction report JSON: {summary_json}")
    print("Done.")


if __name__ == "__main__":
    main()
