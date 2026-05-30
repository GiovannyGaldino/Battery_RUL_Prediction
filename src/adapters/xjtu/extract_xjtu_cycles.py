# -*- coding: utf-8 -*-
"""
Extract charge and discharge cycle segments from the XJTU battery aging dataset.

Expected raw layout example:
    data/raw/XJTU/Batch-1/2C_battery-1.mat

Outputs:
    data/interim/xjtu/extracted/xjtu_charge_extracted_cycles.parquet
    data/interim/xjtu/extracted/xjtu_discharge_extracted_cycles.parquet
    data/reports/xjtu/extraction_report.json
    data/reports/xjtu/extraction_file_report.csv

The script keeps charge and discharge in separate files and uses tqdm progress
bars for file-level and cycle-level processing.
"""

from __future__ import annotations

import argparse
import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.io import loadmat

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from xjtu_schema import (
    ADAPTER_VERSION,
    CLEANING_VERSION,
    DATASET_NAME,
    DATASET_NAME_LOWER,
    STANDARD_CYCLE_COLUMNS,
    as_1d_float,
    as_string,
    clean_file,
    compute_power,
    cumulative_capacity_ah,
    cumulative_energy_wh,
    default_extracted_dir,
    default_raw_dir,
    default_report_dir,
    ensure_standard_columns,
    extracted_file,
    finite_float,
    safe_max,
    safe_mean,
    safe_median_diff,
    safe_min,
    utc_now_iso,
    write_json,
)

DEFAULT_FIRST_CYCLE_USED = 2
DEFAULT_RESCALE_REL_TOL = 0.05
MIN_POINTS_TO_EXTRACT = 5


@dataclass
class PhaseSegment:
    time_min: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_c: np.ndarray
    method: str


def load_mat_file(mat_path: Path) -> dict[str, Any]:
    return loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False,
        simplify_cells=False,
    )


def get_field(obj: Any, field: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field, default)
    if hasattr(obj, field):
        return getattr(obj, field)
    try:
        if hasattr(obj, "dtype") and obj.dtype.names and field in obj.dtype.names:
            return obj[field]
    except Exception:
        pass
    try:
        return obj[field]
    except Exception:
        return default


def iter_mat_struct_array(value: Any) -> list[Any]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=object).squeeze()
    if arr.ndim == 0:
        return [arr.item()]
    return [item for item in arr.flat]


def find_main_struct(mat: dict[str, Any]) -> tuple[Any, Any]:
    summary = mat.get("summary", None)
    data = mat.get("data", None)
    if summary is not None and data is not None:
        return summary, data

    for key, value in mat.items():
        if str(key).startswith("__"):
            continue
        summary = get_field(value, "summary", None)
        data = get_field(value, "data", None)
        if summary is not None and data is not None:
            return summary, data

    raise KeyError("Could not locate MATLAB fields 'summary' and 'data'.")


def get_summary_vector(summary: Any, field: str) -> np.ndarray:
    return as_1d_float(get_field(summary, field, None))


def get_cycle_vector(cycle: Any, field: str) -> np.ndarray:
    return as_1d_float(get_field(cycle, field, None))


def parse_batch_name(mat_path: Path, raw_dir: Path) -> str:
    try:
        for part in mat_path.relative_to(raw_dir).parts:
            if part.lower().startswith("batch"):
                return part
    except Exception:
        pass
    for part in mat_path.parts:
        if part.lower().startswith("batch"):
            return part
    return mat_path.parent.name


def parse_batch_number(batch_name: str) -> int | None:
    match = re.search(r"(\d+)", batch_name)
    return int(match.group(1)) if match else None


def parse_battery_number(file_name: str) -> int | None:
    match = re.search(r"battery[-_ ]?(\d+)", file_name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_c_rate_from_name(file_name: str) -> float:
    stem = Path(file_name).stem
    first_token = stem.split("_")[0]
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)C", first_token, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return np.nan


def build_battery_id(batch_name: str, file_name: str) -> str:
    batch_number = parse_batch_number(batch_name)
    battery_number = parse_battery_number(file_name)
    batch_number = batch_number if batch_number is not None else 0
    battery_number = battery_number if battery_number is not None else 0
    return f"XJTU_Batch{batch_number}_{battery_number:03d}"


def clean_and_align_raw_vectors(
    time_min: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    temperature_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = [
        as_1d_float(time_min),
        as_1d_float(voltage_v),
        as_1d_float(current_a),
        as_1d_float(temperature_c),
    ]
    lengths = [len(x) for x in arrays if len(x) > 0]
    if len(lengths) < 3:
        return tuple(np.array([], dtype=float) for _ in range(4))

    min_len = min(lengths)
    time_min, voltage_v, current_a, temperature_c = [x[:min_len] for x in arrays]

    if len(temperature_c) == 0:
        temperature_c = np.full(min_len, np.nan)

    mask = np.isfinite(time_min) & np.isfinite(voltage_v) & np.isfinite(current_a)
    # Temperature may be fully missing, so do not require it to be finite.
    time_min = time_min[mask]
    voltage_v = voltage_v[mask]
    current_a = current_a[mask]
    temperature_c = temperature_c[mask]

    if time_min.size == 0:
        return tuple(np.array([], dtype=float) for _ in range(4))

    order = np.argsort(time_min, kind="mergesort")
    return time_min[order], voltage_v[order], current_a[order], temperature_c[order]


def split_by_relative_time_zero(
    time_min: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    temperature_c: np.ndarray,
) -> tuple[PhaseSegment | None, PhaseSegment | None]:
    if len(time_min) < MIN_POINTS_TO_EXTRACT:
        return None, None

    zero_idx = np.where(np.isclose(time_min, 0.0, atol=1e-12))[0]
    if len(zero_idx) < 3:
        return None, None

    bounds = list(zero_idx) + [len(time_min)]
    c0, c1 = bounds[0], bounds[1]
    d0 = bounds[2]
    d1 = bounds[3] if len(bounds) > 3 else len(time_min)

    charge = PhaseSegment(
        time_min=time_min[c0:c1],
        voltage_v=voltage_v[c0:c1],
        current_a=current_a[c0:c1],
        temperature_c=temperature_c[c0:c1],
        method="relative_time_zero_segment",
    )
    discharge = PhaseSegment(
        time_min=time_min[d0:d1],
        voltage_v=voltage_v[d0:d1],
        current_a=current_a[d0:d1],
        temperature_c=temperature_c[d0:d1],
        method="relative_time_zero_segment",
    )

    if len(charge.voltage_v) < MIN_POINTS_TO_EXTRACT:
        charge = None
    if len(discharge.voltage_v) < MIN_POINTS_TO_EXTRACT:
        discharge = None
    return charge, discharge


def split_by_current_sign(
    time_min: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    temperature_c: np.ndarray,
) -> tuple[PhaseSegment | None, PhaseSegment | None]:
    charge_mask = current_a > 0
    discharge_mask = current_a < 0

    charge = None
    discharge = None
    if int(np.count_nonzero(charge_mask)) >= MIN_POINTS_TO_EXTRACT:
        charge = PhaseSegment(
            time_min=time_min[charge_mask],
            voltage_v=voltage_v[charge_mask],
            current_a=current_a[charge_mask],
            temperature_c=temperature_c[charge_mask],
            method="current_sign",
        )
    if int(np.count_nonzero(discharge_mask)) >= MIN_POINTS_TO_EXTRACT:
        discharge = PhaseSegment(
            time_min=time_min[discharge_mask],
            voltage_v=voltage_v[discharge_mask],
            current_a=current_a[discharge_mask],
            temperature_c=temperature_c[discharge_mask],
            method="current_sign",
        )
    return charge, discharge


def normalize_phase_time(phase: PhaseSegment) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_s = as_1d_float(phase.time_min) * 60.0
    if time_s.size:
        time_s = time_s - time_s[0]
    return (
        time_s.astype(float),
        as_1d_float(phase.voltage_v).astype(float),
        as_1d_float(phase.current_a).astype(float),
        as_1d_float(phase.temperature_c).astype(float),
    )


def get_summary_value(summary_vector: np.ndarray, zero_based_cycle_idx: int) -> float:
    if summary_vector.size == 0:
        return np.nan
    if 0 <= zero_based_cycle_idx < len(summary_vector):
        return float(summary_vector[zero_based_cycle_idx])
    if 0 <= zero_based_cycle_idx - 1 < len(summary_vector):
        return float(summary_vector[zero_based_cycle_idx - 1])
    return np.nan


def maybe_rescale_capacity(q_ah: np.ndarray, summary_capacity_ah: float, rel_tol: float) -> tuple[np.ndarray, bool]:
    q = as_1d_float(q_ah)
    if q.size == 0:
        return q, False
    if not np.isfinite(summary_capacity_ah) or summary_capacity_ah <= 0:
        return q, False

    q_end = safe_max(q)
    if not np.isfinite(q_end) or q_end <= 0:
        return np.linspace(0.0, summary_capacity_ah, len(q)), True

    rel_error = abs(q_end - summary_capacity_ah) / max(abs(summary_capacity_ah), 1e-12)
    if rel_error > rel_tol:
        return q * (summary_capacity_ah / q_end), True
    return q, False


def basic_scalar_metadata(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    temperature_c: np.ndarray,
    capacity_ah: np.ndarray,
    energy_wh: np.ndarray,
) -> dict[str, float | int]:
    return {
        "n_points": int(len(time_s)),
        "duration_s": float(time_s[-1] - time_s[0]) if len(time_s) >= 2 else np.nan,
        "sample_period_median_s": safe_median_diff(time_s),
        "voltage_min_v": safe_min(voltage_v),
        "voltage_max_v": safe_max(voltage_v),
        "voltage_start_v": float(voltage_v[0]) if len(voltage_v) else np.nan,
        "voltage_end_v": float(voltage_v[-1]) if len(voltage_v) else np.nan,
        "current_min_a": safe_min(current_a),
        "current_max_a": safe_max(current_a),
        "current_mean_a": safe_mean(current_a),
        "temperature_min_c": safe_min(temperature_c),
        "temperature_max_c": safe_max(temperature_c),
        "temperature_mean_c": safe_mean(temperature_c),
        "phase_capacity_ah": float(capacity_ah[-1]) if len(capacity_ah) else np.nan,
        "phase_energy_wh": float(energy_wh[-1]) if len(energy_wh) else np.nan,
    }


def build_phase_row(
    *,
    mat_path: Path,
    raw_dir: Path,
    battery_id: str,
    battery_index: int,
    cycle_number: int,
    phase: str,
    phase_segment: PhaseSegment,
    cycle_life: float,
    summary_capacity_ah: float,
    batch_name: str,
    description: str,
    rescale_rel_tol: float,
) -> dict[str, Any]:
    time_s, voltage_v, current_a, temperature_c = normalize_phase_time(phase_segment)

    # Keep raw current convention in extraction. The cleaning script enforces the
    # project convention: charge positive and discharge negative.
    capacity_ah = cumulative_capacity_ah(time_s, current_a)
    capacity_ah, capacity_rescaled = maybe_rescale_capacity(capacity_ah, summary_capacity_ah, rescale_rel_tol)
    power_w = compute_power(voltage_v, current_a)
    energy_wh = cumulative_energy_wh(time_s, voltage_v, current_a)

    n = min(len(time_s), len(voltage_v), len(current_a), len(temperature_c), len(capacity_ah), len(power_w), len(energy_wh))
    time_s = time_s[:n]
    voltage_v = voltage_v[:n]
    current_a = current_a[:n]
    temperature_c = temperature_c[:n]
    capacity_ah = capacity_ah[:n]
    power_w = power_w[:n]
    energy_wh = energy_wh[:n]

    relative_source_path = str(mat_path.relative_to(raw_dir)).replace("\\", "/") if mat_path.is_relative_to(raw_dir) else str(mat_path)
    source_cycle_id = f"{mat_path.stem}_cycle_{cycle_number:05d}"
    cycle_pair_id = f"{battery_id}_cycle_{cycle_number:05d}"
    phase_capacity_ah = float(capacity_ah[-1]) if len(capacity_ah) else np.nan
    remaining_cycles = cycle_life - cycle_number if np.isfinite(cycle_life) else np.nan
    remaining_cycles_norm = remaining_cycles / cycle_life if np.isfinite(remaining_cycles) and np.isfinite(cycle_life) and cycle_life > 0 else np.nan
    c_rate = parse_c_rate_from_name(mat_path.name)

    warnings = []
    if capacity_rescaled:
        warnings.append("capacity_vector_rescaled_to_summary_capacity")
    if description:
        warnings.append(f"source_description={description}")

    temperature_is_measured = bool(np.any(np.isfinite(temperature_c)))
    scalars = basic_scalar_metadata(time_s, voltage_v, current_a, temperature_c, capacity_ah, energy_wh)
    now = utc_now_iso()

    row = {
        "dataset": DATASET_NAME,
        "source_file": mat_path.name,
        "source_path": relative_source_path,
        "battery_id": battery_id,
        "battery_index": int(battery_index),
        "cell_id_original": mat_path.stem,
        "cycle_index": int(cycle_number),
        "cycle_number": float(cycle_number),
        "source_cycle_id": source_cycle_id,
        "cycle_pair_id": cycle_pair_id,
        "phase": phase,
        "cycle_life": float(cycle_life) if np.isfinite(cycle_life) else np.nan,
        "remaining_cycles": float(remaining_cycles) if np.isfinite(remaining_cycles) else np.nan,
        "remaining_cycles_norm": float(remaining_cycles_norm) if np.isfinite(remaining_cycles_norm) else np.nan,
        "eol_criterion": "provided_by_dataset" if np.isfinite(cycle_life) else "unknown",
        "eol_capacity_ah": np.nan,
        "initial_capacity_ah": np.nan,
        "nominal_capacity_ah": np.nan,
        "soh": np.nan,
        "time_s": time_s.astype(float).tolist(),
        "voltage_v": voltage_v.astype(float).tolist(),
        "current_a": current_a.astype(float).tolist(),
        "temperature_c": temperature_c.astype(float).tolist(),
        "capacity_ah": capacity_ah.astype(float).tolist(),
        "energy_wh": energy_wh.astype(float).tolist(),
        "power_w": power_w.astype(float).tolist(),
        **scalars,
        "voltage_source": "measured",
        "current_source": "measured",
        "temperature_source": "measured" if temperature_is_measured else "unknown_nan",
        "capacity_source": "integrated_current",
        "energy_source": "integrated_power",
        "chemistry": "unknown",
        "form_factor": "unknown",
        "manufacturer": "unknown",
        "nominal_voltage_v": np.nan,
        "ambient_temperature_c": np.nan,
        "charge_protocol": f"XJTU {batch_name}; extracted charge segment" if phase == "charge" else f"XJTU {batch_name}; paired charge protocol unavailable in this row",
        "discharge_protocol": f"XJTU {batch_name}; extracted discharge segment" if phase == "discharge" else f"XJTU {batch_name}; paired discharge protocol unavailable in this row",
        "c_rate_charge": c_rate if phase == "charge" else np.nan,
        "c_rate_discharge": c_rate if phase == "discharge" else np.nan,
        "dod": np.nan,
        "policy_readable": f"{batch_name}; {mat_path.stem}; split={phase_segment.method}",
        "barcode": mat_path.stem,
        "is_valid_cycle": bool(n >= MIN_POINTS_TO_EXTRACT and np.isfinite(phase_capacity_ah) and phase_capacity_ah > 0),
        "is_model_usable": False,
        "exclude_reason": "",
        "cleaning_warnings": warnings,
        "n_removed_points": 0,
        "trim_start_index": 0,
        "trim_end_index": int(max(n - 1, 0)),
        "has_voltage": bool(np.any(np.isfinite(voltage_v))),
        "has_current": bool(np.any(np.isfinite(current_a))),
        "has_temperature": temperature_is_measured,
        "has_capacity": bool(np.any(np.isfinite(capacity_ah))),
        "has_energy": bool(np.any(np.isfinite(energy_wh))),
        "adapter_version": ADAPTER_VERSION,
        "cleaning_version": CLEANING_VERSION,
        "created_at": now,
    }

    return row



def extract_one_file(
    mat_path: Path,
    raw_dir: Path,
    battery_index: int,
    first_cycle_used: int,
    rescale_rel_tol: float,
    charge_writer: "BufferedParquetWriter",
    discharge_writer: "BufferedParquetWriter",
) -> dict[str, Any]:
    """Extract one MAT file and immediately stream rows to parquet writers.

    This function intentionally does not return cycle-row lists. Each accepted
    phase row is written to a small in-memory parquet buffer and then discarded,
    preventing the adapter from accumulating the full XJTU dataset in RAM.
    """
    batch_name = parse_batch_name(mat_path, raw_dir)
    battery_id = build_battery_id(batch_name, mat_path.name)
    report = {
        "dataset": DATASET_NAME,
        "source_file": mat_path.name,
        "source_path": str(mat_path.relative_to(raw_dir)).replace("\\", "/") if mat_path.is_relative_to(raw_dir) else str(mat_path),
        "battery_id": battery_id,
        "battery_index": int(battery_index),
        "batch": batch_name,
        "status": "ok",
        "error": "",
        "n_data_cycles": 0,
        "n_charge_rows": 0,
        "n_discharge_rows": 0,
        "n_cycles_without_defensible_segment": 0,
        "cycle_life": np.nan,
    }

    mat = None
    cycles = []
    try:
        mat = load_mat_file(mat_path)
        summary, data = find_main_struct(mat)
        cycles = iter_mat_struct_array(data)
        report["n_data_cycles"] = len(cycles)

        cycle_life_vec = get_summary_vector(summary, "cycle_life")
        summary_qd = get_summary_vector(summary, "discharge_capacity_Ah")
        summary_qc = get_summary_vector(summary, "charge_capacity_Ah")
        cycle_life = float(cycle_life_vec[0]) if cycle_life_vec.size else finite_float(get_field(summary, "cycle_life", np.nan))
        report["cycle_life"] = cycle_life

        start_idx = max(first_cycle_used - 1, 0)
        cycle_iter = range(start_idx, len(cycles))
        for zero_based_idx in tqdm(cycle_iter, desc=f"{mat_path.stem} cycles", leave=False, unit="cycle"):
            cycle = cycles[zero_based_idx]
            cycle_number = zero_based_idx + 1
            rel_time_min = get_cycle_vector(cycle, "relative_time_min")
            voltage_v = get_cycle_vector(cycle, "voltage_V")
            current_a = get_cycle_vector(cycle, "current_A")
            temperature_c = get_cycle_vector(cycle, "temperature_C")
            description = as_string(get_field(cycle, "description", ""))

            if len(temperature_c) == 0 and len(voltage_v) > 0:
                temperature_c = np.full(len(voltage_v), np.nan)

            rel_time_min, voltage_v, current_a, temperature_c = clean_and_align_raw_vectors(
                rel_time_min,
                voltage_v,
                current_a,
                temperature_c,
            )
            if len(voltage_v) < MIN_POINTS_TO_EXTRACT:
                report["n_cycles_without_defensible_segment"] += 1
                continue

            charge, discharge = split_by_relative_time_zero(rel_time_min, voltage_v, current_a, temperature_c)
            if charge is None or discharge is None:
                fallback_charge, fallback_discharge = split_by_current_sign(rel_time_min, voltage_v, current_a, temperature_c)
                charge = charge if charge is not None else fallback_charge
                discharge = discharge if discharge is not None else fallback_discharge

            if charge is None and discharge is None:
                report["n_cycles_without_defensible_segment"] += 1
                continue

            qd_summary = get_summary_value(summary_qd, zero_based_idx)
            qc_summary = get_summary_value(summary_qc, zero_based_idx)

            if charge is not None:
                row = build_phase_row(
                    mat_path=mat_path,
                    raw_dir=raw_dir,
                    battery_id=battery_id,
                    battery_index=battery_index,
                    cycle_number=cycle_number,
                    phase="charge",
                    phase_segment=charge,
                    cycle_life=cycle_life,
                    summary_capacity_ah=qc_summary,
                    batch_name=batch_name,
                    description=description,
                    rescale_rel_tol=rescale_rel_tol,
                )
                charge_writer.write(row)
                report["n_charge_rows"] += 1
                del row

            if discharge is not None:
                row = build_phase_row(
                    mat_path=mat_path,
                    raw_dir=raw_dir,
                    battery_id=battery_id,
                    battery_index=battery_index,
                    cycle_number=cycle_number,
                    phase="discharge",
                    phase_segment=discharge,
                    cycle_life=cycle_life,
                    summary_capacity_ah=qd_summary,
                    batch_name=batch_name,
                    description=description,
                    rescale_rel_tol=rescale_rel_tol,
                )
                discharge_writer.write(row)
                report["n_discharge_rows"] += 1
                del row

            # Explicitly release per-cycle arrays. This matters for large MATLAB cells.
            del cycle, rel_time_min, voltage_v, current_a, temperature_c, charge, discharge

    except Exception as exc:
        report["status"] = "error"
        report["error"] = repr(exc)
    finally:
        del cycles
        if mat is not None:
            del mat
        gc.collect()

    return report


def discover_mat_files(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.rglob("*.mat"))


def coerce_row_for_parquet(row: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with vector-like fields normalized as Python lists."""
    out = {}
    vector_cols = {"time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah", "energy_wh", "power_w"}
    list_string_cols = {"cleaning_warnings"}
    for col in STANDARD_CYCLE_COLUMNS:
        value = row.get(col, np.nan)
        if col in vector_cols:
            arr = as_1d_float(value).astype("float32", copy=False)
            out[col] = arr.tolist()
        elif col in list_string_cols:
            if isinstance(value, list):
                out[col] = [str(item) for item in value]
            elif value is None or (isinstance(value, float) and np.isnan(value)):
                out[col] = []
            else:
                out[col] = [str(value)]
        else:
            out[col] = value
    return out


def standard_arrow_schema() -> pa.Schema:
    """Arrow schema used to write extracted cycle files in bounded-memory chunks."""
    vector_float = pa.list_(pa.float32())
    list_string = pa.list_(pa.string())

    int_cols = {"battery_index", "cycle_index", "n_points", "n_removed_points", "trim_start_index", "trim_end_index"}
    bool_cols = {"is_valid_cycle", "is_model_usable", "has_voltage", "has_current", "has_temperature", "has_capacity", "has_energy"}
    vector_cols = {"time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah", "energy_wh", "power_w"}
    list_string_cols = {"cleaning_warnings"}
    string_cols = {
        "dataset", "source_file", "source_path", "battery_id", "cell_id_original", "source_cycle_id",
        "cycle_pair_id", "phase", "eol_criterion", "voltage_source", "current_source",
        "temperature_source", "capacity_source", "energy_source", "chemistry", "form_factor",
        "manufacturer", "charge_protocol", "discharge_protocol", "policy_readable", "barcode",
        "exclude_reason", "adapter_version", "cleaning_version", "created_at",
    }

    fields = []
    for col in STANDARD_CYCLE_COLUMNS:
        if col in vector_cols:
            dtype = vector_float
        elif col in list_string_cols:
            dtype = list_string
        elif col in int_cols:
            dtype = pa.int64()
        elif col in bool_cols:
            dtype = pa.bool_()
        elif col in string_cols:
            dtype = pa.string()
        else:
            dtype = pa.float64()
        fields.append(pa.field(col, dtype))
    return pa.schema(fields)


class BufferedParquetWriter:
    """Small-buffer parquet writer for cycle rows with long vector columns."""

    def __init__(self, output_path: Path, chunk_size: int, description: str):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_size = max(int(chunk_size), 1)
        self.description = description
        self.schema = standard_arrow_schema()
        self.writer: pq.ParquetWriter | None = None
        self.buffer: list[dict[str, Any]] = []
        self.n_rows = 0
        self._closed = False

    def write(self, row: dict[str, Any]) -> None:
        self.buffer.append(coerce_row_for_parquet(row))
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        chunk_df = ensure_standard_columns(pd.DataFrame(self.buffer))
        table = pa.Table.from_pandas(chunk_df, schema=self.schema, preserve_index=False, safe=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.output_path, table.schema, compression="snappy")
        self.writer.write_table(table)
        self.n_rows += len(self.buffer)
        self.buffer.clear()
        del chunk_df, table
        gc.collect()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        if self.writer is None:
            empty_df = ensure_standard_columns(pd.DataFrame())
            empty_table = pa.Table.from_pandas(empty_df, schema=self.schema, preserve_index=False, safe=False)
            pq.write_table(empty_table, self.output_path)
            del empty_df, empty_table
        else:
            self.writer.close()
        self._closed = True
        gc.collect()


def save_extraction_report(
    file_reports: list[dict[str, Any]],
    extracted_dir: Path,
    report_dir: Path,
    parquet_chunk_size: int,
) -> dict[str, Any]:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    charge_path = extracted_file("charge", extracted_dir)
    discharge_path = extracted_file("discharge", extracted_dir)
    file_report_path = report_dir / "extraction_file_report.csv"
    pd.DataFrame(file_reports).to_csv(file_report_path, index=False, encoding="utf-8-sig")

    ok_reports = [item for item in file_reports if item.get("status") == "ok"]
    charge_batteries = {str(item.get("battery_id", "")) for item in ok_reports if int(item.get("n_charge_rows", 0) or 0) > 0}
    discharge_batteries = {str(item.get("battery_id", "")) for item in ok_reports if int(item.get("n_discharge_rows", 0) or 0) > 0}

    report = {
        "dataset": DATASET_NAME,
        "adapter_version": ADAPTER_VERSION,
        "memory_mode": "streaming_per_cycle_to_parquet",
        "n_files": int(len(file_reports)),
        "n_loaded_files": int(sum(1 for item in file_reports if item.get("status") == "ok")),
        "n_failed_files": int(sum(1 for item in file_reports if item.get("status") != "ok")),
        "n_charge_rows": int(sum(int(item.get("n_charge_rows", 0) or 0) for item in file_reports)),
        "n_discharge_rows": int(sum(int(item.get("n_discharge_rows", 0) or 0) for item in file_reports)),
        "n_charge_batteries": int(len(charge_batteries)),
        "n_discharge_batteries": int(len(discharge_batteries)),
        "parquet_chunk_size": int(parquet_chunk_size),
        "outputs": {
            "charge_extracted_cycles": str(charge_path),
            "discharge_extracted_cycles": str(discharge_path),
            "file_report_csv": str(file_report_path),
            "extraction_report_json": str(report_dir / "extraction_report.json"),
        },
    }
    write_json(report_dir / "extraction_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract XJTU charge and discharge cycles to standardized extracted parquet files.")
    parser.add_argument("--raw-dir", type=str, default=None, help="Raw XJTU directory. Default: data/raw/XJTU")
    parser.add_argument("--extracted-dir", type=str, default=None, help="Output directory for extracted parquet files.")
    parser.add_argument("--report-dir", type=str, default=None, help="Report directory. Default: data/reports/xjtu")
    parser.add_argument("--first-cycle-used", type=int, default=DEFAULT_FIRST_CYCLE_USED, help="First original cycle to extract. Default: 2.")
    parser.add_argument("--rescale-rel-tol", type=float, default=DEFAULT_RESCALE_REL_TOL, help="Relative tolerance for rescaling capacity to summary capacity. Default: 0.05.")
    parser.add_argument("--limit-files", type=int, default=None, help="Process only the first N files for testing.")
    parser.add_argument("--parquet-chunk-size", type=int, default=5, help="Rows kept in RAM before parquet flush. Default: 5. Use 1 for the lowest memory use.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else default_raw_dir().resolve()
    extracted_dir = Path(args.extracted_dir).resolve() if args.extracted_dir else default_extracted_dir().resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else default_report_dir().resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw XJTU directory not found: {raw_dir}")

    mat_files = discover_mat_files(raw_dir)
    if args.limit_files is not None:
        mat_files = mat_files[: args.limit_files]
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under: {raw_dir}")

    print("=" * 80)
    print("XJTU cycle extraction")
    print("=" * 80)
    print(f"Raw directory:       {raw_dir}")
    print(f"Extracted directory: {extracted_dir}")
    print(f"Report directory:    {report_dir}")
    print(f"MAT files found:     {len(mat_files)}")
    print(f"Memory mode:         streaming per cycle")
    print(f"Parquet chunk size:  {args.parquet_chunk_size}")

    extracted_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    charge_writer = BufferedParquetWriter(extracted_file("charge", extracted_dir), args.parquet_chunk_size, "charge")
    discharge_writer = BufferedParquetWriter(extracted_file("discharge", extracted_dir), args.parquet_chunk_size, "discharge")

    file_reports: list[dict[str, Any]] = []
    try:
        for battery_index, mat_path in enumerate(tqdm(mat_files, desc="XJTU files", unit="file"), start=1):
            report = extract_one_file(
                mat_path=mat_path,
                raw_dir=raw_dir,
                battery_index=battery_index,
                first_cycle_used=args.first_cycle_used,
                rescale_rel_tol=args.rescale_rel_tol,
                charge_writer=charge_writer,
                discharge_writer=discharge_writer,
            )
            file_reports.append(report)
            gc.collect()
    finally:
        charge_writer.close()
        discharge_writer.close()

    report = save_extraction_report(file_reports, extracted_dir, report_dir, parquet_chunk_size=args.parquet_chunk_size)

    print("\nExtraction completed.")
    print(f"Charge rows:      {report['n_charge_rows']}")
    print(f"Discharge rows:   {report['n_discharge_rows']}")
    print(f"Failed files:     {report['n_failed_files']}")
    print(f"Charge output:    {report['outputs']['charge_extracted_cycles']}")
    print(f"Discharge output: {report['outputs']['discharge_extracted_cycles']}")


if __name__ == "__main__":
    main()
