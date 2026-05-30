# -*- coding: utf-8 -*-
"""
Clean and standardize extracted XJTU charge/discharge cycles.

Version 1.3 uses a two-pass streaming strategy. The first pass computes robust
initial capacity references per battery and phase from valid, representative
cycles. The second pass writes cleaned rows using those references without
loading the full dataset into memory.

Inputs:
    data/interim/xjtu/extracted/xjtu_charge_extracted_cycles.parquet
    data/interim/xjtu/extracted/xjtu_discharge_extracted_cycles.parquet

Outputs:
    data/interim/xjtu/clean/xjtu_charge_clean_cycles.parquet
    data/interim/xjtu/clean/xjtu_discharge_clean_cycles.parquet
    data/interim/xjtu/clean/xjtu_charge_rejected_cycles.csv
    data/interim/xjtu/clean/xjtu_discharge_rejected_cycles.csv
    data/reports/xjtu/cleaning_report.json
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from xjtu_schema import (
    CLEANING_VERSION,
    STANDARD_CYCLE_COLUMNS,
    VECTOR_COLUMNS,
    as_1d_float,
    clean_file,
    compute_power,
    cumulative_capacity_ah,
    cumulative_energy_wh,
    default_clean_dir,
    default_extracted_dir,
    default_report_dir,
    ensure_standard_columns,
    extracted_file,
    rejected_file,
    safe_max,
    safe_mean,
    safe_median_diff,
    safe_min,
    utc_now_iso,
    validate_schema,
    write_json,
)

DEFAULT_MIN_POINTS = 20
DEFAULT_MIN_VOLTAGE_RANGE = 0.05
DEFAULT_REBOUND_TOLERANCE_V = 0.03
DEFAULT_EOL_FRACTION = 0.80
DEFAULT_CAPACITY_RESCALE_REL_TOL = 0.05
DEFAULT_REFERENCE_MIN_CAPACITY_AH = 1.0
DEFAULT_REFERENCE_MAX_CAPACITY_AH = 2.3
DEFAULT_REFERENCE_WINDOW_CYCLES = 5


def load_extracted_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        return ensure_standard_columns(pd.DataFrame())
    df = pd.read_parquet(path)
    return ensure_standard_columns(df)


def align_vectors(row: pd.Series) -> tuple[dict[str, np.ndarray], int]:
    arrays = {col: as_1d_float(row.get(col, [])) for col in VECTOR_COLUMNS}
    positive_lengths = [len(arr) for arr in arrays.values() if len(arr) > 0]
    if not positive_lengths:
        return {col: np.array([], dtype=float) for col in VECTOR_COLUMNS}, 0

    min_len = min(positive_lengths)
    original_max_len = max(positive_lengths)

    for col in VECTOR_COLUMNS:
        if len(arrays[col]) == 0:
            arrays[col] = np.full(min_len, np.nan, dtype=float)
        else:
            arrays[col] = arrays[col][:min_len].astype(float)

    base_mask = np.isfinite(arrays["time_s"]) & np.isfinite(arrays["voltage_v"])
    if np.any(np.isfinite(arrays["current_a"])):
        base_mask = base_mask & np.isfinite(arrays["current_a"])
    # Temperature, capacity, energy, and power may be reconstructed later.

    for col in VECTOR_COLUMNS:
        arrays[col] = arrays[col][base_mask]

    removed = int(original_max_len - len(arrays["time_s"]))
    return arrays, removed


def sort_and_deduplicate_time(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    time_s = arrays["time_s"]
    if len(time_s) == 0:
        return arrays, 0

    order = np.argsort(time_s, kind="mergesort")
    for col in VECTOR_COLUMNS:
        arrays[col] = arrays[col][order]

    t = arrays["time_s"]
    keep = np.ones(len(t), dtype=bool)
    if len(t) >= 2:
        keep[1:] = np.diff(t) > 0
    removed = int(np.count_nonzero(~keep))

    for col in VECTOR_COLUMNS:
        arrays[col] = arrays[col][keep]

    if len(arrays["time_s"]):
        arrays["time_s"] = arrays["time_s"] - arrays["time_s"][0]

    return arrays, removed


def trim_voltage_edges(
    arrays: dict[str, np.ndarray],
    phase: str,
    rebound_tolerance_v: float,
) -> tuple[dict[str, np.ndarray], int, int, list[str]]:
    warnings: list[str] = []
    voltage = arrays["voltage_v"]
    if len(voltage) < 5 or not np.any(np.isfinite(voltage)):
        return arrays, 0, max(len(voltage) - 1, 0), warnings

    start = 0
    end = len(voltage) - 1

    # Remove post-cutoff voltage rebound in discharge curves.
    if phase == "discharge":
        min_idx = int(np.nanargmin(voltage))
        if min_idx < len(voltage) - 2 and voltage[-1] > voltage[min_idx] + rebound_tolerance_v:
            end = min_idx
            warnings.append("trimmed_post_cutoff_voltage_rebound")

    # Remove artificial voltage drop after charge termination.
    elif phase == "charge":
        max_idx = int(np.nanargmax(voltage))
        if max_idx < len(voltage) - 2 and voltage[-1] < voltage[max_idx] - rebound_tolerance_v:
            end = max_idx
            warnings.append("trimmed_post_charge_voltage_drop")

    if end < start:
        end = len(voltage) - 1

    trimmed = {col: values[start : end + 1] for col, values in arrays.items()}
    removed = int(len(voltage) - len(trimmed["voltage_v"]))
    if removed > 0 and len(trimmed["time_s"]):
        trimmed["time_s"] = trimmed["time_s"] - trimmed["time_s"][0]
    return trimmed, start, end, warnings


def enforce_current_sign(current_a: np.ndarray, phase: str) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    current = as_1d_float(current_a)
    if not np.any(np.isfinite(current)):
        return current, warnings

    mean_current = safe_mean(current)
    if phase == "charge" and np.isfinite(mean_current) and mean_current < 0:
        current = -current
        warnings.append("flipped_current_sign_to_positive_charge")
    elif phase == "discharge" and np.isfinite(mean_current) and mean_current > 0:
        current = -current
        warnings.append("flipped_current_sign_to_negative_discharge")
    return current, warnings


def maybe_rescale_capacity_to_reference(
    capacity_ah: np.ndarray,
    reference_capacity_ah: float,
    rel_tol: float,
) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    capacity = as_1d_float(capacity_ah)
    if len(capacity) == 0:
        return capacity, warnings
    if not np.isfinite(reference_capacity_ah) or reference_capacity_ah <= 0:
        return capacity, warnings

    end_value = float(capacity[-1]) if np.isfinite(capacity[-1]) else np.nan
    if not np.isfinite(end_value) or end_value <= 0:
        capacity = np.linspace(0.0, reference_capacity_ah, len(capacity))
        warnings.append("capacity_rebuilt_as_linear_vector_from_reference")
        return capacity, warnings

    rel_error = abs(end_value - reference_capacity_ah) / max(abs(reference_capacity_ah), 1e-12)
    if rel_error > rel_tol:
        capacity = capacity * (reference_capacity_ah / end_value)
        warnings.append("capacity_rescaled_to_extracted_reference_capacity")
    return capacity, warnings


def can_interpolate_qv_curve(voltage_v: np.ndarray, capacity_ah: np.ndarray, min_voltage_range: float) -> bool:
    voltage = as_1d_float(voltage_v)
    capacity = as_1d_float(capacity_ah)
    n = min(len(voltage), len(capacity))
    if n < 5:
        return False
    voltage = voltage[:n]
    capacity = capacity[:n]
    mask = np.isfinite(voltage) & np.isfinite(capacity)
    voltage = voltage[mask]
    capacity = capacity[mask]
    if len(voltage) < 5:
        return False
    if safe_max(voltage) - safe_min(voltage) < min_voltage_range:
        return False
    return len(np.unique(np.round(voltage, 5))) >= 5


def build_scalars(
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
        "phase_capacity_ah": float(capacity_ah[-1]) if len(capacity_ah) and np.isfinite(capacity_ah[-1]) else np.nan,
        "phase_energy_wh": float(energy_wh[-1]) if len(energy_wh) and np.isfinite(energy_wh[-1]) else np.nan,
    }


def validity_flags(
    phase: str,
    scalars: dict[str, Any],
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    capacity_ah: np.ndarray,
    cycle_life: float,
    remaining_cycles: float,
    min_points: int,
    min_voltage_range: float,
) -> tuple[bool, bool, str]:
    reasons: list[str] = []

    if scalars["n_points"] < min_points:
        reasons.append("insufficient_points")
    if len(time_s) < 2 or not np.all(np.diff(time_s) > 0):
        reasons.append("time_not_strictly_increasing")
    if not np.any(np.isfinite(voltage_v)):
        reasons.append("missing_voltage")
    if not np.any(np.isfinite(current_a)) and not np.any(np.isfinite(capacity_ah)):
        reasons.append("missing_current_and_capacity")
    if not np.isfinite(scalars["phase_capacity_ah"]) or scalars["phase_capacity_ah"] <= 0:
        reasons.append("invalid_phase_capacity")
    if not np.isfinite(scalars["voltage_min_v"]) or not np.isfinite(scalars["voltage_max_v"]):
        reasons.append("invalid_voltage_range")
    elif scalars["voltage_max_v"] - scalars["voltage_min_v"] < min_voltage_range:
        reasons.append("insufficient_voltage_range")
    if not np.isfinite(scalars["duration_s"]) or scalars["duration_s"] <= 0:
        reasons.append("invalid_duration")

    current_mean = scalars["current_mean_a"]
    if np.any(np.isfinite(current_a)):
        if phase == "charge" and np.isfinite(current_mean) and current_mean <= 0:
            reasons.append("invalid_charge_current_sign")
        if phase == "discharge" and np.isfinite(current_mean) and current_mean >= 0:
            reasons.append("invalid_discharge_current_sign")

    is_valid_cycle = len(reasons) == 0
    interpolation_ok = can_interpolate_qv_curve(voltage_v, capacity_ah, min_voltage_range)
    is_model_usable = bool(
        is_valid_cycle
        and np.isfinite(cycle_life)
        and np.isfinite(remaining_cycles)
        and remaining_cycles > 0
        and np.isfinite(scalars["phase_capacity_ah"])
        and interpolation_ok
        and phase in {"charge", "discharge"}
    )

    if is_valid_cycle and not is_model_usable:
        if not np.isfinite(cycle_life):
            reasons.append("missing_cycle_life")
        elif not np.isfinite(remaining_cycles) or remaining_cycles <= 0:
            reasons.append("non_positive_remaining_cycles")
        elif not interpolation_ok:
            reasons.append("qv_interpolation_not_possible")

    return is_valid_cycle, is_model_usable, ";".join(reasons)


def clean_one_row(
    row: pd.Series,
    min_points: int,
    min_voltage_range: float,
    rebound_tolerance_v: float,
    eol_fraction: float,
    capacity_rescale_rel_tol: float,
) -> dict[str, Any]:
    phase = str(row.get("phase", "")).lower().strip()
    warnings: list[str] = []
    original_lengths = [len(as_1d_float(row.get(col, []))) for col in VECTOR_COLUMNS]
    original_length = max(original_lengths) if original_lengths else 0

    arrays, removed_invalid = align_vectors(row)
    arrays, removed_duplicate = sort_and_deduplicate_time(arrays)
    arrays, trim_start, trim_end, trim_warnings = trim_voltage_edges(arrays, phase, rebound_tolerance_v)
    warnings.extend(trim_warnings)
    warnings.extend([str(x) for x in row.get("cleaning_warnings", []) if str(x)])

    arrays["current_a"], sign_warnings = enforce_current_sign(arrays["current_a"], phase)
    warnings.extend(sign_warnings)

    # Reconstruct capacity and energy after sign normalization and trimming.
    if np.any(np.isfinite(arrays["current_a"])):
        reconstructed_capacity = cumulative_capacity_ah(arrays["time_s"], arrays["current_a"])
        reference_capacity = float(row.get("phase_capacity_ah", np.nan))
        reconstructed_capacity, cap_warnings = maybe_rescale_capacity_to_reference(
            reconstructed_capacity,
            reference_capacity,
            capacity_rescale_rel_tol,
        )
        arrays["capacity_ah"] = reconstructed_capacity
        warnings.extend(cap_warnings)
        capacity_source = "integrated_current"
    else:
        capacity_source = str(row.get("capacity_source", "missing")) or "missing"

    arrays["power_w"] = compute_power(arrays["voltage_v"], arrays["current_a"])
    arrays["energy_wh"] = cumulative_energy_wh(arrays["time_s"], arrays["voltage_v"], arrays["current_a"])

    n = min(len(arrays[col]) for col in VECTOR_COLUMNS)
    for col in VECTOR_COLUMNS:
        arrays[col] = arrays[col][:n]

    scalars = build_scalars(
        arrays["time_s"], arrays["voltage_v"], arrays["current_a"], arrays["temperature_c"], arrays["capacity_ah"], arrays["energy_wh"]
    )

    cycle_life = float(row.get("cycle_life", np.nan)) if pd.notna(row.get("cycle_life", np.nan)) else np.nan
    cycle_number = float(row.get("cycle_number", np.nan)) if pd.notna(row.get("cycle_number", np.nan)) else np.nan
    remaining_cycles = cycle_life - cycle_number if np.isfinite(cycle_life) and np.isfinite(cycle_number) else np.nan
    remaining_cycles_norm = remaining_cycles / cycle_life if np.isfinite(remaining_cycles) and np.isfinite(cycle_life) and cycle_life > 0 else np.nan

    is_valid_cycle, is_model_usable, exclude_reason = validity_flags(
        phase,
        scalars,
        arrays["time_s"],
        arrays["voltage_v"],
        arrays["current_a"],
        arrays["capacity_ah"],
        cycle_life,
        remaining_cycles,
        min_points,
        min_voltage_range,
    )

    has_temperature = bool(np.any(np.isfinite(arrays["temperature_c"])))
    temperature_source = "measured" if has_temperature else "unknown_nan"
    n_removed_points = int(max(original_length - n, 0) + removed_invalid + removed_duplicate)

    out = row.to_dict()
    out.update({
        "cycle_life": cycle_life,
        "remaining_cycles": remaining_cycles,
        "remaining_cycles_norm": remaining_cycles_norm,
        "eol_criterion": str(row.get("eol_criterion", "provided_by_dataset")) or "provided_by_dataset",
        "eol_capacity_ah": np.nan,
        "initial_capacity_ah": np.nan,
        "nominal_capacity_ah": float(row.get("nominal_capacity_ah", np.nan)) if pd.notna(row.get("nominal_capacity_ah", np.nan)) else np.nan,
        "soh": np.nan,
        **{col: arrays[col].astype(float).tolist() for col in VECTOR_COLUMNS},
        **scalars,
        "voltage_source": "measured" if np.any(np.isfinite(arrays["voltage_v"])) else "missing",
        "current_source": "measured" if np.any(np.isfinite(arrays["current_a"])) else "missing",
        "temperature_source": temperature_source,
        "capacity_source": capacity_source,
        "energy_source": "integrated_power" if np.any(np.isfinite(arrays["energy_wh"])) else "missing",
        "is_valid_cycle": bool(is_valid_cycle),
        "is_model_usable": bool(is_model_usable),
        "exclude_reason": "" if is_model_usable else exclude_reason,
        "cleaning_warnings": sorted(set(warnings)),
        "n_removed_points": n_removed_points,
        "trim_start_index": int(trim_start),
        "trim_end_index": int(trim_end),
        "has_voltage": bool(np.any(np.isfinite(arrays["voltage_v"]))),
        "has_current": bool(np.any(np.isfinite(arrays["current_a"]))),
        "has_temperature": has_temperature,
        "has_capacity": bool(np.any(np.isfinite(arrays["capacity_ah"]))),
        "has_energy": bool(np.any(np.isfinite(arrays["energy_wh"]))),
        "cleaning_version": CLEANING_VERSION,
        "created_at": utc_now_iso(),
    })

    return out


def assign_initial_capacity_and_soh(df: pd.DataFrame, eol_fraction: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["initial_capacity_ah"] = np.nan
    out["soh"] = np.nan
    out["eol_capacity_ah"] = np.nan

    for (battery_id, phase), idx in out.groupby(["battery_id", "phase"]).groups.items():
        sub = out.loc[list(idx)].sort_values("cycle_number")
        usable_capacity = sub.loc[np.isfinite(sub["phase_capacity_ah"]) & (sub["phase_capacity_ah"] > 0), "phase_capacity_ah"]
        if usable_capacity.empty:
            continue
        initial = float(usable_capacity.iloc[0])
        out.loc[list(idx), "initial_capacity_ah"] = initial
        out.loc[list(idx), "eol_capacity_ah"] = initial * eol_fraction if np.isfinite(initial) else np.nan
        out.loc[list(idx), "soh"] = out.loc[list(idx), "phase_capacity_ah"] / initial if initial > 0 else np.nan

    return out


def assign_cycle_indices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values(["battery_id", "phase", "cycle_number"]).copy()
    out["cycle_index"] = out.groupby(["battery_id", "phase"]).cumcount() + 1
    return out.sort_values(["battery_id", "phase", "cycle_index"]).reset_index(drop=True)


def clean_phase_dataframe(
    df: pd.DataFrame,
    phase: str,
    min_points: int,
    min_voltage_range: float,
    rebound_tolerance_v: float,
    eol_fraction: float,
    capacity_rescale_rel_tol: float,
) -> pd.DataFrame:
    df = ensure_standard_columns(df)
    if df.empty:
        return df
    df = df[df["phase"].astype(str).str.lower() == phase].copy()
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Cleaning {phase}", unit="cycle"):
        rows.append(
            clean_one_row(
                row,
                min_points=min_points,
                min_voltage_range=min_voltage_range,
                rebound_tolerance_v=rebound_tolerance_v,
                eol_fraction=eol_fraction,
                capacity_rescale_rel_tol=capacity_rescale_rel_tol,
            )
        )
    out = ensure_standard_columns(pd.DataFrame(rows))
    out = assign_initial_capacity_and_soh(out, eol_fraction)
    out = assign_cycle_indices(out)
    out = ensure_standard_columns(out)
    validate_schema(out)
    return out


def standard_arrow_schema() -> pa.Schema:
    vector_float = pa.list_(pa.float32())
    list_string = pa.list_(pa.string())
    int_cols = {"battery_index", "cycle_index", "n_points", "n_removed_points", "trim_start_index", "trim_end_index"}
    bool_cols = {"is_valid_cycle", "is_model_usable", "has_voltage", "has_current", "has_temperature", "has_capacity", "has_energy"}
    vector_cols = set(VECTOR_COLUMNS)
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


def coerce_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_standard_columns(df)
    if out.empty:
        return out
    out = out.copy()
    for col in VECTOR_COLUMNS:
        out[col] = out[col].apply(lambda value: as_1d_float(value).astype("float32", copy=False).tolist())
    out["cleaning_warnings"] = out["cleaning_warnings"].apply(
        lambda value: [str(item) for item in value] if isinstance(value, list) else ([] if pd.isna(value) else [str(value)])
    )
    return out


def write_dataframe_to_parquet_chunked(df: pd.DataFrame, output_path: Path, chunk_size: int = 50) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = standard_arrow_schema()
    df = ensure_standard_columns(df)
    if df.empty:
        empty_table = pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)
        pq.write_table(empty_table, output_path)
        return

    writer: pq.ParquetWriter | None = None
    try:
        iterator = range(0, len(df), max(int(chunk_size), 1))
        for start_idx in tqdm(iterator, desc=f"Writing {output_path.name}", unit="chunk", leave=False):
            chunk = coerce_dataframe_for_parquet(df.iloc[start_idx:start_idx + chunk_size])
            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False, safe=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            del chunk, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()


def rejected_view(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "dataset", "source_file", "source_path", "battery_id", "cycle_index",
            "cycle_number", "phase", "cycle_life", "remaining_cycles",
            "is_valid_cycle", "is_model_usable", "exclude_reason", "cleaning_warnings",
            "n_points", "duration_s", "voltage_min_v", "voltage_max_v", "phase_capacity_ah",
        ])
    cols = [
        "dataset", "source_file", "source_path", "battery_id", "cycle_index",
        "cycle_number", "phase", "cycle_life", "remaining_cycles",
        "is_valid_cycle", "is_model_usable", "exclude_reason", "cleaning_warnings",
        "n_points", "duration_s", "voltage_min_v", "voltage_max_v", "phase_capacity_ah",
    ]
    rejected = df[~df["is_model_usable"].astype(bool)].copy()
    return rejected[cols]


def save_clean_outputs(cleaned: dict[str, pd.DataFrame], clean_dir: Path, report_dir: Path, parquet_chunk_size: int = 50) -> dict[str, Any]:
    clean_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    summary: dict[str, Any] = {}
    for phase, df in cleaned.items():
        parquet_path = clean_file(phase, clean_dir)
        reject_path = rejected_file(phase, clean_dir)
        write_dataframe_to_parquet_chunked(df, parquet_path, chunk_size=parquet_chunk_size)
        rejected_view(df).to_csv(reject_path, index=False, encoding="utf-8-sig")
        outputs[f"{phase}_clean_cycles"] = str(parquet_path)
        outputs[f"{phase}_rejected_cycles"] = str(reject_path)
        summary[phase] = {
            "n_rows": int(len(df)),
            "n_batteries": int(df["battery_id"].nunique()) if not df.empty else 0,
            "n_valid_cycles": int(df["is_valid_cycle"].sum()) if not df.empty else 0,
            "n_model_usable_cycles": int(df["is_model_usable"].sum()) if not df.empty else 0,
            "n_rejected_cycles": int((~df["is_model_usable"].astype(bool)).sum()) if not df.empty else 0,
            "exclude_reason_counts": df.loc[~df["is_model_usable"].astype(bool), "exclude_reason"].value_counts(dropna=False).to_dict() if not df.empty else {},
        }

    report = {
        "cleaning_version": CLEANING_VERSION,
        "summary": summary,
        "outputs": outputs,
    }
    report_path = report_dir / "cleaning_report.json"
    report["outputs"]["cleaning_report_json"] = str(report_path)
    write_json(report_path, report)
    return report




class BufferedCleanParquetWriter:
    """Small-buffer parquet writer used by the streaming cleaner."""

    def __init__(self, output_path: Path, chunk_size: int):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_size = max(int(chunk_size), 1)
        self.schema = standard_arrow_schema()
        self.writer: pq.ParquetWriter | None = None
        self.buffer: list[dict[str, Any]] = []
        self.n_rows = 0
        self.closed = False

    def write(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        chunk = coerce_dataframe_for_parquet(pd.DataFrame(self.buffer))
        table = pa.Table.from_pandas(chunk, schema=self.schema, preserve_index=False, safe=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.output_path, table.schema, compression="snappy")
        self.writer.write_table(table)
        self.n_rows += len(self.buffer)
        self.buffer.clear()
        del chunk, table
        gc.collect()

    def close(self) -> None:
        if self.closed:
            return
        self.flush()
        if self.writer is None:
            empty_df = ensure_standard_columns(pd.DataFrame())
            empty_table = pa.Table.from_pandas(empty_df, schema=self.schema, preserve_index=False, safe=False)
            pq.write_table(empty_table, self.output_path)
            del empty_df, empty_table
        else:
            self.writer.close()
        self.closed = True
        gc.collect()


def rejected_columns() -> list[str]:
    return [
        "dataset", "source_file", "source_path", "battery_id", "cycle_index",
        "cycle_number", "phase", "cycle_life", "remaining_cycles",
        "is_valid_cycle", "is_model_usable", "exclude_reason", "cleaning_warnings",
        "n_points", "duration_s", "voltage_min_v", "voltage_max_v", "phase_capacity_ah",
    ]


def write_rejected_row_csv(row: dict[str, Any], reject_path: Path, header: bool) -> None:
    cols = rejected_columns()
    reject_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{col: row.get(col, np.nan) for col in cols}]).to_csv(
        reject_path,
        mode="a",
        index=False,
        header=header,
        encoding="utf-8-sig",
    )


def initialize_rejected_csv(reject_path: Path) -> None:
    reject_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=rejected_columns()).to_csv(reject_path, index=False, encoding="utf-8-sig")



def is_reference_capacity_candidate(
    cleaned: dict[str, Any],
    reference_min_capacity_ah: float,
    reference_max_capacity_ah: float,
) -> bool:
    """Return True when a cleaned row is suitable for initial-capacity reference."""
    if not bool(cleaned.get("is_valid_cycle", False)):
        return False
    cap = cleaned.get("phase_capacity_ah", np.nan)
    try:
        cap = float(cap)
    except Exception:
        return False
    if not np.isfinite(cap):
        return False
    if cap <= 0:
        return False
    if np.isfinite(reference_min_capacity_ah) and cap < reference_min_capacity_ah:
        return False
    if np.isfinite(reference_max_capacity_ah) and cap > reference_max_capacity_ah:
        return False
    return True


def collect_reference_capacities_streaming(
    *,
    input_path: Path,
    phase: str,
    min_points: int,
    min_voltage_range: float,
    rebound_tolerance_v: float,
    eol_fraction: float,
    capacity_rescale_rel_tol: float,
    read_batch_size: int,
    reference_min_capacity_ah: float,
    reference_max_capacity_ah: float,
    reference_window_cycles: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    """First pass: compute robust initial-capacity references per battery/phase.

    This pass intentionally does not write rows. It keeps only a few reference
    capacities for each battery, which avoids the RAM problem observed when the
    whole XJTU dataset was kept as Python objects.
    """
    references: dict[tuple[str, str], dict[str, Any]] = {}
    window = max(int(reference_window_cycles), 1)

    if not input_path.exists():
        return references

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows if pf.metadata is not None else None
    if total_rows == 0:
        return references

    progress = tqdm(total=total_rows, desc=f"Reference pass {phase}", unit="cycle")
    try:
        for batch in pf.iter_batches(batch_size=max(int(read_batch_size), 1)):
            df_batch = ensure_standard_columns(batch.to_pandas())
            if df_batch.empty:
                progress.update(len(batch))
                continue
            df_batch = df_batch[df_batch["phase"].astype(str).str.lower() == phase]
            for _, row in df_batch.iterrows():
                cleaned = clean_one_row(
                    row,
                    min_points=min_points,
                    min_voltage_range=min_voltage_range,
                    rebound_tolerance_v=rebound_tolerance_v,
                    eol_fraction=eol_fraction,
                    capacity_rescale_rel_tol=capacity_rescale_rel_tol,
                )
                battery_id = str(cleaned.get("battery_id", ""))
                key = (battery_id, phase)
                if battery_id:
                    item = references.setdefault(
                        key,
                        {
                            "candidate_capacities": [],
                            "fallback_valid_capacities": [],
                            "n_reference_candidates": 0,
                            "n_valid_positive": 0,
                            "reference_mode": "missing",
                        },
                    )
                    cap = float(cleaned.get("phase_capacity_ah", np.nan)) if pd.notna(cleaned.get("phase_capacity_ah", np.nan)) else np.nan
                    if bool(cleaned.get("is_valid_cycle", False)) and np.isfinite(cap) and cap > 0:
                        item["n_valid_positive"] += 1
                        if len(item["fallback_valid_capacities"]) < window:
                            item["fallback_valid_capacities"].append(cap)
                    if is_reference_capacity_candidate(cleaned, reference_min_capacity_ah, reference_max_capacity_ah):
                        item["n_reference_candidates"] += 1
                        if len(item["candidate_capacities"]) < window:
                            item["candidate_capacities"].append(cap)
                progress.update(1)
                del cleaned
            del df_batch, batch
            gc.collect()
    finally:
        progress.close()

    # Finalize reference capacity with robust median of the first representative cycles.
    for key, item in references.items():
        candidates = [float(x) for x in item.get("candidate_capacities", []) if np.isfinite(x)]
        fallbacks = [float(x) for x in item.get("fallback_valid_capacities", []) if np.isfinite(x)]
        if candidates:
            item["initial_capacity_ah"] = float(np.nanmedian(candidates))
            item["reference_mode"] = "median_first_representative_valid_cycles"
        elif fallbacks:
            item["initial_capacity_ah"] = float(np.nanmedian(fallbacks))
            item["reference_mode"] = "fallback_median_first_valid_positive_cycles"
        else:
            item["initial_capacity_ah"] = np.nan
            item["reference_mode"] = "missing_no_valid_positive_capacity"
        item["eol_capacity_ah"] = (
            float(item["initial_capacity_ah"] * eol_fraction)
            if np.isfinite(item.get("initial_capacity_ah", np.nan))
            else np.nan
        )

    return references


def apply_reference_capacity(
    cleaned: dict[str, Any],
    reference_info: dict[str, Any] | None,
    eol_fraction: float,
) -> dict[str, Any]:
    """Fill initial capacity, EOL capacity, and SOH using precomputed references."""
    if reference_info is None:
        init = np.nan
        eol = np.nan
        mode = "missing_reference_entry"
    else:
        init = float(reference_info.get("initial_capacity_ah", np.nan)) if pd.notna(reference_info.get("initial_capacity_ah", np.nan)) else np.nan
        eol = float(reference_info.get("eol_capacity_ah", np.nan)) if pd.notna(reference_info.get("eol_capacity_ah", np.nan)) else np.nan
        mode = str(reference_info.get("reference_mode", "unknown"))

    cap = float(cleaned.get("phase_capacity_ah", np.nan)) if pd.notna(cleaned.get("phase_capacity_ah", np.nan)) else np.nan
    cleaned["initial_capacity_ah"] = init
    cleaned["eol_capacity_ah"] = eol if np.isfinite(eol) else (init * eol_fraction if np.isfinite(init) else np.nan)
    cleaned["soh"] = cap / init if np.isfinite(cap) and np.isfinite(init) and init > 0 else np.nan

    warnings = cleaned.get("cleaning_warnings", [])
    if not isinstance(warnings, list):
        warnings = [] if pd.isna(warnings) else [str(warnings)]
    if mode.startswith("fallback"):
        warnings.append("initial_capacity_reference_used_fallback_valid_cycles")
    elif mode.startswith("missing"):
        warnings.append("initial_capacity_reference_missing")
    cleaned["cleaning_warnings"] = sorted(set(str(w) for w in warnings if str(w)))
    return cleaned


def clean_phase_file_streaming(
    *,
    input_path: Path,
    output_path: Path,
    reject_path: Path,
    phase: str,
    min_points: int,
    min_voltage_range: float,
    rebound_tolerance_v: float,
    eol_fraction: float,
    capacity_rescale_rel_tol: float,
    parquet_chunk_size: int,
    read_batch_size: int,
    reference_min_capacity_ah: float,
    reference_max_capacity_ah: float,
    reference_window_cycles: int,
) -> dict[str, Any]:
    """Clean one extracted phase file using bounded memory.

    Version 1.3 uses two streaming passes:
    1. collect a robust initial-capacity reference per battery/phase using only
       valid and representative cycles;
    2. clean again and write rows directly to parquet/CSV using those references.

    The second pass preserves bounded RAM use while avoiding the earlier SOH
    issue in which short invalid discharge segments could become the reference
    initial capacity.
    """
    references = collect_reference_capacities_streaming(
        input_path=input_path,
        phase=phase,
        min_points=min_points,
        min_voltage_range=min_voltage_range,
        rebound_tolerance_v=rebound_tolerance_v,
        eol_fraction=eol_fraction,
        capacity_rescale_rel_tol=capacity_rescale_rel_tol,
        read_batch_size=read_batch_size,
        reference_min_capacity_ah=reference_min_capacity_ah,
        reference_max_capacity_ah=reference_max_capacity_ah,
        reference_window_cycles=reference_window_cycles,
    )

    writer = BufferedCleanParquetWriter(output_path, parquet_chunk_size)
    initialize_rejected_csv(reject_path)

    summary = {
        "n_rows": 0,
        "n_batteries": 0,
        "n_valid_cycles": 0,
        "n_model_usable_cycles": 0,
        "n_rejected_cycles": 0,
        "exclude_reason_counts": {},
        "reference_capacity_policy": {
            "reference_min_capacity_ah": reference_min_capacity_ah,
            "reference_max_capacity_ah": reference_max_capacity_ah,
            "reference_window_cycles": reference_window_cycles,
            "n_reference_entries": len(references),
            "n_fallback_references": int(sum(1 for item in references.values() if str(item.get("reference_mode", "")).startswith("fallback"))),
            "n_missing_references": int(sum(1 for item in references.values() if str(item.get("reference_mode", "")).startswith("missing"))),
        },
    }
    batteries: set[str] = set()
    cycle_counter: dict[tuple[str, str], int] = {}

    if not input_path.exists():
        writer.close()
        return summary

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows if pf.metadata is not None else None
    if total_rows == 0:
        writer.close()
        return summary

    progress = tqdm(total=total_rows, desc=f"Cleaning {phase}", unit="cycle")
    try:
        for batch in pf.iter_batches(batch_size=max(int(read_batch_size), 1)):
            df_batch = ensure_standard_columns(batch.to_pandas())
            if df_batch.empty:
                progress.update(len(batch))
                continue
            df_batch = df_batch[df_batch["phase"].astype(str).str.lower() == phase]
            for _, row in df_batch.iterrows():
                cleaned = clean_one_row(
                    row,
                    min_points=min_points,
                    min_voltage_range=min_voltage_range,
                    rebound_tolerance_v=rebound_tolerance_v,
                    eol_fraction=eol_fraction,
                    capacity_rescale_rel_tol=capacity_rescale_rel_tol,
                )

                battery_id = str(cleaned.get("battery_id", ""))
                key = (battery_id, phase)
                batteries.add(battery_id)
                cycle_counter[key] = cycle_counter.get(key, 0) + 1
                cleaned["cycle_index"] = cycle_counter[key]
                cleaned = apply_reference_capacity(cleaned, references.get(key), eol_fraction)

                writer.write(cleaned)

                summary["n_rows"] += 1
                if bool(cleaned.get("is_valid_cycle", False)):
                    summary["n_valid_cycles"] += 1
                if bool(cleaned.get("is_model_usable", False)):
                    summary["n_model_usable_cycles"] += 1
                else:
                    summary["n_rejected_cycles"] += 1
                    reason = str(cleaned.get("exclude_reason", ""))
                    summary["exclude_reason_counts"][reason] = summary["exclude_reason_counts"].get(reason, 0) + 1
                    write_rejected_row_csv(cleaned, reject_path, header=False)

                progress.update(1)
                del cleaned
            del df_batch, batch
            gc.collect()
    finally:
        progress.close()
        writer.close()

    summary["n_batteries"] = len([b for b in batteries if b])

    # Compact reference diagnostics for report inspection without bloating JSON.
    reference_values = [float(item.get("initial_capacity_ah", np.nan)) for item in references.values() if np.isfinite(item.get("initial_capacity_ah", np.nan))]
    if reference_values:
        summary["reference_capacity_distribution"] = {
            "min": float(np.nanmin(reference_values)),
            "q1": float(np.nanpercentile(reference_values, 25)),
            "median": float(np.nanmedian(reference_values)),
            "q3": float(np.nanpercentile(reference_values, 75)),
            "max": float(np.nanmax(reference_values)),
        }
    else:
        summary["reference_capacity_distribution"] = {}
    return summary


def save_clean_report(summary: dict[str, Any], outputs: dict[str, str], report_dir: Path) -> dict[str, Any]:
    report = {
        "cleaning_version": CLEANING_VERSION,
        "memory_mode": "two_pass_streaming_reference_then_arrow_batches_to_parquet",
        "summary": summary,
        "outputs": outputs,
    }
    report_path = report_dir / "cleaning_report.json"
    report["outputs"]["cleaning_report_json"] = str(report_path)
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean extracted XJTU charge and discharge cycles.")
    parser.add_argument("--extracted-dir", type=str, default=None, help="Input directory with extracted parquet files.")
    parser.add_argument("--clean-dir", type=str, default=None, help="Output directory for clean parquet files.")
    parser.add_argument("--report-dir", type=str, default=None, help="Report directory. Default: data/reports/xjtu")
    parser.add_argument("--min-points", type=int, default=DEFAULT_MIN_POINTS, help="Minimum samples required per cycle. Default: 20.")
    parser.add_argument("--min-voltage-range", type=float, default=DEFAULT_MIN_VOLTAGE_RANGE, help="Minimum voltage range in V. Default: 0.05.")
    parser.add_argument("--rebound-tolerance-v", type=float, default=DEFAULT_REBOUND_TOLERANCE_V, help="Voltage rebound/drop trimming tolerance in V. Default: 0.03.")
    parser.add_argument("--eol-fraction", type=float, default=DEFAULT_EOL_FRACTION, help="Fraction of initial capacity used to fill eol_capacity_ah. Default: 0.80.")
    parser.add_argument("--capacity-rescale-rel-tol", type=float, default=DEFAULT_CAPACITY_RESCALE_REL_TOL, help="Relative tolerance for preserving extracted summary capacity. Default: 0.05.")
    parser.add_argument("--parquet-chunk-size", type=int, default=5, help="Rows kept in RAM before parquet flush. Default: 5. Use 1 for the lowest memory use.")
    parser.add_argument("--read-batch-size", type=int, default=5, help="Rows read from extracted parquet per Arrow batch. Default: 5. Use 1 for the lowest memory use.")
    parser.add_argument("--reference-min-capacity-ah", type=float, default=DEFAULT_REFERENCE_MIN_CAPACITY_AH, help="Minimum phase capacity accepted for initial-capacity reference. Default: 1.0 Ah.")
    parser.add_argument("--reference-max-capacity-ah", type=float, default=DEFAULT_REFERENCE_MAX_CAPACITY_AH, help="Maximum phase capacity accepted for initial-capacity reference. Default: 2.3 Ah.")
    parser.add_argument("--reference-window-cycles", type=int, default=DEFAULT_REFERENCE_WINDOW_CYCLES, help="Number of first representative cycles used to compute median initial capacity. Default: 5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extracted_dir = Path(args.extracted_dir).resolve() if args.extracted_dir else default_extracted_dir().resolve()
    clean_dir = Path(args.clean_dir).resolve() if args.clean_dir else default_clean_dir().resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else default_report_dir().resolve()

    clean_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("XJTU cycle cleaning")
    print("=" * 80)
    print(f"Extracted directory: {extracted_dir}")
    print(f"Clean directory:     {clean_dir}")
    print(f"Report directory:    {report_dir}")
    print("Memory mode:         two-pass streaming Arrow batches")
    print(f"Read batch size:     {args.read_batch_size}")
    print(f"Parquet chunk size:  {args.parquet_chunk_size}")
    print(f"Reference capacity:  first {args.reference_window_cycles} valid representative cycles")
    print(f"Reference capacity range: [{args.reference_min_capacity_ah}, {args.reference_max_capacity_ah}] Ah")

    summary: dict[str, Any] = {}
    outputs: dict[str, str] = {}
    for phase in ["charge", "discharge"]:
        input_path = extracted_file(phase, extracted_dir)
        parquet_path = clean_file(phase, clean_dir)
        reject_path = rejected_file(phase, clean_dir)
        phase_summary = clean_phase_file_streaming(
            input_path=input_path,
            output_path=parquet_path,
            reject_path=reject_path,
            phase=phase,
            min_points=args.min_points,
            min_voltage_range=args.min_voltage_range,
            rebound_tolerance_v=args.rebound_tolerance_v,
            eol_fraction=args.eol_fraction,
            capacity_rescale_rel_tol=args.capacity_rescale_rel_tol,
            parquet_chunk_size=args.parquet_chunk_size,
            read_batch_size=args.read_batch_size,
            reference_min_capacity_ah=args.reference_min_capacity_ah,
            reference_max_capacity_ah=args.reference_max_capacity_ah,
            reference_window_cycles=args.reference_window_cycles,
        )
        summary[phase] = phase_summary
        outputs[f"{phase}_clean_cycles"] = str(parquet_path)
        outputs[f"{phase}_rejected_cycles"] = str(reject_path)

    report = save_clean_report(summary, outputs, report_dir)

    print("\nCleaning completed.")
    for phase in ["charge", "discharge"]:
        data = report["summary"][phase]
        print(
            f"{phase}: rows={data['n_rows']}, valid={data['n_valid_cycles']}, "
            f"model_usable={data['n_model_usable_cycles']}, rejected={data['n_rejected_cycles']}"
        )
        print(f"  output: {report['outputs'][phase + '_clean_cycles']}")


if __name__ == "__main__":
    main()
