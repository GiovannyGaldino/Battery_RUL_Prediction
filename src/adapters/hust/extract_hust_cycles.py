#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract HUST charge and discharge cycle segments from raw pickle files.

Output files:
    data/interim/hust/extracted/hust_charge_extracted_cycles.parquet
    data/interim/hust/extracted/hust_discharge_extracted_cycles.parquet
    data/reports/hust/extraction_report.json

This script does not perform final physical cleaning. It only converts defensible
charge/discharge segments from the original HUST pickle structure into extracted
phase-specific rows. Final schema compliance is handled by clean_hust_cycles.py.

Example:
    python src/adapters/hust/extract_hust_cycles.py \
        --base-dir "D:/USUÁRIOS/Marcellus/OneDrive/Doutorado/BATERIAS - Ressubmissão/Batteries"
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .hust_schema import (
        ADAPTER_VERSION,
        DATASET_NAME,
        DEFAULT_BASE_DIR,
        DEFAULT_TEMPERATURE_C,
        EXTRACTED_COLUMNS,
        RAW_REQUIRED_COLUMNS,
        TEMPERATURE_CANDIDATES,
        dump_json,
        empty_extracted_dataframe,
        ensure_columns,
        estimate_cycle_life_from_rul,
        get_dict_value_flexible,
        get_tqdm,
        listify,
        relative_path,
        resolve_base_paths,
        safe_float,
        safe_int,
        safe_numeric_array,
        utc_now_iso,
        write_parquet,
        describe_raw_dir,
    )
except ImportError:  # Allows direct execution from the script folder.
    from hust_schema import (  # type: ignore
        ADAPTER_VERSION,
        DATASET_NAME,
        DEFAULT_BASE_DIR,
        DEFAULT_TEMPERATURE_C,
        EXTRACTED_COLUMNS,
        RAW_REQUIRED_COLUMNS,
        TEMPERATURE_CANDIDATES,
        dump_json,
        empty_extracted_dataframe,
        ensure_columns,
        estimate_cycle_life_from_rul,
        get_dict_value_flexible,
        get_tqdm,
        listify,
        relative_path,
        resolve_base_paths,
        safe_float,
        safe_int,
        safe_numeric_array,
        utc_now_iso,
        write_parquet,
        describe_raw_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HUST charge/discharge cycles from raw pickle files.")
    parser.add_argument("--base-dir", type=str, default=str(DEFAULT_BASE_DIR), help="Project root directory.")
    parser.add_argument("--raw-dir", type=str, default=None, help="Directory containing HUST .pkl files.")
    parser.add_argument("--interim-dir", type=str, default=None, help="Dataset interim directory. Default: <base>/data/interim/hust")
    parser.add_argument("--reports-dir", type=str, default=None, help="Dataset reports directory. Default: <base>/data/reports/hust")
    parser.add_argument("--temperature-c", type=float, default=DEFAULT_TEMPERATURE_C, help="Fallback ambient temperature when no measured temperature exists.")
    parser.add_argument("--no-per-file-output", action="store_true", help="Do not save per-raw-file extracted parquet files.")
    return parser.parse_args()


def find_pickle_files(raw_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.pkl", "*.pickle", "*.pk"):
        files.extend(raw_dir.rglob(pattern))
    return sorted(set(files))


def load_pickle_file(file_path: Path) -> dict[Any, Any]:
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict in {file_path}, got {type(obj)!r}.")
    return obj


def normalize_hust_pickle_structure(raw_obj: dict[Any, Any]) -> dict[Any, Any]:
    """
    Accept both structures:
        {battery_id: {"data": {cycle: DataFrame}, "rul": {...}, "dq": {...}}, ...}
    and:
        {"data": {cycle: DataFrame}, "rul": {...}, "dq": {...}}
    """
    if {"data", "rul"}.issubset(set(raw_obj.keys())):
        return {"battery_000": raw_obj}
    return raw_obj


def infer_temperature_column(df: pd.DataFrame) -> str | None:
    for col in TEMPERATURE_CANDIDATES:
        if col in df.columns:
            return col
    return None


def select_phase_dataframe(df_cycle: pd.DataFrame, phase: str) -> pd.DataFrame:
    if "Status" not in df_cycle.columns:
        return pd.DataFrame()
    status = df_cycle["Status"].astype(str).str.lower()
    if phase == "charge":
        # Avoid capturing rows labeled as discharge because the word "charge" is
        # contained inside "discharge".
        mask = status.str.contains("charge", na=False) & ~status.str.contains("discharge", na=False)
    elif phase == "discharge":
        mask = status.str.contains("discharge", na=False)
    else:
        raise ValueError(f"Unsupported phase: {phase}")
    return df_cycle.loc[mask].copy()


def extract_phase_vectors(
    df_phase: pd.DataFrame,
    phase: str,
    fallback_temperature_c: float,
) -> tuple[dict[str, Any] | None, str]:
    if df_phase.empty:
        return None, "empty_phase"

    missing = RAW_REQUIRED_COLUMNS - set(df_phase.columns)
    if missing:
        return None, f"missing_raw_columns:{sorted(missing)}"

    time_s = safe_numeric_array(df_phase["Time (s)"])
    voltage_v = safe_numeric_array(df_phase["Voltage (V)"])
    current_a = safe_numeric_array(df_phase["Current (mA)"]) / 1000.0
    capacity_ah = safe_numeric_array(df_phase["Capacity (mAh)"]) / 1000.0

    temp_col = infer_temperature_column(df_phase)
    if temp_col is not None:
        temperature_c = safe_numeric_array(df_phase[temp_col])
        temperature_source = "measured"
    else:
        temperature_c = np.full(time_s.shape, float(fallback_temperature_c), dtype=float)
        temperature_source = "nominal_constant"

    n = min(len(time_s), len(voltage_v), len(current_a), len(capacity_ah), len(temperature_c))
    if n <= 0:
        return None, "empty_vectors"

    status_values = sorted(set(df_phase["Status"].astype(str).dropna().unique().tolist()))
    return {
        "time_s": time_s[:n],
        "voltage_v": voltage_v[:n],
        "current_a": current_a[:n],
        "temperature_c": temperature_c[:n],
        "capacity_ah": capacity_ah[:n],
        "temperature_source": temperature_source,
        "capacity_source": "measured",
        "status_values": status_values,
        "n_points_raw": int(n),
    }, "accepted"


def build_extracted_row(
    *,
    raw_dir: Path,
    file_path: Path,
    battery_id_raw: Any,
    battery_index: int,
    cycle_key: Any,
    phase: str,
    phase_vectors: dict[str, Any],
    rul_dict: Any,
    dq_dict: Any,
) -> dict[str, Any]:
    cycle_number = safe_float(cycle_key)
    if not np.isfinite(cycle_number):
        cycle_number = float(safe_int(cycle_key, default=-1))

    remaining_raw = get_dict_value_flexible(rul_dict, cycle_key)
    remaining_cycles_raw = safe_float(remaining_raw)

    cycle_life_raw = np.nan
    if np.isfinite(cycle_number) and np.isfinite(remaining_cycles_raw):
        cycle_life_raw = cycle_number + remaining_cycles_raw
    else:
        cycle_life_raw = estimate_cycle_life_from_rul(rul_dict)

    summary_capacity_ah = np.nan
    dq_value = get_dict_value_flexible(dq_dict, cycle_key)
    if dq_value is not None:
        summary_capacity_ah = safe_float(dq_value)
    if not np.isfinite(summary_capacity_ah):
        cap = safe_numeric_array(phase_vectors["capacity_ah"])
        if cap.size:
            summary_capacity_ah = float(np.nanmax(cap) - np.nanmin(cap))

    battery_id = str(battery_id_raw)
    if not battery_id.upper().startswith("HUST"):
        battery_id = f"HUST_{battery_id}"

    source_cycle_id = f"{Path(file_path).stem}:{battery_id_raw}:cycle_{cycle_key}"
    cycle_pair_id = f"{battery_id}_cycle_{cycle_number:g}" if np.isfinite(cycle_number) else f"{battery_id}_cycle_{cycle_key}"

    return {
        "dataset": DATASET_NAME,
        "source_file": file_path.name,
        "source_path": relative_path(file_path, raw_dir),
        "battery_id": battery_id,
        "battery_index": int(battery_index),
        "cell_id_original": str(battery_id_raw),
        "cycle_number": float(cycle_number) if np.isfinite(cycle_number) else np.nan,
        "source_cycle_id": source_cycle_id,
        "cycle_pair_id": cycle_pair_id,
        "phase": phase,
        "remaining_cycles_raw": float(remaining_cycles_raw) if np.isfinite(remaining_cycles_raw) else np.nan,
        "cycle_life_raw": float(cycle_life_raw) if np.isfinite(cycle_life_raw) else np.nan,
        "summary_capacity_ah": float(summary_capacity_ah) if np.isfinite(summary_capacity_ah) else np.nan,
        "time_s": listify(phase_vectors["time_s"]),
        "voltage_v": listify(phase_vectors["voltage_v"]),
        "current_a": listify(phase_vectors["current_a"]),
        "temperature_c": listify(phase_vectors["temperature_c"]),
        "capacity_ah": listify(phase_vectors["capacity_ah"]),
        "temperature_source": phase_vectors["temperature_source"],
        "capacity_source": phase_vectors["capacity_source"],
        "status_values": phase_vectors["status_values"],
        "n_points_raw": int(phase_vectors["n_points_raw"]),
        "extract_warnings": [],
        "adapter_version": ADAPTER_VERSION,
        "created_at": utc_now_iso(),
    }


def extract_file(
    file_path: Path,
    raw_dir: Path,
    fallback_temperature_c: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obj = normalize_hust_pickle_structure(load_pickle_file(file_path))
    rows: list[dict[str, Any]] = []
    reasons = Counter()
    phase_counts = Counter()
    batteries_seen = 0
    cycles_seen = 0

    tqdm = get_tqdm()
    iterator = tqdm(list(obj.items()), desc=f"batteries:{file_path.name}", unit="battery", leave=False)
    for battery_index, (battery_id_raw, battery_data) in enumerate(iterator):
        if not isinstance(battery_data, dict):
            reasons["invalid_battery_structure"] += 1
            continue
        batteries_seen += 1
        cycles = battery_data.get("data", {})
        rul_dict = battery_data.get("rul", {})
        dq_dict = battery_data.get("dq", {})
        if not isinstance(cycles, dict):
            reasons["invalid_cycle_dictionary"] += 1
            continue

        cycle_items = list(cycles.items())
        cycle_iterator = tqdm(cycle_items, desc=f"cycles:{battery_id_raw}", unit="cycle", leave=False)
        for cycle_key, df_cycle in cycle_iterator:
            cycles_seen += 1
            if not isinstance(df_cycle, pd.DataFrame) or df_cycle.empty:
                reasons["invalid_or_empty_cycle_dataframe"] += 1
                continue
            missing = RAW_REQUIRED_COLUMNS - set(df_cycle.columns)
            if missing:
                reasons[f"missing_raw_columns:{sorted(missing)}"] += 1
                continue

            for phase in ("charge", "discharge"):
                df_phase = select_phase_dataframe(df_cycle, phase)
                phase_vectors, status = extract_phase_vectors(df_phase, phase, fallback_temperature_c)
                if phase_vectors is None:
                    reasons[f"{phase}:{status}"] += 1
                    continue
                row = build_extracted_row(
                    raw_dir=raw_dir,
                    file_path=file_path,
                    battery_id_raw=battery_id_raw,
                    battery_index=battery_index,
                    cycle_key=cycle_key,
                    phase=phase,
                    phase_vectors=phase_vectors,
                    rul_dict=rul_dict,
                    dq_dict=dq_dict,
                )
                rows.append(row)
                phase_counts[phase] += 1

    report = {
        "source_file": file_path.name,
        "source_path": relative_path(file_path, raw_dir),
        "batteries_seen": batteries_seen,
        "cycles_seen": cycles_seen,
        "rows_extracted": len(rows),
        "phase_counts": dict(phase_counts),
        "rejection_reasons": dict(reasons),
    }
    return rows, report


def main() -> None:
    args = parse_args()
    paths = resolve_base_paths(args)
    raw_dir = paths["raw_dir"]
    extracted_dir = paths["extracted_dir"]
    reports_dir = paths["reports_dir"]
    extracted_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = find_pickle_files(raw_dir)
    if not pkl_files:
        raise FileNotFoundError(describe_raw_dir(raw_dir))

    all_rows: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []

    tqdm = get_tqdm()
    for file_path in tqdm(pkl_files, desc="HUST raw files", unit="file"):
        try:
            rows, file_report = extract_file(file_path, raw_dir, args.temperature_c)
        except Exception as exc:
            rows = []
            file_report = {
                "source_file": file_path.name,
                "source_path": relative_path(file_path, raw_dir),
                "fatal_error": f"{type(exc).__name__}: {exc}",
                "rows_extracted": 0,
            }
        all_rows.extend(rows)
        file_reports.append(file_report)

        if rows and not args.no_per_file_output:
            df_file = ensure_columns(pd.DataFrame(rows), EXTRACTED_COLUMNS)
            for phase in ("charge", "discharge"):
                df_phase = df_file[df_file["phase"] == phase].reset_index(drop=True)
                output_file = extracted_dir / f"{file_path.stem}_{phase}_extracted_cycles.parquet"
                write_parquet(ensure_columns(df_phase, EXTRACTED_COLUMNS), output_file)

    df_all = ensure_columns(pd.DataFrame(all_rows), EXTRACTED_COLUMNS) if all_rows else empty_extracted_dataframe()

    output_files = {}
    for phase in ("charge", "discharge"):
        df_phase = df_all[df_all["phase"] == phase].reset_index(drop=True)
        output_file = extracted_dir / f"hust_{phase}_extracted_cycles.parquet"
        write_parquet(ensure_columns(df_phase, EXTRACTED_COLUMNS), output_file)
        output_files[phase] = str(output_file)

    total_phase_counts = df_all["phase"].value_counts(dropna=False).to_dict() if not df_all.empty else {}
    rejection_counter = Counter()
    for item in file_reports:
        rejection_counter.update(item.get("rejection_reasons", {}))

    report = {
        "dataset": DATASET_NAME,
        "adapter_version": ADAPTER_VERSION,
        "raw_dir": str(raw_dir),
        "extracted_dir": str(extracted_dir),
        "n_raw_files": len(pkl_files),
        "n_extracted_rows": int(len(df_all)),
        "phase_counts": {str(k): int(v) for k, v in total_phase_counts.items()},
        "rejection_reasons": dict(rejection_counter),
        "output_files": output_files,
        "files": file_reports,
        "created_at": utc_now_iso(),
    }
    dump_json(report, reports_dir / "extraction_report.json")

    print("\n[OK] HUST extraction completed.")
    print(f"[OK] Charge rows: {int((df_all['phase'] == 'charge').sum()) if not df_all.empty else 0}")
    print(f"[OK] Discharge rows: {int((df_all['phase'] == 'discharge').sum()) if not df_all.empty else 0}")
    print(f"[OK] Report: {reports_dir / 'extraction_report.json'}")


if __name__ == "__main__":
    main()
