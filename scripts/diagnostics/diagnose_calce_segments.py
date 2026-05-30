#!/usr/bin/env python
"""
Temporary diagnostic script for CALCE cycle segmentation and scaling issues.

Purpose
-------
This script does NOT modify any project data. It inspects the raw CALCE Excel files
and, when available, the extracted/clean Parquet files produced by the adapter.

It was created to investigate visual issues such as:
  - charge curves that appear truncated before the end of charge;
  - voltage decreasing before the charge phase is apparently complete;
  - discharge curves apparently cut at specific voltages;
  - capacity scale inconsistencies between cycles or cells;
  - wrong dominant-segment selection from raw CALCE workbooks.

Expected project layout
-----------------------
Run from the project root, or pass --project-root:

    python scripts/diagnostics/diagnose_calce_segments.py --project-root .

Required raw data layout:

    data/raw/CALCE/<cell_id>/*.xlsx

Outputs
-------
    data/reports/calce/diagnostics/calce_raw_sheet_summary.csv
    data/reports/calce/diagnostics/calce_segment_candidates.csv
    data/reports/calce/diagnostics/calce_selected_segment_summary.csv
    data/reports/calce/diagnostics/calce_clean_cycle_summary.csv
    data/reports/calce/diagnostics/calce_diagnostic_findings.json
    data/reports/calce/diagnostics/plots/...

Interpretation
--------------
Use the CSV files first. The plots are intended for visual confirmation.
Important diagnostic columns include:

    measured_capacity_span_ah
    integrated_current_capacity_ah
    measured_to_integrated_ratio
    capacity_ratio_to_nominal
    voltage_start_v, voltage_end_v, voltage_min_v, voltage_max_v
    charge_end_voltage_low_flag
    discharge_end_voltage_high_flag
    capacity_scale_suspect_flag
    selected_by_adapter_rule

Notes
-----
This script imports utility functions from:

    src/adapters/calce/extract_calce_cycles.py

Therefore, it should be used with the current CALCE adapter present in the project.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TARGET_CELLS = [
    # Main problematic cells reported during visual inspection.
    "CS2_5", "CS2_6", "CS2_7", "CS2_24", "CS2_25",
    # Cells reported as apparently more consistent, but with charge-end concern.
    "CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38",
    # CX2 cells: generally usable, but with possible incomplete charge behavior.
    "CX2_3", "CX2_4", "CX2_8", "CX2_16", "CX2_31", "CX2_33", "CX2_34", "CX2_35", "CX2_36",
]


@dataclass
class SegmentRecord:
    cell_id: str
    file_order: int
    source_file: str
    source_path: str
    sheet_name: str
    phase: str
    raw_segment_index: int
    selected_by_adapter_rule: bool
    n_points: int
    row_start: int
    row_end: int
    duration_s: float
    sample_period_median_s: float
    voltage_start_v: float
    voltage_end_v: float
    voltage_min_v: float
    voltage_max_v: float
    voltage_span_v: float
    current_min_a: float
    current_max_a: float
    current_mean_a: float
    measured_capacity_start_ah: float
    measured_capacity_end_ah: float
    measured_capacity_span_ah: float
    integrated_current_capacity_ah: float
    measured_to_integrated_ratio: float
    nominal_capacity_ah: float
    capacity_ratio_to_nominal: float
    charge_end_voltage_low_flag: bool
    discharge_end_voltage_high_flag: bool
    voltage_span_small_flag: bool
    capacity_scale_suspect_flag: bool
    capacity_too_small_flag: bool
    capacity_too_large_flag: bool
    negative_voltage_flag: bool
    high_voltage_flag: bool
    notes: str


def finite(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def safe_min(values: Any) -> float:
    arr = finite(values)
    return float(np.min(arr)) if arr.size else math.nan


def safe_max(values: Any) -> float:
    arr = finite(values)
    return float(np.max(arr)) if arr.size else math.nan


def safe_first(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr[0]) if arr.size else math.nan


def safe_last(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr[-1]) if arr.size else math.nan


def safe_mean(values: Any) -> float:
    arr = finite(values)
    return float(np.mean(arr)) if arr.size else math.nan


def median_period(time_s: np.ndarray) -> float:
    t = finite(time_s)
    if t.size < 2:
        return math.nan
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size else math.nan


def integrate_abs_current_capacity(time_s: np.ndarray, current_a: np.ndarray) -> float:
    t = np.asarray(time_s, dtype=float)
    i = np.asarray(current_a, dtype=float)
    mask = np.isfinite(t) & np.isfinite(i)
    if mask.sum() < 2:
        return math.nan
    t = t[mask]
    i = np.abs(i[mask])
    order = np.argsort(t)
    t = t[order]
    i = i[order]
    # Remove duplicate timestamps to avoid zero/negative integration artifacts.
    keep = np.r_[True, np.diff(t) > 0]
    t = t[keep]
    i = i[keep]
    if t.size < 2:
        return math.nan
    return float(np.trapz(i, t) / 3600.0)


def localize_time(time_s: np.ndarray) -> np.ndarray:
    t = np.asarray(time_s, dtype=float)
    valid = finite(t)
    if valid.size == 0:
        return np.arange(len(t), dtype=float)
    return t - valid[0]


def capacity_span_for_indices(df: pd.DataFrame, col: str | None, indices: np.ndarray, scale_capacity_to_ah, safe_numeric) -> float:
    if col is None or col not in df.columns or indices.size == 0:
        return math.nan
    q = scale_capacity_to_ah(safe_numeric(df.iloc[indices][col]), col)
    qf = finite(q)
    if qf.size == 0:
        return math.nan
    return float(np.max(qf) - np.min(qf))


def prepare_imports(project_root: Path):
    adapter_dir = project_root / "src" / "adapters" / "calce"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"CALCE adapter directory not found: {adapter_dir}")
    sys.path.insert(0, str(adapter_dir))
    import extract_calce_cycles as calce  # type: ignore
    return calce


def choose_capacity_column(calce, colmap, phase: str) -> tuple[str | None, str]:
    # Use the adapter function when available.
    if hasattr(calce, "choose_capacity_column"):
        return calce.choose_capacity_column(colmap, phase)
    if phase == "charge" and getattr(colmap, "charge_capacity_ah", None):
        return colmap.charge_capacity_ah, "measured"
    if phase == "discharge" and getattr(colmap, "discharge_capacity_ah", None):
        return colmap.discharge_capacity_ah, "measured"
    if getattr(colmap, "capacity_ah", None):
        return colmap.capacity_ah, "measured"
    return None, "missing"


def make_segment_record(calce, *, raw_root: Path, path: Path, cell_id: str, file_order: int,
                        sheet_name: str, df: pd.DataFrame, colmap, phase: str,
                        segment_index: int, indices: np.ndarray, selected: bool,
                        charge_end_voltage_min: float, discharge_end_voltage_max: float,
                        min_voltage_span: float, low_capacity_ratio: float,
                        high_capacity_ratio: float) -> SegmentRecord:
    if getattr(colmap, "time_s", None) is not None:
        time_s = localize_time(calce.safe_numeric(df.iloc[indices][colmap.time_s]))
    else:
        time_s = np.arange(len(indices), dtype=float)

    voltage_v = calce.scale_voltage_to_v(calce.safe_numeric(df.iloc[indices][colmap.voltage_v]), colmap.voltage_v)
    current_a = calce.scale_current_to_a(calce.safe_numeric(df.iloc[indices][colmap.current_a]), colmap.current_a)
    capacity_col, _ = choose_capacity_column(calce, colmap, phase)
    if capacity_col is not None:
        q_measured = calce.scale_capacity_to_ah(calce.safe_numeric(df.iloc[indices][capacity_col]), capacity_col)
        q_f = finite(q_measured)
        q_start = float(q_f[0]) if q_f.size else math.nan
        q_end = float(q_f[-1]) if q_f.size else math.nan
        q_span = float(np.max(q_f) - np.min(q_f)) if q_f.size else math.nan
    else:
        q_start = q_end = q_span = math.nan

    q_int = integrate_abs_current_capacity(time_s, current_a)
    nominal = calce.infer_nominal_capacity_ah(cell_id)
    ratio_nominal = q_span / nominal if np.isfinite(q_span) and np.isfinite(nominal) and nominal > 0 else math.nan
    measured_to_int = q_span / q_int if np.isfinite(q_span) and np.isfinite(q_int) and q_int > 1e-9 else math.nan

    v_start = safe_first(voltage_v)
    v_end = safe_last(voltage_v)
    v_min = safe_min(voltage_v)
    v_max = safe_max(voltage_v)
    v_span = v_max - v_min if np.isfinite(v_max) and np.isfinite(v_min) else math.nan

    notes: list[str] = []
    if phase == "charge" and np.isfinite(v_end) and v_end < charge_end_voltage_min:
        notes.append("charge_end_voltage_below_expected_full_charge")
    if phase == "discharge" and np.isfinite(v_end) and v_end > discharge_end_voltage_max:
        notes.append("discharge_end_voltage_above_expected_cutoff")
    if np.isfinite(measured_to_int) and (measured_to_int < 0.50 or measured_to_int > 2.00):
        notes.append("measured_capacity_disagrees_with_current_integration")
    if np.isfinite(ratio_nominal) and ratio_nominal < low_capacity_ratio:
        notes.append("capacity_span_too_small_vs_nominal")
    if np.isfinite(ratio_nominal) and ratio_nominal > high_capacity_ratio:
        notes.append("capacity_span_too_large_vs_nominal")
    if np.isfinite(v_span) and v_span < min_voltage_span:
        notes.append("voltage_span_small")
    if np.isfinite(v_min) and v_min < 0:
        notes.append("negative_voltage")
    if np.isfinite(v_max) and v_max > 4.45:
        notes.append("high_voltage")

    return SegmentRecord(
        cell_id=cell_id,
        file_order=file_order,
        source_file=path.name,
        source_path=str(path.relative_to(raw_root)),
        sheet_name=sheet_name,
        phase=phase,
        raw_segment_index=int(segment_index),
        selected_by_adapter_rule=bool(selected),
        n_points=int(indices.size),
        row_start=int(indices[0]) if indices.size else -1,
        row_end=int(indices[-1]) if indices.size else -1,
        duration_s=float(safe_last(time_s) - safe_first(time_s)) if indices.size else math.nan,
        sample_period_median_s=median_period(time_s),
        voltage_start_v=v_start,
        voltage_end_v=v_end,
        voltage_min_v=v_min,
        voltage_max_v=v_max,
        voltage_span_v=v_span,
        current_min_a=safe_min(current_a),
        current_max_a=safe_max(current_a),
        current_mean_a=safe_mean(current_a),
        measured_capacity_start_ah=q_start,
        measured_capacity_end_ah=q_end,
        measured_capacity_span_ah=q_span,
        integrated_current_capacity_ah=q_int,
        measured_to_integrated_ratio=measured_to_int,
        nominal_capacity_ah=nominal,
        capacity_ratio_to_nominal=ratio_nominal,
        charge_end_voltage_low_flag=(phase == "charge" and np.isfinite(v_end) and v_end < charge_end_voltage_min),
        discharge_end_voltage_high_flag=(phase == "discharge" and np.isfinite(v_end) and v_end > discharge_end_voltage_max),
        voltage_span_small_flag=(np.isfinite(v_span) and v_span < min_voltage_span),
        capacity_scale_suspect_flag=(np.isfinite(measured_to_int) and (measured_to_int < 0.50 or measured_to_int > 2.00)),
        capacity_too_small_flag=(np.isfinite(ratio_nominal) and ratio_nominal < low_capacity_ratio),
        capacity_too_large_flag=(np.isfinite(ratio_nominal) and ratio_nominal > high_capacity_ratio),
        negative_voltage_flag=(np.isfinite(v_min) and v_min < 0),
        high_voltage_flag=(np.isfinite(v_max) and v_max > 4.45),
        notes="|".join(notes),
    )


def plot_segment(calce, *, outdir: Path, path: Path, sheet_name: str, df: pd.DataFrame, colmap,
                 indices: np.ndarray, cell_id: str, file_order: int, phase: str, tag: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if getattr(colmap, "time_s", None) is not None:
        t_all = localize_time(calce.safe_numeric(df[colmap.time_s]))
        t_seg = localize_time(calce.safe_numeric(df.iloc[indices][colmap.time_s]))
    else:
        t_all = np.arange(len(df), dtype=float)
        t_seg = np.arange(len(indices), dtype=float)

    voltage_all = calce.scale_voltage_to_v(calce.safe_numeric(df[colmap.voltage_v]), colmap.voltage_v)
    current_all = calce.scale_current_to_a(calce.safe_numeric(df[colmap.current_a]), colmap.current_a)
    voltage_seg = calce.scale_voltage_to_v(calce.safe_numeric(df.iloc[indices][colmap.voltage_v]), colmap.voltage_v)
    current_seg = calce.scale_current_to_a(calce.safe_numeric(df.iloc[indices][colmap.current_a]), colmap.current_a)
    capacity_col, _ = choose_capacity_column(calce, colmap, phase)
    if capacity_col is not None:
        q_seg = calce.scale_capacity_to_ah(calce.safe_numeric(df.iloc[indices][capacity_col]), capacity_col)
        q_seg = q_seg - np.nanmin(q_seg) if np.isfinite(q_seg).any() else q_seg
    else:
        q_seg = np.full(len(indices), np.nan)

    safe_name = f"{file_order:04d}_{cell_id}_{phase}_{tag}_{Path(path).stem}_{sheet_name}".replace("/", "_").replace("\\", "_").replace(" ", "_")

    # Plot 1: selected segment V x Q
    fig = plt.figure(figsize=(8, 5))
    plt.plot(q_seg, voltage_seg, linewidth=1.0)
    plt.xlabel("Measured phase capacity span, Q (Ah)")
    plt.ylabel("Voltage (V)")
    plt.title(f"{cell_id} | {phase} | file {file_order} | {path.name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / f"{safe_name}_v_vs_q.png", dpi=160)
    plt.close(fig)

    # Plot 2: raw full sheet time traces with selected segment window.
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(t_all, voltage_all, linewidth=0.8, label="Voltage (V)")
    ax1.set_xlabel("Raw sheet localized time (s)")
    ax1.set_ylabel("Voltage (V)")
    ax1.grid(True, alpha=0.3)
    if indices.size:
        t0 = t_all[indices[0]] if indices[0] < len(t_all) else math.nan
        t1 = t_all[indices[-1]] if indices[-1] < len(t_all) else math.nan
        if np.isfinite(t0) and np.isfinite(t1):
            ax1.axvspan(t0, t1, alpha=0.2, label="Selected segment")
    ax2 = ax1.twinx()
    ax2.plot(t_all, current_all, linewidth=0.6, linestyle="--", label="Current (A)")
    ax2.set_ylabel("Current (A)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title(f"Raw sheet traces | {cell_id} | {phase} | {path.name} | {sheet_name}")
    plt.tight_layout()
    fig.savefig(outdir / f"{safe_name}_raw_time_voltage_current.png", dpi=160)
    plt.close(fig)

    # Plot 3: segment time traces with measured Q if available.
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(t_seg, voltage_seg, linewidth=0.9, label="Voltage (V)")
    ax1.set_xlabel("Segment localized time (s)")
    ax1.set_ylabel("Voltage (V)")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(t_seg, current_seg, linewidth=0.7, linestyle="--", label="Current (A)")
    if np.isfinite(q_seg).any():
        ax2.plot(t_seg, q_seg, linewidth=0.9, linestyle=":", label="Measured Q span (Ah)")
    ax2.set_ylabel("Current (A) / Q (Ah)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title(f"Selected segment traces | {cell_id} | {phase} | {path.name}")
    plt.tight_layout()
    fig.savefig(outdir / f"{safe_name}_segment_time_traces.png", dpi=160)
    plt.close(fig)


def summarize_clean_outputs(project_root: Path, outdir: Path) -> pd.DataFrame:
    records = []
    clean_dir = project_root / "data" / "interim" / "calce" / "clean"
    for phase in ["charge", "discharge"]:
        parquet_path = clean_dir / f"calce_{phase}_clean_cycles.parquet"
        if not parquet_path.exists():
            continue
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            records.append({"phase": phase, "error": str(exc)})
            continue
        if df.empty:
            records.append({"phase": phase, "n_rows": 0})
            continue
        group = df.groupby("battery_id", dropna=False).agg(
            n_rows=("battery_id", "size"),
            n_valid=("is_valid_cycle", "sum"),
            n_model_usable=("is_model_usable", "sum"),
            cycle_min=("cycle_number", "min"),
            cycle_max=("cycle_number", "max"),
            remaining_min=("remaining_cycles", "min"),
            remaining_max=("remaining_cycles", "max"),
            phase_capacity_min=("phase_capacity_ah", "min"),
            phase_capacity_median=("phase_capacity_ah", "median"),
            phase_capacity_max=("phase_capacity_ah", "max"),
            voltage_start_median=("voltage_start_v", "median"),
            voltage_end_median=("voltage_end_v", "median"),
        ).reset_index()
        group.insert(0, "phase", phase)
        records.extend(group.to_dict(orient="records"))
    out = pd.DataFrame(records)
    if not out.empty:
        out.to_csv(outdir / "calce_clean_cycle_summary.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CALCE segmentation and scaling issues without modifying project data.")
    parser.add_argument("--project-root", default=None, help="Project root. Defaults to current directory search.")
    parser.add_argument("--raw-dir", default=None, help="Raw CALCE directory. Defaults to data/raw/CALCE.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to data/reports/calce/diagnostics.")
    parser.add_argument("--cells", nargs="*", default=DEFAULT_TARGET_CELLS, help="CALCE cell ids to inspect, e.g. CS2_5 CS2_6 CX2_16. Use ALL for all cells.")
    parser.add_argument("--file-pattern", default="**/*.xls*", help="Glob pattern below raw CALCE directory.")
    parser.add_argument("--max-files-per-cell", type=int, default=0, help="0 means all files. Otherwise limit each cell.")
    parser.add_argument("--max-plots-per-cell-phase", type=int, default=8, help="Maximum selected diagnostic plots per cell and phase.")
    parser.add_argument("--phase-current-threshold-c-rate", type=float, default=0.05)
    parser.add_argument("--charge-end-voltage-min", type=float, default=4.05, help="Flag charge segments ending below this voltage.")
    parser.add_argument("--discharge-end-voltage-max", type=float, default=3.05, help="Flag discharge segments ending above this voltage.")
    parser.add_argument("--min-voltage-span", type=float, default=0.25)
    parser.add_argument("--low-capacity-ratio", type=float, default=0.30, help="Flag selected segment capacity span below this fraction of nominal capacity.")
    parser.add_argument("--high-capacity-ratio", type=float, default=1.30, help="Flag selected segment capacity span above this fraction of nominal capacity.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else Path.cwd().resolve()
    raw_dir = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else project_root / "data" / "raw" / "CALCE"
    outdir = Path(args.output_dir).expanduser().resolve() if args.output_dir else project_root / "data" / "reports" / "calce" / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    plot_root = outdir / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    calce = prepare_imports(project_root)

    files_by_cell, discovery_issues = calce.discover_excel_files(raw_dir, args.file_pattern)
    requested = args.cells
    if len(requested) == 1 and requested[0].upper() == "ALL":
        target_cells = sorted(files_by_cell.keys())
    else:
        target_cells = requested

    raw_sheet_records: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    findings: dict[str, Any] = {
        "script": "diagnose_calce_segments.py",
        "diagnostic_version": "calce_diagnostic_v6.0.0",
        "raw_dir": str(raw_dir),
        "output_dir": str(outdir),
        "target_cells": target_cells,
        "discovery_issues_count": len(discovery_issues),
        "cells": {},
    }

    plot_counts: dict[tuple[str, str], int] = {}

    for cell_id in target_cells:
        paths = files_by_cell.get(cell_id, [])
        if args.max_files_per_cell and args.max_files_per_cell > 0:
            paths = paths[:args.max_files_per_cell]
        threshold_a = calce.dynamic_current_threshold_a(cell_id, args.phase_current_threshold_c_rate)
        cell_findings = {
            "n_files": len(paths),
            "n_sheets": 0,
            "n_raw_segments": 0,
            "n_selected_segments": 0,
            "flags": {},
        }
        for file_order, path in enumerate(paths, start=1):
            sheets, read_issues = calce.read_excel_sheets(path)
            if read_issues:
                raw_sheet_records.extend(read_issues)
            for sheet_name, sheet_df, colmap, sheet_warnings in sheets:
                cell_findings["n_sheets"] += 1
                if getattr(colmap, "time_s", None) is not None:
                    time_values = calce.safe_numeric(sheet_df[colmap.time_s])
                    order = np.argsort(np.where(np.isfinite(time_values), time_values, np.inf))
                    sheet_df = sheet_df.iloc[order].reset_index(drop=True)

                raw_sheet_records.append({
                    "cell_id": cell_id,
                    "file_order": file_order,
                    "source_file": path.name,
                    "source_path": str(path.relative_to(raw_dir)),
                    "sheet_name": sheet_name,
                    "n_rows": len(sheet_df),
                    "time_col": getattr(colmap, "time_s", None),
                    "current_col": getattr(colmap, "current_a", None),
                    "voltage_col": getattr(colmap, "voltage_v", None),
                    "charge_capacity_col": getattr(colmap, "charge_capacity_ah", None),
                    "discharge_capacity_col": getattr(colmap, "discharge_capacity_ah", None),
                    "generic_capacity_col": getattr(colmap, "capacity_ah", None),
                    "warnings": "|".join(sheet_warnings),
                })

                if not getattr(colmap, "current_a", None) or not getattr(colmap, "voltage_v", None):
                    continue

                raw_segments = calce.split_phase_segments(sheet_df, colmap.current_a, threshold_a)
                selected_segments = calce.select_dominant_phase_segments(sheet_df, colmap, raw_segments, nominal_capacity)
                selected_key = {(p, int(i[0]), int(i[-1])) for p, i, _ in selected_segments if len(i)}

                cell_findings["n_raw_segments"] += len(raw_segments)
                cell_findings["n_selected_segments"] += len(selected_segments)

                for phase, indices, segment_index in raw_segments:
                    selected = (phase, int(indices[0]), int(indices[-1])) in selected_key if len(indices) else False
                    rec = make_segment_record(
                        calce,
                        raw_root=raw_dir,
                        path=path,
                        cell_id=cell_id,
                        file_order=file_order,
                        sheet_name=sheet_name,
                        df=sheet_df,
                        colmap=colmap,
                        phase=phase,
                        segment_index=segment_index,
                        indices=indices,
                        selected=selected,
                        charge_end_voltage_min=args.charge_end_voltage_min,
                        discharge_end_voltage_max=args.discharge_end_voltage_max,
                        min_voltage_span=args.min_voltage_span,
                        low_capacity_ratio=args.low_capacity_ratio,
                        high_capacity_ratio=args.high_capacity_ratio,
                    )
                    rec_dict = asdict(rec)
                    segment_records.append(rec_dict)
                    if selected:
                        selected_records.append(rec_dict)

                    # Plot selected segments, plus any clearly suspicious non-selected segment in limited quantity.
                    should_plot = selected and (
                        rec.notes or
                        rec.cell_id in {"CS2_5", "CS2_6", "CS2_7", "CS2_24", "CS2_25"} or
                        rec.charge_end_voltage_low_flag or
                        rec.discharge_end_voltage_high_flag or
                        rec.capacity_scale_suspect_flag
                    )
                    key = (cell_id, phase)
                    if should_plot and plot_counts.get(key, 0) < args.max_plots_per_cell_phase:
                        tag = "selected_suspicious" if rec.notes else "selected_reference"
                        plot_segment(
                            calce,
                            outdir=plot_root / cell_id / phase,
                            path=path,
                            sheet_name=sheet_name,
                            df=sheet_df,
                            colmap=colmap,
                            indices=indices,
                            cell_id=cell_id,
                            file_order=file_order,
                            phase=phase,
                            tag=tag,
                        )
                        plot_counts[key] = plot_counts.get(key, 0) + 1

        findings["cells"][cell_id] = cell_findings

    raw_sheet_df = pd.DataFrame(raw_sheet_records)
    segment_df = pd.DataFrame(segment_records)
    selected_df = pd.DataFrame(selected_records)

    raw_sheet_df.to_csv(outdir / "calce_raw_sheet_summary.csv", index=False)
    segment_df.to_csv(outdir / "calce_segment_candidates.csv", index=False)
    selected_df.to_csv(outdir / "calce_selected_segment_summary.csv", index=False)

    clean_summary_df = summarize_clean_outputs(project_root, outdir)

    if not selected_df.empty:
        flag_cols = [
            "charge_end_voltage_low_flag", "discharge_end_voltage_high_flag",
            "voltage_span_small_flag", "capacity_scale_suspect_flag",
            "capacity_too_small_flag", "capacity_too_large_flag",
            "negative_voltage_flag", "high_voltage_flag",
        ]
        flag_summary = {}
        for cell_id, grp in selected_df.groupby("cell_id"):
            flag_summary[cell_id] = {
                "n_selected": int(len(grp)),
                **{col: int(grp[col].sum()) for col in flag_cols if col in grp.columns},
                "median_capacity_ratio_to_nominal": float(grp["capacity_ratio_to_nominal"].median()) if "capacity_ratio_to_nominal" in grp else math.nan,
                "median_measured_to_integrated_ratio": float(grp["measured_to_integrated_ratio"].median()) if "measured_to_integrated_ratio" in grp else math.nan,
            }
        findings["selected_segment_flag_summary"] = flag_summary

        # Most suspicious selected segments for quick inspection.
        suspicious = selected_df[selected_df["notes"].astype(str).str.len() > 0].copy()
        if not suspicious.empty:
            suspicious.to_csv(outdir / "calce_selected_suspicious_segments.csv", index=False)
            findings["n_selected_suspicious_segments"] = int(len(suspicious))
        else:
            findings["n_selected_suspicious_segments"] = 0

    findings["output_files"] = {
        "raw_sheet_summary": str(outdir / "calce_raw_sheet_summary.csv"),
        "segment_candidates": str(outdir / "calce_segment_candidates.csv"),
        "selected_segment_summary": str(outdir / "calce_selected_segment_summary.csv"),
        "clean_cycle_summary": str(outdir / "calce_clean_cycle_summary.csv"),
        "selected_suspicious_segments": str(outdir / "calce_selected_suspicious_segments.csv"),
        "plots": str(plot_root),
    }

    with (outdir / "calce_diagnostic_findings.json").open("w", encoding="utf-8") as fp:
        json.dump(findings, fp, indent=2, ensure_ascii=False)

    print("[OK] CALCE diagnostic completed.")
    print(f"[OK] Output directory: {outdir}")
    print(f"[OK] Raw sheet rows: {len(raw_sheet_df)}")
    print(f"[OK] Raw segment candidates: {len(segment_df)}")
    print(f"[OK] Selected segments: {len(selected_df)}")
    if not clean_summary_df.empty:
        print(f"[OK] Clean summary rows: {len(clean_summary_df)}")
    print("[NEXT] Inspect calce_selected_suspicious_segments.csv and the plots directory.")


if __name__ == "__main__":
    main()
