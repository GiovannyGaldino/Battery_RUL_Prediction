# -*- coding: utf-8 -*-
"""
Audit cleaned Oxford/ORA charge and discharge cycle files.

Inputs:
    data/interim/ora/clean/ora_charge_clean_cycles.parquet
    data/interim/ora/clean/ora_discharge_clean_cycles.parquet

Outputs:
    data/reports/ora/audit_report.json
    data/reports/ora/audit_summary.csv
    data/reports/ora/anomaly_list.csv
    data/reports/ora/plots/<phase>/*.png

The audit validates the official clean-cycle schema, summarizes signal and target
availability, and generates visual inspection plots for voltage-capacity curves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

try:
    from .ora_schema import (
        DATASET_FOLDER,
        DATASET_NAME,
        STANDARD_CYCLE_COLUMNS,
        can_interpolate_qv_curve,
        ensure_dir,
        finite_array,
        infer_default_base_dir,
        safe_array,
        validate_standard_schema,
        write_json,
    )
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from ora_schema import (  # type: ignore
        DATASET_FOLDER,
        DATASET_NAME,
        STANDARD_CYCLE_COLUMNS,
        can_interpolate_qv_curve,
        ensure_dir,
        finite_array,
        infer_default_base_dir,
        safe_array,
        validate_standard_schema,
        write_json,
    )


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "count": 0,
            "min": None,
            "q1": None,
            "median": None,
            "mean": None,
            "q3": None,
            "max": None,
            "std": None,
        }
    return {
        "count": int(values.count()),
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)),
    }


def load_clean_file(path: Path, phase: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Clean {phase} file not found: {path}")
    df = pd.read_parquet(path)
    validate_standard_schema(df)
    if not df.empty:
        wrong_phase = sorted(set(df["phase"].dropna().astype(str)) - {phase})
        if wrong_phase:
            raise ValueError(f"Unexpected phase values in {path}: {wrong_phase}")
    return df


def summarize_phase(df: pd.DataFrame, phase: str) -> dict[str, Any]:
    if df.empty:
        return {
            "phase": phase,
            "n_rows": 0,
            "n_batteries": 0,
            "n_valid_cycles": 0,
            "n_model_usable_cycles": 0,
        }

    return {
        "phase": phase,
        "n_rows": int(len(df)),
        "n_batteries": int(df["battery_id"].nunique()),
        "n_valid_cycles": int(df["is_valid_cycle"].sum()),
        "n_model_usable_cycles": int(df["is_model_usable"].sum()),
        "valid_ratio": float(df["is_valid_cycle"].mean()),
        "model_usable_ratio": float(df["is_model_usable"].mean()),
        "rows_by_battery": df["battery_id"].value_counts().sort_index().to_dict(),
        "model_usable_by_battery": df.groupby("battery_id")["is_model_usable"].sum().astype(int).to_dict(),
        "exclusion_reasons": df["exclude_reason"].fillna("").replace("", "model_usable_or_no_reason").value_counts().to_dict(),
        "voltage_source_counts": df["voltage_source"].value_counts(dropna=False).to_dict(),
        "current_source_counts": df["current_source"].value_counts(dropna=False).to_dict(),
        "temperature_source_counts": df["temperature_source"].value_counts(dropna=False).to_dict(),
        "capacity_source_counts": df["capacity_source"].value_counts(dropna=False).to_dict(),
        "energy_source_counts": df["energy_source"].value_counts(dropna=False).to_dict(),
        "voltage_min_summary": numeric_summary(df["voltage_min_v"]),
        "voltage_max_summary": numeric_summary(df["voltage_max_v"]),
        "phase_capacity_summary": numeric_summary(df["phase_capacity_ah"]),
        "temperature_min_summary": numeric_summary(df["temperature_min_c"]),
        "temperature_max_summary": numeric_summary(df["temperature_max_c"]),
        "temperature_mean_summary": numeric_summary(df["temperature_mean_c"]),
        "sample_period_median_s_summary": numeric_summary(df["sample_period_median_s"]),
        "cycle_life_summary": numeric_summary(df["cycle_life"]),
        "remaining_cycles_summary": numeric_summary(df["remaining_cycles"]),
        "remaining_cycles_norm_summary": numeric_summary(df["remaining_cycles_norm"]),
        "n_points_summary": numeric_summary(df["n_points"]),
        "duration_s_summary": numeric_summary(df["duration_s"]),
    }


def row_anomalies(row: pd.Series) -> list[str]:
    anomalies: list[str] = []
    voltage = finite_array(row.get("voltage_v", []))
    capacity = finite_array(row.get("capacity_ah", []))
    time_s = finite_array(row.get("time_s", []))

    if capacity.size >= 2 and np.any(np.diff(capacity) < -1e-7):
        anomalies.append("non_monotonic_capacity")

    if voltage.size >= 2:
        voltage_range = float(np.nanmax(voltage) - np.nanmin(voltage))
        if voltage_range < 0.05:
            anomalies.append("insufficient_voltage_range")
        if row.get("phase") == "discharge" and voltage[0] < voltage[-1]:
            anomalies.append("discharge_voltage_increases_from_start_to_end")
        if row.get("phase") == "charge" and voltage[0] > voltage[-1]:
            anomalies.append("charge_voltage_decreases_from_start_to_end")

    duration = np.nan
    if time_s.size >= 2:
        if np.any(np.diff(time_s) <= 0):
            anomalies.append("non_increasing_time")
        duration = float(time_s[-1] - time_s[0])
        if duration <= 0:
            anomalies.append("non_positive_duration")
        elif duration < 60:
            anomalies.append("suspiciously_short_duration")
        elif duration > 20000:
            anomalies.append("suspiciously_long_duration")
    else:
        anomalies.append("too_few_time_points")

    if not can_interpolate_qv_curve(row.get("voltage_v", []), row.get("capacity_ah", [])):
        anomalies.append("cannot_interpolate_qv_curve")

    temperature_mean = row.get("temperature_mean_c", np.nan)
    if np.isfinite(temperature_mean) and (float(temperature_mean) < 15.0 or float(temperature_mean) > 70.0):
        anomalies.append("temperature_mean_outside_expected_ora_range")

    phase_capacity = row.get("phase_capacity_ah", np.nan)
    if not np.isfinite(phase_capacity) or float(phase_capacity) <= 0:
        anomalies.append("invalid_phase_capacity")
    elif float(phase_capacity) > 1.5:
        anomalies.append("suspiciously_high_phase_capacity")

    if np.isfinite(phase_capacity) and np.isfinite(duration) and duration > 0:
        implied_abs_current_a = float(phase_capacity) * 3600.0 / float(duration)
        if implied_abs_current_a < 0.05 or implied_abs_current_a > 2.0:
            anomalies.append("capacity_duration_current_inconsistent")

    return anomalies


def build_anomaly_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="anomaly scan", unit="cycle"):
        anomalies = row_anomalies(row)
        if anomalies:
            rows.append({
                "dataset": row.get("dataset"),
                "battery_id": row.get("battery_id"),
                "cycle_index": row.get("cycle_index"),
                "cycle_number": row.get("cycle_number"),
                "source_cycle_id": row.get("source_cycle_id"),
                "cycle_pair_id": row.get("cycle_pair_id"),
                "phase": row.get("phase"),
                "is_valid_cycle": row.get("is_valid_cycle"),
                "is_model_usable": row.get("is_model_usable"),
                "exclude_reason": row.get("exclude_reason"),
                "phase_capacity_ah": row.get("phase_capacity_ah"),
                "voltage_min_v": row.get("voltage_min_v"),
                "voltage_max_v": row.get("voltage_max_v"),
                "duration_s": row.get("duration_s"),
                "anomalies": ";".join(anomalies),
            })
    return pd.DataFrame(rows)


def plot_voltage_vs_capacity(df: pd.DataFrame, phase: str, output_dir: Path) -> int:
    if df.empty:
        return 0
    ensure_dir(output_dir)
    count = 0

    grouped = df[df["is_valid_cycle"] == True].groupby("battery_id")
    for battery_id, group in tqdm(grouped, desc=f"plots {phase}", unit="battery"):
        safe_battery = str(battery_id).replace("/", "_").replace("\\", "_")

        # Plot 1: voltage on y-axis, capacity on x-axis.
        plt.figure(figsize=(8, 5))
        for _, row in group.sort_values("cycle_number").iterrows():
            voltage = finite_array(row.get("voltage_v", []))
            capacity = finite_array(row.get("capacity_ah", []))
            n = min(voltage.size, capacity.size)
            if n >= 2:
                plt.plot(capacity[:n], voltage[:n], linewidth=0.8, alpha=0.35)
        plt.xlabel("Capacity [Ah]")
        plt.ylabel("Voltage [V]")
        plt.title(f"{DATASET_NAME} {phase} | {battery_id} | V vs Q")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{safe_battery}_{phase}_voltage_vs_capacity.png", dpi=160)
        plt.close()
        count += 1

        # Plot 2: capacity on y-axis, voltage on x-axis.
        plt.figure(figsize=(8, 5))
        for _, row in group.sort_values("cycle_number").iterrows():
            voltage = finite_array(row.get("voltage_v", []))
            capacity = finite_array(row.get("capacity_ah", []))
            n = min(voltage.size, capacity.size)
            if n >= 2:
                plt.plot(voltage[:n], capacity[:n], linewidth=0.8, alpha=0.35)
        plt.xlabel("Voltage [V]")
        plt.ylabel("Capacity [Ah]")
        plt.title(f"{DATASET_NAME} {phase} | {battery_id} | Q vs V")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{safe_battery}_{phase}_capacity_vs_voltage.png", dpi=160)
        plt.close()
        count += 1

    return count


def plot_capacity_evolution(df: pd.DataFrame, phase: str, output_dir: Path) -> Path | None:
    if df.empty:
        return None
    ensure_dir(output_dir)
    plt.figure(figsize=(9, 5))
    for battery_id, group in df[df["is_valid_cycle"] == True].groupby("battery_id"):
        group = group.sort_values("cycle_number")
        plt.plot(group["cycle_number"], group["phase_capacity_ah"], marker=".", linewidth=0.9, alpha=0.85, label=str(battery_id))
    plt.xlabel("Cycle number")
    plt.ylabel("Phase capacity [Ah]")
    plt.title(f"{DATASET_NAME} {phase} capacity evolution")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = output_dir / f"{phase}_capacity_evolution.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def audit_clean_cycles(
    *,
    charge_file: Path,
    discharge_file: Path,
    report_dir: Path,
) -> dict[str, Any]:
    ensure_dir(report_dir)
    plot_dir = report_dir / "plots"
    ensure_dir(plot_dir)

    phase_frames = {
        "charge": load_clean_file(charge_file, "charge"),
        "discharge": load_clean_file(discharge_file, "discharge"),
    }

    phase_summaries: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    anomaly_frames: list[pd.DataFrame] = []
    plot_counts: dict[str, int] = {}

    for phase, df in phase_frames.items():
        phase_summaries[phase] = summarize_phase(df, phase)
        summary_rows.append({
            "dataset": DATASET_NAME,
            "phase": phase,
            "n_rows": phase_summaries[phase].get("n_rows", 0),
            "n_batteries": phase_summaries[phase].get("n_batteries", 0),
            "n_valid_cycles": phase_summaries[phase].get("n_valid_cycles", 0),
            "n_model_usable_cycles": phase_summaries[phase].get("n_model_usable_cycles", 0),
            "valid_ratio": phase_summaries[phase].get("valid_ratio", np.nan),
            "model_usable_ratio": phase_summaries[phase].get("model_usable_ratio", np.nan),
        })

        anomaly = build_anomaly_table(df)
        if not anomaly.empty:
            anomaly_frames.append(anomaly)

        plot_counts[phase] = plot_voltage_vs_capacity(df, phase, plot_dir / phase)
        capacity_plot = plot_capacity_evolution(df, phase, plot_dir / phase)
        if capacity_plot is not None:
            plot_counts[f"{phase}_capacity_evolution"] = 1

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(report_dir / "audit_summary.csv", index=False, encoding="utf-8-sig")

    if anomaly_frames:
        anomaly_df = pd.concat(anomaly_frames, ignore_index=True)
    else:
        anomaly_df = pd.DataFrame(columns=[
            "dataset", "battery_id", "cycle_index", "cycle_number", "source_cycle_id",
            "cycle_pair_id", "phase", "is_valid_cycle", "is_model_usable",
            "exclude_reason", "phase_capacity_ah", "voltage_min_v", "voltage_max_v",
            "duration_s", "anomalies",
        ])
    anomaly_df.to_csv(report_dir / "anomaly_list.csv", index=False, encoding="utf-8-sig")

    report = {
        "dataset": DATASET_NAME,
        "schema_columns": STANDARD_CYCLE_COLUMNS,
        "source_files": {
            "charge": str(charge_file),
            "discharge": str(discharge_file),
        },
        "phase_summaries": phase_summaries,
        "audit_summary_file": str(report_dir / "audit_summary.csv"),
        "anomaly_list_file": str(report_dir / "anomaly_list.csv"),
        "plot_directory": str(plot_dir),
        "plot_counts": plot_counts,
        "n_anomalies": int(len(anomaly_df)),
    }
    write_json(report_dir / "audit_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    default_base_dir = infer_default_base_dir()
    parser = argparse.ArgumentParser(description="Audit cleaned Oxford/ORA charge and discharge cycle files.")
    parser.add_argument("--base-dir", type=str, default=str(default_base_dir), help="Project root directory.")
    parser.add_argument(
        "--clean-dir",
        type=str,
        default=str(default_base_dir / "data" / "interim" / DATASET_FOLDER / "clean"),
        help="Directory containing ORA clean parquet files.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=str(default_base_dir / "data" / "reports" / DATASET_FOLDER),
        help="Directory where audit reports and plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_dir = Path(args.clean_dir)
    report_dir = Path(args.report_dir)

    report = audit_clean_cycles(
        charge_file=clean_dir / "ora_charge_clean_cycles.parquet",
        discharge_file=clean_dir / "ora_discharge_clean_cycles.parquet",
        report_dir=report_dir,
    )

    print("\nOxford/ORA audit completed.")
    print(f"Audit report:  {report_dir / 'audit_report.json'}")
    print(f"Audit summary: {report_dir / 'audit_summary.csv'}")
    print(f"Anomaly list:  {report_dir / 'anomaly_list.csv'}")
    print(f"Plots:         {report_dir / 'plots'}")
    for phase, summary in report["phase_summaries"].items():
        print(
            f"{phase}: rows={summary.get('n_rows', 0)} | "
            f"valid={summary.get('n_valid_cycles', 0)} | "
            f"usable={summary.get('n_model_usable_cycles', 0)}"
        )


if __name__ == "__main__":
    main()
