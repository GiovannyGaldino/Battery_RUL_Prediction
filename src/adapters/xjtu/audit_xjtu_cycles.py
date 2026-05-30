# -*- coding: utf-8 -*-
"""
Audit the clean XJTU charge/discharge cycle files before feature extraction.

Inputs:
    data/interim/xjtu/clean/xjtu_charge_clean_cycles.parquet
    data/interim/xjtu/clean/xjtu_discharge_clean_cycles.parquet

Outputs:
    data/reports/xjtu/audit_report.json
    data/reports/xjtu/audit_summary.csv
    data/reports/xjtu/anomaly_cycles.csv
    data/reports/xjtu/plots_qv_voltage_y/*.png
    data/reports/xjtu/plots_vq_capacity_y/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

import matplotlib.pyplot as plt

from xjtu_schema import (
    STANDARD_CYCLE_COLUMNS,
    as_1d_float,
    clean_file,
    default_clean_dir,
    default_report_dir,
    ensure_standard_columns,
    safe_max,
    safe_min,
    validate_schema,
    write_json,
)

DEFAULT_MAX_CYCLES_PER_BATTERY_PLOT = 120


def load_clean_phase(phase: str, clean_dir: Path) -> pd.DataFrame:
    path = clean_file(phase, clean_dir)
    if not path.exists():
        return ensure_standard_columns(pd.DataFrame())
    df = pd.read_parquet(path)
    df = ensure_standard_columns(df)
    validate_schema(df)
    return df


def describe_numeric(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(values.count()),
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }


def signal_source_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for col in ["voltage_source", "current_source", "temperature_source", "capacity_source", "energy_source"]:
        if col in df.columns:
            out[col] = {str(k): int(v) for k, v in df[col].value_counts(dropna=False).to_dict().items()}
    return out


def summarize_phase(df: pd.DataFrame, phase: str) -> dict[str, Any]:
    if df.empty:
        return {
            "phase": phase,
            "n_rows": 0,
            "n_batteries": 0,
            "n_valid_cycles": 0,
            "n_model_usable_cycles": 0,
            "n_rejected_cycles": 0,
            "exclude_reason_counts": {},
            "signal_source_counts": {},
        }

    return {
        "phase": phase,
        "n_rows": int(len(df)),
        "n_batteries": int(df["battery_id"].nunique()),
        "n_valid_cycles": int(df["is_valid_cycle"].sum()),
        "n_model_usable_cycles": int(df["is_model_usable"].sum()),
        "n_rejected_cycles": int((~df["is_model_usable"].astype(bool)).sum()),
        "exclude_reason_counts": {str(k): int(v) for k, v in df.loc[~df["is_model_usable"].astype(bool), "exclude_reason"].value_counts(dropna=False).to_dict().items()},
        "signal_source_counts": signal_source_counts(df),
        "voltage_range": {
            "voltage_min_v": describe_numeric(df["voltage_min_v"]),
            "voltage_max_v": describe_numeric(df["voltage_max_v"]),
            "voltage_start_v": describe_numeric(df["voltage_start_v"]),
            "voltage_end_v": describe_numeric(df["voltage_end_v"]),
        },
        "capacity_range": {
            "phase_capacity_ah": describe_numeric(df["phase_capacity_ah"]),
            "initial_capacity_ah": describe_numeric(df["initial_capacity_ah"]),
            "soh": describe_numeric(df["soh"]),
        },
        "temperature_range": {
            "temperature_min_c": describe_numeric(df["temperature_min_c"]),
            "temperature_max_c": describe_numeric(df["temperature_max_c"]),
            "temperature_mean_c": describe_numeric(df["temperature_mean_c"]),
        },
        "target_distribution": {
            "cycle_life": describe_numeric(df["cycle_life"]),
            "remaining_cycles": describe_numeric(df["remaining_cycles"]),
            "remaining_cycles_norm": describe_numeric(df["remaining_cycles_norm"]),
        },
    }


def build_summary_table(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return pd.DataFrame(columns=[
            "phase", "battery_id", "n_rows", "n_valid_cycles", "n_model_usable_cycles",
            "n_rejected_cycles", "cycle_life", "cycle_number_min", "cycle_number_max",
            "capacity_min_ah", "capacity_max_ah", "voltage_min_v", "voltage_max_v",
        ])

    grouped = df_all.groupby(["phase", "battery_id"], dropna=False)
    rows = []
    for (phase, battery_id), group in grouped:
        rows.append({
            "phase": phase,
            "battery_id": battery_id,
            "n_rows": int(len(group)),
            "n_valid_cycles": int(group["is_valid_cycle"].sum()),
            "n_model_usable_cycles": int(group["is_model_usable"].sum()),
            "n_rejected_cycles": int((~group["is_model_usable"].astype(bool)).sum()),
            "cycle_life": float(pd.to_numeric(group["cycle_life"], errors="coerce").dropna().iloc[0]) if pd.to_numeric(group["cycle_life"], errors="coerce").dropna().size else np.nan,
            "cycle_number_min": float(pd.to_numeric(group["cycle_number"], errors="coerce").min()),
            "cycle_number_max": float(pd.to_numeric(group["cycle_number"], errors="coerce").max()),
            "capacity_min_ah": float(pd.to_numeric(group["phase_capacity_ah"], errors="coerce").min()),
            "capacity_max_ah": float(pd.to_numeric(group["phase_capacity_ah"], errors="coerce").max()),
            "voltage_min_v": float(pd.to_numeric(group["voltage_min_v"], errors="coerce").min()),
            "voltage_max_v": float(pd.to_numeric(group["voltage_max_v"], errors="coerce").max()),
        })
    return pd.DataFrame(rows).sort_values(["phase", "battery_id"]).reset_index(drop=True)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting anomalies", unit="cycle"):
        voltage = as_1d_float(row.get("voltage_v", []))
        capacity = as_1d_float(row.get("capacity_ah", []))
        time_s = as_1d_float(row.get("time_s", []))
        phase = str(row.get("phase", ""))
        warnings = []

        if len(time_s) >= 2 and np.any(np.diff(time_s) <= 0):
            warnings.append("time_not_strictly_increasing")
        if len(capacity) >= 2 and np.any(np.diff(capacity) < -1e-8):
            warnings.append("non_monotonic_capacity")
        if len(voltage) >= 2 and safe_max(voltage) - safe_min(voltage) < 0.05:
            warnings.append("insufficient_voltage_range")
        if phase == "discharge" and len(voltage) >= 5:
            min_idx = int(np.nanargmin(voltage))
            if min_idx < len(voltage) - 2 and voltage[-1] > voltage[min_idx] + 0.03:
                warnings.append("suspicious_discharge_voltage_rebound")
        if phase == "charge" and len(voltage) >= 5:
            max_idx = int(np.nanargmax(voltage))
            if max_idx < len(voltage) - 2 and voltage[-1] < voltage[max_idx] - 0.03:
                warnings.append("suspicious_charge_voltage_drop")
        if not bool(row.get("is_model_usable", False)):
            warnings.append("not_model_usable")

        if warnings:
            rows.append({
                "dataset": row.get("dataset"),
                "source_file": row.get("source_file"),
                "source_path": row.get("source_path"),
                "battery_id": row.get("battery_id"),
                "cycle_index": row.get("cycle_index"),
                "cycle_number": row.get("cycle_number"),
                "phase": row.get("phase"),
                "n_points": row.get("n_points"),
                "duration_s": row.get("duration_s"),
                "voltage_min_v": row.get("voltage_min_v"),
                "voltage_max_v": row.get("voltage_max_v"),
                "phase_capacity_ah": row.get("phase_capacity_ah"),
                "is_valid_cycle": row.get("is_valid_cycle"),
                "is_model_usable": row.get("is_model_usable"),
                "exclude_reason": row.get("exclude_reason"),
                "anomaly_flags": ";".join(sorted(set(warnings))),
            })
    return pd.DataFrame(rows)


def sample_cycles_for_plot(df_battery: pd.DataFrame, max_cycles: int) -> pd.DataFrame:
    df_battery = df_battery.sort_values("cycle_number")
    if len(df_battery) <= max_cycles:
        return df_battery
    indices = np.linspace(0, len(df_battery) - 1, max_cycles).astype(int)
    return df_battery.iloc[indices]


def plot_qv_voltage_y(df: pd.DataFrame, output_dir: Path, max_cycles: int, dpi: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_plots = 0
    if df.empty:
        return n_plots

    for (phase, battery_id), group in tqdm(df.groupby(["phase", "battery_id"]), desc="Plotting Q(V): voltage on y", unit="battery"):
        plot_group = sample_cycles_for_plot(group, max_cycles)
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted = 0
        for _, row in plot_group.iterrows():
            q = as_1d_float(row.get("capacity_ah", []))
            v = as_1d_float(row.get("voltage_v", []))
            n = min(len(q), len(v))
            if n < 5:
                continue
            ax.plot(q[:n], v[:n], linewidth=0.6, alpha=0.35)
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            continue
        ax.set_xlabel("Accumulated phase capacity, capacity_ah (Ah)")
        ax.set_ylabel("Voltage, voltage_v (V)")
        ax.set_title(f"{battery_id} | {phase} | cycles plotted: {plotted}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{battery_id}_{phase}_qv_voltage_y.png", dpi=dpi)
        plt.close(fig)
        n_plots += 1
    return n_plots


def plot_vq_capacity_y(df: pd.DataFrame, output_dir: Path, max_cycles: int, dpi: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_plots = 0
    if df.empty:
        return n_plots

    for (phase, battery_id), group in tqdm(df.groupby(["phase", "battery_id"]), desc="Plotting V(Q): capacity on y", unit="battery"):
        plot_group = sample_cycles_for_plot(group, max_cycles)
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted = 0
        for _, row in plot_group.iterrows():
            q = as_1d_float(row.get("capacity_ah", []))
            v = as_1d_float(row.get("voltage_v", []))
            n = min(len(q), len(v))
            if n < 5:
                continue
            ax.plot(v[:n], q[:n], linewidth=0.6, alpha=0.35)
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            continue
        ax.set_xlabel("Voltage, voltage_v (V)")
        ax.set_ylabel("Accumulated phase capacity, capacity_ah (Ah)")
        ax.set_title(f"{battery_id} | {phase} | cycles plotted: {plotted}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{battery_id}_{phase}_vq_capacity_y.png", dpi=dpi)
        plt.close(fig)
        n_plots += 1
    return n_plots


def build_report(df_all: pd.DataFrame, phase_frames: dict[str, pd.DataFrame], report_dir: Path, n_qv_plots: int, n_vq_plots: int) -> dict[str, Any]:
    missing_schema_by_phase = {}
    for phase, df in phase_frames.items():
        missing_schema_by_phase[phase] = [col for col in STANDARD_CYCLE_COLUMNS if col not in df.columns]

    report = {
        "schema_validation": {
            "required_columns": STANDARD_CYCLE_COLUMNS,
            "missing_columns_by_phase": missing_schema_by_phase,
            "schema_ok": all(len(v) == 0 for v in missing_schema_by_phase.values()),
        },
        "global_counts": {
            "n_rows": int(len(df_all)),
            "n_batteries": int(df_all["battery_id"].nunique()) if not df_all.empty else 0,
            "n_valid_cycles": int(df_all["is_valid_cycle"].sum()) if not df_all.empty else 0,
            "n_model_usable_cycles": int(df_all["is_model_usable"].sum()) if not df_all.empty else 0,
            "n_rejected_cycles": int((~df_all["is_model_usable"].astype(bool)).sum()) if not df_all.empty else 0,
        },
        "by_phase": {phase: summarize_phase(df, phase) for phase, df in phase_frames.items()},
        "outputs": {
            "audit_report_json": str(report_dir / "audit_report.json"),
            "audit_summary_csv": str(report_dir / "audit_summary.csv"),
            "anomaly_cycles_csv": str(report_dir / "anomaly_cycles.csv"),
            "plots_qv_voltage_y_dir": str(report_dir / "plots_qv_voltage_y"),
            "plots_vq_capacity_y_dir": str(report_dir / "plots_vq_capacity_y"),
            "n_qv_voltage_y_plots": int(n_qv_plots),
            "n_vq_capacity_y_plots": int(n_vq_plots),
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit clean XJTU charge and discharge cycle files.")
    parser.add_argument("--clean-dir", type=str, default=None, help="Input clean directory. Default: data/interim/xjtu/clean")
    parser.add_argument("--report-dir", type=str, default=None, help="Output report directory. Default: data/reports/xjtu")
    parser.add_argument("--max-cycles-per-battery-plot", type=int, default=DEFAULT_MAX_CYCLES_PER_BATTERY_PLOT, help="Maximum curves plotted per battery and phase. Default: 120.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI. Default: 150.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip visual inspection plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_dir = Path(args.clean_dir).resolve() if args.clean_dir else default_clean_dir().resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else default_report_dir().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("XJTU clean-cycle audit")
    print("=" * 80)
    print(f"Clean directory:  {clean_dir}")
    print(f"Report directory: {report_dir}")

    phase_frames = {phase: load_clean_phase(phase, clean_dir) for phase in ["charge", "discharge"]}
    df_all = pd.concat([phase_frames["charge"], phase_frames["discharge"]], ignore_index=True)
    df_all = ensure_standard_columns(df_all)

    summary_table = build_summary_table(df_all)
    summary_path = report_dir / "audit_summary.csv"
    summary_table.to_csv(summary_path, index=False, encoding="utf-8-sig")

    anomalies = detect_anomalies(df_all) if not df_all.empty else pd.DataFrame()
    anomaly_path = report_dir / "anomaly_cycles.csv"
    anomalies.to_csv(anomaly_path, index=False, encoding="utf-8-sig")

    n_qv_plots = 0
    n_vq_plots = 0
    if not args.skip_plots:
        n_qv_plots = plot_qv_voltage_y(
            df_all,
            report_dir / "plots_qv_voltage_y",
            max_cycles=args.max_cycles_per_battery_plot,
            dpi=args.dpi,
        )
        n_vq_plots = plot_vq_capacity_y(
            df_all,
            report_dir / "plots_vq_capacity_y",
            max_cycles=args.max_cycles_per_battery_plot,
            dpi=args.dpi,
        )

    report = build_report(df_all, phase_frames, report_dir, n_qv_plots, n_vq_plots)
    write_json(report_dir / "audit_report.json", report)

    print("\nAudit completed.")
    print(f"Rows:          {report['global_counts']['n_rows']}")
    print(f"Batteries:     {report['global_counts']['n_batteries']}")
    print(f"Model usable:  {report['global_counts']['n_model_usable_cycles']}")
    print(f"Rejected:      {report['global_counts']['n_rejected_cycles']}")
    print(f"Audit report:  {report['outputs']['audit_report_json']}")
    print(f"Summary CSV:   {summary_path}")
    print(f"Anomalies CSV: {anomaly_path}")


if __name__ == "__main__":
    main()
