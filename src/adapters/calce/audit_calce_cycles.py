"""
Audit CALCE clean charge and discharge cycle files.

Inputs:
    data/interim/calce/clean/calce_charge_clean_cycles.parquet
    data/interim/calce/clean/calce_discharge_clean_cycles.parquet

Outputs:
    data/reports/calce/audit_report.json
    data/reports/calce/audit_summary.csv
    data/reports/calce/anomaly_list.csv
    data/reports/calce/plots/*.png
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET = "CALCE"
DATASET_DIRNAME = "calce"
AUDIT_VERSION = "calce_audit_v6.0.0"

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


def resolve_project_root(project_root_arg: str | None) -> Path:
    if project_root_arg:
        return Path(project_root_arg).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() or (candidate / "src").exists():
            return candidate
    return current


def as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, np.ndarray):
        return pd.to_numeric(pd.Series(value.tolist()), errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    if isinstance(value, (list, tuple)):
        return pd.to_numeric(pd.Series(list(value)), errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    if isinstance(value, float) and math.isnan(value):
        return np.array([], dtype=float)
    return pd.to_numeric(pd.Series([value]), errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def finite_stats(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "min": math.nan, "q1": math.nan, "median": math.nan, "mean": math.nan, "q3": math.nan, "max": math.nan}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "q1": float(np.percentile(finite, 25)),
        "median": float(np.percentile(finite, 50)),
        "mean": float(np.mean(finite)),
        "q3": float(np.percentile(finite, 75)),
        "max": float(np.max(finite)),
    }


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    missing = [col for col in STANDARD_CYCLE_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in STANDARD_CYCLE_COLUMNS]
    order_ok = list(df.columns) == STANDARD_CYCLE_COLUMNS if not missing and not extra else False
    return {"missing_columns": missing, "extra_columns": extra, "order_ok": order_ok, "is_valid_schema": not missing and not extra}


def read_clean_file(path: Path, phase: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)
    df = pd.read_parquet(path)
    if not df.empty:
        df = df.copy()
        df["phase"] = df["phase"].fillna(phase)
    return df


def summarize_phase(df: pd.DataFrame, phase: str) -> dict[str, Any]:
    schema = validate_schema(df)
    if df.empty:
        return {
            "phase": phase,
            "schema": schema,
            "n_rows": 0,
            "n_batteries": 0,
            "n_valid_cycles": 0,
            "n_model_usable_cycles": 0,
            "n_rejected_cycles": 0,
            "by_battery": {},
            "exclude_reason_counts": {},
            "signal_source_counts": {},
            "ranges": {},
        }

    valid = df["is_valid_cycle"].fillna(False).astype(bool)
    model_usable = df["is_model_usable"].fillna(False).astype(bool)
    rejected = ~model_usable
    by_battery = df.groupby("battery_id", dropna=False).agg(
        n_rows=("cycle_index", "count"),
        n_valid_cycles=("is_valid_cycle", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
        n_model_usable_cycles=("is_model_usable", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
        cycle_min=("cycle_index", "min"),
        cycle_max=("cycle_index", "max"),
        capacity_min_ah=("phase_capacity_ah", "min"),
        capacity_max_ah=("phase_capacity_ah", "max"),
        cycle_life=("cycle_life", "first"),
        initial_capacity_ah=("initial_capacity_ah", "first"),
    )

    signal_source_counts = {}
    for col in ["voltage_source", "current_source", "temperature_source", "capacity_source", "energy_source"]:
        if col in df.columns:
            signal_source_counts[col] = {str(k): int(v) for k, v in df[col].astype(str).value_counts(dropna=False).to_dict().items()}

    ranges = {
        "voltage_min_v": finite_stats(df["voltage_min_v"]),
        "voltage_max_v": finite_stats(df["voltage_max_v"]),
        "voltage_start_v": finite_stats(df["voltage_start_v"]),
        "voltage_end_v": finite_stats(df["voltage_end_v"]),
        "phase_capacity_ah": finite_stats(df["phase_capacity_ah"]),
        "phase_energy_wh": finite_stats(df["phase_energy_wh"]),
        "temperature_min_c": finite_stats(df["temperature_min_c"]),
        "temperature_max_c": finite_stats(df["temperature_max_c"]),
        "temperature_mean_c": finite_stats(df["temperature_mean_c"]),
        "cycle_life": finite_stats(df["cycle_life"]),
        "remaining_cycles": finite_stats(df["remaining_cycles"]),
        "remaining_cycles_norm": finite_stats(df["remaining_cycles_norm"]),
        "soh": finite_stats(df["soh"]),
    }

    return {
        "phase": phase,
        "schema": schema,
        "n_rows": int(len(df)),
        "n_batteries": int(df["battery_id"].nunique()),
        "n_valid_cycles": int(valid.sum()),
        "n_model_usable_cycles": int(model_usable.sum()),
        "n_rejected_cycles": int(rejected.sum()),
        "by_battery": by_battery.to_dict(orient="index"),
        "exclude_reason_counts": {str(k): int(v) for k, v in df.loc[rejected, "exclude_reason"].astype(str).value_counts(dropna=False).to_dict().items()},
        "signal_source_counts": signal_source_counts,
        "ranges": ranges,
    }


def capacity_is_nonmonotonic(capacity: np.ndarray, tolerance: float = 1e-7) -> bool:
    finite = capacity[np.isfinite(capacity)]
    if finite.size < 2:
        return False
    return bool(np.nanmin(np.diff(finite)) < -abs(tolerance))


def insufficient_voltage_range(voltage: np.ndarray, min_span: float = 0.10) -> bool:
    finite = voltage[np.isfinite(voltage)]
    if finite.size < 2:
        return True
    return bool(np.nanmax(finite) - np.nanmin(finite) < min_span)


def suspicious_voltage_edge(phase: str, voltage: np.ndarray, threshold_v: float = 0.03) -> bool:
    finite = voltage[np.isfinite(voltage)]
    if finite.size < 5:
        return False
    if phase == "discharge":
        return bool(finite[-1] - np.nanmin(finite) > threshold_v)
    if phase == "charge":
        return bool(np.nanmax(finite) - finite[-1] > threshold_v)
    return False


def build_anomaly_list(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(records)
    for _, row in df.iterrows():
        voltage = as_float_array(row.get("voltage_v"))
        capacity = as_float_array(row.get("capacity_ah"))
        phase = str(row.get("phase"))
        reasons: list[str] = []
        if capacity_is_nonmonotonic(capacity):
            reasons.append("nonmonotonic_capacity")
        if insufficient_voltage_range(voltage):
            reasons.append("insufficient_voltage_range")
        if suspicious_voltage_edge(phase, voltage):
            reasons.append("suspicious_voltage_edge")
        if pd.to_numeric(pd.Series([row.get("duration_s", np.nan)]), errors="coerce").iloc[0] <= 0:
            reasons.append("nonpositive_duration")
        if not bool(row.get("is_model_usable", False)):
            reasons.append("not_model_usable")
        if reasons:
            records.append({
                "dataset": row.get("dataset"),
                "phase": phase,
                "battery_id": row.get("battery_id"),
                "cycle_index": row.get("cycle_index"),
                "cycle_number": row.get("cycle_number"),
                "source_file": row.get("source_file"),
                "source_cycle_id": row.get("source_cycle_id"),
                "phase_capacity_ah": row.get("phase_capacity_ah"),
                "voltage_start_v": row.get("voltage_start_v"),
                "voltage_end_v": row.get("voltage_end_v"),
                "exclude_reason": row.get("exclude_reason"),
                "anomaly_reasons": "|".join(dict.fromkeys(reasons)),
            })
    return pd.DataFrame(records)


def choose_plot_cycles(group: pd.DataFrame, max_cycles: int) -> pd.DataFrame:
    if len(group) <= max_cycles:
        return group
    positions = np.linspace(0, len(group) - 1, max_cycles).round().astype(int)
    return group.iloc[sorted(set(positions))]


def plot_battery_curves(df: pd.DataFrame, output_dir: Path, max_cycles_per_battery: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files: list[str] = []
    if df.empty:
        return plot_files
    for (phase, battery_id), group in df.groupby(["phase", "battery_id"], sort=True):
        group = group.sort_values("cycle_index")
        selected = choose_plot_cycles(group, max_cycles_per_battery)
        if selected.empty:
            continue

        plt.figure(figsize=(8, 6))
        for _, row in selected.iterrows():
            q = as_float_array(row.get("capacity_ah"))
            v = as_float_array(row.get("voltage_v"))
            n = min(len(q), len(v))
            if n >= 2:
                plt.plot(q[:n], v[:n], linewidth=0.8, alpha=0.7)
        plt.xlabel("capacity_ah")
        plt.ylabel("voltage_v")
        plt.title(f"{DATASET} {phase} V x Q - {battery_id}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        file_vq = output_dir / f"{phase}_{battery_id}_v_vs_q.png"
        plt.savefig(file_vq, dpi=160)
        plt.close()
        plot_files.append(str(file_vq))

        plt.figure(figsize=(8, 6))
        for _, row in selected.iterrows():
            q = as_float_array(row.get("capacity_ah"))
            v = as_float_array(row.get("voltage_v"))
            n = min(len(q), len(v))
            if n >= 2:
                plt.plot(v[:n], q[:n], linewidth=0.8, alpha=0.7)
        plt.xlabel("voltage_v")
        plt.ylabel("capacity_ah")
        plt.title(f"{DATASET} {phase} Q x V - {battery_id}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        file_qv = output_dir / f"{phase}_{battery_id}_q_vs_v.png"
        plt.savefig(file_qv, dpi=160)
        plt.close()
        plot_files.append(str(file_qv))
    return plot_files


def make_summary_csv(report: dict[str, Any], output_file: Path) -> None:
    rows = []
    for phase, summary in report["phases"].items():
        rows.append({
            "dataset": DATASET,
            "phase": phase,
            "schema_valid": summary["schema"]["is_valid_schema"],
            "schema_order_ok": summary["schema"].get("order_ok"),
            "n_rows": summary["n_rows"],
            "n_batteries": summary["n_batteries"],
            "n_valid_cycles": summary["n_valid_cycles"],
            "n_model_usable_cycles": summary["n_model_usable_cycles"],
            "n_rejected_cycles": summary["n_rejected_cycles"],
        })
    pd.DataFrame(rows).to_csv(output_file, index=False, encoding="utf-8-sig")



def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CALCE clean cycle files.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--clean-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--max-cycles-per-battery", type=int, default=40)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    clean_dir = Path(args.clean_dir).expanduser().resolve() if args.clean_dir else project_root / "data" / "interim" / DATASET_DIRNAME / "clean"
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else project_root / "data" / "reports" / DATASET_DIRNAME
    report_dir.mkdir(parents=True, exist_ok=True)

    charge_df = read_clean_file(clean_dir / "calce_charge_clean_cycles.parquet", "charge")
    discharge_df = read_clean_file(clean_dir / "calce_discharge_clean_cycles.parquet", "discharge")
    combined = pd.concat([charge_df, discharge_df], ignore_index=True) if not (charge_df.empty and discharge_df.empty) else pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)

    anomaly_df = build_anomaly_list(combined)
    anomaly_file = report_dir / "anomaly_list.csv"
    anomaly_df.to_csv(anomaly_file, index=False, encoding="utf-8-sig")

    plot_files: list[str] = []
    if not args.skip_plots:
        plot_files = plot_battery_curves(combined, report_dir / "plots", args.max_cycles_per_battery)

    report = {
        "dataset": DATASET,
        "audit_version": AUDIT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "clean_dir": str(clean_dir),
        "report_dir": str(report_dir),
        "phases": {
            "charge": summarize_phase(charge_df, "charge"),
            "discharge": summarize_phase(discharge_df, "discharge"),
        },
        "anomaly_file": str(anomaly_file),
        "n_anomalies": int(len(anomaly_df)),
        "plot_files": plot_files,
    }

    report_file = report_dir / "audit_report.json"
    summary_file = report_dir / "audit_summary.csv"
    with report_file.open("w", encoding="utf-8") as fp:
        json.dump(json_safe(report), fp, indent=2, ensure_ascii=False)
    make_summary_csv(report, summary_file)

    print(f"[OK] Audit report: {report_file}")
    print(f"[OK] Audit summary: {summary_file}")
    print(f"[OK] Anomaly list: {anomaly_file}")
    if plot_files:
        print(f"[OK] Plots generated: {len(plot_files)}")


if __name__ == "__main__":
    main()
