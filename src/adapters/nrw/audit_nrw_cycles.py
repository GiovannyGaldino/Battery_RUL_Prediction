"""
Audit NASA Random Walk (NRW) clean charge and discharge cycle files.

The audit verifies schema compliance, counts valid/model-usable cycles, summarizes
signals and targets, lists anomalies, and generates voltage-capacity plots.

Run:
    python src/adapters/nrw/audit_nrw_cycles.py
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable


DATASET_CODE = "NRW"
DATASET_FOLDER = "nrw"
AUDIT_VERSION = "nrw_audit_v1.0"

DEFAULT_BASE_DIR = Path.cwd()

MIN_VALID_POINTS = 20
MIN_QV_VOLTAGE_RANGE = 0.10

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(value, dtype=float).squeeze()
        return np.atleast_1d(arr).astype(float).ravel()
    except Exception:
        return np.array([], dtype=float)


def finite_series_summary(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "min": math.nan, "q1": math.nan, "median": math.nan, "mean": math.nan, "q3": math.nan, "max": math.nan}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "q1": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "q3": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    missing = [col for col in STANDARD_CYCLE_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in STANDARD_CYCLE_COLUMNS]
    in_order = list(df.columns[: len(STANDARD_CYCLE_COLUMNS)]) == STANDARD_CYCLE_COLUMNS and len(df.columns) == len(STANDARD_CYCLE_COLUMNS)
    return {
        "schema_ok": len(missing) == 0 and len(extra) == 0 and in_order,
        "missing_columns": missing,
        "extra_columns": extra,
        "columns_in_official_order": in_order,
    }


def vector_lengths(row: pd.Series) -> dict[str, int]:
    cols = ["time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah", "energy_wh", "power_w"]
    return {col: int(len(safe_float_array(row.get(col)))) for col in cols}


def detect_cycle_anomalies(row: pd.Series) -> list[str]:
    anomalies: list[str] = []
    lengths = vector_lengths(row)
    nonzero_lengths = [length for length in lengths.values() if length > 0]
    if len(set(nonzero_lengths)) > 1:
        anomalies.append("vector_length_mismatch")

    time = safe_float_array(row.get("time_s"))
    voltage = safe_float_array(row.get("voltage_v"))
    capacity = safe_float_array(row.get("capacity_ah"))

    if len(time) < MIN_VALID_POINTS:
        anomalies.append("too_few_points")
    if len(time) > 1 and not np.all(np.diff(time) > 0):
        anomalies.append("time_not_strictly_increasing")
    if len(capacity) > 1 and np.nanmin(np.diff(capacity)) < -1e-8:
        anomalies.append("capacity_not_monotonic")
    if len(voltage) >= MIN_VALID_POINTS:
        voltage_range = float(np.nanmax(voltage) - np.nanmin(voltage))
        if voltage_range < MIN_QV_VOLTAGE_RANGE:
            anomalies.append("insufficient_voltage_range")
        phase = str(row.get("phase", ""))
        if phase == "discharge" and int(np.nanargmin(voltage)) < len(voltage) - 2:
            if float(np.nanmax(voltage[int(np.nanargmin(voltage)):]) - np.nanmin(voltage)) > 0.020:
                anomalies.append("possible_discharge_post_cutoff_rebound")
        if phase == "charge" and int(np.nanargmax(voltage)) < len(voltage) - 2:
            if float(np.nanmax(voltage) - np.nanmin(voltage[int(np.nanargmax(voltage)):])) > 0.050:
                anomalies.append("possible_charge_post_cutoff_drop")
    else:
        anomalies.append("missing_voltage_vector")

    if not bool(row.get("is_valid_cycle", False)):
        anomalies.append("invalid_cycle_flag")
    if not bool(row.get("is_model_usable", False)):
        anomalies.append("not_model_usable")

    return list(dict.fromkeys(anomalies))


def summarize_phase(df: pd.DataFrame, phase: str, schema_report: dict[str, Any]) -> dict[str, Any]:
    if df.empty:
        return {
            "phase": phase,
            "rows": 0,
            "batteries": 0,
            "schema": schema_report,
        }

    signal_sources = {}
    for col in ["voltage_source", "current_source", "temperature_source", "capacity_source", "energy_source"]:
        signal_sources[col] = df[col].astype(str).value_counts(dropna=False).to_dict() if col in df.columns else {}

    exclude_reasons = {}
    if "exclude_reason" in df.columns:
        reasons = df.loc[df["exclude_reason"].astype(str) != "", "exclude_reason"]
        if not reasons.empty:
            exclude_reasons = reasons.str.get_dummies(sep=";").sum().sort_values(ascending=False).astype(int).to_dict()

    by_battery = (
        df.groupby("battery_id")
        .agg(
            rows=("cycle_index", "count"),
            valid=("is_valid_cycle", "sum"),
            model_usable=("is_model_usable", "sum"),
            first_cycle=("cycle_number", "min"),
            last_cycle=("cycle_number", "max"),
            cycle_life=("cycle_life", "first"),
            initial_capacity_ah=("initial_capacity_ah", "first"),
            final_capacity_ah=("phase_capacity_ah", "last"),
        )
        .reset_index()
    )

    return {
        "phase": phase,
        "rows": int(len(df)),
        "batteries": int(df["battery_id"].nunique()),
        "valid_cycles": int(df["is_valid_cycle"].astype(bool).sum()),
        "model_usable_cycles": int(df["is_model_usable"].astype(bool).sum()),
        "rejected_or_not_usable": int(((~df["is_valid_cycle"].astype(bool)) | (~df["is_model_usable"].astype(bool))).sum()),
        "schema": schema_report,
        "exclude_reasons": exclude_reasons,
        "signal_sources": signal_sources,
        "voltage_min_v": finite_series_summary(df["voltage_min_v"]),
        "voltage_max_v": finite_series_summary(df["voltage_max_v"]),
        "phase_capacity_ah": finite_series_summary(df["phase_capacity_ah"]),
        "temperature_mean_c": finite_series_summary(df["temperature_mean_c"]),
        "cycle_life": finite_series_summary(df["cycle_life"]),
        "remaining_cycles": finite_series_summary(df["remaining_cycles"]),
        "remaining_cycles_norm": finite_series_summary(df["remaining_cycles_norm"]),
        "by_battery": by_battery.to_dict(orient="records"),
    }


def load_phase_file(input_file: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not input_file.exists():
        return pd.DataFrame(), {
            "schema_ok": False,
            "missing_columns": STANDARD_CYCLE_COLUMNS.copy(),
            "extra_columns": [],
            "columns_in_official_order": False,
            "file_missing": True,
        }
    df = pd.read_parquet(input_file)
    return df, validate_schema(df)


def plot_v_vs_q(df: pd.DataFrame, phase: str, output_dir: Path, max_cycles_per_battery: int | None) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[str] = []
    if df.empty:
        return output_files

    for battery_id, group in tqdm(df.groupby("battery_id"), desc=f"Plotting {phase} batteries", unit="battery"):
        group = group.sort_values("cycle_number")
        if max_cycles_per_battery is not None and len(group) > max_cycles_per_battery:
            idx = np.linspace(0, len(group) - 1, max_cycles_per_battery).round().astype(int)
            group = group.iloc[idx]

        plt.figure(figsize=(8, 6))
        for _, row in group.iterrows():
            voltage = safe_float_array(row["voltage_v"])
            capacity = safe_float_array(row["capacity_ah"])
            n = min(len(voltage), len(capacity))
            if n < MIN_VALID_POINTS:
                continue
            plt.plot(capacity[:n], voltage[:n], linewidth=0.8, alpha=0.55)
        plt.xlabel("Accumulated phase capacity, Q(V) [Ah]")
        plt.ylabel("Voltage [V]")
        plt.title(f"{DATASET_CODE} {phase} - {battery_id} - V vs Q(V)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = output_dir / f"{battery_id}_{phase}_v_vs_q.png"
        plt.savefig(out, dpi=250)
        plt.close()
        output_files.append(str(out))
    return output_files


def plot_q_vs_v(df: pd.DataFrame, phase: str, output_dir: Path, max_cycles_per_battery: int | None) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[str] = []
    if df.empty:
        return output_files

    for battery_id, group in tqdm(df.groupby("battery_id"), desc=f"Plotting {phase} Q-vs-V", unit="battery"):
        group = group.sort_values("cycle_number")
        if max_cycles_per_battery is not None and len(group) > max_cycles_per_battery:
            idx = np.linspace(0, len(group) - 1, max_cycles_per_battery).round().astype(int)
            group = group.iloc[idx]

        plt.figure(figsize=(8, 6))
        for _, row in group.iterrows():
            voltage = safe_float_array(row["voltage_v"])
            capacity = safe_float_array(row["capacity_ah"])
            n = min(len(voltage), len(capacity))
            if n < MIN_VALID_POINTS:
                continue
            plt.plot(voltage[:n], capacity[:n], linewidth=0.8, alpha=0.55)
        plt.xlabel("Voltage [V]")
        plt.ylabel("Accumulated phase capacity, Q(V) [Ah]")
        plt.title(f"{DATASET_CODE} {phase} - {battery_id} - Q(V) vs V")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = output_dir / f"{battery_id}_{phase}_q_vs_v.png"
        plt.savefig(out, dpi=250)
        plt.close()
        output_files.append(str(out))
    return output_files


def plot_capacity_fade(df: pd.DataFrame, phase: str, output_dir: Path) -> str | None:
    if df.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for battery_id, group in df.groupby("battery_id"):
        group = group.sort_values("cycle_number")
        plt.plot(group["cycle_number"], group["phase_capacity_ah"], marker="o", markersize=2, linewidth=0.8, alpha=0.7)
    plt.xlabel("Reference cycle number")
    plt.ylabel("Phase capacity [Ah]")
    plt.title(f"{DATASET_CODE} {phase} capacity by reference cycle")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / f"{DATASET_FOLDER}_{phase}_capacity_by_cycle.png"
    plt.savefig(out, dpi=250)
    plt.close()
    return str(out)


def build_anomaly_table(charge_df: pd.DataFrame, discharge_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phase, df in [("charge", charge_df), ("discharge", discharge_df)]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            anomalies = detect_cycle_anomalies(row)
            if anomalies:
                rows.append({
                    "dataset": row.get("dataset", DATASET_CODE),
                    "phase": phase,
                    "battery_id": row.get("battery_id", ""),
                    "cycle_index": row.get("cycle_index", math.nan),
                    "cycle_number": row.get("cycle_number", math.nan),
                    "source_file": row.get("source_file", ""),
                    "source_cycle_id": row.get("source_cycle_id", ""),
                    "anomalies": ";".join(anomalies),
                    "exclude_reason": row.get("exclude_reason", ""),
                    "n_points": row.get("n_points", math.nan),
                    "phase_capacity_ah": row.get("phase_capacity_ah", math.nan),
                    "voltage_min_v": row.get("voltage_min_v", math.nan),
                    "voltage_max_v": row.get("voltage_max_v", math.nan),
                    "remaining_cycles": row.get("remaining_cycles", math.nan),
                })
    return pd.DataFrame(rows)


def write_audit_summary_csv(report: dict[str, Any], output_file: Path) -> None:
    rows = []
    for phase in ["charge", "discharge"]:
        summary = report.get(phase, {})
        rows.append({
            "dataset": DATASET_CODE,
            "phase": phase,
            "rows": summary.get("rows", 0),
            "batteries": summary.get("batteries", 0),
            "valid_cycles": summary.get("valid_cycles", 0),
            "model_usable_cycles": summary.get("model_usable_cycles", 0),
            "rejected_or_not_usable": summary.get("rejected_or_not_usable", 0),
            "schema_ok": summary.get("schema", {}).get("schema_ok", False),
            "voltage_min_global": summary.get("voltage_min_v", {}).get("min", math.nan),
            "voltage_max_global": summary.get("voltage_max_v", {}).get("max", math.nan),
            "capacity_min_ah": summary.get("phase_capacity_ah", {}).get("min", math.nan),
            "capacity_max_ah": summary.get("phase_capacity_ah", {}).get("max", math.nan),
            "remaining_cycles_min": summary.get("remaining_cycles", {}).get("min", math.nan),
            "remaining_cycles_max": summary.get("remaining_cycles", {}).get("max", math.nan),
        })
    pd.DataFrame(rows).to_csv(output_file, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit NASA Random Walk clean cycle files.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Project root directory containing data/ and src/.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Input directory. Defaults to <base-dir>/data/interim/nrw/clean.")
    parser.add_argument("--reports-dir", type=Path, default=None, help="Report directory. Defaults to <base-dir>/data/reports/nrw.")
    parser.add_argument("--max-cycles-per-battery", type=int, default=None, help="Optional plotting downsample per battery.")
    parser.add_argument("--no-plots", action="store_true", help="Skip generation of visual inspection plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    input_dir = args.input_dir or (base_dir / "data" / "interim" / DATASET_FOLDER / "clean")
    reports_dir = args.reports_dir or (base_dir / "data" / "reports" / DATASET_FOLDER)
    plots_dir = reports_dir / "plots"

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    charge_file = input_dir / f"{DATASET_FOLDER}_charge_clean_cycles.parquet"
    discharge_file = input_dir / f"{DATASET_FOLDER}_discharge_clean_cycles.parquet"
    report_file = reports_dir / "audit_report.json"
    summary_file = reports_dir / "audit_summary.csv"
    anomaly_file = reports_dir / "audit_anomalies.csv"

    charge_df, charge_schema = load_phase_file(charge_file)
    discharge_df, discharge_schema = load_phase_file(discharge_file)

    charge_plots: dict[str, Any] = {}
    discharge_plots: dict[str, Any] = {}
    if not args.no_plots:
        charge_phase_dir = plots_dir / "charge"
        discharge_phase_dir = plots_dir / "discharge"
        charge_plots = {
            "v_vs_q": plot_v_vs_q(charge_df, "charge", charge_phase_dir / "v_vs_q", args.max_cycles_per_battery),
            "q_vs_v": plot_q_vs_v(charge_df, "charge", charge_phase_dir / "q_vs_v", args.max_cycles_per_battery),
            "capacity_by_cycle": plot_capacity_fade(charge_df, "charge", charge_phase_dir),
        }
        discharge_plots = {
            "v_vs_q": plot_v_vs_q(discharge_df, "discharge", discharge_phase_dir / "v_vs_q", args.max_cycles_per_battery),
            "q_vs_v": plot_q_vs_v(discharge_df, "discharge", discharge_phase_dir / "q_vs_v", args.max_cycles_per_battery),
            "capacity_by_cycle": plot_capacity_fade(discharge_df, "discharge", discharge_phase_dir),
        }

    anomaly_df = build_anomaly_table(charge_df, discharge_df)
    anomaly_df.to_csv(anomaly_file, index=False, encoding="utf-8-sig")

    report = {
        "dataset": DATASET_CODE,
        "audit_version": AUDIT_VERSION,
        "created_at": utc_now_iso(),
        "input_dir": str(input_dir),
        "charge_file": str(charge_file),
        "discharge_file": str(discharge_file),
        "charge": summarize_phase(charge_df, "charge", charge_schema),
        "discharge": summarize_phase(discharge_df, "discharge", discharge_schema),
        "anomaly_file": str(anomaly_file),
        "n_anomalies": int(len(anomaly_df)),
        "plots": {
            "charge": charge_plots,
            "discharge": discharge_plots,
        },
    }

    with report_file.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    write_audit_summary_csv(report, summary_file)

    print("\nAudit completed.")
    print(f"Audit report:   {report_file}")
    print(f"Audit summary:  {summary_file}")
    print(f"Anomaly list:   {anomaly_file}")
    if not args.no_plots:
        print(f"Plots folder:   {plots_dir}")
    print("\nCycle counts:")
    for phase in ["charge", "discharge"]:
        summary = report[phase]
        print(
            f"  {phase:<9} rows={summary.get('rows', 0):5d} | "
            f"valid={summary.get('valid_cycles', 0):5d} | "
            f"model_usable={summary.get('model_usable_cycles', 0):5d} | "
            f"schema_ok={summary.get('schema', {}).get('schema_ok', False)}"
        )


if __name__ == "__main__":
    main()
