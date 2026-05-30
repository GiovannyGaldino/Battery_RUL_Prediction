"""
Audit NASA PCoE clean charge and discharge cycle files.

Outputs
-------
data/reports/pcoe/audit_report.json
data/reports/pcoe/audit_summary.csv
data/reports/pcoe/anomaly_list.csv
data/reports/pcoe/plots/*.png

The audit checks schema compliance, cycle counts, signal availability, voltage and
capacity ranges, target distribution, and produces visual inspection plots for both
charge and discharge branches.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

try:
    from .pcoe_schema import DEFAULT_BASE_DIR, STANDARD_CYCLE_COLUMNS, validate_clean_schema
except ImportError:  # Allows direct execution from the adapter folder.
    from pcoe_schema import DEFAULT_BASE_DIR, STANDARD_CYCLE_COLUMNS, validate_clean_schema  # type: ignore


# ---------------------------------------------------------------------------
# CLI and paths
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit NASA PCoE clean cycle files.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Project root containing data/interim and data/reports.",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=None,
        help="PCoE interim directory. Default: <base-dir>/data/interim/pcoe.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="PCoE report directory. Default: <base-dir>/data/reports/pcoe.",
    )
    parser.add_argument(
        "--max-cycles-per-plot",
        type=int,
        default=80,
        help="Maximum number of cycles plotted per battery and phase.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip visual inspection plots.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    base_dir = args.base_dir
    interim_dir = args.interim_dir or base_dir / "data" / "interim" / "pcoe"
    reports_dir = args.reports_dir or base_dir / "data" / "reports" / "pcoe"
    clean_dir = interim_dir / "clean"
    plots_dir = reports_dir / "plots"

    for path in [reports_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "interim_dir": interim_dir,
        "clean_dir": clean_dir,
        "reports_dir": reports_dir,
        "plots_dir": plots_dir,
        "charge_clean_file": clean_dir / "pcoe_charge_clean_cycles.parquet",
        "discharge_clean_file": clean_dir / "pcoe_discharge_clean_cycles.parquet",
        "audit_report_file": reports_dir / "audit_report.json",
        "audit_summary_file": reports_dir / "audit_summary.csv",
        "anomaly_file": reports_dir / "anomaly_list.csv",
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def as_array(value: Any) -> np.ndarray:
    try:
        if isinstance(value, np.ndarray):
            return value.astype(float, copy=False).reshape(-1)
        if isinstance(value, list):
            return np.asarray(value, dtype=float).reshape(-1)
        if pd.isna(value):
            return np.asarray([], dtype=float)
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def numeric_stats(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if len(values) == 0:
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
        "count": int(len(values)),
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def load_clean_files(paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    schema_checks: dict[str, Any] = {}

    for phase in ["charge", "discharge"]:
        file_path = paths[f"{phase}_clean_file"]
        if not file_path.exists():
            raise FileNotFoundError(f"Missing clean {phase} file: {file_path}")
        df = pd.read_parquet(file_path)
        try:
            validate_clean_schema(df.columns.tolist())
            schema_checks[phase] = {
                "file": str(file_path),
                "ok": True,
                "n_rows": int(len(df)),
                "n_columns": int(len(df.columns)),
            }
        except Exception as exc:
            schema_checks[phase] = {
                "file": str(file_path),
                "ok": False,
                "error": str(exc),
                "n_rows": int(len(df)),
                "n_columns": int(len(df.columns)),
            }
            raise
        if not df.empty:
            frames.append(df[STANDARD_CYCLE_COLUMNS])

    if frames:
        return pd.concat(frames, ignore_index=True), schema_checks
    return pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS), schema_checks


# ---------------------------------------------------------------------------
# Audit calculations
# ---------------------------------------------------------------------------


def build_phase_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows

    for (phase, battery_id), group in df.groupby(["phase", "battery_id"]):
        rows.append(
            {
                "phase": phase,
                "battery_id": battery_id,
                "n_cycles": int(len(group)),
                "n_valid_cycles": int(group["is_valid_cycle"].sum()),
                "n_model_usable_cycles": int(group["is_model_usable"].sum()),
                "cycle_life": float(pd.to_numeric(group["cycle_life"], errors="coerce").max())
                if pd.to_numeric(group["cycle_life"], errors="coerce").notna().any() else np.nan,
                "first_cycle_number": float(pd.to_numeric(group["cycle_number"], errors="coerce").min()),
                "last_cycle_number": float(pd.to_numeric(group["cycle_number"], errors="coerce").max()),
                "capacity_min_ah": float(pd.to_numeric(group["phase_capacity_ah"], errors="coerce").min()),
                "capacity_max_ah": float(pd.to_numeric(group["phase_capacity_ah"], errors="coerce").max()),
                "voltage_min_v": float(pd.to_numeric(group["voltage_min_v"], errors="coerce").min()),
                "voltage_max_v": float(pd.to_numeric(group["voltage_max_v"], errors="coerce").max()),
                "temperature_source_modes": ";".join(sorted(group["temperature_source"].dropna().astype(str).unique().tolist())),
            }
        )
    return rows


def signal_availability(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    return {
        "voltage_source": df["voltage_source"].value_counts(dropna=False).to_dict(),
        "current_source": df["current_source"].value_counts(dropna=False).to_dict(),
        "temperature_source": df["temperature_source"].value_counts(dropna=False).to_dict(),
        "capacity_source": df["capacity_source"].value_counts(dropna=False).to_dict(),
        "energy_source": df["energy_source"].value_counts(dropna=False).to_dict(),
        "has_voltage": df["has_voltage"].value_counts(dropna=False).to_dict(),
        "has_current": df["has_current"].value_counts(dropna=False).to_dict(),
        "has_temperature": df["has_temperature"].value_counts(dropna=False).to_dict(),
        "has_capacity": df["has_capacity"].value_counts(dropna=False).to_dict(),
        "has_energy": df["has_energy"].value_counts(dropna=False).to_dict(),
    }


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        voltage = as_array(row["voltage_v"])
        capacity = as_array(row["capacity_ah"])
        time_s = as_array(row["time_s"])
        issues: list[str] = []

        if len(capacity) >= 2 and np.nanmin(np.diff(capacity)) < -1e-6:
            issues.append("non_monotonic_capacity")
        if len(voltage) < 2 or float(np.nanmax(voltage) - np.nanmin(voltage)) < 0.05:
            issues.append("insufficient_voltage_range")
        if len(time_s) >= 2:
            duration = float(time_s[-1] - time_s[0])
            if duration <= 0 or duration > 24 * 3600:
                issues.append("abnormal_duration")
        else:
            issues.append("invalid_time_vector")

        phase_capacity = float(row.get("phase_capacity_ah", np.nan))
        initial_capacity = float(row.get("initial_capacity_ah", np.nan))
        eol_capacity = float(row.get("eol_capacity_ah", np.nan))
        cycle_life = float(row.get("cycle_life", np.nan))
        cycle_number = float(row.get("cycle_number", np.nan))
        if np.isfinite(phase_capacity) and phase_capacity < 0.20:
            issues.append("very_low_phase_capacity")
        if np.isfinite(phase_capacity) and np.isfinite(initial_capacity) and initial_capacity > 0:
            if phase_capacity / initial_capacity < 0.50:
                issues.append("low_phase_capacity_ratio")
        if (
            str(row.get("phase", "")).lower() == "discharge"
            and np.isfinite(phase_capacity)
            and np.isfinite(eol_capacity)
            and np.isfinite(cycle_life)
            and np.isfinite(cycle_number)
            and cycle_number < cycle_life
            and phase_capacity < eol_capacity
        ):
            issues.append("discharge_capacity_below_eol_before_cycle_life")

        phase = str(row["phase"])
        if len(voltage) >= 4:
            if phase == "discharge":
                min_idx = int(np.nanargmin(voltage))
                if min_idx < len(voltage) - 3:
                    rebound = float(np.nanmax(voltage[min_idx:]) - voltage[min_idx])
                    if rebound > 0.03:
                        issues.append("possible_untrimmed_voltage_rebound")
            elif phase == "charge":
                max_idx = int(np.nanargmax(voltage))
                if max_idx < len(voltage) - 3:
                    drop_after_max = float(voltage[max_idx] - np.nanmin(voltage[max_idx:]))
                    if drop_after_max > 0.05 and float(voltage[-1]) < 4.0:
                        issues.append("possible_terminal_charge_voltage_drop")

        if issues:
            records.append(
                {
                    "phase": row["phase"],
                    "battery_id": row["battery_id"],
                    "cycle_number": row["cycle_number"],
                    "source_file": row["source_file"],
                    "source_cycle_id": row["source_cycle_id"],
                    "cycle_pair_id": row["cycle_pair_id"],
                    "issues": ";".join(issues),
                    "n_points": row["n_points"],
                    "duration_s": row["duration_s"],
                    "phase_capacity_ah": row["phase_capacity_ah"],
                    "voltage_min_v": row["voltage_min_v"],
                    "voltage_max_v": row["voltage_max_v"],
                    "is_model_usable": row["is_model_usable"],
                    "exclude_reason": row["exclude_reason"],
                }
            )
    return pd.DataFrame(records)


def build_report(df: pd.DataFrame, schema_checks: dict[str, Any], anomaly_df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "created_at": utc_now(),
            "schema_checks": schema_checks,
            "n_rows": 0,
            "message": "Clean files are empty.",
        }

    by_phase = {}
    for phase, group in df.groupby("phase"):
        by_phase[phase] = {
            "n_rows": int(len(group)),
            "n_batteries": int(group["battery_id"].nunique()),
            "n_valid_cycles": int(group["is_valid_cycle"].sum()),
            "n_model_usable_cycles": int(group["is_model_usable"].sum()),
            "exclude_reason_counts": group["exclude_reason"].replace("", "model_usable").value_counts(dropna=False).to_dict(),
            "voltage_min_v": numeric_stats(group["voltage_min_v"]),
            "voltage_max_v": numeric_stats(group["voltage_max_v"]),
            "voltage_start_v": numeric_stats(group["voltage_start_v"]),
            "voltage_end_v": numeric_stats(group["voltage_end_v"]),
            "phase_capacity_ah": numeric_stats(group["phase_capacity_ah"]),
            "temperature_mean_c": numeric_stats(group["temperature_mean_c"]),
            "cycle_life": numeric_stats(group["cycle_life"]),
            "remaining_cycles": numeric_stats(group["remaining_cycles"]),
            "remaining_cycles_norm": numeric_stats(group["remaining_cycles_norm"]),
        }

    return {
        "created_at": utc_now(),
        "schema_checks": schema_checks,
        "n_rows": int(len(df)),
        "n_batteries": int(df["battery_id"].nunique()),
        "phases": sorted(df["phase"].unique().tolist()),
        "by_phase": by_phase,
        "signal_availability": signal_availability(df),
        "n_anomalies": int(len(anomaly_df)),
        "n_model_usable_anomalies": int(anomaly_df["is_model_usable"].sum()) if not anomaly_df.empty and "is_model_usable" in anomaly_df.columns else 0,
        "anomaly_issue_counts": anomaly_df["issues"].value_counts(dropna=False).to_dict() if not anomaly_df.empty else {},
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_battery_phase_curves(
    group: pd.DataFrame,
    *,
    phase: str,
    battery_id: str,
    output_dir: Path,
    max_cycles: int,
) -> None:
    ordered = group.sort_values("cycle_number")
    if len(ordered) > max_cycles:
        positions = np.linspace(0, len(ordered) - 1, max_cycles).astype(int)
        ordered = ordered.iloc[positions]

    # Voltage vs capacity: x = capacity, y = voltage.
    plt.figure(figsize=(8, 5))
    for _, row in ordered.iterrows():
        capacity = as_array(row["capacity_ah"])
        voltage = as_array(row["voltage_v"])
        n = min(len(capacity), len(voltage))
        if n >= 2:
            plt.plot(capacity[:n], voltage[:n], linewidth=0.8, alpha=0.45)
    plt.xlabel("Capacity (Ah)")
    plt.ylabel("Voltage (V)")
    plt.title(f"{battery_id} - {phase} - V x Q")
    plt.tight_layout()
    plt.savefig(output_dir / f"{battery_id}_{phase}_voltage_vs_capacity.png", dpi=160)
    plt.close()

    # Capacity vs voltage: x = voltage, y = capacity.
    plt.figure(figsize=(8, 5))
    for _, row in ordered.iterrows():
        capacity = as_array(row["capacity_ah"])
        voltage = as_array(row["voltage_v"])
        n = min(len(capacity), len(voltage))
        if n >= 2:
            plt.plot(voltage[:n], capacity[:n], linewidth=0.8, alpha=0.45)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Capacity (Ah)")
    plt.title(f"{battery_id} - {phase} - Q x V")
    plt.tight_layout()
    plt.savefig(output_dir / f"{battery_id}_{phase}_capacity_vs_voltage.png", dpi=160)
    plt.close()


def generate_plots(df: pd.DataFrame, plots_dir: Path, max_cycles: int) -> int:
    if df.empty:
        return 0
    count = 0
    grouped = list(df.groupby(["phase", "battery_id"]))
    for (phase, battery_id), group in tqdm(grouped, desc="Generating audit plots", unit="battery-phase"):
        phase_dir = plots_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        plot_battery_phase_curves(
            group,
            phase=str(phase),
            battery_id=str(battery_id),
            output_dir=phase_dir,
            max_cycles=max_cycles,
        )
        count += 2
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    df, schema_checks = load_clean_files(paths)
    anomaly_df = detect_anomalies(df)
    anomaly_df.to_csv(paths["anomaly_file"], index=False, encoding="utf-8")

    summary_rows = build_phase_summary(df)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(paths["audit_summary_file"], index=False, encoding="utf-8")

    n_plots = 0
    if not args.skip_plots:
        n_plots = generate_plots(df, paths["plots_dir"], args.max_cycles_per_plot)

    report = build_report(df, schema_checks, anomaly_df)
    report["audit_summary_file"] = str(paths["audit_summary_file"])
    report["anomaly_file"] = str(paths["anomaly_file"])
    report["plots_dir"] = str(paths["plots_dir"])
    report["n_plots_generated"] = int(n_plots)

    with open(paths["audit_report_file"], "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("Audit completed.")
    print(f"Audit report:  {paths['audit_report_file']}")
    print(f"Audit summary: {paths['audit_summary_file']}")
    print(f"Anomalies:     {paths['anomaly_file']}")
    print(f"Plots:         {paths['plots_dir']}")


if __name__ == "__main__":
    main()
