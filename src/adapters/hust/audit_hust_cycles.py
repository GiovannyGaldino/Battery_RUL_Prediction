#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audit cleaned HUST charge/discharge cycle files and generate grouped inspection plots.

Input files:
    data/interim/hust/clean/hust_charge_clean_cycles.parquet
    data/interim/hust/clean/hust_discharge_clean_cycles.parquet

Output files:
    data/reports/hust/audit_report.json
    data/reports/hust/audit_summary.csv
    data/reports/hust/anomaly_list.csv
    data/reports/hust/plots/curves_by_battery/plot_manifest.csv
    data/reports/hust/plots/curves_by_battery/<phase>/voltage_vs_capacity/*.png
    data/reports/hust/plots/curves_by_battery/<phase>/capacity_vs_voltage/*.png

The plotting step creates one image per battery and phase, overlaying all selected
cycles from the same battery in the same figure. This replaces the previous behavior
that produced one image per cycle.

Example using the active Conda environment directly:
    python \
        src/adapters/hust/audit_hust_cycles.py

Example with a custom project root:
    python \
        src/adapters/hust/audit_hust_cycles.py \
        --base-dir "."
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .hust_schema import (
        DATASET_NAME,
        DEFAULT_BASE_DIR,
        STANDARD_CYCLE_COLUMNS,
        dump_json,
        get_tqdm,
        read_parquet_if_exists,
        resolve_base_paths,
        safe_float,
        safe_numeric_array,
        utc_now_iso,
    )
except ImportError:
    from hust_schema import (  # type: ignore
        DATASET_NAME,
        DEFAULT_BASE_DIR,
        STANDARD_CYCLE_COLUMNS,
        dump_json,
        get_tqdm,
        read_parquet_if_exists,
        resolve_base_paths,
        safe_float,
        safe_numeric_array,
        utc_now_iso,
    )


AUDIT_VERSION = "hust_audit_v1.1_grouped_plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit cleaned HUST charge/discharge cycle files and generate grouped battery plots."
    )
    parser.add_argument("--base-dir", type=str, default=str(DEFAULT_BASE_DIR), help="Project root directory.")
    parser.add_argument(
        "--interim-dir",
        type=str,
        default=None,
        help="Dataset interim directory. Default: <base>/data/interim/hust",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Dataset reports directory. Default: <base>/data/reports/hust",
    )
    parser.add_argument(
        "--max-cycles-per-battery",
        type=int,
        default=0,
        help=(
            "Maximum number of cycles to draw per battery and phase. "
            "Use 0 or a negative value to plot all selected cycles."
        ),
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=1,
        help="Draw one cycle every N selected rows within each battery. Use 1 for all selected cycles.",
    )
    parser.add_argument(
        "--plot-all-cycles",
        action="store_true",
        help=(
            "Plot all valid cycles, including cycles that are not model-usable. "
            "By default, plots use model-usable cycles when available."
        ),
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip visual inspection plots.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Resolution of generated PNG figures.",
    )
    return parser.parse_args()


def describe_numeric(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    finite = s[np.isfinite(s)]
    if finite.empty:
        return {
            "count": int(len(s)),
            "valid": 0,
            "missing": int(s.isna().sum()),
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": int(len(s)),
        "valid": int(len(finite)),
        "missing": int(s.isna().sum()),
        "min": float(finite.min()),
        "q1": float(finite.quantile(0.25)),
        "median": float(finite.median()),
        "q3": float(finite.quantile(0.75)),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()) if len(finite) > 1 else 0.0,
    }


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    missing = [col for col in STANDARD_CYCLE_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in STANDARD_CYCLE_COLUMNS]
    order_ok = (
        list(df.columns[: len(STANDARD_CYCLE_COLUMNS)]) == STANDARD_CYCLE_COLUMNS
        if len(df.columns) >= len(STANDARD_CYCLE_COLUMNS)
        else False
    )
    return {
        "columns_total": int(len(df.columns)),
        "missing_columns": missing,
        "extra_columns": extra,
        "official_order_ok": bool(order_ok),
        "schema_ok": len(missing) == 0,
    }


def vector_length_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    vector_columns = ["time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah", "energy_wh", "power_w"]
    tqdm = get_tqdm()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="vector check", unit="cycle", leave=False):
        lengths = {
            col: len(row[col]) if isinstance(row.get(col), (list, tuple, np.ndarray)) else 0
            for col in vector_columns
            if col in row
        }
        if len(set(lengths.values())) > 1 or any(v <= 0 for v in lengths.values()):
            issues.append(
                {
                    "battery_id": row.get("battery_id", ""),
                    "phase": row.get("phase", ""),
                    "cycle_number": row.get("cycle_number", np.nan),
                    "source_cycle_id": row.get("source_cycle_id", ""),
                    "issue": "unequal_or_empty_vector_lengths",
                    "value": str(lengths),
                }
            )
    return issues


def find_anomalies(df: pd.DataFrame) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    tqdm = get_tqdm()
    iterator = tqdm(df.iterrows(), total=len(df), desc="anomaly scan", unit="cycle", leave=False)
    for _, row in iterator:
        battery_id = row.get("battery_id", "")
        phase = row.get("phase", "")
        cycle_number = row.get("cycle_number", np.nan)
        source_cycle_id = row.get("source_cycle_id", "")
        voltage_v = safe_numeric_array(row.get("voltage_v"))
        capacity_ah = safe_numeric_array(row.get("capacity_ah"))
        time_s = safe_numeric_array(row.get("time_s"))

        def add(issue: str, value: Any = "") -> None:
            anomalies.append(
                {
                    "battery_id": battery_id,
                    "phase": phase,
                    "cycle_number": cycle_number,
                    "source_cycle_id": source_cycle_id,
                    "issue": issue,
                    "value": value,
                }
            )

        if len(time_s) < 2 or not np.all(np.diff(time_s) > 0):
            add("time_not_strictly_increasing")
        if len(capacity_ah) < 2 or np.any(np.diff(capacity_ah) < -1e-8):
            add("non_monotonic_capacity")
        if len(voltage_v) and float(np.nanmax(voltage_v) - np.nanmin(voltage_v)) < 0.03:
            add("insufficient_voltage_range", float(np.nanmax(voltage_v) - np.nanmin(voltage_v)))
        if safe_float(row.get("duration_s")) <= 0:
            add("non_positive_duration", row.get("duration_s"))
        if safe_float(row.get("phase_capacity_ah")) <= 0:
            add("non_positive_phase_capacity", row.get("phase_capacity_ah"))
        if str(row.get("exclude_reason", "")) and bool(row.get("is_model_usable", False)):
            add("model_usable_with_exclude_reason", row.get("exclude_reason"))
        if not bool(row.get("is_valid_cycle", False)) and bool(row.get("is_model_usable", False)):
            add("model_usable_but_invalid")
    return anomalies


def signal_availability(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    cols = ["has_voltage", "has_current", "has_temperature", "has_capacity", "has_energy"]
    out: dict[str, Any] = {}
    for col in cols:
        if col in df.columns:
            out[col] = df[col].value_counts(dropna=False).to_dict()
    for col in ["voltage_source", "current_source", "temperature_source", "capacity_source", "energy_source"]:
        if col in df.columns:
            out[col] = df[col].astype(str).value_counts(dropna=False).to_dict()
    return out


def phase_summary(df: pd.DataFrame, phase: str) -> dict[str, Any]:
    schema = validate_schema(df)
    vector_issues = vector_length_issues(df) if not df.empty else []
    anomalies = find_anomalies(df) if not df.empty else []
    summary = {
        "phase": phase,
        "rows": int(len(df)),
        "batteries": int(df["battery_id"].nunique()) if "battery_id" in df.columns and not df.empty else 0,
        "valid_cycles": int(df["is_valid_cycle"].sum()) if "is_valid_cycle" in df.columns and not df.empty else 0,
        "model_usable_cycles": int(df["is_model_usable"].sum()) if "is_model_usable" in df.columns and not df.empty else 0,
        "schema": schema,
        "exclude_reason_counts": df["exclude_reason"].astype(str).value_counts(dropna=False).to_dict() if "exclude_reason" in df.columns and not df.empty else {},
        "signal_availability": signal_availability(df),
        "voltage": {
            "min": describe_numeric(df["voltage_min_v"]) if "voltage_min_v" in df.columns else {},
            "max": describe_numeric(df["voltage_max_v"]) if "voltage_max_v" in df.columns else {},
            "start": describe_numeric(df["voltage_start_v"]) if "voltage_start_v" in df.columns else {},
            "end": describe_numeric(df["voltage_end_v"]) if "voltage_end_v" in df.columns else {},
        },
        "capacity": describe_numeric(df["phase_capacity_ah"]) if "phase_capacity_ah" in df.columns else {},
        "temperature_mean": describe_numeric(df["temperature_mean_c"]) if "temperature_mean_c" in df.columns else {},
        "cycle_life": describe_numeric(df["cycle_life"]) if "cycle_life" in df.columns else {},
        "remaining_cycles": describe_numeric(df["remaining_cycles"]) if "remaining_cycles" in df.columns else {},
        "remaining_cycles_norm": describe_numeric(df["remaining_cycles_norm"]) if "remaining_cycles_norm" in df.columns else {},
        "vector_length_issue_count": int(len(vector_issues)),
        "anomaly_count": int(len(anomalies)),
    }
    return summary


def make_summary_rows(phase_reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for phase, report in phase_reports.items():
        rows.append(
            {
                "dataset": DATASET_NAME,
                "phase": phase,
                "rows": report.get("rows", 0),
                "batteries": report.get("batteries", 0),
                "valid_cycles": report.get("valid_cycles", 0),
                "model_usable_cycles": report.get("model_usable_cycles", 0),
                "schema_ok": report.get("schema", {}).get("schema_ok", False),
                "missing_columns": ";".join(report.get("schema", {}).get("missing_columns", [])),
                "vector_length_issue_count": report.get("vector_length_issue_count", 0),
                "anomaly_count": report.get("anomaly_count", 0),
                "phase_capacity_ah_median": report.get("capacity", {}).get("median"),
                "voltage_min_v_median": report.get("voltage", {}).get("min", {}).get("median"),
                "voltage_max_v_median": report.get("voltage", {}).get("max", {}).get("median"),
                "remaining_cycles_median": report.get("remaining_cycles", {}).get("median"),
            }
        )
    return pd.DataFrame(rows)


def sanitize_filename(value: Any) -> str:
    text = str(value) if value is not None else "unknown"
    out = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "unknown"


def select_cycles_for_plotting(df: pd.DataFrame, plot_all_cycles: bool, plot_every: int, max_cycles_per_battery: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    selected = df.copy()
    if not plot_all_cycles:
        usable = selected[selected.get("is_model_usable", False) == True].copy()
        if not usable.empty:
            selected = usable
        else:
            valid = selected[selected.get("is_valid_cycle", False) == True].copy()
            if not valid.empty:
                selected = valid
    else:
        valid = selected[selected.get("is_valid_cycle", False) == True].copy()
        if not valid.empty:
            selected = valid

    sort_cols = [col for col in ["battery_id", "cycle_index", "cycle_number"] if col in selected.columns]
    if sort_cols:
        selected = selected.sort_values(sort_cols).reset_index(drop=True)

    plot_every = max(1, int(plot_every))
    max_cycles = int(max_cycles_per_battery)

    chunks: list[pd.DataFrame] = []
    for _, group in selected.groupby("battery_id", sort=True):
        group = group.iloc[::plot_every].copy()
        if max_cycles > 0 and len(group) > max_cycles:
            # Uniformly sample the cycle history instead of truncating early life only.
            idx = np.linspace(0, len(group) - 1, max_cycles).round().astype(int)
            group = group.iloc[np.unique(idx)].copy()
        chunks.append(group)

    if not chunks:
        return selected.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True)


def plot_grouped_curves_for_battery(
    group: pd.DataFrame,
    phase: str,
    battery_id: str,
    output_root: Path,
    dpi: int,
) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    safe_battery = sanitize_filename(battery_id)

    plot_specs = [
        {
            "plot_type": "voltage_vs_capacity",
            "subdir": "voltage_vs_capacity",
            "x_label": "Capacity [Ah]",
            "y_label": "Voltage [V]",
            "file_suffix": "v_vs_q",
            "x_vector": "capacity_ah",
            "y_vector": "voltage_v",
        },
        {
            "plot_type": "capacity_vs_voltage",
            "subdir": "capacity_vs_voltage",
            "x_label": "Voltage [V]",
            "y_label": "Capacity [Ah]",
            "file_suffix": "q_vs_v",
            "x_vector": "voltage_v",
            "y_vector": "capacity_ah",
        },
    ]

    for spec in plot_specs:
        output_dir = output_root / phase / spec["subdir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"hust_{phase}_{safe_battery}_{spec['file_suffix']}.png"

        fig, ax = plt.subplots(figsize=(8.5, 6.0))
        n_plotted = 0
        min_cycle = np.nan
        max_cycle = np.nan

        cycles = pd.to_numeric(group.get("cycle_number"), errors="coerce")
        if cycles.notna().any():
            min_cycle = float(cycles.min())
            max_cycle = float(cycles.max())

        for _, row in group.iterrows():
            x = safe_numeric_array(row.get(spec["x_vector"]))
            y = safe_numeric_array(row.get(spec["y_vector"]))
            if len(x) < 2 or len(y) < 2:
                continue
            n = min(len(x), len(y))
            x = x[:n]
            y = y[:n]
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 2:
                continue
            ax.plot(x[valid], y[valid], linewidth=0.55, alpha=0.22)
            n_plotted += 1

        if n_plotted == 0:
            plt.close(fig)
            continue

        ax.set_xlabel(spec["x_label"])
        ax.set_ylabel(spec["y_label"])
        ax.set_title(f"HUST {phase} curves | {battery_id} | n={n_plotted}")
        subtitle = []
        if np.isfinite(min_cycle) and np.isfinite(max_cycle):
            subtitle.append(f"cycles {min_cycle:g}–{max_cycle:g}")
        if subtitle:
            ax.text(0.01, 0.99, " | ".join(subtitle), transform=ax.transAxes, va="top", ha="left", fontsize=9)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_file, dpi=dpi)
        plt.close(fig)

        manifest_rows.append(
            {
                "dataset": DATASET_NAME,
                "phase": phase,
                "battery_id": battery_id,
                "plot_type": spec["plot_type"],
                "n_cycles_plotted": int(n_plotted),
                "min_cycle_number": min_cycle if np.isfinite(min_cycle) else "",
                "max_cycle_number": max_cycle if np.isfinite(max_cycle) else "",
                "output_file": str(output_file),
            }
        )

    return manifest_rows


def generate_grouped_plots(
    df: pd.DataFrame,
    phase: str,
    output_root: Path,
    plot_all_cycles: bool,
    plot_every: int,
    max_cycles_per_battery: int,
    dpi: int,
) -> pd.DataFrame:
    if df.empty or "battery_id" not in df.columns:
        return pd.DataFrame()

    selected = select_cycles_for_plotting(
        df=df,
        plot_all_cycles=plot_all_cycles,
        plot_every=plot_every,
        max_cycles_per_battery=max_cycles_per_battery,
    )
    if selected.empty:
        return pd.DataFrame()

    manifest_rows: list[dict[str, Any]] = []
    tqdm = get_tqdm()
    grouped = list(selected.groupby("battery_id", sort=True))
    for battery_id, group in tqdm(grouped, total=len(grouped), desc=f"grouped plots:{phase}", unit="battery"):
        manifest_rows.extend(
            plot_grouped_curves_for_battery(
                group=group,
                phase=phase,
                battery_id=str(battery_id),
                output_root=output_root,
                dpi=dpi,
            )
        )
    return pd.DataFrame(manifest_rows)


def main() -> None:
    args = parse_args()
    paths = resolve_base_paths(args)
    clean_dir = paths["clean_dir"]
    reports_dir = paths["reports_dir"]
    plots_root = reports_dir / "plots" / "curves_by_battery"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    phase_reports: dict[str, dict[str, Any]] = {}
    all_anomalies: list[dict[str, Any]] = []
    all_plot_manifest: list[pd.DataFrame] = []

    for phase in ("charge", "discharge"):
        clean_file = clean_dir / f"hust_{phase}_clean_cycles.parquet"
        df = read_parquet_if_exists(clean_file, STANDARD_CYCLE_COLUMNS)
        report = phase_summary(df, phase)
        report["input_file"] = str(clean_file)
        phase_reports[phase] = report

        if not df.empty:
            all_anomalies.extend(find_anomalies(df))
            all_anomalies.extend(vector_length_issues(df))

        if not args.skip_plots:
            manifest = generate_grouped_plots(
                df=df,
                phase=phase,
                output_root=plots_root,
                plot_all_cycles=bool(args.plot_all_cycles),
                plot_every=int(args.plot_every),
                max_cycles_per_battery=int(args.max_cycles_per_battery),
                dpi=int(args.dpi),
            )
            if not manifest.empty:
                all_plot_manifest.append(manifest)

    if all_plot_manifest:
        plot_manifest = pd.concat(all_plot_manifest, ignore_index=True)
    else:
        plot_manifest = pd.DataFrame(
            columns=[
                "dataset", "phase", "battery_id", "plot_type", "n_cycles_plotted",
                "min_cycle_number", "max_cycle_number", "output_file"
            ]
        )

    plot_manifest_file = plots_root / "plot_manifest.csv"
    plot_manifest.to_csv(plot_manifest_file, index=False, encoding="utf-8")

    audit_report = {
        "dataset": DATASET_NAME,
        "audit_version": AUDIT_VERSION,
        "clean_dir": str(clean_dir),
        "reports_dir": str(reports_dir),
        "phases": phase_reports,
        "grouped_plots": {
            "enabled": not bool(args.skip_plots),
            "root_dir": str(plots_root),
            "manifest_file": str(plot_manifest_file),
            "n_plot_files": int(len(plot_manifest)),
            "n_battery_phase_groups": int(
                plot_manifest[["phase", "battery_id"]].drop_duplicates().shape[0]
                if not plot_manifest.empty else 0
            ),
            "plot_all_cycles": bool(args.plot_all_cycles),
            "plot_every": int(args.plot_every),
            "max_cycles_per_battery": int(args.max_cycles_per_battery),
        },
        "created_at": utc_now_iso(),
    }

    dump_json(audit_report, reports_dir / "audit_report.json")
    make_summary_rows(phase_reports).to_csv(reports_dir / "audit_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(all_anomalies).to_csv(reports_dir / "anomaly_list.csv", index=False, encoding="utf-8")

    print("\n[OK] HUST audit completed.")
    for phase, report in phase_reports.items():
        print(
            f"[OK] {phase}: rows={report['rows']}, valid={report['valid_cycles']}, "
            f"usable={report['model_usable_cycles']}, anomalies={report['anomaly_count']}"
        )
    print(f"[OK] Report: {reports_dir / 'audit_report.json'}")
    print(f"[OK] Summary: {reports_dir / 'audit_summary.csv'}")
    print(f"[OK] Anomalies: {reports_dir / 'anomaly_list.csv'}")
    print(f"[OK] Plot manifest: {plot_manifest_file}")
    if not args.skip_plots:
        print(f"[OK] Grouped plot files: {len(plot_manifest)}")
        print(f"[OK] Plot root: {plots_root}")


if __name__ == "__main__":
    main()
