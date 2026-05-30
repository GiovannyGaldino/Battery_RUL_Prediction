r"""
MATR/Severson dataset adapter - audit step (version 1.1).

This script audits clean MATR charge/discharge cycle files and creates JSON/CSV
reports plus visual inspection plots for Q(V) and V(Q) curves.

Inputs:
    data/interim/matr/clean/matr_charge_clean_cycles.parquet
    data/interim/matr/clean/matr_discharge_clean_cycles.parquet

Outputs:
    data/reports/matr/audit_report.json
    data/reports/matr/audit_summary.csv
    data/reports/matr/anomaly_list.csv
    data/reports/matr/audit_anomaly_list.csv
    data/reports/matr/cleaning_anomaly_list.csv
    data/reports/matr/article_anomaly_summary.csv
    data/reports/matr/plots/<phase>/voltage_vs_capacity/*.png
    data/reports/matr/plots/<phase>/capacity_vs_voltage/*.png
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
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable | None = None, **kwargs):
        return iterable if iterable is not None else (lambda x: x)


DATASET = "MATR"
AUDIT_VERSION = "matr_adapter_v1_1_audit_2026-05-17"

DEFAULT_BASE_DIR = Path(
    r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries"
)

STANDARD_CYCLE_COLUMNS = [
    "dataset", "source_file", "source_path", "battery_id", "battery_index",
    "cell_id_original", "cycle_index", "cycle_number", "source_cycle_id",
    "cycle_pair_id", "phase", "cycle_life", "remaining_cycles",
    "remaining_cycles_norm", "eol_criterion", "eol_capacity_ah",
    "initial_capacity_ah", "nominal_capacity_ah", "soh", "time_s",
    "voltage_v", "current_a", "temperature_c", "capacity_ah", "energy_wh",
    "power_w", "n_points", "duration_s", "sample_period_median_s",
    "voltage_min_v", "voltage_max_v", "voltage_start_v", "voltage_end_v",
    "current_min_a", "current_max_a", "current_mean_a", "temperature_min_c",
    "temperature_max_c", "temperature_mean_c", "phase_capacity_ah",
    "phase_energy_wh", "voltage_source", "current_source", "temperature_source",
    "capacity_source", "energy_source", "chemistry", "form_factor", "manufacturer",
    "nominal_voltage_v", "ambient_temperature_c", "charge_protocol",
    "discharge_protocol", "c_rate_charge", "c_rate_discharge", "dod",
    "policy_readable", "barcode", "is_valid_cycle", "is_model_usable",
    "exclude_reason", "cleaning_warnings", "n_removed_points", "trim_start_index",
    "trim_end_index", "has_voltage", "has_current", "has_temperature",
    "has_capacity", "has_energy", "adapter_version", "cleaning_version",
    "created_at",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_base_dir(base_dir_arg: str | None) -> Path:
    if base_dir_arg:
        return Path(base_dir_arg).expanduser().resolve()
    if DEFAULT_BASE_DIR.exists():
        return DEFAULT_BASE_DIR
    return Path.cwd().resolve()


def safe_vector(x: Any) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float).squeeze()
        return np.atleast_1d(arr)
    except Exception:
        return np.array([], dtype=float)


def finite_values(x: Any) -> np.ndarray:
    arr = safe_vector(x)
    return arr[np.isfinite(arr)]


def describe_numeric(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def sanitize_filename(name: str) -> str:
    invalid = '<>:"/\\|?*'
    out = str(name)
    for ch in invalid:
        out = out.replace(ch, "_")
    return out


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(json_safe(v) for v in value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, np.ndarray)) else False:
        return None
    return value


def validate_schema(df: pd.DataFrame, phase: str) -> dict[str, Any]:
    missing = [col for col in STANDARD_CYCLE_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in STANDARD_CYCLE_COLUMNS]
    order_ok = list(df.columns) == STANDARD_CYCLE_COLUMNS if not missing and not extra else False
    return {
        "phase": phase,
        "schema_ok": len(missing) == 0 and len(extra) == 0,
        "column_order_ok": bool(order_ok),
        "missing_columns": missing,
        "extra_columns": extra,
    }


def vector_lengths(row: pd.Series) -> dict[str, int]:
    return {
        col: len(safe_vector(row[col]))
        for col in ["time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah", "energy_wh", "power_w"]
        if col in row
    }


def is_monotonic_non_decreasing(x: Any) -> bool:
    values = finite_values(x)
    if values.size < 2:
        return False
    return bool(np.all(np.diff(values) >= -1e-10))


def is_time_increasing(x: Any) -> bool:
    values = finite_values(x)
    if values.size < 2:
        return False
    return bool(np.all(np.diff(values) > 0))


def build_anomaly_list(df: pd.DataFrame, phase: str, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    iterator = tqdm(df.iterrows(), total=len(df), desc=f"Auditing anomalies {phase}", unit="cycle")
    for _, row in iterator:
        anomalies: list[str] = []
        lengths = vector_lengths(row)
        if len(set(lengths.values())) != 1:
            anomalies.append("vector_length_mismatch")
        if int(row.get("n_points", 0)) < args.min_points_warning:
            anomalies.append("few_points")
        if not is_time_increasing(row.get("time_s", [])):
            anomalies.append("time_not_strictly_increasing")
        if not is_monotonic_non_decreasing(row.get("capacity_ah", [])):
            anomalies.append("capacity_not_monotonic")
        if pd.to_numeric(row.get("phase_capacity_ah", np.nan), errors="coerce") <= 0:
            anomalies.append("non_positive_phase_capacity")
        soh = pd.to_numeric(row.get("soh", np.nan), errors="coerce")
        if not np.isfinite(soh):
            anomalies.append("soh_not_finite")
        elif soh <= args.min_soh_model_usable or soh > args.max_soh_model_usable:
            anomalies.append("abnormal_soh_model_exclusion")
        elif soh < args.min_soh_warning or soh > args.max_soh_warning:
            anomalies.append("abnormal_soh_warning")
        voltage_span = float(row.get("voltage_max_v", np.nan)) - float(row.get("voltage_min_v", np.nan))
        if not np.isfinite(voltage_span) or voltage_span < args.min_voltage_span_warning:
            anomalies.append("insufficient_voltage_span")
        if phase == "charge" and float(row.get("current_mean_a", np.nan)) <= 0:
            anomalies.append("charge_current_not_positive")
        if phase == "discharge" and float(row.get("current_mean_a", np.nan)) >= 0:
            anomalies.append("discharge_current_not_negative")
        if float(row.get("voltage_start_v", np.nan)) < args.suspicious_voltage_low:
            anomalies.append("suspicious_low_start_voltage")
        if float(row.get("voltage_end_v", np.nan)) < args.suspicious_voltage_low:
            anomalies.append("suspicious_low_end_voltage")
        if float(row.get("voltage_start_v", np.nan)) > args.suspicious_voltage_high:
            anomalies.append("suspicious_high_start_voltage")
        if float(row.get("voltage_end_v", np.nan)) > args.suspicious_voltage_high:
            anomalies.append("suspicious_high_end_voltage")
        if not bool(row.get("is_model_usable", False)):
            reason = str(row.get("exclude_reason", "not_model_usable"))
            if reason:
                anomalies.append(reason)

        if anomalies:
            rows.append(
                {
                    "dataset": DATASET,
                    "phase": phase,
                    "battery_id": row.get("battery_id", ""),
                    "cycle_index": row.get("cycle_index", np.nan),
                    "cycle_number": row.get("cycle_number", np.nan),
                    "source_cycle_id": row.get("source_cycle_id", ""),
                    "anomalies": ";".join(sorted(set(anomalies))),
                    "n_points": row.get("n_points", np.nan),
                    "voltage_start_v": row.get("voltage_start_v", np.nan),
                    "voltage_end_v": row.get("voltage_end_v", np.nan),
                    "voltage_min_v": row.get("voltage_min_v", np.nan),
                    "voltage_max_v": row.get("voltage_max_v", np.nan),
                    "phase_capacity_ah": row.get("phase_capacity_ah", np.nan),
                    "initial_capacity_ah": row.get("initial_capacity_ah", np.nan),
                    "soh": row.get("soh", np.nan),
                    "remaining_cycles": row.get("remaining_cycles", np.nan),
                    "exclude_reason": row.get("exclude_reason", ""),
                }
            )
    return pd.DataFrame(rows)


def phase_summary(df: pd.DataFrame, phase: str, schema_info: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "phase": phase,
        "schema": schema_info,
        "rows": int(len(df)),
        "unique_batteries": int(df["battery_id"].nunique()) if not df.empty else 0,
        "valid_cycles": int(df["is_valid_cycle"].sum()) if not df.empty else 0,
        "model_usable_cycles": int(df["is_model_usable"].sum()) if not df.empty else 0,
        "exclude_reason_counts": {},
        "signal_sources": {},
        "availability": {},
        "voltage_range": {},
        "capacity_range": {},
        "temperature_range": {},
        "target_distribution": {},
    }
    if df.empty:
        return summary

    for col in ["exclude_reason", "voltage_source", "current_source", "temperature_source", "capacity_source", "energy_source"]:
        if col in df.columns:
            counts = df[col].fillna("<NA>").astype(str).value_counts(dropna=False).to_dict()
            if col == "exclude_reason":
                summary["exclude_reason_counts"] = {str(k): int(v) for k, v in counts.items()}
            else:
                summary["signal_sources"][col] = {str(k): int(v) for k, v in counts.items()}

    for col in ["has_voltage", "has_current", "has_temperature", "has_capacity", "has_energy"]:
        summary["availability"][col] = {str(k): int(v) for k, v in df[col].value_counts(dropna=False).to_dict().items()}

    summary["voltage_range"] = {
        "voltage_min_v": describe_numeric(df["voltage_min_v"]),
        "voltage_max_v": describe_numeric(df["voltage_max_v"]),
        "voltage_start_v": describe_numeric(df["voltage_start_v"]),
        "voltage_end_v": describe_numeric(df["voltage_end_v"]),
    }
    summary["capacity_range"] = {
        "phase_capacity_ah": describe_numeric(df["phase_capacity_ah"]),
        "initial_capacity_ah": describe_numeric(df["initial_capacity_ah"]),
        "soh": describe_numeric(df["soh"]),
    }
    summary["temperature_range"] = {
        "temperature_min_c": describe_numeric(df["temperature_min_c"]),
        "temperature_max_c": describe_numeric(df["temperature_max_c"]),
        "temperature_mean_c": describe_numeric(df["temperature_mean_c"]),
    }
    summary["target_distribution"] = {
        "cycle_life": describe_numeric(df["cycle_life"]),
        "remaining_cycles": describe_numeric(df["remaining_cycles"]),
        "remaining_cycles_norm": describe_numeric(df["remaining_cycles_norm"]),
    }
    return summary


def build_summary_csv_rows(df: pd.DataFrame, phase: str) -> list[dict[str, Any]]:
    if df.empty:
        return [
            {
                "dataset": DATASET,
                "phase": phase,
                "battery_id": "<empty>",
                "n_cycles": 0,
                "n_model_usable": 0,
                "cycle_life": np.nan,
                "capacity_min_ah": np.nan,
                "capacity_max_ah": np.nan,
                "voltage_min_v": np.nan,
                "voltage_max_v": np.nan,
                "temperature_mean_c": np.nan,
            }
        ]

    rows = []
    for battery_id, group in df.groupby("battery_id"):
        rows.append(
            {
                "dataset": DATASET,
                "phase": phase,
                "battery_id": battery_id,
                "n_cycles": int(len(group)),
                "n_model_usable": int(group["is_model_usable"].sum()),
                "cycle_life": float(pd.to_numeric(group["cycle_life"], errors="coerce").dropna().max()) if pd.to_numeric(group["cycle_life"], errors="coerce").notna().any() else np.nan,
                "capacity_min_ah": float(group["phase_capacity_ah"].min()),
                "capacity_max_ah": float(group["phase_capacity_ah"].max()),
                "soh_min": float(pd.to_numeric(group["soh"], errors="coerce").min()),
                "soh_max": float(pd.to_numeric(group["soh"], errors="coerce").max()),
                "n_abnormal_soh": int(((pd.to_numeric(group["soh"], errors="coerce") <= 0.05) | (pd.to_numeric(group["soh"], errors="coerce") > 1.20)).sum()),
                "voltage_min_v": float(group["voltage_min_v"].min()),
                "voltage_max_v": float(group["voltage_max_v"].max()),
                "temperature_mean_c": float(group["temperature_mean_c"].mean()) if group["temperature_mean_c"].notna().any() else np.nan,
            }
        )
    return rows


def plot_battery_curves(df: pd.DataFrame, phase: str, report_dir: Path, args: argparse.Namespace) -> None:
    if df.empty or args.no_plots:
        return

    base_plot_dir = report_dir / "plots" / phase
    v_vs_q_dir = base_plot_dir / "voltage_vs_capacity"
    q_vs_v_dir = base_plot_dir / "capacity_vs_voltage"
    v_vs_q_dir.mkdir(parents=True, exist_ok=True)
    q_vs_v_dir.mkdir(parents=True, exist_ok=True)

    battery_items = list(df.groupby("battery_id"))
    iterator = tqdm(battery_items, desc=f"Plotting {phase}", unit="battery")

    for battery_id, group in iterator:
        group = group.sort_values("cycle_index")
        if args.plot_only_model_usable:
            group = group[group["is_model_usable"]]
        if group.empty:
            continue
        if args.max_cycles_per_battery is not None and len(group) > args.max_cycles_per_battery:
            group = group.iloc[: args.max_cycles_per_battery]
        if args.plot_every_n_cycles > 1:
            group = group.iloc[:: args.plot_every_n_cycles]

        safe_name = sanitize_filename(battery_id)

        # V versus Q(V): x = capacity, y = voltage.
        fig, ax = plt.subplots(figsize=(8, 5))
        for _, row in group.iterrows():
            q = safe_vector(row["capacity_ah"])
            v = safe_vector(row["voltage_v"])
            n = min(len(q), len(v))
            if n >= 2:
                ax.plot(q[:n], v[:n], linewidth=0.7, alpha=0.55)
        ax.set_title(f"{phase.capitalize()} V vs Q - {battery_id}")
        ax.set_xlabel("Capacity, Q (Ah)")
        ax.set_ylabel("Voltage, V (V)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(v_vs_q_dir / f"{safe_name}_{phase}_voltage_vs_capacity.png", dpi=args.plot_dpi)
        plt.close(fig)

        # Q(V): x = voltage, y = capacity.
        fig, ax = plt.subplots(figsize=(8, 5))
        for _, row in group.iterrows():
            q = safe_vector(row["capacity_ah"])
            v = safe_vector(row["voltage_v"])
            n = min(len(q), len(v))
            if n >= 2:
                ax.plot(v[:n], q[:n], linewidth=0.7, alpha=0.55)
        ax.set_title(f"{phase.capitalize()} Q(V) - {battery_id}")
        ax.set_xlabel("Voltage, V (V)")
        ax.set_ylabel("Capacity, Q (Ah)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(q_vs_v_dir / f"{safe_name}_{phase}_capacity_vs_voltage.png", dpi=args.plot_dpi)
        plt.close(fig)


def read_phase_file(clean_dir: Path, phase: str) -> pd.DataFrame:
    file_path = clean_dir / f"matr_{phase}_clean_cycles.parquet"
    if not file_path.exists():
        return pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)
    return pd.read_parquet(file_path)



def load_cleaning_anomalies(clean_dir: Path) -> pd.DataFrame:
    """Load detailed anomaly records produced by clean_matr_cycles.py v1.1."""
    frames: list[pd.DataFrame] = []
    for phase in ["charge", "discharge"]:
        file_path = clean_dir / f"matr_{phase}_cycle_anomalies.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if "phase" not in df.columns:
                df["phase"] = phase
            df["anomaly_origin"] = "cleaning"
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def audit_anomalies_to_long(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the compact audit anomaly list into long-form records."""
    rows: list[dict[str, Any]] = []
    if audit_df.empty or "anomalies" not in audit_df.columns:
        return pd.DataFrame()
    for _, row in audit_df.iterrows():
        anomaly_text = str(row.get("anomalies", ""))
        if not anomaly_text:
            continue
        for anomaly_type in anomaly_text.split(";"):
            anomaly_type = anomaly_type.strip()
            if not anomaly_type:
                continue
            rows.append(
                {
                    "dataset": DATASET,
                    "phase": row.get("phase", ""),
                    "battery_id": row.get("battery_id", ""),
                    "cycle_index": row.get("cycle_index", np.nan),
                    "cycle_number": row.get("cycle_number", np.nan),
                    "source_cycle_id": row.get("source_cycle_id", ""),
                    "anomaly_type": anomaly_type,
                    "anomaly_group": "audit",
                    "severity": "major" if str(row.get("exclude_reason", "")) else "warning",
                    "metric_name": "audit_flag",
                    "metric_value": anomaly_type,
                    "affects_modeling": bool(str(row.get("exclude_reason", ""))),
                    "exclude_reason": row.get("exclude_reason", ""),
                    "anomaly_origin": "audit",
                }
            )
    return pd.DataFrame(rows)


def build_article_anomaly_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact anomaly table suitable for the article/results section."""
    if long_df.empty or "anomaly_type" not in long_df.columns:
        return pd.DataFrame(columns=[
            "dataset", "phase", "anomaly_origin", "anomaly_group", "anomaly_type",
            "severity", "records", "affected_batteries", "affected_cycles",
            "model_affecting_records",
        ])

    for col in ["phase", "anomaly_origin", "anomaly_group", "anomaly_type", "severity"]:
        if col not in long_df.columns:
            long_df[col] = "unknown"
    if "affects_modeling" not in long_df.columns:
        long_df["affects_modeling"] = False

    group_cols = ["phase", "anomaly_origin", "anomaly_group", "anomaly_type", "severity"]
    rows = []
    for keys, group in long_df.groupby(group_cols, dropna=False):
        phase, origin, group_name, anomaly_type, severity = keys
        cycle_cols = [c for c in ["phase", "battery_id", "cycle_number", "source_cycle_id"] if c in group.columns]
        rows.append(
            {
                "dataset": DATASET,
                "phase": phase,
                "anomaly_origin": origin,
                "anomaly_group": group_name,
                "anomaly_type": anomaly_type,
                "severity": severity,
                "records": int(len(group)),
                "affected_batteries": int(group["battery_id"].nunique()) if "battery_id" in group.columns else 0,
                "affected_cycles": int(group[cycle_cols].drop_duplicates().shape[0]) if cycle_cols else int(len(group)),
                "model_affecting_records": int(group["affects_modeling"].astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["phase", "anomaly_origin", "records"], ascending=[True, True, False])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit clean MATR charge/discharge cycles.")
    parser.add_argument("--base-dir", type=str, default=None, help="Project root directory. Defaults to the known Batteries path or current directory.")
    parser.add_argument("--clean-dir", type=str, default=None, help="Clean cycle directory. Defaults to <base-dir>/data/interim/matr/clean.")
    parser.add_argument("--report-dir", type=str, default=None, help="Audit report directory. Defaults to <base-dir>/data/reports/matr.")
    parser.add_argument("--no-plots", action="store_true", help="Disable visual inspection plots.")
    parser.add_argument("--plot-every-n-cycles", type=int, default=10, help="Plot one out of every N cycles per battery. Use 1 to plot all.")
    parser.add_argument("--max-cycles-per-battery", type=int, default=None, help="Optional cap on plotted cycles per battery.")
    parser.add_argument("--plot-only-model-usable", action="store_true", help="Plot only model-usable cycles.")
    parser.add_argument("--plot-dpi", type=int, default=180)
    parser.add_argument("--min-points-warning", type=int, default=50)
    parser.add_argument("--min-voltage-span-warning", type=float, default=0.20)
    parser.add_argument("--suspicious-voltage-low", type=float, default=2.00)
    parser.add_argument("--suspicious-voltage-high", type=float, default=3.70)
    parser.add_argument("--min-soh-model-usable", type=float, default=0.05)
    parser.add_argument("--max-soh-model-usable", type=float, default=1.20)
    parser.add_argument("--min-soh-warning", type=float, default=0.50)
    parser.add_argument("--max-soh-warning", type=float, default=1.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    clean_dir = Path(args.clean_dir).expanduser().resolve() if args.clean_dir else base_dir / "data" / "interim" / "matr" / "clean"
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else base_dir / "data" / "reports" / "matr"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("MATR audit - version 1.1")
    print("=" * 88)
    print(f"Base directory : {base_dir}")
    print(f"Clean directory: {clean_dir}")
    print(f"Report directory: {report_dir}")

    audit_report: dict[str, Any] = {
        "dataset": DATASET,
        "audit_version": AUDIT_VERSION,
        "created_at": utc_now_iso(),
        "phases": {},
    }
    summary_rows: list[dict[str, Any]] = []
    anomaly_frames: list[pd.DataFrame] = []

    for phase in tqdm(["charge", "discharge"], desc="Audit phases", unit="phase"):
        df = read_phase_file(clean_dir, phase)
        schema_info = validate_schema(df, phase)
        audit_report["phases"][phase] = phase_summary(df, phase, schema_info)
        summary_rows.extend(build_summary_csv_rows(df, phase))
        anomaly_frames.append(build_anomaly_list(df, phase, args))
        plot_battery_curves(df, phase, report_dir, args)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = report_dir / "audit_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    anomaly_df = pd.concat(anomaly_frames, ignore_index=True) if anomaly_frames else pd.DataFrame()
    audit_anomaly_csv = report_dir / "audit_anomaly_list.csv"
    anomaly_df.to_csv(audit_anomaly_csv, index=False, encoding="utf-8-sig")

    cleaning_anomalies = load_cleaning_anomalies(clean_dir)
    cleaning_anomaly_csv = report_dir / "cleaning_anomaly_list.csv"
    cleaning_anomalies.to_csv(cleaning_anomaly_csv, index=False, encoding="utf-8-sig")

    audit_long = audit_anomalies_to_long(anomaly_df)
    combined_long = pd.concat([cleaning_anomalies, audit_long], ignore_index=True, sort=False)
    anomaly_csv = report_dir / "anomaly_list.csv"
    combined_long.to_csv(anomaly_csv, index=False, encoding="utf-8-sig")

    article_summary = build_article_anomaly_summary(combined_long)
    article_summary_csv = report_dir / "article_anomaly_summary.csv"
    article_summary.to_csv(article_summary_csv, index=False, encoding="utf-8-sig")

    audit_report["article_anomaly_summary"] = {
        "records": int(len(combined_long)),
        "summary_rows": int(len(article_summary)),
        "files": {
            "combined_anomaly_list": str(anomaly_csv),
            "audit_anomaly_list": str(audit_anomaly_csv),
            "cleaning_anomaly_list": str(cleaning_anomaly_csv),
            "article_anomaly_summary": str(article_summary_csv),
        },
    }

    report_json = report_dir / "audit_report.json"
    with report_json.open("w", encoding="utf-8") as fp:
        json.dump(json_safe(audit_report), fp, indent=2, ensure_ascii=False)

    print(f"Audit report           : {report_json}")
    print(f"Audit summary          : {summary_csv}")
    print(f"Combined anomaly list  : {anomaly_csv}")
    print(f"Audit anomaly list     : {audit_anomaly_csv}")
    print(f"Cleaning anomaly list  : {cleaning_anomaly_csv}")
    print(f"Article anomaly summary: {article_summary_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
