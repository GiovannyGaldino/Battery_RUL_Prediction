# -*- coding: utf-8 -*-
"""
Plot clean lithium-ion battery charge/discharge curves by battery.

This script reads standardized clean-cycle parquet files produced by the dataset
adapters and creates one image per battery and phase. Each image overlays all
valid cycles from the same battery, instead of creating one figure per cycle.

Default outputs
---------------
data/reports/<dataset>/plots/curves_by_battery/
├─ charge/
│  ├─ voltage_vs_capacity/
│  └─ capacity_vs_voltage/
└─ discharge/
   ├─ voltage_vs_capacity/
   └─ capacity_vs_voltage/

Examples
--------
# Plot all charge and discharge curves for one dataset:
python src/validation/plot_clean_cycle_curves_by_battery.py --dataset hust

# Plot all available datasets:
python src/validation/plot_clean_cycle_curves_by_battery.py --dataset all

# Plot only discharge curves from RWTH using all cycles:
python src/validation/plot_clean_cycle_curves_by_battery.py --dataset rwth --phase discharge

# Plot only model-usable cycles:
python src/validation/plot_clean_cycle_curves_by_battery.py --dataset rwth --only-model-usable

Notes
-----
- The script assumes the official clean-cycle schema with vector columns
  voltage_v and capacity_ah.
- By default, all valid cycles from each battery are plotted. Use
  --max-cycles-per-battery N only if the figures become unreadable or too slow.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DEFAULT_BASE_DIR = Path.cwd()

PHASES = ("charge", "discharge")
PLOT_KINDS = ("voltage_vs_capacity", "capacity_vs_voltage")


@dataclass(frozen=True)
class CleanFile:
    dataset_dir_name: str
    dataset_label: str
    phase: str
    path: Path


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(text: Any) -> str:
    safe = str(text).strip()
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "unknown"


def safe_array(value: Any) -> np.ndarray:
    """Convert list-like parquet values or stringified vectors to a float array."""
    if value is None:
        return np.array([], dtype=float)

    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return np.array([], dtype=float)
        try:
            arr = np.asarray(ast.literal_eval(text))
        except Exception:
            text = text.replace(";", ",").replace("[", "").replace("]", "")
            values: list[float] = []
            for part in text.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    values.append(float(part))
                except ValueError:
                    continue
            arr = np.asarray(values)
    else:
        try:
            arr = np.asarray(value)
        except Exception:
            return np.array([], dtype=float)

    try:
        out = np.atleast_1d(np.asarray(arr, dtype=float).squeeze())
    except Exception:
        return np.array([], dtype=float)

    if out.ndim != 1:
        out = out.reshape(-1)
    return out


def choose_alpha(n_curves: int, user_alpha: float | None) -> float:
    if user_alpha is not None:
        return float(user_alpha)
    if n_curves <= 20:
        return 0.55
    if n_curves <= 100:
        return 0.30
    if n_curves <= 500:
        return 0.12
    if n_curves <= 1500:
        return 0.06
    return 0.035


def choose_linewidth(n_curves: int, user_linewidth: float | None) -> float:
    if user_linewidth is not None:
        return float(user_linewidth)
    if n_curves <= 100:
        return 0.8
    if n_curves <= 1000:
        return 0.55
    return 0.35


def downsample_vector(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points).round().astype(int)
    idx = np.unique(idx)
    return x[idx], y[idx]


def read_clean_dataframe(clean_file: CleanFile) -> pd.DataFrame:
    df = pd.read_parquet(clean_file.path)
    if "phase" not in df.columns:
        df["phase"] = clean_file.phase
    if "dataset" not in df.columns:
        df["dataset"] = clean_file.dataset_label.upper()
    if "battery_id" not in df.columns:
        raise ValueError(f"Missing required column 'battery_id' in {clean_file.path}")
    if "voltage_v" not in df.columns or "capacity_ah" not in df.columns:
        raise ValueError(f"Missing required vector columns voltage_v/capacity_ah in {clean_file.path}")
    return df


def discover_clean_files(base_dir: Path, dataset: str, phases: Iterable[str]) -> list[CleanFile]:
    interim_dir = base_dir / "data" / "interim"
    selected_phases = set(phases)
    clean_files: list[CleanFile] = []

    if dataset.lower() == "all":
        candidate_dirs = sorted(path for path in interim_dir.glob("*/clean") if path.is_dir())
    else:
        candidate_dirs = [interim_dir / dataset.lower() / "clean"]
        if not candidate_dirs[0].exists():
            candidate_dirs = [interim_dir / dataset / "clean"]

    for clean_dir in candidate_dirs:
        if not clean_dir.exists():
            continue
        dataset_dir_name = clean_dir.parent.name
        for phase in PHASES:
            if phase not in selected_phases:
                continue
            matches = sorted(clean_dir.glob(f"*_{phase}_clean_cycles.parquet"))
            for path in matches:
                prefix = path.name.replace(f"_{phase}_clean_cycles.parquet", "")
                clean_files.append(
                    CleanFile(
                        dataset_dir_name=dataset_dir_name,
                        dataset_label=prefix.upper(),
                        phase=phase,
                        path=path,
                    )
                )
    return clean_files


def filter_cycles(df: pd.DataFrame, only_valid: bool, only_model_usable: bool) -> pd.DataFrame:
    out = df.copy()
    if only_model_usable and "is_model_usable" in out.columns:
        out = out[out["is_model_usable"].fillna(False).astype(bool)]
    elif only_valid and "is_valid_cycle" in out.columns:
        out = out[out["is_valid_cycle"].fillna(False).astype(bool)]
    return out


def sort_cycles(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in ["cycle_number", "cycle_index", "source_cycle_id", "source_file"] if col in df.columns]
    if not sort_cols:
        return df
    return df.sort_values(sort_cols, kind="mergesort")


def sample_cycles(df: pd.DataFrame, max_cycles: int) -> tuple[pd.DataFrame, bool]:
    if max_cycles <= 0 or len(df) <= max_cycles:
        return df, False
    idx = np.linspace(0, len(df) - 1, max_cycles).round().astype(int)
    idx = np.unique(idx)
    return df.iloc[idx].copy(), True


def plot_one_battery(
    group: pd.DataFrame,
    dataset_label: str,
    phase: str,
    battery_id: str,
    plot_kind: str,
    output_file: Path,
    max_points_per_curve: int,
    line_alpha: float | None,
    line_width: float | None,
    dpi: int,
    figsize: tuple[float, float],
    title_prefix: str,
) -> dict[str, Any]:
    n_rows = len(group)
    alpha = choose_alpha(n_rows, line_alpha)
    linewidth = choose_linewidth(n_rows, line_width)

    fig, ax = plt.subplots(figsize=figsize)
    plotted = 0

    for _, row in group.iterrows():
        voltage = safe_array(row.get("voltage_v"))
        capacity = safe_array(row.get("capacity_ah"))
        n = min(len(voltage), len(capacity))
        if n < 2:
            continue

        voltage = voltage[:n]
        capacity = capacity[:n]
        finite = np.isfinite(voltage) & np.isfinite(capacity)
        if finite.sum() < 2:
            continue
        voltage = voltage[finite]
        capacity = capacity[finite]

        if plot_kind == "voltage_vs_capacity":
            x, y = capacity, voltage
        elif plot_kind == "capacity_vs_voltage":
            x, y = voltage, capacity
        else:
            raise ValueError(f"Unsupported plot kind: {plot_kind}")

        x, y = downsample_vector(x, y, max_points=max_points_per_curve)
        ax.plot(x, y, linewidth=linewidth, alpha=alpha)
        plotted += 1

    if plot_kind == "voltage_vs_capacity":
        ax.set_xlabel("Capacity, Q (Ah)")
        ax.set_ylabel("Voltage, V (V)")
        subtitle = "V(Q): voltage versus capacity"
    else:
        ax.set_xlabel("Voltage, V (V)")
        ax.set_ylabel("Capacity, Q (Ah)")
        subtitle = "Q(V): capacity versus voltage"

    ax.set_title(f"{title_prefix}{dataset_label} | {phase} | {battery_id}\n{subtitle} | {plotted} curves")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    ensure_dir(output_file.parent)
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)

    return {
        "dataset": dataset_label,
        "phase": phase,
        "battery_id": battery_id,
        "plot_kind": plot_kind,
        "n_available_cycles": int(n_rows),
        "n_plotted_curves": int(plotted),
        "output_file": str(output_file),
    }


def plot_clean_file(clean_file: CleanFile, args: argparse.Namespace) -> list[dict[str, Any]]:
    df = read_clean_dataframe(clean_file)
    phase = clean_file.phase
    df = df[df["phase"].astype(str).str.lower().eq(phase)].copy()
    df = filter_cycles(df, only_valid=args.only_valid, only_model_usable=args.only_model_usable)

    if df.empty:
        return []

    report_dataset_dir = args.report_dir or args.base_dir / "data" / "reports" / clean_file.dataset_dir_name
    base_plot_dir = report_dataset_dir / "plots" / "curves_by_battery" / phase
    manifest_rows: list[dict[str, Any]] = []

    grouped = list(df.groupby("battery_id", sort=True, dropna=False))
    for battery_id, group in tqdm(grouped, desc=f"Plotting {clean_file.dataset_label} {phase}", unit="battery"):
        battery_id_str = str(battery_id)
        group = sort_cycles(group)
        group, sampled = sample_cycles(group, args.max_cycles_per_battery)
        safe_battery = sanitize_filename(battery_id_str)

        for plot_kind in args.plot_kind:
            output_file = (
                base_plot_dir
                / plot_kind
                / f"{sanitize_filename(clean_file.dataset_label.lower())}_{phase}_{safe_battery}_{plot_kind}_all_cycles.png"
            )
            if args.skip_existing and output_file.exists():
                manifest_rows.append({
                    "dataset": clean_file.dataset_label,
                    "phase": phase,
                    "battery_id": battery_id_str,
                    "plot_kind": plot_kind,
                    "n_available_cycles": int(len(group)),
                    "n_plotted_curves": None,
                    "output_file": str(output_file),
                    "status": "skipped_existing",
                    "sampled": bool(sampled),
                })
                continue

            row = plot_one_battery(
                group=group,
                dataset_label=clean_file.dataset_label,
                phase=phase,
                battery_id=battery_id_str,
                plot_kind=plot_kind,
                output_file=output_file,
                max_points_per_curve=args.max_points_per_curve,
                line_alpha=args.line_alpha,
                line_width=args.line_width,
                dpi=args.dpi,
                figsize=(args.fig_width, args.fig_height),
                title_prefix=args.title_prefix,
            )
            row["status"] = "created"
            row["sampled"] = bool(sampled)
            manifest_rows.append(row)

    return manifest_rows


def parse_plot_kind(value: str) -> list[str]:
    value = value.lower().strip()
    if value == "both":
        return list(PLOT_KINDS)
    if value not in PLOT_KINDS:
        raise argparse.ArgumentTypeError(f"plot kind must be one of: both, {', '.join(PLOT_KINDS)}")
    return [value]


def parse_phase(value: str) -> list[str]:
    value = value.lower().strip()
    if value == "both":
        return list(PHASES)
    if value not in PHASES:
        raise argparse.ArgumentTypeError("phase must be one of: both, charge, discharge")
    return [value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all clean charge/discharge curves by battery, one image per battery and phase."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--dataset", type=str, default="rwth", help="Dataset folder name under data/interim, or 'all'.")
    parser.add_argument("--phase", type=parse_phase, default=list(PHASES), help="both, charge, or discharge.")
    parser.add_argument("--plot-kind", type=parse_plot_kind, default=list(PLOT_KINDS), help="both, voltage_vs_capacity, or capacity_vs_voltage.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Optional custom report directory for a single dataset.")
    parser.add_argument("--only-valid", action="store_true", default=True, help="Plot only is_valid_cycle=True rows when available. Default: true.")
    parser.add_argument("--include-invalid", dest="only_valid", action="store_false", help="Also plot invalid cycles.")
    parser.add_argument("--only-model-usable", action="store_true", help="Plot only is_model_usable=True rows when available.")
    parser.add_argument("--max-cycles-per-battery", type=int, default=0, help="0 means plot all cycles. Positive values sample cycles uniformly.")
    parser.add_argument("--max-points-per-curve", type=int, default=0, help="0 means plot all points from each curve.")
    parser.add_argument("--line-alpha", type=float, default=None, help="Optional fixed line transparency. If omitted, alpha is automatic.")
    parser.add_argument("--line-width", type=float, default=None, help="Optional fixed line width. If omitted, width is automatic.")
    parser.add_argument("--fig-width", type=float, default=9.5)
    parser.add_argument("--fig-height", type=float, default=6.5)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--title-prefix", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phases = args.phase
    clean_files = discover_clean_files(args.base_dir, dataset=args.dataset, phases=phases)

    if not clean_files:
        raise FileNotFoundError(
            f"No clean parquet files found for dataset={args.dataset!r} and phases={phases}. "
            f"Expected files under {args.base_dir / 'data' / 'interim'}/*/clean."
        )

    all_manifest_rows: list[dict[str, Any]] = []
    for clean_file in clean_files:
        rows = plot_clean_file(clean_file, args)
        all_manifest_rows.extend(rows)

    if all_manifest_rows:
        manifest = pd.DataFrame(all_manifest_rows)
        if args.dataset.lower() == "all":
            out_dir = ensure_dir(args.base_dir / "data" / "reports" / "global" / "curves_by_battery")
        else:
            dataset_dir = clean_files[0].dataset_dir_name
            out_dir = ensure_dir(args.report_dir or args.base_dir / "data" / "reports" / dataset_dir / "plots" / "curves_by_battery")
        manifest_csv = out_dir / "plot_manifest.csv"
        manifest_json = out_dir / "plot_summary.json"
        manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
        summary = {
            "created_at": now_iso(),
            "dataset_argument": args.dataset,
            "n_clean_files": len(clean_files),
            "n_plot_records": int(len(manifest)),
            "n_images_created": int((manifest["status"] == "created").sum()) if "status" in manifest else int(len(manifest)),
            "n_batteries": int(manifest["battery_id"].nunique()) if "battery_id" in manifest else 0,
            "phases": sorted(manifest["phase"].dropna().unique().tolist()) if "phase" in manifest else [],
            "plot_kinds": sorted(manifest["plot_kind"].dropna().unique().tolist()) if "plot_kind" in manifest else [],
            "manifest_csv": str(manifest_csv),
        }
        manifest_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\nCurve plotting completed.")
        print(f"Images listed in: {manifest_csv}")
        print(f"Summary: {manifest_json}")
        print(f"Images created: {summary['n_images_created']}")
    else:
        print("No curves were plotted after filtering.")


if __name__ == "__main__":
    main()
