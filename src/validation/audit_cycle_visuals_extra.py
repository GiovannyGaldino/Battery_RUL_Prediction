#!/usr/bin/env python3
"""
Extra audit script for visual inspection of cleaned battery cycles.

For each battery and phase:
- Select up to N equally spaced cycles (default: 5)
- For each selected cycle, generate one figure with 6 aligned plots vs time:
  voltage, current, capacity, energy, power, temperature
- Generate a single V(Q) figure overlaying the selected cycles

This script is complementary to the main audit pipeline and does not replace it.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


DATASET_ALIASES = {
    "ORA": "OXFORD",
    "OXFORD": "OXFORD",
    "RW": "NRW",
    "NASA_RANDOM_WALK": "NRW",
    "RANDOM_WALK": "NRW",
}

DEFAULT_INTERIM_ROOT = "data/interim"
DEFAULT_OUTPUT_ROOT = "reports/audit_extra"


def canonicalize_dataset_name(name: object) -> str:
    if pd.isna(name):
        return "UNKNOWN"
    text = str(name).strip()
    if not text:
        return "UNKNOWN"
    return DATASET_ALIASES.get(text.upper(), text.upper())


def safe_slug(text: object) -> str:
    s = str(text)
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "unknown"


def parse_vector(value: object) -> np.ndarray:
    """Convert a parquet cell content to a float numpy array."""
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, np.ndarray):
        try:
            return value.astype(float, copy=False)
        except Exception:
            return np.asarray([], dtype=float)
    if isinstance(value, (list, tuple, pd.Series)):
        try:
            return np.asarray(value, dtype=float)
        except Exception:
            return np.asarray([], dtype=float)
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return np.asarray([], dtype=float)
        if txt.startswith("[") and txt.endswith("]"):
            try:
                parsed = ast.literal_eval(txt)
                return np.asarray(parsed, dtype=float)
            except Exception:
                return np.asarray([], dtype=float)
    try:
        return np.asarray(value, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def trim_to_same_length(*arrays: np.ndarray) -> List[np.ndarray]:
    lengths = [len(a) for a in arrays if a is not None and len(a) > 0]
    if not lengths:
        return [np.asarray([], dtype=float) for _ in arrays]
    m = min(lengths)
    return [np.asarray(a[:m], dtype=float) if a is not None and len(a) else np.asarray([], dtype=float) for a in arrays]


def compute_energy_from_power(time_s: np.ndarray, power_w: np.ndarray) -> np.ndarray:
    if len(time_s) < 2 or len(power_w) != len(time_s):
        return np.full(len(time_s), np.nan, dtype=float)
    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0
    dt = np.where(np.isfinite(dt) & (dt >= 0), dt, 0.0)
    return np.cumsum(np.abs(power_w) * dt) / 3600.0


def ensure_power_energy(row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_s = parse_vector(row.get("time_s"))
    voltage_v = parse_vector(row.get("voltage_v"))
    current_a = parse_vector(row.get("current_a"))
    temperature_c = parse_vector(row.get("temperature_c"))
    capacity_ah = parse_vector(row.get("capacity_ah"))
    energy_wh = parse_vector(row.get("energy_wh"))
    power_w = parse_vector(row.get("power_w"))

    time_s, voltage_v, current_a, capacity_ah = trim_to_same_length(time_s, voltage_v, current_a, capacity_ah)

    if len(power_w) != len(time_s):
        if len(voltage_v) == len(current_a) == len(time_s):
            power_w = voltage_v * current_a
        else:
            power_w = np.full(len(time_s), np.nan, dtype=float)

    if len(energy_wh) != len(time_s):
        energy_wh = compute_energy_from_power(time_s, power_w)

    if len(temperature_c) != len(time_s):
        if len(temperature_c) == 0:
            temperature_c = np.full(len(time_s), np.nan, dtype=float)
        else:
            temperature_c = np.asarray(temperature_c[: len(time_s)], dtype=float)
            if len(temperature_c) < len(time_s):
                pad = np.full(len(time_s) - len(temperature_c), np.nan, dtype=float)
                temperature_c = np.concatenate([temperature_c, pad])

    return time_s, voltage_v, current_a, capacity_ah, energy_wh, power_w, temperature_c


def equally_spaced_indices(n: int, k: int) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=int)
    k = max(1, int(k))
    if n <= k:
        return np.arange(n, dtype=int)
    idx = np.round(np.linspace(0, n - 1, k)).astype(int)
    idx = np.unique(idx)
    if len(idx) < k:
        used = set(idx.tolist())
        for cand in range(n):
            if cand not in used:
                idx = np.append(idx, cand)
                used.add(cand)
            if len(idx) >= k:
                break
    return np.sort(idx[:k])


def should_skip(path: Path, skip_existing: bool, force: bool) -> bool:
    if force:
        return False
    return bool(skip_existing and path.exists())


def discover_clean_cycle_files(interim_root: Path) -> List[Tuple[str, str, Path]]:
    results: List[Tuple[str, str, Path]] = []

    # Primary layout: data/interim/<dataset>/clean/<dataset>_<phase>_clean_cycles.parquet
    if interim_root.exists():
        for ds_dir in sorted(interim_root.iterdir()):
            if not ds_dir.is_dir():
                continue
            clean_dir = ds_dir / "clean"
            if not clean_dir.exists() or not clean_dir.is_dir():
                continue
            for phase in ("charge", "discharge"):
                for f in sorted(clean_dir.glob(f"*_{phase}_clean_cycles.parquet")):
                    results.append((canonicalize_dataset_name(ds_dir.name), phase, f))

    if results:
        return results

    # Fallback layout used by feature extraction v3: data/interim/charge/*.parquet and data/interim/discharge/*.parquet
    for phase in ("charge", "discharge"):
        phase_dir = interim_root / phase
        if phase_dir.exists() and phase_dir.is_dir():
            for f in sorted(phase_dir.glob("*.parquet")):
                stem = f.stem.lower()
                if "clean" not in stem and "cycle" not in stem:
                    continue
                ds_name = stem.split(f"_{phase}")[0] if f"_{phase}" in stem else "ALL"
                results.append((canonicalize_dataset_name(ds_name), phase, f))

    return results


def plot_single_cycle_grid(
    row: pd.Series,
    out_path: Path,
    dataset: str,
    battery_id: str,
    phase: str,
    cycle_number: object,
    dpi: int,
    plot_format: str,
) -> bool:
    time_s, voltage_v, current_a, capacity_ah, energy_wh, power_w, temperature_c = ensure_power_energy(row)
    if len(time_s) == 0:
        return False

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    series = [
        (voltage_v, "Voltage (V)"),
        (current_a, "Current (A)"),
        (capacity_ah, "Capacity (Ah)"),
        (energy_wh, "Energy (Wh)"),
        (power_w, "Power (W)"),
        (temperature_c, "Temperature (°C)"),
    ]

    for ax, (y, ylabel) in zip(axes, series):
        if len(y) == len(time_s):
            ax.plot(time_s, y, linewidth=1.4)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    cycle_idx = row.get("cycle_index", np.nan)
    rul = row.get("remaining_cycles", np.nan)
    fig.suptitle(
        f"{dataset} | {battery_id} | {phase} | cycle_number={cycle_number} | "
        f"cycle_index={cycle_idx} | remaining_cycles={rul}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=plot_format)
    plt.close(fig)
    return True


def plot_vq_overlay(
    selected_rows: pd.DataFrame,
    out_path: Path,
    dataset: str,
    battery_id: str,
    phase: str,
    dpi: int,
    plot_format: str,
) -> bool:
    if selected_rows.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, 7))
    plotted = 0
    sort_cols = [c for c in ["cycle_number", "cycle_index"] if c in selected_rows.columns]
    rows = selected_rows.sort_values(by=sort_cols, kind="mergesort") if sort_cols else selected_rows

    for _, row in rows.iterrows():
        _, voltage_v, _, capacity_ah, _, _, _ = ensure_power_energy(row)
        if len(voltage_v) == 0 or len(capacity_ah) == 0:
            continue
        m = min(len(voltage_v), len(capacity_ah))
        q = capacity_ah[:m]
        v = voltage_v[:m]
        finite = np.isfinite(q) & np.isfinite(v)
        if finite.sum() < 2:
            continue
        cycle_number = row.get("cycle_number", "?")
        ax.plot(q[finite], v[finite], linewidth=1.5, alpha=0.9, label=f"cycle {cycle_number}")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_xlabel("Capacity (Ah)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"V(Q) overlay | {dataset} | {battery_id} | {phase} | {plotted} selected cycles")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=plot_format)
    plt.close(fig)
    return True


def filter_list(values: Sequence[str]) -> List[str]:
    vals = [str(v).strip() for v in values if str(v).strip()]
    if not vals or any(v.lower() == "all" for v in vals):
        return ["all"]
    return vals


def select_cycles_per_group(group_df: pd.DataFrame, sample_cycles: int) -> pd.DataFrame:
    sort_cols = [c for c in ["cycle_number", "cycle_index"] if c in group_df.columns]
    g = group_df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True) if sort_cols else group_df.reset_index(drop=True)
    idx = equally_spaced_indices(len(g), sample_cycles)
    sel = g.iloc[idx].copy().reset_index(drop=True)
    sel["selection_position"] = np.arange(1, len(sel) + 1)
    sel["n_cycles_in_group"] = len(g)
    return sel


def process_file(
    file_dataset: str,
    file_phase: str,
    file_path: Path,
    args: argparse.Namespace,
    sampled_records: List[Dict[str, object]],
) -> Dict[str, int]:
    df = pd.read_parquet(file_path, engine=args.engine)
    if df.empty:
        return {"groups": 0, "cycle_figures": 0, "vq_figures": 0}

    if "dataset" not in df.columns:
        df["dataset"] = file_dataset
    if "phase" not in df.columns:
        df["phase"] = file_phase
    if "battery_id" not in df.columns:
        raise ValueError(f"Missing required column battery_id in {file_path}")

    df["dataset"] = df["dataset"].map(canonicalize_dataset_name)
    df["phase"] = df["phase"].astype(str).str.strip().str.lower()
    df["battery_id"] = df["battery_id"].astype(str)

    selected_datasets = filter_list(args.datasets)
    selected_phases = filter_list(args.phases)
    if selected_datasets != ["all"]:
        allowed = {canonicalize_dataset_name(x) for x in selected_datasets}
        df = df[df["dataset"].isin(allowed)].copy()
    if selected_phases != ["all"]:
        allowed_phases = {x.lower() for x in selected_phases}
        df = df[df["phase"].isin(allowed_phases)].copy()
    if df.empty:
        return {"groups": 0, "cycle_figures": 0, "vq_figures": 0}

    # Prefer model-usable cycles when the flag exists and contains at least one True.
    if "is_model_usable" in df.columns:
        mask = df["is_model_usable"].fillna(True).astype(bool)
        if mask.any():
            df = df[mask].copy()
    elif "is_valid_cycle" in df.columns:
        mask = df["is_valid_cycle"].fillna(True).astype(bool)
        if mask.any():
            df = df[mask].copy()
    if df.empty:
        return {"groups": 0, "cycle_figures": 0, "vq_figures": 0}

    grouped = list(df.groupby(["dataset", "battery_id", "phase"], sort=True, dropna=False))
    if args.max_batteries and args.max_batteries > 0:
        grouped = grouped[: args.max_batteries]

    counts = {"groups": 0, "cycle_figures": 0, "vq_figures": 0}

    for (dataset, battery_id, phase), g in tqdm(
        grouped,
        desc=f"Batteries in {file_path.name}",
        leave=False,
        unit="battery-phase",
    ):
        counts["groups"] += 1
        selected = select_cycles_per_group(g, args.sample_raw_cycles)

        base_dir = (
            Path(args.output_root)
            / safe_slug(dataset)
            / "selected_cycles"
            / safe_slug(battery_id)
            / safe_slug(phase)
        )

        for _, row in selected.iterrows():
            cycle_number = row.get("cycle_number", "unknown")
            out_path = base_dir / f"{safe_slug(f'cycle_{cycle_number}')}_signals.{args.plot_format}"
            if not should_skip(out_path, args.skip_existing, args.force):
                ok = plot_single_cycle_grid(
                    row=row,
                    out_path=out_path,
                    dataset=str(dataset),
                    battery_id=str(battery_id),
                    phase=str(phase),
                    cycle_number=cycle_number,
                    dpi=args.dpi,
                    plot_format=args.plot_format,
                )
                if ok:
                    counts["cycle_figures"] += 1

            sampled_records.append(
                {
                    "dataset": dataset,
                    "battery_id": battery_id,
                    "phase": phase,
                    "cycle_number": row.get("cycle_number", np.nan),
                    "cycle_index": row.get("cycle_index", np.nan),
                    "cycle_life": row.get("cycle_life", np.nan),
                    "remaining_cycles": row.get("remaining_cycles", np.nan),
                    "remaining_cycles_norm": row.get("remaining_cycles_norm", np.nan),
                    "selection_position": row.get("selection_position", np.nan),
                    "n_cycles_in_group": row.get("n_cycles_in_group", np.nan),
                    "source_file": row.get("source_file", np.nan),
                    "source_parquet": str(file_path),
                }
            )

        vq_out = base_dir / f"VQ_selected_cycles_overlay.{args.plot_format}"
        if not should_skip(vq_out, args.skip_existing, args.force):
            ok = plot_vq_overlay(
                selected_rows=selected,
                out_path=vq_out,
                dataset=str(dataset),
                battery_id=str(battery_id),
                phase=str(phase),
                dpi=args.dpi,
                plot_format=args.plot_format,
            )
            if ok:
                counts["vq_figures"] += 1

    return counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extra audit: selected-cycle raw-signal grids and V(Q) overlays.")
    p.add_argument("--interim-root", default=DEFAULT_INTERIM_ROOT)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--datasets", nargs="*", default=["all"], help="Datasets to process, e.g. CALCE HUST or all.")
    p.add_argument("--phases", nargs="*", default=["charge", "discharge"], help="Phases to process, e.g. charge discharge or all.")
    p.add_argument("--engine", choices=["fastparquet", "pyarrow"], default="fastparquet")
    p.add_argument("--sample-raw-cycles", type=int, default=5, help="Number of equally spaced cycles selected per battery/phase.")
    p.add_argument("--max-batteries", type=int, default=0, help="Optional limit of battery/phase groups per file for debugging. 0 means no limit.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--plot-format", choices=["png", "jpg", "jpeg", "pdf", "svg"], default="png")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--save-selection-table", action="store_true", help="Save CSV with all selected cycles.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    interim_root = Path(args.interim_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = discover_clean_cycle_files(interim_root)
    if not files:
        print(f"No clean cycle parquet files found under: {interim_root}")
        return 1

    selected_datasets = filter_list(args.datasets)
    selected_phases = filter_list(args.phases)
    if selected_datasets != ["all"]:
        ds_allowed = {canonicalize_dataset_name(x) for x in selected_datasets}
        files = [item for item in files if item[0] in ds_allowed or item[0] == "ALL"]
    if selected_phases != ["all"]:
        ph_allowed = {x.lower() for x in selected_phases}
        files = [item for item in files if item[1].lower() in ph_allowed]
    if not files:
        print("No files remain after dataset/phase filtering.")
        return 1

    sampled_records: List[Dict[str, object]] = []
    total_groups = 0
    total_cycle_figs = 0
    total_vq_figs = 0

    for dataset, phase, path in tqdm(files, desc="Clean cycle files", unit="file"):
        counts = process_file(dataset, phase, path, args, sampled_records)
        total_groups += counts["groups"]
        total_cycle_figs += counts["cycle_figures"]
        total_vq_figs += counts["vq_figures"]

    if args.save_selection_table or sampled_records:
        global_tables = output_root / "global" / "tables"
        global_tables.mkdir(parents=True, exist_ok=True)
        sel_df = pd.DataFrame(sampled_records)
        if not sel_df.empty:
            sel_df = sel_df.sort_values(
                by=["dataset", "battery_id", "phase", "selection_position", "cycle_number"],
                kind="mergesort",
            ).reset_index(drop=True)
        sel_df.to_csv(global_tables / "selected_cycles_visual_audit.csv", index=False, encoding="utf-8-sig")

    print("\nExtra visual audit completed.")
    print(f"Battery-phase groups processed: {total_groups}")
    print(f"Single-cycle 6-signal figures: {total_cycle_figs}")
    print(f"V(Q) overlay figures: {total_vq_figs}")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
