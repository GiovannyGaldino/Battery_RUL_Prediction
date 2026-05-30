# -*- coding: utf-8 -*-
"""
Generate Figure 4 for the battery RUL article.

Figure 4. Example of protocol-induced voltage changes in HUST cell 9-4.
(a) Raw voltage, current, and temperature for selected discharge cycles;
(b) corresponding discharge V(Q) curves for the same cycles.

Default input priority:
1) C:\Artigo Baterias de Litio\Batteries\data\interim\discharge\hust_discharge_clean_cycles.parquet
2) C:\Artigo Baterias de Litio\Batteries\data\interim\hust\clean\hust_discharge_clean_cycles.parquet

Outputs:
C:\Artigo Baterias de Litio\Batteries\reports\article_figures\figure4_hust_cell_9_4\figure4a_hust_cell_9_4_raw_signals.png
C:\Artigo Baterias de Litio\Batteries\reports\article_figures\figure4_hust_cell_9_4\figure4b_hust_cell_9_4_vq.png
C:\Artigo Baterias de Litio\Batteries\reports\article_figures\figure4_hust_cell_9_4\figure4_selected_cycles.csv
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_PROJECT_ROOT = Path(r"C:\Artigo Baterias de Litio\Batteries")
DEFAULT_DATA_ROOT = DEFAULT_PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "reports" / "article_figures" / "figure4_hust_cell_9_4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 4 for HUST cell 9-4.")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--battery", default="9-4", help="Battery identifier fragment. Default: 9-4")
    parser.add_argument("--first-cycle", type=float, default=4.0, help="First selected cycle. Default: 4")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-format", choices=["png", "pdf", "tiff", "svg"], default="png")
    parser.add_argument("--engine", choices=["auto", "fastparquet", "pyarrow"], default="auto")
    parser.add_argument("--show-capacity-in-panel-a", action="store_true", help="Use capacity instead of temperature in panel (a).")
    return parser.parse_args()


def choose_engine(engine: str) -> str:
    if engine in {"fastparquet", "pyarrow"}:
        return engine
    try:
        import fastparquet  # noqa: F401
        return "fastparquet"
    except Exception:
        pass
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        pass
    raise RuntimeError("Install fastparquet or pyarrow to read parquet files.")


def vector_to_float_array(value: Any) -> np.ndarray:
    """Convert list-like parquet cells or stringified vectors to 1-D float arrays."""
    if value is None:
        return np.array([], dtype=float)

    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, pd.Series):
        arr = value.to_numpy()
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
            arr = np.asarray([p for p in re.split(r"[,\s]+", text) if p])
    else:
        try:
            if pd.isna(value):
                return np.array([], dtype=float)
        except Exception:
            pass
        arr = np.asarray([value])

    try:
        out = np.asarray(arr, dtype=float).reshape(-1)
    except Exception:
        converted = []
        for item in np.asarray(arr, dtype=object).reshape(-1):
            try:
                converted.append(float(item))
            except Exception:
                continue
        out = np.asarray(converted, dtype=float)

    return out[np.isfinite(out)]


def align_vectors(*vectors: np.ndarray) -> list[np.ndarray]:
    lengths = [len(v) for v in vectors if len(v) > 0]
    if not lengths:
        return [np.array([], dtype=float) for _ in vectors]
    n = min(lengths)
    return [v[:n] if len(v) else np.full(n, np.nan) for v in vectors]


def find_hust_discharge_file(data_root: Path) -> Path:
    candidates = [
        data_root / "interim" / "discharge" / "hust_discharge_clean_cycles.parquet",
        data_root / "interim" / "hust" / "clean" / "hust_discharge_clean_cycles.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = list((data_root / "interim").glob("**/*hust*discharge*clean_cycles.parquet"))
    if matches:
        return sorted(matches, key=lambda p: len(str(p)))[0]

    raise FileNotFoundError(
        "Could not find HUST discharge clean-cycle parquet. Checked global and per-dataset folders."
    )


def normalize_battery_text(text: Any) -> str:
    return str(text).lower().replace("_", "-").replace(" ", "")


def select_battery(df: pd.DataFrame, battery_fragment: str) -> pd.DataFrame:
    if "battery_id" not in df.columns:
        raise KeyError("Column 'battery_id' was not found.")

    target = normalize_battery_text(battery_fragment)
    mask = df["battery_id"].map(lambda x: target in normalize_battery_text(x))
    out = df.loc[mask].copy()

    if out.empty:
        available = sorted(df["battery_id"].astype(str).unique())[:50]
        raise ValueError(
            f"No battery matching '{battery_fragment}' was found. First available IDs: {available}"
        )
    return out


def scalar_float(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return np.nan
    try:
        value = row[col]
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = vector_to_float_array(value)
            return float(arr[-1]) if len(arr) else np.nan
        return float(value)
    except Exception:
        return np.nan


def get_phase_capacity(row: pd.Series) -> float:
    for col in ["phase_capacity_ah", "target_capacity_ah", "input_phase_capacity_ah"]:
        val = scalar_float(row, col)
        if np.isfinite(val) and val > 0:
            return val
    arr = vector_to_float_array(row.get("capacity_ah", None))
    return float(arr[-1]) if len(arr) else np.nan


def select_eol_cycle(df: pd.DataFrame) -> pd.Series:
    """Select the row corresponding to EOL. Priority: cycle_life; SOH/capacity <= 0.8; last cycle."""
    work = df.sort_values("cycle_number").copy()

    # 1) If cycle_life is available, use the row closest to cycle_life.
    if "cycle_life" in work.columns:
        cl = pd.to_numeric(work["cycle_life"], errors="coerce").dropna()
        if not cl.empty:
            eol_cycle = float(cl.median())
            idx = (pd.to_numeric(work["cycle_number"], errors="coerce") - eol_cycle).abs().idxmin()
            return work.loc[idx]

    # 2) If SOH is available, use first row at or below 80%.
    if "soh" in work.columns:
        soh = pd.to_numeric(work["soh"], errors="coerce")
        below = work.loc[soh <= 0.8]
        if not below.empty:
            return below.iloc[0]

    # 3) Compute capacity ratio from first valid capacity.
    capacities = work.apply(get_phase_capacity, axis=1).astype(float)
    valid = capacities[np.isfinite(capacities) & (capacities > 0)]
    if not valid.empty:
        initial = float(valid.iloc[:5].median()) if len(valid) >= 5 else float(valid.iloc[0])
        below_idx = capacities.index[capacities <= 0.8 * initial]
        if len(below_idx) > 0:
            return work.loc[below_idx[0]]

    # 4) Fallback: last available cycle.
    return work.iloc[-1]


def select_nearest_cycle(df: pd.DataFrame, target_cycle: float) -> pd.Series:
    cycles = pd.to_numeric(df["cycle_number"], errors="coerce")
    idx = (cycles - target_cycle).abs().idxmin()
    return df.loc[idx]


def selected_rows_for_figure(df: pd.DataFrame, first_cycle: float) -> pd.DataFrame:
    work = df.copy()
    work["cycle_number"] = pd.to_numeric(work["cycle_number"], errors="coerce")
    work = work[np.isfinite(work["cycle_number"])].sort_values("cycle_number")
    if work.empty:
        raise ValueError("No finite cycle_number values were found for the selected battery.")

    first = select_nearest_cycle(work, first_cycle)
    eol = select_eol_cycle(work)
    mid_target = (float(first["cycle_number"]) + float(eol["cycle_number"])) / 2.0
    mid = select_nearest_cycle(work, mid_target)

    selected = pd.DataFrame([first, mid, eol]).drop_duplicates(subset=["cycle_number"], keep="first")
    selected = selected.sort_values("cycle_number").reset_index(drop=True)
    return selected


def label_for_cycle(cycle_number: Any) -> str:
    return f"Cycle {int(round(float(cycle_number)))}"


def make_panel_a(selected: pd.DataFrame, output_path: Path, dpi: int, use_capacity: bool) -> None:
    third_col = "capacity_ah" if use_capacity else "temperature_c"
    third_label = "Capacity (Ah)" if use_capacity else "Temperature (°C)"

    fig, axes = plt.subplots(3, 1, figsize=(4.7, 5.2), sharex=True, constrained_layout=True)
    specs = [
        ("voltage_v", "Voltage (V)"),
        ("current_a", "Current (A)"),
        (third_col, third_label),
    ]

    for row in selected.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        t = vector_to_float_array(row_s.get("time_s", None))
        v = vector_to_float_array(row_s.get("voltage_v", None))
        i = vector_to_float_array(row_s.get("current_a", None))
        temp = vector_to_float_array(row_s.get("temperature_c", None))
        q = vector_to_float_array(row_s.get("capacity_ah", None))
        t, v, i, temp, q = align_vectors(t, v, i, temp, q)

        # Use minutes to improve readability for cycle-level curves.
        x = t / 60.0 if len(t) else t
        values = {
            "voltage_v": v,
            "current_a": i,
            "temperature_c": temp,
            "capacity_ah": q,
        }
        label = label_for_cycle(row_s["cycle_number"])
        for ax, (col, _) in zip(axes, specs):
            y = values[col]
            if len(x) and len(y):
                ax.plot(x, y, linewidth=1.2, label=label)

    for ax, (_, ylabel) in zip(axes, specs):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (min)")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_panel_b(selected: pd.DataFrame, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.7, 5.2), constrained_layout=True)

    for row in selected.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        q = vector_to_float_array(row_s.get("capacity_ah", None))
        v = vector_to_float_array(row_s.get("voltage_v", None))
        q, v = align_vectors(q, v)
        if len(q) and len(v):
            order = np.argsort(q)
            ax.plot(q[order], v[order], linewidth=1.4, label=label_for_cycle(row_s["cycle_number"]))

    ax.set_xlabel("Capacity, Q (Ah)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    parquet_file = find_hust_discharge_file(data_root)
    engine = choose_engine(args.engine)
    df = pd.read_parquet(parquet_file, engine=engine)

    if "feature_level" in df.columns:
        df = df[df["feature_level"].astype(str).str.lower().eq("phase")].copy()

    if "phase" in df.columns:
        df = df[df["phase"].astype(str).str.lower().eq("discharge")].copy()

    df_cell = select_battery(df, args.battery)
    selected = selected_rows_for_figure(df_cell, args.first_cycle)

    selected_info = selected[[c for c in [
        "dataset", "battery_id", "phase", "cycle_number", "cycle_index", "cycle_life",
        "remaining_cycles", "remaining_cycles_norm", "soh", "phase_capacity_ah",
        "initial_capacity_ah", "eol_criterion"
    ] if c in selected.columns]].copy()
    selected_info.to_csv(output_root / "figure4_selected_cycles.csv", index=False, encoding="utf-8-sig")

    fig_format = args.fig_format
    panel_a = output_root / f"figure4a_hust_cell_9_4_raw_signals.{fig_format}"
    panel_b = output_root / f"figure4b_hust_cell_9_4_vq.{fig_format}"

    make_panel_a(selected, panel_a, dpi=args.dpi, use_capacity=args.show_capacity_in_panel_a)
    make_panel_b(selected, panel_b, dpi=args.dpi)

    print("Figure 4 generated.")
    print(f"Input file: {parquet_file}")
    print("Selected cycles:", ", ".join(str(int(round(float(x)))) for x in selected["cycle_number"].tolist()))
    print(f"Panel (a): {panel_a}")
    print(f"Panel (b): {panel_b}")
    print(f"Selected-cycle table: {output_root / 'figure4_selected_cycles.csv'}")


if __name__ == "__main__":
    main()
