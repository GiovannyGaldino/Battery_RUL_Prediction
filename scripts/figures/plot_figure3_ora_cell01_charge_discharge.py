# -*- coding: utf-8 -*-
r"""
Figure 3 — ORA/Oxford cell01 raw-signal audit.

This script generates one figure for charge and one figure for discharge.
Each figure has six vertically aligned plots sharing one x-axis:

1. voltage vs local phase time
2. current vs local phase time
3. capacity vs local phase time
4. temperature vs local phase time
5. energy vs local phase time
6. power vs local phase time

Each subplot has its own y-axis. The selected curves correspond to early-,
middle-, and late-life cycles.

Example PowerShell command:

& python `
  ".\scripts\article\plot_figure3_ora_cell01_charge_discharge.py" `
  --data-root ".\data" `
  --output-root ".\reports\article_figures\figure3_ora_cell01_raw_signal_audit" `
  --engine fastparquet `
  --formats png pdf `
  --dpi 600 `
  --line-width 2.8 `
  --font-size 15 `
  --tick-size 12 `
  --legend-size 16
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


DEFAULT_PROJECT_ROOT = Path.cwd()
DEFAULT_DATA_ROOT = DEFAULT_PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "reports" / "article_figures" / "figure3_ora_cell01_raw_signal_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ORA/Oxford cell01 charge/discharge raw-signal audit figures with six shared-x subplots."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset-tokens", default="ora,oxford")
    parser.add_argument("--battery", default="cell01")
    parser.add_argument("--phases", nargs="+", choices=["charge", "discharge"], default=["charge", "discharge"])
    parser.add_argument("--selection-mode", choices=["linspace", "eol"], default="linspace")
    parser.add_argument("--n-cycles", type=int, default=3)
    parser.add_argument("--engine", choices=["auto", "fastparquet", "pyarrow"], default="auto")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg", "tiff"], default=["png"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--fig-width", type=float, default=12.5)
    parser.add_argument("--fig-height", type=float, default=13.5)
    parser.add_argument("--line-width", type=float, default=2.8)
    parser.add_argument("--font-size", type=float, default=15.0)
    parser.add_argument("--tick-size", type=float, default=12.0)
    parser.add_argument("--legend-size", type=float, default=16.0)
    parser.add_argument("--title-size", type=float, default=17.0)
    parser.add_argument("--time-unit", choices=["s", "min"], default="min")
    parser.add_argument("--max-available-list", type=int, default=80)
    return parser.parse_args()


def configure_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.labelsize": args.font_size,
        "axes.titlesize": args.font_size,
        "xtick.labelsize": args.tick_size,
        "ytick.labelsize": args.tick_size,
        "legend.fontsize": args.legend_size,
        "figure.titlesize": args.title_size,
        "lines.linewidth": args.line_width,
    })


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


def split_tokens(text: str) -> list[str]:
    return [token.strip().lower() for token in text.split(",") if token.strip()]


def normalize_key(value: Any) -> str:
    """Normalize text for robust matching: cell_01, cell-01, and cell01 become cell01."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def vector_to_float_array(value: Any) -> np.ndarray:
    """Convert list-like parquet cells or stringified vectors to a 1-D float array."""
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
            arr = np.asarray([part for part in re.split(r"[,\s]+", text) if part])
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


def first_finite_scalar(row: pd.Series, columns: list[str]) -> float:
    for col in columns:
        if col not in row.index:
            continue
        value = row[col]
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = vector_to_float_array(value)
            if len(arr):
                val = float(arr[-1])
                if np.isfinite(val):
                    return val
        else:
            try:
                val = float(value)
                if np.isfinite(val):
                    return val
            except Exception:
                continue
    return np.nan


def align_to_time(time_s: np.ndarray, values: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Align vectors by the shortest available non-empty length and sort by time."""
    lengths = []
    if len(time_s):
        lengths.append(len(time_s))
    lengths += [len(arr) for arr in values.values() if len(arr)]

    if not lengths:
        return np.array([], dtype=float), {key: np.array([], dtype=float) for key in values}

    n = min(lengths)
    t = time_s[:n].astype(float) if len(time_s) else np.arange(n, dtype=float)

    aligned = {}
    for key, arr in values.items():
        aligned[key] = arr[:n].astype(float) if len(arr) else np.full(n, np.nan, dtype=float)

    if len(t) > 1:
        order = np.argsort(t)
        t = t[order]
        aligned = {key: arr[order] for key, arr in aligned.items()}

    if len(t) and np.isfinite(t[0]):
        t = t - t[0]

    return t, aligned


def integrate_capacity_from_current(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    if len(time_s) < 2 or len(current_a) < 2:
        return np.array([], dtype=float)
    n = min(len(time_s), len(current_a))
    dt = np.diff(time_s[:n])
    i_abs = np.abs(current_a[:n])
    dq = 0.5 * (i_abs[:-1] + i_abs[1:]) * dt / 3600.0
    return np.r_[0.0, np.cumsum(dq)]


def integrate_energy_from_power(time_s: np.ndarray, power_w: np.ndarray) -> np.ndarray:
    if len(time_s) < 2 or len(power_w) < 2:
        return np.array([], dtype=float)
    n = min(len(time_s), len(power_w))
    dt = np.diff(time_s[:n])
    p_abs = np.abs(power_w[:n])
    de = 0.5 * (p_abs[:-1] + p_abs[1:]) * dt / 3600.0
    return np.r_[0.0, np.cumsum(de)]


def infer_dataset_from_path(path: Path) -> str:
    text = normalize_key(path.name)
    for token in ["oxford", "ora", "matr", "hust", "calce", "pcoe", "rwth", "xjtu", "rw", "nrw"]:
        if token in text:
            return token.upper()
    return path.stem.split("_")[0].upper()


def find_clean_cycle_files(data_root: Path, phase: str, dataset_tokens: list[str]) -> list[Path]:
    interim = data_root / "interim"
    candidates: list[Path] = []

    global_phase_dir = interim / phase
    if global_phase_dir.exists():
        candidates.extend(sorted(global_phase_dir.glob("*clean*.parquet")))
        candidates.extend(sorted(global_phase_dir.glob(f"*{phase}*.parquet")))

    candidates.extend(sorted(interim.glob(f"*/clean/*{phase}*clean*.parquet")))
    candidates.extend(sorted(interim.glob("*/clean/*clean*.parquet")))

    norm_tokens = [normalize_key(token) for token in dataset_tokens]
    unique = []
    seen = set()

    for path in candidates:
        if not path.exists():
            continue
        path_text = normalize_key(path)
        phase_ok = normalize_key(phase) in path_text
        dataset_ok = any(token in path_text for token in norm_tokens)

        if not (phase_ok and dataset_ok):
            continue

        key = str(path).lower()
        if key not in seen:
            unique.append(path)
            seen.add(key)

    if not unique:
        raise FileNotFoundError(
            f"No parquet files found for phase='{phase}' and dataset_tokens={dataset_tokens} under {interim}"
        )

    return unique


def load_phase_dataset(data_root: Path, phase: str, dataset_tokens: list[str], engine: str) -> pd.DataFrame:
    files = find_clean_cycle_files(data_root, phase, dataset_tokens)
    frames: list[pd.DataFrame] = []

    print(f"Files found for {phase}:")
    for path in files:
        print(f"  - {path}")
        try:
            df = pd.read_parquet(path, engine=engine)
        except Exception as exc:
            print(f"Warning: could not read {path}: {exc}")
            continue

        if df.empty:
            continue

        if "dataset" not in df.columns:
            df["dataset"] = infer_dataset_from_path(path)

        if "phase" not in df.columns:
            df["phase"] = phase

        df["source_parquet"] = str(path)

        if "feature_level" in df.columns:
            df = df[df["feature_level"].astype(str).str.lower().eq("phase")].copy()

        if "phase" in df.columns:
            df = df[df["phase"].astype(str).str.lower().eq(phase)].copy()

        if "dataset" in df.columns:
            norm_tokens = [normalize_key(token) for token in dataset_tokens]
            dataset_mask = df["dataset"].map(lambda x: any(token in normalize_key(x) for token in norm_tokens))
            if dataset_mask.any():
                df = df[dataset_mask].copy()

        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No readable rows found for phase='{phase}', dataset_tokens={dataset_tokens}")

    out = pd.concat(frames, ignore_index=True, sort=False)

    required = {"battery_id", "cycle_number"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Missing required columns in loaded dataframe: {sorted(missing)}")

    return out


def filter_battery(df: pd.DataFrame, battery_fragment: str, max_available: int) -> pd.DataFrame:
    target = normalize_key(battery_fragment)

    mask = df["battery_id"].map(lambda x: target in normalize_key(x))
    out = df.loc[mask].copy()

    if out.empty and "cell_id_original" in df.columns:
        mask = df["cell_id_original"].map(lambda x: target in normalize_key(x))
        out = df.loc[mask].copy()

    if out.empty:
        available = sorted(df["battery_id"].astype(str).unique())[:max_available]
        raise ValueError(
            f"No battery matching fragment '{battery_fragment}' was found. "
            f"First available battery_id values: {available}"
        )

    out["cycle_number"] = pd.to_numeric(out["cycle_number"], errors="coerce")
    out = out[np.isfinite(out["cycle_number"])].sort_values("cycle_number").reset_index(drop=True)

    if out.empty:
        raise ValueError(f"Battery '{battery_fragment}' has no finite cycle_number values.")

    return out


def select_cycles(df_battery: pd.DataFrame, n_cycles: int, mode: str) -> pd.DataFrame:
    work = df_battery.sort_values("cycle_number").reset_index(drop=True).copy()
    n_cycles = max(1, int(n_cycles))

    if len(work) <= n_cycles:
        return work.copy()

    if mode == "eol" and "cycle_life" in work.columns:
        cycle_life_values = pd.to_numeric(work["cycle_life"], errors="coerce")
        if cycle_life_values.notna().any():
            first_cycle = float(work["cycle_number"].iloc[0])
            last_available = float(work["cycle_number"].iloc[-1])
            eol_cycle = float(cycle_life_values.dropna().median())
            late_cycle = min(eol_cycle, last_available)

            if late_cycle > first_cycle:
                targets = np.linspace(first_cycle, late_cycle, n_cycles)
                cycles = work["cycle_number"].to_numpy(dtype=float)
                indices = sorted(set(int(np.argmin(np.abs(cycles - target))) for target in targets))
                return work.iloc[indices].copy().reset_index(drop=True)

    indices = np.linspace(0, len(work) - 1, n_cycles).round().astype(int)
    indices = sorted(set(indices))
    return work.iloc[indices].copy().reset_index(drop=True)


def get_signal_vectors(row: pd.Series) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, str]]:
    notes: dict[str, str] = {}

    time_s = vector_to_float_array(row.get("time_s", None))
    voltage = vector_to_float_array(row.get("voltage_v", None))
    current = vector_to_float_array(row.get("current_a", None))
    temperature = vector_to_float_array(row.get("temperature_c", None))
    capacity = vector_to_float_array(row.get("capacity_ah", None))
    energy = vector_to_float_array(row.get("energy_wh", None))
    power = vector_to_float_array(row.get("power_w", None))

    t0, aligned0 = align_to_time(
        time_s,
        {
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "capacity": capacity,
            "energy": energy,
            "power": power,
        },
    )

    voltage = aligned0["voltage"]
    current = aligned0["current"]
    temperature = aligned0["temperature"]
    capacity = aligned0["capacity"]
    energy = aligned0["energy"]
    power = aligned0["power"]

    if (not len(power)) or np.all(~np.isfinite(power)):
        if len(voltage) and len(current) and np.isfinite(voltage).any() and np.isfinite(current).any():
            power = voltage * current
            notes["power"] = "reconstructed as voltage x current"
        else:
            notes["power"] = "missing"

    if (not len(capacity)) or np.all(~np.isfinite(capacity)):
        reconstructed = integrate_capacity_from_current(t0, current)
        if len(reconstructed):
            capacity = reconstructed
            notes["capacity"] = "reconstructed by current integration"
        else:
            notes["capacity"] = "missing"

    if (not len(energy)) or np.all(~np.isfinite(energy)):
        reconstructed = integrate_energy_from_power(t0, power)
        if len(reconstructed):
            energy = reconstructed
            notes["energy"] = "reconstructed by power integration"
        else:
            notes["energy"] = "missing"

    if (not len(temperature)) or np.all(~np.isfinite(temperature)):
        temp_scalar = first_finite_scalar(
            row,
            [
                "temperature_mean_c",
                "input_temperature_mean_c",
                "ambient_temperature_c",
                "input_ambient_temperature_c",
                "temperature_min_c",
                "input_temperature_min_c",
            ],
        )
        if np.isfinite(temp_scalar) and len(t0):
            temperature = np.full(len(t0), temp_scalar, dtype=float)
            notes["temperature"] = "constant scalar/ambient temperature used"
        else:
            notes["temperature"] = "missing"

    t, aligned = align_to_time(
        t0,
        {
            "Voltage (V)": voltage,
            "Current (A)": current,
            "Capacity (Ah)": capacity,
            "Temperature (deg C)": temperature,
            "Energy (Wh)": energy,
            "Power (W)": power,
        },
    )

    return t, aligned, notes


def cycle_label(row: pd.Series, position: int, total: int) -> str:
    cycle = row.get("cycle_number", np.nan)

    if total == 3:
        prefix = ["Early", "Middle", "Late"][position]
    else:
        prefix = f"Selected {position + 1}"

    try:
        if np.isfinite(float(cycle)):
            return f"{prefix} - Cycle {int(round(float(cycle)))}"
    except Exception:
        pass

    return prefix


def plot_shared_x_signals(
    selected_rows: pd.DataFrame,
    phase: str,
    output_base: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    signal_order = [
        "Voltage (V)",
        "Current (A)",
        "Capacity (Ah)",
        "Temperature (deg C)",
        "Energy (Wh)",
        "Power (W)",
    ]

    fig, axes = plt.subplots(
        len(signal_order),
        1,
        figsize=(args.fig_width, args.fig_height),
        sharex=True,
        constrained_layout=True,
    )

    if len(signal_order) == 1:
        axes = [axes]

    line_styles = ["-", "--", ":"]
    selected_info_rows = []

    for position, (_, row) in enumerate(selected_rows.iterrows()):
        t, signals, notes = get_signal_vectors(row)

        if not len(t):
            continue

        x = t / 60.0 if args.time_unit == "min" else t
        label = cycle_label(row, position=position, total=len(selected_rows))
        linestyle = line_styles[position % len(line_styles)]

        for ax, signal_name in zip(axes, signal_order):
            y = signals.get(signal_name, np.array([], dtype=float))
            if len(y) and np.isfinite(y).any():
                ax.plot(
                    x,
                    y,
                    linestyle=linestyle,
                    linewidth=args.line_width,
                    label=label,
                )

        selected_info_rows.append(
            {
                "phase": phase,
                "dataset": row.get("dataset", ""),
                "battery_id": row.get("battery_id", ""),
                "cell_id_original": row.get("cell_id_original", ""),
                "cycle_number": row.get("cycle_number", np.nan),
                "cycle_index": row.get("cycle_index", np.nan),
                "cycle_life": row.get("cycle_life", np.nan),
                "remaining_cycles": row.get("remaining_cycles", np.nan),
                "remaining_cycles_norm": row.get("remaining_cycles_norm", np.nan),
                "soh": row.get("soh", np.nan),
                "source_parquet": row.get("source_parquet", ""),
                "notes": "; ".join(f"{key}: {value}" for key, value in notes.items()),
            }
        )

    panel_letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for ax, signal_name, letter in zip(axes, signal_order, panel_letters):
        ax.set_ylabel(signal_name)
        ax.grid(True, alpha=0.25)
        ax.text(
            0.01,
            0.88,
            letter,
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            ha="left",
            fontsize=args.font_size,
        )
        if not ax.lines:
            ax.text(
                0.5,
                0.5,
                "Signal unavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

    axes[-1].set_xlabel("Local phase time (min)" if args.time_unit == "min" else "Local phase time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=min(len(labels), 3),
        frameon=False,
        fontsize=args.legend_size,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])


    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        out = output_base.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)
    return pd.DataFrame(selected_info_rows)


def run_phase(phase: str, args: argparse.Namespace) -> pd.DataFrame:
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_tokens = split_tokens(args.dataset_tokens)
    engine = choose_engine(args.engine)

    print("-" * 80)
    print(f"Processing phase: {phase}")
    print(f"Dataset tokens: {dataset_tokens}")
    print(f"Battery fragment: {args.battery}")

    df = load_phase_dataset(data_root, phase=phase, dataset_tokens=dataset_tokens, engine=engine)
    df_battery = filter_battery(df, battery_fragment=args.battery, max_available=args.max_available_list)
    selected = select_cycles(df_battery, n_cycles=args.n_cycles, mode=args.selection_mode)

    selected_cycles = ", ".join(str(int(round(float(c)))) for c in selected["cycle_number"].tolist())
    print(f"Selected cycles for {phase}: {selected_cycles}")

    output_base = output_root / f"figure3_ora_cell01_{phase}_shared_x_six_y"
    selected_info = plot_shared_x_signals(selected, phase=phase, output_base=output_base, args=args)

    csv_path = output_base.with_name(output_base.name + "_selected_cycles.csv")
    selected_info.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Selected-cycle table: {csv_path}")

    return selected_info


def main() -> None:
    args = parse_args()
    configure_matplotlib(args)

    all_info = []
    for phase in args.phases:
        all_info.append(run_phase(phase, args))

    if all_info:
        output_root = Path(args.output_root)
        combined = pd.concat(all_info, ignore_index=True, sort=False)
        combined_path = output_root / "figure3_ora_cell01_selected_cycles_all.csv"
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print("-" * 80)
        print("Finished.")
        print(f"Combined selected-cycle table: {combined_path}")


if __name__ == "__main__":
    main()
