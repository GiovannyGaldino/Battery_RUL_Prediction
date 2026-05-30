# -*- coding: utf-8 -*-
r"""
Figure 5 — Normalized capacity trajectories used for EOL and RUL reconstruction.

Six-image version:
    - one complete-life image is generated for each selected battery; CALCE CS2_33 complete view is limited to cycle 23;
    - one zoom image is generated for each selected battery;
    - zoom images show only the first 15% of the available cycle-number range;
    - zoom images always show at least five cycles when available;
    - the y-axis shows absolute phase capacity in Ah, not normalized capacity;
    - horizontal reference lines are drawn only at 1.0 x nominal capacity and 0.8 x nominal capacity.

Memory-safe version:
    - prefers global files in data/interim/<phase>/ when they exist;
    - avoids reading duplicated per-dataset clean files;
    - reads only scalar/metadata columns needed for this figure.

This script generates a combined figure with normalized phase capacity versus cycle
number for three representative cells:

    1) MATR 2019 b033
    2) HUST 9-2
    3) CALCE CS2_33

The figure highlights:
    - short initial capacity increase;
    - long-term degradation;
    - local capacity recovery;
    - EOL threshold crossing at 0.80 and 0.70;
    - different lifetime scales among public datasets.

Default phase:
    discharge

Recommended PowerShell command:

& python `
  ".\scripts\article\plot_figure5_normalized_capacity_trajectories.py" `
  --data-root ".\data" `
  --output-root ".\reports\article_figures\figure5_normalized_capacity_trajectories" `
  --engine fastparquet `
  --formats png pdf `
  --dpi 600 `
  --line-width 2.6 `
  --marker-size 3.8 `
  --font-size 14 `
  --tick-size 12 `
  --legend-size 12 `
  --initial-window 5 `
  --initial-method median

Outputs:
    figure5_normalized_capacity_trajectories.png
    figure5_normalized_capacity_trajectories.pdf
    figure5_selected_batteries_summary.csv
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_ROOT = Path(
    r".\reports\article_figures\figure5_normalized_capacity_trajectories"
)


@dataclass
class BatterySpec:
    dataset_tokens: list[str]
    battery_fragment: str
    extra_filter: str
    label: str


DEFAULT_BATTERIES = [
    BatterySpec(["matr"], "b033", "2019", "MATR 2019 b033"),
    BatterySpec(["hust"], "9-2", "", "HUST 9-2"),
    BatterySpec(["calce"], "cs2_33", "", "CALCE CS2_33"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 5 normalized capacity trajectories."
    )

    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--phase", choices=["charge", "discharge"], default="discharge")

    parser.add_argument("--initial-window", type=int, default=5)
    parser.add_argument("--initial-method", choices=["median", "mean"], default="median")

    parser.add_argument("--engine", choices=["auto", "fastparquet", "pyarrow"], default="auto")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg", "tiff"], default=["png"])
    parser.add_argument("--dpi", type=int, default=600)

    parser.add_argument("--fig-width", type=float, default=17.5)
    parser.add_argument("--fig-height", type=float, default=10.8)
    parser.add_argument("--line-width", type=float, default=2.6)
    parser.add_argument("--marker-size", type=float, default=3.8)
    parser.add_argument("--font-size", type=float, default=17.0)
    parser.add_argument("--tick-size", type=float, default=14.0)
    parser.add_argument("--legend-size", type=float, default=12.5)
    parser.add_argument("--title-size", type=float, default=17.0)

    parser.add_argument("--y-min", type=float, default=0.65)
    parser.add_argument("--y-max", type=float, default=1.08)
    parser.add_argument("--max-available-list", type=int, default=120)
    parser.add_argument("--show-individual-figures", action="store_true")
    parser.add_argument("--zoom-fraction", type=float, default=0.15,
                        help="Fraction of the cycle-number range shown in zoom figures. Default: 0.15")
    parser.add_argument("--min-zoom-cycles", type=int, default=5,
                        help="Minimum number of cycles shown in zoom figures when available. Default: 5")
    parser.add_argument("--calce-full-max-cycle", type=float, default=23.0,
                        help="Maximum cycle shown in the complete-life CALCE CS2_33 figure. Default: 23.")

    return parser.parse_args()


def configure_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.labelsize": args.font_size,
            "axes.titlesize": args.title_size,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
            "legend.fontsize": args.legend_size,
            "lines.linewidth": args.line_width,
        }
    )


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


def normalize_key(value: Any) -> str:
    """Normalize strings for robust matching: CS2_33, CS2-33 and cs2 33 become cs233."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def vector_to_float_array(value: Any) -> np.ndarray:
    """Convert list-like parquet cells or stringified lists to a 1-D float array."""
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
            text = text.replace("[", "").replace("]", "").replace(";", ",")
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


def scalar_from_row(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return np.nan

    value = row[col]

    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = vector_to_float_array(value)
        return float(arr[-1]) if len(arr) else np.nan

    try:
        return float(value)
    except Exception:
        return np.nan


def infer_dataset_from_path(path: Path) -> str:
    text = normalize_key(path.name)
    for token in ["matr", "hust", "calce", "oxford", "ora", "rwth", "xjtu", "pcoe", "nasa"]:
        if token in text:
            return token.upper()
    return path.stem.split("_")[0].upper()


def find_clean_cycle_files(data_root: Path, phase: str, dataset_tokens: list[str]) -> list[Path]:
    """Find candidate clean-cycle parquet files.

    Memory-safe rule:
        Prefer global files in data/interim/<phase>/.
        Only fall back to data/interim/<dataset>/clean/ if no global file is found.

    This avoids reading the same dataset twice and prevents memory pressure when
    both global and per-dataset clean files are present.
    """
    interim = data_root / "interim"
    norm_tokens = [normalize_key(t) for t in dataset_tokens]
    phase_key = normalize_key(phase)

    def keep(path: Path) -> bool:
        path_text = normalize_key(path)
        phase_ok = phase_key in path_text
        dataset_ok = any(token in path_text for token in norm_tokens)
        return phase_ok and dataset_ok

    # 1) Prefer global phase folder: data/interim/discharge/*.parquet
    global_phase_dir = interim / phase
    global_candidates: list[Path] = []
    if global_phase_dir.exists():
        global_candidates.extend(sorted(global_phase_dir.glob("*clean*.parquet")))
        global_candidates.extend(sorted(global_phase_dir.glob(f"*{phase}*.parquet")))

    global_selected: list[Path] = []
    seen = set()
    for path in global_candidates:
        if path.exists() and keep(path):
            key = str(path).lower()
            if key not in seen:
                global_selected.append(path)
                seen.add(key)

    if global_selected:
        return global_selected

    # 2) Fallback: per-dataset clean folders.
    local_candidates: list[Path] = []
    local_candidates.extend(sorted(interim.glob(f"*/clean/*{phase}*clean*.parquet")))
    local_candidates.extend(sorted(interim.glob("*/clean/*clean*.parquet")))

    local_selected: list[Path] = []
    seen = set()
    for path in local_candidates:
        if path.exists() and keep(path):
            key = str(path).lower()
            if key not in seen:
                local_selected.append(path)
                seen.add(key)

    if not local_selected:
        raise FileNotFoundError(
            f"No parquet files found for phase='{phase}' and dataset_tokens={dataset_tokens} under {interim}"
        )

    return local_selected


def get_parquet_columns(path: Path, engine: str) -> list[str] | None:
    """Return parquet columns without loading the full dataframe."""
    try:
        if engine == "fastparquet":
            from fastparquet import ParquetFile
            return list(ParquetFile(str(path)).columns)
        if engine == "pyarrow":
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(str(path)).schema_arrow.names)
    except Exception:
        return None
    return None


def read_parquet_minimal(path: Path, engine: str) -> pd.DataFrame:
    """Read only the columns needed for Figure 5.

    This figure only needs cycle number, battery identifiers, traceability columns,
    and scalar capacity columns. It does not need voltage/current/time vectors.
    """
    desired_columns = [
        "dataset",
        "battery_id",
        "cell_id_original",
        "cycle_number",
        "cycle_index",
        "cycle_life",
        "remaining_cycles",
        "remaining_cycles_norm",
        "soh",
        "phase",
        "source_file",
        "source_path",
        "policy_readable",
        "barcode",
        "phase_capacity_ah",
        "target_capacity_ah",
        "input_phase_capacity_ah",
        "initial_capacity_ah",
        "target_initial_capacity_ah",
        "eol_criterion",
    ]

    available = get_parquet_columns(path, engine=engine)
    if available is None:
        # Fallback. This can use more memory, but keeps compatibility.
        return pd.read_parquet(path, engine=engine)

    cols = [col for col in desired_columns if col in available]

    # Required minimum for the script.
    required_min = {"battery_id", "cycle_number"}
    if not required_min.issubset(set(cols)):
        return pd.read_parquet(path, engine=engine)

    return pd.read_parquet(path, engine=engine, columns=cols)


def load_phase_dataset(data_root: Path, phase: str, dataset_tokens: list[str], engine: str) -> pd.DataFrame:
    files = find_clean_cycle_files(data_root, phase, dataset_tokens)
    frames: list[pd.DataFrame] = []

    print(f"\nFiles found for phase='{phase}', dataset_tokens={dataset_tokens}:")
    for path in files:
        print(f"  - {path}")

        try:
            df = read_parquet_minimal(path, engine=engine)
        except Exception as exc:
            print(f"    Warning: could not read file: {exc}")
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
            norm_tokens = [normalize_key(t) for t in dataset_tokens]
            mask = df["dataset"].map(lambda x: any(t in normalize_key(x) for t in norm_tokens))
            if mask.any():
                df = df[mask].copy()

        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No readable rows found for phase='{phase}' and dataset_tokens={dataset_tokens}")

    out = pd.concat(frames, ignore_index=True, sort=False)

    required = {"battery_id", "cycle_number"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Missing required columns in loaded data: {sorted(missing)}")

    return out


def filter_battery(df: pd.DataFrame, battery_fragment: str, extra_filter: str, max_available: int) -> pd.DataFrame:
    """Filter selected battery using battery_id and optional traceability columns."""
    target = normalize_key(battery_fragment)

    text_cols = [
        c for c in [
            "battery_id",
            "cell_id_original",
            "source_file",
            "source_path",
            "source_parquet",
            "policy_readable",
            "barcode",
        ]
        if c in df.columns
    ]

    if not text_cols:
        text_cols = ["battery_id"]

    mask = np.zeros(len(df), dtype=bool)
    for col in text_cols:
        mask |= df[col].astype(str).map(lambda x: target in normalize_key(x)).to_numpy()

    out = df.loc[mask].copy()

    if out.empty:
        available = sorted(df["battery_id"].astype(str).unique())[:max_available]
        raise ValueError(
            f"No battery matching fragment '{battery_fragment}' was found. "
            f"First available battery_id values: {available}"
        )

    if extra_filter.strip():
        token = normalize_key(extra_filter)

        extra_mask = np.zeros(len(out), dtype=bool)
        for col in text_cols:
            extra_mask |= out[col].astype(str).map(lambda x: token in normalize_key(x)).to_numpy()

        if extra_mask.any():
            out = out.loc[extra_mask].copy()
        else:
            print(f"    Warning: extra filter '{extra_filter}' did not match. Continuing without it.")

    out["cycle_number"] = pd.to_numeric(out["cycle_number"], errors="coerce")
    out = out[np.isfinite(out["cycle_number"])].sort_values("cycle_number").reset_index(drop=True)

    if out.empty:
        raise ValueError(f"Battery '{battery_fragment}' has no finite cycle_number values after filtering.")

    return out


def get_phase_capacity(row: pd.Series) -> float:
    """Return the scalar phase capacity in Ah.

    Priority:
        1) phase_capacity_ah
        2) target_capacity_ah
        3) input_phase_capacity_ah
        4) last value of capacity_ah vector
    """
    for col in ["phase_capacity_ah", "target_capacity_ah", "input_phase_capacity_ah"]:
        val = scalar_from_row(row, col)
        if np.isfinite(val) and val > 0:
            return val

    arr = vector_to_float_array(row.get("capacity_ah", None))
    if len(arr):
        val = float(arr[-1])
        if np.isfinite(val) and val > 0:
            return val

    return np.nan


def robust_initial_capacity(capacity: pd.Series, window: int, method: str) -> float:
    valid = pd.to_numeric(capacity, errors="coerce")
    valid = valid[np.isfinite(valid) & (valid > 0)]

    if valid.empty:
        return np.nan

    early = valid.iloc[: max(1, int(window))]

    if method == "mean":
        return float(early.mean())

    return float(early.median())


def first_crossing(cycle: np.ndarray, y: np.ndarray, threshold: float) -> float:
    mask = np.isfinite(cycle) & np.isfinite(y)
    cycle = cycle[mask]
    y = y[mask]

    idx = np.where(y <= threshold)[0]
    if len(idx) == 0:
        return np.nan

    return float(cycle[idx[0]])


def first_below_nominal(cycle: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(cycle) & np.isfinite(y)
    cycle = cycle[mask]
    y = y[mask]

    idx = np.where(y < 1.0)[0]
    if len(idx) == 0:
        return np.nan

    return float(cycle[idx[0]])


def prepare_battery_dataframe(
    df_battery: pd.DataFrame,
    initial_window: int,
    initial_method: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    work = df_battery.copy()

    work["cycle_number"] = pd.to_numeric(work["cycle_number"], errors="coerce")
    work = work[np.isfinite(work["cycle_number"])].sort_values("cycle_number").reset_index(drop=True)

    work["phase_capacity_ah_plot"] = work.apply(get_phase_capacity, axis=1).astype(float)
    work = work[np.isfinite(work["phase_capacity_ah_plot"]) & (work["phase_capacity_ah_plot"] > 0)].copy()
    work = work.sort_values("cycle_number").reset_index(drop=True)

    if work.empty:
        raise ValueError("No valid positive phase capacity values found for selected battery.")

    q0 = robust_initial_capacity(
        work["phase_capacity_ah_plot"],
        window=initial_window,
        method=initial_method,
    )

    if not np.isfinite(q0) or q0 <= 0:
        raise ValueError("Could not estimate a valid initial capacity.")

    work["normalized_capacity"] = work["phase_capacity_ah_plot"] / q0

    cycle = work["cycle_number"].to_numpy(dtype=float)
    y = work["normalized_capacity"].to_numpy(dtype=float)

    early_n = min(len(y), max(3, initial_window + 2))
    initial_peak_increase_pct = float((np.nanmax(y[:early_n]) - 1.0) * 100.0) if len(y) else np.nan

    summary = {
        "initial_capacity_estimate_ah": q0,
        "first_below_1p00_cycle": first_below_nominal(cycle, y),
        "first_cross_0p80_cycle": first_crossing(cycle, y, 0.80),
        "first_cross_0p75_cycle": first_crossing(cycle, y, 0.75),
        "first_cross_0p70_cycle": first_crossing(cycle, y, 0.70),
        "last_cycle": float(cycle[-1]) if len(cycle) else np.nan,
        "n_valid_cycles": int(len(work)),
        "initial_peak_increase_pct": initial_peak_increase_pct,
        "min_normalized_capacity": float(np.nanmin(y)) if len(y) else np.nan,
        "max_normalized_capacity": float(np.nanmax(y)) if len(y) else np.nan,
        "min_phase_capacity_ah": float(np.nanmin(work["phase_capacity_ah_plot"].to_numpy(dtype=float))) if len(work) else np.nan,
        "max_phase_capacity_ah": float(np.nanmax(work["phase_capacity_ah_plot"].to_numpy(dtype=float))) if len(work) else np.nan,
    }

    return work, summary


def format_cycle(value: float) -> str:
    if np.isfinite(value):
        return str(int(round(float(value))))
    return "n/a"


def annotate_crossing(ax: plt.Axes, x_cross: float, threshold: float, color: str, label: str) -> None:
    if not np.isfinite(x_cross):
        return

    ax.axvline(x_cross, color=color, linestyle=":", linewidth=1.3, alpha=0.85)
    ax.annotate(
        f"{label} {int(round(x_cross))}",
        xy=(x_cross, threshold),
        xytext=(5, 7),
        textcoords="offset points",
        fontsize=9,
        color=color,
        ha="left",
        va="bottom",
    )



def get_nominal_capacity_for_plot(df_plot: pd.DataFrame, q0: float) -> float:
    """Return nominal capacity in Ah for plotting reference lines.

    Priority:
        1) nominal_capacity_ah column, if available and positive;
        2) input_nominal_capacity_ah column, if available and positive;
        3) target_nominal_capacity_ah column, if available and positive;
        4) robust initial capacity Q0 as a conservative fallback.
    """
    candidate_columns = [
        "nominal_capacity_ah",
        "input_nominal_capacity_ah",
        "target_nominal_capacity_ah",
    ]

    for col in candidate_columns:
        if col not in df_plot.columns:
            continue

        values = pd.to_numeric(df_plot[col], errors="coerce")
        values = values[np.isfinite(values) & (values > 0)]
        if not values.empty:
            return float(values.median())

    return float(q0) if np.isfinite(q0) and q0 > 0 else np.nan


def make_zoom_mask_by_fraction(cycle: np.ndarray, fraction: float = 0.15, min_cycles: int = 5) -> np.ndarray:
    """Keep the first fraction of the available cycle range, with at least min_cycles rows.

    The mask is based on sorted cycle_number values. It uses the cycle-number
    range for the percentage cutoff and then expands the mask if fewer than
    min_cycles observations were selected.
    """
    cycle = np.asarray(cycle, dtype=float)
    finite = np.isfinite(cycle)

    if not finite.any():
        return finite

    cmin = float(np.nanmin(cycle[finite]))
    cmax = float(np.nanmax(cycle[finite]))

    if cmax <= cmin:
        base_mask = finite.copy()
    else:
        cutoff = cmin + float(fraction) * (cmax - cmin)
        base_mask = finite & (cycle <= cutoff)

    finite_indices = np.where(finite)[0]
    selected_indices = np.where(base_mask)[0]

    if len(selected_indices) >= min_cycles or len(finite_indices) <= len(selected_indices):
        return base_mask

    # Ensure at least min_cycles first valid rows are included.
    base_mask = base_mask.copy()
    n = min(int(min_cycles), len(finite_indices))
    base_mask[finite_indices[:n]] = True
    return base_mask



def is_calce_battery(spec: BatterySpec) -> bool:
    """Return True for the selected CALCE cell."""
    label_key = normalize_key(spec.label)
    fragment_key = normalize_key(spec.battery_fragment)
    dataset_key = "".join(normalize_key(t) for t in spec.dataset_tokens)
    return ("calce" in label_key) or ("calce" in dataset_key) or ("cs233" in label_key) or ("cs233" in fragment_key)


def build_plot_arrays(
    df_plot: pd.DataFrame,
    zoom: bool,
    zoom_fraction: float,
    min_zoom_cycles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return x and phase-capacity arrays for full or zoom plotting."""
    x_full = df_plot["cycle_number"].to_numpy(dtype=float)
    q_full = df_plot["phase_capacity_ah_plot"].to_numpy(dtype=float)

    valid = np.isfinite(x_full) & np.isfinite(q_full)

    if zoom:
        keep = valid & make_zoom_mask_by_fraction(
            x_full,
            fraction=zoom_fraction,
            min_cycles=min_zoom_cycles,
        )
    else:
        keep = valid

    x = x_full[keep]
    q = q_full[keep]

    if len(x) == 0:
        x = x_full[valid]
        q = q_full[valid]

    return x, q



def get_capacity_plot_data(
    spec: BatterySpec,
    df_plot: pd.DataFrame,
    summary: dict[str, float],
    args: argparse.Namespace,
    zoom: bool,
) -> dict[str, Any]:
    """Build all arrays and scalar values needed to draw one panel."""
    x, q = build_plot_arrays(
        df_plot,
        zoom=zoom,
        zoom_fraction=args.zoom_fraction,
        min_zoom_cycles=args.min_zoom_cycles,
    )

    # Specific rule requested for CALCE CS2_33:
    # in the complete-life panel only, show the trajectory only up to cycle 23.
    if (not zoom) and is_calce_battery(spec):
        keep_calce = np.isfinite(x) & (x <= args.calce_full_max_cycle)
        if keep_calce.any():
            x = x[keep_calce]
            q = q[keep_calce]

    q0 = summary.get("initial_capacity_estimate_ah", np.nan)
    q_nom = get_nominal_capacity_for_plot(df_plot, q0=q0)

    q_plot_valid = np.asarray(q, dtype=float)
    q_plot_valid = q_plot_valid[np.isfinite(q_plot_valid) & (q_plot_valid > 0)]

    if len(q_plot_valid):
        q_max = float(np.nanmax(q_plot_valid))
    else:
        q_series = pd.to_numeric(df_plot.get("phase_capacity_ah_plot", pd.Series(dtype=float)), errors="coerce")
        q_series = q_series[np.isfinite(q_series) & (q_series > 0)]
        q_max = float(q_series.max()) if not q_series.empty else np.nan

    eol80_cycle = summary.get("first_cross_0p80_cycle", np.nan)

    n0 = min(len(df_plot), max(1, int(args.initial_window)))
    x0 = np.array([], dtype=float)
    q0_points = np.array([], dtype=float)

    if n0 > 0:
        x0_full = df_plot["cycle_number"].to_numpy(dtype=float)[:n0]
        q0_full = df_plot["phase_capacity_ah_plot"].to_numpy(dtype=float)[:n0]

        if zoom:
            zoom_mask_full = make_zoom_mask_by_fraction(
                df_plot["cycle_number"].to_numpy(dtype=float),
                fraction=args.zoom_fraction,
                min_cycles=args.min_zoom_cycles,
            )[:n0]
        else:
            zoom_mask_full = np.ones(n0, dtype=bool)

        keep0 = zoom_mask_full & np.isfinite(x0_full) & np.isfinite(q0_full)
        x0 = x0_full[keep0]
        q0_points = q0_full[keep0]

    return {
        "x": x,
        "q": q,
        "x0": x0,
        "q0_points": q0_points,
        "q0": q0,
        "q_nom": q_nom,
        "q_max": q_max,
        "eol80_cycle": eol80_cycle,
    }


def draw_capacity_panel(
    ax: plt.Axes,
    spec: BatterySpec,
    df_plot: pd.DataFrame,
    summary: dict[str, float],
    args: argparse.Namespace,
    zoom: bool,
    panel_label: str,
) -> None:
    """Draw one capacity panel on a provided axis."""
    data = get_capacity_plot_data(
        spec=spec,
        df_plot=df_plot,
        summary=summary,
        args=args,
        zoom=zoom,
    )

    x = data["x"]
    q = data["q"]
    x0 = data["x0"]
    q0_points = data["q0_points"]
    q0 = data["q0"]
    q_nom = data["q_nom"]
    q_max = data["q_max"]
    eol80_cycle = data["eol80_cycle"]

    ax.plot(
        x,
        q,
        linewidth=args.line_width,
        marker="o",
        markersize=args.marker_size,
        label=spec.label,
    )

    # Only the two requested horizontal lines.
    if np.isfinite(q_nom):
        ax.axhline(
            q_nom,
            color="gray",
            linestyle="-",
            linewidth=1.25,
            alpha=0.78,
            label="1.0 x nominal capacity",
        )
        ax.axhline(
            0.8 * q_nom,
            color="tab:red",
            linestyle="--",
            linewidth=1.55,
            alpha=0.92,
            label="0.8 x nominal capacity",
        )

    if len(x0):
        ax.scatter(
            x0,
            q0_points,
            s=max(24, args.marker_size * 10),
            zorder=4,
            label="Initial-capacity window",
        )

    suffix = "Zoom: initial 15%" if zoom else "Complete trajectory"
    ax.set_title(f"{panel_label} {spec.label}\n{suffix}", fontweight="bold", loc="left")
    ax.set_xlabel("Cycle number")
    ax.grid(True, alpha=0.25)

    # Dynamic y-limits based on plotted data and reference lines.
    y_candidates = []
    if len(q):
        y_candidates.extend(q[np.isfinite(q)].tolist())
    if np.isfinite(q_nom):
        y_candidates.extend([q_nom, 0.8 * q_nom])
    if np.isfinite(q0):
        y_candidates.append(q0)
    if np.isfinite(q_max):
        y_candidates.append(q_max)

    if y_candidates:
        y_arr = np.asarray(y_candidates, dtype=float)
        y_arr = y_arr[np.isfinite(y_arr)]
        if len(y_arr):
            ymin = float(np.nanmin(y_arr))
            ymax = float(np.nanmax(y_arr))
            margin = 0.07 * max(ymax - ymin, 1e-9)
            ax.set_ylim(ymin - margin, ymax + margin)

    if len(x):
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        if xmax > xmin:
            x_margin = 0.015 * max(xmax - xmin, 1e-9)
            ax.set_xlim(xmin - x_margin, xmax + x_margin)
        else:
            ax.set_xlim(xmin - 1, xmax + 1)

    qmax_text = f"{q_max:.3f} Ah" if np.isfinite(q_max) else "not available"
    legend_title = (
        f"Q0 = {q0:.3f} Ah\n"
        f"Qmax = {qmax_text}\n"
        f"EOL = {format_cycle(eol80_cycle)}"
    )

    # One legend block per panel, containing curve, reference lines, marker, and summary.
    legend = ax.legend(
        loc="best",
        frameon=True,
        fontsize=args.legend_size,
        title=legend_title,
        title_fontsize=args.legend_size,
        borderpad=0.8,
        labelspacing=0.45,
        handlelength=2.4,
    )
    legend.get_frame().set_alpha(0.88)
    legend.get_frame().set_edgecolor("0.75")


def plot_one_battery_capacity_figure(
    spec: BatterySpec,
    df_plot: pd.DataFrame,
    summary: dict[str, float],
    output_base: Path,
    args: argparse.Namespace,
    zoom: bool,
) -> None:
    """Compatibility function for optional single-panel figures."""
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(7.8, 5.6),
        constrained_layout=True,
    )

    draw_capacity_panel(
        ax=ax,
        spec=spec,
        df_plot=df_plot,
        summary=summary,
        args=args,
        zoom=zoom,
        panel_label="",
    )
    ax.set_ylabel("Phase capacity (Ah)")

    output_base.parent.mkdir(parents=True, exist_ok=True)

    for fmt in args.formats:
        out = output_base.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


def make_combined_figure(
    prepared: list[tuple[BatterySpec, pd.DataFrame, dict[str, float]]],
    output_base: Path,
    args: argparse.Namespace,
) -> None:
    """Generate one 3-column x 2-row figure.

    Row 1: complete trajectories.
    Row 2: initial 15% zoom views.
    """
    n_cols = len(prepared)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(args.fig_width, args.fig_height),
        constrained_layout=True,
    )

    if n_cols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    panel_labels_top = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    panel_labels_bottom = ["(d)", "(e)", "(f)", "(g)", "(h)"]

    for col, (spec, df_plot, summary) in enumerate(prepared):
        draw_capacity_panel(
            ax=axes[0, col],
            spec=spec,
            df_plot=df_plot,
            summary=summary,
            args=args,
            zoom=False,
            panel_label=panel_labels_top[col],
        )
        draw_capacity_panel(
            ax=axes[1, col],
            spec=spec,
            df_plot=df_plot,
            summary=summary,
            args=args,
            zoom=True,
            panel_label=panel_labels_bottom[col],
        )

        if col == 0:
            axes[0, col].set_ylabel("Phase capacity (Ah)")
            axes[1, col].set_ylabel("Phase capacity (Ah)")
        else:
            axes[0, col].set_ylabel("")
            axes[1, col].set_ylabel("")

    output_base.parent.mkdir(parents=True, exist_ok=True)

    for fmt in args.formats:
        out = output_base.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


def make_individual_figure(
    spec: BatterySpec,
    df_plot: pd.DataFrame,
    summary: dict[str, float],
    output_base: Path,
    args: argparse.Namespace,
) -> None:
    """Compatibility wrapper for optional individual figures."""
    plot_one_battery_capacity_figure(
        spec=spec,
        df_plot=df_plot,
        summary=summary,
        output_base=output_base,
        args=args,
        zoom=False,
    )


def main() -> None:
    args = parse_args()
    configure_matplotlib(args)

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    engine = choose_engine(args.engine)

    prepared: list[tuple[BatterySpec, pd.DataFrame, dict[str, float]]] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in DEFAULT_BATTERIES:
        print("-" * 80)
        print(f"Processing {spec.label}")

        df = load_phase_dataset(
            data_root=data_root,
            phase=args.phase,
            dataset_tokens=spec.dataset_tokens,
            engine=engine,
        )

        df_battery = filter_battery(
            df,
            battery_fragment=spec.battery_fragment,
            extra_filter=spec.extra_filter,
            max_available=args.max_available_list,
        )

        df_plot, summary = prepare_battery_dataframe(
            df_battery,
            initial_window=args.initial_window,
            initial_method=args.initial_method,
        )

        prepared.append((spec, df_plot, summary))

        summary_rows.append(
            {
                "label": spec.label,
                "dataset_tokens": ",".join(spec.dataset_tokens),
                "battery_fragment": spec.battery_fragment,
                "extra_filter": spec.extra_filter,
                "phase": args.phase,
                "initial_method": args.initial_method,
                "initial_window": args.initial_window,
                **summary,
            }
        )

    combined_base = output_root / "figure5_capacity_panel_3x2_complete_and_zoom"
    make_combined_figure(prepared, combined_base, args=args)

    if args.show_individual_figures:
        for spec, df_plot, summary in prepared:
            safe_name = normalize_key(spec.label)
            out_base = output_root / f"figure5_{safe_name}_normalized_capacity"
            make_individual_figure(spec, df_plot, summary, out_base, args=args)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / "figure5_selected_batteries_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("-" * 80)
    print("Figure 5 generation finished: 3x2 panel generated, with complete-life row and initial-15pct zoom row. Legend now uses EOL.")
    print(f"Summary table: {summary_path}")

    for row in summary_rows:
        print(
            f"{row['label']}: "
            f"Q0={row['initial_capacity_estimate_ah']:.4f} Ah, "
            f"first<1.00={row['first_below_1p00_cycle']}, "
            f"0.80={row['first_cross_0p80_cycle']}, "
            f"0.75={row['first_cross_0p75_cycle']}, "
            f"0.70={row['first_cross_0p70_cycle']}, "
            f"last={row['last_cycle']}, "
            f"initial_peak_increase_pct={row['initial_peak_increase_pct']:.3f}%"
        )


if __name__ == "__main__":
    main()
