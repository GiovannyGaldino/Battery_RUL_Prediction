"""
audit_essential_battery_curves.py

Essential visual audit for lithium-ion battery RUL datasets.

Updated v4:
- Capacity normalized plots use real cycle index/number on the x-axis, not normalized cycle life.
- Only capacity is normalized.
- Only the y-axis is limited from 0.6 to 1.1.
- Raw curves remain unchanged.
- Output folders do not split charge and discharge into separate subfolders; phase remains in file names.
- If feature_level exists, only feature_level == 'phase' rows are used; stage-level rows are excluded.

Main outputs:
1) capacity_normalized_vs_cycle_life
   - one plot for each battery
   - one plot with all batteries from each dataset and phase
   - fixed x-axis limits from 0.6 to 1.1

2) raw_curves
   - one figure per battery
   - 5 vertically aligned plots sharing the same time axis:
       voltage, current, temperature, capacity, power
   - 3 cycles per battery: first, middle, last

Design decisions:
- Charge and discharge are audited separately, but their output images are saved
  together in the same dataset folders. The phase remains explicit in the filenames.
- Capacity is prioritized over energy because it is the more consolidated
  degradation indicator in the RUL literature.
- The script does not modify the input files.
- The script does not read data/raw.
- The default input is data/interim/charge and data/interim/discharge.

Default project root:
<project-root>
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PROJECT_ROOT = "."
DEFAULT_DATA_ROOT = "data"

PHASES = ("charge", "discharge")

SCALAR_BASE_COLUMNS = [
    "dataset",
    "battery_id",
    "phase",
    "cycle_number",
    "cycle_index",
    "cycle_life",
    "remaining_cycles",
    "remaining_cycles_norm",
    "phase_capacity_ah",
    "initial_capacity_ah",
    "soh",
    "is_valid_cycle",
    "is_model_usable",
    "exclude_reason",
    "feature_level",
]

VECTOR_BASE_COLUMNS = [
    "dataset",
    "battery_id",
    "phase",
    "cycle_number",
    "cycle_index",
    "cycle_life",
    "time_s",
    "voltage_v",
    "current_a",
    "temperature_c",
    "capacity_ah",
    "power_w",
    "n_points",
    "feature_level",
]


@dataclass
class InputFile:
    dataset: str
    phase: str
    path: str
    n_rows: int
    n_columns: int
    status: str
    error: str


@dataclass
class CapacitySummary:
    dataset: str
    phase: str
    battery_id: str
    n_cycles: int
    first_cycle: float
    last_cycle: float
    initial_capacity_ah: float
    first_capacity_ah: float
    last_capacity_ah: float
    min_capacity_norm: float
    max_capacity_norm: float
    final_capacity_norm: float
    capacity_drop_pct: float
    output_file: str


@dataclass
class RawSelection:
    dataset: str
    phase: str
    battery_id: str
    selection_position: str
    row_id: int
    cycle_number: float
    cycle_index: float
    n_cycles_in_battery: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate essential visual audit plots for battery clean-cycle parquet files."
    )

    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Project root. Default: <project-root>",
    )

    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Data root. Default: <project-root>\\data",
    )

    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Output root. Default: "
            "<project-root>\\reports\\auditoria_essencial\\run_YYYYMMDD_HHMMSS"
        ),
    )

    parser.add_argument(
        "--input-mode",
        choices=["global", "per_dataset"],
        default="global",
        help=(
            "global: read data/interim/charge/*.parquet and data/interim/discharge/*.parquet. "
            "per_dataset: read data/interim/<dataset>/clean/*_clean_cycles.parquet. "
            "Default: global."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset filter, e.g. --datasets calce hust matr.",
    )

    parser.add_argument(
        "--phases",
        nargs="*",
        choices=["charge", "discharge"],
        default=list(PHASES),
        help="Phases to audit. Default: charge discharge.",
    )

    parser.add_argument(
        "--skip-capacity-plots",
        action="store_true",
        help="Skip capacity normalized plots.",
    )

    parser.add_argument(
        "--skip-raw-curves",
        action="store_true",
        help="Skip raw curve plots.",
    )

    parser.add_argument(
        "--include-stage-level",
        action="store_true",
        help=(
            "Include stage-level rows when feature_level exists. "
            "Default behavior is to keep only complete phase rows."
        ),
    )

    parser.add_argument(
        "--raw-cycles-per-battery",
        type=int,
        default=3,
        help="Number of cycles for raw curves. Default: 3. For this project, use 3.",
    )

    parser.add_argument(
        "--initial-capacity-method",
        choices=["early_median", "first", "input"],
        default="early_median",
        help=(
            "Method for normalization. early_median uses the median of the first N cycles; "
            "first uses the first valid capacity; input uses initial_capacity_ah if available. "
            "Default: early_median."
        ),
    )

    parser.add_argument(
        "--initial-window",
        type=int,
        default=5,
        help="Number of early cycles used by early_median. Default: 5.",
    )

    parser.add_argument(
        "--capacity-x",
        choices=["cycle_life_fraction", "cycle_number", "cycle_index"],
        default="cycle_index",
        help=(
            "X axis for capacity plots. Use real cycle index/number, not normalized cycle life. "
            "Default: cycle_index."
        ),
    )

    parser.add_argument(
        "--max-batteries",
        type=int,
        default=None,
        help="Optional limit of batteries per dataset/phase, useful for testing.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure resolution. Default: 160.",
    )

    parser.add_argument(
        "--fig-format",
        choices=["png", "pdf"],
        default="png",
        help="Figure format. Default: png.",
    )

    parser.add_argument(
        "--engine",
        choices=["auto", "fastparquet", "pyarrow"],
        default="auto",
        help="Parquet engine. Default: auto, preferring fastparquet.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing figures in the output folder.",
    )

    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(value: Any, max_len: int = 120) -> str:
    text = str(value)
    text = text.strip()
    text = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._-")
    if not text:
        text = "unknown"
    return text[:max_len]


def infer_dataset_from_path(path: Path) -> str:
    name = path.name.lower()

    for phase in PHASES:
        token = f"_{phase}_clean_cycles"
        if token in name:
            return name.split(token)[0].lower()

    if "_clean_cycles" in name:
        return name.split("_clean_cycles")[0].lower()

    return path.stem.lower()


def infer_phase_from_path(path: Path) -> str:
    lower = str(path).lower().replace("\\", "/")
    name = path.name.lower()

    if "/charge/" in lower or "_charge_" in name:
        return "charge"
    if "/discharge/" in lower or "_discharge_" in name:
        return "discharge"

    return "unknown"


def get_parquet_columns(path: Path, engine: str = "auto") -> list[str]:
    if engine in ("auto", "fastparquet"):
        try:
            from fastparquet import ParquetFile

            pf = ParquetFile(str(path))
            return list(pf.columns)
        except Exception:
            if engine == "fastparquet":
                raise

    if engine in ("auto", "pyarrow"):
        try:
            import pyarrow.parquet as pq

            return list(pq.ParquetFile(str(path)).schema.names)
        except Exception:
            if engine == "pyarrow":
                raise

    # Final fallback: read no rows if possible is not generally supported.
    # Read only metadata is unavailable, so raise a clear error.
    raise RuntimeError(
        f"Could not read parquet metadata for {path}. "
        "Install fastparquet or pyarrow in the active environment."
    )


def choose_engine(engine: str = "auto") -> str:
    if engine == "fastparquet":
        return "fastparquet"
    if engine == "pyarrow":
        return "pyarrow"

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

    raise RuntimeError("Neither fastparquet nor pyarrow is available.")


def read_parquet_selected_columns(
    path: Path,
    columns: list[str],
    engine: str,
) -> pd.DataFrame:
    available = get_parquet_columns(path, engine=engine)
    selected = [c for c in columns if c in available]

    if not selected:
        raise RuntimeError(f"No requested columns were found in {path}")

    real_engine = choose_engine(engine)
    return pd.read_parquet(path, columns=selected, engine=real_engine)


def get_parquet_n_rows(path: Path, engine: str = "auto") -> int:
    if engine in ("auto", "fastparquet"):
        try:
            from fastparquet import ParquetFile

            return int(ParquetFile(str(path)).count())
        except Exception:
            if engine == "fastparquet":
                raise

    if engine in ("auto", "pyarrow"):
        try:
            import pyarrow.parquet as pq

            meta = pq.ParquetFile(str(path)).metadata
            return int(meta.num_rows)
        except Exception:
            if engine == "pyarrow":
                raise

    return -1


def find_input_files(
    data_root: Path,
    input_mode: str,
    phases: list[str],
    datasets: Optional[list[str]],
    engine: str,
) -> list[Path]:
    interim = data_root / "interim"
    files: list[Path] = []

    if input_mode == "global":
        for phase in phases:
            phase_dir = interim / phase
            if not phase_dir.exists():
                continue

            for path in sorted(phase_dir.glob("*.parquet")):
                lower = path.name.lower()
                if "_clean_cycles" not in lower:
                    continue
                files.append(path)

    elif input_mode == "per_dataset":
        for clean_dir in sorted(interim.glob("*/clean")):
            for phase in phases:
                pattern = f"*_{phase}_clean_cycles.parquet"
                files.extend(sorted(clean_dir.glob(pattern)))

    if datasets:
        dataset_filter = {d.lower() for d in datasets}
        files = [
            p for p in files
            if infer_dataset_from_path(p).lower() in dataset_filter
        ]

    # Remove duplicates while preserving order.
    seen = set()
    unique_files = []
    for p in files:
        key = str(p.resolve()).lower()
        if key not in seen:
            unique_files.append(p)
            seen.add(key)

    return unique_files


def to_numeric_scalar(value: Any) -> float:
    try:
        if value is None:
            return np.nan
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = to_numeric_vector(value)
            if arr.size == 0:
                return np.nan
            return float(arr[-1])
        return float(value)
    except Exception:
        return np.nan


def to_numeric_vector(value: Any) -> np.ndarray:
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
            parsed = ast.literal_eval(text)
            arr = np.asarray(parsed)
        except Exception:
            # Fallback for strings like "1, 2, 3" or "1 2 3"
            text = text.replace(";", ",")
            parts = re.split(r"[,\s]+", text)
            parts = [p for p in parts if p]
            arr = np.asarray(parts)
    else:
        try:
            if pd.isna(value):
                return np.array([], dtype=float)
        except Exception:
            pass
        arr = np.asarray([value])

    try:
        arr = np.asarray(arr, dtype=float).reshape(-1)
    except Exception:
        converted = []
        for item in np.asarray(arr, dtype=object).reshape(-1):
            try:
                converted.append(float(item))
            except Exception:
                converted.append(np.nan)
        arr = np.asarray(converted, dtype=float)

    return arr


def finite_or_nan(value: Any) -> float:
    v = to_numeric_scalar(value)
    return float(v) if np.isfinite(v) else np.nan


def sort_cycle_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = []
    if "cycle_number" in df.columns:
        sort_cols.append("cycle_number")
    if "cycle_index" in df.columns:
        sort_cols.append("cycle_index")
    if not sort_cols:
        return df

    out = df.copy()
    for c in sort_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out.sort_values(sort_cols, kind="mergesort")


def filter_complete_phase_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only full-phase rows when a hierarchical feature-level column exists.

    The feature extraction v3 pipeline can create rows with:
    - feature_level == "phase": complete phase vector;
    - feature_level == "stage": internal subsegment separated by temporal gaps.

    This audit must use only complete phase trajectories. Clean-cycle files usually
    do not contain feature_level, so they are returned unchanged.
    """
    if "feature_level" not in df.columns:
        return df

    level = df["feature_level"].astype(str).str.strip().str.lower()
    return df[level.eq("phase")].copy()


def get_phase_capacity_series(df: pd.DataFrame) -> pd.Series:
    if "phase_capacity_ah" in df.columns:
        cap = pd.to_numeric(df["phase_capacity_ah"], errors="coerce")
        if cap.notna().any():
            return cap

    if "capacity_ah" in df.columns:
        return df["capacity_ah"].apply(finite_or_nan)

    return pd.Series(np.nan, index=df.index, dtype=float)


def compute_initial_capacity(
    cap: pd.Series,
    input_initial: Optional[pd.Series],
    method: str,
    initial_window: int,
) -> float:
    cap = pd.to_numeric(cap, errors="coerce")
    valid_cap = cap[np.isfinite(cap) & (cap > 0)]

    if method == "input" and input_initial is not None:
        initial_values = pd.to_numeric(input_initial, errors="coerce")
        initial_values = initial_values[np.isfinite(initial_values) & (initial_values > 0)]
        if len(initial_values) > 0:
            return float(initial_values.iloc[0])

    if len(valid_cap) == 0:
        return np.nan

    if method == "first":
        return float(valid_cap.iloc[0])

    # early_median
    window = max(int(initial_window), 1)
    early = valid_cap.iloc[:window]
    if len(early) == 0:
        return float(valid_cap.iloc[0])
    return float(np.nanmedian(early.to_numpy(dtype=float)))


def compute_x_axis(df: pd.DataFrame, capacity_x: str) -> tuple[np.ndarray, str]:
    """
    Return the x-axis for capacity degradation plots.

    Important:
    - Capacity may be normalized on the y-axis.
    - Cycle life must not be normalized on the x-axis by default.
    - The preferred x-axis is cycle_index because the project schema defines it
      as sequential within each battery and phase, starting at 1.
    """

    if capacity_x == "cycle_index" and "cycle_index" in df.columns:
        x = pd.to_numeric(df["cycle_index"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).any():
            return x, "Cycle index"

    if capacity_x == "cycle_number" and "cycle_number" in df.columns:
        x = pd.to_numeric(df["cycle_number"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).any():
            return x, "Cycle number"

    # Normalized cycle life is available only if explicitly requested.
    if capacity_x == "cycle_life_fraction":
        if "cycle_number" in df.columns and "cycle_life" in df.columns:
            cycle = pd.to_numeric(df["cycle_number"], errors="coerce").to_numpy(dtype=float)
            life = pd.to_numeric(df["cycle_life"], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                x = cycle / life
            if np.isfinite(x).any():
                return x, "Normalized cycle life (cycle_number / cycle_life)"

        if "cycle_index" in df.columns and "cycle_life" in df.columns:
            cycle = pd.to_numeric(df["cycle_index"], errors="coerce").to_numpy(dtype=float)
            life = pd.to_numeric(df["cycle_life"], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                x = cycle / life
            if np.isfinite(x).any():
                return x, "Normalized cycle life (cycle_index / cycle_life)"

    # Non-normalized fallbacks.
    if "cycle_index" in df.columns:
        x = pd.to_numeric(df["cycle_index"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).any():
            return x, "Cycle index"

    if "cycle_number" in df.columns:
        x = pd.to_numeric(df["cycle_number"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).any():
            return x, "Cycle number"

    # Last fallback starts at 1, not 0.
    x = np.arange(1, len(df) + 1, dtype=float)
    return x, "Ordered cycle position"


def save_capacity_plot_one_battery(
    group: pd.DataFrame,
    dataset: str,
    phase: str,
    battery_id: str,
    output_file: Path,
    args: argparse.Namespace,
) -> Optional[CapacitySummary]:
    if output_file.exists() and not args.overwrite:
        pass

    group = sort_cycle_dataframe(group)
    cap = get_phase_capacity_series(group)

    initial_input = group["initial_capacity_ah"] if "initial_capacity_ah" in group.columns else None
    initial = compute_initial_capacity(
        cap=cap,
        input_initial=initial_input,
        method=args.initial_capacity_method,
        initial_window=args.initial_window,
    )

    if not np.isfinite(initial) or initial <= 0:
        return None

    cap_norm = cap.astype(float) / initial
    valid_mask = np.isfinite(cap_norm)

    if valid_mask.sum() < 2:
        return None

    x, xlabel = compute_x_axis(group, args.capacity_x)

    x = np.asarray(x, dtype=float)
    y = cap_norm.to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return None

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not output_file.exists() or args.overwrite:
        plt.figure(figsize=(8.5, 5.0))
        plt.plot(x[valid], y[valid], marker=".", linewidth=1.2, markersize=3)
        plt.axhline(0.80, linestyle="--", linewidth=1.0, label="80% capacity")
        plt.ylim(0.6, 1.1)
        plt.xlabel(xlabel)
        plt.ylabel("Normalized capacity")
        plt.title(f"{dataset} | {phase} | {battery_id}\nNormalized capacity vs cycle")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_file, dpi=args.dpi)
        plt.close()

    sorted_group = group.reset_index(drop=True)
    cycle_values = (
        pd.to_numeric(sorted_group["cycle_number"], errors="coerce")
        if "cycle_number" in sorted_group.columns
        else pd.Series(np.arange(len(sorted_group)))
    )

    valid_cap = cap[valid_mask]

    return CapacitySummary(
        dataset=dataset,
        phase=phase,
        battery_id=str(battery_id),
        n_cycles=int(valid_mask.sum()),
        first_cycle=float(cycle_values.iloc[0]) if len(cycle_values) else np.nan,
        last_cycle=float(cycle_values.iloc[-1]) if len(cycle_values) else np.nan,
        initial_capacity_ah=float(initial),
        first_capacity_ah=float(valid_cap.iloc[0]) if len(valid_cap) else np.nan,
        last_capacity_ah=float(valid_cap.iloc[-1]) if len(valid_cap) else np.nan,
        min_capacity_norm=float(np.nanmin(y[valid])),
        max_capacity_norm=float(np.nanmax(y[valid])),
        final_capacity_norm=float(y[valid][-1]),
        capacity_drop_pct=float((1.0 - y[valid][-1]) * 100.0),
        output_file=str(output_file),
    )


def save_capacity_plot_all_batteries(
    df: pd.DataFrame,
    dataset: str,
    phase: str,
    output_file: Path,
    args: argparse.Namespace,
) -> None:
    if output_file.exists() and not args.overwrite:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9.0, 5.5))

    n_plotted = 0

    for battery_id, group in df.groupby("battery_id", sort=True):
        group = sort_cycle_dataframe(group)

        cap = get_phase_capacity_series(group)
        initial_input = group["initial_capacity_ah"] if "initial_capacity_ah" in group.columns else None
        initial = compute_initial_capacity(
            cap=cap,
            input_initial=initial_input,
            method=args.initial_capacity_method,
            initial_window=args.initial_window,
        )

        if not np.isfinite(initial) or initial <= 0:
            continue

        x, xlabel = compute_x_axis(group, args.capacity_x)
        y = cap.astype(float).to_numpy(dtype=float) / initial

        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            continue

        plt.plot(x[valid], y[valid], linewidth=0.9, alpha=0.45)
        n_plotted += 1

    plt.axhline(0.80, linestyle="--", linewidth=1.0, label="80% capacity")
    plt.ylim(0.6, 1.1)
    plt.xlabel(xlabel if n_plotted else "Cycle")
    plt.ylabel("Normalized capacity")
    plt.title(
        f"{dataset} | {phase} | all batteries\n"
        f"Normalized capacity vs cycle ({n_plotted} batteries)"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file, dpi=args.dpi)
    plt.close()


def select_raw_cycle_rows(
    scalar_df: pd.DataFrame,
    dataset: str,
    phase: str,
    raw_cycles_per_battery: int,
    max_batteries: Optional[int],
) -> list[RawSelection]:
    selections: list[RawSelection] = []

    battery_ids = sorted(scalar_df["battery_id"].dropna().astype(str).unique())
    if max_batteries is not None:
        battery_ids = battery_ids[: max_batteries]

    for battery_id in battery_ids:
        group = scalar_df[scalar_df["battery_id"].astype(str) == str(battery_id)].copy()
        group = sort_cycle_dataframe(group)

        if len(group) == 0:
            continue

        n = len(group)
        k = min(max(raw_cycles_per_battery, 1), n)
        idx = np.linspace(0, n - 1, k).round().astype(int)
        idx = np.unique(idx)

        labels = []
        if len(idx) == 1:
            labels = ["selected"]
        elif len(idx) == 2:
            labels = ["first", "last"]
        else:
            labels = ["first"] + [f"middle_{i}" for i in range(1, len(idx) - 1)] + ["last"]
            if len(idx) == 3:
                labels = ["first", "middle", "last"]

        group_reset = group.reset_index(drop=True)

        for label, i in zip(labels, idx):
            row = group_reset.iloc[int(i)]
            selections.append(
                RawSelection(
                    dataset=dataset,
                    phase=phase,
                    battery_id=str(battery_id),
                    selection_position=label,
                    row_id=int(row["__row_id"]),
                    cycle_number=float(row.get("cycle_number", np.nan))
                    if pd.notna(row.get("cycle_number", np.nan))
                    else np.nan,
                    cycle_index=float(row.get("cycle_index", np.nan))
                    if pd.notna(row.get("cycle_index", np.nan))
                    else np.nan,
                    n_cycles_in_battery=int(n),
                )
            )

    return selections


def align_vectors(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    lengths = [len(v) for v in vectors.values() if len(v) > 0]
    if not lengths:
        return vectors

    n = min(lengths)
    out = {}
    for key, arr in vectors.items():
        if len(arr) == 0:
            out[key] = np.full(n, np.nan)
        else:
            out[key] = arr[:n]
    return out


def prepare_raw_vectors(row: pd.Series) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    time = to_numeric_vector(row.get("time_s", None))
    voltage = to_numeric_vector(row.get("voltage_v", None))
    current = to_numeric_vector(row.get("current_a", None))
    temperature = to_numeric_vector(row.get("temperature_c", None))
    capacity = to_numeric_vector(row.get("capacity_ah", None))
    power = to_numeric_vector(row.get("power_w", None))

    vectors = {
        "time": time,
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "capacity": capacity,
        "power": power,
    }

    vectors = align_vectors(vectors)

    if len(vectors["time"]) == 0:
        n = max((len(v) for v in vectors.values()), default=0)
        if n <= 0:
            return np.array([], dtype=float), vectors, False
        vectors["time"] = np.arange(n, dtype=float)

    time = vectors["time"].astype(float)

    finite_time = np.isfinite(time)
    if finite_time.sum() >= 1:
        first = time[finite_time][0]
        time = time - first

    # Sort by time if needed.
    if np.isfinite(time).sum() >= 2:
        order = np.argsort(np.where(np.isfinite(time), time, np.inf))
        time = time[order]
        for key in vectors:
            vectors[key] = vectors[key][order]

    # Reconstruct power if invalid and voltage/current are available.
    power = vectors.get("power", np.array([], dtype=float))
    voltage = vectors.get("voltage", np.array([], dtype=float))
    current = vectors.get("current", np.array([], dtype=float))

    if (
        len(power) == 0
        or np.isfinite(power).sum() < 2
    ) and len(voltage) == len(current) and len(voltage) > 0:
        vectors["power"] = voltage * current

    return time, vectors, len(time) > 1


def plot_raw_curves_for_battery(
    rows: list[tuple[str, pd.Series]],
    dataset: str,
    phase: str,
    battery_id: str,
    output_file: Path,
    args: argparse.Namespace,
) -> bool:
    if output_file.exists() and not args.overwrite:
        return True

    if not rows:
        return False

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(11.0, 12.0),
        sharex=True,
        constrained_layout=False,
    )

    plot_specs = [
        ("voltage", "Voltage [V]"),
        ("current", "Current [A]"),
        ("temperature", "Temperature [°C]"),
        ("capacity", "Capacity [Ah]"),
        ("power", "Power [W]"),
    ]

    plotted_any = False

    for selection_position, row in rows:
        time, vectors, ok = prepare_raw_vectors(row)
        if not ok:
            continue

        cycle_number = row.get("cycle_number", np.nan)
        cycle_index = row.get("cycle_index", np.nan)

        try:
            cycle_label = f"cycle={float(cycle_number):g}"
        except Exception:
            try:
                cycle_label = f"cycle_index={float(cycle_index):g}"
            except Exception:
                cycle_label = selection_position

        label = f"{selection_position} ({cycle_label})"

        for ax, (key, ylabel) in zip(axes, plot_specs):
            y = vectors.get(key, np.array([], dtype=float))
            if len(y) != len(time):
                continue
            valid = np.isfinite(time) & np.isfinite(y)
            if valid.sum() < 2:
                continue
            ax.plot(time[valid], y[valid], linewidth=1.0, label=label)
            plotted_any = True

    for ax, (_, ylabel) in zip(axes, plot_specs):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(
        f"{dataset} | {phase} | {battery_id}\n"
        "Raw curves for first, middle, and last selected cycles"
    )
    axes[0].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_file, dpi=args.dpi)
    plt.close(fig)

    return plotted_any


def iter_selected_raw_rows_fastparquet(
    path: Path,
    selected_row_ids: set[int],
    vector_columns: list[str],
    engine: str,
) -> pd.DataFrame:
    if not selected_row_ids:
        return pd.DataFrame()

    # Preferred memory-efficient path.
    if engine in ("auto", "fastparquet"):
        try:
            from fastparquet import ParquetFile

            pf = ParquetFile(str(path))
            available = list(pf.columns)
            columns = [c for c in vector_columns if c in available]

            kept_parts = []
            offset = 0

            for rg in pf.iter_row_groups(columns=columns):
                n = len(rg)
                if n == 0:
                    continue

                ids = np.arange(offset, offset + n, dtype=int)
                mask = np.isin(ids, list(selected_row_ids))

                if mask.any():
                    part = rg.loc[mask].copy()
                    part["__row_id"] = ids[mask]
                    kept_parts.append(part)

                offset += n

            if kept_parts:
                return pd.concat(kept_parts, ignore_index=True)
            return pd.DataFrame()

        except Exception as exc:
            if engine == "fastparquet":
                raise
            warnings.warn(
                f"fastparquet row-group reading failed for {path}: {exc}. "
                "Falling back to full selected-column read."
            )

    # Fallback: read vector columns fully.
    df = read_parquet_selected_columns(path, vector_columns, engine=engine)
    df["__row_id"] = np.arange(len(df), dtype=int)
    return df[df["__row_id"].isin(selected_row_ids)].copy()


def write_csv_dataclass(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    dict_rows = [asdict(r) if not isinstance(r, dict) else r for r in rows]
    fieldnames = list(dict_rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)


def write_markdown_report(
    path: Path,
    input_files: list[InputFile],
    capacity_summaries: list[CapacitySummary],
    raw_selections: list[RawSelection],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    n_capacity = len(capacity_summaries)
    n_raw_selected = len(raw_selections)
    n_input_ok = sum(1 for f in input_files if f.status == "ok")
    n_input_error = sum(1 for f in input_files if f.status != "ok")

    by_dataset_phase = defaultdict(lambda: {"files": 0, "rows": 0, "capacity_plots": 0, "raw_selected": 0})

    for f in input_files:
        key = (f.dataset, f.phase)
        by_dataset_phase[key]["files"] += 1
        by_dataset_phase[key]["rows"] += max(f.n_rows, 0)

    for c in capacity_summaries:
        key = (c.dataset, c.phase)
        by_dataset_phase[key]["capacity_plots"] += 1

    for r in raw_selections:
        key = (r.dataset, r.phase)
        by_dataset_phase[key]["raw_selected"] += 1

    lines = [
        "# Essential Battery Curve Audit",
        "",
        f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Scope",
        "",
        "- Normalized capacity vs real cycle index/number.",
        "- Raw curves with voltage, current, temperature, capacity, and power aligned by time.",
        "- Three cycles per battery for raw curves: first, middle, and last.",
        "- Charge and discharge audited separately.",
        "",
        "## Configuration",
        "",
        f"- Project root: `{args.project_root}`",
        f"- Data root: `{args.data_root}`",
        f"- Input mode: `{args.input_mode}`",
        f"- Phases: `{', '.join(args.phases)}`",
        f"- Initial capacity method: `{args.initial_capacity_method}`",
        f"- Initial window: `{args.initial_window}`",
        f"- Capacity x axis: `{args.capacity_x}` — real cycle axis unless explicitly set to `cycle_life_fraction`",
        "",
        "## Summary",
        "",
        f"- Input files read successfully: `{n_input_ok}`",
        f"- Input files with errors: `{n_input_error}`",
        f"- Battery-level capacity plots generated/listed: `{n_capacity}`",
        f"- Raw selected cycle rows: `{n_raw_selected}`",
        "",
        "## Dataset-phase summary",
        "",
        "| Dataset | Phase | Files | Rows | Capacity plots | Raw selected rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for (dataset, phase), values in sorted(by_dataset_phase.items()):
        lines.append(
            f"| {dataset} | {phase} | {values['files']} | {values['rows']} | "
            f"{values['capacity_plots']} | {values['raw_selected']} |"
        )

    lines.extend(
        [
            "",
            "## Output folders",
            "",
            "- `capacity_normalized_vs_cycle_life/by_battery`: one image per battery.",
            "- `capacity_normalized_vs_cycle_life/by_dataset`: one image per dataset and phase with all batteries.",
            "- `raw_curves`: one 5-panel image per battery.",
            "- `global/tables`: CSV summaries and selected cycles.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def process_file(path: Path, output_root: Path, args: argparse.Namespace) -> tuple[InputFile, list[CapacitySummary], list[RawSelection]]:
    dataset = infer_dataset_from_path(path)
    phase = infer_phase_from_path(path)

    if phase not in args.phases:
        return (
            InputFile(
                dataset=dataset,
                phase=phase,
                path=str(path),
                n_rows=0,
                n_columns=0,
                status="skipped_phase",
                error="",
            ),
            [],
            [],
        )

    try:
        columns = get_parquet_columns(path, engine=args.engine)
        scalar_columns = list(SCALAR_BASE_COLUMNS)

        # Use vector fallback only if phase_capacity_ah does not exist.
        if "phase_capacity_ah" not in columns and "capacity_ah" in columns:
            scalar_columns.append("capacity_ah")

        scalar_columns = [c for c in scalar_columns if c in columns]

        df = read_parquet_selected_columns(path, scalar_columns, engine=args.engine)
        df["__row_id"] = np.arange(len(df), dtype=int)

        if "dataset" not in df.columns:
            df["dataset"] = dataset
        else:
            df["dataset"] = df["dataset"].fillna(dataset).astype(str)

        if "phase" not in df.columns:
            df["phase"] = phase
        else:
            df["phase"] = df["phase"].fillna(phase).astype(str)

        if "battery_id" not in df.columns:
            raise RuntimeError("Missing required column: battery_id")

        df["battery_id"] = df["battery_id"].astype(str)

        if not args.include_stage_level:
            before_phase_filter = len(df)
            df = filter_complete_phase_rows(df)
            after_phase_filter = len(df)
            if before_phase_filter != after_phase_filter:
                print(
                    f"[INFO] Complete-phase filter: kept {after_phase_filter} of "
                    f"{before_phase_filter} rows from {path.name}"
                )

        # Optional model-usable filter only when the column exists.
        if "is_model_usable" in df.columns:
            usable = df["is_model_usable"]
            # Keep rows unless explicitly False. This avoids losing rows when the column is object/NaN.
            df = df[usable.astype(str).str.lower().ne("false")].copy()

        if args.max_batteries is not None:
            battery_ids = sorted(df["battery_id"].dropna().astype(str).unique())
            keep = set(battery_ids[: args.max_batteries])
            df = df[df["battery_id"].isin(keep)].copy()

        input_file = InputFile(
            dataset=dataset,
            phase=phase,
            path=str(path),
            n_rows=int(len(df)),
            n_columns=int(len(columns)),
            status="ok",
            error="",
        )

        capacity_summaries: list[CapacitySummary] = []
        raw_selections: list[RawSelection] = []

        if not args.skip_capacity_plots:
            battery_ids = sorted(df["battery_id"].dropna().astype(str).unique())

            for battery_id in battery_ids:
                group = df[df["battery_id"].astype(str) == str(battery_id)].copy()
                output_file = (
                    output_root
                    / "capacity_normalized_vs_cycle_life"
                    / "by_battery"
                    / safe_name(dataset)
                    / f"{safe_name(dataset)}_{safe_name(phase)}_{safe_name(battery_id)}.{args.fig_format}"
                )

                summary = save_capacity_plot_one_battery(
                    group=group,
                    dataset=dataset,
                    phase=phase,
                    battery_id=battery_id,
                    output_file=output_file,
                    args=args,
                )
                if summary is not None:
                    capacity_summaries.append(summary)

            output_all = (
                output_root
                / "capacity_normalized_vs_cycle_life"
                / "by_dataset"
                / safe_name(dataset)
                / f"{safe_name(dataset)}_{safe_name(phase)}_all_batteries.{args.fig_format}"
            )
            save_capacity_plot_all_batteries(
                df=df,
                dataset=dataset,
                phase=phase,
                output_file=output_all,
                args=args,
            )

        if not args.skip_raw_curves:
            raw_selections = select_raw_cycle_rows(
                scalar_df=df,
                dataset=dataset,
                phase=phase,
                raw_cycles_per_battery=args.raw_cycles_per_battery,
                max_batteries=args.max_batteries,
            )

            selected_ids = {s.row_id for s in raw_selections}

            vector_columns = [c for c in VECTOR_BASE_COLUMNS if c in columns]
            # Make sure row identity is reconstructed from row-group offset.
            selected_rows = iter_selected_raw_rows_fastparquet(
                path=path,
                selected_row_ids=selected_ids,
                vector_columns=vector_columns,
                engine=args.engine,
            )

            if not selected_rows.empty:
                if "dataset" not in selected_rows.columns:
                    selected_rows["dataset"] = dataset
                if "phase" not in selected_rows.columns:
                    selected_rows["phase"] = phase

                selection_by_row_id = {
                    s.row_id: s.selection_position for s in raw_selections
                }

                # Preserve selected order: first, middle, last.
                selected_rows["__selection_position"] = selected_rows["__row_id"].map(selection_by_row_id)

                for battery_id, group in selected_rows.groupby("battery_id", sort=True):
                    rows_for_plot = []

                    # Sort by row_id to keep first/middle/last order.
                    group = group.sort_values("__row_id")

                    for _, row in group.iterrows():
                        rows_for_plot.append(
                            (
                                str(row.get("__selection_position", "selected")),
                                row,
                            )
                        )

                    output_file = (
                        output_root
                        / "raw_curves"
                        / safe_name(dataset)
                        / f"{safe_name(dataset)}_{safe_name(phase)}_{safe_name(battery_id)}.{args.fig_format}"
                    )

                    plot_raw_curves_for_battery(
                        rows=rows_for_plot,
                        dataset=dataset,
                        phase=phase,
                        battery_id=str(battery_id),
                        output_file=output_file,
                        args=args,
                    )

        return input_file, capacity_summaries, raw_selections

    except Exception as exc:
        return (
            InputFile(
                dataset=dataset,
                phase=phase,
                path=str(path),
                n_rows=-1,
                n_columns=-1,
                status="error",
                error=str(exc),
            ),
            [],
            [],
        )


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root)
    data_root = Path(args.data_root)

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = project_root / "reports" / "auditoria_essencial" / f"run_{timestamp()}"

    output_root.mkdir(parents=True, exist_ok=True)

    input_files = find_input_files(
        data_root=data_root,
        input_mode=args.input_mode,
        phases=args.phases,
        datasets=args.datasets,
        engine=args.engine,
    )

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Data root: {data_root}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Input mode: {args.input_mode}")
    print(f"[INFO] Files found: {len(input_files)}")

    if not input_files:
        print("[ERROR] No input parquet files found.", file=sys.stderr)
        return 1

    try:
        from tqdm import tqdm

        iterator = tqdm(input_files, desc="Auditing files", unit="file")
    except Exception:
        iterator = input_files

    all_input_reports: list[InputFile] = []
    all_capacity_summaries: list[CapacitySummary] = []
    all_raw_selections: list[RawSelection] = []

    for path in iterator:
        print(f"[INFO] Processing: {path}")
        input_report, capacity_summaries, raw_selections = process_file(
            path=path,
            output_root=output_root,
            args=args,
        )
        all_input_reports.append(input_report)
        all_capacity_summaries.extend(capacity_summaries)
        all_raw_selections.extend(raw_selections)

    tables_dir = output_root / "global" / "tables"
    reports_dir = output_root / "global" / "reports"

    write_csv_dataclass(tables_dir / "input_files.csv", all_input_reports)
    write_csv_dataclass(tables_dir / "capacity_summary.csv", all_capacity_summaries)
    write_csv_dataclass(tables_dir / "raw_curve_selected_cycles.csv", all_raw_selections)

    write_markdown_report(
        path=reports_dir / "audit_essential_report.md",
        input_files=all_input_reports,
        capacity_summaries=all_capacity_summaries,
        raw_selections=all_raw_selections,
        args=args,
    )

    n_ok = sum(1 for r in all_input_reports if r.status == "ok")
    n_error = sum(1 for r in all_input_reports if r.status == "error")

    print("[DONE] Essential audit finished.")
    print(f"[DONE] Files OK: {n_ok}")
    print(f"[DONE] Files with errors: {n_error}")
    print(f"[DONE] Capacity plots/listed: {len(all_capacity_summaries)}")
    print(f"[DONE] Raw selected cycle rows: {len(all_raw_selections)}")
    print(f"[DONE] Output: {output_root}")

    return 0 if n_error == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
