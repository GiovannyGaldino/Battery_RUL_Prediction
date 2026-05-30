"""
build_hierarchical_features_v3.py

Patched version v3.1: faster vector parsing, closed-form slopes, optional --fast-mode, and faster target lookup.

Vector-aware hierarchical feature extraction for cleaned battery charge/discharge files.

Input expected
--------------
This script uses only:

    <input-root>/charge/*.parquet
    <input-root>/discharge/*.parquet

Each row is expected to represent one battery life segment/cycle-phase record, with
vector-like signal columns stored inside cells:

    time_s
    voltage_v
    current_a
    temperature_c
    capacity_ah
    energy_wh
    power_w

The previous v2 extractor incorrectly treated these columns as scalar columns.
This v3 script parses them as vectors/lists and extracts real curve-level features.

Main design
-----------
1. Charge and discharge are treated as independent life trajectories.
   Grouping key for target construction:

       dataset, battery_id, phase_folder

   where phase_folder is charge or discharge.

2. Capacity validation and target remodeling are done before feature extraction.
   The script analyzes phase_capacity_ah by battery and phase independently.

3. The script recomputes:
   - validated capacity
   - initial capacity
   - recomputed SOH
   - EOL at 80% of initial capacity
   - EOL at 70% of initial capacity
   - remaining steps/cycles until EOL80 and EOL70

4. Feature extraction is resumable.
   Each input Parquet generates one output feature Parquet plus a .done marker.

5. Output levels:
   - phase: features for the full vector in the original row
   - stage: optional subsegments inside the phase, split by large time gaps

Typical usage
-------------

python src/feature_extraction/build_hierarchical_features_v3.py --input-root data/interim --output-root data/processed/features_hierarchical_v3 --reports-root reports/feature_extraction_v3 --workers 2 --resume

Important
---------
Do not write output to features_hierarchical_v2. Use features_hierarchical_v3.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import fastparquet  # noqa: F401
    from fastparquet import ParquetFile
except Exception as exc:
    raise RuntimeError("fastparquet is required. Install with: pip install fastparquet") from exc


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_INPUT_ROOT = Path("data/interim")
DEFAULT_OUTPUT_ROOT = Path("data/processed/features_hierarchical_v3")
DEFAULT_REPORTS_ROOT = Path("reports/feature_extraction_v3")

PHASE_FOLDERS = ("charge", "discharge")

VECTOR_COLUMNS = [
    "time_s",
    "voltage_v",
    "current_a",
    "temperature_c",
    "capacity_ah",
    "energy_wh",
    "power_w",
]

IDENTITY_COLUMNS = [
    "dataset",
    "source_file",
    "source_path",
    "battery_id",
    "battery_index",
    "cell_id_original",
    "cycle_index",
    "cycle_number",
    "source_cycle_id",
    "cycle_pair_id",
    "phase",
]

TARGET_SOURCE_COLUMNS = [
    "soh",
    "remaining_cycles",
    "remaining_cycles_norm",
    "cycle_life",
    "eol_criterion",
    "eol_capacity_ah",
    "initial_capacity_ah",
    "nominal_capacity_ah",
]

SCALAR_INPUT_COLUMNS = [
    "n_points",
    "duration_s",
    "sample_period_median_s",
    "voltage_min_v",
    "voltage_max_v",
    "voltage_start_v",
    "voltage_end_v",
    "current_min_a",
    "current_max_a",
    "current_mean_a",
    "temperature_min_c",
    "temperature_max_c",
    "temperature_mean_c",
    "phase_capacity_ah",
    "phase_energy_wh",
    "chemistry",
    "form_factor",
    "manufacturer",
    "nominal_voltage_v",
    "ambient_temperature_c",
    "charge_protocol",
    "discharge_protocol",
    "c_rate_charge",
    "c_rate_discharge",
    "dod",
    "policy_readable",
    "barcode",
    "is_valid_cycle",
    "is_model_usable",
    "exclude_reason",
    "cleaning_warnings",
    "n_removed_points",
    "trim_start_index",
    "trim_end_index",
    "has_voltage",
    "has_current",
    "has_temperature",
    "has_capacity",
    "has_energy",
]

TARGET_OUTPUT_COLUMNS = [
    "source_file_key",
    "source_row_id",
    "phase_folder",
    "dataset",
    "battery_id",
    "cycle_index",
    "cycle_number",
    "phase",
    "target_capacity_ah",
    "target_capacity_source",
    "target_capacity_valid",
    "target_initial_capacity_ah",
    "target_initial_capacity_method",
    "target_capacity_ratio_to_initial",
    "target_soh_recomputed",
    "target_soh_recomputed_clipped_0_1p2",
    "target_eol80_order_index",
    "target_eol80_cycle_index",
    "target_eol80_capacity_ah",
    "target_eol80_found",
    "target_after_eol80",
    "target_rul80_steps_signed",
    "target_rul80_steps_clipped",
    "target_rul80_cycle_index_signed",
    "target_rul80_cycle_index_clipped",
    "target_eol70_order_index",
    "target_eol70_cycle_index",
    "target_eol70_capacity_ah",
    "target_eol70_found",
    "target_after_eol70",
    "target_rul70_steps_signed",
    "target_rul70_steps_clipped",
    "target_rul70_cycle_index_signed",
    "target_rul70_cycle_index_clipped",
    "capacity_trend_slope_per_step",
    "capacity_trend_spearman",
    "capacity_positive_step_rate",
    "capacity_negative_step_rate",
    "capacity_median_abs_step_ah",
    "capacity_validation_status",
    "capacity_validation_notes",
    "input_soh",
    "input_remaining_cycles",
    "input_remaining_cycles_norm",
    "input_cycle_life",
    "input_initial_capacity_ah",
    "input_nominal_capacity_ah",
]


@dataclass
class InputFile:
    phase_folder: str
    dataset_name: str
    path: str
    output_path: str
    done_path: str
    error_path: str
    job_id: str


# =============================================================================
# Generic utilities
# =============================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_filename(value: str, max_len: int = 140) -> str:
    value = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value)).strip("_")
    return (value or "file")[:max_len]


def short_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def normalize_path_key(path: Path | str) -> str:
    return str(Path(path)).replace("\\", "/").lower()


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def finite_numeric(value: Any) -> float:
    out = safe_float(value)
    return out if np.isfinite(out) else np.nan


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """
    Safe trapezoidal integration compatible with NumPy versions where np.trapz
    is unavailable or deprecated.

    Equivalent to integral y dx using the trapezoidal rule.
    """
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)

    n = min(len(yy), len(xx))
    if n < 2:
        return np.nan

    yy = yy[:n]
    xx = xx[:n]

    valid = np.isfinite(yy) & np.isfinite(xx)
    if valid.sum() < 2:
        return np.nan

    yy = yy[valid]
    xx = xx[valid]

    order = np.argsort(xx, kind="mergesort")
    xx = xx[order]
    yy = yy[order]

    dx = np.diff(xx)
    finite_dx = np.isfinite(dx) & (dx >= 0)

    if finite_dx.sum() == 0:
        return np.nan

    y_mid_sum = 0.5 * (yy[:-1] + yy[1:])
    return safe_float(np.sum(y_mid_sum[finite_dx] * dx[finite_dx]))


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def file_is_valid(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def read_parquet_columns(path: Path) -> List[str]:
    return list(ParquetFile(str(path)).columns)


def read_parquet_selected(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    available = set(read_parquet_columns(path))
    selected = [col for col in columns if col in available]

    if not selected:
        return pd.DataFrame()

    return pd.read_parquet(path, columns=selected, engine="fastparquet")


def infer_dataset_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"_(charge|discharge).*", "", stem)
    return stem.upper() if stem else "UNKNOWN"


def parse_phase_dirs(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return list(PHASE_FOLDERS)

    output: List[str] = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                output.append(part)

    return list(dict.fromkeys(output))


# =============================================================================
# Vector parsing
# =============================================================================

def parse_vector(value: Any) -> np.ndarray:
    """
    Convert a cell value into a 1D float64 numpy array.

    v3.1 optimization: avoid constructing a pandas Series for every vector.
    In the MATR file this function is called millions of times; using
    ``pd.to_numeric(pd.Series(...))`` inside the hot loop can dominate runtime.
    The fast path below handles the common cases produced by fastparquet
    (list/tuple/ndarray of numeric values) with ``np.asarray(..., dtype=float)``.
    """

    if value is None:
        return np.asarray([], dtype=np.float64)

    try:
        if isinstance(value, float) and math.isnan(value):
            return np.asarray([], dtype=np.float64)
    except Exception:
        pass

    raw: Any

    if isinstance(value, np.ndarray):
        raw = value
    elif isinstance(value, (list, tuple)):
        raw = value
    elif isinstance(value, pd.Series):
        raw = value.to_numpy()
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return np.asarray([], dtype=np.float64)
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
            # Prefer np.fromstring for long numeric lists; it is much faster than ast.literal_eval.
            text2 = text.strip("[]()")
            try:
                arr = np.fromstring(text2.replace(";", ","), sep=",", dtype=np.float64)
                if arr.size > 0:
                    return arr[np.isfinite(arr)]
            except Exception:
                pass
            try:
                raw = ast.literal_eval(text)
            except Exception:
                raw = re.split(r"[,\s;]+", text2)
        else:
            try:
                scalar = float(value)
            except Exception:
                return np.asarray([], dtype=np.float64)
            return np.asarray([scalar], dtype=np.float64) if np.isfinite(scalar) else np.asarray([], dtype=np.float64)
    else:
        try:
            scalar = float(value)
            return np.asarray([scalar], dtype=np.float64) if np.isfinite(scalar) else np.asarray([], dtype=np.float64)
        except Exception:
            raw = value

    try:
        arr = np.asarray(raw, dtype=np.float64)
    except Exception:
        try:
            arr = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=np.float64)
        except Exception:
            return np.asarray([], dtype=np.float64)

    if arr.ndim > 1:
        arr = arr.reshape(-1)

    return arr[np.isfinite(arr)]


def align_by_time(vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Align vectors by the shortest available length among time_s and the signal
    columns that are needed for time-based features.
    """

    lengths = [len(v) for v in vectors.values() if isinstance(v, np.ndarray) and len(v) > 0]

    if not lengths:
        return vectors

    n = min(lengths)

    aligned = {}
    for key, arr in vectors.items():
        if isinstance(arr, np.ndarray) and len(arr) >= n:
            aligned[key] = arr[:n]
        else:
            aligned[key] = arr

    if "time_s" in aligned and len(aligned["time_s"]) == n:
        t = aligned["time_s"]
        if len(t) > 1:
            order = np.argsort(t, kind="mergesort")
            if not np.all(order == np.arange(len(t))):
                for key, arr in list(aligned.items()):
                    if isinstance(arr, np.ndarray) and len(arr) == n:
                        aligned[key] = arr[order]

    return aligned


# =============================================================================
# Capacity and target remodeling
# =============================================================================

def discover_input_files(
    input_root: Path,
    output_root: Path,
    phase_dirs: Sequence[str],
    datasets: Sequence[str],
    max_files: Optional[int],
) -> List[InputFile]:
    wanted = {d.upper() for d in datasets if d}
    files: List[InputFile] = []

    for phase_folder in phase_dirs:
        folder = input_root / phase_folder

        if not folder.exists():
            print(f"[WARNING] Missing phase folder: {folder}", file=sys.stderr)
            continue

        phase_files = sorted(folder.glob("*.parquet"), key=lambda p: str(p).lower())

        if max_files is not None:
            phase_files = phase_files[:max_files]

        for path in phase_files:
            dataset_name = infer_dataset_from_filename(path)

            if wanted and dataset_name.upper() not in wanted:
                continue

            job_id = short_hash(f"{phase_folder}/{path.name}")
            out_dir = output_root / phase_folder / dataset_name.lower()
            output_path = out_dir / f"{path.stem}__{job_id}.features.parquet"
            done_path = Path(str(output_path) + ".done")
            error_path = out_dir / f"{path.stem}__{job_id}.error.txt"

            files.append(
                InputFile(
                    phase_folder=phase_folder,
                    dataset_name=dataset_name,
                    path=str(path),
                    output_path=str(output_path),
                    done_path=str(done_path),
                    error_path=str(error_path),
                    job_id=job_id,
                )
            )

    return files


def load_scalar_index(files: Sequence[InputFile]) -> pd.DataFrame:
    columns = list(dict.fromkeys(
        IDENTITY_COLUMNS
        + TARGET_SOURCE_COLUMNS
        + SCALAR_INPUT_COLUMNS
    ))

    frames: List[pd.DataFrame] = []

    for item in files:
        path = Path(item.path)
        df = read_parquet_selected(path, columns)

        df["source_file_key"] = normalize_path_key(path)
        df["source_row_id"] = np.arange(len(df), dtype=np.int64)
        df["phase_folder"] = item.phase_folder
        df["dataset_from_file_name"] = item.dataset_name

        if "dataset" not in df.columns:
            df["dataset"] = item.dataset_name

        if "phase" not in df.columns:
            df["phase"] = item.phase_folder

        if "battery_id" not in df.columns:
            raise ValueError(f"Missing battery_id in {path}")

        if "cycle_index" not in df.columns and "cycle_number" not in df.columns:
            raise ValueError(f"Missing cycle_index/cycle_number in {path}")

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)

    for col in ["dataset", "battery_id", "phase", "phase_folder", "source_file_key"]:
        if col in out.columns:
            out[col] = out[col].astype("string")

    for col in ["cycle_index", "cycle_number", "phase_capacity_ah", "soh", "remaining_cycles", "remaining_cycles_norm"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def choose_capacity_series(group: pd.DataFrame) -> Tuple[pd.Series, str]:
    """
    Choose the capacity used to rebuild SOH/RUL targets.

    Priority:
    1. phase_capacity_ah, because the audit showed it exists in all charge/discharge files.
    2. capacity-like fallback from existing scalar columns if needed.

    The returned capacity must be positive and finite.
    """

    if "phase_capacity_ah" in group.columns:
        cap = pd.to_numeric(group["phase_capacity_ah"], errors="coerce")
        valid_count = int(((cap > 0) & np.isfinite(cap)).sum())
        if valid_count > 0:
            return cap, "phase_capacity_ah"

    # Fallbacks, kept for robustness.
    for col in ["capacity_ah", "discharge_capacity_ah", "charge_capacity_ah"]:
        if col in group.columns:
            cap = pd.to_numeric(group[col], errors="coerce")
            valid_count = int(((cap > 0) & np.isfinite(cap)).sum())
            if valid_count > 0:
                return cap, col

    return pd.Series(np.nan, index=group.index), "none"


def compute_initial_capacity(
    cap: pd.Series,
    method: str,
    initial_window: int,
    initial_fraction: float,
) -> Tuple[float, str]:
    valid = cap[(cap > 0) & np.isfinite(cap)]

    if valid.empty:
        return np.nan, "no_valid_capacity"

    n = len(valid)
    window_n = max(1, int(min(initial_window, max(initial_window, math.ceil(n * initial_fraction)))))
    window_n = min(window_n, n)

    early = valid.iloc[:window_n]

    if method == "early_max":
        return safe_float(early.max()), f"early_max_{window_n}"
    if method == "early_mean":
        return safe_float(early.mean()), f"early_mean_{window_n}"

    # Default: early median is robust to a single abnormal first cycle.
    return safe_float(early.median()), f"early_median_{window_n}"


def find_eol_position(
    cap: pd.Series,
    initial_capacity: float,
    threshold_ratio: float,
    rolling_window: int,
    consecutive: int,
    min_position: int,
) -> Tuple[float, float, float]:
    """
    Returns:
        eol_position, eol_cycle_like_value, eol_capacity_value

    eol_position is the row order within the battery/phase group.
    """

    if not np.isfinite(initial_capacity) or initial_capacity <= 0:
        return np.nan, np.nan, np.nan

    values = pd.to_numeric(cap, errors="coerce").astype(float)
    smooth = values.rolling(window=rolling_window, min_periods=1).median()

    threshold = threshold_ratio * initial_capacity
    below = (smooth <= threshold) & values.notna()

    if len(below) == 0:
        return np.nan, np.nan, np.nan

    consecutive = max(1, int(consecutive))
    min_position = max(0, int(min_position))

    arr = below.to_numpy(dtype=bool)

    found_pos: Optional[int] = None

    for pos in range(min_position, len(arr)):
        end = min(len(arr), pos + consecutive)
        if end - pos < consecutive:
            break
        if bool(arr[pos:end].all()):
            found_pos = pos
            break

    if found_pos is None and consecutive > 1:
        # Fallback: first below-threshold point.
        below_positions = np.where(arr[min_position:])[0]
        if len(below_positions):
            found_pos = int(below_positions[0] + min_position)

    if found_pos is None:
        return np.nan, np.nan, np.nan

    eol_capacity = safe_float(values.iloc[found_pos])
    return float(found_pos), np.nan, eol_capacity


def compute_trend_stats(cap: pd.Series) -> Dict[str, Any]:
    valid = pd.to_numeric(cap, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = valid.dropna()

    out = {
        "capacity_trend_slope_per_step": np.nan,
        "capacity_trend_spearman": np.nan,
        "capacity_positive_step_rate": np.nan,
        "capacity_negative_step_rate": np.nan,
        "capacity_median_abs_step_ah": np.nan,
    }

    if len(valid) < 3:
        return out

    y = valid.to_numpy(dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)

    try:
        slope = np.polyfit(x, y, deg=1)[0]
    except Exception:
        slope = np.nan

    try:
        spearman = pd.Series(x).corr(pd.Series(y), method="spearman")
    except Exception:
        spearman = np.nan

    diffs = np.diff(y)
    nonzero = diffs[np.isfinite(diffs)]

    if len(nonzero):
        positive_rate = float((nonzero > 0).mean())
        negative_rate = float((nonzero < 0).mean())
        med_abs_step = safe_float(np.median(np.abs(nonzero)))
    else:
        positive_rate = np.nan
        negative_rate = np.nan
        med_abs_step = np.nan

    out.update(
        {
            "capacity_trend_slope_per_step": safe_float(slope),
            "capacity_trend_spearman": safe_float(spearman),
            "capacity_positive_step_rate": positive_rate,
            "capacity_negative_step_rate": negative_rate,
            "capacity_median_abs_step_ah": med_abs_step,
        }
    )

    return out


def validate_capacity_group(
    group: pd.DataFrame,
    initial_capacity: float,
    cap: pd.Series,
    nominal_col: Optional[str] = "nominal_capacity_ah",
) -> Tuple[str, str]:
    notes: List[str] = []
    status = "ok"

    valid = cap[(cap > 0) & np.isfinite(cap)]

    if valid.empty:
        return "invalid", "no positive finite capacity"

    if not np.isfinite(initial_capacity) or initial_capacity <= 0:
        return "invalid", "invalid initial capacity"

    # Capacity should be in a physically plausible range relative to the initial capacity.
    ratio = valid / initial_capacity
    high_ratio = float((ratio > 1.35).mean())
    low_ratio = float((ratio < 0.05).mean())

    if high_ratio > 0.01:
        status = "warning"
        notes.append(f"{high_ratio:.3f} of capacity values exceed 1.35 * initial capacity")

    if low_ratio > 0.01:
        status = "warning"
        notes.append(f"{low_ratio:.3f} of capacity values below 0.05 * initial capacity")

    if nominal_col and nominal_col in group.columns:
        nominal = pd.to_numeric(group[nominal_col], errors="coerce")
        nominal_valid = nominal[(nominal > 0) & np.isfinite(nominal)]
        if not nominal_valid.empty:
            nominal_ref = safe_float(nominal_valid.median())
            if np.isfinite(nominal_ref) and nominal_ref > 0:
                initial_to_nominal = initial_capacity / nominal_ref
                if initial_to_nominal < 0.4 or initial_to_nominal > 1.6:
                    status = "warning"
                    notes.append(f"initial/nominal ratio = {initial_to_nominal:.3f}")

    trend = compute_trend_stats(cap)
    slope = trend["capacity_trend_slope_per_step"]
    spearman = trend["capacity_trend_spearman"]

    if np.isfinite(slope) and slope > 0:
        status = "warning"
        notes.append(f"positive capacity slope = {slope:.6g}")

    if np.isfinite(spearman) and spearman > 0.20:
        status = "warning"
        notes.append(f"positive capacity Spearman trend = {spearman:.3f}")

    return status, "; ".join(notes)


def build_targets(
    scalar_df: pd.DataFrame,
    initial_method: str,
    initial_window: int,
    initial_fraction: float,
    eol_rolling_window: int,
    eol_consecutive: int,
    eol_min_position: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build redesigned targets independently for each:
        dataset, battery_id, phase_folder

    Here charge and discharge are independent life trajectories.
    """

    if scalar_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    cycle_sort_col = "cycle_index" if "cycle_index" in scalar_df.columns else "cycle_number"

    records: List[pd.DataFrame] = []
    summary_records: List[Dict[str, Any]] = []

    group_cols = ["dataset", "battery_id", "phase_folder"]

    for (dataset, battery_id, phase_folder), group in tqdm(
        scalar_df.groupby(group_cols, dropna=False, sort=False),
        desc="Rebuilding targets",
    ):
        group = group.copy()
        group = group.sort_values(cycle_sort_col, kind="mergesort").reset_index(drop=False)
        original_index_col = "index"

        cap, cap_source = choose_capacity_series(group)
        cap = pd.to_numeric(cap, errors="coerce")
        cap_valid = (cap > 0) & np.isfinite(cap)

        initial_capacity, initial_method_used = compute_initial_capacity(
            cap=cap,
            method=initial_method,
            initial_window=initial_window,
            initial_fraction=initial_fraction,
        )

        trend = compute_trend_stats(cap)

        status, notes = validate_capacity_group(
            group=group,
            initial_capacity=initial_capacity,
            cap=cap,
            nominal_col="nominal_capacity_ah",
        )

        eol80_pos, _, eol80_capacity = find_eol_position(
            cap=cap,
            initial_capacity=initial_capacity,
            threshold_ratio=0.80,
            rolling_window=eol_rolling_window,
            consecutive=eol_consecutive,
            min_position=eol_min_position,
        )

        eol70_pos, _, eol70_capacity = find_eol_position(
            cap=cap,
            initial_capacity=initial_capacity,
            threshold_ratio=0.70,
            rolling_window=eol_rolling_window,
            consecutive=eol_consecutive,
            min_position=eol_min_position,
        )

        if np.isfinite(eol80_pos):
            eol80_pos_int = int(eol80_pos)
            eol80_cycle = safe_float(group.loc[eol80_pos_int, cycle_sort_col])
            eol80_found = True
        else:
            eol80_pos_int = None
            eol80_cycle = np.nan
            eol80_found = False

        if np.isfinite(eol70_pos):
            eol70_pos_int = int(eol70_pos)
            eol70_cycle = safe_float(group.loc[eol70_pos_int, cycle_sort_col])
            eol70_found = True
        else:
            eol70_pos_int = None
            eol70_cycle = np.nan
            eol70_found = False

        order_index = np.arange(len(group), dtype=np.int64)
        cycle_values = pd.to_numeric(group[cycle_sort_col], errors="coerce").to_numpy(dtype=np.float64)

        soh_recomputed = cap / initial_capacity if np.isfinite(initial_capacity) and initial_capacity > 0 else np.nan
        soh_clipped = pd.Series(soh_recomputed).clip(lower=0, upper=1.2).to_numpy(dtype=np.float64)

        if eol80_found:
            rul80_steps_signed = eol80_pos_int - order_index
            rul80_steps_clipped = np.maximum(rul80_steps_signed, 0)
            rul80_cycle_signed = eol80_cycle - cycle_values
            rul80_cycle_clipped = np.maximum(rul80_cycle_signed, 0)
            after80 = order_index > eol80_pos_int
        else:
            rul80_steps_signed = np.full(len(group), np.nan)
            rul80_steps_clipped = np.full(len(group), np.nan)
            rul80_cycle_signed = np.full(len(group), np.nan)
            rul80_cycle_clipped = np.full(len(group), np.nan)
            after80 = np.full(len(group), False)

        if eol70_found:
            rul70_steps_signed = eol70_pos_int - order_index
            rul70_steps_clipped = np.maximum(rul70_steps_signed, 0)
            rul70_cycle_signed = eol70_cycle - cycle_values
            rul70_cycle_clipped = np.maximum(rul70_cycle_signed, 0)
            after70 = order_index > eol70_pos_int
        else:
            rul70_steps_signed = np.full(len(group), np.nan)
            rul70_steps_clipped = np.full(len(group), np.nan)
            rul70_cycle_signed = np.full(len(group), np.nan)
            rul70_cycle_clipped = np.full(len(group), np.nan)
            after70 = np.full(len(group), False)

        target_part = pd.DataFrame(
            {
                "source_file_key": group["source_file_key"].astype(str).to_numpy(),
                "source_row_id": group["source_row_id"].to_numpy(),
                "phase_folder": group["phase_folder"].astype(str).to_numpy(),
                "dataset": group["dataset"].astype(str).to_numpy(),
                "battery_id": group["battery_id"].astype(str).to_numpy(),
                "cycle_index": pd.to_numeric(group.get("cycle_index", np.nan), errors="coerce"),
                "cycle_number": pd.to_numeric(group.get("cycle_number", np.nan), errors="coerce"),
                "phase": group["phase"].astype(str).to_numpy() if "phase" in group.columns else group["phase_folder"].astype(str).to_numpy(),
                "target_capacity_ah": cap.to_numpy(dtype=np.float64),
                "target_capacity_source": cap_source,
                "target_capacity_valid": cap_valid.to_numpy(dtype=bool),
                "target_initial_capacity_ah": initial_capacity,
                "target_initial_capacity_method": initial_method_used,
                "target_capacity_ratio_to_initial": pd.to_numeric(soh_recomputed, errors="coerce"),
                "target_soh_recomputed": pd.to_numeric(soh_recomputed, errors="coerce"),
                "target_soh_recomputed_clipped_0_1p2": soh_clipped,
                "target_eol80_order_index": eol80_pos if eol80_found else np.nan,
                "target_eol80_cycle_index": eol80_cycle,
                "target_eol80_capacity_ah": eol80_capacity,
                "target_eol80_found": eol80_found,
                "target_after_eol80": after80,
                "target_rul80_steps_signed": rul80_steps_signed,
                "target_rul80_steps_clipped": rul80_steps_clipped,
                "target_rul80_cycle_index_signed": rul80_cycle_signed,
                "target_rul80_cycle_index_clipped": rul80_cycle_clipped,
                "target_eol70_order_index": eol70_pos if eol70_found else np.nan,
                "target_eol70_cycle_index": eol70_cycle,
                "target_eol70_capacity_ah": eol70_capacity,
                "target_eol70_found": eol70_found,
                "target_after_eol70": after70,
                "target_rul70_steps_signed": rul70_steps_signed,
                "target_rul70_steps_clipped": rul70_steps_clipped,
                "target_rul70_cycle_index_signed": rul70_cycle_signed,
                "target_rul70_cycle_index_clipped": rul70_cycle_clipped,
                "capacity_trend_slope_per_step": trend["capacity_trend_slope_per_step"],
                "capacity_trend_spearman": trend["capacity_trend_spearman"],
                "capacity_positive_step_rate": trend["capacity_positive_step_rate"],
                "capacity_negative_step_rate": trend["capacity_negative_step_rate"],
                "capacity_median_abs_step_ah": trend["capacity_median_abs_step_ah"],
                "capacity_validation_status": status,
                "capacity_validation_notes": notes,
                "input_soh": pd.to_numeric(group["soh"], errors="coerce") if "soh" in group.columns else np.nan,
                "input_remaining_cycles": pd.to_numeric(group["remaining_cycles"], errors="coerce") if "remaining_cycles" in group.columns else np.nan,
                "input_remaining_cycles_norm": pd.to_numeric(group["remaining_cycles_norm"], errors="coerce") if "remaining_cycles_norm" in group.columns else np.nan,
                "input_cycle_life": pd.to_numeric(group["cycle_life"], errors="coerce") if "cycle_life" in group.columns else np.nan,
                "input_initial_capacity_ah": pd.to_numeric(group["initial_capacity_ah"], errors="coerce") if "initial_capacity_ah" in group.columns else np.nan,
                "input_nominal_capacity_ah": pd.to_numeric(group["nominal_capacity_ah"], errors="coerce") if "nominal_capacity_ah" in group.columns else np.nan,
            }
        )

        records.append(target_part)

        summary_records.append(
            {
                "dataset": dataset,
                "battery_id": battery_id,
                "phase_folder": phase_folder,
                "n_rows": int(len(group)),
                "n_valid_capacity": int(cap_valid.sum()),
                "capacity_source": cap_source,
                "initial_capacity_ah": initial_capacity,
                "initial_capacity_method": initial_method_used,
                "capacity_first_ah": safe_float(cap[cap_valid].iloc[0]) if cap_valid.any() else np.nan,
                "capacity_last_ah": safe_float(cap[cap_valid].iloc[-1]) if cap_valid.any() else np.nan,
                "capacity_min_ah": safe_float(cap[cap_valid].min()) if cap_valid.any() else np.nan,
                "capacity_max_ah": safe_float(cap[cap_valid].max()) if cap_valid.any() else np.nan,
                "capacity_end_to_initial_ratio": safe_float(cap[cap_valid].iloc[-1] / initial_capacity) if cap_valid.any() and np.isfinite(initial_capacity) and initial_capacity > 0 else np.nan,
                "eol80_found": eol80_found,
                "eol80_order_index": eol80_pos if eol80_found else np.nan,
                "eol80_cycle_index": eol80_cycle,
                "eol80_capacity_ah": eol80_capacity,
                "eol70_found": eol70_found,
                "eol70_order_index": eol70_pos if eol70_found else np.nan,
                "eol70_cycle_index": eol70_cycle,
                "eol70_capacity_ah": eol70_capacity,
                "trend_slope_per_step": trend["capacity_trend_slope_per_step"],
                "trend_spearman": trend["capacity_trend_spearman"],
                "positive_step_rate": trend["capacity_positive_step_rate"],
                "negative_step_rate": trend["capacity_negative_step_rate"],
                "capacity_validation_status": status,
                "capacity_validation_notes": notes,
            }
        )

    targets = pd.concat(records, ignore_index=True, sort=False) if records else pd.DataFrame()
    summary = pd.DataFrame(summary_records)

    return targets, summary


# =============================================================================
# Feature engineering
# =============================================================================

def numeric_stats(prefix: str, arr: np.ndarray) -> Dict[str, float]:
    out = {
        f"{prefix}_n_valid": 0,
        f"{prefix}_first": np.nan,
        f"{prefix}_last": np.nan,
        f"{prefix}_min": np.nan,
        f"{prefix}_max": np.nan,
        f"{prefix}_mean": np.nan,
        f"{prefix}_std": np.nan,
        f"{prefix}_median": np.nan,
        f"{prefix}_q05": np.nan,
        f"{prefix}_q95": np.nan,
        f"{prefix}_delta": np.nan,
        f"{prefix}_range": np.nan,
        f"{prefix}_abs_diff_mean": np.nan,
        f"{prefix}_abs_diff_max": np.nan,
        f"{prefix}_sign_change_count": np.nan,
    }

    if arr is None or len(arr) == 0:
        return out

    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return out

    out[f"{prefix}_n_valid"] = int(len(x))
    out[f"{prefix}_first"] = safe_float(x[0])
    out[f"{prefix}_last"] = safe_float(x[-1])
    out[f"{prefix}_min"] = safe_float(np.min(x))
    out[f"{prefix}_max"] = safe_float(np.max(x))
    out[f"{prefix}_mean"] = safe_float(np.mean(x))
    out[f"{prefix}_std"] = safe_float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    out[f"{prefix}_median"] = safe_float(np.median(x))
    out[f"{prefix}_q05"] = safe_float(np.quantile(x, 0.05))
    out[f"{prefix}_q95"] = safe_float(np.quantile(x, 0.95))
    out[f"{prefix}_delta"] = safe_float(x[-1] - x[0])
    out[f"{prefix}_range"] = safe_float(np.max(x) - np.min(x))

    if len(x) > 1:
        dx = np.diff(x)
        finite_dx = dx[np.isfinite(dx)]
        if len(finite_dx):
            out[f"{prefix}_abs_diff_mean"] = safe_float(np.mean(np.abs(finite_dx)))
            out[f"{prefix}_abs_diff_max"] = safe_float(np.max(np.abs(finite_dx)))
            signs = np.sign(finite_dx)
            nonzero = signs[signs != 0]
            if len(nonzero) > 1:
                out[f"{prefix}_sign_change_count"] = int(np.sum(nonzero[1:] != nonzero[:-1]))
            else:
                out[f"{prefix}_sign_change_count"] = 0

    return out


def slope_vs_time(prefix: str, y: np.ndarray, t: np.ndarray, *, fast_mode: bool = False) -> Dict[str, float]:
    out = {
        f"{prefix}_slope_vs_time": np.nan,
        f"{prefix}_corr_vs_time": np.nan,
    }

    if y is None or t is None or len(y) < 2 or len(t) < 2:
        return out

    n = min(len(y), len(t))
    x = np.asarray(t[:n], dtype=np.float64)
    z = np.asarray(y[:n], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(z)

    if valid.sum() < 2:
        return out

    x = x[valid]
    z = z[valid]

    if x.size < 2:
        return out

    x_mean = float(np.mean(x))
    z_mean = float(np.mean(z))
    dx = x - x_mean
    dz = z - z_mean
    denom = float(np.dot(dx, dx))
    z_denom = float(np.dot(dz, dz))

    if denom <= 0 or z_denom <= 0:
        return out

    # Closed-form least-squares slope is much faster than np.polyfit for millions of calls.
    slope = float(np.dot(dx, dz) / denom)

    if fast_mode:
        # Fast mode stores Pearson correlation in the existing *_corr_vs_time column.
        # This is intended for large article-scale runs where runtime is more important
        # than Spearman rank correlation.
        corr = float(np.dot(dx, dz) / math.sqrt(denom * z_denom))
    else:
        try:
            corr = pd.Series(x).corr(pd.Series(z), method="spearman")
        except Exception:
            corr = np.nan

    out[f"{prefix}_slope_vs_time"] = safe_float(slope)
    out[f"{prefix}_corr_vs_time"] = safe_float(corr)
    return out


def time_gap_features(t: np.ndarray, min_gap_seconds: float, gap_factor: float) -> Tuple[Dict[str, Any], np.ndarray]:
    out = {
        "time_vector_duration_s": np.nan,
        "time_dt_min_s": np.nan,
        "time_dt_mean_s": np.nan,
        "time_dt_median_s": np.nan,
        "time_dt_max_s": np.nan,
        "time_gap_threshold_s": np.nan,
        "time_gap_count": 0,
        "time_gap_total_s": 0.0,
        "time_gap_max_s": 0.0,
        "time_segment_count": 0,
    }

    if t is None or len(t) < 2:
        out["time_segment_count"] = 1 if t is not None and len(t) > 0 else 0
        return out, np.asarray([], dtype=int)

    x = np.asarray(t, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        out["time_segment_count"] = 1 if len(x) > 0 else 0
        return out, np.asarray([], dtype=int)

    dt = np.diff(x)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]

    out["time_vector_duration_s"] = safe_float(np.nanmax(x) - np.nanmin(x))

    if len(positive_dt) == 0:
        out["time_segment_count"] = 1
        return out, np.asarray([], dtype=int)

    threshold = max(float(min_gap_seconds), float(gap_factor) * float(np.median(positive_dt)))
    gap_indices = np.where(np.isfinite(dt) & (dt > threshold))[0]
    gap_values = dt[gap_indices] if len(gap_indices) else np.asarray([], dtype=np.float64)

    out.update(
        {
            "time_dt_min_s": safe_float(np.min(positive_dt)),
            "time_dt_mean_s": safe_float(np.mean(positive_dt)),
            "time_dt_median_s": safe_float(np.median(positive_dt)),
            "time_dt_max_s": safe_float(np.max(positive_dt)),
            "time_gap_threshold_s": safe_float(threshold),
            "time_gap_count": int(len(gap_indices)),
            "time_gap_total_s": safe_float(np.sum(gap_values)) if len(gap_values) else 0.0,
            "time_gap_max_s": safe_float(np.max(gap_values)) if len(gap_values) else 0.0,
            "time_segment_count": int(len(gap_indices) + 1),
        }
    )

    return out, gap_indices


def boundary_jump_features(
    prefix: str,
    arr: np.ndarray,
    gap_indices: np.ndarray,
) -> Dict[str, float]:
    out = {
        f"{prefix}_gap_jump_n": 0,
        f"{prefix}_gap_jump_mean": np.nan,
        f"{prefix}_gap_jump_abs_mean": np.nan,
        f"{prefix}_gap_jump_abs_max": np.nan,
        f"{prefix}_gap_jump_abs_sum": np.nan,
    }

    if arr is None or len(arr) == 0 or gap_indices is None or len(gap_indices) == 0:
        return out

    values = np.asarray(arr, dtype=np.float64)
    jumps = []

    for idx in gap_indices:
        if idx < 0 or idx + 1 >= len(values):
            continue
        a = values[idx]
        b = values[idx + 1]
        if np.isfinite(a) and np.isfinite(b):
            jumps.append(float(b - a))

    if not jumps:
        return out

    x = np.asarray(jumps, dtype=np.float64)
    ax = np.abs(x)

    out[f"{prefix}_gap_jump_n"] = int(len(x))
    out[f"{prefix}_gap_jump_mean"] = safe_float(np.mean(x))
    out[f"{prefix}_gap_jump_abs_mean"] = safe_float(np.mean(ax))
    out[f"{prefix}_gap_jump_abs_max"] = safe_float(np.max(ax))
    out[f"{prefix}_gap_jump_abs_sum"] = safe_float(np.sum(ax))

    return out


def integration_features(vectors: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {
        "integral_current_signed_ah": np.nan,
        "integral_current_abs_ah": np.nan,
        "integral_energy_signed_wh_from_vi": np.nan,
        "integral_energy_abs_wh_from_vi": np.nan,
        "integral_power_signed_wh": np.nan,
        "integral_power_abs_wh": np.nan,
    }

    t = vectors.get("time_s", np.asarray([]))
    i = vectors.get("current_a", np.asarray([]))
    v = vectors.get("voltage_v", np.asarray([]))
    p = vectors.get("power_w", np.asarray([]))

    if len(t) < 2:
        return out

    if len(i) >= 2:
        n = min(len(t), len(i))
        tt = t[:n]
        ii = i[:n]
        valid = np.isfinite(tt) & np.isfinite(ii)
        if valid.sum() >= 2:
            tt = tt[valid]
            ii = ii[valid]
            out["integral_current_signed_ah"] = safe_float(trapezoid_integral(ii, tt) / 3600.0)
            out["integral_current_abs_ah"] = safe_float(trapezoid_integral(np.abs(ii), tt) / 3600.0)

    if len(i) >= 2 and len(v) >= 2:
        n = min(len(t), len(i), len(v))
        tt = t[:n]
        ii = i[:n]
        vv = v[:n]
        valid = np.isfinite(tt) & np.isfinite(ii) & np.isfinite(vv)
        if valid.sum() >= 2:
            tt = tt[valid]
            pp = ii[valid] * vv[valid]
            out["integral_energy_signed_wh_from_vi"] = safe_float(trapezoid_integral(pp, tt) / 3600.0)
            out["integral_energy_abs_wh_from_vi"] = safe_float(trapezoid_integral(np.abs(pp), tt) / 3600.0)

    if len(p) >= 2:
        n = min(len(t), len(p))
        tt = t[:n]
        pp = p[:n]
        valid = np.isfinite(tt) & np.isfinite(pp)
        if valid.sum() >= 2:
            tt = tt[valid]
            pp = pp[valid]
            out["integral_power_signed_wh"] = safe_float(trapezoid_integral(pp, tt) / 3600.0)
            out["integral_power_abs_wh"] = safe_float(trapezoid_integral(np.abs(pp), tt) / 3600.0)

    return out


def make_stage_slices(t: np.ndarray, gap_indices: np.ndarray) -> List[Tuple[int, int]]:
    if t is None or len(t) == 0:
        return []

    n = len(t)

    if gap_indices is None or len(gap_indices) == 0:
        return [(0, n)]

    starts = [0] + [int(idx) + 1 for idx in gap_indices]
    ends = [int(idx) + 1 for idx in gap_indices] + [n]

    return [(s, e) for s, e in zip(starts, ends) if e > s]


def feature_record_from_vectors(
    base: Dict[str, Any],
    vectors: Dict[str, np.ndarray],
    feature_level: str,
    stage_index: int,
    stage_total: int,
    min_gap_seconds: float,
    gap_factor: float,
    fast_mode: bool = False,
) -> Dict[str, Any]:
    vectors = align_by_time(vectors)

    record = dict(base)
    record["feature_level"] = feature_level
    record["stage_index"] = int(stage_index)
    record["stage_total"] = int(stage_total)

    for col in VECTOR_COLUMNS:
        prefix = col.replace("_s", "").replace("_v", "").replace("_a", "").replace("_c", "").replace("_ah", "").replace("_wh", "").replace("_w", "")
        # Keep readable names.
        prefix_map = {
            "time": "time",
            "voltage": "voltage",
            "current": "current",
            "temperature": "temperature",
            "capacity": "capacity",
            "energy": "energy",
            "power": "power",
        }
        prefix = prefix_map.get(prefix, prefix)
        arr = vectors.get(col, np.asarray([], dtype=np.float64))
        record.update(numeric_stats(prefix, arr))

    t = vectors.get("time_s", np.asarray([], dtype=np.float64))

    for col, prefix in [
        ("voltage_v", "voltage"),
        ("current_a", "current"),
        ("temperature_c", "temperature"),
        ("capacity_ah", "capacity"),
        ("energy_wh", "energy"),
        ("power_w", "power"),
    ]:
        record.update(slope_vs_time(prefix, vectors.get(col, np.asarray([])), t, fast_mode=fast_mode))

    time_features, gap_indices = time_gap_features(
        t=t,
        min_gap_seconds=min_gap_seconds,
        gap_factor=gap_factor,
    )
    record.update(time_features)

    for col, prefix in [
        ("voltage_v", "voltage"),
        ("current_a", "current"),
        ("temperature_c", "temperature"),
        ("capacity_ah", "capacity"),
        ("energy_wh", "energy"),
        ("power_w", "power"),
    ]:
        record.update(boundary_jump_features(prefix, vectors.get(col, np.asarray([])), gap_indices))

    record.update(integration_features(vectors))

    # Differences between extracted/integrated quantities and input aggregate values, when available.
    if "input_phase_capacity_ah" in record and np.isfinite(record.get("integral_current_abs_ah", np.nan)):
        record["capacity_abs_integral_minus_input_phase_capacity_ah"] = (
            record["integral_current_abs_ah"] - safe_float(record.get("input_phase_capacity_ah"))
        )
    else:
        record["capacity_abs_integral_minus_input_phase_capacity_ah"] = np.nan

    if "input_phase_energy_wh" in record and np.isfinite(record.get("integral_energy_abs_wh_from_vi", np.nan)):
        record["energy_abs_integral_minus_input_phase_energy_wh"] = (
            record["integral_energy_abs_wh_from_vi"] - safe_float(record.get("input_phase_energy_wh"))
        )
    else:
        record["energy_abs_integral_minus_input_phase_energy_wh"] = np.nan

    return record


def build_base_record(
    row: Dict[str, Any],
    target_row: Dict[str, Any],
    input_path: str,
    phase_folder: str,
    source_row_id: int,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "source_parquet_file": input_path,
        "source_file_key": normalize_path_key(input_path),
        "source_row_id": int(source_row_id),
        "phase_folder": phase_folder,
    }

    for col in IDENTITY_COLUMNS:
        if col in row:
            base[col] = row[col]

    # Force standardized phase folder as a modeling dimension.
    base["phase_folder"] = phase_folder

    # Preserve scalar input columns with an input_ prefix.
    for col in SCALAR_INPUT_COLUMNS:
        if col in row:
            base[f"input_{col}"] = row[col]

    # Add rebuilt targets.
    for col in TARGET_OUTPUT_COLUMNS:
        if col in target_row:
            base[col] = target_row[col]

    return base


def process_input_file(job_dict: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    job = InputFile(**job_dict)

    input_path = Path(job.path)
    output_path = Path(job.output_path)
    done_path = Path(job.done_path)
    error_path = Path(job.error_path)

    resume = bool(options["resume"])
    force = bool(options["force"])
    target_table_path = Path(options["target_table_path"])
    compression = options["compression"]
    include_stages = bool(options["include_stages"])
    min_gap_seconds = float(options["min_gap_seconds"])
    gap_factor = float(options["gap_factor"])
    fast_mode = bool(options.get("fast_mode", False))

    result = {
        "job_id": job.job_id,
        "phase_folder": job.phase_folder,
        "dataset_name": job.dataset_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "unknown",
        "started_at": now_iso(),
        "finished_at": None,
        "input_rows": None,
        "output_rows": None,
        "error": None,
    }

    try:
        if resume and not force and file_is_valid(output_path) and file_is_valid(done_path):
            result["status"] = "skipped_done"
            result["finished_at"] = now_iso()
            return result

        if force:
            for p in [output_path, done_path, error_path]:
                if p.exists():
                    p.unlink()

        ensure_dir(output_path.parent)

        input_columns = read_parquet_columns(input_path)
        columns_to_read = [
            col for col in list(dict.fromkeys(IDENTITY_COLUMNS + TARGET_SOURCE_COLUMNS + SCALAR_INPUT_COLUMNS + VECTOR_COLUMNS))
            if col in input_columns
        ]

        df = pd.read_parquet(input_path, columns=columns_to_read, engine="fastparquet")
        input_rows = len(df)

        target_df = pd.read_parquet(target_table_path, engine="fastparquet")
        file_key = normalize_path_key(input_path)
        target_df = target_df[target_df["source_file_key"].astype(str) == file_key].copy()
        target_df = target_df.set_index("source_row_id", drop=False)
        target_records = target_df.to_dict(orient="index")

        if len(target_df) != input_rows:
            print(
                f"[WARNING] Target rows ({len(target_df)}) != input rows ({input_rows}) for {input_path}",
                file=sys.stderr,
            )

        tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")

        if tmp_path.exists():
            tmp_path.unlink()

        buffer: List[Dict[str, Any]] = []
        total_output_rows = 0
        schema_columns: Optional[List[str]] = None
        wrote_any = False
        batch_size = int(options["write_batch_size"])

        def flush_buffer() -> None:
            nonlocal buffer, total_output_rows, schema_columns, wrote_any

            if not buffer:
                return

            batch = pd.DataFrame(buffer)

            if schema_columns is None:
                schema_columns = list(batch.columns)
            else:
                for col in schema_columns:
                    if col not in batch.columns:
                        batch[col] = np.nan
                new_cols = [col for col in batch.columns if col not in schema_columns]
                if new_cols:
                    # Keep schema stable by extending it. This is safer when a rare column appears later.
                    schema_columns.extend(new_cols)
                batch = batch[schema_columns]

            for col in batch.select_dtypes(include=["float64"]).columns:
                batch[col] = pd.to_numeric(batch[col], downcast="float")

            for col in batch.select_dtypes(include=["int64"]).columns:
                batch[col] = pd.to_numeric(batch[col], downcast="integer")

            if wrote_any:
                # If schema has expanded, append can fail. To avoid silent corruption,
                # keep all rows in stable schema from the first batches. The code above
                # adds new cols but append schema cannot change after first write.
                # Therefore, new feature columns should not appear after first batch.
                fastparquet.write(str(tmp_path), batch, compression=compression, append=True)
            else:
                fastparquet.write(str(tmp_path), batch, compression=compression)
                wrote_any = True

            total_output_rows += len(batch)
            buffer = []
            del batch
            gc.collect()

        cols = list(df.columns)

        for source_row_id, row_tuple in enumerate(tqdm(df.itertuples(index=False, name=None), total=input_rows, desc=f"{job.dataset_name}-{job.phase_folder}", leave=False)):
            row = dict(zip(cols, row_tuple))

            target_row = target_records.get(source_row_id, {})

            base = build_base_record(
                row=row,
                target_row=target_row,
                input_path=str(input_path),
                phase_folder=job.phase_folder,
                source_row_id=source_row_id,
            )

            vectors = {col: parse_vector(row.get(col)) for col in VECTOR_COLUMNS if col in row}
            vectors = align_by_time(vectors)

            phase_record = feature_record_from_vectors(
                base=base,
                vectors=vectors,
                feature_level="phase",
                stage_index=-1,
                stage_total=-1,
                min_gap_seconds=min_gap_seconds,
                gap_factor=gap_factor,
                fast_mode=fast_mode,
            )
            buffer.append(phase_record)

            if include_stages:
                t = vectors.get("time_s", np.asarray([], dtype=np.float64))
                _, gap_indices = time_gap_features(t, min_gap_seconds=min_gap_seconds, gap_factor=gap_factor)
                slices = make_stage_slices(t, gap_indices)

                for stage_index, (start, end) in enumerate(slices):
                    stage_vectors: Dict[str, np.ndarray] = {}
                    for col, arr in vectors.items():
                        if isinstance(arr, np.ndarray) and len(arr) >= end:
                            stage_vectors[col] = arr[start:end]
                        else:
                            stage_vectors[col] = arr

                    stage_record = feature_record_from_vectors(
                        base=base,
                        vectors=stage_vectors,
                        feature_level="stage",
                        stage_index=stage_index,
                        stage_total=len(slices),
                        min_gap_seconds=min_gap_seconds,
                        gap_factor=gap_factor,
                        fast_mode=fast_mode,
                    )
                    stage_record["stage_start_point_index"] = int(start)
                    stage_record["stage_end_point_index"] = int(end - 1)
                    buffer.append(stage_record)

            if len(buffer) >= batch_size:
                flush_buffer()

        flush_buffer()

        if output_path.exists():
            output_path.unlink()

        tmp_path.replace(output_path)

        done_payload = {
            "job_id": job.job_id,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "phase_folder": job.phase_folder,
            "dataset_name": job.dataset_name,
            "finished_at": now_iso(),
            "input_rows": int(input_rows),
            "output_rows": int(total_output_rows),
        }
        done_path.write_text(json.dumps(done_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if error_path.exists():
            error_path.unlink()

        result.update(
            {
                "status": "done",
                "finished_at": now_iso(),
                "input_rows": int(input_rows),
                "output_rows": int(total_output_rows),
            }
        )

        del df
        del target_df
        gc.collect()

        return result

    except Exception as exc:
        tb = traceback.format_exc()
        ensure_dir(error_path.parent)
        error_path.write_text(tb, encoding="utf-8")

        result.update(
            {
                "status": "failed",
                "finished_at": now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return result


# =============================================================================
# Reports
# =============================================================================

def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def create_dataset_summary(targets: pd.DataFrame) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame()

    cycle_col = "cycle_index" if "cycle_index" in targets.columns else "cycle_number"

    base = (
        targets.groupby(["dataset", "phase_folder"], dropna=False)
        .agg(
            rows=("source_row_id", "count"),
            batteries=("battery_id", "nunique"),
            cycles=(cycle_col, "nunique"),
            capacity_valid_rate=("target_capacity_valid", "mean"),
            median_initial_capacity_ah=("target_initial_capacity_ah", "median"),
            median_capacity_ah=("target_capacity_ah", "median"),
        )
        .reset_index()
    )

    # EOL flags are repeated for every row of the same battery/phase trajectory.
    # Therefore, count unique batteries, not rows.
    battery_eol = (
        targets.groupby(["dataset", "phase_folder", "battery_id"], dropna=False)
        .agg(
            eol80_found=("target_eol80_found", "max"),
            eol70_found=("target_eol70_found", "max"),
        )
        .reset_index()
    )

    eol_counts = (
        battery_eol.groupby(["dataset", "phase_folder"], dropna=False)
        .agg(
            eol80_batteries=("eol80_found", "sum"),
            eol70_batteries=("eol70_found", "sum"),
        )
        .reset_index()
    )

    summary = (
        base.merge(eol_counts, on=["dataset", "phase_folder"], how="left")
        .sort_values(["dataset", "phase_folder"])
        .reset_index(drop=True)
    )

    return summary


def write_markdown_report(
    path: Path,
    input_files: Sequence[InputFile],
    targets: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    job_results: Optional[pd.DataFrame] = None,
) -> None:
    def md_preview(df: pd.DataFrame, n: int = 20) -> str:
        if df is None or df.empty:
            return "_No data._"
        try:
            return df.head(n).to_markdown(index=False)
        except Exception:
            return df.head(n).to_string(index=False)

    lines: List[str] = []
    lines.append("# Hierarchical Features v3 Extraction Report")
    lines.append("")
    lines.append(f"Generated at: `{now_iso()}`")
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Input files: `{len(input_files)}`")
    lines.append(f"- Target rows: `{len(targets)}`")
    lines.append("")
    lines.append("## Dataset and phase summary")
    lines.append("")
    lines.append(md_preview(dataset_summary, 50))
    lines.append("")
    lines.append("## Capacity life summary preview")
    lines.append("")
    preview_cols = [
        "dataset",
        "battery_id",
        "phase_folder",
        "n_rows",
        "initial_capacity_ah",
        "capacity_first_ah",
        "capacity_last_ah",
        "capacity_end_to_initial_ratio",
        "eol80_found",
        "eol80_cycle_index",
        "eol70_found",
        "eol70_cycle_index",
        "trend_spearman",
        "capacity_validation_status",
        "capacity_validation_notes",
    ]
    preview_cols = [c for c in preview_cols if c in capacity_summary.columns]
    lines.append(md_preview(capacity_summary[preview_cols] if not capacity_summary.empty else capacity_summary, 50))
    lines.append("")
    lines.append("## Job results")
    lines.append("")
    lines.append(md_preview(job_results if job_results is not None else pd.DataFrame(), 50))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Charge and discharge were treated as independent life trajectories.")
    lines.append("- `target_soh_recomputed` is calculated as `target_capacity_ah / target_initial_capacity_ah`.")
    lines.append("- EOL80 and EOL70 are based on the first sustained crossing below 80% or 70% of the recomputed initial capacity.")
    lines.append("- Both signed and clipped RUL columns are produced. For modeling, prefer clipped columns unless you intentionally model post-EOL behavior.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Runner
# =============================================================================

def run_parallel_jobs(
    jobs: Sequence[InputFile],
    reports_dir: Path,
    options: Dict[str, Any],
    workers: int,
) -> pd.DataFrame:
    ensure_dir(reports_dir)

    job_records = [asdict(job) for job in jobs]
    write_csv(pd.DataFrame(job_records), reports_dir / "job_plan.csv")

    status_path = reports_dir / "status.jsonl"

    results: List[Dict[str, Any]] = []

    if workers <= 1:
        for job in tqdm(job_records, total=len(job_records), desc="Extracting features"):
            result = process_input_file(job, options)
            results.append(result)
            write_jsonl(status_path, result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_input_file, job, options) for job in job_records]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
                result = future.result()
                results.append(result)
                write_jsonl(status_path, result)

    results_df = pd.DataFrame(results)
    write_csv(results_df, reports_dir / "summary.csv")

    failures = results_df[results_df["status"] == "failed"] if not results_df.empty else pd.DataFrame()
    if not failures.empty:
        write_csv(failures, reports_dir / "failures.csv")

    return results_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vector-aware hierarchical feature extraction v3 for charge/discharge interim files."
    )

    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)

    parser.add_argument(
        "--phase-dirs",
        nargs="*",
        default=list(PHASE_FOLDERS),
        help="Phase folders inside input root. Default: charge discharge.",
    )

    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset to process. Can be repeated or comma-separated. Example: --dataset HUST --dataset MATR",
    )

    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)

    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "gzip", "brotli", "none", "None"],
    )

    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=5000,
        help="Number of feature records buffered before writing to Parquet.",
    )

    parser.add_argument(
        "--include-stages",
        action="store_true",
        default=True,
        help="Include stage-level features split by time gaps. Enabled by default.",
    )

    parser.add_argument(
        "--no-stages",
        dest="include_stages",
        action="store_false",
        help="Disable stage-level rows and write only phase-level rows.",
    )

    parser.add_argument("--min-gap-seconds", type=float, default=60.0)
    parser.add_argument("--gap-factor", type=float, default=20.0)

    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help=(
            "Optimize the hot loop for very large files: use closed-form slopes, "
            "avoid pandas where possible, and store Pearson instead of Spearman in *_corr_vs_time."
        ),
    )

    parser.add_argument(
        "--initial-capacity-method",
        choices=["early_median", "early_mean", "early_max"],
        default="early_median",
    )

    parser.add_argument("--initial-window", type=int, default=5)
    parser.add_argument("--initial-fraction", type=float, default=0.02)
    parser.add_argument("--eol-rolling-window", type=int, default=5)
    parser.add_argument("--eol-consecutive", type=int, default=3)
    parser.add_argument("--eol-min-position", type=int, default=3)

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build targets and job plan. Do not extract vector features.",
    )

    return parser


def parse_dataset_args(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(part.upper())
    return list(dict.fromkeys(out))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    reports_root: Path = args.reports_root
    reports_dir = reports_root / run_id()

    ensure_dir(output_root)
    ensure_dir(reports_dir)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    datasets = parse_dataset_args(args.dataset)

    input_files = discover_input_files(
        input_root=input_root,
        output_root=output_root,
        phase_dirs=args.phase_dirs,
        datasets=datasets,
        max_files=args.max_files,
    )

    if not input_files:
        raise FileNotFoundError("No input Parquet files found in charge/discharge folders.")

    print("Feature extraction v3 plan")
    print("-" * 60)
    print(f"Input root:     {input_root}")
    print(f"Output root:    {output_root}")
    print(f"Reports dir:    {reports_dir}")
    print(f"Input files:    {len(input_files)}")
    print(f"Phase dirs:     {', '.join(args.phase_dirs)}")
    print(f"Datasets:       {', '.join(datasets) if datasets else 'all'}")
    print(f"Workers:        {args.workers}")
    print(f"Resume:         {args.resume}")
    print(f"Force:          {args.force}")
    print(f"Include stages: {args.include_stages}")
    print(f"Fast mode:      {args.fast_mode}")

    write_csv(pd.DataFrame([asdict(f) for f in input_files]), reports_dir / "input_files.csv")

    print("\nLoading scalar index for target reconstruction...")
    scalar_df = load_scalar_index(input_files)
    write_csv(scalar_df.head(1000), reports_dir / "scalar_index_preview.csv")

    print("\nRebuilding capacity, SOH and RUL targets...")
    targets, capacity_summary = build_targets(
        scalar_df=scalar_df,
        initial_method=args.initial_capacity_method,
        initial_window=args.initial_window,
        initial_fraction=args.initial_fraction,
        eol_rolling_window=args.eol_rolling_window,
        eol_consecutive=args.eol_consecutive,
        eol_min_position=args.eol_min_position,
    )

    targets_dir = output_root / "_targets"
    ensure_dir(targets_dir)

    target_table_path = targets_dir / "targets_v3.parquet"
    capacity_summary_path = targets_dir / "capacity_life_summary_v3.csv"

    targets.to_parquet(target_table_path, engine="fastparquet", compression="snappy", index=False)
    write_csv(targets, reports_dir / "targets_v3.csv")
    write_csv(capacity_summary, reports_dir / "capacity_life_summary_v3.csv")
    write_csv(capacity_summary, capacity_summary_path)

    dataset_summary = create_dataset_summary(targets)
    write_csv(dataset_summary, reports_dir / "dataset_phase_target_summary.csv")

    metadata = {
        "generated_at": now_iso(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "reports_dir": str(reports_dir),
        "input_files": len(input_files),
        "targets_rows": len(targets),
        "initial_capacity_method": args.initial_capacity_method,
        "initial_window": args.initial_window,
        "initial_fraction": args.initial_fraction,
        "eol_rolling_window": args.eol_rolling_window,
        "eol_consecutive": args.eol_consecutive,
        "eol_min_position": args.eol_min_position,
    }
    (reports_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        write_markdown_report(
            path=reports_dir / "extraction_report.md",
            input_files=input_files,
            targets=targets,
            capacity_summary=capacity_summary,
            dataset_summary=dataset_summary,
            job_results=None,
        )
        print("\nDry run finished. Targets and reports were created, but features were not extracted.")
        print(f"Reports: {reports_dir}")
        return

    compression = None if str(args.compression).lower() == "none" else args.compression

    options = {
        "resume": bool(args.resume),
        "force": bool(args.force),
        "target_table_path": str(target_table_path),
        "compression": compression,
        "include_stages": bool(args.include_stages),
        "min_gap_seconds": float(args.min_gap_seconds),
        "gap_factor": float(args.gap_factor),
        "write_batch_size": int(args.write_batch_size),
        "fast_mode": bool(args.fast_mode),
    }

    print("\nStarting vector-aware feature extraction...")
    results_df = run_parallel_jobs(
        jobs=input_files,
        reports_dir=reports_dir,
        options=options,
        workers=max(1, int(args.workers)),
    )

    write_markdown_report(
        path=reports_dir / "extraction_report.md",
        input_files=input_files,
        targets=targets,
        capacity_summary=capacity_summary,
        dataset_summary=dataset_summary,
        job_results=results_df,
    )

    print("\nFeature extraction v3 finished.")
    print("-" * 60)
    print(f"Reports:      {reports_dir}")
    print(f"Features:     {output_root}")
    print(f"Targets:      {target_table_path}")
    print(f"Cap. summary: {capacity_summary_path}")

    if not results_df.empty:
        print("\nStatus counts:")
        print(results_df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on Windows.
    main()
