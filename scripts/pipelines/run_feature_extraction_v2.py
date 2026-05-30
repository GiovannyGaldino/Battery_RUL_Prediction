"""
run_feature_extraction_v2.py

Resumable hierarchical feature extraction for battery datasets.

Main characteristics
--------------------
- Reads processed/interim Parquet files.
- Writes one feature Parquet per input file/job.
- Uses a ".done" marker to resume without wasting previous work.
- Uses controlled multiprocessing with ProcessPoolExecutor.
- Uses tqdm progress bars.
- Uses fastparquet as the Parquet engine.
- Avoids multiple workers writing to the same output file.
- Skips auxiliary folders named "charge" and "discharge" when using --all, unless they are explicitly selected with --dataset.
- Detects phase segments using time gaps.
- Generates execution reports: job_plan.csv, summary.csv, failures.csv, status.jsonl, output_index.csv.

Typical usage from the project root
-----------------------------------

python scripts/pipelines/run_feature_extraction_v2.py ^
  --input-root "C:/Artigo Baterias de Litio/Batteries/data/interim" ^
  --output-root "C:/Artigo Baterias de Litio/Batteries/data/processed/features_hierarchical_v2" ^
  --reports-root "C:/Artigo Baterias de Litio/Batteries/reports/feature_extraction_v2" ^
  --all ^
  --workers 2 ^
  --resume ^
  --suffixes ".parquet"

PowerShell one-line example
---------------------------

& C:\\Users\\Giovanny\\anaconda3\\envs\\batteries\\python.exe ".\\scripts\\pipelines\\run_feature_extraction_v2.py" --input-root "C:\\Artigo Baterias de Litio\\Batteries\\data\\interim" --output-root "C:\\Artigo Baterias de Litio\\Batteries\\data\\processed\\features_hierarchical_v2" --reports-root "C:\\Artigo Baterias de Litio\\Batteries\\reports\\feature_extraction_v2" --all --workers 2 --resume --suffixes ".parquet"

Notes
-----
Use --workers 1 if RAM usage is too high.
Do not use --force unless you intentionally want to reprocess already completed jobs.
"""

from __future__ import annotations

import argparse
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
    raise RuntimeError(
        "fastparquet is required. Install it with: pip install fastparquet"
    ) from exc


# =============================================================================
# Default paths
# =============================================================================

DEFAULT_INPUT_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/interim")
DEFAULT_OUTPUT_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/processed/features_hierarchical_v2")
DEFAULT_REPORTS_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/reports/feature_extraction_v2")

# Process only Parquet by default. CSV files in data/interim are often auxiliary
# reports, rejected-cycle lists, anomaly summaries, etc.
DEFAULT_SUFFIXES = (".parquet",)

# These folders are usually phase-level folders, not real datasets.
DEFAULT_EXCLUDED_DATASET_DIRS = {"charge", "discharge"}


# =============================================================================
# Column aliases
# =============================================================================

COLUMN_ALIASES: Dict[str, List[str]] = {
    "dataset": ["dataset", "database", "source", "base"],
    "battery_id": [
        "battery_id", "battery", "cell_id", "cell", "cell_name",
        "battery_name", "batteryid", "cellid"
    ],
    "cycle": [
        "cycle", "cycle_id", "cycle_index", "cycle_idx", "cycle_number",
        "cycle_no", "cycle_num", "n_cycle"
    ],
    "phase": [
        "phase", "step_type", "step", "operation", "mode", "state",
        "type", "status"
    ],
    "time": [
        "time", "time_s", "time_sec", "time_seconds", "test_time",
        "test_time_s", "elapsed_time", "elapsed_time_s", "relative_time",
        "relative_time_s", "t"
    ],
    "voltage": [
        "voltage", "voltage_v", "v", "u", "voltage_measured",
        "terminal_voltage"
    ],
    "current": ["current", "current_a", "i", "current_measured"],
    "temperature": [
        "temperature", "temperature_c", "temp", "temp_c",
        "ambient_temperature", "temperature_measured"
    ],
    "capacity": [
        "capacity", "capacity_ah", "charge_capacity",
        "discharge_capacity", "q", "q_ah", "ah"
    ],
    "energy": [
        "energy", "energy_wh", "charge_energy",
        "discharge_energy", "wh"
    ],
    "soh": ["soh", "state_of_health"],
    "rul": ["rul", "remaining_useful_life"],
}

NUMERIC_COLUMNS = [
    "cycle", "time", "voltage", "current", "temperature",
    "capacity", "energy", "soh", "rul"
]

TEXT_COLUMNS = ["dataset", "battery_id", "phase"]


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Job:
    job_id: str
    dataset: str
    input_path: str
    output_path: str
    done_path: str
    error_path: str


# =============================================================================
# Generic utilities
# =============================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def safe_filename(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value))
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "file")[:max_len]


def short_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def file_is_valid(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def parse_list_argument(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    items: List[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                items.append(part)
    return list(dict.fromkeys(items))


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


# =============================================================================
# Dataset and job discovery
# =============================================================================

def discover_datasets(
    input_root: Path,
    requested_datasets: Sequence[str],
    use_all: bool,
    excluded_dirs: Sequence[str],
) -> List[Tuple[str, Path]]:
    """Return a list of (dataset_name, dataset_path)."""

    excluded = {x.strip().lower() for x in excluded_dirs if x.strip()}

    if requested_datasets:
        output: List[Tuple[str, Path]] = []
        for name in requested_datasets:
            path_inside_root = input_root / name
            path_direct = Path(name)

            if path_inside_root.exists():
                output.append((path_inside_root.name, path_inside_root))
            elif path_direct.exists():
                output.append((path_direct.name, path_direct))
            else:
                print(f"[WARNING] Dataset path not found: {name}", file=sys.stderr)

        return output

    if not input_root.exists():
        return []

    child_dirs = [p for p in input_root.iterdir() if p.is_dir()]

    if child_dirs:
        output = []
        for path in sorted(child_dirs, key=lambda p: p.name.lower()):
            if path.name.lower() in excluded:
                print(f"[INFO] Skipping auxiliary directory in --all mode: {path.name}")
                continue
            output.append((path.name, path))
        return output

    return [(input_root.name or "default", input_root)]


def discover_input_files(dataset_path: Path, suffixes: Sequence[str]) -> List[Path]:
    suffix_set = {s.lower() for s in suffixes}
    files: List[Path] = []

    for path in dataset_path.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in suffix_set:
            files.append(path)

    return sorted(files, key=lambda p: str(p).lower())


def build_jobs(
    input_root: Path,
    output_root: Path,
    datasets: Sequence[Tuple[str, Path]],
    suffixes: Sequence[str],
    max_files: Optional[int],
) -> List[Job]:
    jobs: List[Job] = []

    for dataset_name, dataset_path in datasets:
        files = discover_input_files(dataset_path, suffixes)

        if max_files is not None:
            files = files[:max_files]

        for input_path in files:
            try:
                relative_path = input_path.relative_to(input_root)
            except ValueError:
                relative_path = Path(input_path.name)

            relative_text = str(relative_path).replace("\\", "/")
            job_hash = short_hash(f"{dataset_name}/{relative_text}")
            stem = safe_filename(input_path.stem)

            output_dir = output_root / safe_filename(dataset_name) / "parts"
            output_path = output_dir / f"{stem}__{job_hash}.features.parquet"
            done_path = output_dir / f"{stem}__{job_hash}.features.parquet.done"
            error_path = output_dir / f"{stem}__{job_hash}.error.txt"

            jobs.append(
                Job(
                    job_id=job_hash,
                    dataset=dataset_name,
                    input_path=str(input_path),
                    output_path=str(output_path),
                    done_path=str(done_path),
                    error_path=str(error_path),
                )
            )

    return jobs


# =============================================================================
# Reading and standardization
# =============================================================================

def get_parquet_columns(path: Path) -> List[str]:
    parquet_file = ParquetFile(str(path))
    return list(parquet_file.columns)


def infer_column_map(columns: Sequence[str]) -> Dict[str, str]:
    normalized_to_original = {normalize_name(col): col for col in columns}
    mapping: Dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = normalize_name(alias)
            if key in normalized_to_original:
                mapping[canonical] = normalized_to_original[key]
                break

    return mapping


def infer_columns_to_read(columns: Sequence[str]) -> List[str]:
    mapping = infer_column_map(columns)
    selected = list(dict.fromkeys(mapping.values()))

    # If the input does not match expected aliases, read all columns to avoid
    # silent data loss.
    if len(selected) < 3:
        return list(columns)

    return selected


def read_input_table(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Only Parquet is supported by this compact version. Got: {path}")

    columns = get_parquet_columns(path)
    selected_columns = infer_columns_to_read(columns)
    column_map = infer_column_map(columns)

    df = pd.read_parquet(path, columns=selected_columns, engine="fastparquet")
    return df, column_map


def infer_phase_from_current(df: pd.DataFrame) -> pd.Series:
    if "current" not in df.columns:
        return pd.Series(["unknown"] * len(df), index=df.index, dtype="string")

    current = pd.to_numeric(df["current"], errors="coerce")
    abs_current = current.abs().replace([np.inf, -np.inf], np.nan).dropna()

    if abs_current.empty:
        threshold = 1e-9
    else:
        threshold = max(float(abs_current.quantile(0.05)) * 0.01, 1e-9)

    phase = np.where(
        current > threshold,
        "charge",
        np.where(current < -threshold, "discharge", "rest"),
    )

    return pd.Series(phase, index=df.index, dtype="string")


def normalize_phase_series(phase: pd.Series, current: Optional[pd.Series]) -> pd.Series:
    s = phase.astype("string").str.strip().str.lower().fillna("unknown")

    discharge_mask = s.str.contains(
        "discharge|descarga|dchg|decharge", regex=True, na=False
    )
    charge_mask = s.str.contains(
        "charge|carga|chg", regex=True, na=False
    ) & ~discharge_mask
    rest_mask = s.str.contains(
        "rest|repouso|relax|pause|pausa|idle|calendar", regex=True, na=False
    )

    out = s.copy()
    out[discharge_mask] = "discharge"
    out[charge_mask] = "charge"
    out[rest_mask] = "rest"

    unknown_mask = out.isna() | (out == "") | (out == "unknown")
    if current is not None and unknown_mask.any():
        inferred = infer_phase_from_current(pd.DataFrame({"current": current}))
        out[unknown_mask] = inferred[unknown_mask]

    return out.astype("string")


def standardize_dataframe(df: pd.DataFrame, column_map: Dict[str, str], job: Job) -> pd.DataFrame:
    rename = {
        original: canonical
        for canonical, original in column_map.items()
        if original in df.columns
    }

    df = df.rename(columns=rename)

    if "dataset" not in df.columns:
        df["dataset"] = job.dataset

    if "battery_id" not in df.columns:
        df["battery_id"] = Path(job.input_path).stem

    if "cycle" not in df.columns:
        df["cycle"] = 0

    if "time" not in df.columns:
        df["time"] = np.arange(len(df), dtype=np.float32)
        df["time_is_synthetic"] = True
    else:
        df["time_is_synthetic"] = False

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "phase" not in df.columns:
        df["phase"] = infer_phase_from_current(df)
    else:
        current = df["current"] if "current" in df.columns else None
        df["phase"] = normalize_phase_series(df["phase"], current)

    # Memory reduction.
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in df.select_dtypes(include=["int64", "int32", "int16"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("unknown").astype("category")

    keep_columns = [
        "dataset", "battery_id", "cycle", "phase", "time", "time_is_synthetic",
        "voltage", "current", "temperature", "capacity", "energy", "soh", "rul"
    ]
    keep_columns = [col for col in keep_columns if col in df.columns]

    return df[keep_columns]


# =============================================================================
# Feature calculation
# =============================================================================

def finite_array(values: Any) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def numeric_stats(values: Any, prefix: str) -> Dict[str, float]:
    arr = finite_array(values)

    output = {
        f"{prefix}_n_valid": float(arr.size),
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
    }

    if arr.size == 0:
        return output

    output[f"{prefix}_first"] = safe_float(arr[0])
    output[f"{prefix}_last"] = safe_float(arr[-1])
    output[f"{prefix}_min"] = safe_float(np.min(arr))
    output[f"{prefix}_max"] = safe_float(np.max(arr))
    output[f"{prefix}_mean"] = safe_float(np.mean(arr))
    output[f"{prefix}_std"] = safe_float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    output[f"{prefix}_median"] = safe_float(np.median(arr))
    output[f"{prefix}_q05"] = safe_float(np.quantile(arr, 0.05))
    output[f"{prefix}_q95"] = safe_float(np.quantile(arr, 0.95))
    output[f"{prefix}_delta"] = safe_float(arr[-1] - arr[0])
    output[f"{prefix}_range"] = safe_float(np.max(arr) - np.min(arr))

    return output


def ordered_group(group: pd.DataFrame) -> pd.DataFrame:
    if "time" not in group.columns:
        return group
    return group.sort_values("time", kind="mergesort")


def time_gap_info(
    group: pd.DataFrame,
    min_gap_seconds: float,
    gap_factor: float,
) -> Dict[str, Any]:
    if "time" not in group.columns or len(group) < 2:
        return {
            "duration_s": 0.0 if len(group) > 0 else np.nan,
            "dt_median_s": np.nan,
            "dt_mean_s": np.nan,
            "dt_max_s": np.nan,
            "gap_threshold_s": min_gap_seconds,
            "gap_count": 0,
            "segment_count": 1 if len(group) > 0 else 0,
            "gap_total_s": 0.0,
            "gap_max_s": 0.0,
            "gap_indices": np.array([], dtype=int),
        }

    g = ordered_group(group)
    t = pd.to_numeric(g["time"], errors="coerce").to_numpy(dtype=np.float64)

    if np.isfinite(t).sum() < 2:
        return {
            "duration_s": np.nan,
            "dt_median_s": np.nan,
            "dt_mean_s": np.nan,
            "dt_max_s": np.nan,
            "gap_threshold_s": min_gap_seconds,
            "gap_count": 0,
            "segment_count": 1,
            "gap_total_s": 0.0,
            "gap_max_s": 0.0,
            "gap_indices": np.array([], dtype=int),
        }

    dt = np.diff(t)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]

    if positive_dt.size == 0:
        threshold = min_gap_seconds
        gap_indices = np.array([], dtype=int)
    else:
        dt_median = float(np.median(positive_dt))
        threshold = max(float(min_gap_seconds), float(gap_factor) * dt_median)
        gap_indices = np.where(np.isfinite(dt) & (dt > threshold))[0]

    gap_values = dt[gap_indices] if gap_indices.size else np.array([], dtype=np.float64)

    return {
        "duration_s": safe_float(np.nanmax(t) - np.nanmin(t)),
        "dt_median_s": safe_float(np.median(positive_dt)) if positive_dt.size else np.nan,
        "dt_mean_s": safe_float(np.mean(positive_dt)) if positive_dt.size else np.nan,
        "dt_max_s": safe_float(np.max(positive_dt)) if positive_dt.size else np.nan,
        "gap_threshold_s": safe_float(threshold),
        "gap_count": int(gap_indices.size),
        "segment_count": int(gap_indices.size + 1),
        "gap_total_s": safe_float(np.sum(gap_values)) if gap_values.size else 0.0,
        "gap_max_s": safe_float(np.max(gap_values)) if gap_values.size else 0.0,
        "gap_indices": gap_indices,
    }


def boundary_jump_stats(
    group: pd.DataFrame,
    gap_indices: np.ndarray,
    column: str,
    prefix: str,
) -> Dict[str, float]:
    output = {
        f"{prefix}_boundary_jump_n": 0.0,
        f"{prefix}_boundary_jump_mean": np.nan,
        f"{prefix}_boundary_jump_abs_mean": np.nan,
        f"{prefix}_boundary_jump_abs_max": np.nan,
        f"{prefix}_boundary_jump_abs_sum": np.nan,
    }

    if column not in group.columns or gap_indices.size == 0:
        return output

    g = ordered_group(group)
    values = pd.to_numeric(g[column], errors="coerce").to_numpy(dtype=np.float64)
    jumps: List[float] = []

    for idx in gap_indices:
        if idx < 0 or idx + 1 >= len(values):
            continue

        before = values[idx]
        after = values[idx + 1]

        if np.isfinite(before) and np.isfinite(after):
            jumps.append(float(after - before))

    if not jumps:
        return output

    arr = np.asarray(jumps, dtype=np.float64)
    abs_arr = np.abs(arr)

    output[f"{prefix}_boundary_jump_n"] = float(arr.size)
    output[f"{prefix}_boundary_jump_mean"] = safe_float(np.mean(arr))
    output[f"{prefix}_boundary_jump_abs_mean"] = safe_float(np.mean(abs_arr))
    output[f"{prefix}_boundary_jump_abs_max"] = safe_float(np.max(abs_arr))
    output[f"{prefix}_boundary_jump_abs_sum"] = safe_float(np.sum(abs_arr))

    return output


def integration_features(group: pd.DataFrame) -> Dict[str, float]:
    output = {
        "integrated_capacity_signed_ah": np.nan,
        "integrated_capacity_abs_ah": np.nan,
        "integrated_energy_signed_wh": np.nan,
        "integrated_energy_abs_wh": np.nan,
    }

    if "time" not in group.columns or "current" not in group.columns or len(group) < 2:
        return output

    g = ordered_group(group)
    t = pd.to_numeric(g["time"], errors="coerce").to_numpy(dtype=np.float64)
    i = pd.to_numeric(g["current"], errors="coerce").to_numpy(dtype=np.float64)

    valid = np.isfinite(t) & np.isfinite(i)
    if valid.sum() < 2:
        return output

    t = t[valid]
    i = i[valid]

    dt = np.diff(t)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)
    i_mid = 0.5 * (i[:-1] + i[1:])

    output["integrated_capacity_signed_ah"] = safe_float(np.sum(i_mid * dt) / 3600.0)
    output["integrated_capacity_abs_ah"] = safe_float(np.sum(np.abs(i_mid) * dt) / 3600.0)

    if "voltage" in g.columns:
        v = pd.to_numeric(g.loc[valid, "voltage"], errors="coerce").to_numpy(dtype=np.float64)

        if np.isfinite(v).sum() >= 2:
            p = v * i
            p_mid = 0.5 * (p[:-1] + p[1:])
            output["integrated_energy_signed_wh"] = safe_float(np.sum(p_mid * dt) / 3600.0)
            output["integrated_energy_abs_wh"] = safe_float(np.sum(np.abs(p_mid) * dt) / 3600.0)

    return output


def make_feature_row(
    group: pd.DataFrame,
    level: str,
    dataset: Any,
    battery_id: Any,
    cycle: Any,
    phase: Any,
    segment_index: int,
    min_gap_seconds: float,
    gap_factor: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "feature_level": level,
        "dataset": str(dataset),
        "battery_id": str(battery_id),
        "cycle": safe_float(cycle),
        "phase": str(phase),
        "segment_index": int(segment_index),
        "n_samples": int(len(group)),
        "time_is_synthetic": bool(group["time_is_synthetic"].iloc[0])
        if "time_is_synthetic" in group.columns and len(group) else False,
    }

    gap = time_gap_info(group, min_gap_seconds=min_gap_seconds, gap_factor=gap_factor)
    gap_indices = gap.pop("gap_indices")
    row.update(gap)

    for col in ["time", "voltage", "current", "temperature", "capacity", "energy", "soh", "rul"]:
        if col in group.columns:
            row.update(numeric_stats(group[col], col))

    row.update(boundary_jump_stats(group, gap_indices, "voltage", "voltage"))
    row.update(boundary_jump_stats(group, gap_indices, "current", "current"))
    row.update(integration_features(group))

    return row


def segment_slices(
    group: pd.DataFrame,
    min_gap_seconds: float,
    gap_factor: float,
) -> List[pd.DataFrame]:
    if len(group) <= 1:
        return [group]

    g = ordered_group(group)
    gap = time_gap_info(g, min_gap_seconds=min_gap_seconds, gap_factor=gap_factor)
    gap_indices = gap["gap_indices"]

    if gap_indices.size == 0:
        return [g]

    starts = [0] + [int(idx) + 1 for idx in gap_indices]
    ends = [int(idx) + 1 for idx in gap_indices] + [len(g)]

    segments = [g.iloc[start:end] for start, end in zip(starts, ends) if end > start]
    return segments if segments else [g]


def extract_hierarchical_features(
    df: pd.DataFrame,
    min_gap_seconds: float,
    gap_factor: float,
    inner_progress: bool,
) -> pd.DataFrame:
    required = {"dataset", "battery_id", "cycle", "phase", "time"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}")

    rows: List[Dict[str, Any]] = []

    cycle_group_cols = ["dataset", "battery_id", "cycle"]
    phase_group_cols = ["dataset", "battery_id", "cycle", "phase"]

    cycle_grouped = df.groupby(cycle_group_cols, observed=True, sort=False)
    cycle_iter: Iterable[Any] = cycle_grouped

    if inner_progress:
        cycle_iter = tqdm(cycle_grouped, total=cycle_grouped.ngroups, desc="Cycles", leave=False)

    for (dataset, battery_id, cycle), group in cycle_iter:
        rows.append(
            make_feature_row(
                group=group,
                level="cycle",
                dataset=dataset,
                battery_id=battery_id,
                cycle=cycle,
                phase="all",
                segment_index=-1,
                min_gap_seconds=min_gap_seconds,
                gap_factor=gap_factor,
            )
        )

    phase_grouped = df.groupby(phase_group_cols, observed=True, sort=False)
    phase_iter: Iterable[Any] = phase_grouped

    if inner_progress:
        phase_iter = tqdm(phase_grouped, total=phase_grouped.ngroups, desc="Phases", leave=False)

    for (dataset, battery_id, cycle, phase), group in phase_iter:
        rows.append(
            make_feature_row(
                group=group,
                level="phase",
                dataset=dataset,
                battery_id=battery_id,
                cycle=cycle,
                phase=phase,
                segment_index=-1,
                min_gap_seconds=min_gap_seconds,
                gap_factor=gap_factor,
            )
        )

        segments = segment_slices(group, min_gap_seconds=min_gap_seconds, gap_factor=gap_factor)

        for segment_index, segment in enumerate(segments):
            row = make_feature_row(
                group=segment,
                level="phase_segment",
                dataset=dataset,
                battery_id=battery_id,
                cycle=cycle,
                phase=phase,
                segment_index=segment_index,
                min_gap_seconds=min_gap_seconds,
                gap_factor=gap_factor,
            )
            row["segment_total_in_phase"] = len(segments)
            rows.append(row)

    features = pd.DataFrame(rows)

    for col in features.select_dtypes(include=["float64"]).columns:
        features[col] = pd.to_numeric(features[col], downcast="float")

    for col in features.select_dtypes(include=["int64"]).columns:
        features[col] = pd.to_numeric(features[col], downcast="integer")

    for col in ["feature_level", "dataset", "battery_id", "phase"]:
        if col in features.columns:
            features[col] = features[col].astype("string")

    return features


# =============================================================================
# Job processing
# =============================================================================

def process_job(job_dict: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    job = Job(**job_dict)

    input_path = Path(job.input_path)
    output_path = Path(job.output_path)
    done_path = Path(job.done_path)
    error_path = Path(job.error_path)

    resume = bool(options["resume"])
    force = bool(options["force"])
    compression = options["compression"]
    min_gap_seconds = float(options["min_gap_seconds"])
    gap_factor = float(options["gap_factor"])
    inner_progress = bool(options["inner_progress"])

    result: Dict[str, Any] = {
        "job_id": job.job_id,
        "dataset": job.dataset,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "unknown",
        "started_at": now_iso(),
        "finished_at": None,
        "input_rows": None,
        "feature_rows": None,
        "error": None,
    }

    try:
        if resume and not force and file_is_valid(output_path) and file_is_valid(done_path):
            result["status"] = "skipped_done"
            result["finished_at"] = now_iso()
            return result

        if force:
            for path in [output_path, done_path, error_path]:
                if path.exists():
                    path.unlink()

        ensure_dir(output_path.parent)

        df_raw, column_map = read_input_table(input_path)
        input_rows = len(df_raw)

        df = standardize_dataframe(df_raw, column_map=column_map, job=job)
        del df_raw
        gc.collect()

        features = extract_hierarchical_features(
            df,
            min_gap_seconds=min_gap_seconds,
            gap_factor=gap_factor,
            inner_progress=inner_progress,
        )

        del df
        gc.collect()

        tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")

        if tmp_path.exists():
            tmp_path.unlink()

        features.to_parquet(
            tmp_path,
            engine="fastparquet",
            compression=compression,
            index=False,
        )

        if output_path.exists():
            output_path.unlink()

        tmp_path.replace(output_path)

        done_payload = {
            "job_id": job.job_id,
            "dataset": job.dataset,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "finished_at": now_iso(),
            "input_rows": int(input_rows),
            "feature_rows": int(len(features)),
        }

        done_path.write_text(
            json.dumps(done_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if error_path.exists():
            error_path.unlink()

        result["status"] = "done"
        result["finished_at"] = now_iso()
        result["input_rows"] = int(input_rows)
        result["feature_rows"] = int(len(features))

        del features
        gc.collect()

        return result

    except Exception as exc:
        traceback_text = traceback.format_exc()
        ensure_dir(error_path.parent)
        error_path.write_text(traceback_text, encoding="utf-8")

        result["status"] = "failed"
        result["finished_at"] = now_iso()
        result["error"] = f"{type(exc).__name__}: {exc}"

        gc.collect()
        return result


# =============================================================================
# Execution and reports
# =============================================================================

def run_jobs(
    jobs: List[Job],
    reports_dir: Path,
    options: Dict[str, Any],
    workers: int,
) -> pd.DataFrame:
    ensure_dir(reports_dir)

    status_jsonl = reports_dir / "status.jsonl"
    job_records = [asdict(job) for job in jobs]

    pd.DataFrame(job_records).to_csv(
        reports_dir / "job_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not jobs:
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []

    if workers <= 1:
        for job_dict in tqdm(job_records, total=len(job_records), desc="Extracting features"):
            result = process_job(job_dict, options)
            results.append(result)
            write_jsonl(status_jsonl, result)
    else:
        worker_options = dict(options)
        worker_options["inner_progress"] = False

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_job, job_dict, worker_options)
                for job_dict in job_records
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
                result = future.result()
                results.append(result)
                write_jsonl(status_jsonl, result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        reports_dir / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not results_df.empty and "status" in results_df.columns:
        failures = results_df[results_df["status"] == "failed"]
        if not failures.empty:
            failures.to_csv(
                reports_dir / "failures.csv",
                index=False,
                encoding="utf-8-sig",
            )

    return results_df


def write_output_index(output_root: Path, reports_dir: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for path in output_root.rglob("*.features.parquet"):
        done_path = Path(str(path) + ".done")

        dataset_name = path.parent.parent.name if path.parent.name == "parts" else path.parent.name

        records.append(
            {
                "dataset": dataset_name,
                "feature_file": str(path),
                "done": done_path.exists(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )

    index_df = pd.DataFrame(records)

    if not index_df.empty:
        index_df = index_df.sort_values(["dataset", "feature_file"])

    index_df.to_csv(
        reports_dir / "output_index.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return index_df


def print_summary(results_df: pd.DataFrame, reports_dir: Path, output_root: Path) -> None:
    if results_df.empty:
        print("No jobs were found.")
        return

    counts = results_df["status"].value_counts(dropna=False).to_dict()

    total = len(results_df)
    done = int(counts.get("done", 0))
    skipped = int(counts.get("skipped_done", 0))
    failed = int(counts.get("failed", 0))

    print("\nExtraction summary")
    print("-" * 60)
    print(f"Total jobs:         {total}")
    print(f"Done now:           {done}")
    print(f"Skipped by resume:  {skipped}")
    print(f"Failed:             {failed}")
    print(f"Output root:        {output_root}")
    print(f"Reports directory:  {reports_dir}")

    if failed:
        print("\nSome jobs failed.")
        print(f"Check: {reports_dir / 'failures.csv'}")
        print("Also check the .error.txt files next to each failed output path.")


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable hierarchical feature extraction for battery datasets."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Input root directory. Default: {DEFAULT_INPUT_ROOT}",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Feature output root directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )

    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help=f"Reports root directory. Default: {DEFAULT_REPORTS_ROOT}",
    )

    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset name or folder. Can be repeated or comma-separated.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all dataset folders inside --input-root.",
    )

    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=list(DEFAULT_EXCLUDED_DATASET_DIRS),
        help="Dataset/folder name to skip in --all mode. Can be repeated.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers. Use 1 or 2 if RAM usage is high.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip jobs that already have output Parquet and .done marker. Enabled by default.",
    )

    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess jobs even if .done exists. Use with caution.",
    )

    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "gzip", "brotli", "none", "None"],
        help="Parquet compression. Default: snappy.",
    )

    parser.add_argument(
        "--suffixes",
        default=",".join(DEFAULT_SUFFIXES),
        help='Input file suffixes. Default: ".parquet".',
    )

    parser.add_argument(
        "--min-gap-seconds",
        type=float,
        default=60.0,
        help="Minimum time gap used to split a phase into segments. Default: 60 seconds.",
    )

    parser.add_argument(
        "--gap-factor",
        type=float,
        default=20.0,
        help="Multiplier over median dt used to detect time-gap segments. Default: 20.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of files per dataset. Useful for testing.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the job plan only. Do not process files.",
    )

    parser.add_argument(
        "--inner-progress",
        action="store_true",
        help="Show cycle/phase progress inside each file. Use only with --workers 1.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    reports_root: Path = args.reports_root

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    requested_datasets = parse_list_argument(args.dataset)
    excluded_datasets = parse_list_argument(args.exclude_dataset)

    suffixes = tuple(
        suffix.strip().lower()
        for suffix in str(args.suffixes).split(",")
        if suffix.strip()
    )

    if not suffixes:
        suffixes = DEFAULT_SUFFIXES

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    reports_dir = reports_root / run_id

    ensure_dir(output_root)
    ensure_dir(reports_dir)

    datasets = discover_datasets(
        input_root=input_root,
        requested_datasets=requested_datasets,
        use_all=args.all,
        excluded_dirs=excluded_datasets,
    )

    jobs = build_jobs(
        input_root=input_root,
        output_root=output_root,
        datasets=datasets,
        suffixes=suffixes,
        max_files=args.max_files,
    )

    print("Execution plan")
    print("-" * 60)
    print(f"Input root:       {input_root}")
    print(f"Output root:      {output_root}")
    print(f"Reports root:     {reports_dir}")
    print(f"Datasets:         {', '.join([d[0] for d in datasets]) if datasets else 'none'}")
    print(f"Input suffixes:   {', '.join(suffixes)}")
    print(f"Jobs/files:       {len(jobs)}")
    print(f"Workers:          {args.workers}")
    print(f"Resume:           {args.resume}")
    print(f"Force:            {args.force}")

    pd.DataFrame([asdict(job) for job in jobs]).to_csv(
        reports_dir / "job_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.dry_run:
        print("\nDry run finished. No files were processed.")
        print(f"Job plan saved to: {reports_dir / 'job_plan.csv'}")
        return

    compression = None if str(args.compression).lower() == "none" else args.compression

    options: Dict[str, Any] = {
        "resume": bool(args.resume),
        "force": bool(args.force),
        "compression": compression,
        "min_gap_seconds": float(args.min_gap_seconds),
        "gap_factor": float(args.gap_factor),
        "inner_progress": bool(args.inner_progress and args.workers <= 1),
    }

    workers = max(1, int(args.workers))

    results_df = run_jobs(
        jobs=jobs,
        reports_dir=reports_dir,
        options=options,
        workers=workers,
    )

    write_output_index(output_root=output_root, reports_dir=reports_dir)
    print_summary(results_df=results_df, reports_dir=reports_dir, output_root=output_root)


if __name__ == "__main__":
    # Required on Windows when using ProcessPoolExecutor.
    main()
