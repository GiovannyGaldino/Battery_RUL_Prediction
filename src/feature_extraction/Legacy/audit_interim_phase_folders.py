"""
audit_interim_phase_folders.py

Audit the cleaned interim phase folders:

    data/interim/charge
    data/interim/discharge

Purpose
-------
This script checks whether the Parquet files inside these two folders are suitable
for feature extraction.

It does NOT scan dataset-specific clean/extracted folders such as:
    data/interim/hust/clean
    data/interim/matr/extracted

It only audits:
    <input-root>/charge
    <input-root>/discharge

Main checks
-----------
1. File inventory:
   - number of files
   - number of rows and columns
   - size on disk
   - dataset inferred from file content/name
   - phase inferred from folder/file/content

2. Schema consistency:
   - available columns
   - missing required columns
   - dtypes
   - missing values
   - infinite values

3. Signal column format:
   - checks if time_s, voltage_v, current_a, temperature_c, capacity_ah, energy_wh
     are scalar values, list/vector-like values, strings, or empty.
   - this is important because feature extraction must know whether it can compute
     curve-level features from vectors or should use existing scalar aggregate columns.

4. Existing aggregate feature availability:
   - n_points
   - duration_s
   - sample_period_median_s
   - voltage_min_v, voltage_max_v, voltage_start_v, voltage_end_v
   - current_min_a, current_max_a, current_mean_a
   - temperature_min_c, temperature_max_c, temperature_mean_c
   - phase_capacity_ah, phase_energy_wh

5. Target availability:
   - soh
   - remaining_cycles
   - remaining_cycles_norm
   - cycle_life

6. Duplicates:
   - duplicated keys by dataset, battery_id, cycle_index/cycle_number, phase

7. Dataset summary:
   - batteries by dataset and phase
   - cycles by battery
   - rows by dataset and phase

8. Readiness decision:
   - ready_for_scalar_feature_extraction
   - ready_for_vector_feature_extraction
   - needs_attention

Outputs
-------
CSV:
- file_inventory.csv
- columns_inventory.csv
- missing_values_by_file.csv
- signal_column_format_audit.csv
- aggregate_feature_availability.csv
- target_availability.csv
- duplicate_keys_report.csv
- dataset_phase_summary.csv
- cycles_per_battery.csv
- extraction_readiness.csv

Markdown:
- audit_report.md

Typical PowerShell usage
------------------------

& C:\\Users\\Giovanny\\anaconda3\\envs\\batteries\\python.exe ".\\src\\feature_extraction\\audit_interim_phase_folders.py" --input-root "C:\\Artigo Baterias de Litio\\Batteries\\data\\interim" --reports-root "C:\\Artigo Baterias de Litio\\Batteries\\reports\\interim_phase_folder_audit"

If you saved this file in scripts/pipelines instead:
& C:\\Users\\Giovanny\\anaconda3\\envs\\batteries\\python.exe ".\\scripts\\pipelines\\audit_interim_phase_folders.py" --input-root "C:\\Artigo Baterias de Litio\\Batteries\\data\\interim" --reports-root "C:\\Artigo Baterias de Litio\\Batteries\\reports\\interim_phase_folder_audit"
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/interim")
DEFAULT_REPORTS_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/reports/interim_phase_folder_audit")
DEFAULT_PHASE_DIRS = ("charge", "discharge")

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

TARGET_COLUMNS = [
    "soh",
    "remaining_cycles",
    "remaining_cycles_norm",
    "cycle_life",
    "eol_capacity_ah",
    "initial_capacity_ah",
    "nominal_capacity_ah",
]

SIGNAL_COLUMNS = [
    "time_s",
    "voltage_v",
    "current_a",
    "temperature_c",
    "capacity_ah",
    "energy_wh",
    "power_w",
]

AGGREGATE_COLUMNS = [
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
]

QUALITY_COLUMNS = [
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

MINIMAL_REQUIRED_COLUMNS = [
    "dataset",
    "battery_id",
    "phase",
]

REQUIRED_ONE_OF_CYCLE_COLUMNS = [
    "cycle_index",
    "cycle_number",
]

REQUIRED_ONE_OF_TARGET_COLUMNS = [
    "soh",
    "remaining_cycles",
    "remaining_cycles_norm",
]

REQUIRED_FOR_SCALAR_EXTRACTION = [
    "n_points",
    "duration_s",
    "voltage_min_v",
    "voltage_max_v",
    "voltage_start_v",
    "voltage_end_v",
    "current_min_a",
    "current_max_a",
    "current_mean_a",
    "phase_capacity_ah",
]

REQUIRED_FOR_VECTOR_EXTRACTION = [
    "time_s",
    "voltage_v",
    "current_a",
]


@dataclass
class FileAudit:
    phase_folder: str
    file_path: str
    file_name: str
    dataset_inferred_from_name: str
    dataset_inferred_from_content: str
    phase_inferred_from_content: str
    rows: int
    columns: int
    size_mb: float
    has_minimal_required_columns: bool
    has_cycle_column: bool
    has_target_column: bool
    scalar_required_available_rate: float
    vector_required_available_rate: float
    likely_signal_storage_mode: str
    ready_for_scalar_feature_extraction: bool
    ready_for_vector_feature_extraction: bool
    readiness_status: str
    notes: str


def now_id() -> str:
    return datetime.now().strftime("audit_%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def parse_list_argument(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    output: List[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                output.append(part)
    return list(dict.fromkeys(output))


def infer_dataset_from_filename(path: Path) -> str:
    # Examples:
    # hust_charge_clean_cycles.parquet -> hust
    # calce_discharge_clean_cycles.parquet -> calce
    stem = path.stem.lower()
    stem = re.sub(r"_(charge|discharge).*", "", stem)
    return stem or "unknown"


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def is_sequence_like(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return True

    return False


def maybe_parse_sequence_string(value: Any) -> Optional[List[Any]]:
    if not isinstance(value, str):
        return None

    text = value.strip()

    if not text:
        return None

    if not ((text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")"))):
        return None

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None

    if isinstance(parsed, (list, tuple)):
        return list(parsed)

    return None


def classify_cell_value(value: Any) -> Tuple[str, Optional[int]]:
    """
    Returns:
        (kind, length)

    kind:
        missing
        numeric_scalar
        sequence
        sequence_string
        string
        other
    """

    if value is None:
        return "missing", None

    try:
        if pd.isna(value):
            return "missing", None
    except Exception:
        pass

    if is_sequence_like(value):
        try:
            return "sequence", len(value)
        except Exception:
            return "sequence", None

    parsed = maybe_parse_sequence_string(value)
    if parsed is not None:
        return "sequence_string", len(parsed)

    if isinstance(value, str):
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric_value):
            return "numeric_scalar", None
        return "string", None

    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isfinite(value):
            return "numeric_scalar", None
        return "missing", None

    return "other", None


def classify_column_format(series: pd.Series, max_samples: int) -> Dict[str, Any]:
    non_null = series.dropna()

    if non_null.empty:
        return {
            "storage_mode": "all_missing",
            "sampled_non_null": 0,
            "numeric_scalar_count": 0,
            "sequence_count": 0,
            "sequence_string_count": 0,
            "string_count": 0,
            "other_count": 0,
            "min_sequence_length": np.nan,
            "median_sequence_length": np.nan,
            "max_sequence_length": np.nan,
        }

    sample = non_null.head(max_samples)

    counts = {
        "numeric_scalar": 0,
        "sequence": 0,
        "sequence_string": 0,
        "string": 0,
        "other": 0,
        "missing": 0,
    }
    lengths: List[int] = []

    for value in sample:
        kind, length = classify_cell_value(value)
        counts[kind] = counts.get(kind, 0) + 1
        if length is not None:
            lengths.append(int(length))

    sampled = int(len(sample))
    sequence_total = counts["sequence"] + counts["sequence_string"]

    if sequence_total / max(sampled, 1) >= 0.50:
        storage_mode = "vector_like"
    elif counts["numeric_scalar"] / max(sampled, 1) >= 0.50:
        storage_mode = "numeric_scalar"
    elif counts["string"] / max(sampled, 1) >= 0.50:
        storage_mode = "string"
    else:
        storage_mode = "mixed_or_unknown"

    return {
        "storage_mode": storage_mode,
        "sampled_non_null": sampled,
        "numeric_scalar_count": counts["numeric_scalar"],
        "sequence_count": counts["sequence"],
        "sequence_string_count": counts["sequence_string"],
        "string_count": counts["string"],
        "other_count": counts["other"],
        "min_sequence_length": min(lengths) if lengths else np.nan,
        "median_sequence_length": float(np.median(lengths)) if lengths else np.nan,
        "max_sequence_length": max(lengths) if lengths else np.nan,
    }


def numeric_missing_and_inf(series: pd.Series) -> Tuple[int, int, int, float, float, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    arr = numeric.to_numpy(dtype=np.float64)

    missing = int(pd.isna(numeric).sum())
    inf_count = int(np.isinf(arr).sum())
    finite_mask = np.isfinite(arr)
    finite_count = int(finite_mask.sum())

    if finite_count:
        min_value = safe_float(np.nanmin(arr[finite_mask]))
        max_value = safe_float(np.nanmax(arr[finite_mask]))
        mean_value = safe_float(np.nanmean(arr[finite_mask]))
    else:
        min_value = np.nan
        max_value = np.nan
        mean_value = np.nan

    return missing, inf_count, finite_count, min_value, max_value, mean_value


def read_parquet_file(path: Path, engine: str) -> pd.DataFrame:
    if engine == "auto":
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception:
            return pd.read_parquet(path)

    return pd.read_parquet(path, engine=engine)


def discover_files(input_root: Path, phase_dirs: Sequence[str], max_files: Optional[int]) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []

    for phase_dir in phase_dirs:
        folder = input_root / phase_dir

        if not folder.exists():
            print(f"[WARNING] Phase folder not found: {folder}", file=sys.stderr)
            continue

        phase_files = sorted(folder.glob("*.parquet"), key=lambda p: str(p).lower())

        if max_files is not None:
            phase_files = phase_files[:max_files]

        for path in phase_files:
            files.append((phase_dir, path))

    return files


def infer_single_value(series: pd.Series, default: str = "unknown") -> str:
    if series.empty:
        return default
    values = series.dropna().astype(str).unique()
    if len(values) == 0:
        return default
    if len(values) == 1:
        return str(values[0])
    return "multiple:" + "|".join(sorted(map(str, values))[:10])


def column_presence_rate(columns: Sequence[str], required: Sequence[str]) -> float:
    if not required:
        return 1.0
    available = sum(1 for col in required if col in columns)
    return available / len(required)


def make_columns_inventory(df: pd.DataFrame, file_path: Path, phase_folder: str, dataset_name: str) -> pd.DataFrame:
    records = []

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_rate = missing_count / max(len(df), 1)

        dtype = str(s.dtype)

        if pd.api.types.is_numeric_dtype(s):
            missing_numeric, inf_count, finite_count, min_value, max_value, mean_value = numeric_missing_and_inf(s)
        else:
            inf_count = 0
            finite_count = np.nan
            min_value = np.nan
            max_value = np.nan
            mean_value = np.nan

        try:
            unique_count = int(s.nunique(dropna=True))
        except Exception:
            unique_count = np.nan

        records.append(
            {
                "phase_folder": phase_folder,
                "dataset": dataset_name,
                "file": str(file_path),
                "column": col,
                "dtype": dtype,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "unique_count": unique_count,
                "inf_count": inf_count,
                "finite_count": finite_count,
                "min": min_value,
                "max": max_value,
                "mean": mean_value,
            }
        )

    return pd.DataFrame(records)


def make_signal_audit(df: pd.DataFrame, file_path: Path, phase_folder: str, dataset_name: str, max_samples: int) -> pd.DataFrame:
    records = []

    for col in SIGNAL_COLUMNS:
        if col not in df.columns:
            records.append(
                {
                    "phase_folder": phase_folder,
                    "dataset": dataset_name,
                    "file": str(file_path),
                    "column": col,
                    "exists": False,
                    "dtype": None,
                    "missing_rate": np.nan,
                    "storage_mode": "missing_column",
                    "sampled_non_null": 0,
                    "numeric_scalar_count": 0,
                    "sequence_count": 0,
                    "sequence_string_count": 0,
                    "string_count": 0,
                    "other_count": 0,
                    "min_sequence_length": np.nan,
                    "median_sequence_length": np.nan,
                    "max_sequence_length": np.nan,
                }
            )
            continue

        result = classify_column_format(df[col], max_samples=max_samples)
        record = {
            "phase_folder": phase_folder,
            "dataset": dataset_name,
            "file": str(file_path),
            "column": col,
            "exists": True,
            "dtype": str(df[col].dtype),
            "missing_rate": float(df[col].isna().mean()) if len(df) else np.nan,
        }
        record.update(result)
        records.append(record)

    return pd.DataFrame(records)


def make_aggregate_availability(df: pd.DataFrame, file_path: Path, phase_folder: str, dataset_name: str) -> pd.DataFrame:
    records = []

    for col in AGGREGATE_COLUMNS:
        exists = col in df.columns

        if exists:
            missing_rate = float(df[col].isna().mean()) if len(df) else np.nan
            if pd.api.types.is_numeric_dtype(df[col]):
                _, inf_count, finite_count, min_value, max_value, mean_value = numeric_missing_and_inf(df[col])
            else:
                inf_count = 0
                finite_count = np.nan
                min_value = np.nan
                max_value = np.nan
                mean_value = np.nan
        else:
            missing_rate = np.nan
            inf_count = np.nan
            finite_count = np.nan
            min_value = np.nan
            max_value = np.nan
            mean_value = np.nan

        records.append(
            {
                "phase_folder": phase_folder,
                "dataset": dataset_name,
                "file": str(file_path),
                "column": col,
                "exists": exists,
                "missing_rate": missing_rate,
                "inf_count": inf_count,
                "finite_count": finite_count,
                "min": min_value,
                "max": max_value,
                "mean": mean_value,
            }
        )

    return pd.DataFrame(records)


def make_target_availability(df: pd.DataFrame, file_path: Path, phase_folder: str, dataset_name: str) -> pd.DataFrame:
    records = []

    for col in TARGET_COLUMNS:
        exists = col in df.columns

        if exists:
            missing_rate = float(df[col].isna().mean()) if len(df) else np.nan
            if pd.api.types.is_numeric_dtype(df[col]):
                _, inf_count, finite_count, min_value, max_value, mean_value = numeric_missing_and_inf(df[col])
            else:
                inf_count = 0
                finite_count = np.nan
                min_value = np.nan
                max_value = np.nan
                mean_value = np.nan
        else:
            missing_rate = np.nan
            inf_count = np.nan
            finite_count = np.nan
            min_value = np.nan
            max_value = np.nan
            mean_value = np.nan

        records.append(
            {
                "phase_folder": phase_folder,
                "dataset": dataset_name,
                "file": str(file_path),
                "column": col,
                "exists": exists,
                "missing_rate": missing_rate,
                "inf_count": inf_count,
                "finite_count": finite_count,
                "min": min_value,
                "max": max_value,
                "mean": mean_value,
            }
        )

    return pd.DataFrame(records)


def make_duplicate_report(df: pd.DataFrame, file_path: Path, phase_folder: str, dataset_name: str) -> pd.DataFrame:
    possible_key_cols = ["dataset", "battery_id", "cycle_index", "cycle_number", "phase"]

    key_cols = [col for col in possible_key_cols if col in df.columns]

    if "cycle_index" in key_cols and "cycle_number" in key_cols:
        # Prefer cycle_index for uniqueness. Keep cycle_number for context only if needed.
        key_cols = [col for col in key_cols if col != "cycle_number"]

    if not key_cols:
        return pd.DataFrame(
            [
                {
                    "phase_folder": phase_folder,
                    "dataset": dataset_name,
                    "file": str(file_path),
                    "key_columns": "",
                    "duplicate_rows": np.nan,
                    "duplicate_groups": np.nan,
                    "status": "no_key_columns",
                }
            ]
        )

    duplicated_mask = df.duplicated(subset=key_cols, keep=False)
    duplicate_rows = int(duplicated_mask.sum())

    if duplicate_rows:
        duplicate_groups = (
            df.loc[duplicated_mask]
            .groupby(key_cols, dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        duplicate_group_count = int(len(duplicate_groups))
    else:
        duplicate_group_count = 0

    return pd.DataFrame(
        [
            {
                "phase_folder": phase_folder,
                "dataset": dataset_name,
                "file": str(file_path),
                "key_columns": "|".join(key_cols),
                "duplicate_rows": duplicate_rows,
                "duplicate_groups": duplicate_group_count,
                "status": "ok" if duplicate_rows == 0 else "duplicates_found",
            }
        ]
    )


def make_dataset_phase_summary(all_keys: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if all_keys.empty:
        return pd.DataFrame(), pd.DataFrame()

    group_cols = ["dataset", "phase_folder"]

    agg_dict = {
        "rows": ("row_id", "count"),
        "files": ("file", "nunique"),
    }

    if "battery_id" in all_keys.columns:
        agg_dict["batteries"] = ("battery_id", "nunique")

    if "cycle_index" in all_keys.columns:
        agg_dict["cycles"] = ("cycle_index", "nunique")
    elif "cycle_number" in all_keys.columns:
        agg_dict["cycles"] = ("cycle_number", "nunique")

    summary = (
        all_keys.groupby(group_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .sort_values(["dataset", "phase_folder"])
    )

    if "battery_id" in all_keys.columns and ("cycle_index" in all_keys.columns or "cycle_number" in all_keys.columns):
        cycle_col = "cycle_index" if "cycle_index" in all_keys.columns else "cycle_number"
        cycles_per_battery = (
            all_keys.groupby(["dataset", "phase_folder", "battery_id"], dropna=False)
            .agg(
                rows=("row_id", "count"),
                n_cycles=(cycle_col, "nunique"),
                min_cycle=(cycle_col, "min"),
                max_cycle=(cycle_col, "max"),
            )
            .reset_index()
            .sort_values(["dataset", "phase_folder", "battery_id"])
        )
    else:
        cycles_per_battery = pd.DataFrame()

    return summary, cycles_per_battery


def determine_likely_signal_storage(signal_audit: pd.DataFrame) -> str:
    if signal_audit.empty:
        return "unknown"

    existing = signal_audit[signal_audit["exists"] == True]

    if existing.empty:
        return "missing_signal_columns"

    modes = existing["storage_mode"].value_counts(dropna=False).to_dict()

    if modes.get("vector_like", 0) >= 2:
        return "vector_like"

    if modes.get("numeric_scalar", 0) >= 2:
        return "numeric_scalar"

    if modes.get("all_missing", 0) >= 2:
        return "mostly_missing"

    return "mixed_or_unknown"


def evaluate_readiness(
    columns: Sequence[str],
    signal_audit: pd.DataFrame,
    aggregate_availability: pd.DataFrame,
    target_availability: pd.DataFrame,
    duplicate_status: str,
    rows: int,
) -> Tuple[bool, bool, str, str]:
    columns_set = set(columns)

    missing_minimal = [col for col in MINIMAL_REQUIRED_COLUMNS if col not in columns_set]
    has_cycle = any(col in columns_set for col in REQUIRED_ONE_OF_CYCLE_COLUMNS)
    has_target = target_availability.query("exists == True and missing_rate < 1.0").shape[0] > 0

    aggregate_available = aggregate_availability.query("exists == True and missing_rate < 1.0")
    scalar_required_present = set(aggregate_available["column"]).intersection(REQUIRED_FOR_SCALAR_EXTRACTION)
    scalar_rate = len(scalar_required_present) / len(REQUIRED_FOR_SCALAR_EXTRACTION)

    signal_existing = signal_audit.query("exists == True and missing_rate < 1.0")
    signal_required_present = set(signal_existing["column"]).intersection(REQUIRED_FOR_VECTOR_EXTRACTION)
    vector_rate = len(signal_required_present) / len(REQUIRED_FOR_VECTOR_EXTRACTION)

    likely_storage = determine_likely_signal_storage(signal_audit)

    ready_scalar = (
        rows > 0
        and not missing_minimal
        and has_cycle
        and has_target
        and scalar_rate >= 0.70
    )

    ready_vector = (
        rows > 0
        and not missing_minimal
        and has_cycle
        and has_target
        and vector_rate >= 1.0
        and likely_storage == "vector_like"
    )

    notes = []

    if missing_minimal:
        notes.append(f"missing minimal columns: {missing_minimal}")

    if not has_cycle:
        notes.append("missing cycle_index or cycle_number")

    if not has_target:
        notes.append("no valid target column among soh/remaining_cycles/remaining_cycles_norm")

    if scalar_rate < 0.70:
        notes.append(f"low scalar aggregate availability: {scalar_rate:.2f}")

    if vector_rate < 1.0:
        notes.append(f"missing vector signal columns for raw curve extraction: {vector_rate:.2f}")

    if likely_storage != "vector_like":
        notes.append(f"signal columns storage mode is {likely_storage}; vector extraction may not be possible directly")

    if duplicate_status == "duplicates_found":
        notes.append("duplicate keys detected")

    if ready_vector:
        status = "ready_for_vector_feature_extraction"
    elif ready_scalar:
        status = "ready_for_scalar_feature_extraction"
    else:
        status = "needs_attention"

    return ready_scalar, ready_vector, status, "; ".join(notes)


def audit_file(
    phase_folder: str,
    path: Path,
    input_root: Path,
    parquet_engine: str,
    max_signal_samples: int,
) -> Tuple[
    FileAudit,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    df = read_parquet_file(path, engine=parquet_engine)

    dataset_from_name = infer_dataset_from_filename(path)

    dataset_from_content = infer_single_value(df["dataset"]) if "dataset" in df.columns else dataset_from_name
    phase_from_content = infer_single_value(df["phase"]) if "phase" in df.columns else phase_folder

    if dataset_from_content.startswith("multiple:"):
        dataset_for_report = dataset_from_name
    else:
        dataset_for_report = dataset_from_content

    columns_inventory = make_columns_inventory(df, path, phase_folder, dataset_for_report)
    signal_audit = make_signal_audit(df, path, phase_folder, dataset_for_report, max_signal_samples)
    aggregate_availability = make_aggregate_availability(df, path, phase_folder, dataset_for_report)
    target_availability = make_target_availability(df, path, phase_folder, dataset_for_report)
    duplicate_report = make_duplicate_report(df, path, phase_folder, dataset_for_report)

    duplicate_status = duplicate_report["status"].iloc[0] if not duplicate_report.empty else "unknown"

    ready_scalar, ready_vector, readiness_status, notes = evaluate_readiness(
        columns=list(df.columns),
        signal_audit=signal_audit,
        aggregate_availability=aggregate_availability,
        target_availability=target_availability,
        duplicate_status=duplicate_status,
        rows=len(df),
    )

    columns = list(df.columns)
    scalar_required_rate = column_presence_rate(columns, REQUIRED_FOR_SCALAR_EXTRACTION)
    vector_required_rate = column_presence_rate(columns, REQUIRED_FOR_VECTOR_EXTRACTION)
    likely_storage = determine_likely_signal_storage(signal_audit)

    file_audit = FileAudit(
        phase_folder=phase_folder,
        file_path=str(path),
        file_name=path.name,
        dataset_inferred_from_name=dataset_from_name,
        dataset_inferred_from_content=dataset_from_content,
        phase_inferred_from_content=phase_from_content,
        rows=int(len(df)),
        columns=int(df.shape[1]),
        size_mb=round(path.stat().st_size / (1024 * 1024), 4),
        has_minimal_required_columns=all(col in df.columns for col in MINIMAL_REQUIRED_COLUMNS),
        has_cycle_column=any(col in df.columns for col in REQUIRED_ONE_OF_CYCLE_COLUMNS),
        has_target_column=target_availability.query("exists == True and missing_rate < 1.0").shape[0] > 0,
        scalar_required_available_rate=scalar_required_rate,
        vector_required_available_rate=vector_required_rate,
        likely_signal_storage_mode=likely_storage,
        ready_for_scalar_feature_extraction=ready_scalar,
        ready_for_vector_feature_extraction=ready_vector,
        readiness_status=readiness_status,
        notes=notes,
    )

    key_cols = [col for col in ["dataset", "battery_id", "cycle_index", "cycle_number", "phase"] if col in df.columns]
    keys_df = df[key_cols].copy() if key_cols else pd.DataFrame()
    keys_df["row_id"] = np.arange(len(df), dtype=np.int64)
    keys_df["file"] = str(path)
    keys_df["phase_folder"] = phase_folder
    if "dataset" not in keys_df.columns:
        keys_df["dataset"] = dataset_for_report

    del df

    return (
        file_audit,
        columns_inventory,
        signal_audit,
        aggregate_availability,
        target_availability,
        duplicate_report,
        keys_df,
    )


def df_to_markdown_preview(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data._"

    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)


def write_markdown_report(
    path: Path,
    file_inventory: pd.DataFrame,
    readiness: pd.DataFrame,
    dataset_phase_summary: pd.DataFrame,
    signal_audit: pd.DataFrame,
    aggregate_availability: pd.DataFrame,
    target_availability: pd.DataFrame,
    duplicate_report: pd.DataFrame,
    missing_values: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Interim Charge/Discharge Folder Audit")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Audited files: `{len(file_inventory)}`")
    if not file_inventory.empty:
        lines.append(f"- Total rows: `{int(file_inventory['rows'].sum())}`")
        lines.append(f"- Total size MB: `{round(float(file_inventory['size_mb'].sum()), 3)}`")
        status_counts = file_inventory["readiness_status"].value_counts(dropna=False).to_dict()
        lines.append(f"- Readiness status counts: `{status_counts}`")
    lines.append("")

    lines.append("## File readiness")
    lines.append("")
    preview_cols = [
        "phase_folder",
        "file_name",
        "dataset_inferred_from_content",
        "rows",
        "columns",
        "likely_signal_storage_mode",
        "ready_for_scalar_feature_extraction",
        "ready_for_vector_feature_extraction",
        "readiness_status",
        "notes",
    ]
    preview_cols = [col for col in preview_cols if col in file_inventory.columns]
    lines.append(df_to_markdown_preview(file_inventory[preview_cols], 50))
    lines.append("")

    lines.append("## Dataset and phase summary")
    lines.append("")
    lines.append(df_to_markdown_preview(dataset_phase_summary, 50))
    lines.append("")

    lines.append("## Signal column storage modes")
    lines.append("")
    if not signal_audit.empty:
        signal_summary = (
            signal_audit.groupby(["phase_folder", "dataset", "column", "storage_mode"], dropna=False)
            .size()
            .reset_index(name="files")
            .sort_values(["dataset", "phase_folder", "column", "storage_mode"])
        )
    else:
        signal_summary = pd.DataFrame()
    lines.append(df_to_markdown_preview(signal_summary, 80))
    lines.append("")

    lines.append("## Aggregate feature availability")
    lines.append("")
    if not aggregate_availability.empty:
        agg_summary = (
            aggregate_availability.groupby("column", dropna=False)
            .agg(
                files=("file", "nunique"),
                exists_rate=("exists", "mean"),
                mean_missing_rate=("missing_rate", "mean"),
            )
            .reset_index()
            .sort_values(["exists_rate", "column"], ascending=[True, True])
        )
    else:
        agg_summary = pd.DataFrame()
    lines.append(df_to_markdown_preview(agg_summary, 80))
    lines.append("")

    lines.append("## Target availability")
    lines.append("")
    if not target_availability.empty:
        target_summary = (
            target_availability.groupby("column", dropna=False)
            .agg(
                files=("file", "nunique"),
                exists_rate=("exists", "mean"),
                mean_missing_rate=("missing_rate", "mean"),
                global_min=("min", "min"),
                global_max=("max", "max"),
                global_mean=("mean", "mean"),
            )
            .reset_index()
            .sort_values(["exists_rate", "column"], ascending=[False, True])
        )
    else:
        target_summary = pd.DataFrame()
    lines.append(df_to_markdown_preview(target_summary, 50))
    lines.append("")

    lines.append("## Duplicate keys")
    lines.append("")
    lines.append(df_to_markdown_preview(duplicate_report, 50))
    lines.append("")

    lines.append("## Highest missing-value columns")
    lines.append("")
    if not missing_values.empty:
        miss_preview = missing_values.sort_values("missing_rate", ascending=False).head(50)
    else:
        miss_preview = pd.DataFrame()
    lines.append(df_to_markdown_preview(miss_preview, 50))
    lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- `ready_for_scalar_feature_extraction` means the files contain enough existing aggregate columns to build a robust feature table without using raw vectors.")
    lines.append("- `ready_for_vector_feature_extraction` means the signal columns appear to contain vector/list-like data and can potentially be used to compute curve-shape features directly.")
    lines.append("- If signal columns are `numeric_scalar`, then each row is already cycle/phase-level and raw curve extraction is not possible directly from those columns.")
    lines.append("- If signal columns are `vector_like`, extraction should parse arrays from `time_s`, `voltage_v`, `current_a`, etc.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = argparse.ArgumentParser(
        description="Audit only data/interim/charge and data/interim/discharge Parquet files."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Root folder containing charge and discharge folders. Default: {DEFAULT_INPUT_ROOT}",
    )

    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help=f"Output reports root. Default: {DEFAULT_REPORTS_ROOT}",
    )

    parser.add_argument(
        "--phase-dirs",
        nargs="*",
        default=list(DEFAULT_PHASE_DIRS),
        help="Phase folders to audit inside --input-root. Default: charge discharge.",
    )

    parser.add_argument(
        "--parquet-engine",
        default="fastparquet",
        choices=["fastparquet", "pyarrow", "auto"],
        help="Parquet engine. Default: fastparquet.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files per phase folder. Useful for quick tests.",
    )

    parser.add_argument(
        "--max-signal-samples",
        type=int,
        default=30,
        help="Number of non-null values sampled per signal column to infer storage format. Default: 30.",
    )

    args = parser.parse_args()

    input_root: Path = args.input_root
    reports_root: Path = args.reports_root

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    run_dir = reports_root / now_id()
    ensure_dir(run_dir)

    files = discover_files(
        input_root=input_root,
        phase_dirs=args.phase_dirs,
        max_files=args.max_files,
    )

    print("Interim phase folder audit")
    print("-" * 60)
    print(f"Input root:       {input_root}")
    print(f"Phase folders:    {', '.join(args.phase_dirs)}")
    print(f"Files found:      {len(files)}")
    print(f"Reports dir:      {run_dir}")

    if not files:
        raise FileNotFoundError("No Parquet files found in the selected phase folders.")

    file_audits: List[FileAudit] = []
    columns_inventory_frames: List[pd.DataFrame] = []
    signal_audit_frames: List[pd.DataFrame] = []
    aggregate_availability_frames: List[pd.DataFrame] = []
    target_availability_frames: List[pd.DataFrame] = []
    duplicate_report_frames: List[pd.DataFrame] = []
    keys_frames: List[pd.DataFrame] = []

    for phase_folder, path in files:
        print(f"Auditing: {path}")

        (
            file_audit,
            columns_inventory,
            signal_audit,
            aggregate_availability,
            target_availability,
            duplicate_report,
            keys_df,
        ) = audit_file(
            phase_folder=phase_folder,
            path=path,
            input_root=input_root,
            parquet_engine=args.parquet_engine,
            max_signal_samples=args.max_signal_samples,
        )

        file_audits.append(file_audit)
        columns_inventory_frames.append(columns_inventory)
        signal_audit_frames.append(signal_audit)
        aggregate_availability_frames.append(aggregate_availability)
        target_availability_frames.append(target_availability)
        duplicate_report_frames.append(duplicate_report)
        keys_frames.append(keys_df)

    file_inventory = pd.DataFrame([asdict(x) for x in file_audits])
    columns_inventory_all = pd.concat(columns_inventory_frames, ignore_index=True) if columns_inventory_frames else pd.DataFrame()
    signal_audit_all = pd.concat(signal_audit_frames, ignore_index=True) if signal_audit_frames else pd.DataFrame()
    aggregate_availability_all = pd.concat(aggregate_availability_frames, ignore_index=True) if aggregate_availability_frames else pd.DataFrame()
    target_availability_all = pd.concat(target_availability_frames, ignore_index=True) if target_availability_frames else pd.DataFrame()
    duplicate_report_all = pd.concat(duplicate_report_frames, ignore_index=True) if duplicate_report_frames else pd.DataFrame()
    all_keys = pd.concat(keys_frames, ignore_index=True, sort=False) if keys_frames else pd.DataFrame()

    dataset_phase_summary, cycles_per_battery = make_dataset_phase_summary(all_keys)

    missing_values = columns_inventory_all[
        [
            "phase_folder",
            "dataset",
            "file",
            "column",
            "dtype",
            "missing_count",
            "missing_rate",
            "inf_count",
            "unique_count",
        ]
    ].copy() if not columns_inventory_all.empty else pd.DataFrame()

    if not missing_values.empty:
        missing_values = missing_values.sort_values(["missing_rate", "inf_count"], ascending=[False, False])

    extraction_readiness = file_inventory[
        [
            "phase_folder",
            "file_name",
            "dataset_inferred_from_name",
            "dataset_inferred_from_content",
            "phase_inferred_from_content",
            "rows",
            "columns",
            "likely_signal_storage_mode",
            "ready_for_scalar_feature_extraction",
            "ready_for_vector_feature_extraction",
            "readiness_status",
            "notes",
        ]
    ].copy()

    # Write CSV reports.
    write_csv(file_inventory, run_dir / "file_inventory.csv")
    write_csv(columns_inventory_all, run_dir / "columns_inventory.csv")
    write_csv(missing_values, run_dir / "missing_values_by_file.csv")
    write_csv(signal_audit_all, run_dir / "signal_column_format_audit.csv")
    write_csv(aggregate_availability_all, run_dir / "aggregate_feature_availability.csv")
    write_csv(target_availability_all, run_dir / "target_availability.csv")
    write_csv(duplicate_report_all, run_dir / "duplicate_keys_report.csv")
    write_csv(dataset_phase_summary, run_dir / "dataset_phase_summary.csv")
    write_csv(cycles_per_battery, run_dir / "cycles_per_battery.csv")
    write_csv(extraction_readiness, run_dir / "extraction_readiness.csv")

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(input_root),
        "reports_dir": str(run_dir),
        "phase_dirs": list(args.phase_dirs),
        "files_found": len(files),
        "parquet_engine": args.parquet_engine,
        "max_signal_samples": args.max_signal_samples,
    }

    (run_dir / "audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_markdown_report(
        path=run_dir / "audit_report.md",
        file_inventory=file_inventory,
        readiness=extraction_readiness,
        dataset_phase_summary=dataset_phase_summary,
        signal_audit=signal_audit_all,
        aggregate_availability=aggregate_availability_all,
        target_availability=target_availability_all,
        duplicate_report=duplicate_report_all,
        missing_values=missing_values,
    )

    print("\nAudit finished.")
    print("-" * 60)
    print(f"Reports saved to: {run_dir}")

    if not file_inventory.empty:
        print("\nReadiness status:")
        print(file_inventory["readiness_status"].value_counts(dropna=False).to_string())

        print("\nSignal storage modes:")
        print(file_inventory["likely_signal_storage_mode"].value_counts(dropna=False).to_string())

        n_scalar_ready = int(file_inventory["ready_for_scalar_feature_extraction"].sum())
        n_vector_ready = int(file_inventory["ready_for_vector_feature_extraction"].sum())
        print(f"\nReady for scalar feature extraction: {n_scalar_ready}/{len(file_inventory)}")
        print(f"Ready for vector feature extraction: {n_vector_ready}/{len(file_inventory)}")


if __name__ == "__main__":
    main()
