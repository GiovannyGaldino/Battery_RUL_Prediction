"""
Extract CALCE lithium-ion battery charge and discharge cycles from Excel files.

This script reconstructs the CALCE adapter using the legacy CALCE extraction logic
as a reference, but it writes the new project outputs required by the standard
charge/discharge branch workflow.

Default input layout:
    data/raw/CALCE/<cell_id>/<workbook>.xlsx

Default outputs:
    data/interim/calce/extracted/calce_charge_extracted_cycles.parquet
    data/interim/calce/extracted/calce_discharge_extracted_cycles.parquet
    data/interim/calce/extracted/calce_extraction_report.json

Important assumptions inherited from the legacy CALCE scripts:
    - TXT-to-XLSX conversion is not performed here; converted Excel files are read.
    - Cell IDs are inferred from path patterns such as CS2_8, CS_2_8, CX2_16.
    - Known non-cycling files are ignored by keyword.
    - CS2 cells are treated as nominal 1.10 Ah and CX2 cells as nominal 1.35 Ah
      unless overridden by command-line arguments.
    - When measured temperature is absent, a 24 °C constant vector is used and
      marked as literature_constant.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

DATASET = "CALCE"
DATASET_DIRNAME = "calce"
ADAPTER_VERSION = "calce_extractor_v6.0.0"

SUPPORTED_INPUT_EXTENSIONS = {".xlsx", ".xls"}
IGNORED_INPUT_EXTENSIONS = {".txt", ".csv"}
EXCLUDE_KEYWORDS = [
    "calibration",
    "logicerror",
    "brokenchannel",
    "tricklecharge",
    "thumbs",
    "self discharge",
    "self_discharge",
]

DEFAULT_TEMPERATURE_C = 24.0
DEFAULT_PHASE_CURRENT_THRESHOLD_C_RATE = 0.05
MIN_POINTS_PER_EXTRACTED_SEGMENT = 20
HEADER_SCAN_MAX_ROWS = 25
MAX_GAP_POINTS_TO_MERGE = 10

STANDARD_CYCLE_COLUMNS = [
    "dataset", "source_file", "source_path", "battery_id", "battery_index",
    "cell_id_original", "cycle_index", "cycle_number", "source_cycle_id",
    "cycle_pair_id", "phase",
    "cycle_life", "remaining_cycles", "remaining_cycles_norm", "eol_criterion",
    "eol_capacity_ah", "initial_capacity_ah", "nominal_capacity_ah", "soh",
    "time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah",
    "energy_wh", "power_w",
    "n_points", "duration_s", "sample_period_median_s", "voltage_min_v",
    "voltage_max_v", "voltage_start_v", "voltage_end_v", "current_min_a",
    "current_max_a", "current_mean_a", "temperature_min_c",
    "temperature_max_c", "temperature_mean_c", "phase_capacity_ah",
    "phase_energy_wh",
    "voltage_source", "current_source", "temperature_source", "capacity_source",
    "energy_source",
    "chemistry", "form_factor", "manufacturer", "nominal_voltage_v",
    "ambient_temperature_c", "charge_protocol", "discharge_protocol",
    "c_rate_charge", "c_rate_discharge", "dod", "policy_readable", "barcode",
    "is_valid_cycle", "is_model_usable", "exclude_reason", "cleaning_warnings",
    "n_removed_points", "trim_start_index", "trim_end_index", "has_voltage",
    "has_current", "has_temperature", "has_capacity", "has_energy",
    "adapter_version", "cleaning_version", "created_at",
]

EXTRACTION_COLUMNS = [
    "extraction_warnings",
    "raw_sheet_name",
    "raw_cycle_column",
    "raw_phase_segment_index",
    "source_row_start",
    "source_row_end",
]

COLUMN_ALIASES = {
    "cycle_number": [
        "cycle_index", "cycle index", "cycle", "cycle number", "cycle_number",
        "cycle#", "cycle #", "cyc#", "cyc #", "cycle no", "cycle_id",
    ],
    "step_index": [
        "step_index", "step index", "step", "step number", "step_number",
        "step#", "step #", "step_id",
    ],
    "time_s": [
        "test_time_s", "test time(s)", "test_time(s)", "test time", "test time (s)",
        "time_s", "time(s)", "time", "test_time", "testtime", "test_time_sec",
        "time sec",
    ],
    "step_time_s": [
        "step_time_s", "step time(s)", "step_time(s)", "step time", "step time (s)",
        "steptime",
    ],
    "voltage_v": [
        "voltage_v", "voltage(v)", "voltage (v)", "voltage", "voltage_measured",
        "voltage_measured(v)", "voltage measured", "voltage/v", "v", "mv", "voltage(mv)", "voltage (mv)", "voltage/mv",
    ],
    "current_a": [
        "current_a", "current(a)", "current (a)", "current", "current_measured",
        "current_measured(a)", "current measured", "current/a", "i", "ma", "current(ma)", "current (ma)", "current/ma",
    ],
    "charge_capacity_ah": [
        "charge_capacity_ah", "charge capacity(ah)", "charge_capacity(ah)",
        "charge capacity", "charge_capacity", "charge cap(ah)", "charge cap",
        "charge_capacity/ah", "qcharge", "qc_ah", "charge", "charge capacity(mah)", "charge capacity (mah)", "charge_capacity(mah)", "charge/mah",
    ],
    "discharge_capacity_ah": [
        "discharge_capacity_ah", "discharge capacity(ah)", "discharge_capacity(ah)",
        "discharge capacity", "discharge_capacity", "discharge cap(ah)",
        "discharge cap", "discharge_capacity/ah", "qdischarge", "qd_ah", "discharge", "discharge capacity(mah)", "discharge capacity (mah)", "discharge_capacity(mah)", "discharge/mah",
    ],
    "capacity_ah": ["capacity_ah", "capacity(ah)", "capacity", "cap(ah)", "cap_ah", "capacity(mah)", "capacity (mah)", "cap(mah)", "mah"],
    "charge_energy_wh": [
        "charge_energy_wh", "charge energy(wh)", "charge_energy(wh)",
        "charge energy", "charge_energy", "charge energy(mwh)", "charge_energy(mwh)",
    ],
    "discharge_energy_wh": [
        "discharge_energy_wh", "discharge energy(wh)", "discharge_energy(wh)",
        "discharge energy", "discharge_energy", "discharge energy(mwh)", "discharge_energy(mwh)",
    ],
    "energy_wh": ["energy_wh", "energy(wh)", "energy", "energy (wh)", "energy(mwh)", "energy (mwh)", "mwh"],
    "temperature_c": [
        "temperature_c", "temperature(c)", "temperature (c)", "temperature",
        "temp", "temp(c)", "temp_c", "aux_temperature(c)", "aux temperature(c)",
        "temperature_measured",
    ],
}


@dataclass(frozen=True)
class ColumnMap:
    cycle_number: str | None = None
    step_index: str | None = None
    time_s: str | None = None
    step_time_s: str | None = None
    voltage_v: str | None = None
    current_a: str | None = None
    charge_capacity_ah: str | None = None
    discharge_capacity_ah: str | None = None
    capacity_ah: str | None = None
    charge_energy_wh: str | None = None
    discharge_energy_wh: str | None = None
    energy_wh: str | None = None
    temperature_c: str | None = None


def normalize_column_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("[", "(").replace("]", ")")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\(\s*", "(", text)
    text = re.sub(r"\s*\)\s*", ")", text)
    return text.strip()


def find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized_to_original: dict[str, str] = {}
    for col in df.columns:
        normalized_to_original.setdefault(normalize_column_name(col), col)
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in normalized_to_original:
            return normalized_to_original[key]
    return None


def detect_columns(df: pd.DataFrame) -> ColumnMap:
    values = {standard: find_column(df, aliases) for standard, aliases in COLUMN_ALIASES.items()}
    return ColumnMap(**values)


def safe_numeric(values: Iterable[Any]) -> np.ndarray:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def column_name_has_unit(column_name: str | None, unit: str) -> bool:
    if column_name is None:
        return False
    normalized = normalize_column_name(column_name)
    unit = unit.lower()
    return (
        normalized == unit
        or f"({unit})" in normalized
        or f"/{unit}" in normalized
        or normalized.endswith(f" {unit}")
    )


def median_abs_finite(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    return float(np.nanmedian(np.abs(finite)))


def scale_voltage_to_v(values: np.ndarray, column_name: str | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    med = median_abs_finite(arr)
    if column_name_has_unit(column_name, "mv") or (np.isfinite(med) and med > 20.0):
        return arr / 1000.0
    return arr


def scale_current_to_a(values: np.ndarray, column_name: str | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    med = median_abs_finite(arr)
    if column_name_has_unit(column_name, "ma") or (np.isfinite(med) and med > 20.0):
        return arr / 1000.0
    return arr


def scale_capacity_to_ah(values: np.ndarray, column_name: str | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    span = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else math.nan
    max_abs = float(np.nanmax(np.abs(finite))) if finite.size else math.nan
    if (
        column_name_has_unit(column_name, "mah")
        or (np.isfinite(span) and span > 20.0)
        or (np.isfinite(max_abs) and max_abs > 20.0)
    ):
        return arr / 1000.0
    return arr


def scale_energy_to_wh(values: np.ndarray, column_name: str | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    span = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else math.nan
    max_abs = float(np.nanmax(np.abs(finite))) if finite.size else math.nan
    if (
        column_name_has_unit(column_name, "mwh")
        or (np.isfinite(span) and span > 100.0)
        or (np.isfinite(max_abs) and max_abs > 100.0)
    ):
        return arr / 1000.0
    return arr


def finite_list(values: np.ndarray | list[float] | None) -> list[float]:
    if values is None:
        return []
    arr = np.asarray(values, dtype=float)
    return [float(x) if np.isfinite(x) else math.nan for x in arr]


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def safe_min(values: np.ndarray) -> float:
    finite = finite_values(values)
    return float(np.min(finite)) if finite.size else math.nan


def safe_max(values: np.ndarray) -> float:
    finite = finite_values(values)
    return float(np.max(finite)) if finite.size else math.nan


def safe_mean(values: np.ndarray) -> float:
    finite = finite_values(values)
    return float(np.mean(finite)) if finite.size else math.nan


def safe_first(values: np.ndarray) -> float:
    finite = finite_values(values)
    return float(finite[0]) if finite.size else math.nan


def safe_last(values: np.ndarray) -> float:
    finite = finite_values(values)
    return float(finite[-1]) if finite.size else math.nan


def median_sample_period(time_s: np.ndarray) -> float:
    finite = finite_values(time_s)
    if finite.size < 2:
        return math.nan
    diffs = np.diff(finite)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size else math.nan


def resolve_project_root(project_root_arg: str | None) -> Path:
    if project_root_arg:
        return Path(project_root_arg).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "raw" / "CALCE").exists() or (candidate / "src").exists():
            return candidate
    return current


def infer_cell_id(path: Path) -> str | None:
    text = str(path).replace("\\", "/")
    patterns = [
        r"(CS2|CX2)[_/\\\-\s]?(\d+)",
        r"(CS)[_/\\\-\s]?2[_/\\\-\s]?(\d+)",
        r"(CX)[_/\\\-\s]?2[_/\\\-\s]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        prefix = match.group(1).upper()
        number = int(match.group(2))
        if prefix == "CS":
            prefix = "CS2"
        elif prefix == "CX":
            prefix = "CX2"
        return f"{prefix}_{number}"
    return None


def infer_nominal_capacity_ah(cell_id: str, override: float = math.nan) -> float:
    if np.isfinite(override):
        return float(override)
    text = str(cell_id).upper()
    if text.startswith("CS2") or "_CS2_" in f"_{text}_":
        return 1.10
    if text.startswith("CX2") or "_CX2_" in f"_{text}_":
        return 1.35
    return math.nan


def infer_battery_index(cell_id: str, sequential_index: int) -> int:
    # Keep the index unique inside the dataset. The cell number is preserved in cell_id_original.
    return int(sequential_index)


def dynamic_current_threshold_a(cell_id: str, c_rate: float, nominal_override: float = math.nan) -> float:
    nominal = infer_nominal_capacity_ah(cell_id, nominal_override)
    if np.isfinite(nominal):
        return float(abs(c_rate) * nominal)
    return 0.05


def should_exclude_file(path: Path) -> bool:
    name = str(path).lower()
    return any(keyword in name for keyword in EXCLUDE_KEYWORDS)


def get_file_date_order(path: Path) -> tuple[int, int, int, str]:
    nums = re.findall(r"\d+", path.stem)
    if len(nums) < 4:
        return (9999, 99, 99, path.name)
    try:
        month = int(nums[-3])
        day = int(nums[-2])
        year_2d = int(nums[-1])
        year = 2000 + year_2d
        if not 1 <= month <= 12:
            month = 99
        if not 1 <= day <= 31:
            day = 99
        return (year, month, day, path.name)
    except Exception:
        return (9999, 99, 99, path.name)


def load_selection_report(selection_report: Path, raw_dir: Path) -> tuple[dict[str, list[Path]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    with selection_report.open("r", encoding="utf-8") as fp:
        selection = json.load(fp)
    files_by_cell: dict[str, list[Path]] = defaultdict(list)
    for item in selection.get("selected", []):
        cell_id = item.get("cell_id")
        relative_path = item.get("relative_path")
        if not cell_id or not relative_path:
            issues.append({"stage": "selection_report", "warning": "invalid_selected_item", "item": item})
            continue
        path = raw_dir / relative_path
        if path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
            issues.append({"file": str(relative_path), "stage": "selection_report", "status": "ignored_extension"})
            continue
        if not path.exists():
            issues.append({"file": str(relative_path), "stage": "selection_report", "warning": "selected_file_not_found"})
            continue
        if should_exclude_file(path):
            issues.append({"file": str(relative_path), "stage": "selection_report", "status": "excluded_by_keyword"})
            continue
        files_by_cell[str(cell_id)].append(path)
    return {k: sorted(set(v), key=get_file_date_order) for k, v in files_by_cell.items()}, issues


def discover_excel_files(raw_dir: Path, file_pattern: str) -> tuple[dict[str, list[Path]], list[dict[str, Any]]]:
    files_by_cell: dict[str, list[Path]] = defaultdict(list)
    issues: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob(file_pattern)):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        ext = path.suffix.lower()
        rel = str(path.relative_to(raw_dir))
        if ext in IGNORED_INPUT_EXTENSIONS:
            issues.append({"file": rel, "stage": "discovery", "status": "ignored_extension", "extension": ext})
            continue
        if ext not in SUPPORTED_INPUT_EXTENSIONS:
            issues.append({"file": rel, "stage": "discovery", "status": "unsupported_extension", "extension": ext})
            continue
        if should_exclude_file(path):
            issues.append({"file": rel, "stage": "discovery", "status": "excluded_by_keyword"})
            continue
        cell_id = infer_cell_id(path)
        if cell_id is None:
            issues.append({"file": rel, "stage": "discovery", "status": "unknown_cell_id"})
            continue
        files_by_cell[cell_id].append(path)
    return {k: sorted(set(v), key=get_file_date_order) for k, v in files_by_cell.items()}, issues


def get_files_by_cell(
    raw_dir: Path,
    file_pattern: str,
    selection_report: Path | None,
    use_selection_report: bool,
) -> tuple[dict[str, list[Path]], list[dict[str, Any]], str]:
    if use_selection_report and selection_report and selection_report.exists():
        files_by_cell, issues = load_selection_report(selection_report, raw_dir)
        if files_by_cell:
            return files_by_cell, issues, "selection_report"
        issues.append({"stage": "selection_report", "warning": "selection_report_empty_falling_back_to_scan"})
    files_by_cell, issues = discover_excel_files(raw_dir, file_pattern)
    return files_by_cell, issues, "raw_directory_scan"


def is_candidate_sheet(sheet_name: str) -> bool:
    text = str(sheet_name).strip().lower()
    return any(keyword in text for keyword in ["channel", "record", "data", "test", "sheet"])


def clean_loaded_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Remove empty rows/columns and duplicated labels after an Excel read."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.duplicated()].copy()
    # Remove fully empty or autogenerated unnamed columns that survived because of comments.
    unnamed_cols = [c for c in df.columns if str(c).strip().lower().startswith("unnamed") and df[c].isna().all()]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def column_detection_score(colmap: ColumnMap) -> int:
    score = 0
    if colmap.voltage_v is not None:
        score += 10
    if colmap.current_a is not None:
        score += 10
    if colmap.time_s is not None:
        score += 3
    if colmap.charge_capacity_ah is not None:
        score += 2
    if colmap.discharge_capacity_ah is not None:
        score += 2
    if colmap.capacity_ah is not None:
        score += 1
    if colmap.temperature_c is not None:
        score += 1
    return score


def read_sheet_with_header_probe(path: Path, sheet_name: str) -> tuple[pd.DataFrame | None, ColumnMap | None, int | None, list[str], list[dict[str, Any]]]:
    """Read a CALCE sheet and infer the true header row.

    Some CALCE source-data workbooks contain comments or metadata before the
    actual table.  The v6 reader first tries the default header row and then
    scans the first HEADER_SCAN_MAX_ROWS rows, selecting the header that exposes
    voltage and current columns with the best signal metadata score.
    """
    local_issues: list[dict[str, Any]] = []
    best: tuple[int, int, pd.DataFrame, ColumnMap] | None = None
    header_candidates = [0, *range(1, HEADER_SCAN_MAX_ROWS + 1)]
    seen_headers: set[int] = set()
    for header in header_candidates:
        if header in seen_headers:
            continue
        seen_headers.add(header)
        try:
            df_try = pd.read_excel(path, sheet_name=sheet_name, header=header)
        except Exception as exc:
            if header == 0:
                local_issues.append({"file": str(path), "sheet": str(sheet_name), "stage": "read_sheet", "header": header, "error": repr(exc)})
            continue
        df_try = clean_loaded_sheet(df_try)
        if df_try.empty:
            continue
        colmap_try = detect_columns(df_try)
        score = column_detection_score(colmap_try)
        if best is None or score > best[0]:
            best = (score, header, df_try, colmap_try)
        if colmap_try.voltage_v is not None and colmap_try.current_a is not None and score >= 23:
            break

    if best is None:
        return None, None, None, [], local_issues
    score, header, df_best, colmap_best = best
    warnings: list[str] = []
    if header != 0:
        warnings.append(f"excel_header_row_detected_at_{header}")
    if colmap_best.voltage_v is None or colmap_best.current_a is None:
        local_issues.append({
            "file": str(path),
            "sheet": str(sheet_name),
            "stage": "column_detection",
            "warning": "missing_voltage_or_current_column_after_header_probe",
            "best_header": header,
            "best_score": score,
            "columns": [str(c) for c in df_best.columns],
        })
        return None, None, header, warnings, local_issues
    if colmap_best.time_s is None:
        warnings.append("time_reconstructed_from_sample_index")
    return df_best, colmap_best, header, warnings, local_issues


def read_excel_sheets(path: Path) -> tuple[list[tuple[str, pd.DataFrame, ColumnMap, list[str]]], list[dict[str, Any]]]:
    out: list[tuple[str, pd.DataFrame, ColumnMap, list[str]]] = []
    issues: list[dict[str, Any]] = []
    try:
        xls = pd.ExcelFile(path)
    except Exception as exc:
        issues.append({"file": str(path), "stage": "open_excel", "error": repr(exc), "traceback": traceback.format_exc()})
        return out, issues

    candidate_sheets = [sheet for sheet in xls.sheet_names if is_candidate_sheet(sheet)]
    if not candidate_sheets:
        candidate_sheets = list(xls.sheet_names)
        issues.append({
            "file": str(path),
            "stage": "sheet_selection",
            "warning": "no_candidate_sheet_name_found_using_all_sheets",
            "sheets": list(xls.sheet_names),
        })

    for sheet_name in candidate_sheets:
        df, colmap, header_row, sheet_warnings, local_issues = read_sheet_with_header_probe(path, str(sheet_name))
        issues.extend(local_issues)
        if df is None or colmap is None:
            continue
        if df.empty:
            issues.append({"file": str(path), "sheet": str(sheet_name), "stage": "read_sheet", "warning": "empty_sheet_after_header_probe"})
            continue
        out.append((str(sheet_name), df, colmap, sheet_warnings))
    return out, issues


def merge_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if mask.size == 0 or max_gap <= 0:
        return mask
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            i += 1
            continue
        start = i
        while i < n and not mask[i]:
            i += 1
        end = i
        if start > 0 and end < n and mask[start - 1] and mask[end] and (end - start) <= max_gap:
            mask[start:end] = True
    return mask


def contiguous_true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    in_block = False
    start = 0
    for idx, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and not in_block:
            start = idx
            in_block = True
        elif not value and in_block:
            ranges.append((start, idx))
            in_block = False
    if in_block:
        ranges.append((start, len(mask)))
    return ranges


def split_phase_segments(df: pd.DataFrame, current_column: str, threshold_a: float) -> list[tuple[str, np.ndarray, int]]:
    current = scale_current_to_a(safe_numeric(df[current_column]), current_column)
    segments: list[tuple[str, np.ndarray, int]] = []
    for phase, mask in [
        ("charge", current >= threshold_a),
        ("discharge", current <= -threshold_a),
    ]:
        merged = merge_small_gaps(mask, MAX_GAP_POINTS_TO_MERGE)
        phase_counter = 0
        for start, end in contiguous_true_ranges(merged):
            rel_indices = np.arange(start, end, dtype=int)
            if rel_indices.size < MIN_POINTS_PER_EXTRACTED_SEGMENT:
                continue
            phase_counter += 1
            segments.append((phase, rel_indices, phase_counter))
    return segments



def score_phase_segment(
    *,
    phase: str,
    voltage_v: np.ndarray,
    capacity_span_ah: float,
    n_points: int,
    nominal_capacity_ah: float,
) -> float:
    """Score candidate segments using capacity, voltage coverage, and endpoint plausibility."""
    finite_v = voltage_v[np.isfinite(voltage_v)]
    if finite_v.size == 0:
        return -1e9
    v_start = float(finite_v[0])
    v_end = float(finite_v[-1])
    v_span = float(np.nanmax(finite_v) - np.nanmin(finite_v))
    q = capacity_span_ah if np.isfinite(capacity_span_ah) else 0.0
    nominal = nominal_capacity_ah if np.isfinite(nominal_capacity_ah) and nominal_capacity_ah > 0 else 1.0
    q_ratio = q / nominal
    endpoint_bonus = 0.0
    endpoint_penalty = 0.0
    if phase == "charge":
        endpoint_bonus += 2.0 if v_end >= 3.90 else 0.0
        endpoint_penalty += 4.0 if v_end < 3.70 else 0.0
        endpoint_penalty += 1.5 if v_span < 0.35 else 0.0
    elif phase == "discharge":
        endpoint_bonus += 2.0 if v_end <= 3.35 else 0.0
        endpoint_penalty += 4.0 if v_end > 3.50 else 0.0
        endpoint_penalty += 1.5 if v_span < 0.50 else 0.0
    capacity_bonus = min(max(q_ratio, 0.0), 1.5) * 10.0
    length_bonus = min(n_points / 1000.0, 2.0)
    return capacity_bonus + endpoint_bonus + length_bonus - endpoint_penalty


def select_dominant_phase_segments(
    df: pd.DataFrame,
    colmap: ColumnMap,
    segments: list[tuple[str, np.ndarray, int]],
    nominal_capacity_ah: float,
) -> list[tuple[str, np.ndarray, int]]:
    """Keep only the dominant charge and discharge segment for each source cycle.

    The legacy CALCE files often contain rest periods, CV tails, current sign
    oscillations, and small diagnostic pulses inside the same workbook. A pure
    sign-based segmentation therefore creates many pseudo-cycles. For the
    standardized RUL pipeline, each source electrochemical cycle should contribute
    at most one charge row and one discharge row. The dominant segment is selected
    by measured phase-capacity span when available and by segment length otherwise.
    """
    selected: list[tuple[str, np.ndarray, int]] = []
    for phase in ("charge", "discharge"):
        candidates = [seg for seg in segments if seg[0] == phase]
        if not candidates:
            continue

        capacity_column, _ = choose_capacity_column(colmap, phase)
        scored: list[tuple[float, int, tuple[str, np.ndarray, int]]] = []
        for seg in candidates:
            _, rel_indices, _ = seg
            capacity_span = math.nan
            if capacity_column is not None and capacity_column in df.columns:
                q = scale_capacity_to_ah(safe_numeric(df.iloc[rel_indices][capacity_column]), capacity_column)
                finite_q = q[np.isfinite(q)]
                if finite_q.size:
                    capacity_span = float(np.nanmax(finite_q) - np.nanmin(finite_q))
            v = scale_voltage_to_v(safe_numeric(df.iloc[rel_indices][colmap.voltage_v]), colmap.voltage_v) if colmap.voltage_v else np.array([], dtype=float)
            score = score_phase_segment(
                phase=phase,
                voltage_v=v,
                capacity_span_ah=capacity_span,
                n_points=len(rel_indices),
                nominal_capacity_ah=nominal_capacity_ah,
            )
            scored.append((score, len(rel_indices), seg))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen_phase, chosen_indices, _ = scored[0][2]
        selected.append((chosen_phase, chosen_indices, 1))
    return selected


def normalize_measured_capacity(values: np.ndarray) -> np.ndarray:
    q = np.asarray(values, dtype=float)
    finite = q[np.isfinite(q)]
    if finite.size == 0:
        return np.full_like(q, np.nan, dtype=float)
    q1 = q - finite[0]
    finite_q1 = q1[np.isfinite(q1)]
    if finite_q1.size and np.nanmax(finite_q1) <= 0:
        q1 = finite[0] - q
    finite_q1 = q1[np.isfinite(q1)]
    if finite_q1.size:
        q1 = q1 - np.nanmin(finite_q1)
    return q1


def choose_capacity_column(colmap: ColumnMap, phase: str) -> tuple[str | None, str]:
    if phase == "charge" and colmap.charge_capacity_ah is not None:
        return colmap.charge_capacity_ah, "measured"
    if phase == "discharge" and colmap.discharge_capacity_ah is not None:
        return colmap.discharge_capacity_ah, "measured"
    if colmap.capacity_ah is not None:
        return colmap.capacity_ah, "measured"
    return None, "missing"


def choose_energy_column(colmap: ColumnMap, phase: str) -> tuple[str | None, str]:
    if phase == "charge" and colmap.charge_energy_wh is not None:
        return colmap.charge_energy_wh, "measured"
    if phase == "discharge" and colmap.discharge_energy_wh is not None:
        return colmap.discharge_energy_wh, "measured"
    if colmap.energy_wh is not None:
        return colmap.energy_wh, "measured"
    return None, "missing"


def localize_time(time_s: np.ndarray) -> np.ndarray:
    t = np.asarray(time_s, dtype=float)
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return np.arange(len(t), dtype=float)
    return t - finite[0]


def make_row(
    *,
    project_root: Path,
    raw_root: Path,
    file_path: Path,
    sheet_name: str,
    df: pd.DataFrame,
    colmap: ColumnMap,
    segment_indices: np.ndarray,
    phase: str,
    segment_index: int,
    battery_id: str,
    battery_index: int,
    cell_id_original: str,
    cycle_number: float,
    cycle_index: int,
    nominal_capacity_ah: float,
    chemistry: str,
    form_factor: str,
    manufacturer: str,
    nominal_voltage_v: float,
    ambient_temperature_c: float,
    charge_protocol: str,
    discharge_protocol: str,
    c_rate_charge: float,
    c_rate_discharge: float,
    dod: float,
    extraction_warnings: list[str],
) -> dict[str, Any]:
    phase_df = df.iloc[segment_indices].copy()
    if colmap.time_s is not None:
        time_s = localize_time(safe_numeric(phase_df[colmap.time_s]))
    else:
        time_s = np.arange(len(phase_df), dtype=float)
        extraction_warnings.append("time_reconstructed_from_sample_index")

    voltage_v = (
        scale_voltage_to_v(safe_numeric(phase_df[colmap.voltage_v]), colmap.voltage_v)
        if colmap.voltage_v else np.full(len(phase_df), np.nan)
    )
    current_a = (
        scale_current_to_a(safe_numeric(phase_df[colmap.current_a]), colmap.current_a)
        if colmap.current_a else np.full(len(phase_df), np.nan)
    )

    if colmap.temperature_c is not None:
        temperature_raw = safe_numeric(phase_df[colmap.temperature_c])
        if np.isfinite(temperature_raw).any():
            temperature_c = temperature_raw
            temperature_source = "measured"
            has_temperature = True
        else:
            temperature_c = np.full(len(phase_df), ambient_temperature_c if np.isfinite(ambient_temperature_c) else DEFAULT_TEMPERATURE_C)
            temperature_source = "literature_constant"
            has_temperature = False
            extraction_warnings.append("temperature_column_without_finite_values_filled_with_constant")
    else:
        fill_temperature = ambient_temperature_c if np.isfinite(ambient_temperature_c) else DEFAULT_TEMPERATURE_C
        temperature_c = np.full(len(phase_df), fill_temperature, dtype=float)
        temperature_source = "nominal_constant" if np.isfinite(ambient_temperature_c) else "literature_constant"
        has_temperature = False
        extraction_warnings.append("temperature_missing_filled_with_constant")

    capacity_column, capacity_source = choose_capacity_column(colmap, phase)
    if capacity_column:
        capacity_ah = normalize_measured_capacity(
            scale_capacity_to_ah(safe_numeric(phase_df[capacity_column]), capacity_column)
        )
        has_capacity = bool(np.isfinite(capacity_ah).any())
        if not has_capacity:
            capacity_source = "missing"
    else:
        capacity_ah = np.full(len(phase_df), np.nan, dtype=float)
        has_capacity = False

    energy_column, energy_source = choose_energy_column(colmap, phase)
    if energy_column:
        energy_wh = normalize_measured_capacity(
            scale_energy_to_wh(safe_numeric(phase_df[energy_column]), energy_column)
        )
        has_energy = bool(np.isfinite(energy_wh).any())
        if not has_energy:
            energy_source = "missing"
    else:
        energy_wh = np.full(len(phase_df), np.nan, dtype=float)
        has_energy = False

    power_w = voltage_v * current_a if np.isfinite(voltage_v).any() and np.isfinite(current_a).any() else np.full(len(phase_df), np.nan)

    rel_source_path = str(file_path.relative_to(project_root)) if file_path.resolve().is_relative_to(project_root.resolve()) else str(file_path)
    raw_cycle_id = f"{file_path.stem}|sheet={sheet_name}|cycle={cycle_number:g}"
    cycle_pair_id = f"{battery_id}_cycle_{cycle_number:g}"
    source_cycle_id = f"{raw_cycle_id}|phase={phase}|segment={segment_index}"

    row = {
        "dataset": DATASET,
        "source_file": file_path.name,
        "source_path": rel_source_path,
        "battery_id": battery_id,
        "battery_index": int(battery_index),
        "cell_id_original": cell_id_original,
        "cycle_index": int(cycle_index),
        "cycle_number": float(cycle_number) if np.isfinite(cycle_number) else math.nan,
        "source_cycle_id": source_cycle_id,
        "cycle_pair_id": cycle_pair_id,
        "phase": phase,
        "cycle_life": math.nan,
        "remaining_cycles": math.nan,
        "remaining_cycles_norm": math.nan,
        "eol_criterion": "80pct_initial_capacity",
        "eol_capacity_ah": math.nan,
        "initial_capacity_ah": math.nan,
        "nominal_capacity_ah": float(nominal_capacity_ah) if np.isfinite(nominal_capacity_ah) else math.nan,
        "soh": math.nan,
        "time_s": finite_list(time_s),
        "voltage_v": finite_list(voltage_v),
        "current_a": finite_list(current_a),
        "temperature_c": finite_list(temperature_c),
        "capacity_ah": finite_list(capacity_ah),
        "energy_wh": finite_list(energy_wh),
        "power_w": finite_list(power_w),
        "n_points": int(len(phase_df)),
        "duration_s": safe_last(time_s) - safe_first(time_s) if len(time_s) else math.nan,
        "sample_period_median_s": median_sample_period(time_s),
        "voltage_min_v": safe_min(voltage_v),
        "voltage_max_v": safe_max(voltage_v),
        "voltage_start_v": safe_first(voltage_v),
        "voltage_end_v": safe_last(voltage_v),
        "current_min_a": safe_min(current_a),
        "current_max_a": safe_max(current_a),
        "current_mean_a": safe_mean(current_a),
        "temperature_min_c": safe_min(temperature_c),
        "temperature_max_c": safe_max(temperature_c),
        "temperature_mean_c": safe_mean(temperature_c),
        "phase_capacity_ah": safe_last(capacity_ah),
        "phase_energy_wh": safe_last(energy_wh),
        "voltage_source": "measured" if np.isfinite(voltage_v).any() else "missing",
        "current_source": "measured" if np.isfinite(current_a).any() else "missing",
        "temperature_source": temperature_source,
        "capacity_source": capacity_source,
        "energy_source": energy_source,
        "chemistry": chemistry,
        "form_factor": form_factor,
        "manufacturer": manufacturer,
        "nominal_voltage_v": float(nominal_voltage_v) if np.isfinite(nominal_voltage_v) else math.nan,
        "ambient_temperature_c": float(ambient_temperature_c) if np.isfinite(ambient_temperature_c) else (DEFAULT_TEMPERATURE_C if temperature_source == "literature_constant" else math.nan),
        "charge_protocol": charge_protocol,
        "discharge_protocol": discharge_protocol,
        "c_rate_charge": float(c_rate_charge) if np.isfinite(c_rate_charge) else math.nan,
        "c_rate_discharge": float(c_rate_discharge) if np.isfinite(c_rate_discharge) else math.nan,
        "dod": float(dod) if np.isfinite(dod) else math.nan,
        "policy_readable": f"{cell_id_original}; source_file={file_path.name}; sheet={sheet_name}",
        "barcode": cell_id_original,
        "is_valid_cycle": False,
        "is_model_usable": False,
        "exclude_reason": "not_cleaned_yet",
        "cleaning_warnings": [],
        "n_removed_points": 0,
        "trim_start_index": 0,
        "trim_end_index": int(len(phase_df) - 1) if len(phase_df) else -1,
        "has_voltage": bool(np.isfinite(voltage_v).any()),
        "has_current": bool(np.isfinite(current_a).any()),
        "has_temperature": has_temperature,
        "has_capacity": has_capacity,
        "has_energy": has_energy,
        "adapter_version": ADAPTER_VERSION,
        "cleaning_version": "not_cleaned_yet",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extraction_warnings": sorted(set(extraction_warnings)),
        "raw_sheet_name": sheet_name,
        "raw_cycle_column": colmap.cycle_number or "reconstructed_from_file_order",
        "raw_phase_segment_index": int(segment_index),
        "source_row_start": int(segment_indices[0]) if len(segment_indices) else -1,
        "source_row_end": int(segment_indices[-1]) if len(segment_indices) else -1,
    }
    return row


def group_sheet_by_cycle(df: pd.DataFrame, colmap: ColumnMap, fallback_cycle_number: float) -> list[tuple[np.ndarray, float, list[str]]]:
    """Return one file-order cycle for each CALCE sheet.

    CALCE converted workbooks often contain a column named Cycle_Index or similar,
    but in this dataset this column is not a reliable electrochemical life-cycle
    counter for RUL modeling. In several files it behaves as an internal cycler
    counter or step counter, producing thousands of pseudo-cycles per cell. The
    project-level CALCE unit used here is the chronological source workbook. Each
    workbook/sheet therefore contributes at most one dominant charge segment and
    one dominant discharge segment, and the model cycle number is reconstructed
    from the ordered file sequence within each cell.
    """
    warnings = ["cycle_number_reconstructed_from_file_order"]
    if colmap.cycle_number is not None:
        warnings.append("source_cycle_column_ignored_for_calce_file_level_cycle")
    return [(np.arange(len(df), dtype=int), float(fallback_cycle_number), warnings)]


def extract_file(
    *,
    project_root: Path,
    raw_root: Path,
    file_path: Path,
    battery_id: str,
    battery_index: int,
    cell_id_original: str,
    fallback_cycle_number: float,
    nominal_capacity_ah: float,
    phase_current_threshold_a: float,
    chemistry: str,
    form_factor: str,
    manufacturer: str,
    nominal_voltage_v: float,
    ambient_temperature_c: float,
    charge_protocol: str,
    discharge_protocol: str,
    c_rate_charge: float,
    c_rate_discharge: float,
    dod: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    sheets, read_issues = read_excel_sheets(file_path)
    issues.extend(read_issues)
    if not sheets:
        issues.append({"file": str(file_path.relative_to(raw_root)), "stage": "read_excel_sheets", "warning": "no_valid_data_sheets"})
        return rows, issues

    local_pair_counter = 0
    for sheet_name, sheet_df, colmap, sheet_warnings in sheets:
        if colmap.time_s is not None:
            time_values = safe_numeric(sheet_df[colmap.time_s])
            order = np.argsort(np.where(np.isfinite(time_values), time_values, np.inf))
            sheet_df = sheet_df.iloc[order].reset_index(drop=True)
        groups = group_sheet_by_cycle(sheet_df, colmap, fallback_cycle_number)
        for group_indices, cycle_number, cycle_warnings in groups:
            cycle_df = sheet_df.iloc[group_indices].reset_index(drop=True)
            raw_segments = split_phase_segments(cycle_df, colmap.current_a, phase_current_threshold_a)
            segments = select_dominant_phase_segments(cycle_df, colmap, raw_segments, nominal_capacity_ah)
            if not segments:
                issues.append({
                    "file": str(file_path.relative_to(raw_root)),
                    "sheet": sheet_name,
                    "cycle_number": cycle_number,
                    "stage": "phase_segmentation",
                    "warning": "no_charge_or_discharge_segment",
                })
                continue
            # Increment once per electrochemical cycle pair, not once per phase.
            local_pair_counter += 1
            for phase, rel_indices, phase_segment_index in segments:
                extraction_warnings = list(sheet_warnings) + list(cycle_warnings)
                if len(rel_indices) < MIN_POINTS_PER_EXTRACTED_SEGMENT:
                    extraction_warnings.append("short_phase_segment_less_than_minimum_points")
                rows.append(make_row(
                    project_root=project_root,
                    raw_root=raw_root,
                    file_path=file_path,
                    sheet_name=sheet_name,
                    df=cycle_df,
                    colmap=colmap,
                    segment_indices=rel_indices,
                    phase=phase,
                    segment_index=phase_segment_index,
                    battery_id=battery_id,
                    battery_index=battery_index,
                    cell_id_original=cell_id_original,
                    cycle_number=cycle_number if np.isfinite(cycle_number) else float(local_pair_counter),
                    cycle_index=local_pair_counter,
                    nominal_capacity_ah=nominal_capacity_ah,
                    chemistry=chemistry,
                    form_factor=form_factor,
                    manufacturer=manufacturer,
                    nominal_voltage_v=nominal_voltage_v,
                    ambient_temperature_c=ambient_temperature_c,
                    charge_protocol=charge_protocol,
                    discharge_protocol=discharge_protocol,
                    c_rate_charge=c_rate_charge,
                    c_rate_discharge=c_rate_discharge,
                    dod=dod,
                    extraction_warnings=extraction_warnings,
                ))
    return rows, issues


def make_empty_extracted_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS + EXTRACTION_COLUMNS)


def ensure_extracted_schema(df: pd.DataFrame) -> pd.DataFrame:
    for column in STANDARD_CYCLE_COLUMNS + EXTRACTION_COLUMNS:
        if column not in df.columns:
            df[column] = pd.Series(dtype="object")
    return df[STANDARD_CYCLE_COLUMNS + EXTRACTION_COLUMNS]



def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CALCE charge and discharge cycles from converted Excel files.")
    parser.add_argument("--project-root", default=None, help="Batteries project root. Defaults to auto-detection.")
    parser.add_argument("--raw-dir", default=None, help="CALCE raw directory. Defaults to data/raw/CALCE.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to data/interim/calce/extracted.")
    parser.add_argument("--file-pattern", default="**/*.xlsx", help="Glob pattern below raw dir. Default: **/*.xlsx")
    parser.add_argument("--selection-report", default=None, help="Optional CALCE file selection report JSON.")
    parser.add_argument("--ignore-selection-report", action="store_true", help="Ignore an existing file selection report and scan raw-dir instead.")
    parser.add_argument("--phase-current-threshold-c-rate", type=float, default=DEFAULT_PHASE_CURRENT_THRESHOLD_C_RATE)
    parser.add_argument("--nominal-capacity-ah", type=float, default=float("nan"), help="Override nominal capacity for all CALCE cells.")
    parser.add_argument("--nominal-voltage-v", type=float, default=float("nan"))
    parser.add_argument("--ambient-temperature-c", type=float, default=float("nan"))
    parser.add_argument("--chemistry", default="unknown")
    parser.add_argument("--form-factor", default="unknown")
    parser.add_argument("--manufacturer", default="unknown")
    parser.add_argument("--charge-protocol", default="CALCE source cycling protocol")
    parser.add_argument("--discharge-protocol", default="CALCE source cycling protocol")
    parser.add_argument("--c-rate-charge", type=float, default=float("nan"))
    parser.add_argument("--c-rate-discharge", type=float, default=float("nan"))
    parser.add_argument("--dod", type=float, default=float("nan"))
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    raw_root = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else project_root / "data" / "raw" / "CALCE"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else project_root / "data" / "interim" / DATASET_DIRNAME / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise FileNotFoundError(f"CALCE raw directory not found: {raw_root}")

    selection_report = Path(args.selection_report).expanduser().resolve() if args.selection_report else project_root / "data" / "reports" / "calce_file_selection_report.json"
    files_by_cell, discovery_issues, input_mode = get_files_by_cell(
        raw_dir=raw_root,
        file_pattern=args.file_pattern,
        selection_report=selection_report,
        use_selection_report=not args.ignore_selection_report,
    )

    if not files_by_cell:
        raise FileNotFoundError(f"No CALCE Excel files found under {raw_root}")

    cell_ids = sorted(files_by_cell)
    battery_index_map = {cell_id: infer_battery_index(cell_id, idx + 1) for idx, cell_id in enumerate(cell_ids)}
    fallback_cycle_counter_by_cell: dict[str, int] = defaultdict(int)

    charge_rows: list[dict[str, Any]] = []
    discharge_rows: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = list(discovery_issues)
    failed_reads: list[dict[str, Any]] = []

    for cell_id in cell_ids:
        battery_id = f"{DATASET}_{cell_id}"
        nominal_capacity_ah = infer_nominal_capacity_ah(cell_id, args.nominal_capacity_ah)
        threshold_a = dynamic_current_threshold_a(cell_id, args.phase_current_threshold_c_rate, args.nominal_capacity_ah)
        for file_path in files_by_cell[cell_id]:
            fallback_cycle_counter_by_cell[cell_id] += 1
            fallback_cycle_number = float(fallback_cycle_counter_by_cell[cell_id])
            try:
                rows, issues = extract_file(
                    project_root=project_root,
                    raw_root=raw_root,
                    file_path=file_path,
                    battery_id=battery_id,
                    battery_index=battery_index_map[cell_id],
                    cell_id_original=cell_id,
                    fallback_cycle_number=fallback_cycle_number,
                    nominal_capacity_ah=nominal_capacity_ah,
                    phase_current_threshold_a=threshold_a,
                    chemistry=args.chemistry,
                    form_factor=args.form_factor,
                    manufacturer=args.manufacturer,
                    nominal_voltage_v=args.nominal_voltage_v,
                    ambient_temperature_c=args.ambient_temperature_c,
                    charge_protocol=args.charge_protocol,
                    discharge_protocol=args.discharge_protocol,
                    c_rate_charge=args.c_rate_charge,
                    c_rate_discharge=args.c_rate_discharge,
                    dod=args.dod,
                )
            except Exception as exc:
                failed_reads.append({
                    "file": str(file_path.relative_to(raw_root)),
                    "cell_id": cell_id,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
                continue
            all_issues.extend(issues)
            for row in rows:
                if row["phase"] == "charge":
                    charge_rows.append(row)
                elif row["phase"] == "discharge":
                    discharge_rows.append(row)

    charge_df = ensure_extracted_schema(pd.DataFrame(charge_rows) if charge_rows else make_empty_extracted_frame())
    discharge_df = ensure_extracted_schema(pd.DataFrame(discharge_rows) if discharge_rows else make_empty_extracted_frame())

    charge_output = output_dir / "calce_charge_extracted_cycles.parquet"
    discharge_output = output_dir / "calce_discharge_extracted_cycles.parquet"
    charge_df.to_parquet(charge_output, index=False)
    discharge_df.to_parquet(discharge_output, index=False)

    report = {
        "dataset": DATASET,
        "adapter_version": ADAPTER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "raw_root": str(raw_root),
        "input_mode": input_mode,
        "selection_report": str(selection_report) if selection_report.exists() else None,
        "parameters": {
            "supported_input_extensions": sorted(SUPPORTED_INPUT_EXTENSIONS),
            "ignored_input_extensions": sorted(IGNORED_INPUT_EXTENSIONS),
            "exclude_keywords": EXCLUDE_KEYWORDS,
            "phase_current_threshold_c_rate": args.phase_current_threshold_c_rate,
            "min_points_per_extracted_segment": MIN_POINTS_PER_EXTRACTED_SEGMENT,
            "max_gap_points_to_merge": MAX_GAP_POINTS_TO_MERGE,
            "default_temperature_c": DEFAULT_TEMPERATURE_C,
            "header_scan_max_rows": HEADER_SCAN_MAX_ROWS,
            "cycle_reconstruction": "file_sheet_level_one_dominant_charge_and_discharge_segment",
            "cycle_grouping_policy": "one_file_order_cycle_per_source_workbook_sheet",
        },
        "n_cells": len(cell_ids),
        "files_by_cell": {cell: len(files) for cell, files in files_by_cell.items()},
        "selected_files": {cell: [str(p.relative_to(raw_root)) for p in files] for cell, files in files_by_cell.items()},
        "n_failed_reads": len(failed_reads),
        "failed_reads": failed_reads,
        "n_issues": len(all_issues),
        "issues": all_issues[:2000],
        "issue_truncated": len(all_issues) > 2000,
        "n_charge_rows": int(len(charge_df)),
        "n_discharge_rows": int(len(discharge_df)),
        "outputs": {"charge": str(charge_output), "discharge": str(discharge_output)},
    }
    report_path = output_dir / "calce_extraction_report.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(json_safe(report), fp, indent=2, ensure_ascii=False)

    print(f"[OK] Charge extracted cycles: {charge_output} ({len(charge_df)} rows)")
    print(f"[OK] Discharge extracted cycles: {discharge_output} ({len(discharge_df)} rows)")
    print(f"[OK] Extraction report: {report_path}")


if __name__ == "__main__":
    main()
