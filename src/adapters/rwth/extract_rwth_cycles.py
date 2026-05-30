# -*- coding: utf-8 -*-
"""
RWTH Aachen / ISEA battery dataset adapter - extraction stage.

This script reads the original RWTH ZIP files and extracts defensible charge and
 discharge segments into two independent parquet files. The output is intentionally
conservative: it preserves traceability and does not apply final model-usability
rejections. Final standardization is performed by clean_rwth_cycles.py.

Default project layout
----------------------
Batteries/
  data/raw/RWTH/*.zip
  data/interim/rwth/extracted/rwth_charge_extracted_cycles.parquet
  data/interim/rwth/extracted/rwth_discharge_extracted_cycles.parquet
  data/reports/rwth/extraction/rwth_extraction_report.csv
  data/reports/rwth/extraction/rwth_extraction_summary.json

Run
---
python src/adapters/rwth/extract_rwth_cycles.py
python src/adapters/rwth/extract_rwth_cycles.py --max-zips 20
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DATASET = "RWTH"
ADAPTER_VERSION = "rwth_extract_v1.1"

DEFAULT_BASE_DIR = Path.cwd()
DEFAULT_RAW_DIR = DEFAULT_BASE_DIR / "data" / "raw" / "RWTH"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "data" / "interim" / "rwth" / "extracted"
DEFAULT_REPORT_DIR = DEFAULT_BASE_DIR / "data" / "reports" / "rwth" / "extraction"

TABULAR_EXTENSIONS = {".csv", ".txt", ".dat", ".asc"}

V_MIN_VALID = 1.5
V_MAX_VALID = 5.0
TEMP_MIN_VALID = -30.0
TEMP_MAX_VALID = 100.0
MIN_POINTS_DEFAULT = 20
MIN_DURATION_S_DEFAULT = 60.0
MIN_ABS_CURRENT_A_DEFAULT = 0.05
MIN_VOLTAGE_CHANGE_V_DEFAULT = 0.03

EXTRACTED_COLUMNS = [
    "dataset", "source_file", "source_path", "internal_file", "battery_id", "battery_index",
    "cell_id_original", "source_cycle_id", "cycle_pair_id", "phase", "raw_test_type",
    "segment_role", "segment_index_in_file", "segment_start_row", "segment_end_row",
    "source_timestamp", "source_protocol", "policy_readable", "barcode", "chemistry",
    "form_factor", "manufacturer", "nominal_capacity_ah", "nominal_voltage_v",
    "ambient_temperature_c", "charge_protocol", "discharge_protocol", "c_rate_charge",
    "c_rate_discharge", "dod", "time_s", "voltage_v", "current_a", "temperature_c",
    "capacity_ah", "capacity_source", "temperature_source", "n_points", "duration_s",
    "voltage_start_v", "voltage_end_v", "voltage_min_v", "voltage_max_v",
    "current_mean_a", "phase_capacity_ah", "extraction_warnings", "adapter_version",
    "created_at",
]


@dataclass
class ZipMetadata:
    zip_name: str
    zip_stem: str
    source_path: str
    battery_id: str
    battery_index: int
    cell_id_original: str
    raw_test_type: str
    protocol: str
    timestamp_text: str
    timestamp_iso: str
    channel: str
    raw_parts: str


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_timestamp_from_text(text: str) -> tuple[str, str]:
    match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{6})", text)
    if not match:
        return "", ""
    raw = match.group(1)
    ts = pd.to_datetime(raw, format="%Y-%m-%d %H%M%S", errors="coerce")
    if pd.isna(ts):
        return raw, ""
    return raw, ts.isoformat()


def infer_test_type(stem: str) -> str:
    text = stem.lower()
    if re.search(r"(^|[_=\s\-])eis($|[_=\s\-]|\d)", stem, flags=re.IGNORECASE):
        return "EIS"
    if "bol" in text:
        return "BOL"
    if "eol" in text:
        return "EOL"
    if re.search(r"cuv|checkup|check-up|reference|rpt|cha", stem, flags=re.IGNORECASE):
        return "RPT"
    if re.search(r"zyk|alterung|aging|massenalterung", stem, flags=re.IGNORECASE):
        return "AGING"
    return "UNKNOWN"


def parse_metadata_from_zip_name(zip_path: Path, raw_dir: Path) -> ZipMetadata:
    stem = zip_path.stem
    parts = [p.strip() for p in stem.split("=")]
    raw_parts = " | ".join(parts)

    cell_id_original = "unknown"
    battery_index = -1
    battery_id = "RWTH_unknown"
    cell_match = re.search(r"sanyo\s*(\d+)", stem, flags=re.IGNORECASE)
    if cell_match:
        battery_index = int(cell_match.group(1))
        cell_id_original = f"sanyo {battery_index:03d}"
        battery_id = f"RWTH_sanyo_{battery_index:03d}"

    protocol_match = re.search(r"(TBA[^=\s]*)", stem, flags=re.IGNORECASE)
    protocol = protocol_match.group(1).strip() if protocol_match else "unknown"

    channel_match = re.search(r"(Kreis\s*[\w\-\s]+)", stem, flags=re.IGNORECASE)
    channel = re.sub(r"\s+", " ", channel_match.group(1)).strip() if channel_match else "unknown"

    timestamp_text, timestamp_iso = parse_timestamp_from_text(stem)

    try:
        source_path = str(zip_path.relative_to(raw_dir.parent)).replace("\\", "/")
    except Exception:
        source_path = str(zip_path).replace("\\", "/")

    return ZipMetadata(
        zip_name=zip_path.name,
        zip_stem=stem,
        source_path=source_path,
        battery_id=battery_id,
        battery_index=battery_index,
        cell_id_original=cell_id_original,
        raw_test_type=infer_test_type(stem),
        protocol=protocol,
        timestamp_text=timestamp_text,
        timestamp_iso=timestamp_iso,
        channel=channel,
        raw_parts=raw_parts,
    )


def is_eis_by_name_or_columns(zip_name: str, internal_name: str, columns: Iterable[str] | None = None) -> bool:
    text = f"{zip_name} {internal_name}"
    if (
        re.search(r"(^|[_=\s\-])EIS($|[_=\s\-]|\d)", text, flags=re.IGNORECASE)
        or re.search(r"EISkanal", text, flags=re.IGNORECASE)
        or re.search(r"impedance|impedanz", text, flags=re.IGNORECASE)
    ):
        return True
    if columns is not None:
        cols = {str(c).strip() for c in columns}
        eis_cols = {
            "Zreal1", "Zimg1", "ActFreq", "StartFreq", "EndFreq", "SetFreq",
            "Phase", "Betrag", "Re", "Im", "Frequency", "Freq",
        }
        if len(cols.intersection(eis_cols)) >= 3:
            return True
    return False


def sniff_csv_params_from_bytes(raw: bytes) -> tuple[str, str, str]:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1", "iso-8859-1"]
    text = ""
    encoding = "latin1"
    for candidate in encodings:
        try:
            text = raw.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue

    lines = [line for line in text.splitlines() if line.strip()][:80]
    sample = "\n".join(lines)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        sep = dialect.delimiter
    except Exception:
        counts = {sep_candidate: sample.count(sep_candidate) for sep_candidate in [";", ",", "\t"]}
        sep = max(counts, key=counts.get) if counts else ";"
        if counts.get(sep, 0) == 0:
            sep = ";"

    decimal_comma = len(re.findall(r"\d+,\d+", sample))
    decimal_dot = len(re.findall(r"\d+\.\d+", sample))
    decimal = "," if decimal_comma > decimal_dot and sep != "," else "."
    return encoding, sep, decimal


def read_tabular_from_zip(zip_path: Path, internal_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(internal_name) as file:
            raw = file.read()
    encoding, sep, decimal = sniff_csv_params_from_bytes(raw[:250_000])
    buffer = BytesIO(raw)
    df = pd.read_csv(
        TextIOWrapper(buffer, encoding=encoding, errors="replace"),
        sep=sep,
        decimal=decimal,
        engine="python",
        on_bad_lines="skip",
    )
    return df.dropna(how="all").copy()


def find_column(columns: Iterable[str], patterns: list[str]) -> str | None:
    cols = [str(c).strip() for c in columns]
    lower_to_original = {c.lower(): c for c in cols}
    for pattern in patterns:
        if pattern.lower() in lower_to_original:
            return lower_to_original[pattern.lower()]
    for col in cols:
        for pattern in patterns:
            if re.search(pattern, col, flags=re.IGNORECASE):
                return col
    return None


def find_temperature_column(columns: Iterable[str]) -> str | None:
    preferred = ["Temp1", "Temp5", "Temp7", "Temp12", "Temp", "ActTemp", "Temperatur", "Temperature"]
    return find_column(columns, preferred) or find_column(columns, [r"^temp", r"temper"])


def parse_time_seconds(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().sum() >= max(2, int(0.5 * len(series))):
        first = dt.dropna().iloc[0]
        return (dt - first).dt.total_seconds().to_numpy(dtype=float).copy()
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float).copy()
    finite = values[np.isfinite(values)]
    if finite.size:
        values = values - finite[0]
    return values


def numeric_series(df: pd.DataFrame, column: str | None, default: float = np.nan) -> np.ndarray:
    if column is None:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float).copy()


def prepare_dataframe(df_raw: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    if df_raw.empty:
        return None, "empty_table"

    cols = list(df_raw.columns)
    time_col = find_column(cols, ["Zeit", "Time", "time", "t"])
    voltage_col = find_column(cols, ["Spannung", "Voltage", "U", "Voltage[V]", "Voltage (V)"])
    current_col = find_column(cols, ["Strom", "Current", "I", "Current[A]", "Current (A)"])
    cycle_col = find_column(cols, ["Zyklus", "Cycle", "cycle"])
    step_col = find_column(cols, ["Schritt", "Step", "step"])
    state_col = find_column(cols, ["Zustand", "State", "Status"])
    temp_col = find_temperature_column(cols)

    capacity_cols = {
        "ah_ela": find_column(cols, ["AhEla", "Ah_Ela"]),
        "ah_akku": find_column(cols, ["AhAkku", "Ah_Akku"]),
        "ah_lad": find_column(cols, ["AhLad", "Ah_Lad"]),
        "ah_step": find_column(cols, ["AhStep", "Ah_Step", "Ah", "Capacity", "Q"]),
    }

    missing = []
    if time_col is None:
        missing.append("time")
    if voltage_col is None:
        missing.append("voltage")
    if current_col is None:
        missing.append("current")
    if missing:
        return None, "missing_required_columns:" + ",".join(missing)

    out = pd.DataFrame()
    out["time_s"] = parse_time_seconds(df_raw[time_col])
    out["voltage_v"] = numeric_series(df_raw, voltage_col)
    out["current_a"] = numeric_series(df_raw, current_col)
    out["source_cycle"] = numeric_series(df_raw, cycle_col, default=1.0)
    out["source_step"] = numeric_series(df_raw, step_col)
    out["state"] = df_raw[state_col].astype(str).to_numpy() if state_col else ""
    out["temperature_c"] = numeric_series(df_raw, temp_col)
    out["temperature_source"] = "measured" if temp_col else "unknown_nan"

    for output_col, source_col in capacity_cols.items():
        out[output_col] = numeric_series(df_raw, source_col)

    mask = (
        np.isfinite(out["time_s"])
        & np.isfinite(out["voltage_v"])
        & np.isfinite(out["current_a"])
        & (out["voltage_v"] >= V_MIN_VALID)
        & (out["voltage_v"] <= V_MAX_VALID)
    )
    out = out.loc[mask].copy()
    if out.empty:
        return None, "no_valid_rows_after_basic_filter"

    temperature = out["temperature_c"].to_numpy(dtype=float).copy()
    valid_temperature = np.isfinite(temperature) & (temperature >= TEMP_MIN_VALID) & (temperature <= TEMP_MAX_VALID)
    if valid_temperature.any():
        replacement = float(np.nanmedian(temperature[valid_temperature]))
        temperature[~valid_temperature] = replacement
        out["temperature_source"] = "measured"
    else:
        temperature[:] = np.nan
        out["temperature_source"] = "unknown_nan"
    out["temperature_c"] = temperature

    out = out.sort_values(["source_cycle", "time_s"]).reset_index(drop=True)
    return out, "ok"


def contiguous_true_regions(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= min_len:
                regions.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= min_len:
        regions.append((start, len(mask)))
    return regions


def integrate_abs_current_to_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    n = min(len(time_s), len(current_a))
    if n < 2:
        return np.array([], dtype=float)
    time = np.asarray(time_s[:n], dtype=float)
    current = np.asarray(current_a[:n], dtype=float)
    dt = np.diff(time)
    current_mid = 0.5 * (np.abs(current[1:]) + np.abs(current[:-1]))
    dq = current_mid * dt / 3600.0
    dq = np.where(np.isfinite(dq) & (dq >= 0), dq, 0.0)
    return np.concatenate([[0.0], np.cumsum(dq)])


def build_local_capacity(segment: pd.DataFrame, phase: str) -> tuple[np.ndarray, str]:
    if phase == "charge":
        candidates = ["ah_lad", "ah_step", "ah_akku", "ah_ela"]
    else:
        candidates = ["ah_ela", "ah_step", "ah_akku", "ah_lad"]

    measured_candidates: list[tuple[float, np.ndarray, str]] = []
    for col in candidates:
        arr = segment[col].to_numpy(dtype=float) if col in segment else np.array([], dtype=float)
        if arr.size == 0 or np.isfinite(arr).sum() < max(5, int(0.5 * len(segment))):
            continue
        local = arr - np.nanmin(arr)
        local = np.where(np.isfinite(local), local, np.nan)
        qmax = float(np.nanmax(local)) if np.isfinite(local).any() else np.nan
        if np.isfinite(qmax) and 0.02 <= qmax <= 3.0:
            local = np.maximum.accumulate(np.nan_to_num(local, nan=0.0))
            measured_candidates.append((qmax, local, col))

    if measured_candidates:
        measured_candidates.sort(key=lambda item: item[0], reverse=True)
        _, capacity, column = measured_candidates[0]
        return capacity, f"measured:{column}"

    capacity = integrate_abs_current_to_ah(
        segment["time_s"].to_numpy(dtype=float) - float(np.nanmin(segment["time_s"])),
        segment["current_a"].to_numpy(dtype=float),
    )
    return capacity, "integrated_current"


def edge_median(values: np.ndarray, side: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    n_edge = max(3, int(0.1 * finite.size))
    edge = finite[:n_edge] if side == "start" else finite[-n_edge:]
    return float(np.nanmedian(edge))


def classify_phase_by_voltage(voltage_v: np.ndarray, min_voltage_change_v: float) -> str | None:
    start = edge_median(voltage_v, "start")
    end = edge_median(voltage_v, "end")
    if not np.isfinite(start) or not np.isfinite(end):
        return None
    delta = end - start
    if delta >= min_voltage_change_v:
        return "charge"
    if delta <= -min_voltage_change_v:
        return "discharge"
    return None


def classify_segment_role(test_type: str, phase: str, phase_capacity_ah: float) -> str:
    if not np.isfinite(phase_capacity_ah):
        return f"{phase}_unknown_capacity"
    if phase_capacity_ah >= 1.2:
        if test_type == "BOL":
            return f"{phase}_bol_full"
        if test_type == "EOL":
            return f"{phase}_eol_full"
        if test_type == "RPT":
            return f"{phase}_rpt_full"
        return f"{phase}_full_unclassified"
    if 0.05 <= phase_capacity_ah < 1.2:
        if test_type == "AGING":
            return f"{phase}_aging_partial"
        return f"{phase}_partial_unclassified"
    return f"{phase}_low_capacity"


def source_cycle_id(value: Any) -> str:
    try:
        number = float(value)
        if np.isfinite(number):
            if abs(number - int(number)) < 1e-9:
                return str(int(number))
            return f"{number:.6g}"
    except Exception:
        pass
    return str(value)


def segment_to_row(
    meta: ZipMetadata,
    internal_name: str,
    segment: pd.DataFrame,
    phase: str,
    local_idx: int,
    start_row: int,
    end_row: int,
) -> dict[str, Any]:
    time_s = segment["time_s"].to_numpy(dtype=float)
    time_s = time_s - float(np.nanmin(time_s))
    voltage_v = segment["voltage_v"].to_numpy(dtype=float)
    current_a = segment["current_a"].to_numpy(dtype=float)
    temperature_c = segment["temperature_c"].to_numpy(dtype=float)
    capacity_ah, capacity_source = build_local_capacity(segment, phase)

    n = min(len(time_s), len(voltage_v), len(current_a), len(temperature_c), len(capacity_ah))
    time_s = time_s[:n]
    voltage_v = voltage_v[:n]
    current_a = current_a[:n]
    temperature_c = temperature_c[:n]
    capacity_ah = capacity_ah[:n]

    finite_mask = np.isfinite(time_s) & np.isfinite(voltage_v)
    if np.isfinite(current_a).any():
        finite_mask &= np.isfinite(current_a)
    if np.isfinite(capacity_ah).any():
        finite_mask &= np.isfinite(capacity_ah)

    time_s = time_s[finite_mask]
    voltage_v = voltage_v[finite_mask]
    current_a = current_a[finite_mask]
    temperature_c = temperature_c[finite_mask]
    capacity_ah = capacity_ah[finite_mask]

    source_cycle = source_cycle_id(segment["source_cycle"].iloc[0])
    pair_id = f"{meta.battery_id}__{meta.timestamp_text or meta.zip_stem}__cycle_{source_cycle}"
    phase_capacity_ah = float(np.nanmax(capacity_ah)) if len(capacity_ah) else np.nan
    duration_s = float(np.nanmax(time_s) - np.nanmin(time_s)) if len(time_s) else np.nan

    temperature_source = str(segment["temperature_source"].iloc[0]) if "temperature_source" in segment else "unknown_nan"
    role = classify_segment_role(meta.raw_test_type, phase, phase_capacity_ah)
    warning_list = []
    if capacity_source == "integrated_current":
        warning_list.append("capacity_reconstructed_from_current")
    if temperature_source == "unknown_nan":
        warning_list.append("temperature_missing")

    return {
        "dataset": DATASET,
        "source_file": meta.zip_name,
        "source_path": meta.source_path,
        "internal_file": internal_name,
        "battery_id": meta.battery_id,
        "battery_index": meta.battery_index,
        "cell_id_original": meta.cell_id_original,
        "source_cycle_id": source_cycle,
        "cycle_pair_id": pair_id,
        "phase": phase,
        "raw_test_type": meta.raw_test_type,
        "segment_role": role,
        "segment_index_in_file": int(local_idx),
        "segment_start_row": int(start_row),
        "segment_end_row": int(end_row),
        "source_timestamp": meta.timestamp_iso,
        "source_protocol": meta.protocol,
        "policy_readable": f"{meta.raw_test_type}; {meta.protocol}; {role}",
        "barcode": meta.channel,
        "chemistry": "NMC/graphite",
        "form_factor": "18650",
        "manufacturer": "Sanyo/Panasonic",
        "nominal_capacity_ah": 2.05,
        "nominal_voltage_v": 3.6,
        "ambient_temperature_c": np.nan,
        "charge_protocol": "RWTH source protocol; extracted from voltage-increasing active-current segments",
        "discharge_protocol": "RWTH source protocol; extracted from voltage-decreasing active-current segments",
        "c_rate_charge": np.nan,
        "c_rate_discharge": np.nan,
        "dod": np.nan,
        "time_s": time_s.tolist(),
        "voltage_v": voltage_v.tolist(),
        "current_a": current_a.tolist(),
        "temperature_c": temperature_c.tolist(),
        "capacity_ah": capacity_ah.tolist(),
        "capacity_source": capacity_source,
        "temperature_source": temperature_source,
        "n_points": int(len(time_s)),
        "duration_s": duration_s,
        "voltage_start_v": float(voltage_v[0]) if len(voltage_v) else np.nan,
        "voltage_end_v": float(voltage_v[-1]) if len(voltage_v) else np.nan,
        "voltage_min_v": float(np.nanmin(voltage_v)) if len(voltage_v) else np.nan,
        "voltage_max_v": float(np.nanmax(voltage_v)) if len(voltage_v) else np.nan,
        "current_mean_a": float(np.nanmean(current_a)) if len(current_a) and np.isfinite(current_a).any() else np.nan,
        "phase_capacity_ah": phase_capacity_ah,
        "extraction_warnings": warning_list,
        "adapter_version": ADAPTER_VERSION,
        "created_at": now_iso(),
    }


def segment_prepared_dataframe(
    df: pd.DataFrame,
    meta: ZipMetadata,
    internal_name: str,
    min_points: int,
    min_duration_s: float,
    min_abs_current_a: float,
    min_voltage_change_v: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    rejected_reasons: list[str] = []

    for _, group in df.groupby("source_cycle", dropna=False):
        group = group.sort_values("time_s").reset_index(drop=True)
        if len(group) < min_points:
            rejected_reasons.append("few_points_source_cycle")
            continue

        current = group["current_a"].to_numpy(dtype=float)
        active = np.isfinite(current) & (np.abs(current) >= min_abs_current_a)
        candidate_regions: list[tuple[int, int]] = []
        candidate_regions.extend(contiguous_true_regions(active & (current > min_abs_current_a), min_points))
        candidate_regions.extend(contiguous_true_regions(active & (current < -min_abs_current_a), min_points))
        if not candidate_regions:
            candidate_regions.extend(contiguous_true_regions(active, min_points))
        candidate_regions = sorted(set(candidate_regions))

        local_idx = 0
        for start, end in candidate_regions:
            segment = group.iloc[start:end].copy().reset_index(drop=True)
            if len(segment) < min_points:
                rejected_reasons.append("few_points_segment")
                continue
            time = segment["time_s"].to_numpy(dtype=float)
            duration = float(np.nanmax(time) - np.nanmin(time)) if np.isfinite(time).any() else np.nan
            if not np.isfinite(duration) or duration < min_duration_s:
                rejected_reasons.append("short_segment")
                continue
            phase = classify_phase_by_voltage(segment["voltage_v"].to_numpy(dtype=float), min_voltage_change_v)
            if phase not in {"charge", "discharge"}:
                rejected_reasons.append("ambiguous_voltage_trend")
                continue
            capacity, _ = build_local_capacity(segment, phase)
            qmax = float(np.nanmax(capacity)) if len(capacity) and np.isfinite(capacity).any() else np.nan
            if not np.isfinite(qmax) or qmax <= 0.02:
                rejected_reasons.append("insufficient_phase_capacity")
                continue
            local_idx += 1
            rows.append(segment_to_row(meta, internal_name, segment, phase, local_idx, start, end))

    return rows, rejected_reasons


def zip_internal_tabular_files(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return [
            info.filename
            for info in zf.infolist()
            if (not info.is_dir()) and Path(info.filename).suffix.lower() in TABULAR_EXTENSIONS
        ]


def extract_one_zip(
    zip_path: Path,
    raw_dir: Path,
    min_points: int,
    min_duration_s: float,
    min_abs_current_a: float,
    min_voltage_change_v: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    meta = parse_metadata_from_zip_name(zip_path, raw_dir)
    rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []

    try:
        internal_files = zip_internal_tabular_files(zip_path)
    except Exception as exc:
        report_rows.append({
            **asdict(meta), "internal_file": "", "status": "zip_error",
            "message": f"{type(exc).__name__}: {exc}", "n_rows_table": 0,
            "n_prepared_rows": 0, "n_charge_segments": 0, "n_discharge_segments": 0,
        })
        return rows, report_rows

    if not internal_files:
        report_rows.append({
            **asdict(meta), "internal_file": "", "status": "no_tabular_file_inside_zip",
            "message": "", "n_rows_table": 0, "n_prepared_rows": 0,
            "n_charge_segments": 0, "n_discharge_segments": 0,
        })
        return rows, report_rows

    for internal_name in internal_files:
        try:
            df_raw = read_tabular_from_zip(zip_path, internal_name)
            if is_eis_by_name_or_columns(meta.zip_name, internal_name, df_raw.columns):
                report_rows.append({
                    **asdict(meta), "internal_file": internal_name, "status": "skipped_eis",
                    "message": "", "n_rows_table": int(len(df_raw)), "n_prepared_rows": 0,
                    "n_charge_segments": 0, "n_discharge_segments": 0,
                })
                continue
            df, message = prepare_dataframe(df_raw)
            if df is None:
                report_rows.append({
                    **asdict(meta), "internal_file": internal_name, "status": "skipped_prepare",
                    "message": message, "n_rows_table": int(len(df_raw)), "n_prepared_rows": 0,
                    "n_charge_segments": 0, "n_discharge_segments": 0,
                })
                continue
            segment_rows, reasons = segment_prepared_dataframe(
                df=df,
                meta=meta,
                internal_name=internal_name,
                min_points=min_points,
                min_duration_s=min_duration_s,
                min_abs_current_a=min_abs_current_a,
                min_voltage_change_v=min_voltage_change_v,
            )
            rows.extend(segment_rows)
            n_charge = sum(1 for row in segment_rows if row["phase"] == "charge")
            n_discharge = sum(1 for row in segment_rows if row["phase"] == "discharge")
            report_rows.append({
                **asdict(meta), "internal_file": internal_name, "status": "processed",
                "message": ";".join(sorted(set(reasons))) if reasons else "",
                "n_rows_table": int(len(df_raw)), "n_prepared_rows": int(len(df)),
                "n_source_cycles": int(df["source_cycle"].nunique(dropna=True)),
                "n_charge_segments": int(n_charge), "n_discharge_segments": int(n_discharge),
            })
        except Exception as exc:
            report_rows.append({
                **asdict(meta), "internal_file": internal_name, "status": "read_or_extract_error",
                "message": f"{type(exc).__name__}: {exc}", "n_rows_table": 0,
                "n_prepared_rows": 0, "n_charge_segments": 0, "n_discharge_segments": 0,
            })

    return rows, report_rows


def empty_extracted_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EXTRACTED_COLUMNS)


def write_phase_file(rows: list[dict[str, Any]], phase: str, output_dir: Path) -> Path:
    output_file = output_dir / f"rwth_{phase}_extracted_cycles.parquet"
    phase_rows = [row for row in rows if row.get("phase") == phase]
    df = pd.DataFrame(phase_rows) if phase_rows else empty_extracted_frame()
    for col in EXTRACTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[EXTRACTED_COLUMNS]
    df.to_parquet(output_file, index=False)
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RWTH charge and discharge cycle segments from raw ZIP files.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--max-zips", type=int, default=None, help="Optional limit for quick tests.")
    parser.add_argument("--min-points", type=int, default=MIN_POINTS_DEFAULT)
    parser.add_argument("--min-duration-s", type=float, default=MIN_DURATION_S_DEFAULT)
    parser.add_argument("--min-abs-current-a", type=float, default=MIN_ABS_CURRENT_A_DEFAULT)
    parser.add_argument("--min-voltage-change-v", type=float, default=MIN_VOLTAGE_CHANGE_V_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir or args.base_dir / "data" / "raw" / "RWTH"
    output_dir = ensure_dir(args.output_dir or args.base_dir / "data" / "interim" / "rwth" / "extracted")
    report_dir = ensure_dir(args.report_dir or args.base_dir / "data" / "reports" / "rwth" / "extraction")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    zip_files = sorted(raw_dir.glob("*.zip"))
    if args.max_zips is not None:
        zip_files = zip_files[: args.max_zips]
    if not zip_files:
        raise FileNotFoundError(f"No ZIP files found in: {raw_dir}")

    all_rows: list[dict[str, Any]] = []
    all_report_rows: list[dict[str, Any]] = []
    for zip_path in tqdm(zip_files, desc="Extracting RWTH ZIP files", unit="zip"):
        rows, report_rows = extract_one_zip(
            zip_path=zip_path,
            raw_dir=raw_dir,
            min_points=args.min_points,
            min_duration_s=args.min_duration_s,
            min_abs_current_a=args.min_abs_current_a,
            min_voltage_change_v=args.min_voltage_change_v,
        )
        all_rows.extend(rows)
        all_report_rows.extend(report_rows)

    charge_file = write_phase_file(all_rows, "charge", output_dir)
    discharge_file = write_phase_file(all_rows, "discharge", output_dir)

    report_df = pd.DataFrame(all_report_rows)
    report_file = report_dir / "rwth_extraction_report.csv"
    report_df.to_csv(report_file, index=False, encoding="utf-8-sig")

    summary = {
        "dataset": DATASET,
        "adapter_version": ADAPTER_VERSION,
        "created_at": now_iso(),
        "raw_dir": str(raw_dir),
        "n_zip_files": len(zip_files),
        "n_total_segments": len(all_rows),
        "n_charge_segments": int(sum(1 for row in all_rows if row.get("phase") == "charge")),
        "n_discharge_segments": int(sum(1 for row in all_rows if row.get("phase") == "discharge")),
        "n_batteries": int(pd.Series([row.get("battery_id") for row in all_rows]).nunique()) if all_rows else 0,
        "charge_file": str(charge_file),
        "discharge_file": str(discharge_file),
        "report_file": str(report_file),
        "status_counts": report_df["status"].value_counts(dropna=False).to_dict() if not report_df.empty else {},
    }
    summary_file = report_dir / "rwth_extraction_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nRWTH extraction completed.")
    print(f"Charge file: {charge_file}")
    print(f"Discharge file: {discharge_file}")
    print(f"Report: {report_file}")
    print(f"Summary: {summary_file}")
    print(f"Charge segments: {summary['n_charge_segments']}")
    print(f"Discharge segments: {summary['n_discharge_segments']}")


if __name__ == "__main__":
    main()
