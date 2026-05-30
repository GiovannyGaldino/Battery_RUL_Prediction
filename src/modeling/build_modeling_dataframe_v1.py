#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_modeling_dataframe_v1.py

Build modeling-ready dataframes for lithium-ion battery RUL models from:
  1) hierarchical feature-extraction v3 outputs;
  2) cleaned cycle files, used only to add systematic V(Q) samples.

Main decisions
--------------
- Keeps metadata/traceability columns in the saved dataframes.
- Keeps target columns in the saved dataframes, but writes a separate list of
  model input columns excluding targets, identifiers, and leakage-prone fields.
- Removes redundant feature columns by name rules and, optionally, by sampled
  Spearman correlation.
- Adds only V(Q) samples, using Q = capacity_ah as the monotonic independent
  variable. It does not generate E(Q), V(E), or Q(V) features.
- Writes outputs to a modeling-ready folder with per-dataset/phase pieces and
  optional merged parquet files.

Default paths match the project layout used in this conversation.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import fastparquet  # type: ignore  # noqa: F401
except Exception:
    fastparquet = None  # type: ignore

try:
    from scipy.interpolate import PchipInterpolator  # type: ignore
except Exception:
    PchipInterpolator = None  # type: ignore


DEFAULT_INTERIM_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/interim")
DEFAULT_FEATURES_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/processed/features_hierarchical_v3")
DEFAULT_OUTPUT_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/processed/modeling_dataset_v1")
DEFAULT_REPORTS_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/reports/modeling_dataset_v1")

PHASES = ("charge", "discharge")

DATASET_ALIASES = {
    "ORA": "OXFORD",
    "OXFORD": "OXFORD",
    "RW": "NRW",
    "NRW": "NRW",
    "NASA_RANDOM_WALK": "NRW",
    "NASA-RANDOM-WALK": "NRW",
    "NASA_RANDOMWALK": "NRW",
}

# Metadata/traceability columns are retained in the final dataframes, but never
# listed as model input features.
TRACEABILITY_COLUMNS = [
    "dataset", "battery_id", "battery_index", "cell_id_original",
    "phase", "phase_folder", "feature_level",
    "cycle_number", "cycle_index", "cycle_pair_id",
    "source_cycle_id", "source_file", "source_path", "source_file_key",
    "source_parquet_file", "source_row_id",
    "stage_index", "stage_total", "stage_start_point_index", "stage_end_point_index",
]

# Optional metadata/protocol columns. These are retained for traceability/domain
# analysis, but excluded from model input columns by default.
DOMAIN_METADATA_COLUMNS = [
    "input_barcode", "input_chemistry", "input_form_factor", "input_manufacturer",
    "input_charge_protocol", "input_discharge_protocol", "input_policy_readable",
    "input_nominal_capacity_ah", "input_nominal_voltage_v", "input_ambient_temperature_c",
    "input_c_rate_charge", "input_c_rate_discharge", "input_dod",
    "input_has_voltage", "input_has_current", "input_has_temperature",
    "input_has_capacity", "input_has_energy",
]

QUALITY_COLUMNS = [
    "input_is_valid_cycle", "input_is_model_usable", "input_exclude_reason",
    "input_cleaning_warnings", "input_n_removed_points", "input_trim_start_index",
    "input_trim_end_index", "capacity_validation_status", "capacity_validation_notes",
    "target_capacity_valid",
]

# Keep targets in dataframe, but not as model input features.
TARGET_PATTERNS = [
    r"^target_", r"^input_remaining_cycles$", r"^input_remaining_cycles_norm$",
    r"^remaining_cycles$", r"^remaining_cycles_norm$", r"^input_cycle_life$", r"^cycle_life$",
    r"^input_soh$", r"^soh$",
]

PRIMARY_TARGET = "target_rul80_steps_clipped"
DEFAULT_TARGET_COLUMNS_PRIORITY = [
    "target_rul80_steps_clipped",
    "target_rul80_cycle_index_clipped",
    "target_rul80_steps_signed",
    "target_rul80_cycle_index_signed",
    "target_rul70_steps_clipped",
    "target_rul70_cycle_index_clipped",
    "target_capacity_ah",
    "target_capacity_ratio_to_initial",
    "target_initial_capacity_ah",
    "target_soh_recomputed",
    "target_soh_recomputed_clipped_0_1p2",
    "target_eol80_found",
    "target_eol80_order_index",
    "target_eol80_cycle_index",
    "target_after_eol80",
    "input_remaining_cycles",
    "input_remaining_cycles_norm",
    "input_soh",
]

# Prefer recalculated vector-aware columns over duplicated input_* scalar columns.
RULE_REDUNDANT_MAP = {
    "input_voltage_min_v": "voltage_min",
    "input_voltage_max_v": "voltage_max",
    "input_voltage_start_v": "voltage_first",
    "input_voltage_end_v": "voltage_last",
    "input_current_min_a": "current_min",
    "input_current_max_a": "current_max",
    "input_current_mean_a": "current_mean",
    "input_temperature_min_c": "temperature_min",
    "input_temperature_max_c": "temperature_max",
    "input_temperature_mean_c": "temperature_mean",
    "input_phase_capacity_ah": "capacityh_last",
    "input_phase_energy_wh": "energy_last",
    "input_duration_s": "time_vector_duration_s",
    "input_sample_period_median_s": "time_dt_median_s",
    "input_n_points": "time_n_valid",
    "time_min": "time_first",
    "time_max": "time_last",
    "time_range": "time_delta",
    "capacityh_max": "capacityh_last",
    "capacityh_range": "capacityh_delta",
    "integral_energy_abs_wh_from_vi": "integral_power_abs_wh",
    "integral_energy_signed_wh_from_vi": "integral_power_signed_wh",
}

# Columns that are typically not physical predictors, or likely leakage/domain ID.
# They are retained if present as metadata/targets, but excluded from model_input_columns.
MODEL_EXCLUDE_EXACT = set(TRACEABILITY_COLUMNS + DOMAIN_METADATA_COLUMNS + QUALITY_COLUMNS + [
    "target_capacity_source", "target_initial_capacity_method",
    "target_eol70_found", "target_eol80_found", "target_after_eol70", "target_after_eol80",
])

MODEL_EXCLUDE_REGEX = [
    r"^target_",
    r"^input_remaining_cycles",
    r"^remaining_cycles",
    r"^input_cycle_life$",
    r"^cycle_life$",
    r"^source_",
    r".*_id$",
    r".*_notes$",
    r".*_status$",
    r".*_protocol$",
    r".*_barcode$",
]


@dataclass
class FeatureFile:
    dataset: str
    phase: str
    path: Path


@dataclass
class CleanCycleFile:
    dataset: str
    phase: str
    path: Path
    layout: str


# =============================================================================
# Generic helpers
# =============================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def canonicalize_dataset_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "UNKNOWN"
    name = str(value).strip().upper()
    return DATASET_ALIASES.get(name, name)


def safe_filename(value: Any, max_len: int = 160) -> str:
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value)).strip("_")
    return (text or "value")[:max_len]


def normalize_phase(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"charge", "charging"}:
        return "charge"
    if text in {"discharge", "discharging"}:
        return "discharge"
    return text


def parse_list_argument(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return list(dict.fromkeys(out))


def parse_dataset_args(values: Optional[Sequence[str]]) -> List[str]:
    datasets = parse_list_argument(values)
    if not datasets or any(d.lower() == "all" for d in datasets):
        return []
    return [canonicalize_dataset_name(d) for d in datasets]


def parse_phase_args(values: Optional[Sequence[str]]) -> List[str]:
    phases = parse_list_argument(values)
    if not phases or any(p.lower() == "all" for p in phases):
        return list(PHASES)
    out = [normalize_phase(p) for p in phases if normalize_phase(p) in PHASES]
    return list(dict.fromkeys(out)) or list(PHASES)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_parquet_columns(path: Path, engine: str) -> List[str]:
    if engine == "fastparquet" and fastparquet is not None:
        try:
            return list(fastparquet.ParquetFile(str(path)).columns)
        except Exception:
            pass
    try:
        return list(pd.read_parquet(path, engine=engine).columns)
    except Exception:
        return []


def parquet_row_count(path: Path, engine: str) -> Optional[int]:
    if engine == "fastparquet" and fastparquet is not None:
        try:
            pf = fastparquet.ParquetFile(str(path))
            return int(sum(rg.num_rows for rg in pf.fmd.row_groups))
        except Exception:
            pass
    try:
        return int(len(pd.read_parquet(path, columns=[], engine=engine)))
    except Exception:
        pass
    try:
        return int(len(pd.read_parquet(path, engine=engine)))
    except Exception:
        return None


def robust_read_parquet(path: Path, columns: Optional[Sequence[str]], engine: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=list(columns) if columns else None, engine=engine)
    except Exception:
        if not columns:
            raise
        frames: List[pd.DataFrame] = []
        failed: List[str] = []
        for col in columns:
            try:
                frames.append(pd.read_parquet(path, columns=[col], engine=engine))
            except Exception:
                failed.append(col)
        if not frames:
            raise RuntimeError(f"Could not read any requested columns from {path}. Failed={failed[:20]}")
        df = pd.concat(frames, axis=1)
        if failed:
            missing = pd.DataFrame({col: np.nan for col in failed}, index=df.index)
            df = pd.concat([df, missing], axis=1)
        return df.copy()


def to_numeric_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def is_target_column(col: str) -> bool:
    return any(re.match(pat, col) for pat in TARGET_PATTERNS)


def is_excluded_from_model(col: str) -> bool:
    if col in MODEL_EXCLUDE_EXACT:
        return True
    return any(re.match(pat, col) for pat in MODEL_EXCLUDE_REGEX)


def is_numeric_like(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
        return True
    s = pd.to_numeric(df[col].head(1000), errors="coerce")
    return bool(s.notna().sum() >= max(3, int(0.1 * min(len(s), 1000))))


# =============================================================================
# Vector parsing and V(Q)
# =============================================================================

def parse_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, np.ndarray):
        arr = value.ravel()
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value).ravel()
    elif isinstance(value, pd.Series):
        arr = value.to_numpy().ravel()
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return np.array([], dtype=float)
        try:
            arr = np.asarray(ast.literal_eval(text)).ravel()
        except Exception:
            text = text.strip("[]()")
            parts = re.split(r"[;,\s]+", text)
            vals: List[float] = []
            for part in parts:
                if not part:
                    continue
                try:
                    vals.append(float(part))
                except Exception:
                    vals.append(np.nan)
            arr = np.asarray(vals).ravel()
    else:
        try:
            arr = np.asarray(value).ravel()
        except Exception:
            return np.array([], dtype=float)
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)


def align_two(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if n <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return a[:n].astype(float, copy=False), b[:n].astype(float, copy=False)


def monotonic_q_voltage(q: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q, v = align_two(q, v)
    if len(q) == 0:
        return q, v
    valid = np.isfinite(q) & np.isfinite(v)
    q = q[valid]
    v = v[valid]
    if len(q) < 2:
        return q, v
    order = np.argsort(q, kind="mergesort")
    q = q[order]
    v = v[order]
    # Collapse duplicated Q values using median voltage. This makes Q strictly monotonic.
    temp = pd.DataFrame({"q": q, "v": v})
    grouped = temp.groupby("q", sort=True, as_index=False)["v"].median()
    return grouped["q"].to_numpy(dtype=float), grouped["v"].to_numpy(dtype=float)


def interp_vq(q: np.ndarray, v: np.ndarray, p: int, interpolation: str = "pchip") -> Tuple[np.ndarray, np.ndarray, bool, str]:
    q_m, v_m = monotonic_q_voltage(q, v)
    if len(q_m) < 2:
        return np.full(p, np.nan), np.full(p, np.nan), False, "less_than_two_unique_q"
    q_range = float(np.nanmax(q_m) - np.nanmin(q_m))
    v_range = float(np.nanmax(v_m) - np.nanmin(v_m))
    if not np.isfinite(q_range) or q_range <= 0:
        return np.full(p, np.nan), np.full(p, np.nan), False, "invalid_q_range"
    grid = np.linspace(float(np.nanmin(q_m)), float(np.nanmax(q_m)), p)
    try:
        if interpolation == "pchip" and PchipInterpolator is not None and len(q_m) >= 3:
            y = PchipInterpolator(q_m, v_m, extrapolate=False)(grid)
        else:
            y = np.interp(grid, q_m, v_m)
        y = np.asarray(y, dtype=float)
    except Exception as exc:
        return grid, np.full(p, np.nan), False, f"interpolation_error:{exc!r}"
    return grid, y, bool(np.isfinite(y).sum() >= max(2, int(math.ceil(0.8 * p)))), "ok"


def count_inflections(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    y = y[valid]
    if len(y) < 4:
        return np.nan
    d1 = np.diff(y)
    d2 = np.diff(d1)
    signs = np.sign(d2)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0
    return float(np.sum(signs[1:] != signs[:-1]))


def vq_features_from_row(row: pd.Series, resolutions: Sequence[int], interpolation: str) -> Dict[str, Any]:
    q = parse_vector(row.get("capacity_ah"))
    v = parse_vector(row.get("voltage_v"))
    q_m, v_m = monotonic_q_voltage(q, v)
    base: Dict[str, Any] = {
        "VQ_n_raw_points": int(min(len(q), len(v))),
        "VQ_n_unique_q": int(len(q_m)),
        "VQ_q_min": float(np.nanmin(q_m)) if len(q_m) else np.nan,
        "VQ_q_max": float(np.nanmax(q_m)) if len(q_m) else np.nan,
        "VQ_q_range": float(np.nanmax(q_m) - np.nanmin(q_m)) if len(q_m) else np.nan,
        "VQ_voltage_min": float(np.nanmin(v_m)) if len(v_m) else np.nan,
        "VQ_voltage_max": float(np.nanmax(v_m)) if len(v_m) else np.nan,
        "VQ_voltage_range": float(np.nanmax(v_m) - np.nanmin(v_m)) if len(v_m) else np.nan,
    }
    for p in resolutions:
        grid, y, valid, reason = interp_vq(q_m, v_m, p, interpolation=interpolation)
        base[f"VQ_valid_p{p:03d}"] = int(valid)
        base[f"VQ_invalid_reason_p{p:03d}"] = "" if valid else reason
        for i, val in enumerate(y, start=1):
            base[f"VQ_p{p:03d}_{i:03d}"] = float(val) if np.isfinite(val) else np.nan
        if valid:
            try:
                base[f"VQ_area_p{p:03d}"] = float(np.trapz(y, grid))
            except Exception:
                base[f"VQ_area_p{p:03d}"] = np.nan
            dy = np.diff(y)
            dq = np.diff(grid)
            slopes = np.divide(dy, dq, out=np.full_like(dy, np.nan, dtype=float), where=dq != 0)
            base[f"VQ_slope_mean_p{p:03d}"] = float(np.nanmean(slopes)) if np.isfinite(slopes).any() else np.nan
            base[f"VQ_slope_std_p{p:03d}"] = float(np.nanstd(slopes)) if np.isfinite(slopes).any() else np.nan
            base[f"VQ_slope_min_p{p:03d}"] = float(np.nanmin(slopes)) if np.isfinite(slopes).any() else np.nan
            base[f"VQ_slope_max_p{p:03d}"] = float(np.nanmax(slopes)) if np.isfinite(slopes).any() else np.nan
            base[f"VQ_slope_max_abs_p{p:03d}"] = float(np.nanmax(np.abs(slopes))) if np.isfinite(slopes).any() else np.nan
            base[f"VQ_inflection_count_p{p:03d}"] = count_inflections(y)
        else:
            for suffix in ["area", "slope_mean", "slope_std", "slope_min", "slope_max", "slope_max_abs", "inflection_count"]:
                base[f"VQ_{suffix}_p{p:03d}"] = np.nan
    return base


# =============================================================================
# Discovery
# =============================================================================

def infer_dataset_from_filename(path: Path) -> str:
    name = path.name.lower()
    for token in ["calce", "hust", "matr", "nrw", "ora", "oxford", "pcoe", "rwth", "xjtu"]:
        if token in name:
            return canonicalize_dataset_name(token)
    for part in path.parts:
        low = part.lower()
        if low in {"calce", "hust", "matr", "nrw", "ora", "oxford", "pcoe", "rwth", "xjtu"}:
            return canonicalize_dataset_name(low)
    return "UNKNOWN"


def infer_phase_from_path(path: Path) -> str:
    text = str(path).lower()
    if "discharge" in text:
        return "discharge"
    if "charge" in text:
        return "charge"
    return "unknown"


def discover_feature_files(root: Path, datasets: Sequence[str], phases: Sequence[str]) -> List[FeatureFile]:
    wanted = {d.upper() for d in datasets if d}
    phase_set = set(phases)
    files: List[FeatureFile] = []
    if not root.exists():
        return files
    for path in sorted(root.rglob("*.parquet"), key=lambda p: str(p).lower()):
        lower = str(path).replace("\\", "/").lower()
        if "/_targets/" in lower or ".tmp" in path.name.lower() or path.name.startswith("."):
            continue
        phase = infer_phase_from_path(path)
        if phase not in phase_set:
            continue
        ds = infer_dataset_from_filename(path)
        # Prefer folder after phase when present: root/charge/CALCE/file.parquet
        parts = [p.lower() for p in path.parts]
        if phase in parts:
            idx = parts.index(phase)
            if idx + 1 < len(parts):
                cand = path.parts[idx + 1]
                if cand.lower() not in {"_targets", "merged", "features"}:
                    ds = canonicalize_dataset_name(cand)
        if wanted and ds.upper() not in wanted:
            continue
        files.append(FeatureFile(dataset=ds, phase=phase, path=path))
    return files


def discover_clean_cycle_files(interim_root: Path, datasets: Sequence[str], phases: Sequence[str]) -> List[CleanCycleFile]:
    wanted = {d.upper() for d in datasets if d}
    phase_set = set(phases)
    files: List[CleanCycleFile] = []
    seen: set[str] = set()

    # Prefer global phase folders used by extraction v3.
    for phase in phases:
        folder = interim_root / phase
        if folder.exists():
            for path in sorted(folder.glob("*.parquet"), key=lambda p: str(p).lower()):
                ds = infer_dataset_from_filename(path)
                if wanted and ds.upper() not in wanted:
                    continue
                key = str(path.resolve()).lower()
                if key not in seen:
                    files.append(CleanCycleFile(ds, phase, path, "global_phase_folder"))
                    seen.add(key)

    # Fallback / supplement: per-dataset clean folders.
    for clean_dir in sorted(interim_root.glob("*/clean"), key=lambda p: str(p).lower()):
        ds_default = canonicalize_dataset_name(clean_dir.parent.name)
        for path in sorted(clean_dir.glob("*.parquet"), key=lambda p: str(p).lower()):
            phase = infer_phase_from_path(path)
            if phase not in phase_set:
                continue
            ds = infer_dataset_from_filename(path)
            if ds == "UNKNOWN":
                ds = ds_default
            if wanted and ds.upper() not in wanted:
                continue
            key = str(path.resolve()).lower()
            if key not in seen:
                files.append(CleanCycleFile(ds, phase, path, "dataset_clean_folder"))
                seen.add(key)
    return files


# =============================================================================
# V(Q) extraction
# =============================================================================

def clean_key_columns(df: pd.DataFrame, dataset: str, phase: str) -> pd.DataFrame:
    out = df.copy()
    if "dataset" not in out.columns:
        out["dataset"] = dataset
    if "phase" not in out.columns:
        out["phase"] = phase
    out["dataset"] = out["dataset"].map(canonicalize_dataset_name)
    out["phase"] = out["phase"].map(normalize_phase)
    if "battery_id" in out.columns:
        out["battery_id"] = out["battery_id"].astype(str)
    return out


def key_columns_available(df: pd.DataFrame) -> List[str]:
    candidates = ["dataset", "phase", "battery_id", "cycle_number", "cycle_index"]
    return [c for c in candidates if c in df.columns]


def build_vq_feature_file(item: CleanCycleFile, args: argparse.Namespace) -> Optional[Path]:
    out_dir = args.output_root / "_curve_features" / item.phase / safe_filename(item.dataset)
    ensure_dir(out_dir)
    out_path = out_dir / f"{safe_filename(item.dataset)}_{item.phase}_VQ_features.parquet"
    if out_path.exists() and args.skip_existing and not args.force:
        return out_path

    columns_needed = [
        "dataset", "phase", "battery_id", "cycle_number", "cycle_index", "cycle_pair_id",
        "source_cycle_id", "source_file", "source_path", "capacity_ah", "voltage_v",
    ]
    available = read_parquet_columns(item.path, args.engine)
    cols = [c for c in columns_needed if c in available]
    if "capacity_ah" not in cols or "voltage_v" not in cols or "battery_id" not in cols:
        return None
    df = robust_read_parquet(item.path, cols, args.engine)
    df = clean_key_columns(df, item.dataset, item.phase)
    if args.max_rows_per_file and args.max_rows_per_file > 0:
        df = df.head(args.max_rows_per_file).copy()

    records: List[Dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"V(Q) {item.dataset}/{item.phase}", leave=False, unit="row"):
        rec: Dict[str, Any] = {}
        for c in ["dataset", "phase", "battery_id", "cycle_number", "cycle_index", "cycle_pair_id", "source_cycle_id", "source_file", "source_path"]:
            if c in df.columns:
                rec[c] = row.get(c)
        rec.update(vq_features_from_row(row, args.resolutions, args.interpolation))
        records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        return None
    out.to_parquet(out_path, engine=args.engine, index=False)
    return out_path


def build_all_vq_features(clean_files: Sequence[CleanCycleFile], args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    for item in tqdm(clean_files, desc="Build V(Q) feature files", unit="file"):
        try:
            p = build_vq_feature_file(item, args)
            if p is not None:
                paths.append(p)
        except Exception as exc:
            print(f"[WARNING] V(Q) extraction failed for {item.path}: {exc}", file=sys.stderr)
            traceback.print_exc(limit=2)
    return paths


def load_vq_for_dataset_phase(dataset: str, phase: str, args: argparse.Namespace) -> pd.DataFrame:
    path = args.output_root / "_curve_features" / phase / safe_filename(dataset) / f"{safe_filename(dataset)}_{phase}_VQ_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, engine=args.engine)


# =============================================================================
# Column selection and redundancy
# =============================================================================

def collect_schema(files: Sequence[FeatureFile], engine: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in tqdm(files, desc="Collect feature schema", unit="file"):
        cols = read_parquet_columns(item.path, engine)
        for i, c in enumerate(cols, start=1):
            rows.append({"dataset": item.dataset, "phase": item.phase, "file": str(item.path), "file_name": item.path.name, "column_order": i, "column_name": c})
    return pd.DataFrame(rows)


def collect_vq_schema(vq_paths: Sequence[Path], engine: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in vq_paths:
        cols = read_parquet_columns(path, engine)
        phase = infer_phase_from_path(path)
        ds = infer_dataset_from_filename(path)
        for i, c in enumerate(cols, start=1):
            rows.append({"dataset": ds, "phase": phase, "file": str(path), "file_name": path.name, "column_order": i, "column_name": c})
    return pd.DataFrame(rows)


def split_column_roles(all_columns: Sequence[str]) -> Dict[str, List[str]]:
    cols = list(dict.fromkeys(all_columns))
    trace = [c for c in TRACEABILITY_COLUMNS if c in cols]
    domain = [c for c in DOMAIN_METADATA_COLUMNS if c in cols]
    quality = [c for c in QUALITY_COLUMNS if c in cols]
    targets = [c for c in DEFAULT_TARGET_COLUMNS_PRIORITY if c in cols]
    targets += [c for c in cols if c not in targets and is_target_column(c)]
    targets = list(dict.fromkeys(targets))
    metadata = list(dict.fromkeys(trace + domain + quality))
    return {"metadata": metadata, "targets": targets}


def rule_based_redundant_columns(columns: Sequence[str]) -> pd.DataFrame:
    colset = set(columns)
    rows = []
    for drop, keep in RULE_REDUNDANT_MAP.items():
        if drop in colset and keep in colset:
            rows.append({"dropped_column": drop, "kept_column": keep, "reason": "rule_based_duplicate_or_near_duplicate"})
    return pd.DataFrame(rows)


def candidate_feature_columns(columns: Sequence[str], redundant_drop: Sequence[str]) -> List[str]:
    drop_set = set(redundant_drop)
    candidates = []
    for c in columns:
        if c in drop_set:
            continue
        if c in MODEL_EXCLUDE_EXACT:
            continue
        if is_target_column(c):
            continue
        if is_excluded_from_model(c):
            continue
        # Avoid raw non-numeric/text descriptors as features.
        if c.startswith("input_") and c in DOMAIN_METADATA_COLUMNS:
            continue
        candidates.append(c)
    return list(dict.fromkeys(candidates))


def sample_analysis_frame(feature_files: Sequence[FeatureFile], vq_paths: Sequence[Path], args: argparse.Namespace, candidate_cols: Sequence[str]) -> pd.DataFrame:
    if args.max_correlation_rows <= 0:
        return pd.DataFrame()
    per_file = max(1, args.max_correlation_rows // max(1, len(feature_files)))
    frames: List[pd.DataFrame] = []
    candidate_set = set(candidate_cols)

    for item in tqdm(feature_files, desc="Build redundancy sample", unit="file"):
        cols = read_parquet_columns(item.path, args.engine)
        base_keep = [c for c in ["dataset", "phase", "battery_id", "cycle_number", "cycle_index", PRIMARY_TARGET] if c in cols]
        feat_keep = [c for c in cols if c in candidate_set]
        read_cols = list(dict.fromkeys(base_keep + feat_keep))
        if not read_cols:
            continue
        try:
            df = robust_read_parquet(item.path, read_cols, args.engine)
            df = clean_key_columns(df, item.dataset, item.phase)
            vq = load_vq_for_dataset_phase(item.dataset, item.phase, args)
            if not vq.empty:
                merge_keys = [k for k in ["dataset", "phase", "battery_id", "cycle_number", "cycle_index"] if k in df.columns and k in vq.columns]
                vq_cols = merge_keys + [c for c in vq.columns if c in candidate_set]
                if merge_keys and len(vq_cols) > len(merge_keys):
                    df = df.merge(vq[vq_cols], on=merge_keys, how="left", validate="many_to_one")
            if len(df) > per_file:
                df = df.sample(n=per_file, random_state=args.random_seed)
            frames.append(df)
        except Exception as exc:
            print(f"[WARNING] Sampling failed for {item.path}: {exc}", file=sys.stderr)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).copy()


def compute_corr_redundancy(sample: pd.DataFrame, candidate_cols: Sequence[str], args: argparse.Namespace) -> pd.DataFrame:
    if sample.empty or not args.corr_redundancy:
        return pd.DataFrame()
    cols = [c for c in candidate_cols if c in sample.columns]
    # Keep numeric and sufficiently available columns only.
    numeric_cols: List[str] = []
    stats_rows: List[Tuple[str, float, int, float]] = []
    target = sample[PRIMARY_TARGET] if PRIMARY_TARGET in sample.columns else None
    for c in cols:
        s = to_numeric_float(sample[c])
        finite = float(s.notna().mean())
        nunique = int(s.dropna().nunique())
        if finite >= args.min_finite_for_corr and nunique > 1:
            numeric_cols.append(c)
            target_corr = 0.0
            if target is not None:
                t = to_numeric_float(target)
                valid = s.notna() & t.notna()
                if valid.sum() >= 20 and s[valid].nunique() > 1 and t[valid].nunique() > 1:
                    val = s[valid].corr(t[valid], method="spearman")
                    target_corr = abs(float(val)) if pd.notna(val) else 0.0
            stats_rows.append((c, finite, nunique, target_corr))
    if len(numeric_cols) < 2:
        return pd.DataFrame()

    # Cap columns for correlation if needed, prioritizing target association and completeness.
    if args.max_corr_features and len(numeric_cols) > args.max_corr_features:
        ranked = sorted(stats_rows, key=lambda x: (x[3], x[1], x[2]), reverse=True)[: args.max_corr_features]
        numeric_cols = [x[0] for x in ranked]
        stats_rows = ranked

    stat = {c: {"finite": f, "nunique": n, "target_corr": tc} for c, f, n, tc in stats_rows}
    num = pd.DataFrame({c: to_numeric_float(sample[c]) for c in numeric_cols})
    corr = num.corr(method="spearman", min_periods=50)

    rows: List[Dict[str, Any]] = []
    dropped: set[str] = set()
    for i, a in enumerate(numeric_cols):
        if a in dropped:
            continue
        for b in numeric_cols[i + 1:]:
            if b in dropped:
                continue
            val = corr.loc[a, b]
            if pd.isna(val) or abs(float(val)) < args.corr_redundancy_threshold:
                continue
            # Keep the column with better target association, then completeness, then shorter name.
            sa = stat.get(a, {"finite": 0, "target_corr": 0})
            sb = stat.get(b, {"finite": 0, "target_corr": 0})
            score_a = (sa["target_corr"], sa["finite"], -len(a))
            score_b = (sb["target_corr"], sb["finite"], -len(b))
            keep, drop = (a, b) if score_a >= score_b else (b, a)
            rows.append({
                "dropped_column": drop,
                "kept_column": keep,
                "spearman_corr": float(val),
                "abs_spearman_corr": abs(float(val)),
                "drop_target_corr": stat.get(drop, {}).get("target_corr", np.nan),
                "keep_target_corr": stat.get(keep, {}).get("target_corr", np.nan),
                "reason": "sampled_spearman_redundancy",
            })
            dropped.add(drop)
    return pd.DataFrame(rows)


# =============================================================================
# Modeling dataframe building
# =============================================================================

def prepare_feature_df(item: FeatureFile, args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(item.path, engine=args.engine)
    df = clean_key_columns(df, item.dataset, item.phase)
    return df



def prepare_feature_df_selected_safe(
    item: FeatureFile,
    args: argparse.Namespace,
    metadata_cols: Sequence[str],
    target_cols: Sequence[str],
    selected_features: Sequence[str],
) -> pd.DataFrame:
    """
    Read only the columns needed for the modeling dataframe.

    This avoids fastparquet failures caused by unused columns containing pandas.NA
    in integer-like fields. If a requested column is unreadable, it is skipped and
    recorded in skipped_unreadable_feature_columns.csv.
    """
    required_key_cols = [
        "dataset", "phase", "battery_id", "cycle_number", "cycle_index",
        "feature_level", "stage_index", "stage_total",
        "stage_start_point_index", "stage_end_point_index",
    ]

    wanted_cols = list(dict.fromkeys(
        list(metadata_cols) +
        list(target_cols) +
        list(selected_features) +
        required_key_cols
    ))

    try:
        import fastparquet
        available_cols = list(fastparquet.ParquetFile(str(item.path)).columns)
    except Exception:
        available_cols = None

    if available_cols is None:
        df = pd.read_parquet(item.path, engine=args.engine)
        df = clean_key_columns(df, item.dataset, item.phase)
        return df

    read_cols = [c for c in wanted_cols if c in available_cols]

    if not read_cols:
        df = pd.read_parquet(item.path, engine=args.engine)
        df = clean_key_columns(df, item.dataset, item.phase)
        return df

    try:
        df = pd.read_parquet(item.path, engine=args.engine, columns=read_cols)
        df = clean_key_columns(df, item.dataset, item.phase)
        return df

    except Exception as first_exc:
        print()
        print(f"[WARNING] Direct selective read failed for {item.dataset}/{item.phase}: {item.path.name}")
        print(f"          {repr(first_exc)}")
        print("          Scanning columns to skip unreadable fields...")

        good_cols = []
        bad_rows = []

        for col in tqdm(read_cols, desc=f"Column scan {item.dataset}/{item.phase}", leave=False, unit="col"):
            try:
                pd.read_parquet(item.path, engine=args.engine, columns=[col])
                good_cols.append(col)
            except Exception as exc:
                bad_rows.append({
                    "dataset": item.dataset,
                    "phase": item.phase,
                    "source_file": str(item.path),
                    "column_name": col,
                    "error": repr(exc),
                })

        if not good_cols:
            raise first_exc

        df = pd.read_parquet(item.path, engine=args.engine, columns=good_cols)
        df = clean_key_columns(df, item.dataset, item.phase)

        if bad_rows:
            ensure_dir(args.reports_root)
            bad_path = args.reports_root / "skipped_unreadable_feature_columns.csv"
            bad_df = pd.DataFrame(bad_rows)

            if bad_path.exists():
                old = pd.read_csv(bad_path)
                bad_df = pd.concat([old, bad_df], ignore_index=True)

            bad_df = bad_df.drop_duplicates(
                subset=["dataset", "phase", "source_file", "column_name"],
                keep="last"
            )
            bad_df.to_csv(bad_path, index=False, encoding="utf-8-sig")

            print(f"[WARNING] Skipped {len(bad_rows)} unreadable columns for {item.dataset}/{item.phase}.")
            print(f"          Details saved to: {bad_path}")

        return df


def merge_vq_features(df: pd.DataFrame, dataset: str, phase: str, args: argparse.Namespace, selected_vq_cols: Sequence[str]) -> pd.DataFrame:
    vq = load_vq_for_dataset_phase(dataset, phase, args)
    if vq.empty:
        return df
    vq = clean_key_columns(vq, dataset, phase)
    merge_keys = [k for k in ["dataset", "phase", "battery_id", "cycle_number", "cycle_index"] if k in df.columns and k in vq.columns]
    if not merge_keys:
        return df
    cols = list(dict.fromkeys(merge_keys + [c for c in selected_vq_cols if c in vq.columns]))
    if len(cols) == len(merge_keys):
        return df
    return df.merge(vq[cols], on=merge_keys, how="left", validate="many_to_one")


def ordered_columns(df: pd.DataFrame, metadata_cols: Sequence[str], target_cols: Sequence[str], feature_cols: Sequence[str]) -> List[str]:
    meta = [c for c in metadata_cols if c in df.columns]
    targets = [c for c in target_cols if c in df.columns and c not in meta]
    feats = [c for c in feature_cols if c in df.columns and c not in meta and c not in targets]
    rest = [c for c in df.columns if c not in set(meta + targets + feats)]
    return meta + targets + feats + rest


def write_piece(df: pd.DataFrame, out_path: Path, args: argparse.Namespace) -> None:
    ensure_dir(out_path.parent)
    df.to_parquet(out_path, engine=args.engine, index=False)


def build_modeling_pieces(feature_files: Sequence[FeatureFile], args: argparse.Namespace, metadata_cols: Sequence[str], target_cols: Sequence[str], selected_features: Sequence[str]) -> Dict[str, Any]:
    piece_rows: List[Dict[str, Any]] = []
    selected_vq_cols = [c for c in selected_features if c.startswith("VQ_")]

    for item in tqdm(feature_files, desc="Write modeling pieces", unit="file"):
        try:
            df = prepare_feature_df_selected_safe(item, args, metadata_cols, target_cols, selected_features)
            df = merge_vq_features(df, item.dataset, item.phase, args, selected_vq_cols)
            # Drop redundant/unselected feature columns, but retain metadata/targets.
            keep_cols = ordered_columns(df, metadata_cols, target_cols, selected_features)
            df = df[keep_cols].copy()

            for level_name, level_df in split_by_level(df).items():
                if level_df.empty:
                    continue
                out_dir = args.output_root / "pieces" / level_name / item.phase / safe_filename(item.dataset)
                out_path = out_dir / f"{safe_filename(item.dataset)}_{item.phase}_{level_name}_modeling.parquet"
                write_piece(level_df, out_path, args)
                piece_rows.append({
                    "level": level_name,
                    "dataset": item.dataset,
                    "phase": item.phase,
                    "rows": len(level_df),
                    "columns": len(level_df.columns),
                    "path": str(out_path),
                })
        except Exception as exc:
            piece_rows.append({
                "level": "ERROR", "dataset": item.dataset, "phase": item.phase,
                "rows": 0, "columns": 0, "path": str(item.path), "error": repr(exc),
            })
            traceback.print_exc(limit=2)
    pieces = pd.DataFrame(piece_rows)
    write_csv(pieces, args.reports_root / "modeling_pieces_inventory.csv")
    return {"pieces": piece_rows}


def split_by_level(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {"all_levels": df}
    if "feature_level" in df.columns:
        fl = df["feature_level"].astype(str).str.lower()
        out["phase"] = df[fl.eq("phase")].copy()
        out["stage"] = df[fl.eq("stage")].copy()
    else:
        out["phase"] = df.copy()
        out["stage"] = df.iloc[[]].copy()
    return out


def merge_pieces(args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for level_name in ["phase", "stage", "all_levels"]:
        piece_paths = sorted((args.output_root / "pieces" / level_name).rglob("*.parquet"))
        if not piece_paths:
            continue
        frames: List[pd.DataFrame] = []
        for p in tqdm(piece_paths, desc=f"Merge {level_name}", unit="piece"):
            frames.append(pd.read_parquet(p, engine=args.engine))
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        out_path = args.output_root / f"modeling_{level_name}.parquet"
        merged.to_parquet(out_path, engine=args.engine, index=False)
        rows.append({"level": level_name, "rows": len(merged), "columns": len(merged.columns), "path": str(out_path)})
        # Also split by charge/discharge for immediate model training.
        if "phase" in merged.columns:
            for phase, g in merged.groupby("phase", dropna=False):
                phase_s = normalize_phase(phase)
                out_phase_path = args.output_root / "by_phase" / level_name / f"modeling_{level_name}_{phase_s}.parquet"
                ensure_dir(out_phase_path.parent)
                g.to_parquet(out_phase_path, engine=args.engine, index=False)
                rows.append({"level": f"{level_name}_{phase_s}", "rows": len(g), "columns": len(g.columns), "path": str(out_phase_path)})
    summary = pd.DataFrame(rows)
    write_csv(summary, args.reports_root / "modeling_merged_files.csv")
    return summary


def build_readme(args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    path = args.output_root / "README_MODELING_DATASET.md"
    lines = [
        "# Modeling Dataset v1",
        "",
        f"Generated at: `{now_iso()}`",
        "",
        "## Contents",
        "",
        "- `modeling_phase.parquet`: rows with `feature_level == phase`; recommended first dataset for ML.",
        "- `modeling_stage.parquet`: rows with `feature_level == stage`; use only for stage-level experiments.",
        "- `modeling_all_levels.parquet`: all available feature levels together.",
        "- `by_phase/`: charge/discharge splits.",
        "- `pieces/`: per-dataset/phase parquet pieces.",
        "- `_curve_features/`: V(Q) features computed from cleaned cycle vectors.",
        "",
        "## Target",
        "",
        f"Primary target: `{PRIMARY_TARGET}`.",
        "",
        "## Model input columns",
        "",
        "Use `reports/model_input_columns.csv` to select predictors. Metadata and target columns are retained in the dataframes for traceability, but are not model inputs.",
        "",
        "## Important methodological decisions",
        "",
        "- V(Q) samples were added using `capacity_ah` as the monotonic independent variable.",
        "- Energy-curve samples were not added because E(Q)/V(E) plots were visually almost equivalent to capacity-based curves.",
        "- Redundant scalar features were removed using rule-based duplicates and optional sampled Spearman redundancy.",
        "- Dataset, battery, cycle, source and stage metadata were preserved for auditability and train/test splitting.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main orchestration
# =============================================================================

def run(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_dir(args.output_root)
    ensure_dir(args.reports_root)

    feature_files = discover_feature_files(args.features_root, args.datasets, args.phases)
    clean_files = discover_clean_cycle_files(args.interim_root, args.datasets, args.phases)
    if not feature_files:
        raise RuntimeError(f"No feature files found under {args.features_root}")
    if not clean_files:
        raise RuntimeError(f"No clean cycle files found under {args.interim_root}; required for V(Q) sampling")

    schema = collect_schema(feature_files, args.engine)
    write_csv(schema, args.reports_root / "source_feature_columns_inventory.csv")

    # Build V(Q) features from cleaned cycle vectors.
    vq_paths = build_all_vq_features(clean_files, args)
    vq_schema = collect_vq_schema(vq_paths, args.engine)
    write_csv(vq_schema, args.reports_root / "vq_feature_columns_inventory.csv")

    all_cols = list(dict.fromkeys(schema["column_name"].tolist() + vq_schema["column_name"].tolist()))
    roles = split_column_roles(all_cols)
    metadata_cols = list(dict.fromkeys(roles["metadata"] + [c for c in DOMAIN_METADATA_COLUMNS + QUALITY_COLUMNS if c in all_cols]))
    target_cols = roles["targets"]

    rule_drop_df = rule_based_redundant_columns(all_cols)
    rule_dropped = rule_drop_df["dropped_column"].tolist() if not rule_drop_df.empty else []
    initial_candidates = candidate_feature_columns(all_cols, rule_dropped)

    sample = sample_analysis_frame(feature_files, vq_paths, args, initial_candidates)
    if args.save_redundancy_sample and not sample.empty:
        sample.head(args.save_redundancy_sample_rows).to_parquet(args.output_root / "redundancy_analysis_sample.parquet", engine=args.engine, index=False)

    corr_drop_df = compute_corr_redundancy(sample, initial_candidates, args)
    corr_dropped = corr_drop_df["dropped_column"].tolist() if not corr_drop_df.empty else []
    dropped_all = set(rule_dropped) | set(corr_dropped)
    selected_features = [c for c in initial_candidates if c not in dropped_all]

    # Preserve VQ validity/quality columns as features/diagnostics if present.
    selected_features = list(dict.fromkeys(selected_features))

    write_csv(rule_drop_df, args.reports_root / "dropped_rule_redundant_columns.csv")
    write_csv(corr_drop_df, args.reports_root / "dropped_corr_redundant_columns.csv")
    all_dropped = pd.concat([rule_drop_df, corr_drop_df], ignore_index=True) if not rule_drop_df.empty or not corr_drop_df.empty else pd.DataFrame(columns=["dropped_column", "kept_column", "reason"])
    write_csv(all_dropped, args.reports_root / "dropped_redundant_columns.csv")

    model_input_df = pd.DataFrame({"model_input_column": selected_features})
    write_csv(model_input_df, args.reports_root / "model_input_columns.csv")
    write_csv(pd.DataFrame({"metadata_column": metadata_cols}), args.reports_root / "metadata_columns_retained.csv")
    write_csv(pd.DataFrame({"target_column": target_cols}), args.reports_root / "target_columns_retained.csv")

    # Build and save modeling pieces.
    build_modeling_pieces(feature_files, args, metadata_cols, target_cols, selected_features)
    merged_summary = pd.DataFrame()
    if args.write_merged:
        merged_summary = merge_pieces(args)

    summary = {
        "generated_at": now_iso(),
        "interim_root": str(args.interim_root),
        "features_root": str(args.features_root),
        "output_root": str(args.output_root),
        "reports_root": str(args.reports_root),
        "feature_files": len(feature_files),
        "clean_cycle_files": len(clean_files),
        "vq_feature_files": len(vq_paths),
        "source_unique_columns": int(schema["column_name"].nunique()) if not schema.empty else 0,
        "vq_unique_columns": int(vq_schema["column_name"].nunique()) if not vq_schema.empty else 0,
        "metadata_columns_retained": len(metadata_cols),
        "target_columns_retained": len(target_cols),
        "initial_candidate_features": len(initial_candidates),
        "rule_dropped_redundant": len(rule_dropped),
        "corr_dropped_redundant": len(corr_dropped),
        "final_model_input_features": len(selected_features),
        "primary_target": PRIMARY_TARGET,
        "resolutions": args.resolutions,
        "write_merged": bool(args.write_merged),
        "merged_outputs": merged_summary.to_dict(orient="records") if not merged_summary.empty else [],
    }
    write_json(summary, args.reports_root / "modeling_dataset_summary.json")
    build_readme(args, summary)
    return summary


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build modeling-ready dataframe with selected non-redundant features and V(Q) samples.")
    p.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    p.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    p.add_argument("--datasets", nargs="*", default=["all"])
    p.add_argument("--phases", nargs="*", default=list(PHASES))
    p.add_argument("--engine", choices=["fastparquet", "pyarrow"], default="fastparquet")
    p.add_argument("--resolutions", nargs="*", type=int, default=[3, 9, 15, 30])
    p.add_argument("--interpolation", choices=["linear", "pchip"], default="pchip")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-rows-per-file", type=int, default=0, help="Debug limit for V(Q) extraction. 0 means all rows.")
    p.add_argument("--max-correlation-rows", type=int, default=200000, help="Rows sampled for redundancy analysis. 0 disables correlation redundancy.")
    p.add_argument("--corr-redundancy", action="store_true", default=True, help="Enable sampled Spearman redundancy removal. Enabled by default.")
    p.add_argument("--no-corr-redundancy", dest="corr_redundancy", action="store_false")
    p.add_argument("--corr-redundancy-threshold", type=float, default=0.995)
    p.add_argument("--min-finite-for-corr", type=float, default=0.50)
    p.add_argument("--max-corr-features", type=int, default=0, help="Optional cap for redundancy-correlation features. 0 means no cap.")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--write-merged", action="store_true", default=True)
    p.add_argument("--no-write-merged", dest="write_merged", action="store_false")
    p.add_argument("--save-redundancy-sample", action="store_true")
    p.add_argument("--save-redundancy-sample-rows", type=int, default=20000)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.datasets = parse_dataset_args(args.datasets)
    args.phases = parse_phase_args(args.phases)
    args.resolutions = sorted(set(int(r) for r in args.resolutions if int(r) > 1))

    if args.engine == "fastparquet" and fastparquet is None:
        print("[WARNING] fastparquet is not importable. Install it or use --engine pyarrow.", file=sys.stderr)

    print("Modeling dataframe builder v1")
    print("- Interim root:", args.interim_root)
    print("- Features root:", args.features_root)
    print("- Output root:", args.output_root)
    print("- Reports root:", args.reports_root)
    print("- Datasets:", args.datasets or "all")
    print("- Phases:", args.phases)
    print("- V(Q) resolutions:", args.resolutions)
    print("- Energy-curve sampling: disabled by design")

    summary = run(args)
    print("\nDone.")
    print("Primary output root:", args.output_root)
    print("Reports root:", args.reports_root)
    print("Final model input features:", summary.get("final_model_input_features"))
    print("Primary target:", summary.get("primary_target"))
    return 0



def _sanitize_model_input_csv_after_run():
    """Remove diagnostic/control columns from model_input_columns.csv after the builder finishes."""
    import sys
    import re
    from pathlib import Path
    import pandas as pd

    def get_arg_value(flag, default=None):
        argv = sys.argv
        for i, item in enumerate(argv):
            if item == flag and i + 1 < len(argv):
                return argv[i + 1]
        return default

    reports_root = get_arg_value(
        "--reports-root",
        r"C:\Artigo Baterias de Litio\Batteries\reports\modeling_dataset_v1"
    )

    reports_root = Path(reports_root)

    candidate_files = list(reports_root.glob("*model_input*columns*.csv"))
    candidate_files += list(reports_root.rglob("*model_input*columns*.csv"))

    candidate_files = sorted(set(candidate_files))

    if not candidate_files:
        print("[WARNING] No model_input_columns CSV found for post-run sanitization.")
        return

    exclude_exact = {
        "VQ_n_raw_points",
        "VQ_n_unique_q",
        "VQ_q_min",
        "VQ_q_max",
        "VQ_q_range",
        "VQ_voltage_min",
        "VQ_voltage_max",
        "VQ_voltage_range",
    }

    exclude_regex = [
        r"^VQ_valid_p\d+$",
        r"^VQ_invalid_reason_p\d+$",
        r".*invalid_reason.*",
        r".*_reason$",
    ]

    def should_remove(col):
        col = str(col)
        if col in exclude_exact:
            return True
        return any(re.match(pattern, col) for pattern in exclude_regex)

    for csv_path in candidate_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARNING] Could not read {csv_path}: {exc}")
            continue

        if df.empty:
            continue

        col_name = "model_input_column" if "model_input_column" in df.columns else df.columns[0]

        before = len(df)
        mask_remove = df[col_name].astype(str).map(should_remove)
        removed = df.loc[mask_remove, col_name].astype(str).tolist()
        df2 = df.loc[~mask_remove].copy()
        after = len(df2)

        df2.to_csv(csv_path, index=False, encoding="utf-8-sig")

        removed_path = csv_path.with_name(csv_path.stem + "_removed_diagnostics.csv")
        pd.DataFrame({"removed_column": removed}).to_csv(removed_path, index=False, encoding="utf-8-sig")

        print(f"[OK] Sanitized {csv_path}")
        print(f"     before: {before}")
        print(f"     after : {after}")
        print(f"     removed diagnostic/control columns: {before - after}")
        print(f"     removed list: {removed_path}")


if __name__ == "__main__":
    _exit_code = main()
    _sanitize_model_input_csv_after_run()
    raise SystemExit(_exit_code)
