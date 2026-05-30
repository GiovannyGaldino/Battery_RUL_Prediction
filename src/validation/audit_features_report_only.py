#!/usr/bin/env python3
"""
audit_features_report_only.py

Report-only audit for hierarchical feature-extraction outputs of lithium-ion
battery RUL project.

This script does NOT generate plots. It reads only the extracted feature tables
and optional extraction summaries, then produces CSV/JSON/Markdown reports for:

- feature inventory;
- NaN/completeness/constant-feature diagnostics;
- physical plausibility checks;
- diagnostic outlier analysis;
- target distribution and imbalance;
- feature-target correlations;
- feature redundancy;
- dataset/phase stability and domain shift;
- Q(V)/V(Q) validity summaries;
- final feature-quality scorecard.

Default paths are aligned with the project structure used in the previous audit.
"""

from __future__ import annotations

import argparse
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
    from sklearn.feature_selection import mutual_info_regression  # type: ignore
except Exception:
    mutual_info_regression = None  # type: ignore


DEFAULT_FEATURES_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/processed/features_hierarchical_v3")
DEFAULT_OUTPUT_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/reports/auditoria")
DEFAULT_OUTPUT_SUBDIR = Path(r"global/feature_report_only")

PHASES = ("charge", "discharge")

DATASET_ALIASES = {
    "ORA": "OXFORD",
    "OXFORD": "OXFORD",
    "RW": "NRW",
    "NRW": "NRW",
    "NASA_RANDOM_WALK": "NRW",
    "NASA-RANDOM-WALK": "NRW",
    "NASA_RANDOMWALK": "NRW",
    "RANDOM_WALK": "NRW",
}

ID_LIKE_COLUMNS = {
    "dataset", "battery_id", "phase", "phase_folder", "source_file", "source_path",
    "source_cycle_id", "cycle_pair_id", "cell_id_original", "barcode", "policy_readable",
    "chemistry", "form_factor", "manufacturer", "charge_protocol", "discharge_protocol",
    "adapter_version", "cleaning_version", "created_at", "source_file_key",
    "source_parquet_file", "feature_level", "capacity_validation_status",
    "capacity_validation_notes", "target_capacity_source", "target_initial_capacity_method",
    "eol_criterion", "input_eol_criterion", "input_cleaning_warnings",
    "input_exclude_reason", "exclude_reason",
}

TARGET_CANDIDATES = [
    "target_rul80_steps_clipped",
    "target_rul80_cycle_index_clipped",
    "target_rul80_steps_signed",
    "target_rul80_cycle_index_signed",
    "target_rul80_steps",
    "target_rul80_cycle_index",
    "remaining_cycles",
    "input_remaining_cycles",
    "target_rul70_steps_clipped",
    "target_rul70_cycle_index_clipped",
    "target_rul70_steps_signed",
    "target_rul70_cycle_index_signed",
    "target_rul70_steps",
    "target_rul70_cycle_index",
]

NORM_TARGET_CANDIDATES = [
    "remaining_cycles_norm",
    "input_remaining_cycles_norm",
    "target_rul80_steps_norm",
    "target_rul80_cycle_index_norm",
    "target_rul70_steps_norm",
    "target_rul70_cycle_index_norm",
]

PHYSICAL_RULES = {
    # voltage
    "voltage_min_v": (0.0, 6.0),
    "voltage_max_v": (0.0, 6.0),
    "voltage_start_v": (0.0, 6.0),
    "voltage_end_v": (0.0, 6.0),
    "voltage_min": (0.0, 6.0),
    "voltage_max": (0.0, 6.0),
    "voltage_first": (0.0, 6.0),
    "voltage_last": (0.0, 6.0),
    "input_voltage_min_v": (0.0, 6.0),
    "input_voltage_max_v": (0.0, 6.0),
    # capacity
    "capacity_min_ah": (-1e-9, 200.0),
    "capacity_max_ah": (-1e-9, 200.0),
    "phase_capacity_ah": (-1e-9, 200.0),
    "capacity_min": (-1e-9, 200.0),
    "capacity_max": (-1e-9, 200.0),
    "capacity_last": (-1e-9, 200.0),
    "capacity_delta": (-200.0, 200.0),
    "target_capacity_ah": (0.0, 200.0),
    "target_initial_capacity_ah": (0.0, 200.0),
    "input_phase_capacity_ah": (-1e-9, 200.0),
    # energy
    "phase_energy_wh": (-1e-9, 2000.0),
    "energy_min_wh": (-1e-9, 2000.0),
    "energy_max_wh": (-1e-9, 2000.0),
    "energy_min": (-1e-9, 2000.0),
    "energy_max": (-1e-9, 2000.0),
    "energy_last": (-1e-9, 2000.0),
    "input_phase_energy_wh": (-1e-9, 2000.0),
    # time and sampling
    "duration_s": (1e-12, 1e8),
    "time_vector_duration_s": (1e-12, 1e8),
    "sample_period_median_s": (1e-12, 1e7),
    "time_dt_median_s": (1e-12, 1e7),
    "input_duration_s": (1e-12, 1e8),
    "input_sample_period_median_s": (1e-12, 1e7),
    "n_points": (20.0, 1e8),
    "input_n_points": (20.0, 1e8),
    # temperature
    "temperature_min_c": (-40.0, 100.0),
    "temperature_max_c": (-40.0, 120.0),
    "temperature_mean_c": (-40.0, 100.0),
    "temperature_min": (-40.0, 100.0),
    "temperature_max": (-40.0, 120.0),
    "temperature_mean": (-40.0, 100.0),
    "input_temperature_min_c": (-40.0, 100.0),
    "input_temperature_max_c": (-40.0, 120.0),
    "input_temperature_mean_c": (-40.0, 100.0),
    # degradation indicators
    "target_soh_recomputed": (0.0, 1.5),
    "target_soh_recomputed_clipped_0_1p2": (0.0, 1.2),
    "soh": (0.0, 1.5),
    "input_soh": (0.0, 1.5),
}


@dataclass
class FeatureFile:
    dataset: str
    phase: str
    path: Path
    level_hint: str


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_filename(value: Any, max_len: int = 160) -> str:
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value)).strip("_")
    return (text or "value")[:max_len]


def canonicalize_dataset_name(value: Any, enabled: bool = True) -> str:
    if value is None:
        return "UNKNOWN"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "UNKNOWN"
    except Exception:
        pass
    name = str(value).strip().upper()
    if not enabled:
        return name
    return DATASET_ALIASES.get(name, name)


def canonicalize_dataset_column(df: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    if enabled and "dataset" in df.columns:
        df = df.copy()
        df["dataset"] = df["dataset"].map(lambda x: canonicalize_dataset_name(x, enabled=True))
    return df


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


def parse_dataset_args(values: Optional[Sequence[str]], canonicalize: bool = True) -> List[str]:
    datasets = parse_list_argument(values)
    if not datasets or any(d.lower() == "all" for d in datasets):
        return []
    return [canonicalize_dataset_name(d, enabled=canonicalize) for d in datasets]


def parse_phase_args(values: Optional[Sequence[str]]) -> List[str]:
    phases = parse_list_argument(values)
    if not phases or any(p.lower() == "all" for p in phases):
        return list(PHASES)
    out: List[str] = []
    for p in phases:
        p = p.lower()
        if p in PHASES:
            out.append(p)
    return list(dict.fromkeys(out)) or list(PHASES)


def infer_dataset_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    stem = stem.replace(".features", "")
    stem = re.sub(r"_(charge|discharge).*", "", stem)
    stem = re.sub(r"__.*$", "", stem)
    return canonicalize_dataset_name(stem.upper() if stem else "UNKNOWN", enabled=True)


def infer_phase_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()
    if "charge" in parts or "_charge_" in name or name.startswith("charge"):
        return "charge"
    if "discharge" in parts or "_discharge_" in name or name.startswith("discharge"):
        return "discharge"
    return "unknown"


def infer_level_from_path(path: Path) -> str:
    text = str(path).replace("\\", "/")
    for level in ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "phase", "stage"]:
        if re.search(rf"(^|[/_.-]){re.escape(level)}([/_.-]|$)", text, flags=re.IGNORECASE):
            return level
    return "unknown"


def read_parquet_columns(path: Path, engine: str = "fastparquet") -> List[str]:
    try:
        if engine == "fastparquet" and fastparquet is not None:
            return list(fastparquet.ParquetFile(str(path)).columns)
    except Exception:
        pass
    try:
        return list(pd.read_parquet(path, engine=engine).columns)
    except Exception:
        return []


def parquet_row_count(path: Path, engine: str = "fastparquet") -> Optional[int]:
    try:
        if engine == "fastparquet" and fastparquet is not None:
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


def robust_read_parquet(path: Path, columns: Optional[Sequence[str]] = None, engine: str = "fastparquet") -> pd.DataFrame:
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
            raise RuntimeError(f"Could not read requested columns from {path}. Failed={failed[:20]}")
        df = pd.concat(frames, axis=1)
        for col in failed:
            df[col] = np.nan
        return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_numeric_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s.astype("float64")


def table_to_md(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No data available._"
    return df.head(max_rows).to_markdown(index=False)


# -----------------------------------------------------------------------------
# Discovery and classification
# -----------------------------------------------------------------------------

def discover_feature_files(features_root: Path, datasets: Sequence[str], phases: Sequence[str]) -> List[FeatureFile]:
    wanted = {d.upper() for d in datasets if d}
    phase_set = {p.lower() for p in phases}
    files: List[FeatureFile] = []
    if not features_root.exists():
        return files

    for path in sorted(features_root.rglob("*.parquet"), key=lambda p: str(p).lower()):
        lower = str(path).replace("\\", "/").lower()
        if "/_targets/" in lower or path.name.startswith(".") or ".tmp." in path.name:
            continue
        phase = infer_phase_from_path(path)
        if phase not in phase_set:
            continue
        ds = infer_dataset_from_filename(path)
        parts = [p.lower() for p in path.parts]
        if phase in parts:
            idx = parts.index(phase)
            if idx + 1 < len(parts):
                candidate = path.parts[idx + 1]
                if candidate.lower() not in {"_targets", "merged", "features", "feature_extraction"}:
                    ds = canonicalize_dataset_name(candidate, enabled=True)
        if wanted and ds.upper() not in wanted:
            continue
        files.append(FeatureFile(dataset=canonicalize_dataset_name(ds, enabled=True), phase=phase, path=path, level_hint=infer_level_from_path(path)))
    return files


def classify_feature_group(name: str) -> str:
    s = name.lower()
    if s in ID_LIKE_COLUMNS or s.endswith("_id") or "source" in s:
        return "metadata"
    if s in {"remaining_cycles", "remaining_cycles_norm", "cycle_life"} or s.startswith("target_") or s.startswith("input_remaining"):
        return "target"
    if s.startswith("qv") or "_qv" in s or "qv_" in s:
        return "QV_curve"
    if s.startswith("vq") or "_vq" in s or "vq_" in s:
        return "VQ_curve"
    if "capacity" in s or re.search(r"(^|_)q($|_)", s) or s.startswith("q_"):
        return "capacity"
    if "energy" in s or s.startswith("e_") or "_wh" in s:
        return "energy"
    if "voltage" in s or "_v" in s:
        return "voltage"
    if "current" in s or "_a" in s or "c_rate" in s:
        return "current"
    if "temperature" in s or "temp" in s:
        return "temperature"
    if "power" in s or "_w" in s:
        return "power"
    if "gap" in s or "relax" in s:
        return "time_gap_relaxation"
    if "time" in s or "duration" in s or "sample_period" in s or "dt" in s:
        return "sampling_time"
    if "stage" in s or "segment" in s or s.startswith("seg"):
        return "stage_structure"
    if "roll" in s or "prev" in s or "diff" in s or "slope_last" in s or "trend" in s:
        return "temporal_degradation"
    if s.startswith("is_") or s.startswith("has_") or "valid" in s or "usable" in s or "warning" in s:
        return "quality_flags"
    return "other"


def is_identifier_or_text(col: str) -> bool:
    s = col.lower()
    if col in ID_LIKE_COLUMNS or s in ID_LIKE_COLUMNS:
        return True
    if s.endswith("_id") or s.endswith("_file") or s.endswith("_path"):
        return True
    if "protocol" in s or "criterion" in s or "notes" in s or "status" in s:
        return True
    return False


def infer_numeric_columns(df: pd.DataFrame, max_probe: int = 1000) -> List[str]:
    numeric_cols: List[str] = []
    for col in df.columns:
        if is_identifier_or_text(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            numeric_cols.append(col)
        else:
            probe = pd.to_numeric(df[col].head(max_probe), errors="coerce")
            if len(probe) and probe.notna().sum() >= max(3, int(0.1 * min(len(probe), max_probe))):
                numeric_cols.append(col)
    return numeric_cols


def choose_target_column(columns: Sequence[str], requested: str = "auto") -> Optional[str]:
    if requested and requested.lower() != "auto":
        return requested if requested in columns else None
    for col in TARGET_CANDIDATES:
        if col in columns:
            return col
    return None


def choose_norm_target_column(columns: Sequence[str], requested: str = "auto") -> Optional[str]:
    if requested and requested.lower() != "auto":
        return requested if requested in columns else None
    for col in NORM_TARGET_CANDIDATES:
        if col in columns:
            return col
    return None


def add_target_norm_if_needed(df: pd.DataFrame, target_col: Optional[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    existing = choose_norm_target_column(df.columns, "auto")
    if existing:
        return df, existing
    if not target_col or target_col not in df.columns:
        return df, None
    needed_cols = {"dataset", "battery_id", "phase"}
    out = df.copy()
    y = safe_numeric_series(out[target_col])
    if needed_cols.issubset(out.columns):
        denom = out.groupby(["dataset", "battery_id", "phase"], dropna=False)[target_col].transform(lambda s: safe_numeric_series(s).max())
    elif "battery_id" in out.columns:
        denom = out.groupby(["battery_id"], dropna=False)[target_col].transform(lambda s: safe_numeric_series(s).max())
    else:
        denom = pd.Series(y.max(), index=out.index)
    denom = safe_numeric_series(denom)
    out["target_norm_auto"] = y / denom.replace(0, np.nan)
    return out, "target_norm_auto"


# -----------------------------------------------------------------------------
# Optional extraction summaries
# -----------------------------------------------------------------------------

def normalize_phase_value(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"charge", "charging"}:
        return "charge"
    if text in {"discharge", "discharging"}:
        return "discharge"
    return text


def load_capacity_life_summary(path: Optional[Path], canonicalize: bool = True) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].map(lambda x: canonicalize_dataset_name(x, enabled=canonicalize))
    phase_col = "phase_folder" if "phase_folder" in df.columns else "phase" if "phase" in df.columns else None
    if phase_col:
        df["phase"] = df[phase_col].map(normalize_phase_value)
    else:
        df["phase"] = "unknown"
    if "battery_id" in df.columns:
        df["battery_id"] = df["battery_id"].astype(str)
    return df


def audit_capacity_life_summary(path: Optional[Path], tables_dir: Path, canonicalize: bool = True) -> Dict[str, Any]:
    df = load_capacity_life_summary(path, canonicalize)
    if df.empty:
        return {"capacity_life_summary_rows": 0, "capacity_warning_rows": 0}
    write_csv(df, tables_dir / "capacity_life_summary_loaded.csv")
    status = df.get("capacity_validation_status", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    notes = df.get("capacity_validation_notes", pd.Series("", index=df.index)).fillna("").astype(str).str.strip().str.lower()
    notes = notes.mask(notes.isin(["nan", "none", "null"]), "")
    warning_mask = (~status.isin(["", "ok", "nan", "none", "null"])) | notes.str.contains("warning|below|exceed|ratio|invalid|non|low|high", regex=True, na=False)
    warnings = df.loc[warning_mask].copy()
    write_csv(warnings, tables_dir / "capacity_warning_batteries.csv")
    if not warnings.empty:
        count = warnings.groupby(["dataset", "phase"], dropna=False).size().reset_index(name="warning_series")
        write_csv(count, tables_dir / "capacity_warnings_by_dataset_phase.csv")
    eol_cols = [c for c in ["eol80_found", "eol70_found"] if c in df.columns]
    if eol_cols:
        rows = []
        for (dataset, phase), g in df.groupby(["dataset", "phase"], dropna=False):
            row = {"dataset": dataset, "phase": phase, "series": len(g)}
            if "battery_id" in g.columns:
                row["batteries"] = int(g["battery_id"].nunique())
            for c in eol_cols:
                row[f"{c}_pct"] = float(pd.Series(g[c]).astype(bool).mean() * 100.0)
                row[f"{c}_count"] = int(pd.Series(g[c]).astype(bool).sum())
            rows.append(row)
        write_csv(pd.DataFrame(rows), tables_dir / "eol_coverage_by_dataset_phase.csv")
    return {"capacity_life_summary_rows": int(len(df)), "capacity_warning_rows": int(len(warnings))}


def audit_dataset_phase_summary(path: Optional[Path], tables_dir: Path, canonicalize: bool = True) -> Dict[str, Any]:
    if not path:
        return {"dataset_phase_summary_rows": 0}
    path = Path(path)
    if not path.exists():
        return {"dataset_phase_summary_rows": 0}
    df = pd.read_csv(path)
    if df.empty:
        return {"dataset_phase_summary_rows": 0}
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].map(lambda x: canonicalize_dataset_name(x, enabled=canonicalize))
    phase_col = "phase_folder" if "phase_folder" in df.columns else "phase" if "phase" in df.columns else None
    if phase_col:
        df["phase"] = df[phase_col].map(normalize_phase_value)
    else:
        df["phase"] = "unknown"
    row_col = "rows" if "rows" in df.columns else "target_rows" if "target_rows" in df.columns else None
    if row_col is None:
        df["rows"] = 1
        row_col = "rows"
    df[row_col] = pd.to_numeric(df[row_col], errors="coerce").fillna(0)
    total_rows = float(df[row_col].sum())
    df["row_share_pct"] = df[row_col] / total_rows * 100.0 if total_rows else np.nan
    write_csv(df, tables_dir / "dataset_phase_imbalance_summary.csv")
    by_dataset = df.groupby("dataset", dropna=False).agg(rows=(row_col, "sum")).reset_index()
    by_dataset["row_share_pct"] = by_dataset["rows"] / max(float(by_dataset["rows"].sum()), 1.0) * 100.0
    by_dataset = by_dataset.sort_values("row_share_pct", ascending=False)
    write_csv(by_dataset, tables_dir / "dataset_imbalance_summary.csv")
    return {
        "dataset_phase_summary_rows": int(len(df)),
        "largest_dataset_row_share_pct": float(by_dataset["row_share_pct"].max()) if not by_dataset.empty else np.nan,
    }


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def feature_inventory(files: Sequence[FeatureFile], engine: str) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for item in tqdm(files, desc="Feature inventory", unit="file"):
        cols = read_parquet_columns(item.path, engine)
        n_rows = parquet_row_count(item.path, engine)
        for col in cols:
            group = classify_feature_group(col)
            records.append({
                "feature_name": col,
                "feature_group": group,
                "hierarchical_level": item.level_hint,
                "source_file": str(item.path),
                "phase": item.phase,
                "dataset": item.dataset,
                "n_rows_file": n_rows,
                "is_identifier": is_identifier_or_text(col),
                "is_target": group == "target",
                "is_quality_flag": group == "quality_flags",
            })
    return pd.DataFrame(records)


def completeness_for_df(df: pd.DataFrame, dataset: str, phase: str, file_path: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    numeric_cols = infer_numeric_columns(df)
    for col in numeric_cols:
        s = safe_numeric_series(df[col])
        n = len(s)
        arr = s.to_numpy(dtype=float, na_value=np.nan)
        n_non_nan = int(np.isfinite(arr).sum())
        n_nan = int(n - n_non_nan)
        unique_count = int(s.dropna().nunique()) if n_non_nan else 0
        finite = float(n_non_nan / n * 100.0) if n else np.nan
        nan_pct = float(n_nan / n * 100.0) if n else np.nan
        is_const = bool(unique_count <= 1)
        if is_const:
            availability = "constant"
        elif finite >= 95:
            availability = "excellent"
        elif finite >= 80:
            availability = "good"
        elif finite >= 40:
            availability = "partial"
        elif finite >= 5:
            availability = "poor"
        else:
            availability = "missing"
        records.append({
            "feature_name": col,
            "feature_group": classify_feature_group(col),
            "dataset": dataset,
            "phase": phase,
            "source_file": str(file_path),
            "n_rows": n,
            "n_non_nan": n_non_nan,
            "n_nan": n_nan,
            "nan_pct": nan_pct,
            "finite_pct": finite,
            "n_unique": unique_count,
            "is_constant": is_const,
            "availability_class": availability,
        })
    return pd.DataFrame(records)


def physical_violations_for_df(df: pd.DataFrame, dataset: str, phase: str, file_path: Path, max_records_per_file: int) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for col, (lo, hi) in PHYSICAL_RULES.items():
        if col not in df.columns:
            continue
        s = safe_numeric_series(df[col])
        bad = ((s < lo) | (s > hi)) & s.notna()
        if not bad.any():
            continue
        idxs = list(np.flatnonzero(bad.to_numpy()))[:max_records_per_file]
        for idx in idxs:
            row = df.iloc[idx]
            records.append({
                "dataset": str(row.get("dataset", dataset)),
                "phase": str(row.get("phase", phase)),
                "battery_id": row.get("battery_id", None),
                "cycle_number": row.get("cycle_number", None),
                "cycle_index": row.get("cycle_index", None),
                "feature_name": col,
                "value": s.iloc[idx],
                "rule_violated": f"expected {lo} <= {col} <= {hi}",
                "severity": "warning" if col in {"n_points", "input_n_points"} else "review",
                "source_file": str(file_path),
            })
    for col in ["current_mean_a", "current_mean", "input_current_mean_a"]:
        if col not in df.columns:
            continue
        s = safe_numeric_series(df[col])
        if phase == "charge":
            bad = (s <= 0) & s.notna()
            rule = f"{col} should be positive for charge"
        elif phase == "discharge":
            bad = (s >= 0) & s.notna()
            rule = f"{col} should be negative for discharge"
        else:
            continue
        idxs = list(np.flatnonzero(bad.to_numpy()))[:max_records_per_file]
        for idx in idxs:
            row = df.iloc[idx]
            records.append({
                "dataset": str(row.get("dataset", dataset)),
                "phase": str(row.get("phase", phase)),
                "battery_id": row.get("battery_id", None),
                "cycle_number": row.get("cycle_number", None),
                "cycle_index": row.get("cycle_index", None),
                "feature_name": col,
                "value": s.iloc[idx],
                "rule_violated": rule,
                "severity": "review",
                "source_file": str(file_path),
            })
    return pd.DataFrame(records)


def outliers_for_df(df: pd.DataFrame, dataset: str, phase: str, file_path: Path, max_records_per_file: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_records: List[Dict[str, Any]] = []
    cycle_records: List[Dict[str, Any]] = []
    numeric_cols = infer_numeric_columns(df)
    excluded = set(TARGET_CANDIDATES) | {"cycle_number", "cycle_index", "source_row_id", "stage_index", "stage_total", "order_index"}
    numeric_cols = [c for c in numeric_cols if c not in excluded and classify_feature_group(c) not in {"metadata", "target"}]

    for col in numeric_cols:
        s = safe_numeric_series(df[col])
        valid = s.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        n_valid = len(valid)
        if n_valid < 20 or valid.nunique() <= 1:
            continue
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            lo_iqr, hi_iqr = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask_iqr = ((s < lo_iqr) | (s > hi_iqr)) & s.notna()
        else:
            mask_iqr = pd.Series(False, index=s.index)
        med = valid.median()
        mad = np.median(np.abs(valid - med))
        if np.isfinite(mad) and mad > 0:
            robust_z = 0.6745 * (s - med) / mad
            mask_mad = (robust_z.abs() > 3.5) & s.notna()
        else:
            robust_z = pd.Series(np.nan, index=s.index)
            mask_mad = pd.Series(False, index=s.index)
        p001, p999 = valid.quantile(0.001), valid.quantile(0.999)
        mask_pct = ((s < p001) | (s > p999)) & s.notna()

        for method, mask, score_series in [
            ("IQR", mask_iqr, s),
            ("MAD", mask_mad, robust_z),
            ("P001_P999", mask_pct, s),
        ]:
            count = int(mask.sum())
            summary_records.append({
                "feature_name": col,
                "feature_group": classify_feature_group(col),
                "dataset": dataset,
                "phase": phase,
                "source_file": str(file_path),
                "outlier_method": method,
                "n_valid": n_valid,
                "outlier_count": count,
                "outlier_pct": float(count / len(df) * 100.0) if len(df) else np.nan,
            })
            if count:
                idxs = list(np.flatnonzero(mask.to_numpy()))[:max_records_per_file]
                for idx in idxs:
                    row = df.iloc[idx]
                    cycle_records.append({
                        "dataset": str(row.get("dataset", dataset)),
                        "phase": str(row.get("phase", phase)),
                        "battery_id": row.get("battery_id", None),
                        "cycle_number": row.get("cycle_number", None),
                        "cycle_index": row.get("cycle_index", None),
                        "feature_name": col,
                        "feature_group": classify_feature_group(col),
                        "feature_value": s.iloc[idx],
                        "outlier_method": method,
                        "outlier_score": score_series.iloc[idx] if idx < len(score_series) else np.nan,
                        "source_file": str(file_path),
                    })
    return pd.DataFrame(summary_records), pd.DataFrame(cycle_records)


def target_balance(df: pd.DataFrame, dataset: str, phase: str, target_norm_col: Optional[str]) -> pd.DataFrame:
    if not target_norm_col or target_norm_col not in df.columns:
        return pd.DataFrame()
    s = safe_numeric_series(df[target_norm_col])
    labels = pd.cut(
        s,
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["near_eol", "late_life", "mid_life", "early_life"],
        right=False,
    )
    counts = labels.value_counts(dropna=False).reset_index()
    counts.columns = ["life_range", "n_rows"]
    counts["dataset"] = dataset
    counts["phase"] = phase
    counts["pct"] = counts["n_rows"] / len(df) * 100.0 if len(df) else np.nan
    return counts[["dataset", "phase", "life_range", "n_rows", "pct"]]


def sample_for_correlation(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def build_analysis_frame(files: Sequence[FeatureFile], args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    completeness_frames: List[pd.DataFrame] = []
    violation_frames: List[pd.DataFrame] = []
    outlier_summary_frames: List[pd.DataFrame] = []
    outlier_cycle_frames: List[pd.DataFrame] = []
    target_balance_frames: List[pd.DataFrame] = []
    sampled_frames: List[pd.DataFrame] = []
    failures: List[Dict[str, Any]] = []

    per_file_sample = max(1, args.max_correlation_rows // max(1, len(files))) if args.max_correlation_rows > 0 else 0

    for item in tqdm(files, desc="Feature diagnostics", unit="file"):
        try:
            cols = read_parquet_columns(item.path, args.engine)
            if not cols:
                failures.append({"path": str(item.path), "error": "no_columns_detected"})
                continue
            df = robust_read_parquet(item.path, columns=cols, engine=args.engine)
            if "dataset" not in df.columns:
                df["dataset"] = item.dataset
            if "phase" not in df.columns:
                df["phase"] = item.phase
            df = canonicalize_dataset_column(df, enabled=args.canonicalize_dataset_names)
            if "phase_folder" not in df.columns:
                df["phase_folder"] = item.phase

            target_col = choose_target_column(df.columns, args.target_column)
            df, target_norm_col = add_target_norm_if_needed(df, target_col)
            requested_norm = choose_norm_target_column(df.columns, args.target_norm_column)
            if requested_norm:
                target_norm_col = requested_norm

            comp = completeness_for_df(df, item.dataset, item.phase, item.path)
            completeness_frames.append(comp)

            viol = physical_violations_for_df(df, item.dataset, item.phase, item.path, max_records_per_file=args.max_issue_rows_per_file)
            if not viol.empty:
                violation_frames.append(viol)

            out_sum, out_cyc = outliers_for_df(df, item.dataset, item.phase, item.path, max_records_per_file=args.max_issue_rows_per_file)
            if not out_sum.empty:
                outlier_summary_frames.append(out_sum)
            if not out_cyc.empty:
                outlier_cycle_frames.append(out_cyc)

            tb = target_balance(df, item.dataset, item.phase, target_norm_col)
            if not tb.empty:
                target_balance_frames.append(tb)

            numeric_cols = infer_numeric_columns(df)
            keep_cols = [c for c in ["dataset", "phase", "phase_folder", "battery_id", "cycle_number", "cycle_index", "feature_level"] if c in df.columns]
            if target_col and target_col not in keep_cols:
                keep_cols.append(target_col)
            if target_norm_col and target_norm_col not in keep_cols:
                keep_cols.append(target_norm_col)
            feature_cols = [c for c in numeric_cols if c not in keep_cols and classify_feature_group(c) not in {"metadata"}]
            selected_cols = list(dict.fromkeys(keep_cols + feature_cols))
            sub = df[selected_cols].copy()
            if per_file_sample > 0:
                sub = sample_for_correlation(sub, per_file_sample, args.random_seed)
            sub["__source_file"] = str(item.path)
            sampled_frames.append(sub)
        except Exception as exc:
            failures.append({
                "path": str(item.path),
                "dataset": item.dataset,
                "phase": item.phase,
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=3),
            })

    completeness = pd.concat(completeness_frames, ignore_index=True) if completeness_frames else pd.DataFrame()
    violations = pd.concat(violation_frames, ignore_index=True) if violation_frames else pd.DataFrame()
    outlier_summary = pd.concat(outlier_summary_frames, ignore_index=True) if outlier_summary_frames else pd.DataFrame()
    outlier_cycles = pd.concat(outlier_cycle_frames, ignore_index=True) if outlier_cycle_frames else pd.DataFrame()
    target_bal = pd.concat(target_balance_frames, ignore_index=True) if target_balance_frames else pd.DataFrame()
    analysis = pd.concat(sampled_frames, ignore_index=True) if sampled_frames else pd.DataFrame()
    return analysis, completeness, violations, outlier_summary, outlier_cycles, target_bal, failures


# -----------------------------------------------------------------------------
# Correlation, redundancy, domain shift, curve validity, scorecard
# -----------------------------------------------------------------------------

def corr_pair(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    xx = safe_numeric_series(x)
    yy = safe_numeric_series(y)
    valid = xx.notna() & yy.notna() & np.isfinite(xx) & np.isfinite(yy)
    n = int(valid.sum())
    if n < 5 or xx[valid].nunique() <= 1 or yy[valid].nunique() <= 1:
        return np.nan, np.nan, n
    pearson = float(xx[valid].corr(yy[valid], method="pearson"))
    spearman = float(xx[valid].corr(yy[valid], method="spearman"))
    return pearson, spearman, n


def correlations_for_scope(df: pd.DataFrame, scope_name: str, target_col: str, target_norm_col: Optional[str], args: argparse.Namespace) -> pd.DataFrame:
    if df.empty or target_col not in df.columns:
        return pd.DataFrame()
    numeric_cols = infer_numeric_columns(df)
    excluded = set([target_col]) | set(TARGET_CANDIDATES) | {"cycle_number", "cycle_index", "source_row_id", "stage_index", "stage_total", "order_index"}
    features = [c for c in numeric_cols if c not in excluded and classify_feature_group(c) not in {"metadata", "target"}]
    if args.max_corr_features and len(features) > args.max_corr_features:
        scores = []
        for c in features:
            s = safe_numeric_series(df[c])
            scores.append((c, s.notna().mean(), s.nunique()))
        features = [c for c, _, _ in sorted(scores, key=lambda t: (t[1], t[2]), reverse=True)[:args.max_corr_features]]

    mi_values: Dict[str, float] = {}
    if args.mutual_info and mutual_info_regression is not None and features:
        try:
            sub_cols = [target_col] + features
            sub = df[sub_cols].copy()
            for c in features:
                sub[c] = safe_numeric_series(sub[c])
            sub[target_col] = safe_numeric_series(sub[target_col])
            sub = sub.dropna()
            if len(sub) >= 20:
                X = sub[features].to_numpy(dtype=float)
                yy = sub[target_col].to_numpy(dtype=float)
                mi = mutual_info_regression(X, yy, random_state=args.random_seed)
                mi_values = {c: float(v) for c, v in zip(features, mi)}
        except Exception:
            mi_values = {}

    records: List[Dict[str, Any]] = []
    y = df[target_col]
    y_norm = df[target_norm_col] if target_norm_col and target_norm_col in df.columns else None
    for feat in features:
        pear, spear, n = corr_pair(df[feat], y)
        if y_norm is not None:
            pear_n, spear_n, n_norm = corr_pair(df[feat], y_norm)
        else:
            pear_n, spear_n, n_norm = (np.nan, np.nan, 0)
        s = safe_numeric_series(df[feat])
        records.append({
            "scope": scope_name,
            "feature_name": feat,
            "feature_group": classify_feature_group(feat),
            "target_column": target_col,
            "target_norm_column": target_norm_col,
            "n_valid": n,
            "pearson_r": pear,
            "pearson_abs": abs(pear) if np.isfinite(pear) else np.nan,
            "spearman_r": spear,
            "spearman_abs": abs(spear) if np.isfinite(spear) else np.nan,
            "pearson_norm_r": pear_n,
            "pearson_norm_abs": abs(pear_n) if np.isfinite(pear_n) else np.nan,
            "spearman_norm_r": spear_n,
            "spearman_norm_abs": abs(spear_n) if np.isfinite(spear_n) else np.nan,
            "mutual_info": mi_values.get(feat, np.nan),
            "nan_pct": float(s.isna().mean() * 100.0) if len(s) else np.nan,
        })
    return pd.DataFrame(records)


def compute_correlations(analysis: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if analysis.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    target_col = choose_target_column(analysis.columns, args.target_column)
    analysis, auto_norm = add_target_norm_if_needed(analysis, target_col)
    target_norm_col = choose_norm_target_column(analysis.columns, args.target_norm_column) or auto_norm
    if not target_col:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    corr_global = correlations_for_scope(analysis, "global", target_col, target_norm_col, args)

    phase_frames: List[pd.DataFrame] = []
    if "phase" in analysis.columns:
        for phase, g in tqdm(analysis.groupby("phase", dropna=False), desc="Correlations by phase", leave=False):
            c = correlations_for_scope(g, f"phase={phase}", target_col, target_norm_col, args)
            if not c.empty:
                c["phase"] = phase
                phase_frames.append(c)
    corr_phase = pd.concat(phase_frames, ignore_index=True) if phase_frames else pd.DataFrame()

    ds_phase_frames: List[pd.DataFrame] = []
    if "dataset" in analysis.columns and "phase" in analysis.columns:
        for (dataset, phase), g in tqdm(analysis.groupby(["dataset", "phase"], dropna=False), desc="Correlations by dataset/phase", leave=False):
            c = correlations_for_scope(g, f"dataset={dataset};phase={phase}", target_col, target_norm_col, args)
            if not c.empty:
                c["dataset"] = dataset
                c["phase"] = phase
                ds_phase_frames.append(c)
    corr_ds_phase = pd.concat(ds_phase_frames, ignore_index=True) if ds_phase_frames else pd.DataFrame()
    return corr_global, corr_phase, corr_ds_phase


def compute_redundancy(analysis: pd.DataFrame, corr_global: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if analysis.empty:
        return pd.DataFrame()
    numeric_cols = infer_numeric_columns(analysis)
    target_cols = set(TARGET_CANDIDATES) | {"cycle_number", "cycle_index", "source_row_id", "stage_index", "stage_total", "order_index"}
    features = [c for c in numeric_cols if c not in target_cols and classify_feature_group(c) not in {"metadata", "target"}]
    target_corr = {}
    if corr_global is not None and not corr_global.empty:
        target_corr = dict(zip(corr_global["feature_name"], corr_global["spearman_abs"]))
    records: List[Dict[str, Any]] = []
    for group_name in tqdm(sorted({classify_feature_group(c) for c in features}), desc="Redundancy by group", leave=False):
        group_features = [c for c in features if classify_feature_group(c) == group_name]
        if len(group_features) < 2:
            continue
        if len(group_features) > args.max_redundancy_features_per_group:
            scores = []
            for c in group_features:
                s = safe_numeric_series(analysis[c])
                tc = target_corr.get(c, np.nan)
                scores.append((c, tc if np.isfinite(tc) else 0.0, s.notna().mean()))
            group_features = [c for c, _, _ in sorted(scores, key=lambda t: (t[1], t[2]), reverse=True)[:args.max_redundancy_features_per_group]]
        sub = analysis[group_features].apply(safe_numeric_series)
        corr = sub.corr(method="spearman", min_periods=20)
        for i, a in enumerate(group_features):
            for b in group_features[i + 1:]:
                val = corr.loc[a, b]
                if pd.notna(val) and abs(float(val)) >= args.redundancy_threshold:
                    corr_a = target_corr.get(a, np.nan)
                    corr_b = target_corr.get(b, np.nan)
                    keep, drop = (a, b)
                    if np.nan_to_num(corr_b, nan=-1) > np.nan_to_num(corr_a, nan=-1):
                        keep, drop = (b, a)
                    records.append({
                        "feature_a": a,
                        "feature_b": b,
                        "feature_group": group_name,
                        "correlation": float(val),
                        "nan_pct_a": float(safe_numeric_series(analysis[a]).isna().mean() * 100.0),
                        "nan_pct_b": float(safe_numeric_series(analysis[b]).isna().mean() * 100.0),
                        "target_corr_a": corr_a,
                        "target_corr_b": corr_b,
                        "suggested_keep": keep,
                        "suggested_drop": drop,
                    })
    return pd.DataFrame(records)


def compute_domain_shift(analysis: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if analysis.empty or "dataset" not in analysis.columns or "phase" not in analysis.columns:
        return pd.DataFrame(), pd.DataFrame()
    numeric_cols = infer_numeric_columns(analysis)
    excluded = set(TARGET_CANDIDATES) | {"cycle_number", "cycle_index", "source_row_id", "stage_index", "stage_total", "order_index"}
    features = [c for c in numeric_cols if c not in excluded and classify_feature_group(c) not in {"metadata", "target"}]
    if len(features) > args.max_domain_features:
        scores = []
        for c in features:
            s = safe_numeric_series(analysis[c])
            scores.append((c, s.notna().mean(), s.nunique()))
        features = [c for c, _, _ in sorted(scores, key=lambda t: (t[1], t[2]), reverse=True)[:args.max_domain_features]]

    records: List[Dict[str, Any]] = []
    for feat in tqdm(features, desc="Domain-shift statistics", leave=False):
        for (dataset, phase), g in analysis.groupby(["dataset", "phase"], dropna=False):
            s = pd.to_numeric(safe_numeric_series(g[feat]), errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) == 0:
                records.append({
                    "feature_name": feat, "feature_group": classify_feature_group(feat),
                    "dataset": dataset, "phase": phase, "n_valid": 0,
                    "mean": np.nan, "median": np.nan, "std": np.nan, "iqr": np.nan,
                    "min": np.nan, "max": np.nan, "nan_pct": 100.0,
                })
                continue
            records.append({
                "feature_name": feat,
                "feature_group": classify_feature_group(feat),
                "dataset": dataset,
                "phase": phase,
                "n_valid": len(s),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "iqr": float(np.nanquantile(s.to_numpy(dtype="float64"), 0.75) - np.nanquantile(s.to_numpy(dtype="float64"), 0.25)),
                "min": float(s.min()),
                "max": float(s.max()),
                "nan_pct": float(safe_numeric_series(g[feat]).isna().mean() * 100.0),
            })
    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, pd.DataFrame()
    scores = []
    for feat, g in detail.groupby("feature_name"):
        med = g["median"].dropna()
        if len(med) < 2:
            rel_range = np.nan
            cv_median = np.nan
        else:
            denom = abs(float(med.median())) + 1e-12
            rel_range = float((med.max() - med.min()) / denom)
            cv_median = float(med.std() / denom)
        scores.append({
            "feature_name": feat,
            "feature_group": classify_feature_group(feat),
            "domain_shift_score": rel_range,
            "median_cv_across_domains": cv_median,
            "n_domains": int(g[["dataset", "phase"]].drop_duplicates().shape[0]),
        })
    summary = pd.DataFrame(scores).sort_values("domain_shift_score", ascending=False)
    return detail, summary


def curve_feature_validity(analysis: pd.DataFrame) -> pd.DataFrame:
    if analysis.empty:
        return pd.DataFrame()
    valid_cols = [c for c in analysis.columns if re.match(r"^(QV|VQ)_valid_p\d+", c, flags=re.IGNORECASE)]
    if not valid_cols:
        return pd.DataFrame()
    records = []
    group_cols = [c for c in ["dataset", "phase"] if c in analysis.columns]
    grouped = analysis.groupby(group_cols, dropna=False) if group_cols else [((), analysis)]
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        for col in valid_cols:
            s = safe_numeric_series(g[col])
            records.append({
                **base,
                "curve_valid_column": col,
                "n_rows": len(g),
                "valid_pct": float((s > 0).mean() * 100.0),
                "nan_pct": float(s.isna().mean() * 100.0),
            })
    return pd.DataFrame(records)


def feature_quality_scorecard(
    completeness: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    violations: pd.DataFrame,
    corr_global: pd.DataFrame,
    domain_summary: pd.DataFrame,
    redundant: pd.DataFrame,
) -> pd.DataFrame:
    if completeness.empty:
        return pd.DataFrame()
    comp = completeness.groupby(["feature_name", "feature_group", "phase"], dropna=False).agg(
        finite_pct_global=("finite_pct", "mean"),
        nan_pct_global=("nan_pct", "mean"),
        n_unique_mean=("n_unique", "mean"),
        available_in_datasets=("dataset", lambda x: ",".join(sorted(set(map(str, x))))),
    ).reset_index()
    if not outlier_summary.empty:
        out = outlier_summary.groupby(["feature_name", "phase"], dropna=False).agg(outlier_pct_global=("outlier_pct", "mean")).reset_index()
        comp = comp.merge(out, on=["feature_name", "phase"], how="left")
    else:
        comp["outlier_pct_global"] = np.nan
    if not violations.empty:
        viol = violations.groupby(["feature_name", "phase"], dropna=False).size().reset_index(name="physical_violation_count")
        total_by_phase = completeness.groupby("phase")["n_rows"].sum().to_dict()
        viol["physical_violation_pct"] = viol.apply(lambda r: float(r["physical_violation_count"] / max(total_by_phase.get(r["phase"], 1), 1) * 100.0), axis=1)
        comp = comp.merge(viol[["feature_name", "phase", "physical_violation_pct"]], on=["feature_name", "phase"], how="left")
    else:
        comp["physical_violation_pct"] = np.nan
    if not corr_global.empty:
        corr = corr_global[["feature_name", "spearman_abs", "spearman_norm_abs", "pearson_abs", "mutual_info"]].copy()
        corr = corr.groupby("feature_name", dropna=False).max().reset_index()
        corr = corr.rename(columns={
            "spearman_abs": "abs_spearman_target",
            "spearman_norm_abs": "abs_spearman_target_norm",
            "pearson_abs": "abs_pearson_target",
        })
        comp = comp.merge(corr, on="feature_name", how="left")
    else:
        comp["abs_spearman_target"] = np.nan
        comp["abs_spearman_target_norm"] = np.nan
        comp["abs_pearson_target"] = np.nan
        comp["mutual_info"] = np.nan
    if not domain_summary.empty:
        ds = domain_summary[["feature_name", "domain_shift_score", "median_cv_across_domains"]].drop_duplicates()
        comp = comp.merge(ds, on="feature_name", how="left")
    else:
        comp["domain_shift_score"] = np.nan
        comp["median_cv_across_domains"] = np.nan
    if not redundant.empty:
        drops = set(redundant["suggested_drop"].dropna().astype(str).tolist())
        comp["redundancy_flag"] = comp["feature_name"].astype(str).isin(drops)
    else:
        comp["redundancy_flag"] = False

    statuses = []
    reasons = []
    for _, r in comp.iterrows():
        finite = r.get("finite_pct_global", np.nan)
        nunique = r.get("n_unique_mean", np.nan)
        outp = r.get("outlier_pct_global", np.nan)
        viol = r.get("physical_violation_pct", np.nan)
        corr = r.get("abs_spearman_target", np.nan)
        shift = r.get("domain_shift_score", np.nan)
        red = bool(r.get("redundancy_flag", False))
        status = "review"
        reason = "default review"
        if np.isfinite(finite) and finite < 5:
            status, reason = "drop_missing", "finite_pct_global < 5%"
        elif np.isfinite(nunique) and nunique <= 1:
            status, reason = "drop_constant", "n_unique_mean <= 1"
        elif np.isfinite(viol) and viol > 20:
            status, reason = "drop_physically_invalid", "physical_violation_pct > 20%"
        elif np.isfinite(shift) and shift > 10:
            status, reason = "drop_unstable", "domain_shift_score > 10"
        elif np.isfinite(corr) and corr >= 0.30 and np.isfinite(finite) and finite >= 80 and not red:
            status, reason = "keep", "high target association and good completeness"
        elif np.isfinite(corr) and corr >= 0.10 and np.isfinite(finite) and finite >= 40:
            status, reason = "keep_with_caution", "moderate target association or partial completeness"
        elif np.isfinite(outp) and outp > 30:
            status, reason = "review", "outlier_pct_global > 30%"
        elif red:
            status, reason = "review", "redundancy_flag=True"
        statuses.append(status)
        reasons.append(reason)
    comp["recommended_status"] = statuses
    comp["recommendation_reason"] = reasons
    return comp.sort_values(["recommended_status", "abs_spearman_target"], ascending=[True, False])


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def generate_markdown_report(tables_dir: Path, report_path: Path, run_info: Dict[str, Any]) -> None:
    ensure_dir(report_path.parent)
    inv = read_table(tables_dir / "feature_inventory.csv")
    comp = read_table(tables_dir / "feature_completeness_summary.csv")
    viol = read_table(tables_dir / "physical_limit_violations.csv")
    out_sum = read_table(tables_dir / "feature_outlier_summary.csv")
    target_bal = read_table(tables_dir / "target_balance_summary.csv")
    corr = read_table(tables_dir / "feature_target_correlation_global.csv")
    corr_phase = read_table(tables_dir / "feature_target_correlation_by_phase.csv")
    corr_ds = read_table(tables_dir / "feature_target_correlation_by_dataset_phase.csv")
    red = read_table(tables_dir / "redundant_feature_pairs.csv")
    domain = read_table(tables_dir / "feature_domain_shift_summary.csv")
    curve = read_table(tables_dir / "curve_feature_validity_by_dataset_phase.csv")
    score = read_table(tables_dir / "feature_quality_scorecard.csv")
    cap_warn = read_table(tables_dir / "capacity_warning_batteries.csv")
    imbalance = read_table(tables_dir / "dataset_phase_imbalance_summary.csv")

    lines: List[str] = []
    lines.append("# Feature Report-Only Audit")
    lines.append("")
    lines.append(f"Generated at: `{now_iso()}`")
    lines.append("")
    lines.append("## 1. Run summary")
    lines.append("")
    for k, v in run_info.items():
        lines.append(f"- `{k}`: `{v}`")

    lines.append("")
    lines.append("## 2. What this report did not do")
    lines.append("")
    lines.append("This script did not create raw-curve, V(Q), E(Q), or degradation plots. It only audited extracted feature tables and wrote tabular/Markdown outputs.")

    lines.append("")
    lines.append("## 3. Feature inventory")
    lines.append("")
    if not inv.empty:
        by_group = inv.groupby("feature_group", dropna=False)["feature_name"].nunique().reset_index(name="n_features").sort_values("n_features", ascending=False)
        lines.append(table_to_md(by_group, 30))
    else:
        lines.append("_No feature inventory available._")

    lines.append("")
    lines.append("## 4. Completeness and NaN diagnostics")
    lines.append("")
    if not comp.empty:
        class_counts = comp.groupby(["feature_group", "availability_class"], dropna=False).size().reset_index(name="count").sort_values(["feature_group", "availability_class"])
        lines.append(table_to_md(class_counts, 60))
        low = comp[comp["availability_class"].isin(["missing", "poor", "constant"])]
        if not low.empty:
            lines.append("")
            lines.append("### Lowest availability / constant features")
            show = low.sort_values(["finite_pct", "n_unique"], ascending=[True, True]).head(40)
            lines.append(table_to_md(show[["feature_name", "feature_group", "dataset", "phase", "finite_pct", "nan_pct", "n_unique", "availability_class"]], 40))
    else:
        lines.append("_No completeness table available._")

    lines.append("")
    lines.append("## 5. Physical plausibility")
    lines.append("")
    lines.append(f"Violation rows: `{len(viol)}`")
    if not viol.empty:
        viol_summary = viol.groupby(["feature_name", "rule_violated", "severity"], dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
        lines.append(table_to_md(viol_summary, 30))

    lines.append("")
    lines.append("## 6. Outlier diagnostics")
    lines.append("")
    if not out_sum.empty:
        out_top = out_sum.groupby(["feature_name", "feature_group"], dropna=False)["outlier_pct"].mean().reset_index().sort_values("outlier_pct", ascending=False).head(40)
        lines.append(table_to_md(out_top, 40))
    else:
        lines.append("_No outlier summary available._")

    lines.append("")
    lines.append("## 7. Target distribution and imbalance")
    lines.append("")
    if not target_bal.empty:
        lines.append(table_to_md(target_bal, 60))
    else:
        lines.append("_No target-balance table available._")
    if not imbalance.empty:
        lines.append("")
        lines.append("### Dataset/phase imbalance from extraction summary")
        cols = [c for c in ["dataset", "phase", "rows", "target_rows", "batteries", "n_batteries", "cycles", "n_cycles", "row_share_pct"] if c in imbalance.columns]
        show = imbalance[cols].sort_values("row_share_pct", ascending=False).head(30) if "row_share_pct" in imbalance.columns else imbalance[cols].head(30)
        lines.append(table_to_md(show, 30))

    lines.append("")
    lines.append("## 8. Feature-target correlations")
    lines.append("")
    if not corr.empty:
        top = corr.sort_values("spearman_abs", ascending=False).head(40)
        cols = [c for c in ["feature_name", "feature_group", "target_column", "n_valid", "pearson_r", "spearman_r", "spearman_abs", "spearman_norm_r", "mutual_info", "nan_pct"] if c in top.columns]
        lines.append(table_to_md(top[cols], 40))
    else:
        lines.append("_No global correlation table available._")

    if not corr_phase.empty and "phase" in corr_phase.columns:
        lines.append("")
        lines.append("### Top correlations by phase")
        rows = []
        for phase, g in corr_phase.groupby("phase", dropna=False):
            gg = g.sort_values("spearman_abs", ascending=False).head(15).copy()
            rows.append(gg)
        if rows:
            top_phase = pd.concat(rows, ignore_index=True)
            cols = [c for c in ["phase", "feature_name", "feature_group", "n_valid", "spearman_r", "spearman_abs", "nan_pct"] if c in top_phase.columns]
            lines.append(table_to_md(top_phase[cols], 60))

    if not corr_ds.empty and {"dataset", "phase"}.issubset(corr_ds.columns):
        lines.append("")
        lines.append("### Top correlations by dataset and phase")
        rows = []
        for (dataset, phase), g in corr_ds.groupby(["dataset", "phase"], dropna=False):
            rows.append(g.sort_values("spearman_abs", ascending=False).head(5))
        if rows:
            top_ds = pd.concat(rows, ignore_index=True)
            cols = [c for c in ["dataset", "phase", "feature_name", "feature_group", "n_valid", "spearman_r", "spearman_abs", "nan_pct"] if c in top_ds.columns]
            lines.append(table_to_md(top_ds[cols], 80))

    lines.append("")
    lines.append("## 9. Redundant feature pairs")
    lines.append("")
    if not red.empty:
        show = red.sort_values("correlation", key=lambda s: s.abs(), ascending=False).head(40)
        lines.append(table_to_md(show, 40))
    else:
        lines.append("_No redundant pairs above threshold were found._")

    lines.append("")
    lines.append("## 10. Domain shift / stability")
    lines.append("")
    if not domain.empty:
        show = domain.sort_values("domain_shift_score", ascending=False).head(40)
        lines.append(table_to_md(show, 40))
    else:
        lines.append("_No domain-shift summary available._")

    lines.append("")
    lines.append("## 11. Q(V) and V(Q) validity")
    lines.append("")
    if not curve.empty:
        lines.append(table_to_md(curve.sort_values("valid_pct", ascending=True).head(60), 60))
    else:
        lines.append("_No QV/VQ validity columns were found._")

    lines.append("")
    lines.append("## 12. Capacity-life warnings")
    lines.append("")
    if not cap_warn.empty:
        cols = [c for c in ["dataset", "battery_id", "phase", "capacity_validation_status", "capacity_validation_notes", "initial_capacity_ah", "capacity_last_ah", "capacity_end_to_initial_ratio", "n_rows"] if c in cap_warn.columns]
        lines.append(table_to_md(cap_warn[cols].head(60), 60))
    else:
        lines.append("_No capacity-life warnings were provided or detected._")

    lines.append("")
    lines.append("## 13. Feature-quality scorecard")
    lines.append("")
    if not score.empty:
        status_counts = score["recommended_status"].value_counts(dropna=False).reset_index()
        status_counts.columns = ["recommended_status", "count"]
        lines.append(table_to_md(status_counts, 20))
        lines.append("")
        lines.append("### Recommended keep / keep_with_caution")
        keep = score[score["recommended_status"].isin(["keep", "keep_with_caution"])].copy()
        if "abs_spearman_target" in keep.columns:
            keep = keep.sort_values("abs_spearman_target", ascending=False)
        cols = [c for c in ["feature_name", "feature_group", "phase", "finite_pct_global", "outlier_pct_global", "abs_spearman_target", "abs_spearman_target_norm", "domain_shift_score", "redundancy_flag", "recommended_status", "recommendation_reason"] if c in keep.columns]
        lines.append(table_to_md(keep[cols].head(80), 80))
    else:
        lines.append("_No scorecard available._")

    lines.append("")
    lines.append("## 14. Main output files")
    lines.append("")
    expected = [
        "feature_inventory.csv",
        "feature_completeness_summary.csv",
        "physical_limit_violations.csv",
        "feature_outlier_summary.csv",
        "feature_outlier_events.csv",
        "target_balance_summary.csv",
        "feature_target_correlation_global.csv",
        "feature_target_correlation_by_phase.csv",
        "feature_target_correlation_by_dataset_phase.csv",
        "redundant_feature_pairs.csv",
        "feature_domain_shift_detail.csv",
        "feature_domain_shift_summary.csv",
        "curve_feature_validity_by_dataset_phase.csv",
        "feature_quality_scorecard.csv",
        "feature_report_only_run_info.json",
    ]
    file_rows = []
    for name in expected:
        p = tables_dir / name if name.endswith(".csv") else report_path.parent / name
        if p.exists():
            if p.suffix.lower() == ".csv":
                try:
                    rows = sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")) - 1
                except Exception:
                    rows = "unknown"
            else:
                rows = "exists"
        else:
            rows = "missing"
        file_rows.append({"file": name, "rows/status": rows})
    lines.append(table_to_md(pd.DataFrame(file_rows), 100))

    report_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main audit
# -----------------------------------------------------------------------------

def run_report_only_audit(args: argparse.Namespace) -> Dict[str, Any]:
    output_base = args.output_root / args.output_subdir
    tables_dir = output_base / "tables"
    reports_dir = output_base / "reports"
    ensure_dir(tables_dir)
    ensure_dir(reports_dir)

    aux_summary: Dict[str, Any] = {}
    aux_summary.update(audit_capacity_life_summary(args.capacity_life_summary, tables_dir, args.canonicalize_dataset_names))
    aux_summary.update(audit_dataset_phase_summary(args.dataset_phase_summary, tables_dir, args.canonicalize_dataset_names))

    files = discover_feature_files(args.features_root, args.datasets, args.phases)
    if not files:
        raise RuntimeError(f"No feature parquet files found under {args.features_root}")

    inventory = feature_inventory(files, args.engine)
    write_csv(inventory, tables_dir / "feature_inventory.csv")

    analysis, completeness, violations, outlier_summary, outlier_events, target_bal, failures = build_analysis_frame(files, args)
    write_csv(completeness, tables_dir / "feature_completeness_summary.csv")
    write_csv(violations, tables_dir / "physical_limit_violations.csv")
    write_csv(outlier_summary, tables_dir / "feature_outlier_summary.csv")
    write_csv(outlier_events, tables_dir / "feature_outlier_events.csv")
    write_csv(target_bal, tables_dir / "target_balance_summary.csv")
    write_csv(pd.DataFrame(failures), tables_dir / "feature_file_read_failures.csv")

    if args.save_analysis_sample and not analysis.empty:
        write_csv(analysis.head(args.save_analysis_sample_rows), tables_dir / "analysis_sample.csv")

    corr_global, corr_phase, corr_ds_phase = compute_correlations(analysis, args)
    write_csv(corr_global, tables_dir / "feature_target_correlation_global.csv")
    write_csv(corr_phase, tables_dir / "feature_target_correlation_by_phase.csv")
    write_csv(corr_ds_phase, tables_dir / "feature_target_correlation_by_dataset_phase.csv")

    redundant = compute_redundancy(analysis, corr_global, args)
    write_csv(redundant, tables_dir / "redundant_feature_pairs.csv")

    domain_detail, domain_summary = compute_domain_shift(analysis, args)
    write_csv(domain_detail, tables_dir / "feature_domain_shift_detail.csv")
    write_csv(domain_summary, tables_dir / "feature_domain_shift_summary.csv")

    curve_valid = curve_feature_validity(analysis)
    write_csv(curve_valid, tables_dir / "curve_feature_validity_by_dataset_phase.csv")

    scorecard = feature_quality_scorecard(completeness, outlier_summary, violations, corr_global, domain_summary, redundant)
    write_csv(scorecard, tables_dir / "feature_quality_scorecard.csv")

    summary = {
        "generated_at": now_iso(),
        "features_root": str(args.features_root),
        "output_base": str(output_base),
        "feature_files": len(files),
        "inventory_rows": len(inventory),
        "analysis_rows_sampled": len(analysis),
        "completeness_rows": len(completeness),
        "physical_violation_rows": len(violations),
        "outlier_summary_rows": len(outlier_summary),
        "outlier_event_rows": len(outlier_events),
        "target_balance_rows": len(target_bal),
        "correlation_global_rows": len(corr_global),
        "correlation_phase_rows": len(corr_phase),
        "correlation_dataset_phase_rows": len(corr_ds_phase),
        "redundant_pairs": len(redundant),
        "domain_shift_detail_rows": len(domain_detail),
        "domain_shift_summary_rows": len(domain_summary),
        "curve_validity_rows": len(curve_valid),
        "scorecard_rows": len(scorecard),
        "feature_read_failures": len(failures),
        "target_column": choose_target_column(analysis.columns, args.target_column) if not analysis.empty else None,
        "target_norm_column": choose_norm_target_column(analysis.columns, args.target_norm_column) if not analysis.empty else None,
        **aux_summary,
    }
    write_json(summary, reports_dir / "feature_report_only_run_info.json")
    generate_markdown_report(tables_dir, reports_dir / "feature_report_only.md", summary)
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report-only audit for extracted battery features. No plots are generated.")
    p.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--output-subdir", type=Path, default=DEFAULT_OUTPUT_SUBDIR, help="Subdirectory below output-root for tables/reports.")
    p.add_argument("--capacity-life-summary", type=Path, default=None, help="Optional capacity_life_summary_v3.csv from feature extraction.")
    p.add_argument("--dataset-phase-summary", type=Path, default=None, help="Optional dataset_phase_target_summary.csv from feature extraction.")
    p.add_argument("--canonicalize-dataset-names", dest="canonicalize_dataset_names", action="store_true", default=True)
    p.add_argument("--no-canonicalize-dataset-names", dest="canonicalize_dataset_names", action="store_false")
    p.add_argument("--datasets", nargs="*", default=["all"])
    p.add_argument("--phases", nargs="*", default=list(PHASES))
    p.add_argument("--engine", default="fastparquet", choices=["fastparquet", "pyarrow"])
    p.add_argument("--target-column", default="auto")
    p.add_argument("--target-norm-column", default="auto")
    p.add_argument("--max-correlation-rows", type=int, default=250000, help="Max sampled rows used for correlations/redundancy/domain shift. 0 means all.")
    p.add_argument("--max-corr-features", type=int, default=0, help="Feature cap for target correlations. 0 means no cap.")
    p.add_argument("--max-redundancy-features-per-group", type=int, default=80)
    p.add_argument("--max-domain-features", type=int, default=400)
    p.add_argument("--redundancy-threshold", type=float, default=0.95)
    p.add_argument("--max-issue-rows-per-file", type=int, default=10000)
    p.add_argument("--mutual-info", action="store_true", help="Compute mutual_information_regression if sklearn is available.")
    p.add_argument("--save-analysis-sample", action="store_true")
    p.add_argument("--save-analysis-sample-rows", type=int, default=20000)
    p.add_argument("--random-seed", type=int, default=42)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.datasets = parse_dataset_args(args.datasets, canonicalize=args.canonicalize_dataset_names)
    args.phases = parse_phase_args(args.phases)
    args.output_root = Path(args.output_root)
    args.output_subdir = Path(args.output_subdir)

    if args.engine == "fastparquet" and fastparquet is None:
        print("[WARNING] fastparquet is not importable. Install it or run with --engine pyarrow.", file=sys.stderr)

    print("Feature report-only audit")
    print("- Features root:", args.features_root)
    print("- Output root:", args.output_root / args.output_subdir)
    print("- Datasets:", args.datasets or "all")
    print("- Phases:", args.phases)
    print("- No plots will be generated.")

    summary = run_report_only_audit(args)
    print("Done.")
    print("Main report:", args.output_root / args.output_subdir / "reports" / "feature_report_only.md")
    print("Summary:", json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
