"""
audit_features_general.py

General audit pipeline for cleaned lithium-ion battery cycles and hierarchical
feature-extraction outputs.

Typical use from the repository root:

    python src/validation/audit_features_general.py \
        --interim-root data/interim \
        --features-root data/processed/features_hierarchical_v3 \
        --output-root reports/audit \
        --datasets all \
        --phases charge discharge \
        --sample-raw-cycles 5 \
        --engine fastparquet \
        --mode all \
        --skip-existing

Modes:
    --mode plots      Run cleaned-cycle plots and signal-level summaries.
    --mode features   Run feature-table diagnostics and feature report.
    --mode report     Rebuild the final Markdown report from existing tables.
    --mode all        Run all audit stages.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

try:
    import fastparquet  # type: ignore  # noqa: F401
except Exception:
    fastparquet = None  # type: ignore

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:
    scipy_stats = None  # type: ignore

try:
    from sklearn.feature_selection import mutual_info_regression  # type: ignore
except Exception:
    mutual_info_regression = None  # type: ignore


# =============================================================================
# Defaults and constants
# =============================================================================

DEFAULT_INTERIM_ROOT = Path("data/interim")
DEFAULT_FEATURES_ROOT = Path("data/processed/features_hierarchical_v3")
DEFAULT_OUTPUT_ROOT = Path("reports/audit")

PHASES = ("charge", "discharge")

# Canonical dataset aliases used throughout reports. The adapters and legacy
# scripts may refer to Oxford as ORA, while extracted summaries often use
# OXFORD. Canonicalizing prevents duplicated report folders/tables.
DATASET_ALIASES = {
    "ORA": "OXFORD",
    "OXFORD": "OXFORD",
    "RW": "NRW",
    "NRW": "NRW",
    "NASA_RANDOM_WALK": "NRW",
    "NASA-RANDOM-WALK": "NRW",
    "NASA_RANDOMWALK": "NRW",
}
VECTOR_COLUMNS = ["time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah", "energy_wh", "power_w"]
MIN_CLEAN_COLUMNS = [
    "dataset", "battery_id", "cycle_number", "cycle_index", "cycle_life",
    "remaining_cycles", "phase", "time_s", "voltage_v", "current_a",
    "temperature_c", "capacity_ah",
]
OPTIONAL_CLEAN_COLUMNS = [
    "remaining_cycles_norm", "soh", "energy_wh", "power_w", "phase_capacity_ah",
    "phase_energy_wh", "source_file", "source_path", "source_cycle_id", "cycle_pair_id",
    "nominal_capacity_ah", "initial_capacity_ah", "nominal_voltage_v",
    "ambient_temperature_c", "chemistry", "form_factor", "manufacturer",
    "charge_protocol", "discharge_protocol", "c_rate_charge", "c_rate_discharge",
    "dod", "is_valid_cycle", "is_model_usable", "exclude_reason",
]
ID_LIKE_COLUMNS = {
    "dataset", "battery_id", "phase", "phase_folder", "source_file", "source_path",
    "source_cycle_id", "cycle_pair_id", "cell_id_original", "barcode", "policy_readable",
    "chemistry", "form_factor", "manufacturer", "charge_protocol", "discharge_protocol",
    "adapter_version", "cleaning_version", "created_at", "source_file_key", "source_parquet_file",
    "feature_level", "capacity_validation_status", "capacity_validation_notes",
}
TARGET_CANDIDATES = [
    "target_rul80_steps_clipped",
    "target_rul80_cycle_index_clipped",
    "target_rul80_steps",
    "target_rul80_cycle_index",
    "remaining_cycles",
    "remaining_cycles_norm",
    "input_remaining_cycles",
    "input_remaining_cycles_norm",
    "target_rul70_steps_clipped",
    "target_rul70_cycle_index_clipped",
    "target_rul70_steps",
    "target_rul70_cycle_index",
]
PRIMARY_TARGET_CANDIDATES = [
    "target_rul80_steps_clipped",
    "target_rul80_cycle_index_clipped",
    "target_rul80_steps",
    "target_rul80_cycle_index",
    "remaining_cycles",
    "input_remaining_cycles",
]
NORM_TARGET_CANDIDATES = [
    "remaining_cycles_norm",
    "input_remaining_cycles_norm",
]

PHYSICAL_RULES = {
    "voltage_min_v": (0.0, 6.0),
    "voltage_max_v": (0.0, 6.0),
    "voltage_start_v": (0.0, 6.0),
    "voltage_end_v": (0.0, 6.0),
    "voltage_min": (0.0, 6.0),
    "voltage_max": (0.0, 6.0),
    "voltage_first": (0.0, 6.0),
    "voltage_last": (0.0, 6.0),
    "capacity_min_ah": (-1e-9, 200.0),
    "capacity_max_ah": (-1e-9, 200.0),
    "phase_capacity_ah": (-1e-9, 200.0),
    "capacity_min": (-1e-9, 200.0),
    "capacity_max": (-1e-9, 200.0),
    "capacity_last": (-1e-9, 200.0),
    "capacity_delta": (-200.0, 200.0),
    "energy_min_wh": (-1e-9, 2000.0),
    "energy_max_wh": (-1e-9, 2000.0),
    "phase_energy_wh": (-1e-9, 2000.0),
    "energy_min": (-1e-9, 2000.0),
    "energy_max": (-1e-9, 2000.0),
    "energy_last": (-1e-9, 2000.0),
    "duration_s": (1e-12, 1e8),
    "time_vector_duration_s": (1e-12, 1e8),
    "sample_period_median_s": (1e-12, 1e7),
    "time_dt_median_s": (1e-12, 1e7),
    "n_points": (20.0, 1e8),
    "input_n_points": (20.0, 1e8),
    "temperature_min_c": (-40.0, 100.0),
    "temperature_max_c": (-40.0, 120.0),
    "temperature_mean_c": (-40.0, 100.0),
    "temperature_min": (-40.0, 100.0),
    "temperature_max": (-40.0, 120.0),
    "temperature_mean": (-40.0, 100.0),
    "target_capacity_ah": (0.0, 200.0),
    "target_initial_capacity_ah": (0.0, 200.0),
    "target_soh_recomputed": (0.0, 1.5),
    "target_soh_recomputed_clipped_0_1p2": (0.0, 1.2),
}


@dataclass
class CleanCycleFile:
    dataset: str
    phase: str
    path: Path
    layout: str


@dataclass
class FeatureFile:
    dataset: str
    phase: str
    path: Path
    level_hint: str


# =============================================================================
# Basic helpers
# =============================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_filename(value: Any, max_len: int = 160) -> str:
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value)).strip("_")
    return (text or "value")[:max_len]


def canonicalize_dataset_name(value: Any, enabled: bool = True) -> str:
    """Return a canonical dataset name used in paths, filters and reports."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "UNKNOWN"
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
    out = []
    for p in phases:
        p = p.lower()
        if p in PHASES:
            out.append(p)
    return list(dict.fromkeys(out)) or list(PHASES)


def infer_dataset_from_filename(path: Path) -> str:
    stem = path.stem.lower()
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
    """Read parquet with a fallback for problematic nullable columns."""
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
            for col in failed:
                df[col] = np.nan
        return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def close_fig(fig: plt.Figure) -> None:
    try:
        plt.close(fig)
    except Exception:
        pass


def save_fig(fig: plt.Figure, path: Path, dpi: int = 130) -> None:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    close_fig(fig)


def safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def finite_pct(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    s = safe_numeric_series(series) if not pd.api.types.is_numeric_dtype(series) else series.replace([np.inf, -np.inf], np.nan)
    return float(np.isfinite(s.to_numpy(dtype=float, na_value=np.nan)).mean() * 100.0)


# =============================================================================
# Vector parsing and reconstruction helpers
# =============================================================================

def parse_vector(value: Any) -> np.ndarray:
    """Convert a cell value into a 1D float numpy array."""
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
            parsed = ast.literal_eval(text)
            arr = np.asarray(parsed).ravel()
        except Exception:
            # Try permissive whitespace/comma-separated numeric parsing.
            text = text.strip("[]()")
            parts = re.split(r"[;,\s]+", text)
            vals: List[float] = []
            for p in parts:
                if not p:
                    continue
                try:
                    vals.append(float(p))
                except Exception:
                    vals.append(np.nan)
            arr = np.asarray(vals).ravel()
    else:
        try:
            arr = np.asarray(value).ravel()
        except Exception:
            return np.array([], dtype=float)
    try:
        return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
    except Exception:
        return np.array([], dtype=float)


def align_vectors(*arrays: np.ndarray) -> List[np.ndarray]:
    lens = [len(a) for a in arrays if isinstance(a, np.ndarray) and len(a) > 0]
    if not lens:
        return [np.array([], dtype=float) for _ in arrays]
    n = min(lens)
    return [a[:n] if len(a) >= n else np.full(n, np.nan) for a in arrays]


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    if y is None or x is None or len(y) < 2 or len(x) < 2:
        return np.nan
    y, x = align_vectors(y, x)
    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < 2:
        return np.nan
    yy = y[valid]
    xx = x[valid]
    order = np.argsort(xx, kind="mergesort")
    return float(np.trapz(yy[order], xx[order]))


def reconstruct_power_energy(row: pd.Series) -> Dict[str, np.ndarray]:
    t = parse_vector(row.get("time_s"))
    v = parse_vector(row.get("voltage_v"))
    i = parse_vector(row.get("current_a"))
    q = parse_vector(row.get("capacity_ah"))
    temp = parse_vector(row.get("temperature_c"))
    e = parse_vector(row.get("energy_wh")) if "energy_wh" in row.index else np.array([], dtype=float)
    p = parse_vector(row.get("power_w")) if "power_w" in row.index else np.array([], dtype=float)

    t, v, i, q = align_vectors(t, v, i, q)
    n = len(t)
    if len(temp) == 0 and n:
        temp = np.full(n, np.nan)
    elif n:
        temp = temp[:n] if len(temp) >= n else np.pad(temp, (0, n - len(temp)), constant_values=np.nan)

    if len(p) == 0 and n and len(v) == n and len(i) == n:
        p = v * i
    elif n:
        p = p[:n] if len(p) >= n else np.pad(p, (0, n - len(p)), constant_values=np.nan)

    if len(e) == 0 and n and len(p) == n:
        e = np.zeros(n, dtype=float)
        valid = np.isfinite(t) & np.isfinite(p)
        if n >= 2 and valid.sum() >= 2:
            dt = np.diff(t)
            pp = np.abs(p)
            step = 0.5 * (pp[:-1] + pp[1:]) * dt / 3600.0
            step[~np.isfinite(step)] = 0.0
            step[dt < 0] = 0.0
            e[1:] = np.cumsum(step)
        else:
            e[:] = np.nan
    elif n:
        e = e[:n] if len(e) >= n else np.pad(e, (0, n - len(e)), constant_values=np.nan)

    return {
        "time_s": t,
        "voltage_v": v,
        "current_a": i,
        "temperature_c": temp,
        "capacity_ah": q,
        "power_w": p,
        "energy_wh": e,
    }


def last_finite(arr: np.ndarray) -> float:
    if arr is None or len(arr) == 0:
        return np.nan
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.nan
    return float(valid[-1])


def first_finite(arr: np.ndarray) -> float:
    if arr is None or len(arr) == 0:
        return np.nan
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.nan
    return float(valid[0])


# =============================================================================
# Discovery
# =============================================================================

def discover_clean_cycle_files(interim_root: Path, datasets: Sequence[str], phases: Sequence[str]) -> List[CleanCycleFile]:
    wanted = {d.upper() for d in datasets if d}
    phase_set = {p.lower() for p in phases}
    files: List[CleanCycleFile] = []
    seen: set[str] = set()

    # Layout A: global phase folders: interim_root/charge/*.parquet
    for phase in phases:
        folder = interim_root / phase
        if folder.exists():
            for path in sorted(folder.glob("*.parquet"), key=lambda p: str(p).lower()):
                ds = infer_dataset_from_filename(path)
                if wanted and ds.upper() not in wanted:
                    continue
                key = str(path.resolve()).lower()
                if key not in seen:
                    files.append(CleanCycleFile(dataset=canonicalize_dataset_name(ds, enabled=True), phase=phase, path=path, layout="global_phase_folder"))
                    seen.add(key)

    # Layout B: per-dataset clean folders: interim_root/<dataset>/clean/*_{phase}_clean_cycles.parquet
    for clean_dir in sorted(interim_root.glob("*/clean"), key=lambda p: str(p).lower()):
        ds_default = canonicalize_dataset_name(clean_dir.parent.name, enabled=True)
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
                files.append(CleanCycleFile(dataset=canonicalize_dataset_name(ds, enabled=True), phase=phase, path=path, layout="dataset_clean_folder"))
                seen.add(key)

    return files


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
        # Prefer dataset folder when output follows features_root/phase/dataset/file.features.parquet
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


# =============================================================================
# Feature grouping and targets
# =============================================================================

def classify_feature_group(name: str) -> str:
    s = name.lower()
    if s in ID_LIKE_COLUMNS or s.endswith("_id") or "source" in s:
        return "metadata"
    if s in {"remaining_cycles", "remaining_cycles_norm", "cycle_life"} or s.startswith("target_") or s.startswith("input_remaining"):
        return "target"
    if "capacity" in s or re.search(r"(^|_)q($|_)", s) or s.startswith("q_"):
        if s.startswith("qv") or "_qv" in s or "qv_" in s:
            return "QV_curve"
        return "capacity"
    if "energy" in s or s.startswith("e_") or "_wh" in s:
        return "energy"
    if "voltage" in s or "_v" in s:
        if s.startswith("vq") or "_vq" in s or "vq_" in s:
            return "VQ_curve"
        return "voltage"
    if "current" in s or "_a" in s or "c_rate" in s:
        return "current"
    if "temperature" in s or "temp" in s:
        return "temperature"
    if "power" in s or "_w" in s:
        return "power"
    if "time" in s or "duration" in s or "sample_period" in s or "dt" in s:
        if "gap" in s or "relax" in s:
            return "time_gap_relaxation"
        return "sampling_time"
    if "qv" in s:
        return "QV_curve"
    if "vq" in s:
        return "VQ_curve"
    if "stage" in s or "segment" in s or s.startswith("seg"):
        return "stage_structure"
    if "gap" in s or "relax" in s:
        return "time_gap_relaxation"
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


def choose_target_column(columns: Sequence[str], requested: str = "auto") -> Optional[str]:
    if requested and requested.lower() != "auto":
        return requested if requested in columns else None
    for col in PRIMARY_TARGET_CANDIDATES:
        if col in columns:
            return col
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


# =============================================================================
# Clean-cycle plotting and summaries
# =============================================================================

def select_sampled_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.iloc[[]].copy()
    work = df.sort_values("cycle_number" if "cycle_number" in df.columns else df.index.name or df.columns[0]).reset_index(drop=True)
    m = len(work)
    if m <= n:
        idx = np.arange(m)
    else:
        idx = np.linspace(0, m - 1, n).round().astype(int)
        idx = np.unique(idx)
    return work.iloc[idx].copy()


def plot_sampled_signal(rows: pd.DataFrame, signal_col: str, y_label: str, out_path: Path, dpi: int, plot_format: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    plotted = 0
    for _, row in rows.iterrows():
        vecs = reconstruct_power_energy(row)
        t = vecs["time_s"]
        y = vecs.get(signal_col, np.array([], dtype=float))
        if len(t) == 0 or len(y) == 0:
            continue
        t, y = align_vectors(t, y)
        valid = np.isfinite(t) & np.isfinite(y)
        if valid.sum() < 2:
            continue
        label = f"cycle {row.get('cycle_number', '?')}"
        ax.plot(t[valid], y[valid], linewidth=1.0, alpha=0.85, label=label)
        plotted += 1
    ax.set_xlabel("time_s")
    ax.set_ylabel(y_label)
    title_bits = []
    for col in ["dataset", "battery_id", "phase"]:
        if col in rows.columns and len(rows):
            title_bits.append(str(rows.iloc[0].get(col)))
    ax.set_title(f"{signal_col} vs time_s — {' | '.join(title_bits)} — sampled={plotted}")
    if plotted <= 8:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    save_fig(fig, out_path.with_suffix(f".{plot_format}"), dpi=dpi)


def plot_raw_grid(rows: pd.DataFrame, out_path: Path, dpi: int, plot_format: str) -> None:
    signals = [
        ("voltage_v", "Voltage (V)"),
        ("current_a", "Current (A)"),
        ("capacity_ah", "Capacity (Ah)"),
        ("temperature_c", "Temperature (C)"),
        ("energy_wh", "Energy (Wh)"),
        ("power_w", "Power (W)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    axes = axes.ravel()
    for ax, (sig, ylabel) in zip(axes, signals):
        plotted = 0
        for _, row in rows.iterrows():
            vecs = reconstruct_power_energy(row)
            t = vecs["time_s"]
            y = vecs[sig]
            if len(t) == 0 or len(y) == 0:
                continue
            t, y = align_vectors(t, y)
            valid = np.isfinite(t) & np.isfinite(y)
            if valid.sum() < 2:
                continue
            ax.plot(t[valid], y[valid], linewidth=0.9, alpha=0.85, label=f"cycle {row.get('cycle_number', '?')}")
            plotted += 1
        ax.set_xlabel("time_s")
        ax.set_ylabel(ylabel)
        ax.set_title(sig)
        ax.grid(True, alpha=0.25)
    if len(rows) <= 6 and len(rows) > 0:
        axes[0].legend(fontsize=7)
    title_bits = []
    for col in ["dataset", "battery_id", "phase"]:
        if col in rows.columns and len(rows):
            title_bits.append(str(rows.iloc[0].get(col)))
    fig.suptitle(f"Raw signals sampled — {' | '.join(title_bits)}", y=1.005, fontsize=13)
    save_fig(fig, out_path.with_suffix(f".{plot_format}"), dpi=dpi)


def plot_vq_all_cycles(group: pd.DataFrame, out_path: Path, dpi: int, plot_format: str, max_curves: int = 0) -> None:
    if group.empty:
        return
    work = group.sort_values("cycle_number" if "cycle_number" in group.columns else group.index.name or group.columns[0]).reset_index(drop=True)
    if max_curves and len(work) > max_curves:
        idx = np.linspace(0, len(work) - 1, max_curves).round().astype(int)
        work = work.iloc[np.unique(idx)].copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    first_mid_last = set()
    if len(work) > 0:
        first_mid_last = {0, len(work) // 2, len(work) - 1}
    plotted = 0
    for local_idx, (_, row) in enumerate(work.iterrows()):
        vecs = reconstruct_power_energy(row)
        q = vecs["capacity_ah"]
        v = vecs["voltage_v"]
        q, v = align_vectors(q, v)
        valid = np.isfinite(q) & np.isfinite(v)
        if valid.sum() < 2:
            continue
        lw = 1.4 if local_idx in first_mid_last else 0.45
        alpha = 0.95 if local_idx in first_mid_last else 0.15
        label = f"cycle {row.get('cycle_number', '?')}" if local_idx in first_mid_last else None
        ax.plot(q[valid], v[valid], linewidth=lw, alpha=alpha, label=label)
        plotted += 1
    ax.set_xlabel("capacity_ah")
    ax.set_ylabel("voltage_v")
    title = f"V(Q) all cycles — {group.iloc[0].get('dataset')} | {group.iloc[0].get('battery_id')} | {group.iloc[0].get('phase')} | n={plotted}"
    if max_curves and len(group) > max_curves:
        title += f" (sampled from {len(group)})"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=7)
    save_fig(fig, out_path.with_suffix(f".{plot_format}"), dpi=dpi)


def plot_eq_all_cycles(group: pd.DataFrame, out_path: Path, dpi: int, plot_format: str, max_curves: int = 0) -> None:
    if group.empty:
        return
    work = group.sort_values("cycle_number" if "cycle_number" in group.columns else group.index.name or group.columns[0]).reset_index(drop=True)
    if max_curves and len(work) > max_curves:
        idx = np.linspace(0, len(work) - 1, max_curves).round().astype(int)
        work = work.iloc[np.unique(idx)].copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    first_mid_last = {0, len(work) // 2, len(work) - 1} if len(work) else set()
    plotted = 0
    for local_idx, (_, row) in enumerate(work.iterrows()):
        vecs = reconstruct_power_energy(row)
        q = vecs["capacity_ah"]
        e = vecs["energy_wh"]
        q, e = align_vectors(q, e)
        valid = np.isfinite(q) & np.isfinite(e)
        if valid.sum() < 2:
            continue
        lw = 1.4 if local_idx in first_mid_last else 0.45
        alpha = 0.95 if local_idx in first_mid_last else 0.15
        label = f"cycle {row.get('cycle_number', '?')}" if local_idx in first_mid_last else None
        ax.plot(q[valid], e[valid], linewidth=lw, alpha=alpha, label=label)
        plotted += 1
    ax.set_xlabel("capacity_ah")
    ax.set_ylabel("energy_wh")
    title = f"E(Q) all cycles — {group.iloc[0].get('dataset')} | {group.iloc[0].get('battery_id')} | {group.iloc[0].get('phase')} | n={plotted}"
    if max_curves and len(group) > max_curves:
        title += f" (sampled from {len(group)})"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=7)
    save_fig(fig, out_path.with_suffix(f".{plot_format}"), dpi=dpi)


def add_cycle_totals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "phase_capacity_ah" not in out.columns:
        out["phase_capacity_ah"] = np.nan
    if "phase_energy_wh" not in out.columns:
        out["phase_energy_wh"] = np.nan
    missing_cap = ~np.isfinite(pd.to_numeric(out["phase_capacity_ah"], errors="coerce"))
    missing_ene = ~np.isfinite(pd.to_numeric(out["phase_energy_wh"], errors="coerce"))
    if missing_cap.any() or missing_ene.any():
        caps: List[float] = []
        enes: List[float] = []
        for _, row in out.iterrows():
            vecs = reconstruct_power_energy(row)
            caps.append(last_finite(vecs["capacity_ah"]))
            enes.append(last_finite(vecs["energy_wh"]))
        cap_series = pd.Series(caps, index=out.index)
        ene_series = pd.Series(enes, index=out.index)
        out.loc[missing_cap, "phase_capacity_ah"] = cap_series.loc[missing_cap]
        out.loc[missing_ene, "phase_energy_wh"] = ene_series.loc[missing_ene]
    return out


def plot_degradation_battery(group: pd.DataFrame, out_dir: Path, dpi: int, plot_format: str) -> None:
    if group.empty:
        return
    work = add_cycle_totals(group).sort_values("cycle_number").copy()
    phase = str(work.iloc[0].get("phase", "unknown"))
    cycle = safe_numeric_series(work["cycle_number"])
    cap = safe_numeric_series(work["phase_capacity_ah"])
    ene = safe_numeric_series(work["phase_energy_wh"])

    # Capacity
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    valid = np.isfinite(cycle) & np.isfinite(cap)
    ax.plot(cycle[valid], cap[valid], marker=".", linewidth=1.0)
    ax.set_xlabel("cycle_number")
    ax.set_ylabel("phase_capacity_ah")
    ax.set_title(f"Capacity vs cycle — {work.iloc[0].get('dataset')} | {work.iloc[0].get('battery_id')} | {phase}")
    ax.grid(True, alpha=0.25)
    save_fig(fig, out_dir / f"capacity_vs_cycle_{phase}.{plot_format}", dpi=dpi)

    # Energy
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    valid = np.isfinite(cycle) & np.isfinite(ene)
    ax.plot(cycle[valid], ene[valid], marker=".", linewidth=1.0)
    ax.set_xlabel("cycle_number")
    ax.set_ylabel("phase_energy_wh")
    ax.set_title(f"Energy vs cycle — {work.iloc[0].get('dataset')} | {work.iloc[0].get('battery_id')} | {phase}")
    ax.grid(True, alpha=0.25)
    save_fig(fig, out_dir / f"energy_vs_cycle_{phase}.{plot_format}", dpi=dpi)

    # Normalized comparison
    cap0 = first_finite(cap.to_numpy(dtype=float))
    ene0 = first_finite(ene.to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if np.isfinite(cap0) and cap0 != 0:
        ax.plot(cycle, cap / cap0, marker=".", linewidth=1.0, label="capacity_norm")
    if np.isfinite(ene0) and ene0 != 0:
        ax.plot(cycle, ene / ene0, marker=".", linewidth=1.0, label="energy_norm")
    ax.set_xlabel("cycle_number")
    ax.set_ylabel("normalized value")
    ax.set_title(f"Capacity and energy normalized — {work.iloc[0].get('dataset')} | {work.iloc[0].get('battery_id')} | {phase}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    save_fig(fig, out_dir / f"capacity_energy_normalized_vs_cycle_{phase}.{plot_format}", dpi=dpi)


def plot_degradation_dataset(df: pd.DataFrame, dataset: str, phase: str, out_dir: Path, dpi: int, plot_format: str) -> None:
    if df.empty:
        return
    work = add_cycle_totals(df).sort_values(["battery_id", "cycle_number"]).copy()
    # Capacity all batteries
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for bid, g in work.groupby("battery_id", sort=False):
        c = safe_numeric_series(g["cycle_number"])
        y = safe_numeric_series(g["phase_capacity_ah"])
        valid = np.isfinite(c) & np.isfinite(y)
        if valid.sum() >= 2:
            ax.plot(c[valid], y[valid], linewidth=0.8, alpha=0.45)
    ax.set_xlabel("cycle_number")
    ax.set_ylabel("phase_capacity_ah")
    ax.set_title(f"Capacity degradation — all batteries — {dataset} | {phase}")
    ax.grid(True, alpha=0.25)
    save_fig(fig, out_dir / f"capacity_all_batteries_{phase}.{plot_format}", dpi=dpi)

    # Energy all batteries
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for bid, g in work.groupby("battery_id", sort=False):
        c = safe_numeric_series(g["cycle_number"])
        y = safe_numeric_series(g["phase_energy_wh"])
        valid = np.isfinite(c) & np.isfinite(y)
        if valid.sum() >= 2:
            ax.plot(c[valid], y[valid], linewidth=0.8, alpha=0.45)
    ax.set_xlabel("cycle_number")
    ax.set_ylabel("phase_energy_wh")
    ax.set_title(f"Energy degradation — all batteries — {dataset} | {phase}")
    ax.grid(True, alpha=0.25)
    save_fig(fig, out_dir / f"energy_all_batteries_{phase}.{plot_format}", dpi=dpi)

    # Normalized all batteries
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for bid, g in work.groupby("battery_id", sort=False):
        c = safe_numeric_series(g["cycle_number"])
        cap = safe_numeric_series(g["phase_capacity_ah"])
        ene = safe_numeric_series(g["phase_energy_wh"])
        cap0 = first_finite(cap.to_numpy(dtype=float))
        ene0 = first_finite(ene.to_numpy(dtype=float))
        if np.isfinite(cap0) and cap0 != 0:
            ax.plot(c, cap / cap0, linewidth=0.8, alpha=0.35)
        if np.isfinite(ene0) and ene0 != 0:
            ax.plot(c, ene / ene0, linewidth=0.8, alpha=0.18, linestyle="--")
    ax.set_xlabel("cycle_number")
    ax.set_ylabel("normalized capacity / energy")
    ax.set_title(f"Capacity and energy normalized — all batteries — {dataset} | {phase}")
    ax.grid(True, alpha=0.25)
    save_fig(fig, out_dir / f"capacity_energy_norm_all_batteries_{phase}.{plot_format}", dpi=dpi)


def audit_clean_cycle_plots(files: Sequence[CleanCycleFile], args: argparse.Namespace) -> pd.DataFrame:
    sampled_records: List[Dict[str, Any]] = []
    input_summaries: List[Dict[str, Any]] = []

    columns_needed = list(dict.fromkeys(MIN_CLEAN_COLUMNS + OPTIONAL_CLEAN_COLUMNS))

    for item in tqdm(files, desc="Clean cycle plot audit", unit="file"):
        available = read_parquet_columns(item.path, args.engine)
        cols = [c for c in columns_needed if c in available]
        missing_min = [c for c in MIN_CLEAN_COLUMNS if c not in available]
        input_summaries.append({
            "dataset": canonicalize_dataset_name(item.dataset, enabled=getattr(args, "canonicalize_dataset_names", True)),
            "phase": item.phase,
            "path": str(item.path),
            "layout": item.layout,
            "row_count": parquet_row_count(item.path, args.engine),
            "missing_minimum_columns": ";".join(missing_min),
            "has_energy_wh": "energy_wh" in available,
            "has_power_w": "power_w" in available,
        })
        if missing_min:
            continue
        try:
            df = robust_read_parquet(item.path, columns=cols, engine=args.engine)
        except Exception as exc:
            input_summaries[-1]["read_error"] = repr(exc)
            continue

        if "dataset" not in df.columns:
            df["dataset"] = item.dataset
        if "phase" not in df.columns:
            df["phase"] = item.phase
        df = canonicalize_dataset_column(df, enabled=getattr(args, "canonicalize_dataset_names", True))
        if getattr(args, "priority_warnings_only", False):
            keys = getattr(args, "critical_battery_phase_keys", set())
            if keys:
                key_series = list(zip(
                    df["dataset"].map(lambda x: canonicalize_dataset_name(x, enabled=True)).astype(str),
                    df["battery_id"].astype(str),
                    df["phase"].astype(str).str.lower(),
                ))
                mask = pd.Series([k in keys for k in key_series], index=df.index)
                df = df[mask].copy()
            else:
                df = df.iloc[[]].copy()
        if args.max_batteries and args.max_batteries > 0:
            batteries = list(dict.fromkeys(df["battery_id"].astype(str).tolist()))[: args.max_batteries]
            df = df[df["battery_id"].astype(str).isin(batteries)].copy()

        for (dataset, battery_id, phase), group in tqdm(
            df.groupby(["dataset", "battery_id", "phase"], dropna=False, sort=False),
            desc=f"Plots {item.dataset}/{item.phase}",
            leave=False,
            unit="battery",
        ):
            dataset_s = canonicalize_dataset_name(dataset, enabled=getattr(args, "canonicalize_dataset_names", True))
            battery_s = str(battery_id)
            phase_s = str(phase).lower()
            sample = select_sampled_rows(group, args.sample_raw_cycles)
            sample = sample.reset_index(drop=True)
            for pos, (_, row) in enumerate(sample.iterrows(), start=1):
                sampled_records.append({
                    "dataset": dataset_s,
                    "battery_id": battery_s,
                    "phase": phase_s,
                    "cycle_number": row.get("cycle_number", np.nan),
                    "cycle_index": row.get("cycle_index", np.nan),
                    "cycle_life": row.get("cycle_life", np.nan),
                    "remaining_cycles": row.get("remaining_cycles", np.nan),
                    "remaining_cycles_norm": row.get("remaining_cycles_norm", np.nan),
                    "selection_position": pos,
                    "n_cycles_in_group": len(group),
                    "source_file": str(item.path),
                })

            raw_dir = args.output_root / dataset_s / "raw_curves" / safe_filename(battery_s) / phase_s
            vq_dir = args.output_root / dataset_s / "VQ_curves" / safe_filename(battery_s) / phase_s
            eq_dir = args.output_root / dataset_s / "EQ_curves" / safe_filename(battery_s) / phase_s
            deg_dir = args.output_root / dataset_s / "degradation" / safe_filename(battery_s)

            if args.mode in {"all", "plots"}:
                signal_specs = [
                    ("voltage_v", "Voltage (V)", raw_dir / "voltage_vs_time_sampled"),
                    ("current_a", "Current (A)", raw_dir / "current_vs_time_sampled"),
                    ("capacity_ah", "Capacity (Ah)", raw_dir / "capacity_vs_time_sampled"),
                    ("temperature_c", "Temperature (C)", raw_dir / "temperature_vs_time_sampled"),
                    ("energy_wh", "Energy (Wh)", raw_dir / "energy_vs_time_sampled"),
                    ("power_w", "Power (W)", raw_dir / "power_vs_time_sampled"),
                ]
                for sig, ylabel, out_stem in signal_specs:
                    out_path = out_stem.with_suffix(f".{args.plot_format}")
                    if args.skip_existing and out_path.exists() and not args.force:
                        continue
                    plot_sampled_signal(sample, sig, ylabel, out_stem, dpi=args.dpi, plot_format=args.plot_format)

                grid_out = raw_dir / f"raw_signals_sampled_grid.{args.plot_format}"
                if not (args.skip_existing and grid_out.exists() and not args.force):
                    plot_raw_grid(sample, raw_dir / "raw_signals_sampled_grid", dpi=args.dpi, plot_format=args.plot_format)

                vq_out = vq_dir / f"VQ_all_cycles.{args.plot_format}"
                if not (args.skip_existing and vq_out.exists() and not args.force):
                    plot_vq_all_cycles(group, vq_dir / "VQ_all_cycles", dpi=args.dpi, plot_format=args.plot_format, max_curves=args.max_curves_per_battery_phase)

                eq_out = eq_dir / f"EQ_all_cycles.{args.plot_format}"
                if not (args.skip_existing and eq_out.exists() and not args.force):
                    plot_eq_all_cycles(group, eq_dir / "EQ_all_cycles", dpi=args.dpi, plot_format=args.plot_format, max_curves=args.max_curves_per_battery_phase)

                cap_out = deg_dir / f"capacity_vs_cycle_{phase_s}.{args.plot_format}"
                if not (args.skip_existing and cap_out.exists() and not args.force):
                    plot_degradation_battery(group, deg_dir, dpi=args.dpi, plot_format=args.plot_format)

        # Dataset aggregate plots
        if not df.empty and args.mode in {"all", "plots"}:
            for (dataset, phase), g in df.groupby(["dataset", "phase"], dropna=False, sort=False):
                dataset_s = canonicalize_dataset_name(dataset, enabled=getattr(args, "canonicalize_dataset_names", True))
                phase_s = str(phase).lower()
                out_dir = args.output_root / dataset_s / "degradation"
                cap_out = out_dir / f"capacity_all_batteries_{phase_s}.{args.plot_format}"
                if not (args.skip_existing and cap_out.exists() and not args.force):
                    plot_degradation_dataset(g, dataset_s, phase_s, out_dir, dpi=args.dpi, plot_format=args.plot_format)

    sampled_df = pd.DataFrame(sampled_records)
    summaries_df = pd.DataFrame(input_summaries)
    write_csv(sampled_df, args.output_root / "global" / "tables" / "sampled_raw_curve_cycles.csv")
    write_csv(summaries_df, args.output_root / "global" / "tables" / "input_cycle_files_summary.csv")
    return summaries_df


# =============================================================================
# Feature diagnostics
# =============================================================================

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
                "dataset": canonicalize_dataset_name(item.dataset, enabled=True),
                "n_rows_file": n_rows,
                "is_numeric": None,
                "is_identifier": is_identifier_or_text(col),
                "is_target": group == "target",
                "is_quality_flag": group == "quality_flags",
            })
    return pd.DataFrame(records)


def infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols: List[str] = []
    for col in df.columns:
        if is_identifier_or_text(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            numeric_cols.append(col)
        else:
            # Conservative attempt: only accept if at least a few values convert.
            s = pd.to_numeric(df[col].head(1000), errors="coerce")
            if s.notna().sum() >= max(3, int(0.1 * min(len(s), 1000))):
                numeric_cols.append(col)
    return numeric_cols


def completeness_for_df(df: pd.DataFrame, dataset: str, phase: str, file_path: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    numeric_cols = infer_numeric_columns(df)
    for col in numeric_cols:
        s = safe_numeric_series(df[col])
        n = len(s)
        n_non_nan = int(np.isfinite(s.to_numpy(dtype=float, na_value=np.nan)).sum())
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


def physical_violations_for_df(df: pd.DataFrame, dataset: str, phase: str, file_path: Path, max_records_per_file: int = 100000) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    id_cols = [c for c in ["dataset", "phase", "battery_id", "cycle_number", "cycle_index"] if c in df.columns]
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
    # Current sign convention if current_mean exists.
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


def outliers_for_df(df: pd.DataFrame, dataset: str, phase: str, file_path: Path, max_records_per_file: int = 200000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_records: List[Dict[str, Any]] = []
    cycle_records: List[Dict[str, Any]] = []
    numeric_cols = infer_numeric_columns(df)
    excluded = set(TARGET_CANDIDATES) | {"cycle_number", "cycle_index", "source_row_id", "stage_index", "stage_total"}
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


def target_balance(df: pd.DataFrame, dataset: str, phase: str, target_norm: Optional[str]) -> pd.DataFrame:
    if not target_norm or target_norm not in df.columns:
        return pd.DataFrame()
    s = safe_numeric_series(df[target_norm])
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
            # Read full file for per-file diagnostics. These are tabular features, not vector raw signals.
            df = robust_read_parquet(item.path, columns=cols, engine=args.engine)
            if "dataset" not in df.columns:
                df["dataset"] = item.dataset
            if "phase" not in df.columns:
                df["phase"] = item.phase
            df = canonicalize_dataset_column(df, enabled=getattr(args, "canonicalize_dataset_names", True))
            if "phase_folder" not in df.columns:
                df["phase_folder"] = item.phase
            target_col = choose_target_column(df.columns, args.target_column)
            target_norm_col = choose_norm_target_column(df.columns, args.target_norm_column)

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

            # Build sampled analysis frame for global correlations/redundancy/domain shift.
            numeric_cols = infer_numeric_columns(df)
            keep_cols = [c for c in ["dataset", "phase", "phase_folder", "battery_id", "cycle_number", "cycle_index"] if c in df.columns]
            if target_col and target_col not in keep_cols:
                keep_cols.append(target_col)
            if target_norm_col and target_norm_col not in keep_cols:
                keep_cols.append(target_norm_col)
            feature_cols = [c for c in numeric_cols if c not in keep_cols and classify_feature_group(c) not in {"metadata"}]
            # Drop all-vector/object columns that slipped through.
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


def correlations_for_scope(df: pd.DataFrame, scope_name: str, target_col: str, target_norm_col: Optional[str], max_features: int = 0) -> pd.DataFrame:
    if df.empty or target_col not in df.columns:
        return pd.DataFrame()
    numeric_cols = infer_numeric_columns(df)
    excluded = set([target_col]) | set(TARGET_CANDIDATES) | {"cycle_number", "cycle_index", "source_row_id", "stage_index", "stage_total"}
    features = [c for c in numeric_cols if c not in excluded and classify_feature_group(c) not in {"metadata", "target"}]
    if max_features and len(features) > max_features:
        # Keep more complete, more variable features first.
        scores = []
        for c in features:
            s = safe_numeric_series(df[c])
            scores.append((c, s.notna().mean(), s.nunique()))
        features = [c for c, _, _ in sorted(scores, key=lambda t: (t[1], t[2]), reverse=True)[:max_features]]
    records: List[Dict[str, Any]] = []
    y = df[target_col]
    y_norm = df[target_norm_col] if target_norm_col and target_norm_col in df.columns else None
    mi_values: Dict[str, float] = {}
    if args_global_mutual_info_enabled() and mutual_info_regression is not None and len(features) > 0:
        try:
            # MI only for rows with target and a manageable feature set.
            sub_cols = [target_col] + features
            sub = df[sub_cols].copy()
            for c in features:
                sub[c] = safe_numeric_series(sub[c])
            sub[target_col] = safe_numeric_series(sub[target_col])
            sub = sub.dropna()
            if len(sub) >= 20:
                X = sub[features].to_numpy(dtype=float)
                yy = sub[target_col].to_numpy(dtype=float)
                mi = mutual_info_regression(X, yy, random_state=42)
                mi_values = {c: float(v) for c, v in zip(features, mi)}
        except Exception:
            mi_values = {}

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


def args_global_mutual_info_enabled() -> bool:
    # This indirection avoids passing args into deeply nested calls in this script.
    return getattr(args_global_mutual_info_enabled, "enabled", False)


def compute_correlations(analysis: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if analysis.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    target_col = choose_target_column(analysis.columns, args.target_column)
    target_norm_col = choose_norm_target_column(analysis.columns, args.target_norm_column)
    if not target_col:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    setattr(args_global_mutual_info_enabled, "enabled", args.mutual_info)

    corr_global = correlations_for_scope(analysis, "global", target_col, target_norm_col, max_features=args.max_corr_features)

    phase_frames: List[pd.DataFrame] = []
    if "phase" in analysis.columns:
        for phase, g in analysis.groupby("phase", dropna=False):
            c = correlations_for_scope(g, f"phase={phase}", target_col, target_norm_col, max_features=args.max_corr_features)
            if not c.empty:
                c["phase"] = phase
                phase_frames.append(c)
    corr_phase = pd.concat(phase_frames, ignore_index=True) if phase_frames else pd.DataFrame()

    ds_phase_frames: List[pd.DataFrame] = []
    if "dataset" in analysis.columns and "phase" in analysis.columns:
        for (dataset, phase), g in analysis.groupby(["dataset", "phase"], dropna=False):
            c = correlations_for_scope(g, f"dataset={dataset};phase={phase}", target_col, target_norm_col, max_features=args.max_corr_features)
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
    target_cols = set(TARGET_CANDIDATES) | {"cycle_number", "cycle_index", "source_row_id", "stage_index", "stage_total"}
    features = [c for c in numeric_cols if c not in target_cols and classify_feature_group(c) not in {"metadata", "target"}]
    target_corr = {}
    if corr_global is not None and not corr_global.empty:
        target_corr = dict(zip(corr_global["feature_name"], corr_global["spearman_abs"]))
    records: List[Dict[str, Any]] = []
    for group_name in sorted({classify_feature_group(c) for c in features}):
        group_features = [c for c in features if classify_feature_group(c) == group_name]
        if len(group_features) < 2:
            continue
        if len(group_features) > args.max_redundancy_features_per_group:
            # Keep features most associated with the target and most complete.
            scores = []
            for c in group_features:
                s = safe_numeric_series(analysis[c])
                scores.append((c, target_corr.get(c, 0.0) if np.isfinite(target_corr.get(c, np.nan)) else 0.0, s.notna().mean()))
            group_features = [c for c, _, _ in sorted(scores, key=lambda t: (t[1], t[2]), reverse=True)[: args.max_redundancy_features_per_group]]
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


def compute_domain_shift(analysis: pd.DataFrame, outlier_summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if analysis.empty or "dataset" not in analysis.columns or "phase" not in analysis.columns:
        return pd.DataFrame()
    numeric_cols = infer_numeric_columns(analysis)
    excluded = set(TARGET_CANDIDATES) | {"cycle_number", "cycle_index", "source_row_id", "stage_index", "stage_total"}
    features = [c for c in numeric_cols if c not in excluded and classify_feature_group(c) not in {"metadata", "target"}]
    if len(features) > args.max_domain_features:
        # Use complete, variable features.
        scores = []
        for c in features:
            s = safe_numeric_series(analysis[c])
            scores.append((c, s.notna().mean(), s.nunique()))
        features = [c for c, _, _ in sorted(scores, key=lambda t: (t[1], t[2]), reverse=True)[: args.max_domain_features]]
    records: List[Dict[str, Any]] = []
    for feat in features:
        for (dataset, phase), g in analysis.groupby(["dataset", "phase"], dropna=False):
            s = safe_numeric_series(g[feat]).dropna()
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
                "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
                "min": float(s.min()),
                "max": float(s.max()),
                "nan_pct": float(safe_numeric_series(g[feat]).isna().mean() * 100.0),
            })
    summary = pd.DataFrame(records)
    if summary.empty:
        return summary
    # Domain shift score by relative spread of medians across dataset-phase groups.
    scores = []
    for feat, g in summary.groupby("feature_name"):
        med = g["median"].dropna()
        if len(med) < 2:
            score = np.nan
        else:
            denom = abs(float(med.median())) + 1e-12
            score = float((med.max() - med.min()) / denom)
        scores.append({"feature_name": feat, "domain_shift_score": score})
    scores_df = pd.DataFrame(scores)
    return summary.merge(scores_df, on="feature_name", how="left")


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
            records.append({**base, "curve_valid_column": col, "n_rows": len(g), "valid_pct": float((s > 0).mean() * 100.0), "nan_pct": float(s.isna().mean() * 100.0)})
    return pd.DataFrame(records)


def feature_quality_scorecard(
    completeness: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    violations: pd.DataFrame,
    corr_global: pd.DataFrame,
    domain_shift: pd.DataFrame,
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
        corr = corr_global[["feature_name", "spearman_abs", "spearman_norm_abs"]].copy()
        corr = corr.groupby("feature_name", dropna=False).max().reset_index()
        corr = corr.rename(columns={"spearman_abs": "abs_spearman_remaining_cycles", "spearman_norm_abs": "abs_spearman_remaining_cycles_norm"})
        comp = comp.merge(corr, on="feature_name", how="left")
    else:
        comp["abs_spearman_remaining_cycles"] = np.nan
        comp["abs_spearman_remaining_cycles_norm"] = np.nan
    if not domain_shift.empty:
        ds = domain_shift.groupby("feature_name", dropna=False).agg(domain_shift_score=("domain_shift_score", "max")).reset_index()
        comp = comp.merge(ds, on="feature_name", how="left")
    else:
        comp["domain_shift_score"] = np.nan
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
        corr = r.get("abs_spearman_remaining_cycles", np.nan)
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
    return comp


# =============================================================================
# Plotting feature diagnostics
# =============================================================================

def plot_barh(df: pd.DataFrame, value_col: str, label_col: str, title: str, out_path: Path, dpi: int, plot_format: str, top_n: int = 30) -> None:
    if df.empty or value_col not in df.columns or label_col not in df.columns:
        return
    work = df[[label_col, value_col]].dropna().copy()
    if work.empty:
        return
    work = work.sort_values(value_col, ascending=True).tail(top_n)
    fig_h = max(4.0, min(12.0, 0.28 * len(work) + 1.5))
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    ax.barh(work[label_col].astype(str), work[value_col])
    ax.set_xlabel(value_col)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    save_fig(fig, out_path.with_suffix(f".{plot_format}"), dpi=dpi)


def plot_completeness(completeness: pd.DataFrame, args: argparse.Namespace) -> None:
    if completeness.empty:
        return
    out_dir = args.output_root / "global" / "plots" / "completeness"
    for group, g in completeness.groupby("feature_group"):
        if group in {"metadata", "target", "other"} and len(g) > 100:
            continue
        agg = g.groupby("feature_name", dropna=False)["finite_pct"].mean().reset_index()
        plot_barh(agg, "finite_pct", "feature_name", f"Feature completeness — {group}", out_dir / f"completeness_{safe_filename(group)}", args.dpi, args.plot_format, top_n=args.n_top_features)


def plot_physical_limits(analysis: pd.DataFrame, args: argparse.Namespace) -> None:
    if analysis.empty:
        return
    out_dir = args.output_root / "global" / "plots" / "physical_limits"
    groups = {
        "voltage": [c for c in analysis.columns if any(k in c.lower() for k in ["voltage_min", "voltage_max", "voltage_mean", "voltage_first", "voltage_last"])],
        "capacity": [c for c in analysis.columns if any(k in c.lower() for k in ["capacity_last", "capacity_max", "phase_capacity", "target_capacity"])],
        "energy": [c for c in analysis.columns if any(k in c.lower() for k in ["energy_last", "energy_max", "phase_energy"])],
        "duration": [c for c in analysis.columns if any(k in c.lower() for k in ["duration", "dt_median", "sample_period"])],
        "temperature": [c for c in analysis.columns if any(k in c.lower() for k in ["temperature_min", "temperature_max", "temperature_mean"])],
    }
    for name, cols in groups.items():
        cols = [c for c in cols if c in analysis.columns][:12]
        if not cols:
            continue
        long = []
        for c in cols:
            temp = pd.DataFrame({"feature": c, "value": safe_numeric_series(analysis[c]), "dataset_phase": analysis.get("dataset", "") .astype(str) + "/" + analysis.get("phase", "").astype(str)})
            temp = temp.dropna(subset=["value"])
            if len(temp) > 5000:
                temp = temp.sample(n=5000, random_state=args.random_seed)
            long.append(temp)
        if not long:
            continue
        plot_df = pd.concat(long, ignore_index=True)
        if plot_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(10.5, max(4.5, len(cols) * 0.6)))
        data = [plot_df.loc[plot_df["feature"] == c, "value"].to_numpy() for c in cols]
        ax.boxplot(data, labels=cols, vert=False, showfliers=False)
        ax.set_xlabel("value")
        ax.set_title(f"Physical plausibility ranges — {name}")
        ax.grid(True, axis="x", alpha=0.25)
        save_fig(fig, out_dir / f"boxplot_{name}_by_dataset_phase.{args.plot_format}", dpi=args.dpi)


def plot_outliers(outlier_summary: pd.DataFrame, outlier_cycles: pd.DataFrame, args: argparse.Namespace) -> None:
    out_dir = args.output_root / "global" / "plots" / "outliers"
    if not outlier_summary.empty:
        agg = outlier_summary.groupby(["feature_name", "feature_group"], dropna=False)["outlier_pct"].mean().reset_index()
        agg = agg.sort_values("outlier_pct", ascending=True).tail(args.n_top_features)
        labels = agg["feature_group"].astype(str) + " | " + agg["feature_name"].astype(str)
        fig, ax = plt.subplots(figsize=(10.0, max(4.5, 0.32 * len(agg) + 1.5)))
        ax.barh(labels, agg["outlier_pct"])
        ax.set_xlabel("mean outlier_pct")
        ax.set_title("Top outlier features by group")
        ax.grid(True, axis="x", alpha=0.25)
        save_fig(fig, out_dir / f"top_outlier_features_by_group.{args.plot_format}", dpi=args.dpi)
    if not outlier_cycles.empty:
        by_ds = outlier_cycles.groupby(["dataset", "phase"], dropna=False).size().reset_index(name="outlier_events")
        by_ds["dataset_phase"] = by_ds["dataset"].astype(str) + "/" + by_ds["phase"].astype(str)
        plot_barh(by_ds, "outlier_events", "dataset_phase", "Outlier events by dataset/phase", out_dir / "outlier_rate_by_dataset_phase", args.dpi, args.plot_format, top_n=50)
        by_bat = outlier_cycles.groupby(["dataset", "battery_id"], dropna=False).size().reset_index(name="outlier_events")
        by_bat["battery"] = by_bat["dataset"].astype(str) + "/" + by_bat["battery_id"].astype(str)
        plot_barh(by_bat, "outlier_events", "battery", "Outlier events by battery", out_dir / "outlier_rate_by_battery", args.dpi, args.plot_format, top_n=50)


def plot_target_distributions(analysis: pd.DataFrame, target_balance_df: pd.DataFrame, args: argparse.Namespace) -> None:
    if analysis.empty:
        return
    out_dir = args.output_root / "global" / "plots" / "target"
    target_col = choose_target_column(analysis.columns, args.target_column)
    target_norm_col = choose_norm_target_column(analysis.columns, args.target_norm_column)
    for col, name in [(target_col, "remaining_cycles"), (target_norm_col, "remaining_cycles_norm")]:
        if not col or col not in analysis.columns:
            continue
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        groups = analysis.groupby(["dataset", "phase"], dropna=False) if "dataset" in analysis.columns and "phase" in analysis.columns else [("global", analysis)]
        for key, g in groups:
            s = safe_numeric_series(g[col]).dropna()
            if len(s) < 2:
                continue
            if len(s) > 5000:
                s = s.sample(n=5000, random_state=args.random_seed)
            ax.hist(s, bins=40, alpha=0.25, label=str(key))
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(f"Target distribution by dataset/phase — {col}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=2)
        save_fig(fig, out_dir / f"{name}_distribution_by_dataset_phase.{args.plot_format}", dpi=args.dpi)
    if not target_balance_df.empty:
        pivot = target_balance_df.pivot_table(index=["dataset", "phase"], columns="life_range", values="n_rows", aggfunc="sum", fill_value=0)
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(pivot) + 1.5)))
            pivot.plot(kind="barh", stacked=True, ax=ax)
            ax.set_xlabel("n_rows")
            ax.set_title("Target balance by dataset/phase")
            ax.grid(True, axis="x", alpha=0.25)
            save_fig(fig, out_dir / f"target_balance_by_dataset_phase.{args.plot_format}", dpi=args.dpi)


def plot_correlations(corr_global: pd.DataFrame, corr_phase: pd.DataFrame, args: argparse.Namespace) -> None:
    out_dir = args.output_root / "global" / "plots" / "correlations"
    if not corr_global.empty:
        top = corr_global.sort_values("spearman_abs", ascending=True).tail(20)
        plot_barh(top, "spearman_abs", "feature_name", "Top 20 features by absolute Spearman — global", out_dir / "top20_features_by_spearman_global", args.dpi, args.plot_format, top_n=20)
        for group, g in corr_global.groupby("feature_group", dropna=False):
            topg = g.sort_values("spearman_abs", ascending=True).tail(args.n_top_features)
            plot_barh(topg, "spearman_abs", "feature_name", f"Top features by Spearman — {group}", out_dir / f"top_features_{safe_filename(group)}", args.dpi, args.plot_format, top_n=args.n_top_features)
    if not corr_phase.empty and "phase" in corr_phase.columns:
        for phase, g in corr_phase.groupby("phase", dropna=False):
            top = g.sort_values("spearman_abs", ascending=True).tail(20)
            plot_barh(top, "spearman_abs", "feature_name", f"Top 20 features by absolute Spearman — {phase}", out_dir / f"top20_features_by_spearman_{safe_filename(phase)}", args.dpi, args.plot_format, top_n=20)


def plot_redundancy(analysis: pd.DataFrame, args: argparse.Namespace) -> None:
    if analysis.empty:
        return
    out_dir = args.output_root / "global" / "plots" / "redundancy"
    numeric_cols = infer_numeric_columns(analysis)
    target_cols = set(TARGET_CANDIDATES) | {"cycle_number", "cycle_index", "source_row_id", "stage_index", "stage_total"}
    features = [c for c in numeric_cols if c not in target_cols and classify_feature_group(c) not in {"metadata", "target"}]
    for group in ["capacity", "energy", "temperature", "QV_curve", "VQ_curve", "stage_structure"]:
        cols = [c for c in features if classify_feature_group(c) == group]
        if len(cols) < 2:
            continue
        if len(cols) > args.max_corr_plot_features:
            cols = cols[: args.max_corr_plot_features]
        sub = analysis[cols].apply(safe_numeric_series)
        corr = sub.corr(method="spearman", min_periods=20)
        if corr.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.35), max(5, len(cols) * 0.35)))
        im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=6)
        ax.set_yticklabels(cols, fontsize=6)
        ax.set_title(f"Spearman correlation within group — {group}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_fig(fig, out_dir / f"feature_corr_{safe_filename(group)}_group.{args.plot_format}", dpi=args.dpi)


def plot_domain_shift(domain_shift: pd.DataFrame, args: argparse.Namespace) -> None:
    if domain_shift.empty:
        return
    out_dir = args.output_root / "global" / "plots" / "domain_shift"
    scores = domain_shift[["feature_name", "feature_group", "domain_shift_score"]].drop_duplicates().dropna()
    if scores.empty:
        return
    unstable = scores.sort_values("domain_shift_score", ascending=True).tail(args.n_top_features)
    stable = scores.sort_values("domain_shift_score", ascending=True).head(args.n_top_features)
    unstable["label"] = unstable["feature_group"].astype(str) + " | " + unstable["feature_name"].astype(str)
    stable["label"] = stable["feature_group"].astype(str) + " | " + stable["feature_name"].astype(str)
    plot_barh(unstable, "domain_shift_score", "label", "Most unstable features between datasets/phases", out_dir / "unstable_features_by_dataset", args.dpi, args.plot_format, top_n=args.n_top_features)
    plot_barh(stable, "domain_shift_score", "label", "Most stable features between datasets/phases", out_dir / "stable_features_by_dataset", args.dpi, args.plot_format, top_n=args.n_top_features)


def plot_curve_feature_diagnostics(analysis: pd.DataFrame, curve_valid: pd.DataFrame, corr_global: pd.DataFrame, args: argparse.Namespace) -> None:
    out_dir = args.output_root / "global" / "plots" / "curve_features"
    if not curve_valid.empty:
        curve_valid["label"] = curve_valid.get("dataset", "").astype(str) + "/" + curve_valid.get("phase", "").astype(str) + " | " + curve_valid["curve_valid_column"].astype(str)
        plot_barh(curve_valid, "valid_pct", "label", "Curve feature validity", out_dir / "curve_feature_validity_by_dataset_phase", args.dpi, args.plot_format, top_n=80)
    if analysis.empty:
        return
    # NaN profile and target correlation profile for QV/VQ sample columns.
    for prefix in ["QV", "VQ"]:
        sample_cols = [c for c in analysis.columns if re.match(rf"^{prefix}_p\d+_\d+", c, flags=re.IGNORECASE)]
        by_res: Dict[str, List[str]] = {}
        for c in sample_cols:
            m = re.match(rf"^{prefix}_(p\d+)_", c, flags=re.IGNORECASE)
            if m:
                by_res.setdefault(m.group(1), []).append(c)
        for res, cols in by_res.items():
            cols = sorted(cols)
            if not cols:
                continue
            for phase, g in analysis.groupby("phase", dropna=False) if "phase" in analysis.columns else [("global", analysis)]:
                nan_pct = [float(safe_numeric_series(g[c]).isna().mean() * 100.0) for c in cols]
                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.plot(range(1, len(cols) + 1), nan_pct, marker=".")
                ax.set_xlabel("curve sample index")
                ax.set_ylabel("NaN (%)")
                ax.set_title(f"{prefix} NaN profile — {res} — {phase}")
                ax.grid(True, alpha=0.25)
                save_fig(fig, out_dir / f"{prefix}_nan_profile_{res}_{safe_filename(phase)}.{args.plot_format}", dpi=args.dpi)
                if not corr_global.empty:
                    cg = corr_global[corr_global["feature_name"].isin(cols)].copy()
                    if not cg.empty:
                        cg["idx"] = cg["feature_name"].map({c: i + 1 for i, c in enumerate(cols)})
                        cg = cg.sort_values("idx")
                        fig, ax = plt.subplots(figsize=(10, 4.5))
                        ax.plot(cg["idx"], cg["spearman_r"], marker=".")
                        ax.set_xlabel("curve sample index")
                        ax.set_ylabel("Spearman r with target")
                        ax.set_title(f"{prefix} target-correlation profile — {res} — global")
                        ax.grid(True, alpha=0.25)
                        save_fig(fig, out_dir / f"{prefix}_target_correlation_profile_{res}_global.{args.plot_format}", dpi=args.dpi)


# =============================================================================
# Markdown report
# =============================================================================

def read_table_head(path: Path, n: int = 8) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=n)
    except Exception:
        return pd.DataFrame()


def table_to_md(df: pd.DataFrame, max_rows: int = 8) -> str:
    if df is None or df.empty:
        return "_No data available._"
    return df.head(max_rows).to_markdown(index=False)


def generate_report(args: argparse.Namespace, extra: Optional[Dict[str, Any]] = None) -> None:
    root = args.output_root
    tables = root / "global" / "tables"
    report_path = root / "global" / "reports" / "feature_audit_report.md"
    ensure_dir(report_path.parent)

    files_expected = [
        "sampled_raw_curve_cycles.csv",
        "input_cycle_files_summary.csv",
        "feature_inventory.csv",
        "feature_completeness_summary.csv",
        "physical_limit_violations.csv",
        "feature_outlier_summary.csv",
        "feature_outlier_cycles.csv",
        "target_balance_summary.csv",
        "feature_target_correlation_global.csv",
        "feature_target_correlation_by_phase.csv",
        "feature_target_correlation_by_dataset_phase.csv",
        "redundant_feature_pairs.csv",
        "feature_domain_shift_summary.csv",
        "curve_feature_validity_by_dataset_phase.csv",
        "feature_quality_scorecard.csv",
        "capacity_life_summary_loaded.csv",
        "capacity_warning_batteries.csv",
        "capacity_warnings_by_dataset_phase.csv",
        "eol_coverage_by_dataset_phase.csv",
        "dataset_phase_imbalance_summary.csv",
        "dataset_imbalance_summary.csv",
    ]

    def count_rows(name: str) -> str:
        p = tables / name
        if not p.exists():
            return "missing"
        try:
            return str(sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")) - 1)
        except Exception:
            return "unknown"

    scorecard = read_table_head(tables / "feature_quality_scorecard.csv", n=100000)
    comp = read_table_head(tables / "feature_completeness_summary.csv", n=100000)
    corr = read_table_head(tables / "feature_target_correlation_global.csv", n=100000)
    violations = read_table_head(tables / "physical_limit_violations.csv", n=100000)
    target_balance_df = read_table_head(tables / "target_balance_summary.csv", n=100000)

    lines: List[str] = []
    lines.append("# Feature Audit Report")
    lines.append("")
    lines.append(f"Generated at: `{now_iso()}`")
    lines.append("")
    lines.append("## 1. Executive summary")
    lines.append("")
    if extra:
        for k, v in extra.items():
            lines.append(f"- `{k}`: `{v}`")
    lines.append(f"- Output root: `{root}`")
    if not scorecard.empty and "recommended_status" in scorecard.columns:
        status_counts = scorecard["recommended_status"].value_counts(dropna=False).reset_index()
        status_counts.columns = ["recommended_status", "count"]
        lines.append("")
        lines.append("Recommended feature-status counts:")
        lines.append(table_to_md(status_counts, 20))
    lines.append("")
    lines.append("## 2. Input files analyzed")
    lines.append("")
    lines.append(table_to_md(read_table_head(tables / "input_cycle_files_summary.csv", 20), 20))
    lines.append("")
    lines.append("## 3. Raw signal sampled-curve audit")
    lines.append("")
    lines.append("The raw signal plots use up to `--sample-raw-cycles` equally spaced phase rows per dataset/battery/phase. See `sampled_raw_curve_cycles.csv`.")
    lines.append("")
    lines.append(table_to_md(read_table_head(tables / "sampled_raw_curve_cycles.csv", 10), 10))
    lines.append("")
    lines.append("## 4. V(Q) curves by battery")
    lines.append("")
    lines.append("All available cycles are plotted per battery and phase in `<dataset>/VQ_curves/<battery_id>/<phase>/VQ_all_cycles.*`, unless a max-curve limit was explicitly passed.")
    lines.append("")
    lines.append("## 5. E(Q) curves by battery")
    lines.append("")
    lines.append("All available cycles are plotted per battery and phase in `<dataset>/EQ_curves/<battery_id>/<phase>/EQ_all_cycles.*`, unless a max-curve limit was explicitly passed.")
    lines.append("")
    lines.append("## 6. Capacity and energy degradation by cycle")
    lines.append("")
    lines.append("Per-battery and per-dataset degradation plots are saved under `<dataset>/degradation/`.")
    lines.append("")
    lines.append("## 7. Feature inventory")
    lines.append("")
    lines.append(table_to_md(read_table_head(tables / "feature_inventory.csv", 12), 12))
    lines.append("")
    lines.append("## 8. Feature completeness and NaN analysis")
    lines.append("")
    if not comp.empty:
        poor = comp[comp.get("availability_class", "") .isin(["missing", "poor", "constant"])] if "availability_class" in comp.columns else pd.DataFrame()
        lines.append(f"Features/contexts with missing, poor or constant status: `{len(poor)}`")
        lines.append(table_to_md(poor.sort_values("finite_pct", ascending=True).head(20) if not poor.empty and "finite_pct" in poor.columns else poor, 20))
    else:
        lines.append("_No completeness table found._")
    lines.append("")
    lines.append("## 9. Physical plausibility and limit violations")
    lines.append("")
    lines.append(f"Violation rows: `{len(violations) if not violations.empty else 0}`")
    lines.append(table_to_md(violations.head(20) if not violations.empty else violations, 20))
    lines.append("")
    lines.append("## 10. Outlier analysis")
    lines.append("")
    lines.append(table_to_md(read_table_head(tables / "feature_outlier_summary.csv", 20), 20))
    lines.append("")
    lines.append("## 11. Target distribution and imbalance")
    lines.append("")
    lines.append(table_to_md(target_balance_df.head(20) if not target_balance_df.empty else target_balance_df, 20))
    lines.append("")
    lines.append("### Dataset/phase imbalance from extraction summary")
    lines.append("")
    imbalance = read_table_head(tables / "dataset_phase_imbalance_summary.csv", 40)
    if not imbalance.empty and "row_share_pct" in imbalance.columns:
        show_cols = [c for c in ["dataset", "phase", "rows", "batteries", "cycles", "row_share_pct"] if c in imbalance.columns]
        lines.append(table_to_md(imbalance[show_cols].sort_values("row_share_pct", ascending=False).head(20), 20))
    else:
        lines.append("_No dataset-phase extraction summary was provided._")
    lines.append("")
    lines.append("### Capacity-life warnings from extraction summary")
    lines.append("")
    cap_warn = read_table_head(tables / "capacity_warning_batteries.csv", 40)
    if not cap_warn.empty:
        show_cols = [c for c in ["dataset", "battery_id", "phase", "n_rows", "initial_capacity_ah", "capacity_last_ah", "capacity_end_to_initial_ratio", "capacity_validation_status", "capacity_validation_notes"] if c in cap_warn.columns]
        lines.append(table_to_md(cap_warn[show_cols].head(20), 20))
    else:
        lines.append("_No capacity-life warnings were found or no capacity summary was provided._")
    lines.append("")
    lines.append("## 12. Feature-target correlation")
    lines.append("")
    if not corr.empty and "spearman_abs" in corr.columns:
        top_corr = corr.sort_values("spearman_abs", ascending=False).head(25)
        lines.append(table_to_md(top_corr, 25))
    else:
        lines.append("_No global correlation table found._")
    lines.append("")
    lines.append("## 13. Feature redundancy")
    lines.append("")
    lines.append(table_to_md(read_table_head(tables / "redundant_feature_pairs.csv", 20), 20))
    lines.append("")
    lines.append("## 14. Dataset and phase stability")
    lines.append("")
    domain = read_table_head(tables / "feature_domain_shift_summary.csv", 100000)
    if not domain.empty and "domain_shift_score" in domain.columns:
        top_domain = domain[["feature_name", "feature_group", "domain_shift_score"]].drop_duplicates().dropna().sort_values("domain_shift_score", ascending=False).head(20)
        lines.append(table_to_md(top_domain, 20))
    else:
        lines.append("_No domain-shift table found._")
    lines.append("")
    lines.append("## 15. Q(V), V(Q), and E(Q) feature diagnostics")
    lines.append("")
    lines.append(table_to_md(read_table_head(tables / "curve_feature_validity_by_dataset_phase.csv", 20), 20))
    lines.append("")
    lines.append("## 16. Recommended feature groups for modeling")
    lines.append("")
    if not scorecard.empty and "recommended_status" in scorecard.columns:
        keep = scorecard[scorecard["recommended_status"].isin(["keep", "keep_with_caution"])]
        if "abs_spearman_remaining_cycles" in keep.columns:
            keep = keep.sort_values("abs_spearman_remaining_cycles", ascending=False)
        lines.append(table_to_md(keep.head(40), 40))
    else:
        lines.append("_No scorecard found._")
    lines.append("")
    lines.append("## 17. Critical issues requiring manual inspection")
    lines.append("")
    lines.append("Review, at minimum, features marked as `drop_physically_invalid`, `drop_unstable`, `drop_missing`, datasets/batteries with many outlier events, and batteries listed in `capacity_warning_batteries.csv`.")
    lines.append("")
    lines.append("## 18. Generated files")
    lines.append("")
    file_rows = pd.DataFrame({"file": files_expected, "rows": [count_rows(f) for f in files_expected]})
    lines.append(table_to_md(file_rows, len(file_rows)))
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main feature audit
# =============================================================================

def audit_features(files: Sequence[FeatureFile], args: argparse.Namespace) -> Dict[str, Any]:
    tables_dir = args.output_root / "global" / "tables"
    ensure_dir(tables_dir)

    inventory = feature_inventory(files, args.engine)
    write_csv(inventory, tables_dir / "feature_inventory.csv")

    analysis, completeness, violations, outlier_summary, outlier_cycles, target_bal, failures = build_analysis_frame(files, args)
    write_csv(completeness, tables_dir / "feature_completeness_summary.csv")
    write_csv(violations, tables_dir / "physical_limit_violations.csv")
    write_csv(outlier_summary, tables_dir / "feature_outlier_summary.csv")
    write_csv(outlier_cycles, tables_dir / "feature_outlier_cycles.csv")
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

    domain = compute_domain_shift(analysis, outlier_summary, args)
    write_csv(domain, tables_dir / "feature_domain_shift_summary.csv")

    curve_valid = curve_feature_validity(analysis)
    write_csv(curve_valid, tables_dir / "curve_feature_validity_by_dataset_phase.csv")

    scorecard = feature_quality_scorecard(completeness, outlier_summary, violations, corr_global, domain, redundant)
    write_csv(scorecard, tables_dir / "feature_quality_scorecard.csv")

    if args.mode in {"all", "features"} and not args.no_feature_plots:
        plot_completeness(completeness, args)
        plot_physical_limits(analysis, args)
        plot_outliers(outlier_summary, outlier_cycles, args)
        plot_target_distributions(analysis, target_bal, args)
        plot_correlations(corr_global, corr_phase, args)
        plot_redundancy(analysis, args)
        plot_domain_shift(domain, args)
        plot_curve_feature_diagnostics(analysis, curve_valid, corr_global, args)

    summary = {
        "feature_files": len(files),
        "inventory_rows": len(inventory),
        "analysis_rows_sampled": len(analysis),
        "completeness_rows": len(completeness),
        "physical_violation_rows": len(violations),
        "outlier_summary_rows": len(outlier_summary),
        "outlier_cycle_rows": len(outlier_cycles),
        "correlation_global_rows": len(corr_global),
        "redundant_pairs": len(redundant),
        "domain_shift_rows": len(domain),
        "scorecard_rows": len(scorecard),
        "feature_read_failures": len(failures),
    }
    write_json(summary, args.output_root / "global" / "reports" / "feature_audit_summary.json")
    return summary



# =============================================================================
# Auxiliary extraction-summary audits
# =============================================================================

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


def critical_keys_from_capacity_summary(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if df.empty or not {"dataset", "battery_id", "phase"}.issubset(df.columns):
        return set()
    status = df.get("capacity_validation_status", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    notes = df.get("capacity_validation_notes", pd.Series("", index=df.index)).fillna("").astype(str).str.strip().str.lower()
    notes = notes.mask(notes.isin(["nan", "none", "null"]), "")
    warning_mask = (~status.isin(["", "ok", "nan", "none", "null"])) | notes.str.contains("warning|below|exceed|ratio|invalid|non|low|high", regex=True, na=False)
    crit = df.loc[warning_mask, ["dataset", "battery_id", "phase"]].dropna().copy()
    return set(zip(crit["dataset"].astype(str), crit["battery_id"].astype(str), crit["phase"].astype(str).str.lower()))


def audit_capacity_life_summary(args: argparse.Namespace) -> Dict[str, Any]:
    tables_dir = args.output_root / "global" / "tables"
    plots_dir = args.output_root / "global" / "plots" / "target"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)
    df = load_capacity_life_summary(args.capacity_life_summary, getattr(args, "canonicalize_dataset_names", True))
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
        count["label"] = count["dataset"].astype(str) + "/" + count["phase"].astype(str)
        write_csv(count, tables_dir / "capacity_warnings_by_dataset_phase.csv")
        plot_barh(count, "warning_series", "label", "Capacity warnings by dataset/phase", plots_dir / "capacity_warnings_by_dataset_phase", args.dpi, args.plot_format, top_n=80)

    # Compact EOL coverage table.
    eol_cols = [c for c in ["eol80_found", "eol70_found"] if c in df.columns]
    if eol_cols:
        agg_dict = {c: (c, lambda x: float(pd.Series(x).astype(bool).mean() * 100.0)) for c in eol_cols}
        coverage = df.groupby(["dataset", "phase"], dropna=False).agg(
            series=("battery_id", "count"),
            batteries=("battery_id", lambda x: int(pd.Series(x).nunique())),
            **agg_dict,
        ).reset_index()
        # Rename percentage columns for clarity.
        coverage = coverage.rename(columns={"eol80_found": "eol80_found_pct", "eol70_found": "eol70_found_pct"})
        write_csv(coverage, tables_dir / "eol_coverage_by_dataset_phase.csv")

    return {"capacity_life_summary_rows": int(len(df)), "capacity_warning_rows": int(len(warnings))}


def audit_dataset_phase_summary(args: argparse.Namespace) -> Dict[str, Any]:
    tables_dir = args.output_root / "global" / "tables"
    plots_dir = args.output_root / "global" / "plots" / "target"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)
    path = getattr(args, "dataset_phase_summary", None)
    if not path:
        return {"dataset_phase_summary_rows": 0}
    path = Path(path)
    if not path.exists():
        return {"dataset_phase_summary_rows": 0}
    df = pd.read_csv(path)
    if df.empty:
        return {"dataset_phase_summary_rows": 0}
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].map(lambda x: canonicalize_dataset_name(x, enabled=getattr(args, "canonicalize_dataset_names", True)))
    phase_col = "phase_folder" if "phase_folder" in df.columns else "phase" if "phase" in df.columns else None
    if phase_col:
        df["phase"] = df[phase_col].map(normalize_phase_value)
    else:
        df["phase"] = "unknown"

    row_col = "rows" if "rows" in df.columns else "target_rows" if "target_rows" in df.columns else None
    batt_col = "batteries" if "batteries" in df.columns else "n_batteries" if "n_batteries" in df.columns else None
    cyc_col = "cycles" if "cycles" in df.columns else "n_cycles" if "n_cycles" in df.columns else None
    if row_col is None:
        df["rows"] = 1
        row_col = "rows"
    df[row_col] = pd.to_numeric(df[row_col], errors="coerce").fillna(0)
    total_rows = float(df[row_col].sum())
    df["row_share_pct"] = df[row_col] / total_rows * 100.0 if total_rows else np.nan
    write_csv(df, tables_dir / "dataset_phase_imbalance_summary.csv")

    plot_df = df.copy()
    plot_df["label"] = plot_df["dataset"].astype(str) + "/" + plot_df["phase"].astype(str)
    plot_barh(plot_df, "row_share_pct", "label", "Row share by dataset/phase", plots_dir / "dataset_row_share", args.dpi, args.plot_format, top_n=80)
    if batt_col and batt_col in plot_df.columns:
        plot_df[batt_col] = pd.to_numeric(plot_df[batt_col], errors="coerce")
        plot_barh(plot_df, batt_col, "label", "Battery count by dataset/phase", plots_dir / "battery_count_by_dataset_phase", args.dpi, args.plot_format, top_n=80)
    if cyc_col and cyc_col in plot_df.columns:
        plot_df[cyc_col] = pd.to_numeric(plot_df[cyc_col], errors="coerce")
        plot_barh(plot_df, cyc_col, "label", "Cycle count by dataset/phase", plots_dir / "cycle_count_by_dataset_phase", args.dpi, args.plot_format, top_n=80)

    by_dataset = df.groupby("dataset", dropna=False).agg(rows=(row_col, "sum")).reset_index()
    by_dataset["row_share_pct"] = by_dataset["rows"] / max(float(by_dataset["rows"].sum()), 1.0) * 100.0
    by_dataset = by_dataset.sort_values("row_share_pct", ascending=False)
    write_csv(by_dataset, tables_dir / "dataset_imbalance_summary.csv")
    return {"dataset_phase_summary_rows": int(len(df)), "largest_dataset_row_share_pct": float(by_dataset["row_share_pct"].max()) if not by_dataset.empty else np.nan}

# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="General audit pipeline for cleaned battery cycles and extracted features.")
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--capacity-life-summary", type=Path, default=None, help="Optional capacity_life_summary_v3.csv from the feature-extraction run.")
    parser.add_argument("--dataset-phase-summary", type=Path, default=None, help="Optional dataset_phase_target_summary.csv from the feature-extraction run.")
    parser.add_argument("--canonicalize-dataset-names", dest="canonicalize_dataset_names", action="store_true", default=True, help="Canonicalize dataset aliases such as ORA->OXFORD and RW->NRW. Enabled by default.")
    parser.add_argument("--no-canonicalize-dataset-names", dest="canonicalize_dataset_names", action="store_false", help="Disable dataset alias canonicalization.")
    parser.add_argument("--priority-warnings-only", action="store_true", help="For plot audit, process only battery/phase series flagged in capacity_life_summary_v3.csv.")
    parser.add_argument("--datasets", nargs="*", default=["all"], help="Datasets to audit, e.g. CALCE HUST MATR, or all.")
    parser.add_argument("--phases", nargs="*", default=list(PHASES), help="Phases to audit: charge discharge, or all.")
    parser.add_argument("--mode", choices=["all", "plots", "features", "report"], default="all")
    parser.add_argument("--engine", default="fastparquet", choices=["fastparquet", "pyarrow"], help="Parquet engine.")
    parser.add_argument("--sample-raw-cycles", type=int, default=5)
    parser.add_argument("--max-batteries", type=int, default=0, help="Optional limit per input file for fast debugging. 0 means no limit.")
    parser.add_argument("--max-curves-per-battery-phase", type=int, default=0, help="Optional cap for V(Q)/E(Q) curves. 0 means all cycles.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--plot-format", default="png", choices=["png", "jpg", "jpeg", "pdf", "svg"])
    parser.add_argument("--target-column", default="auto")
    parser.add_argument("--target-norm-column", default="auto")
    parser.add_argument("--max-correlation-rows", type=int, default=200000, help="Maximum sampled feature rows used for global correlations/redundancy.")
    parser.add_argument("--max-corr-features", type=int, default=0, help="Feature cap for target correlations. 0 means no cap.")
    parser.add_argument("--max-redundancy-features-per-group", type=int, default=60)
    parser.add_argument("--max-corr-plot-features", type=int, default=40)
    parser.add_argument("--max-domain-features", type=int, default=300)
    parser.add_argument("--redundancy-threshold", type=float, default=0.95)
    parser.add_argument("--n-top-features", type=int, default=30)
    parser.add_argument("--max-issue-rows-per-file", type=int, default=10000)
    parser.add_argument("--mutual-info", action="store_true", help="Compute mutual_information_regression when sklearn is available.")
    parser.add_argument("--no-feature-plots", action="store_true")
    parser.add_argument("--save-analysis-sample", action="store_true")
    parser.add_argument("--save-analysis-sample-rows", type=int, default=20000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.datasets = parse_dataset_args(args.datasets, canonicalize=args.canonicalize_dataset_names)
    args.phases = parse_phase_args(args.phases)
    args.output_root = Path(args.output_root)
    ensure_dir(args.output_root / "global" / "tables")
    ensure_dir(args.output_root / "global" / "reports")

    run_info: Dict[str, Any] = {
        "generated_at": now_iso(),
        "mode": args.mode,
        "interim_root": str(args.interim_root),
        "features_root": str(args.features_root),
        "output_root": str(args.output_root),
        "datasets": args.datasets or "all",
        "phases": args.phases,
        "sample_raw_cycles": args.sample_raw_cycles,
        "canonicalize_dataset_names": args.canonicalize_dataset_names,
        "priority_warnings_only": args.priority_warnings_only,
        "capacity_life_summary": str(args.capacity_life_summary) if args.capacity_life_summary else None,
        "dataset_phase_summary": str(args.dataset_phase_summary) if args.dataset_phase_summary else None,
    }

    if args.engine == "fastparquet" and fastparquet is None:
        print("[WARNING] fastparquet is not importable in this environment. The script may fail unless it is installed.", file=sys.stderr)

    # Auxiliary extraction summaries are optional, but they improve the audit:
    # - capacity_life_summary_v3.csv identifies battery/phase series that need visual inspection;
    # - dataset_phase_target_summary.csv quantifies dataset imbalance.
    cap_summary_df = load_capacity_life_summary(args.capacity_life_summary, args.canonicalize_dataset_names)
    args.critical_battery_phase_keys = critical_keys_from_capacity_summary(cap_summary_df)
    if args.priority_warnings_only:
        print(f"- Priority warning mode: {len(args.critical_battery_phase_keys)} critical battery/phase keys loaded.")
        if not args.critical_battery_phase_keys:
            print("[WARNING] --priority-warnings-only was requested, but no capacity warning keys were found.", file=sys.stderr)

    aux_summary: Dict[str, Any] = {}
    aux_summary.update(audit_capacity_life_summary(args))
    aux_summary.update(audit_dataset_phase_summary(args))
    run_info.update(aux_summary)

    clean_files = discover_clean_cycle_files(args.interim_root, args.datasets, args.phases)
    feature_files = discover_feature_files(args.features_root, args.datasets, args.phases)
    run_info["clean_cycle_files_found"] = len(clean_files)
    run_info["feature_files_found"] = len(feature_files)
    write_json(run_info, args.output_root / "global" / "reports" / "audit_run_info.json")

    print("General audit pipeline")
    print("- Clean cycle files:", len(clean_files))
    print("- Feature files:", len(feature_files))
    print("- Output root:", args.output_root)

    if args.mode in {"all", "plots"}:
        if not clean_files:
            print("[WARNING] No clean cycle files found. Skipping plot audit.", file=sys.stderr)
        else:
            audit_clean_cycle_plots(clean_files, args)

    summary: Dict[str, Any] = {}
    if args.mode in {"all", "features"}:
        if not feature_files:
            print("[WARNING] No feature files found. Skipping feature diagnostics.", file=sys.stderr)
        else:
            summary = audit_features(feature_files, args)
            run_info.update(summary)
            write_json(run_info, args.output_root / "global" / "reports" / "audit_run_info.json")

    if args.mode in {"all", "features", "plots", "report"}:
        generate_report(args, extra=run_info)

    print("Done. Main report:", args.output_root / "global" / "reports" / "feature_audit_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
