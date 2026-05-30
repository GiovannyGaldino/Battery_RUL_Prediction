"""
run_feature_extraction_v2.py

Extração hierárquica de características para bases de baterias, com:
- retomada automática por marcadores .done;
- paralelismo controlado por arquivo/bateria;
- saída particionada, evitando vários processos escrevendo no mesmo Parquet;
- uso de fastparquet;
- redução de memória por seleção de colunas, downcast e limpeza explícita;
- detecção de segmentos dentro das fases por saltos no tempo;
- relatórios de execução em CSV/JSONL.

Exemplos de uso no PowerShell, a partir da raiz do projeto:

python scripts/pipelines/run_feature_extraction_v2.py --dataset hust --workers 2 --resume
python scripts/pipelines/run_feature_extraction_v2.py --all --workers 2 --resume
python scripts/pipelines/run_feature_extraction_v2.py --all --workers 1 --resume --force

Observação:
- Use --workers 1 ou 2 se a memória ainda estiver alta.
- O script NÃO faz vários processos escreverem no mesmo arquivo Parquet.
- Cada arquivo de entrada gera um arquivo de features independente + marcador .done.
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
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "fastparquet não está instalado ou não pôde ser importado. "
        "Instale com: pip install fastparquet"
    ) from exc


# =============================================================================
# Configurações padrão
# =============================================================================

DEFAULT_INPUT_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/interim")
DEFAULT_OUTPUT_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/processed/features_hierarchical_v2")
DEFAULT_REPORTS_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/reports/feature_extraction_v2")

SUPPORTED_SUFFIXES = (".parquet", ".csv", ".txt", ".dat")

# Possíveis nomes de colunas nas bases já adaptadas.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "dataset": [
        "dataset",
        "database",
        "source",
        "base",
    ],
    "battery_id": [
        "battery_id",
        "battery",
        "cell_id",
        "cell",
        "cell_name",
        "battery_name",
        "batteryid",
        "cellid",
    ],
    "cycle": [
        "cycle",
        "cycle_id",
        "cycle_index",
        "cycle_idx",
        "cycle_number",
        "cycle_no",
        "cycle_num",
        "n_cycle",
    ],
    "phase": [
        "phase",
        "step_type",
        "step",
        "operation",
        "mode",
        "state",
        "type",
        "status",
    ],
    "time": [
        "time",
        "time_s",
        "time_sec",
        "time_seconds",
        "test_time",
        "test_time_s",
        "elapsed_time",
        "elapsed_time_s",
        "relative_time",
        "relative_time_s",
        "t",
    ],
    "voltage": [
        "voltage",
        "voltage_v",
        "v",
        "u",
        "voltage_measured",
        "terminal_voltage",
    ],
    "current": [
        "current",
        "current_a",
        "i",
        "current_measured",
    ],
    "temperature": [
        "temperature",
        "temperature_c",
        "temp",
        "temp_c",
        "ambient_temperature",
        "temperature_measured",
    ],
    "capacity": [
        "capacity",
        "capacity_ah",
        "charge_capacity",
        "discharge_capacity",
        "q",
        "q_ah",
        "ah",
    ],
    "energy": [
        "energy",
        "energy_wh",
        "charge_energy",
        "discharge_energy",
        "wh",
    ],
    "soh": [
        "soh",
        "state_of_health",
    ],
    "rul": [
        "rul",
        "remaining_useful_life",
    ],
}

CANONICAL_COLUMNS = list(COLUMN_ALIASES.keys())
NUMERIC_CANONICAL = [
    "cycle",
    "time",
    "voltage",
    "current",
    "temperature",
    "capacity",
    "energy",
    "soh",
    "rul",
]
TEXT_CANONICAL = ["dataset", "battery_id", "phase"]


# =============================================================================
# Estruturas de dados
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
# Utilitários gerais
# =============================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def safe_filename(text: str, max_len: int = 120) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "file"
    return text[:max_len]


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def file_is_valid(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def parse_dataset_args(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    datasets: List[str] = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                datasets.append(part)
    return list(dict.fromkeys(datasets))


# =============================================================================
# Descoberta de arquivos e jobs
# =============================================================================

def discover_datasets(input_root: Path, requested: Optional[Sequence[str]], use_all: bool) -> List[Tuple[str, Path]]:
    requested = list(requested or [])

    if requested:
        pairs: List[Tuple[str, Path]] = []
        for name in requested:
            dataset_path = input_root / name
            if dataset_path.exists():
                pairs.append((name, dataset_path))
            else:
                # Permite apontar diretamente para uma pasta fora do input_root.
                possible_path = Path(name)
                if possible_path.exists():
                    pairs.append((possible_path.name, possible_path))
                else:
                    print(f"[AVISO] Dataset/pasta não encontrado: {name}", file=sys.stderr)
        return pairs

    # Se não passar dataset, usa todos por padrão.
    child_dirs = [p for p in input_root.iterdir() if p.is_dir()] if input_root.exists() else []

    if child_dirs:
        return [(p.name, p) for p in sorted(child_dirs, key=lambda x: x.name.lower())]

    # Caso os arquivos estejam diretamente no input_root.
    return [(input_root.name or "default", input_root)]


def discover_input_files(dataset_path: Path, suffixes: Sequence[str]) -> List[Path]:
    suffixes = tuple(s.lower() for s in suffixes)
    files: List[Path] = []
    for path in dataset_path.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in suffixes:
            files.append(path)
    return sorted(files, key=lambda p: str(p).lower())


def build_jobs(
    input_root: Path,
    output_root: Path,
    datasets: Sequence[Tuple[str, Path]],
    suffixes: Sequence[str],
    max_files: Optional[int] = None,
) -> List[Job]:
    jobs: List[Job] = []

    for dataset_name, dataset_path in datasets:
        files = discover_input_files(dataset_path, suffixes)
        if max_files is not None:
            files = files[:max_files]

        for input_path in files:
            try:
                rel = input_path.relative_to(input_root)
            except ValueError:
                rel = input_path.name

            rel_str = str(rel).replace("\\", "/")
            job_hash = short_hash(f"{dataset_name}/{rel_str}")
            stem = safe_filename(input_path.stem)

            out_dir = output_root / safe_filename(dataset_name) / "parts"
            output_path = out_dir / f"{stem}__{job_hash}.features.parquet"
            done_path = out_dir / f"{stem}__{job_hash}.features.parquet.done"
            error_path = out_dir / f"{stem}__{job_hash}.error.txt"

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
# Leitura otimizada
# =============================================================================

def get_file_columns(path: Path) -> List[str]:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        pf = ParquetFile(str(path))
        return list(pf.columns)

    if suffix in {".csv", ".txt", ".dat"}:
        # Primeiro tenta separador automático com cabeçalho.
        sample = pd.read_csv(path, nrows=0, sep=None, engine="python")
        return list(sample.columns)

    raise ValueError(f"Formato não suportado: {path.suffix}")


def infer_column_map(columns: Sequence[str]) -> Dict[str, str]:
    normalized_to_original = {normalize_name(c): c for c in columns}
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

    # Se quase nada foi identificado, lê todas as colunas para evitar falha silenciosa.
    # Em bases já adaptadas, normalmente as colunas principais serão detectadas.
    if len(selected) < 3:
        return list(columns)

    return selected


def read_input_table(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    columns = get_file_columns(path)
    selected_columns = infer_columns_to_read(columns)
    column_map = infer_column_map(columns)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path, columns=selected_columns, engine="fastparquet")
    elif suffix in {".csv", ".txt", ".dat"}:
        df = pd.read_csv(
            path,
            usecols=selected_columns,
            sep=None,
            engine="python",
            low_memory=False,
        )
    else:
        raise ValueError(f"Formato não suportado: {path.suffix}")

    return df, column_map


# =============================================================================
# Normalização de dados
# =============================================================================

def standardize_dataframe(df: pd.DataFrame, column_map: Dict[str, str], job: Job) -> pd.DataFrame:
    rename: Dict[str, str] = {}
    for canonical, original in column_map.items():
        if original in df.columns:
            rename[original] = canonical

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

    # Converte colunas numéricas conhecidas.
    for col in NUMERIC_CANONICAL:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Infere fase quando não existir.
    if "phase" not in df.columns:
        df["phase"] = infer_phase_from_current(df)
    else:
        df["phase"] = normalize_phase_series(df["phase"], df.get("current"))

    # Redução de memória.
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64", "int32", "int16"]).columns:
        if col == "cycle":
            df[col] = pd.to_numeric(df[col], downcast="integer")
        else:
            df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in TEXT_CANONICAL:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("unknown").astype("category")

    # Mantém apenas colunas úteis para features.
    keep_cols = [
        c
        for c in [
            "dataset",
            "battery_id",
            "cycle",
            "phase",
            "time",
            "time_is_synthetic",
            "voltage",
            "current",
            "temperature",
            "capacity",
            "energy",
            "soh",
            "rul",
        ]
        if c in df.columns
    ]
    df = df[keep_cols]

    return df


def infer_phase_from_current(df: pd.DataFrame) -> pd.Series:
    if "current" not in df.columns:
        return pd.Series(["unknown"] * len(df), index=df.index, dtype="string")

    current = pd.to_numeric(df["current"], errors="coerce")
    finite_abs = current.abs().replace([np.inf, -np.inf], np.nan).dropna()
    if finite_abs.empty:
        eps = 1e-9
    else:
        eps = max(float(finite_abs.quantile(0.05)) * 0.01, 1e-9)

    phase = np.where(
        current > eps,
        "charge",
        np.where(current < -eps, "discharge", "rest"),
    )
    return pd.Series(phase, index=df.index, dtype="string")


def normalize_phase_series(phase: pd.Series, current: Optional[pd.Series] = None) -> pd.Series:
    s = phase.astype("string").str.strip().str.lower().fillna("unknown")

    # Ordem importante: "descarga" contém "carga".
    discharge_mask = s.str.contains("discharge|descarga|dchg|decharge", regex=True, na=False)
    charge_mask = s.str.contains("charge|carga|chg", regex=True, na=False) & ~discharge_mask
    rest_mask = s.str.contains("rest|repouso|relax|pausa|pause|idle|calendar", regex=True, na=False)

    out = s.copy()
    out[discharge_mask] = "discharge"
    out[charge_mask] = "charge"
    out[rest_mask] = "rest"

    # Se a fase vier numérica/codificada e houver corrente, tenta refinar unknown.
    unknown_mask = out.isna() | (out == "unknown") | (out == "")
    if current is not None and unknown_mask.any():
        inferred = infer_phase_from_current(pd.DataFrame({"current": current}))
        out[unknown_mask] = inferred[unknown_mask]

    return out.astype("string")


# =============================================================================
# Cálculo de features
# =============================================================================

def finite_array(values: Any) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def safe_float(x: Any) -> float:
    try:
        x = float(x)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return np.nan


def numeric_stats(values: Any, prefix: str) -> Dict[str, float]:
    arr = finite_array(values)
    out: Dict[str, float] = {
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
        return out

    out[f"{prefix}_first"] = safe_float(arr[0])
    out[f"{prefix}_last"] = safe_float(arr[-1])
    out[f"{prefix}_min"] = safe_float(np.min(arr))
    out[f"{prefix}_max"] = safe_float(np.max(arr))
    out[f"{prefix}_mean"] = safe_float(np.mean(arr))
    out[f"{prefix}_std"] = safe_float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    out[f"{prefix}_median"] = safe_float(np.median(arr))
    out[f"{prefix}_q05"] = safe_float(np.quantile(arr, 0.05))
    out[f"{prefix}_q95"] = safe_float(np.quantile(arr, 0.95))
    out[f"{prefix}_delta"] = safe_float(arr[-1] - arr[0])
    out[f"{prefix}_range"] = safe_float(np.max(arr) - np.min(arr))
    return out


def ordered_group(group: pd.DataFrame) -> pd.DataFrame:
    if "time" not in group.columns:
        return group
    return group.sort_values("time", kind="mergesort")


def time_gap_info(group: pd.DataFrame, min_gap_seconds: float, gap_factor: float) -> Dict[str, Any]:
    if "time" not in group.columns or len(group) < 2:
        return {
            "duration_s": 0.0,
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
    finite = np.isfinite(t)

    if finite.sum() < 2:
        return {
            "duration_s": np.nan,
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

    dt = np.diff(t)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]

    if positive_dt.size == 0:
        threshold = min_gap_seconds
        gap_indices = np.array([], dtype=int)
        duration = 0.0
        return {
            "duration_s": duration,
            "dt_median_s": np.nan,
            "dt_mean_s": np.nan,
            "dt_max_s": np.nan,
            "gap_threshold_s": threshold,
            "gap_count": 0,
            "segment_count": 1,
            "gap_total_s": 0.0,
            "gap_max_s": 0.0,
            "gap_indices": gap_indices,
        }

    dt_median = float(np.median(positive_dt))
    dt_mean = float(np.mean(positive_dt))
    dt_max = float(np.max(positive_dt))
    threshold = max(float(min_gap_seconds), float(gap_factor) * dt_median)

    gap_mask = np.isfinite(dt) & (dt > threshold)
    gap_indices = np.where(gap_mask)[0]
    gap_values = dt[gap_indices]

    return {
        "duration_s": safe_float(np.nanmax(t) - np.nanmin(t)),
        "dt_median_s": safe_float(dt_median),
        "dt_mean_s": safe_float(dt_mean),
        "dt_max_s": safe_float(dt_max),
        "gap_threshold_s": safe_float(threshold),
        "gap_count": int(gap_indices.size),
        "segment_count": int(gap_indices.size + 1),
        "gap_total_s": safe_float(np.sum(gap_values)) if gap_values.size else 0.0,
        "gap_max_s": safe_float(np.max(gap_values)) if gap_values.size else 0.0,
        "gap_indices": gap_indices,
    }


def boundary_jump_stats(group: pd.DataFrame, gap_indices: np.ndarray, column: str, prefix: str) -> Dict[str, float]:
    out = {
        f"{prefix}_boundary_jump_n": 0.0,
        f"{prefix}_boundary_jump_mean": np.nan,
        f"{prefix}_boundary_jump_abs_mean": np.nan,
        f"{prefix}_boundary_jump_abs_max": np.nan,
        f"{prefix}_boundary_jump_abs_sum": np.nan,
    }

    if column not in group.columns or gap_indices.size == 0:
        return out

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
        return out

    arr = np.asarray(jumps, dtype=np.float64)
    abs_arr = np.abs(arr)
    out[f"{prefix}_boundary_jump_n"] = float(arr.size)
    out[f"{prefix}_boundary_jump_mean"] = safe_float(np.mean(arr))
    out[f"{prefix}_boundary_jump_abs_mean"] = safe_float(np.mean(abs_arr))
    out[f"{prefix}_boundary_jump_abs_max"] = safe_float(np.max(abs_arr))
    out[f"{prefix}_boundary_jump_abs_sum"] = safe_float(np.sum(abs_arr))
    return out


def integration_features(group: pd.DataFrame) -> Dict[str, float]:
    out = {
        "integrated_capacity_signed_ah": np.nan,
        "integrated_capacity_abs_ah": np.nan,
        "integrated_energy_signed_wh": np.nan,
        "integrated_energy_abs_wh": np.nan,
    }

    if "time" not in group.columns or "current" not in group.columns or len(group) < 2:
        return out

    g = ordered_group(group)
    t = pd.to_numeric(g["time"], errors="coerce").to_numpy(dtype=np.float64)
    i = pd.to_numeric(g["current"], errors="coerce").to_numpy(dtype=np.float64)

    valid = np.isfinite(t) & np.isfinite(i)
    if valid.sum() < 2:
        return out

    t = t[valid]
    i = i[valid]
    dt = np.diff(t)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)

    i_mid = 0.5 * (i[:-1] + i[1:])
    out["integrated_capacity_signed_ah"] = safe_float(np.sum(i_mid * dt) / 3600.0)
    out["integrated_capacity_abs_ah"] = safe_float(np.sum(np.abs(i_mid) * dt) / 3600.0)

    if "voltage" in g.columns:
        v_all = pd.to_numeric(g.loc[valid, "voltage"], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(v_all).sum() >= 2:
            p = v_all * i
            p_mid = 0.5 * (p[:-1] + p[1:])
            out["integrated_energy_signed_wh"] = safe_float(np.sum(p_mid * dt) / 3600.0)
            out["integrated_energy_abs_wh"] = safe_float(np.sum(np.abs(p_mid) * dt) / 3600.0)

    return out


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
        "time_is_synthetic": bool(group["time_is_synthetic"].iloc[0]) if "time_is_synthetic" in group.columns and len(group) else False,
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


def segment_slices(group: pd.DataFrame, min_gap_seconds: float, gap_factor: float) -> List[pd.DataFrame]:
    if len(group) <= 1:
        return [group]

    g = ordered_group(group)
    gap = time_gap_info(g, min_gap_seconds=min_gap_seconds, gap_factor=gap_factor)
    gap_indices = gap["gap_indices"]

    if gap_indices.size == 0:
        return [g]

    starts = [0] + [int(idx) + 1 for idx in gap_indices]
    ends = [int(idx) + 1 for idx in gap_indices] + [len(g)]

    segments = []
    for start, end in zip(starts, ends):
        if end > start:
            segments.append(g.iloc[start:end])
    return segments if segments else [g]


def extract_hierarchical_features(
    df: pd.DataFrame,
    min_gap_seconds: float,
    gap_factor: float,
    inner_progress: bool = False,
) -> pd.DataFrame:
    required = {"dataset", "battery_id", "cycle", "phase", "time"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes após normalização: {missing}")

    rows: List[Dict[str, Any]] = []

    cycle_group_cols = ["dataset", "battery_id", "cycle"]
    phase_group_cols = ["dataset", "battery_id", "cycle", "phase"]

    cycle_grouped = df.groupby(cycle_group_cols, observed=True, sort=False)
    cycle_iter: Iterable[Any] = cycle_grouped
    if inner_progress:
        cycle_iter = tqdm(cycle_grouped, total=cycle_grouped.ngroups, desc="Ciclos", leave=False)

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
        phase_iter = tqdm(phase_grouped, total=phase_grouped.ngroups, desc="Fases/segmentos", leave=False)

    for (dataset, battery_id, cycle, phase), group in phase_iter:
        phase_row = make_feature_row(
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
        rows.append(phase_row)

        segments = segment_slices(group, min_gap_seconds=min_gap_seconds, gap_factor=gap_factor)
        for idx, seg in enumerate(segments):
            seg_row = make_feature_row(
                group=seg,
                level="phase_segment",
                dataset=dataset,
                battery_id=battery_id,
                cycle=cycle,
                phase=phase,
                segment_index=idx,
                min_gap_seconds=min_gap_seconds,
                gap_factor=gap_factor,
            )
            seg_row["segment_total_in_phase"] = len(segments)
            rows.append(seg_row)

    features = pd.DataFrame(rows)

    # Downcast final.
    for col in features.select_dtypes(include=["float64"]).columns:
        features[col] = pd.to_numeric(features[col], downcast="float")
    for col in features.select_dtypes(include=["int64"]).columns:
        features[col] = pd.to_numeric(features[col], downcast="integer")

    for col in ["feature_level", "dataset", "battery_id", "phase"]:
        if col in features.columns:
            features[col] = features[col].astype("string")

    return features


# =============================================================================
# Processamento de cada job
# =============================================================================

def process_job(job_dict: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    job = Job(**job_dict)
    input_path = Path(job.input_path)
    output_path = Path(job.output_path)
    done_path = Path(job.done_path)
    error_path = Path(job.error_path)

    resume = bool(options.get("resume", True))
    force = bool(options.get("force", False))
    compression = str(options.get("compression", "snappy"))
    min_gap_seconds = float(options.get("min_gap_seconds", 60.0))
    gap_factor = float(options.get("gap_factor", 20.0))
    inner_progress = bool(options.get("inner_progress", False))

    start = now_iso()

    result: Dict[str, Any] = {
        "job_id": job.job_id,
        "dataset": job.dataset,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "unknown",
        "started_at": start,
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
            for p in [output_path, done_path, error_path]:
                if p.exists():
                    p.unlink()

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

        # Substituição atômica no mesmo diretório.
        if output_path.exists():
            output_path.unlink()
        tmp_path.replace(output_path)

        # Marcador de conclusão. Se o processamento parar antes daqui, o job será refeito.
        done_payload = {
            "job_id": job.job_id,
            "dataset": job.dataset,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "finished_at": now_iso(),
            "input_rows": int(input_rows),
            "feature_rows": int(len(features)),
        }
        done_path.write_text(json.dumps(done_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if error_path.exists():
            error_path.unlink()

        result.update(
            {
                "status": "done",
                "finished_at": now_iso(),
                "input_rows": int(input_rows),
                "feature_rows": int(len(features)),
            }
        )

        del features
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
        gc.collect()
        return result


# =============================================================================
# Execução
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
    pd.DataFrame(job_records).to_csv(reports_dir / "job_plan.csv", index=False, encoding="utf-8-sig")

    if not jobs:
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []

    if workers <= 1:
        for job_dict in tqdm(job_records, total=len(job_records), desc="Extraindo features"):
            result = process_job(job_dict, {**options, "inner_progress": bool(options.get("inner_progress", False))})
            results.append(result)
            write_jsonl(status_jsonl, result)
    else:
        # No Windows, workers excessivos costumam aumentar muito a RAM.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_job, job_dict, {**options, "inner_progress": False}) for job_dict in job_records]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extraindo features"):
                result = future.result()
                results.append(result)
                write_jsonl(status_jsonl, result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(reports_dir / "summary.csv", index=False, encoding="utf-8-sig")

    failed = results_df[results_df["status"] == "failed"] if not results_df.empty else pd.DataFrame()
    if not failed.empty:
        failed.to_csv(reports_dir / "failures.csv", index=False, encoding="utf-8-sig")

    return results_df


def write_output_index(output_root: Path, reports_dir: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for path in output_root.rglob("*.features.parquet"):
        done_path = Path(str(path) + ".done")
        records.append(
            {
                "dataset": path.parent.parent.name if path.parent.name == "parts" else path.parent.name,
                "feature_file": str(path),
                "done": done_path.exists(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["dataset", "feature_file"])
    df.to_csv(reports_dir / "output_index.csv", index=False, encoding="utf-8-sig")
    return df


def print_summary(results: pd.DataFrame, reports_dir: Path, output_root: Path) -> None:
    if results.empty:
        print("Nenhum job encontrado.")
        return

    counts = results["status"].value_counts(dropna=False).to_dict()
    total = len(results)
    done = counts.get("done", 0)
    skipped = counts.get("skipped_done", 0)
    failed = counts.get("failed", 0)

    print("\nResumo da extração")
    print("-" * 60)
    print(f"Total de jobs:       {total}")
    print(f"Concluídos agora:    {done}")
    print(f"Pulados por resume:  {skipped}")
    print(f"Falharam:            {failed}")
    print(f"Saída:               {output_root}")
    print(f"Relatórios:          {reports_dir}")

    if failed:
        print("\nAlguns jobs falharam. Veja:")
        print(f"  {reports_dir / 'failures.csv'}")
        print("  e os arquivos .error.txt ao lado dos Parquets de saída.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extração hierárquica de features de baterias com resume, tqdm e fastparquet."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Pasta de entrada. Padrão: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Pasta de saída das features. Padrão: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help=f"Pasta dos relatórios. Padrão: {DEFAULT_REPORTS_ROOT}",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Nome de uma base ou pasta. Pode repetir ou separar por vírgula. Ex.: --dataset hust --dataset matr",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Processa todas as bases encontradas em --input-root. Se nenhum --dataset for passado, este comportamento já é usado.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Número de processos paralelos. Recomenda-se 1 ou 2 para economizar memória.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Pula jobs que já têm arquivo .parquet e marcador .done. Ativo por padrão.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Desativa retomada automática.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocessa mesmo que já exista .done. Cuidado: sobrescreve os Parquets correspondentes.",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "gzip", "brotli", "None", "none"],
        help="Compressão Parquet. Padrão: snappy.",
    )
    parser.add_argument(
        "--suffixes",
        default=",".join(SUPPORTED_SUFFIXES),
        help="Extensões de entrada separadas por vírgula. Padrão: .parquet,.csv,.txt,.dat",
    )
    parser.add_argument(
        "--min-gap-seconds",
        type=float,
        default=60.0,
        help="Salto mínimo de tempo para dividir fases em segmentos. Padrão: 60 s.",
    )
    parser.add_argument(
        "--gap-factor",
        type=float,
        default=20.0,
        help="Fator sobre o dt mediano para detectar quebra de segmento. Padrão: 20.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limita número de arquivos por dataset. Útil para teste.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas monta o plano de jobs; não executa a extração.",
    )
    parser.add_argument(
        "--inner-progress",
        action="store_true",
        help="Mostra progresso interno de ciclos/fases. Use apenas com --workers 1.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    reports_root: Path = args.reports_root

    if not input_root.exists():
        raise FileNotFoundError(f"Pasta de entrada não encontrada: {input_root}")

    requested_datasets = parse_dataset_args(args.dataset)
    datasets = discover_datasets(input_root, requested_datasets, use_all=args.all)

    suffixes = tuple(s.strip().lower() for s in str(args.suffixes).split(",") if s.strip())
    if not suffixes:
        suffixes = SUPPORTED_SUFFIXES

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    reports_dir = reports_root / run_id
    ensure_dir(reports_dir)
    ensure_dir(output_root)

    jobs = build_jobs(
        input_root=input_root,
        output_root=output_root,
        datasets=datasets,
        suffixes=suffixes,
        max_files=args.max_files,
    )

    # Remove jobs já concluídos do plano quando resume está ativo e force não está ativo.
    # Mesmo assim, eles são contabilizados como skipped_done por process_job se passarem.
    # Mantemos no plano para transparência.

    print("Plano de execução")
    print("-" * 60)
    print(f"Entrada:        {input_root}")
    print(f"Saída:          {output_root}")
    print(f"Relatórios:     {reports_dir}")
    print(f"Datasets:       {', '.join([d[0] for d in datasets]) if datasets else 'nenhum'}")
    print(f"Arquivos/jobs:  {len(jobs)}")
    print(f"Workers:        {args.workers}")
    print(f"Resume:         {args.resume}")
    print(f"Force:          {args.force}")

    pd.DataFrame([asdict(job) for job in jobs]).to_csv(
        reports_dir / "job_plan.csv", index=False, encoding="utf-8-sig"
    )

    if args.dry_run:
        print("\nDry-run concluído. Nenhum arquivo foi processado.")
        print(f"Plano salvo em: {reports_dir / 'job_plan.csv'}")
        return

    compression: Optional[str]
    if str(args.compression).lower() == "none":
        compression = None
    else:
        compression = args.compression

    options: Dict[str, Any] = {
        "resume": bool(args.resume),
        "force": bool(args.force),
        "compression": compression,
        "min_gap_seconds": float(args.min_gap_seconds),
        "gap_factor": float(args.gap_factor),
        "inner_progress": bool(args.inner_progress and args.workers <= 1),
    }

    workers = max(1, int(args.workers))
    results = run_jobs(
        jobs=jobs,
        reports_dir=reports_dir,
        options=options,
        workers=workers,
    )

    write_output_index(output_root=output_root, reports_dir=reports_dir)
    print_summary(results=results, reports_dir=reports_dir, output_root=output_root)


if __name__ == "__main__":
    # Necessário no Windows para ProcessPoolExecutor.
    main()
