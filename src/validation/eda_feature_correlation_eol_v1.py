#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EDA de correlação das features de RUL para baterias de íons de lítio.

Objetivo
--------
Executar uma análise exploratória das características extraídas, separando carga
(charge) e descarga (discharge), usando apenas trajetórias que atingiram EOL e
removendo ciclos/fases com RUL igual a zero. A análise calcula:

1. Correlação de Spearman de cada feature com a variável alvo.
2. Ranking das melhores features por fase e por grupo.
3. Comparação entre grupos de features: Q(V), V(Q), temperatura, tensão,
   corrente, capacidade, energia, potência, tempo/amostragem e estrutura de estágio.
4. Redundância interna por grupo usando correlação cruzada Spearman absoluta.
5. Efeito da resolução de interpolação p = 3, 9, 15 e 30 para Q(V) e V(Q):
   ganho médio de correlação com o alvo versus aumento de redundância.
6. Mapas de calor por grupo, sempre com escala fixa de 0 a 1.

Uso típico no PowerShell
-----------------------
$py = "C:\\Users\\Giovanny\\anaconda3\\envs\\batteries\\python.exe"
$project = "D:\\USUÁRIOS\\Marcellus\\OneDrive\\Doutorado\\BATERIAS - Ressubmissão\\Batteries"
$data = "C:\\Artigo Baterias de Litio\\Batteries\\data"

& $py "$project\\src\\validation\\eda_feature_correlation_eol_v1.py" `
  --features-root "$data\\processed\\features_hierarchical_v3" `
  --output-root "C:\\Artigo Baterias de Litio\\Batteries\\reports\\eda_feature_correlation_eol" `
  --target auto `
  --feature-level phase `
  --min-valid-rows 30 `
  --min-unique 3 `
  --top-n 30 `
  --heatmap-max-features 60

Observação
----------
O script não reconstrói features. Ele audita e analisa os arquivos .features.parquet
já gerados. Se as colunas QV_p003_001, VQ_p003_001 etc. ainda não existirem, o
relatório vai indicar que esses grupos estão ausentes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuração padrão
# =============================================================================

DEFAULT_FEATURES_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/processed/features_hierarchical_v3")
DEFAULT_OUTPUT_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/reports/eda_feature_correlation_eol")

PHASES = ("charge", "discharge")
RESOLUTIONS = (3, 9, 15, 30)

TARGET_PRIORITY = [
    "target_rul80_steps_clipped",
    "target_rul80_cycle_index_clipped",
    "remaining_cycles",
    "input_remaining_cycles",
    "target_rul70_steps_clipped",
    "target_rul70_cycle_index_clipped",
]

EOL_FOUND_BY_TARGET = {
    "target_rul80_steps_clipped": "target_eol80_found",
    "target_rul80_cycle_index_clipped": "target_eol80_found",
    "target_rul70_steps_clipped": "target_eol70_found",
    "target_rul70_cycle_index_clipped": "target_eol70_found",
}

AFTER_EOL_BY_TARGET = {
    "target_rul80_steps_clipped": "target_after_eol80",
    "target_rul80_cycle_index_clipped": "target_after_eol80",
    "target_rul70_steps_clipped": "target_after_eol70",
    "target_rul70_cycle_index_clipped": "target_after_eol70",
}

# Colunas que não devem entrar como features de entrada na correlação.
# A ideia é evitar vazamento de alvo, identificação, ordenação temporal pura e flags.
EXCLUDE_EXACT = {
    "dataset", "phase", "phase_folder", "feature_level", "source_file", "source_path",
    "source_file_key", "source_row_id", "battery_id", "battery_index", "cell_id_original",
    "source_cycle_id", "cycle_pair_id", "stage_id", "created_at", "adapter_version",
    "cleaning_version", "extractor_version", "job_id", "row_id",
    "cycle_life", "remaining_cycles", "remaining_cycles_norm", "soh", "input_soh",
    "input_remaining_cycles", "input_remaining_cycles_norm", "input_cycle_life",
    "target_rul80_steps_clipped", "target_rul80_steps_signed",
    "target_rul80_cycle_index_clipped", "target_rul80_cycle_index_signed",
    "target_rul70_steps_clipped", "target_rul70_steps_signed",
    "target_rul70_cycle_index_clipped", "target_rul70_cycle_index_signed",
    "target_eol80_order_index", "target_eol80_cycle_index", "target_eol80_found",
    "target_after_eol80", "target_eol70_order_index", "target_eol70_cycle_index",
    "target_eol70_found", "target_after_eol70", "target_initial_capacity_method",
    "target_capacity_source", "capacity_validation_status", "capacity_validation_notes",
    "input_is_valid_cycle", "input_is_model_usable", "input_exclude_reason",
    "input_cleaning_warnings", "is_valid_cycle", "is_model_usable", "exclude_reason",
    "cleaning_warnings",
}

EXCLUDE_PREFIXES = (
    "target_",  # evita vazamento, exceto se você remover manualmente depois
)

# Alguns target_* são úteis apenas para filtro, não para feature. Mantemos todos fora.

QUALITY_KEYWORDS = (
    "valid", "usable", "warning", "exclude", "has_", "finite_", "source",
    "quality", "flag", "removed", "trim_",
)


@dataclass
class PhaseFilterSummary:
    phase: str
    rows_loaded: int
    rows_after_feature_level: int
    rows_after_eol_found: int
    rows_after_positive_target: int
    target_column: str
    eol_found_column: str
    after_eol_column: str
    batteries_after_filter: int
    datasets_after_filter: int


# =============================================================================
# Utilitários gerais
# =============================================================================

def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_filename(text: str, max_len: int = 150) -> str:
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(text)).strip("_")
    return (text or "file")[:max_len]


def infer_phase_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "charge" in parts:
        return "charge"
    if "discharge" in parts:
        return "discharge"
    name = path.name.lower()
    if "charge" in name and "discharge" not in name:
        return "charge"
    if "discharge" in name:
        return "discharge"
    return "unknown"


def infer_dataset_from_path(path: Path) -> str:
    name = path.stem.lower()
    name = re.sub(r"__(.)*", "", name)
    name = re.sub(r"_(charge|discharge).*", "", name)
    name = re.sub(r"(charge|discharge)_", "", name)
    return name.upper() if name else "UNKNOWN"


def is_bool_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    vals = series.dropna().unique()
    if len(vals) == 0:
        return False
    if len(vals) <= 2 and set(pd.Series(vals).astype(str).str.lower()).issubset({"true", "false", "0", "1"}):
        return True
    return False


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "sim"])


def numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = {}
    for col in columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # Evita tentar converter listas/vetores salvos como object.
            sample = df[col].dropna().head(5)
            if len(sample) and any(isinstance(v, (list, tuple, dict, np.ndarray)) for v in sample):
                continue
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                out[col] = converted
    return pd.DataFrame(out, index=df.index)


def choose_target(df: pd.DataFrame, requested: str = "auto") -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Target solicitado não existe no dataframe: {requested}")
        return requested

    for col in TARGET_PRIORITY:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() >= 10 and (s > 0).sum() >= 5:
                return col

    raise ValueError(
        "Nenhuma coluna alvo válida foi encontrada. Use --target com uma coluna existente. "
        f"Prioridade testada: {TARGET_PRIORITY}"
    )


def discover_feature_files(features_root: Path) -> List[Path]:
    candidates = []
    for pattern in ("*.features.parquet", "*.parquet"):
        candidates.extend(features_root.rglob(pattern))

    unique = []
    seen = set()
    for p in sorted(candidates, key=lambda x: str(x).lower()):
        s = str(p).lower()
        if s in seen:
            continue
        seen.add(s)
        if "_targets" in s or p.name.startswith(".") or p.suffix.lower() != ".parquet":
            continue
        if ".tmp" in p.name.lower():
            continue
        unique.append(p)
    return unique


def read_feature_files(files: Sequence[Path], phases: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    inventory = []

    for path in files:
        phase_inferred = infer_phase_from_path(path)
        if phase_inferred not in phases and phase_inferred != "unknown":
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            inventory.append({
                "path": str(path), "status": "read_failed", "error": str(exc),
                "phase_inferred": phase_inferred, "dataset_inferred": infer_dataset_from_path(path),
                "rows": np.nan, "columns": np.nan,
            })
            continue

        if df.empty:
            inventory.append({
                "path": str(path), "status": "empty", "error": "", "phase_inferred": phase_inferred,
                "dataset_inferred": infer_dataset_from_path(path), "rows": 0, "columns": len(df.columns),
            })
            continue

        if "phase_folder" not in df.columns:
            df["phase_folder"] = phase_inferred
        if "phase" not in df.columns:
            df["phase"] = phase_inferred
        if "dataset" not in df.columns:
            df["dataset"] = infer_dataset_from_path(path)

        # Normaliza phase_folder quando a coluna existe, mas tem valores ausentes.
        df["phase_folder"] = df["phase_folder"].fillna(phase_inferred).astype(str).str.lower()
        df["_source_feature_file"] = str(path)

        if phase_inferred in phases:
            df = df[df["phase_folder"].isin(phases) | (df["phase_folder"] == "unknown")].copy()
        else:
            df = df[df["phase_folder"].isin(phases)].copy()

        frames.append(df)
        inventory.append({
            "path": str(path), "status": "loaded", "error": "", "phase_inferred": phase_inferred,
            "dataset_inferred": infer_dataset_from_path(path), "rows": len(df), "columns": len(df.columns),
        })

    inv = pd.DataFrame(inventory)
    if not frames:
        return pd.DataFrame(), inv

    all_df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    return all_df, inv


# =============================================================================
# Agrupamento de features
# =============================================================================

def qv_interp_pattern(res: int) -> re.Pattern:
    # Aceita QV_p003_001, qv_p3_001, QV_003_001, qv_3_001.
    r = str(res).zfill(3)
    return re.compile(rf"(?i)^qv[_-]?p?0*{res}[_-]\d+$|^qv[_-]?p?{r}[_-]\d+$")


def vq_interp_pattern(res: int) -> re.Pattern:
    r = str(res).zfill(3)
    return re.compile(rf"(?i)^vq[_-]?p?0*{res}[_-]\d+$|^vq[_-]?p?{r}[_-]\d+$")


def is_qv_any(name: str) -> bool:
    return re.match(r"(?i)^qv[_-]", name) is not None


def is_vq_any(name: str) -> bool:
    return re.match(r"(?i)^vq[_-]", name) is not None


def assign_feature_group(feature: str) -> str:
    f = feature.lower()
    f_no_input = f[6:] if f.startswith("input_") else f

    for res in RESOLUTIONS:
        if qv_interp_pattern(res).match(feature):
            return f"QV_interp_p{res:03d}"
        if vq_interp_pattern(res).match(feature):
            return f"VQ_interp_p{res:03d}"

    if is_qv_any(feature):
        return "QV_curve_stats"
    if is_vq_any(feature):
        return "VQ_curve_stats"

    if f_no_input.startswith("temperature_") or f_no_input.startswith("temp_"):
        return "temperature_stats"
    if f_no_input.startswith("voltage_"):
        return "voltage_stats"
    if f_no_input.startswith("current_"):
        return "current_stats"
    if f_no_input.startswith("capacity_") or f_no_input.startswith("phase_capacity"):
        return "capacity_stats"
    if f_no_input.startswith("energy_") or f_no_input.startswith("phase_energy"):
        return "energy_stats"
    if f_no_input.startswith("power_"):
        return "power_stats"
    if f_no_input.startswith("time_") or "sample_period" in f_no_input or "sampling" in f_no_input or f_no_input == "duration_s":
        return "time_sampling_stats"
    if f_no_input.startswith("stage_") or f_no_input.startswith("active_") or "segment" in f_no_input or "transition" in f_no_input:
        return "stage_structure"
    if "event" in f_no_input or "peak" in f_no_input or "spike" in f_no_input:
        return "individual_events"
    if "trend" in f_no_input or "diff_prev" in f_no_input or "delta_prev" in f_no_input:
        return "temporal_degradation"
    if any(k in f_no_input for k in QUALITY_KEYWORDS):
        return "quality_or_availability"
    return "other_numeric"


def feature_is_excluded(feature: str) -> bool:
    f = feature.lower()
    if feature in EXCLUDE_EXACT or f in EXCLUDE_EXACT:
        return True
    if any(f.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return True
    if f.startswith("_"):
        return True
    if f.endswith("_id") or f.endswith("_key"):
        return True
    return False


def select_candidate_features(df: pd.DataFrame, target_col: str, min_valid_rows: int, min_unique: int) -> Tuple[List[str], pd.DataFrame]:
    candidate_cols = []
    for col in df.columns:
        if col == target_col or feature_is_excluded(col):
            continue
        if col.startswith("_source_"):
            continue
        if is_bool_like(df[col]):
            # Flags e qualidade não são usadas como features preditivas nesta EDA.
            continue
        candidate_cols.append(col)

    num = numeric_frame(df, candidate_cols)
    keep = []
    for col in num.columns:
        s = num[col]
        if s.notna().sum() < min_valid_rows:
            continue
        if s.nunique(dropna=True) < min_unique:
            continue
        keep.append(col)

    return keep, num[keep]


# =============================================================================
# Correlação e redundância
# =============================================================================

def spearman_with_target(num_df: pd.DataFrame, target: pd.Series, min_valid_rows: int) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(target, errors="coerce")

    for col in num_df.columns:
        x = pd.to_numeric(num_df[col], errors="coerce")
        valid = x.notna() & y.notna()
        n = int(valid.sum())
        if n < min_valid_rows:
            continue
        if x[valid].nunique() < 3 or y[valid].nunique() < 3:
            continue
        rho = x[valid].corr(y[valid], method="spearman")
        if pd.isna(rho):
            continue
        rows.append({
            "feature": col,
            "feature_group": assign_feature_group(col),
            "spearman_rho": float(rho),
            "abs_spearman_rho": float(abs(rho)),
            "n_valid": n,
            "missing_fraction": float(1.0 - n / len(y)) if len(y) else np.nan,
            "n_unique": int(x[valid].nunique()),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_spearman_rho", "n_valid"], ascending=[False, False]).reset_index(drop=True)
    return out


def corr_abs_matrix(df: pd.DataFrame, features: Sequence[str], min_valid_rows: int) -> pd.DataFrame:
    if len(features) < 2:
        return pd.DataFrame()
    x = df[list(features)].copy()

    # Remove colunas com baixa cobertura/variância após o subconjunto.
    keep = []
    for col in x.columns:
        s = pd.to_numeric(x[col], errors="coerce")
        if s.notna().sum() >= min_valid_rows and s.nunique(dropna=True) >= 3:
            keep.append(col)
    if len(keep) < 2:
        return pd.DataFrame()

    corr = x[keep].corr(method="spearman").abs()
    return corr


def redundancy_summary(corr_abs: pd.DataFrame, threshold_90: float = 0.90, threshold_95: float = 0.95) -> Dict[str, float]:
    if corr_abs.empty or corr_abs.shape[0] < 2:
        return {
            "n_features": int(corr_abs.shape[0]) if not corr_abs.empty else 0,
            "n_pairs": 0,
            "mean_abs_cross_spearman": np.nan,
            "median_abs_cross_spearman": np.nan,
            "max_abs_cross_spearman": np.nan,
            "pct_pairs_abs_ge_0p90": np.nan,
            "pct_pairs_abs_ge_0p95": np.nan,
        }
    arr = corr_abs.to_numpy(dtype=float)
    mask = np.triu(np.ones_like(arr, dtype=bool), k=1)
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "n_features": int(corr_abs.shape[0]), "n_pairs": 0,
            "mean_abs_cross_spearman": np.nan, "median_abs_cross_spearman": np.nan,
            "max_abs_cross_spearman": np.nan, "pct_pairs_abs_ge_0p90": np.nan,
            "pct_pairs_abs_ge_0p95": np.nan,
        }
    return {
        "n_features": int(corr_abs.shape[0]),
        "n_pairs": int(vals.size),
        "mean_abs_cross_spearman": float(np.mean(vals)),
        "median_abs_cross_spearman": float(np.median(vals)),
        "max_abs_cross_spearman": float(np.max(vals)),
        "pct_pairs_abs_ge_0p90": float(np.mean(vals >= threshold_90)),
        "pct_pairs_abs_ge_0p95": float(np.mean(vals >= threshold_95)),
    }


def select_features_for_heatmap(target_corr: pd.DataFrame, group: str, max_features: int) -> List[str]:
    sub = target_corr[target_corr["feature_group"] == group].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("abs_spearman_rho", ascending=False)
    return sub["feature"].head(max_features).tolist()


# =============================================================================
# Plotagem
# =============================================================================

def save_heatmap(matrix: pd.DataFrame, path: Path, title: str, figsize_base: float = 0.32) -> None:
    if matrix.empty:
        return
    ensure_dir(path.parent)
    n = matrix.shape[0]
    width = max(7.0, min(22.0, n * figsize_base + 4.0))
    height = max(6.0, min(22.0, n * figsize_base + 3.0))

    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    labels = [str(c) for c in matrix.columns]
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|Spearman|", rotation=90)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_barh(df: pd.DataFrame, x: str, y: str, path: Path, title: str, xlabel: str) -> None:
    if df.empty:
        return
    ensure_dir(path.parent)
    d = df.copy().sort_values(x, ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(d))), dpi=180)
    ax.barh(d[y].astype(str), d[x].astype(float))
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    for key, sub in df.groupby("curve_type"):
        sub = sub.sort_values("resolution")
        ax.plot(sub["resolution"], sub["mean_abs_spearman"], marker="o", label=f"{key}: média |rho alvo|")
        ax.plot(sub["resolution"], sub["mean_abs_cross_spearman"], marker="s", linestyle="--", label=f"{key}: redundância média")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Número de pontos de interpolação")
    ax.set_ylabel("Valor médio na escala 0–1")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_top_features_plot(target_corr: pd.DataFrame, phase: str, out_dir: Path, top_n: int) -> None:
    top = target_corr.head(top_n).copy()
    if top.empty:
        return
    top["label"] = top["feature_group"] + " | " + top["feature"]
    save_barh(
        top,
        x="abs_spearman_rho",
        y="label",
        path=out_dir / "plots" / "target" / f"{phase}_top_{top_n}_features_abs_spearman.png",
        title=f"{phase}: top {top_n} features por |Spearman| com o alvo",
        xlabel="|Spearman(feature, alvo)|",
    )


# =============================================================================
# Pipeline principal por fase
# =============================================================================

def filter_phase_df(
    df: pd.DataFrame,
    phase: str,
    target_col: str,
    feature_level: str,
    require_eol_found: bool,
    drop_zero_target: bool,
) -> Tuple[pd.DataFrame, PhaseFilterSummary]:
    phase_df = df[df["phase_folder"].astype(str).str.lower() == phase].copy()
    rows_loaded = len(phase_df)

    if feature_level.lower() != "all" and "feature_level" in phase_df.columns:
        phase_df = phase_df[phase_df["feature_level"].astype(str).str.lower() == feature_level.lower()].copy()
    rows_after_feature_level = len(phase_df)

    eol_col = EOL_FOUND_BY_TARGET.get(target_col, "")
    if require_eol_found and eol_col and eol_col in phase_df.columns:
        phase_df = phase_df[to_bool_series(phase_df[eol_col])].copy()
    rows_after_eol_found = len(phase_df)

    after_eol_col = AFTER_EOL_BY_TARGET.get(target_col, "")
    if after_eol_col and after_eol_col in phase_df.columns:
        # Remove tudo marcado como depois do EOL; o filtro target > 0 remove o ciclo EOL também.
        phase_df = phase_df[~to_bool_series(phase_df[after_eol_col])].copy()

    phase_df[target_col] = pd.to_numeric(phase_df[target_col], errors="coerce")
    if drop_zero_target:
        phase_df = phase_df[phase_df[target_col] > 0].copy()
    else:
        phase_df = phase_df[phase_df[target_col].notna()].copy()
    rows_after_positive_target = len(phase_df)

    summary = PhaseFilterSummary(
        phase=phase,
        rows_loaded=rows_loaded,
        rows_after_feature_level=rows_after_feature_level,
        rows_after_eol_found=rows_after_eol_found,
        rows_after_positive_target=rows_after_positive_target,
        target_column=target_col,
        eol_found_column=eol_col,
        after_eol_column=after_eol_col,
        batteries_after_filter=int(phase_df["battery_id"].nunique()) if "battery_id" in phase_df.columns else 0,
        datasets_after_filter=int(phase_df["dataset"].nunique()) if "dataset" in phase_df.columns else 0,
    )
    return phase_df, summary


def summarize_by_group(target_corr: pd.DataFrame) -> pd.DataFrame:
    if target_corr.empty:
        return pd.DataFrame()
    out = (
        target_corr.groupby("feature_group")
        .agg(
            n_features=("feature", "count"),
            mean_abs_spearman=("abs_spearman_rho", "mean"),
            median_abs_spearman=("abs_spearman_rho", "median"),
            max_abs_spearman=("abs_spearman_rho", "max"),
            mean_missing_fraction=("missing_fraction", "mean"),
        )
        .reset_index()
        .sort_values(["mean_abs_spearman", "max_abs_spearman"], ascending=[False, False])
    )
    return out


def make_top_by_group(target_corr: pd.DataFrame, top_n_each: int) -> pd.DataFrame:
    if target_corr.empty:
        return pd.DataFrame()
    return (
        target_corr.sort_values("abs_spearman_rho", ascending=False)
        .groupby("feature_group", group_keys=False)
        .head(top_n_each)
        .reset_index(drop=True)
    )


def analyze_phase(
    df_all: pd.DataFrame,
    phase: str,
    target_col: str,
    out_root: Path,
    feature_level: str,
    require_eol_found: bool,
    drop_zero_target: bool,
    min_valid_rows: int,
    min_unique: int,
    top_n: int,
    heatmap_max_features: int,
) -> Dict[str, pd.DataFrame]:
    phase_df, filt_summary = filter_phase_df(
        df_all,
        phase=phase,
        target_col=target_col,
        feature_level=feature_level,
        require_eol_found=require_eol_found,
        drop_zero_target=drop_zero_target,
    )

    phase_out = out_root / phase
    ensure_dir(phase_out / "tables")
    ensure_dir(phase_out / "plots" / "correlations")
    ensure_dir(phase_out / "plots" / "redundancy")
    ensure_dir(phase_out / "plots" / "target")

    pd.DataFrame([asdict(filt_summary)]).to_csv(phase_out / "tables" / f"{phase}_filter_summary.csv", index=False)

    if phase_df.empty:
        return {
            "filter_summary": pd.DataFrame([asdict(filt_summary)]),
            "target_corr": pd.DataFrame(),
            "group_summary": pd.DataFrame(),
            "redundancy_summary": pd.DataFrame(),
            "curve_tradeoff": pd.DataFrame(),
            "top_by_group": pd.DataFrame(),
            "feature_inventory": pd.DataFrame(),
        }

    features, num = select_candidate_features(phase_df, target_col, min_valid_rows=min_valid_rows, min_unique=min_unique)
    feature_inventory = pd.DataFrame({
        "feature": features,
        "feature_group": [assign_feature_group(f) for f in features],
        "phase": phase,
    }).sort_values(["feature_group", "feature"])
    feature_inventory.to_csv(phase_out / "tables" / f"{phase}_feature_inventory.csv", index=False)

    target_corr = spearman_with_target(num, phase_df[target_col], min_valid_rows=min_valid_rows)
    if not target_corr.empty:
        target_corr.insert(0, "phase", phase)
    target_corr.to_csv(phase_out / "tables" / f"{phase}_target_spearman_rank_all.csv", index=False)

    group_summary = summarize_by_group(target_corr)
    if not group_summary.empty:
        group_summary.insert(0, "phase", phase)
    group_summary.to_csv(phase_out / "tables" / f"{phase}_target_spearman_by_group.csv", index=False)

    top_by_group = make_top_by_group(target_corr, top_n_each=min(10, top_n))
    top_by_group.to_csv(phase_out / "tables" / f"{phase}_top_features_by_group.csv", index=False)

    save_top_features_plot(target_corr, phase, phase_out, top_n=top_n)
    save_barh(
        group_summary.head(30),
        x="mean_abs_spearman",
        y="feature_group",
        path=phase_out / "plots" / "target" / f"{phase}_group_mean_abs_spearman.png",
        title=f"{phase}: média de |Spearman| com o alvo por grupo de features",
        xlabel="Média de |Spearman(feature, alvo)|",
    )

    # Redundância por grupo e mapas de calor por grupo.
    redundancy_rows = []
    groups = sorted(feature_inventory["feature_group"].unique()) if not feature_inventory.empty else []
    for group in groups:
        selected = select_features_for_heatmap(target_corr, group, heatmap_max_features)
        if len(selected) < 2:
            redundancy_rows.append({"phase": phase, "feature_group": group, **redundancy_summary(pd.DataFrame())})
            continue
        corr_abs = corr_abs_matrix(num, selected, min_valid_rows=min_valid_rows)
        summ = redundancy_summary(corr_abs)
        redundancy_rows.append({"phase": phase, "feature_group": group, **summ})
        if not corr_abs.empty:
            save_heatmap(
                corr_abs,
                phase_out / "plots" / "redundancy" / f"{phase}_{safe_filename(group)}_abs_spearman_heatmap.png",
                title=f"{phase}: redundância interna - {group} (|Spearman|, escala 0–1)",
            )

    redundancy_df = pd.DataFrame(redundancy_rows)
    if not redundancy_df.empty:
        redundancy_df = redundancy_df.sort_values(["mean_abs_cross_spearman", "n_features"], ascending=[False, False])
    redundancy_df.to_csv(phase_out / "tables" / f"{phase}_group_redundancy_summary.csv", index=False)

    # Trade-off específico Q(V) e V(Q): informação versus redundância por resolução.
    tradeoff_rows = []
    for curve_type in ("QV", "VQ"):
        for res in RESOLUTIONS:
            group = f"{curve_type}_interp_p{res:03d}"
            group_features = feature_inventory.loc[feature_inventory["feature_group"] == group, "feature"].tolist()
            corr_sub = target_corr[target_corr["feature_group"] == group].copy() if not target_corr.empty else pd.DataFrame()
            if len(group_features) >= 2:
                corr_abs = corr_abs_matrix(num, group_features, min_valid_rows=min_valid_rows)
                red = redundancy_summary(corr_abs)
            else:
                red = redundancy_summary(pd.DataFrame())

            if corr_sub.empty:
                mean_abs = median_abs = max_abs = np.nan
                best_feature = ""
                best_rho = np.nan
            else:
                mean_abs = float(corr_sub["abs_spearman_rho"].mean())
                median_abs = float(corr_sub["abs_spearman_rho"].median())
                max_abs = float(corr_sub["abs_spearman_rho"].max())
                best = corr_sub.sort_values("abs_spearman_rho", ascending=False).iloc[0]
                best_feature = str(best["feature"])
                best_rho = float(best["spearman_rho"])

            tradeoff_rows.append({
                "phase": phase,
                "curve_type": curve_type,
                "resolution": res,
                "feature_group": group,
                "n_features_available": len(group_features),
                "n_features_correlated_with_target": int(len(corr_sub)),
                "mean_abs_spearman": mean_abs,
                "median_abs_spearman": median_abs,
                "max_abs_spearman": max_abs,
                "best_feature": best_feature,
                "best_feature_spearman_rho": best_rho,
                **red,
            })

    curve_tradeoff = pd.DataFrame(tradeoff_rows)
    curve_tradeoff.to_csv(phase_out / "tables" / f"{phase}_QV_VQ_resolution_tradeoff.csv", index=False)
    save_tradeoff_plot(
        curve_tradeoff,
        phase_out / "plots" / "correlations" / f"{phase}_QV_VQ_information_redundancy_tradeoff.png",
        title=f"{phase}: Q(V) e V(Q) - informação com alvo versus redundância",
    )

    # Heatmaps específicos das interpolações QV/VQ por resolução com escala fixa 0-1.
    for curve_type in ("QV", "VQ"):
        for res in RESOLUTIONS:
            group = f"{curve_type}_interp_p{res:03d}"
            feats = feature_inventory.loc[feature_inventory["feature_group"] == group, "feature"].tolist()
            if len(feats) < 2:
                continue
            corr_abs = corr_abs_matrix(num, feats, min_valid_rows=min_valid_rows)
            if not corr_abs.empty:
                save_heatmap(
                    corr_abs,
                    phase_out / "plots" / "redundancy" / f"{phase}_{group}_abs_spearman_heatmap.png",
                    title=f"{phase}: {group} - correlação cruzada interna (|Spearman|, escala 0–1)",
                )

    return {
        "filter_summary": pd.DataFrame([asdict(filt_summary)]),
        "target_corr": target_corr,
        "group_summary": group_summary,
        "redundancy_summary": redundancy_df,
        "curve_tradeoff": curve_tradeoff,
        "top_by_group": top_by_group,
        "feature_inventory": feature_inventory,
    }


# =============================================================================
# Relatório markdown
# =============================================================================

def fmt_num(x: object, nd: int = 3) -> str:
    try:
        v = float(x)
        if math.isfinite(v):
            return f"{v:.{nd}f}"
    except Exception:
        pass
    return "NA"


def make_report(out_root: Path, results: Dict[str, Dict[str, pd.DataFrame]], args: argparse.Namespace, loaded_inventory: pd.DataFrame) -> None:
    lines = []
    lines.append("# EDA de correlação das features com RUL\n")
    lines.append(f"Execução: `{datetime.now().isoformat(timespec='seconds')}`\n")
    lines.append("## Configuração\n")
    lines.append(f"- Pasta de features: `{args.features_root}`")
    lines.append(f"- Variável alvo: `{args.target_resolved}`")
    lines.append(f"- Feature level: `{args.feature_level}`")
    lines.append(f"- Apenas trajetórias com EOL encontrado: `{args.require_eol_found}`")
    lines.append(f"- Remover RUL igual a zero: `{args.drop_zero_target}`")
    lines.append(f"- Escala dos mapas de calor: `0 a 1`, usando `|Spearman|`\n")

    lines.append("## Arquivos carregados\n")
    if loaded_inventory.empty:
        lines.append("Nenhum arquivo foi carregado.")
    else:
        ok = loaded_inventory[loaded_inventory["status"] == "loaded"]
        lines.append(f"- Arquivos carregados: {len(ok)}")
        lines.append(f"- Arquivos com falha/vazios: {len(loaded_inventory) - len(ok)}\n")

    lines.append("## Resumo por fase\n")
    for phase, res in results.items():
        filt = res.get("filter_summary", pd.DataFrame())
        if not filt.empty:
            r = filt.iloc[0]
            lines.append(f"### {phase}")
            lines.append(f"- Linhas carregadas: {int(r['rows_loaded'])}")
            lines.append(f"- Linhas após filtro de nível: {int(r['rows_after_feature_level'])}")
            lines.append(f"- Linhas após filtro EOL encontrado: {int(r['rows_after_eol_found'])}")
            lines.append(f"- Linhas após remover alvo zero: {int(r['rows_after_positive_target'])}")
            lines.append(f"- Baterias após filtro: {int(r['batteries_after_filter'])}")
            lines.append(f"- Datasets após filtro: {int(r['datasets_after_filter'])}\n")

        group = res.get("group_summary", pd.DataFrame())
        if not group.empty:
            lines.append(f"Principais grupos por média de |Spearman| com o alvo em `{phase}`:")
            for _, row in group.head(10).iterrows():
                lines.append(
                    f"- `{row['feature_group']}`: média={fmt_num(row['mean_abs_spearman'])}, "
                    f"mediana={fmt_num(row['median_abs_spearman'])}, máximo={fmt_num(row['max_abs_spearman'])}, "
                    f"n={int(row['n_features'])}"
                )
            lines.append("")

        trade = res.get("curve_tradeoff", pd.DataFrame())
        if not trade.empty:
            lines.append(f"Comparação Q(V) versus V(Q) por resolução em `{phase}`:")
            for _, row in trade.sort_values(["curve_type", "resolution"]).iterrows():
                lines.append(
                    f"- `{row['feature_group']}`: média alvo={fmt_num(row['mean_abs_spearman'])}, "
                    f"máximo alvo={fmt_num(row['max_abs_spearman'])}, "
                    f"redundância média={fmt_num(row['mean_abs_cross_spearman'])}, "
                    f"pares ≥0,95={fmt_num(row['pct_pairs_abs_ge_0p95'])}"
                )
            lines.append("")

    lines.append("## Como interpretar o trade-off\n")
    lines.append(
        "A resolução com mais pontos só deve ser preferida quando o aumento da média ou do máximo de "
        "|Spearman| com o alvo for relevante em comparação ao aumento da redundância interna. Quando "
        "p=15 ou p=30 apresenta correlação média semelhante a p=3 ou p=9, mas com maior porcentagem "
        "de pares altamente correlacionados, o grupo maior tende a adicionar custo computacional e "
        "multicolinearidade sem ganho proporcional de informação."
    )
    lines.append("")
    lines.append(
        "Para sustentar a hipótese visual indicada anteriormente, verifique nas tabelas `*_QV_VQ_resolution_tradeoff.csv` "
        "se, em carga, os grupos `VQ_interp_*` superam `QV_interp_*`, e se, em descarga, os grupos "
        "`QV_interp_*` superam `VQ_interp_*`."
    )

    report_path = out_root / "eda_feature_correlation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EDA de correlação Spearman e redundância por grupos de features.")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target", type=str, default="auto")
    parser.add_argument("--phases", nargs="+", default=list(PHASES), choices=list(PHASES))
    parser.add_argument("--feature-level", type=str, default="phase", help="Use 'phase' por padrão. Use 'all' para não filtrar.")
    parser.add_argument("--require-eol-found", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-zero-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-valid-rows", type=int, default=30)
    parser.add_argument("--min-unique", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--heatmap-max-features", type=int, default=60)
    return parser


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = build_arg_parser().parse_args()

    out_root = args.output_root / f"run_{now_id()}"
    ensure_dir(out_root)
    ensure_dir(out_root / "global" / "tables")
    ensure_dir(out_root / "global" / "plots")

    files = discover_feature_files(args.features_root)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .parquet encontrado em: {args.features_root}")

    df_all, inv = read_feature_files(files, phases=args.phases)
    inv.to_csv(out_root / "global" / "tables" / "loaded_feature_files.csv", index=False)

    if df_all.empty:
        raise RuntimeError("Nenhum dataframe de features foi carregado com sucesso.")

    target_col = choose_target(df_all, requested=args.target)
    args.target_resolved = target_col

    run_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features_root": str(args.features_root),
        "output_root": str(out_root),
        "target_requested": args.target,
        "target_resolved": target_col,
        "phases": args.phases,
        "feature_level": args.feature_level,
        "require_eol_found": args.require_eol_found,
        "drop_zero_target": args.drop_zero_target,
        "min_valid_rows": args.min_valid_rows,
        "min_unique": args.min_unique,
        "top_n": args.top_n,
        "heatmap_max_features": args.heatmap_max_features,
    }
    (out_root / "global" / "tables" / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for phase in args.phases:
        print(f"\n=== Analisando fase: {phase} ===")
        results[phase] = analyze_phase(
            df_all=df_all,
            phase=phase,
            target_col=target_col,
            out_root=out_root,
            feature_level=args.feature_level,
            require_eol_found=args.require_eol_found,
            drop_zero_target=args.drop_zero_target,
            min_valid_rows=args.min_valid_rows,
            min_unique=args.min_unique,
            top_n=args.top_n,
            heatmap_max_features=args.heatmap_max_features,
        )

    # Tabelas consolidadas.
    concat_specs = {
        "filter_summary_all_phases.csv": "filter_summary",
        "target_spearman_rank_all_phases.csv": "target_corr",
        "target_spearman_by_group_all_phases.csv": "group_summary",
        "group_redundancy_summary_all_phases.csv": "redundancy_summary",
        "QV_VQ_resolution_tradeoff_all_phases.csv": "curve_tradeoff",
        "top_features_by_group_all_phases.csv": "top_by_group",
        "feature_inventory_all_phases.csv": "feature_inventory",
    }
    for filename, key in concat_specs.items():
        frames = [res[key] for res in results.values() if key in res and not res[key].empty]
        if frames:
            pd.concat(frames, axis=0, ignore_index=True, sort=False).to_csv(out_root / "global" / "tables" / filename, index=False)

    # Comparação direta QV x VQ por fase.
    qv_vq_frames = [res["curve_tradeoff"] for res in results.values() if not res["curve_tradeoff"].empty]
    if qv_vq_frames:
        curve_all = pd.concat(qv_vq_frames, ignore_index=True)
        # Melhor curva por fase/resolução.
        best_curve_by_phase_res = (
            curve_all.sort_values("mean_abs_spearman", ascending=False)
            .groupby(["phase", "resolution"], as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        best_curve_by_phase_res.to_csv(out_root / "global" / "tables" / "best_QV_or_VQ_by_phase_resolution.csv", index=False)

        # Melhor resolução por fase e tipo de curva.
        best_res_by_phase_curve = (
            curve_all.sort_values("mean_abs_spearman", ascending=False)
            .groupby(["phase", "curve_type"], as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        best_res_by_phase_curve.to_csv(out_root / "global" / "tables" / "best_resolution_by_phase_curve_type.csv", index=False)

    make_report(out_root, results, args, inv)

    print("\nConcluído.")
    print(f"Relatórios salvos em: {out_root}")
    print(f"Relatório principal: {out_root / 'eda_feature_correlation_report.md'}")


if __name__ == "__main__":
    main()
