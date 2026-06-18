from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


META_COLUMNS = [
    "dataset", "battery_id", "cycle_number", "cycle_index", "phase", "source_file",
    "source_path", "source_cycle_id", "cycle_pair_id", "cell_id_original", "barcode",
    "policy_readable", "cycle_life", "target_eol80_cycle_index", "target_eol70_cycle_index",
    "target_eol80_order_index", "target_eol70_order_index", "target_initial_capacity_ah",
    "target_capacity_ah", "target_soh_recomputed", "input_soh", "soh",
]

# Common metadata names that may appear in X, y, or in a parquet index after reset_index().
BATTERY_ID_ALTERNATIVES = [
    "battery_id", "input_battery_id", "meta_battery_id", "battery", "battery_name",
    "cell_id", "cell_id_original", "input_cell_id_original", "cell", "cell_name",
    "cell_index", "barcode", "input_barcode",
]
DATASET_ALTERNATIVES = [
    "dataset", "input_dataset", "meta_dataset", "dataset_name", "data_set", "database",
    "source_dataset", "study",
]
CYCLE_NUMBER_ALTERNATIVES = [
    "cycle_number", "input_cycle_number", "cycle_index", "input_cycle_index", "cycle", "cycle_id",
]
PHASE_ALTERNATIVES = ["phase", "input_phase", "phase_folder"]


def expand_user_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_dir(path: str | Path) -> Path:
    path = expand_user_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _reset_index_preserving_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Reset index only when it may contain useful metadata.

    Some preparation scripts save identifiers such as battery_id/dataset in a parquet
    index. The previous version used reset_index(drop=True), which discarded these
    fields. This helper preserves named/non-RangeIndex values as ordinary columns.
    """
    if isinstance(df.index, pd.RangeIndex) and df.index.name is None:
        return df.reset_index(drop=True)
    try:
        return df.reset_index()
    except ValueError:
        # Duplicate column name after reset_index. Rename index levels defensively.
        df2 = df.copy()
        if isinstance(df2.index, pd.MultiIndex):
            names = [name if name is not None else f"index_level_{i}" for i, name in enumerate(df2.index.names)]
            names = [name if name not in df2.columns else f"idx_{name}" for name in names]
            df2.index = df2.index.set_names(names)
        else:
            name = df2.index.name or "index"
            if name in df2.columns:
                df2.index = df2.index.rename(f"idx_{name}")
        return df2.reset_index()


def read_parquet_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return _reset_index_preserving_metadata(pd.read_parquet(path))


def _find_target_column(y_df: pd.DataFrame, x_df: pd.DataFrame, target: str) -> Tuple[pd.Series, str]:
    if target in y_df.columns:
        return y_df[target], target
    if target in x_df.columns:
        return x_df[target], target

    # Case-insensitive target fallback.
    target_lower = target.lower()
    for df in (y_df, x_df):
        for col in df.columns:
            if str(col).lower() == target_lower:
                return df[col], str(col)

    numeric_cols = [c for c in y_df.columns if pd.api.types.is_numeric_dtype(y_df[c])]
    # Ignore possible reset index columns when trying to infer single y column.
    numeric_non_index = [c for c in numeric_cols if not str(c).lower().startswith(("index", "level_", "idx_"))]
    if len(numeric_non_index) == 1:
        return y_df[numeric_non_index[0]], str(numeric_non_index[0])
    if len(numeric_cols) == 1:
        return y_df[numeric_cols[0]], str(numeric_cols[0])
    if len(y_df.columns) == 1:
        col = y_df.columns[0]
        return pd.to_numeric(y_df[col], errors="coerce"), str(col)
    raise ValueError(
        f"Não consegui encontrar a coluna alvo '{target}'. Colunas de y: {list(y_df.columns)[:50]}..."
    )


def _normalize_name(name: Any) -> str:
    return str(name).strip().lower()


def _find_first_column(df_list: List[pd.DataFrame], alternatives: List[str]) -> Optional[Tuple[pd.DataFrame, str]]:
    alt_norm = {_normalize_name(a) for a in alternatives}
    # 1) exact/case-insensitive match
    for df in df_list:
        for col in df.columns:
            if _normalize_name(col) in alt_norm:
                return df, col
    # 2) loose contains match for battery/cell/dataset names
    for df in df_list:
        for col in df.columns:
            lower = _normalize_name(col)
            if any(a in lower for a in alt_norm if len(a) >= 5):
                return df, col
    return None


def _clean_string_series(s: pd.Series, default: str) -> pd.Series:
    out = s.copy()
    out = out.where(~pd.isna(out), default)
    out = out.astype(str)
    bad = out.str.strip().str.lower().isin({"", "nan", "none", "null", "<na>"})
    out.loc[bad] = default
    return out


def _available_columns_summary(X_raw: pd.DataFrame, y_raw: pd.DataFrame) -> str:
    x_cols = [str(c) for c in X_raw.columns[:80]]
    y_cols = [str(c) for c in y_raw.columns[:80]]
    return (
        "\nColunas iniciais de X: " + ", ".join(x_cols) +
        "\nColunas iniciais de y: " + ", ".join(y_cols)
    )


def load_phase_xy(input_root: Path, phase: str, target: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:
    input_root = expand_user_path(input_root)
    x_path = input_root / f"{phase}_X.parquet"
    y_path = input_root / f"{phase}_y.parquet"
    X_raw = read_parquet_checked(x_path)
    y_raw = read_parquet_checked(y_path)

    # After preserving possible index metadata, align by row position.
    X_raw = X_raw.reset_index(drop=True)
    y_raw = y_raw.reset_index(drop=True)

    if len(X_raw) != len(y_raw):
        raise ValueError(
            f"Tamanhos incompatíveis para {phase}: X={len(X_raw)} linhas, y={len(y_raw)} linhas."
        )
    y, resolved_target = _find_target_column(y_raw, X_raw, target)
    y = pd.to_numeric(y, errors="coerce").astype(float)

    meta = pd.DataFrame(index=X_raw.index)
    for col in META_COLUMNS:
        if col in X_raw.columns:
            meta[col] = X_raw[col]
        elif col in y_raw.columns:
            meta[col] = y_raw[col]

    # Robust fallback for key metadata that may be named differently or stored in y/index.
    frames = [X_raw, y_raw]

    if "phase" not in meta.columns:
        found = _find_first_column(frames, PHASE_ALTERNATIVES)
        meta["phase"] = found[0][found[1]] if found else phase

    if "dataset" not in meta.columns:
        found = _find_first_column(frames, DATASET_ALTERNATIVES)
        meta["dataset"] = found[0][found[1]] if found else "unknown_dataset"

    if "battery_id" not in meta.columns:
        found = _find_first_column(frames, BATTERY_ID_ALTERNATIVES)
        if found:
            meta["battery_id"] = found[0][found[1]]
        else:
            raise ValueError(
                f"A coluna 'battery_id' é obrigatória para split por bateria na fase {phase}. "
                "Procurei em X, y e no índice preservado do parquet por nomes como "
                f"{BATTERY_ID_ALTERNATIVES}." + _available_columns_summary(X_raw, y_raw)
            )

    if "cycle_number" not in meta.columns:
        found = _find_first_column(frames, CYCLE_NUMBER_ALTERNATIVES)
        if found:
            meta["cycle_number"] = found[0][found[1]]

    meta["phase"] = _clean_string_series(meta["phase"], phase)
    meta["dataset"] = _clean_string_series(meta["dataset"], "unknown_dataset")
    meta["battery_id"] = _clean_string_series(meta["battery_id"], "unknown_battery")

    return X_raw, y, meta.reset_index(drop=True), resolved_target


def filter_valid_target(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    drop_zero_rul: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    mask = np.isfinite(y.to_numpy(dtype=float))
    mask &= y.to_numpy(dtype=float) >= 0
    if drop_zero_rul:
        mask &= y.to_numpy(dtype=float) > 0
    X2 = X.loc[mask].reset_index(drop=True)
    y2 = y.loc[mask].reset_index(drop=True)
    meta2 = meta.loc[mask].reset_index(drop=True)
    return X2, y2, meta2


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def write_text(path: Path, lines: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).rstrip("\n") + "\n")
