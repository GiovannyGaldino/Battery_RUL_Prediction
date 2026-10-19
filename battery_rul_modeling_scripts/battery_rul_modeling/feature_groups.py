from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd


IDENTIFIER_EXACT = {
    "dataset", "battery_id", "phase", "source_file", "source_path", "source_cycle_id",
    "cycle_pair_id", "cell_id_original", "barcode", "policy_readable", "manufacturer",
    "chemistry", "form_factor", "adapter_version", "cleaning_version", "created_at",
}

LEAKAGE_KEYWORDS = [
    "target_", "remaining_cycles", "rul", "eol", "cycle_life", "after_eol",
    "order_index", "life_label", "split", "fold", "battery_index",
]

TRACEABILITY_KEYWORDS = [
    "source", "barcode", "cell_id", "battery_id", "dataset", "phase", "file", "path",
]

CURVE_PATTERNS = [r"(^|_)QV(_|$)", r"(^|_)VQ(_|$)", r"(^|_)EQ(_|$)", r"curve", r"interp", r"p003", r"p009", r"p015", r"p030"]
STAGE_GAP_PATTERNS = [r"stage", r"gap", r"transition", r"rest", r"segment"]
TEMP_PATTERNS = [r"temp", r"temperature"]
ENERGY_PATTERNS = [r"energy", r"power"]
CAPACITY_PATTERNS = [r"capacity", r"soh", r"cap_", r"capacityh"]
GLOBAL_PATTERNS = [
    r"voltage", r"current", r"temperature", r"temp", r"capacity", r"capacityh", r"soh",
    r"energy", r"power", r"time", r"sampling", r"duration", r"n_points",
]


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    lower = name.lower()
    return any(re.search(p, lower, flags=re.IGNORECASE) for p in patterns)


def is_forbidden_feature(col: str, allow_cycle_number_as_feature: bool = False) -> bool:
    lower = col.lower()
    if lower in IDENTIFIER_EXACT:
        return True
    if not allow_cycle_number_as_feature and lower in {"cycle_number", "cycle_index"}:
        return True
    if any(k in lower for k in LEAKAGE_KEYWORDS):
        return True
    if any(k in lower for k in TRACEABILITY_KEYWORDS):
        return True
    return False


def candidate_numeric_features(
    X: pd.DataFrame,
    allow_cycle_number_as_feature: bool = False,
) -> List[str]:
    cols = []
    for col in X.columns:
        if is_forbidden_feature(col, allow_cycle_number_as_feature=allow_cycle_number_as_feature):
            continue
        if pd.api.types.is_bool_dtype(X[col]) or pd.api.types.is_numeric_dtype(X[col]):
            cols.append(col)
    return cols


def select_experiment_features(all_features: List[str], experiment: str) -> List[str]:
    exp = experiment.lower().strip()
    if exp in {"all", "all_features", "baseline_full"}:
        return list(all_features)
    if exp == "no_curves":
        return [c for c in all_features if not _matches_any(c, CURVE_PATTERNS)]
    if exp == "global_only":
        return [
            c for c in all_features
            if _matches_any(c, GLOBAL_PATTERNS)
            and not _matches_any(c, CURVE_PATTERNS)
            and not _matches_any(c, STAGE_GAP_PATTERNS)
        ]
    if exp == "capacity_only":
        return [c for c in all_features if _matches_any(c, CAPACITY_PATTERNS)]
    if exp == "curve_only":
        return [c for c in all_features if _matches_any(c, CURVE_PATTERNS)]
    if exp == "no_temperature":
        return [c for c in all_features if not _matches_any(c, TEMP_PATTERNS)]
    if exp == "no_energy":
        return [c for c in all_features if not _matches_any(c, ENERGY_PATTERNS)]
    if exp == "no_stage_gap":
        return [c for c in all_features if not _matches_any(c, STAGE_GAP_PATTERNS)]
    if exp.startswith("resolution_p"):
        # Keep all non-curve features and curve features from only the requested p-resolution.
        p = exp.replace("resolution_p", "p")
        return [
            c for c in all_features
            if not _matches_any(c, CURVE_PATTERNS) or p in c.lower()
        ]
    raise ValueError(
        f"Experimento desconhecido: {experiment}. Opções: all_features, no_curves, global_only, "
        "capacity_only, curve_only, no_temperature, no_energy, no_stage_gap, resolution_p003, "
        "resolution_p009, resolution_p015, resolution_p030."
    )
