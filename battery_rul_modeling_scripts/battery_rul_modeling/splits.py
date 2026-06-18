from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    split_labels: pd.Series
    battery_table: pd.DataFrame
    summary: pd.DataFrame


def _allocate_batteries_for_dataset(
    batteries: list[str], rng: np.random.Generator, test_size: float, val_size: float
) -> Dict[str, str]:
    batteries = list(batteries)
    rng.shuffle(batteries)
    n = len(batteries)
    labels: Dict[str, str] = {}
    if n <= 0:
        return labels
    if n == 1:
        labels[batteries[0]] = "train"
        return labels
    if n == 2:
        labels[batteries[0]] = "train"
        labels[batteries[1]] = "test"
        return labels

    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size))) if n - n_test >= 3 else 0
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1 if n >= 4 else 0
    test_b = set(batteries[:n_test])
    val_b = set(batteries[n_test:n_test + n_val])
    for b in batteries:
        if b in test_b:
            labels[b] = "test"
        elif b in val_b:
            labels[b] = "val"
        else:
            labels[b] = "train"
    return labels


def make_group_split(
    meta: pd.DataFrame,
    random_seed: int = 42,
    test_size: float = 0.20,
    val_size: float = 0.15,
) -> SplitResult:
    required = {"dataset", "battery_id"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para split agrupado: {missing}")

    battery_table = (
        meta[["dataset", "battery_id"]]
        .drop_duplicates()
        .sort_values(["dataset", "battery_id"])
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(random_seed)
    battery_to_split: Dict[str, str] = {}
    for dataset, group in battery_table.groupby("dataset", sort=True):
        labels = _allocate_batteries_for_dataset(
            group["battery_id"].astype(str).tolist(), rng, test_size, val_size
        )
        battery_to_split.update(labels)

    split_labels = meta["battery_id"].astype(str).map(battery_to_split).fillna("train")
    battery_table["split"] = battery_table["battery_id"].astype(str).map(battery_to_split).fillna("train")

    row_summary = (
        meta.assign(split=split_labels)
        .groupby(["dataset", "split"], dropna=False)
        .agg(n_rows=("battery_id", "size"), n_batteries=("battery_id", "nunique"))
        .reset_index()
    )
    return SplitResult(split_labels=split_labels, battery_table=battery_table, summary=row_summary)


def subsample_rows_by_battery(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    max_rows: int | None,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if max_rows is None or len(X) <= max_rows:
        return X, y, meta
    rng = np.random.default_rng(random_seed)
    # Sampling rows stratified by dataset to avoid taking only the largest block at the top of the table.
    selected_idx = []
    for _, g in meta.groupby("dataset", sort=False):
        frac = len(g) / len(meta)
        n_take = max(1, int(round(max_rows * frac)))
        idx = g.index.to_numpy()
        if len(idx) > n_take:
            idx = rng.choice(idx, size=n_take, replace=False)
        selected_idx.extend(idx.tolist())
    if len(selected_idx) > max_rows:
        selected_idx = rng.choice(np.array(selected_idx), size=max_rows, replace=False).tolist()
    selected_idx = sorted(set(selected_idx))
    return X.loc[selected_idx].reset_index(drop=True), y.loc[selected_idx].reset_index(drop=True), meta.loc[selected_idx].reset_index(drop=True)


def make_lodo_split(
    meta: pd.DataFrame,
    heldout_dataset: str,
    random_seed: int = 42,
    val_size: float = 0.15,
) -> SplitResult:
    """Leave one complete dataset out for testing; split remaining batteries into train/val."""
    required = {"dataset", "battery_id"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para split LODO: {missing}")

    rng = np.random.default_rng(random_seed)
    battery_table = (
        meta[["dataset", "battery_id"]]
        .drop_duplicates()
        .sort_values(["dataset", "battery_id"])
        .reset_index(drop=True)
    )
    labels = {}
    for dataset, group in battery_table.groupby("dataset", sort=True):
        batteries = group["battery_id"].astype(str).tolist()
        if str(dataset) == str(heldout_dataset):
            for b in batteries:
                labels[b] = "test"
            continue
        rng.shuffle(batteries)
        n = len(batteries)
        if n <= 1:
            for b in batteries:
                labels[b] = "train"
            continue
        n_val = max(1, int(round(n * val_size))) if n >= 3 else 1
        n_val = min(n_val, n - 1)
        val_b = set(batteries[:n_val])
        for b in batteries:
            labels[b] = "val" if b in val_b else "train"

    split_labels = meta["battery_id"].astype(str).map(labels).fillna("train")
    battery_table["split"] = battery_table["battery_id"].astype(str).map(labels).fillna("train")
    row_summary = (
        meta.assign(split=split_labels)
        .groupby(["dataset", "split"], dropna=False)
        .agg(n_rows=("battery_id", "size"), n_batteries=("battery_id", "nunique"))
        .reset_index()
    )
    return SplitResult(split_labels=split_labels, battery_table=battery_table, summary=row_summary)
