from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .io_utils import ensure_dir


def _read_existing(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_comparison_tables(output_root: Path) -> None:
    output_root = Path(output_root).expanduser().resolve()
    comparison_dir = ensure_dir(output_root / "comparison")

    global_df = _read_existing([
        output_root / "charge" / "metrics" / "global_metrics_all.csv",
        output_root / "discharge" / "metrics" / "global_metrics_all.csv",
    ])
    dataset_df = _read_existing([
        output_root / "charge" / "metrics" / "dataset_metrics_all.csv",
        output_root / "discharge" / "metrics" / "dataset_metrics_all.csv",
    ])
    battery_df = _read_existing([
        output_root / "charge" / "metrics" / "battery_metrics_all.csv",
        output_root / "discharge" / "metrics" / "battery_metrics_all.csv",
    ])
    rul_abs_df = _read_existing([
        output_root / "charge" / "metrics" / "rul_abs_bin_metrics_all.csv",
        output_root / "discharge" / "metrics" / "rul_abs_bin_metrics_all.csv",
    ])
    rul_norm_df = _read_existing([
        output_root / "charge" / "metrics" / "rul_norm_bin_metrics_all.csv",
        output_root / "discharge" / "metrics" / "rul_norm_bin_metrics_all.csv",
    ])

    if not global_df.empty:
        global_df.to_csv(comparison_dir / "global_metrics_all_phases.csv", index=False)
        test_df = global_df[global_df["split"].eq("test")].copy() if "split" in global_df.columns else global_df.copy()
        ranking = test_df.sort_values(["experiment", "mae", "rmse"], ascending=[True, True, True])
        ranking["rank_mae_within_experiment"] = ranking.groupby("experiment")["mae"].rank(method="dense", ascending=True)
        ranking.to_csv(comparison_dir / "model_ranking.csv", index=False)

        pivot_cols = ["experiment", "model", "split"]
        metrics = ["mae", "rmse", "r2", "smape", "nmae", "nrmse", "median_abs_error", "mean_signed_error"]
        available_metrics = [m for m in metrics if m in test_df.columns]
        if all(c in test_df.columns for c in pivot_cols + ["phase"]):
            pivot = test_df.pivot_table(index=pivot_cols, columns="phase", values=available_metrics, aggfunc="first")
            pivot.columns = [f"{metric}_{phase}" for metric, phase in pivot.columns]
            pivot = pivot.reset_index()
            if "mae_charge" in pivot.columns and "mae_discharge" in pivot.columns:
                pivot["delta_mae_charge_minus_discharge"] = pivot["mae_charge"] - pivot["mae_discharge"]
            if "rmse_charge" in pivot.columns and "rmse_discharge" in pivot.columns:
                pivot["delta_rmse_charge_minus_discharge"] = pivot["rmse_charge"] - pivot["rmse_discharge"]
            pivot.to_csv(comparison_dir / "charge_vs_discharge_metrics.csv", index=False)

    if not dataset_df.empty:
        dataset_df.to_csv(comparison_dir / "dataset_error_comparison.csv", index=False)
    if not battery_df.empty:
        battery_df.to_csv(comparison_dir / "battery_error_comparison.csv", index=False)
    if not rul_abs_df.empty:
        rul_abs_df.to_csv(comparison_dir / "rul_abs_bin_error_comparison.csv", index=False)
    if not rul_norm_df.empty:
        rul_norm_df.to_csv(comparison_dir / "rul_norm_bin_error_comparison.csv", index=False)
