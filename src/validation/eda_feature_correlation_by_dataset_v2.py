#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA por dataset: correlação Spearman entre grupos de features e RUL,
separando charge/discharge e dataset.

Entrada esperada:
  --input-root ~/data/batteries/modeling_from_v3_with_curves_fixed
com arquivos:
  charge_phase_modeling_with_curves.parquet
  discharge_phase_modeling_with_curves.parquet

Saída:
  CSVs e PNGs em --output-root/run_YYYYMMDD_HHMMSS
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_CANDIDATES = [
    "target_rul80_steps_clipped",
    "target_rul80_cycle_index_clipped",
    "remaining_cycles",
    "remaining_cycles_norm",
]

ID_PREFIXES = (
    "source_", "cell_id", "barcode", "policy_readable", "charge_protocol",
    "discharge_protocol", "chemistry", "form_factor", "manufacturer",
)
LEAKAGE_PREFIXES = (
    "target_rul", "target_eol", "target_after_eol", "remaining_cycles",
    "cycle_life", "eol_", "initial_capacity", "target_initial_capacity",
)
EXCLUDE_EXACT = {
    "dataset", "battery_id", "phase", "phase_folder",
    "cycle_index", "cycle_number", "order_index",
    "is_valid_cycle", "is_model_usable",
}


def infer_group(col: str) -> str:
    c = col

    m = re.match(r"^(QV|VQ)_p(\d{3})_\d+$", c)
    if m:
        return f"{m.group(1)}_interp_p{m.group(2)}"

    if c.startswith("QV_"):
        return "QV_curve_stats"
    if c.startswith("VQ_"):
        return "VQ_curve_stats"

    lower = c.lower()
    if "temperature" in lower or lower.startswith("temp"):
        return "temperature_stats"
    if "voltage" in lower or lower.startswith("volt"):
        return "voltage_stats"
    if "current" in lower or lower.startswith("curr") or lower.startswith("integral_current"):
        return "current_stats"
    if "capacity" in lower or lower.startswith("capacityh") or lower.startswith("q_"):
        return "capacity_stats"
    if "energy" in lower:
        return "energy_stats"
    if "power" in lower:
        return "power_stats"
    if "time" in lower or "duration" in lower or "sample" in lower or "sampling" in lower:
        return "time_sampling_stats"
    if "stage" in lower or "rest" in lower or "transition" in lower:
        return "stage_structure"
    if "valid" in lower or "has_" in lower or "finite" in lower:
        return "quality_or_availability"
    return "other_numeric"


def choose_target(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Target requested not found: {requested}")
        return requested
    for t in TARGET_CANDIDATES:
        if t in df.columns:
            return t
    raise ValueError("No target column found. Use --target explicitly.")


def find_phase_file(input_root: Path, phase: str) -> Path:
    candidates = [
        input_root / f"{phase}_phase_modeling_with_curves.parquet",
        input_root / f"{phase}_phase_modeling.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No {phase} parquet found in {input_root}")


def numeric_feature_columns(df: pd.DataFrame, target: str):
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    keep = []
    for c in numeric_cols:
        lc = c.lower()
        if c == target:
            continue
        if c in EXCLUDE_EXACT:
            continue
        if any(lc.startswith(p) for p in ID_PREFIXES):
            continue
        if any(lc.startswith(p) for p in LEAKAGE_PREFIXES):
            continue
        keep.append(c)
    return keep


def filter_df(df: pd.DataFrame, target: str, drop_zero_target=True, require_eol_found=True):
    out = df.copy()
    if require_eol_found:
        eol_col = None
        if target.startswith("target_rul80") and "target_eol80_found" in out.columns:
            eol_col = "target_eol80_found"
        elif target.startswith("target_rul70") and "target_eol70_found" in out.columns:
            eol_col = "target_eol70_found"
        if eol_col is not None:
            out = out[out[eol_col].astype(bool)]
    out = out[pd.to_numeric(out[target], errors="coerce").notna()]
    if drop_zero_target:
        out = out[pd.to_numeric(out[target], errors="coerce") > 0]
    return out


def spearman_by_feature(df, feature_cols, target, min_valid_rows=30, min_unique=3):
    y = pd.to_numeric(df[target], errors="coerce")
    rows = []
    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n < min_valid_rows:
            continue
        if x[mask].nunique(dropna=True) < min_unique:
            continue
        if y[mask].nunique(dropna=True) < min_unique:
            continue
        rho = x[mask].corr(y[mask], method="spearman")
        if pd.notna(rho):
            rows.append({
                "feature": c,
                "feature_group": infer_group(c),
                "spearman_rho": float(rho),
                "abs_spearman_rho": float(abs(rho)),
                "n_valid": n,
                "missing_fraction": float(1 - n / len(df)),
                "n_unique": int(x[mask].nunique(dropna=True)),
            })
    return pd.DataFrame(rows)


def summarize_groups(rank_df):
    if rank_df.empty:
        return pd.DataFrame()
    return (rank_df
        .groupby("feature_group", as_index=False)
        .agg(
            n_features=("feature", "count"),
            mean_abs_spearman=("abs_spearman_rho", "mean"),
            median_abs_spearman=("abs_spearman_rho", "median"),
            max_abs_spearman=("abs_spearman_rho", "max"),
            mean_missing_fraction=("missing_fraction", "mean"),
        )
        .sort_values("mean_abs_spearman", ascending=False)
    )


def redundancy_by_group(df, rank_df, max_features=60):
    rows = []
    for group, sub in rank_df.groupby("feature_group"):
        feats = sub.sort_values("abs_spearman_rho", ascending=False)["feature"].head(max_features).tolist()
        feats = [f for f in feats if f in df.columns]
        if len(feats) < 2:
            rows.append({
                "feature_group": group, "n_features": len(feats), "n_pairs": 0,
                "mean_abs_cross_spearman": np.nan, "median_abs_cross_spearman": np.nan,
                "max_abs_cross_spearman": np.nan,
                "pct_pairs_abs_ge_0p90": np.nan,
                "pct_pairs_abs_ge_0p95": np.nan,
            })
            continue
        corr = df[feats].corr(method="spearman").abs()
        vals = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().values
        vals = vals[np.isfinite(vals)]
        rows.append({
            "feature_group": group,
            "n_features": len(feats),
            "n_pairs": int(len(vals)),
            "mean_abs_cross_spearman": float(np.mean(vals)) if len(vals) else np.nan,
            "median_abs_cross_spearman": float(np.median(vals)) if len(vals) else np.nan,
            "max_abs_cross_spearman": float(np.max(vals)) if len(vals) else np.nan,
            "pct_pairs_abs_ge_0p90": float(np.mean(vals >= 0.90)) if len(vals) else np.nan,
            "pct_pairs_abs_ge_0p95": float(np.mean(vals >= 0.95)) if len(vals) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mean_abs_cross_spearman", ascending=False)


def qv_vq_tradeoff(group_summary, red_summary):
    """
    Build the QV/VQ resolution trade-off table.

    In the by-dataset workflow, group_summary and red_summary may already
    contain context columns such as phase and dataset. These columns are
    intentionally skipped here because they are inserted explicitly in main().
    This avoids duplicate-column errors in pandas.DataFrame.insert().
    """
    context_cols = {"phase", "dataset"}
    rows = []
    for curve_type in ["QV", "VQ"]:
        for res in [3, 9, 15, 30]:
            group = f"{curve_type}_interp_p{res:03d}"
            g = group_summary[group_summary["feature_group"] == group]
            r = red_summary[red_summary["feature_group"] == group]
            if g.empty and r.empty:
                continue
            row = {"curve_type": curve_type, "resolution": res, "feature_group": group}
            if not g.empty:
                for k, v in g.iloc[0].to_dict().items():
                    if k not in context_cols and k != "feature_group":
                        row[k] = v
            if not r.empty:
                for k, v in r.iloc[0].to_dict().items():
                    if k not in context_cols and k != "feature_group":
                        # Avoid overwriting group-level columns silently.
                        # Redundancy-specific columns keep their own names.
                        row[k] = v
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--target", default="auto")
    ap.add_argument("--drop-zero-target", action="store_true")
    ap.add_argument("--require-eol-found", action="store_true")
    ap.add_argument("--min-valid-rows", type=int, default=30)
    ap.add_argument("--min-unique", type=int, default=3)
    ap.add_argument("--heatmap-max-features", type=int, default=60)
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    run = args.output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tables = run / "tables"
    plots = run / "plots"
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    all_rank = []
    all_groups = []
    all_red = []
    all_trade = []
    filter_rows = []

    for phase in ["charge", "discharge"]:
        path = find_phase_file(args.input_root, phase)
        df0 = pd.read_parquet(path)
        target = choose_target(df0, args.target)
        df = filter_df(df0, target, args.drop_zero_target, args.require_eol_found)

        if "dataset" not in df.columns:
            raise ValueError("Column 'dataset' is required for by-dataset analysis.")

        feature_cols = numeric_feature_columns(df, target)

        filter_rows.append({
            "phase": phase,
            "file": str(path),
            "target": target,
            "rows_loaded": len(df0),
            "rows_after_filter": len(df),
            "datasets": int(df["dataset"].nunique()),
            "batteries": int(df["battery_id"].nunique()) if "battery_id" in df.columns else np.nan,
            "n_feature_candidates": len(feature_cols),
        })

        for dataset, dfg in df.groupby("dataset", sort=True):
            rank = spearman_by_feature(dfg, feature_cols, target, args.min_valid_rows, args.min_unique)
            if rank.empty:
                continue
            rank.insert(0, "dataset", dataset)
            rank.insert(0, "phase", phase)

            group = summarize_groups(rank)
            group.insert(0, "dataset", dataset)
            group.insert(0, "phase", phase)

            red = redundancy_by_group(dfg, rank, args.heatmap_max_features)
            red.insert(0, "dataset", dataset)
            red.insert(0, "phase", phase)

            trade = qv_vq_tradeoff(group, red)
            if not trade.empty:
                trade.insert(0, "dataset", dataset)
                trade.insert(0, "phase", phase)

            all_rank.append(rank)
            all_groups.append(group)
            all_red.append(red)
            all_trade.append(trade)

    rank_all = pd.concat(all_rank, ignore_index=True) if all_rank else pd.DataFrame()
    group_all = pd.concat(all_groups, ignore_index=True) if all_groups else pd.DataFrame()
    red_all = pd.concat(all_red, ignore_index=True) if all_red else pd.DataFrame()
    trade_all = pd.concat(all_trade, ignore_index=True) if all_trade else pd.DataFrame()
    filter_all = pd.DataFrame(filter_rows)

    filter_all.to_csv(tables / "dataset_phase_filter_summary.csv", index=False)
    rank_all.to_csv(tables / "dataset_phase_target_spearman_rank_all.csv", index=False)
    group_all.to_csv(tables / "dataset_phase_target_spearman_by_group.csv", index=False)
    red_all.to_csv(tables / "dataset_phase_group_redundancy_summary.csv", index=False)
    trade_all.to_csv(tables / "dataset_phase_QV_VQ_resolution_tradeoff.csv", index=False)

    # Compact top table for article
    if not group_all.empty:
        top_group = (group_all.sort_values(["phase", "dataset", "mean_abs_spearman"], ascending=[True, True, False])
                     .groupby(["phase", "dataset"], as_index=False)
                     .head(5))
        top_group.to_csv(tables / "dataset_phase_top5_groups.csv", index=False)

    # Plot: mean spearman by group and dataset for selected major groups
    major = ["QV_interp_p003", "QV_interp_p009", "QV_interp_p015", "QV_interp_p030",
             "VQ_interp_p003", "VQ_interp_p009", "VQ_interp_p015", "VQ_interp_p030",
             "QV_curve_stats", "VQ_curve_stats", "temperature_stats", "voltage_stats",
             "current_stats", "energy_stats", "capacity_stats", "power_stats", "time_sampling_stats"]

    for phase in ["charge", "discharge"]:
        sub = group_all[(group_all["phase"] == phase) & (group_all["feature_group"].isin(major))]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="feature_group", columns="dataset", values="mean_abs_spearman", aggfunc="mean")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(pivot.values, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Média de |Spearman| por grupo e dataset — {phase}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Grupo de features")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Média de |Spearman| com o alvo")
        plt.tight_layout()
        plt.savefig(plots / f"{phase}_dataset_group_mean_spearman_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    with open(run / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_root": str(args.input_root),
            "output_root": str(run),
            "target_requested": args.target,
            "drop_zero_target": args.drop_zero_target,
            "require_eol_found": args.require_eol_found,
            "min_valid_rows": args.min_valid_rows,
            "min_unique": args.min_unique,
        }, f, indent=2)

    print(f"Done. Output: {run}")


if __name__ == "__main__":
    main()
