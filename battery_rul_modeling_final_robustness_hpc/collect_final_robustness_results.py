#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

RUN_DIRS = [
    "modeling_results_complete_reduced",
    "modeling_results_final_elasticnet_full",
    "modeling_results_final_mlp_reduced",
    "modeling_results_final_repeated_seed_13",
    "modeling_results_final_repeated_seed_42",
    "modeling_results_final_repeated_seed_73",
    "modeling_results_final_internal_cv_global",
]

MASTER_FILES = {
    "global": "all_metrics_global.csv",
    "dataset": "all_metrics_dataset.csv",
    "battery": "all_metrics_battery.csv",
    "rul_bin": "all_metrics_rul_bin.csv",
    "macro_by_dataset": "all_metrics_macro_by_dataset.csv",
}


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return pd.DataFrame()


def add_run_context(df: pd.DataFrame, run_dir: str, path: Path) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["result_run_dir"] = run_dir
    df["source_file"] = str(path)
    return df


def collect_master_tables(outputs_root: Path, output_dir: Path) -> Dict[str, int]:
    counts = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, filename in MASTER_FILES.items():
        frames: List[pd.DataFrame] = []
        for run_dir in RUN_DIRS:
            path = outputs_root / run_dir / filename
            df = read_csv_if_exists(path)
            if not df.empty:
                frames.append(add_run_context(df, run_dir, path))
        if frames:
            out = pd.concat(frames, ignore_index=True)
        else:
            out = pd.DataFrame()
        out_path = output_dir / f"final_robustness_{key}.csv"
        out.to_csv(out_path, index=False)
        counts[key] = len(out)
        print(f"[WRITE] {out_path} rows={len(out)}")

    return counts


def collect_cv_metrics(outputs_root: Path, output_dir: Path) -> int:
    frames: List[pd.DataFrame] = []
    for run_dir in RUN_DIRS:
        root = outputs_root / run_dir
        if not root.exists():
            continue
        for path in root.rglob("cv_metrics.csv"):
            df = read_csv_if_exists(path)
            if df.empty:
                continue
            rel = path.relative_to(root)
            parts = rel.parts
            # Expected: phase/mode/scope/experiment/model/cv_metrics.csv
            context = {
                "result_run_dir": run_dir,
                "source_file": str(path),
            }
            if len(parts) >= 6:
                context.update({
                    "phase_path": parts[0],
                    "mode_path": parts[1],
                    "scope_path": parts[2],
                    "experiment_path": parts[3],
                    "model_path": parts[4],
                })
            for k, v in context.items():
                df[k] = v
            frames.append(df)

    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame()
    out_path = output_dir / "final_robustness_cv_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path} rows={len(out)}")
    return len(out)


def make_status_summary(output_dir: Path) -> None:
    global_path = output_dir / "final_robustness_global.csv"
    df = read_csv_if_exists(global_path)
    summary = {}
    if not df.empty:
        group_cols = [c for c in ["result_run_dir", "phase", "mode", "experiment", "model", "random_state"] if c in df.columns]
        summary["n_global_rows"] = int(len(df))
        summary["available_result_run_dirs"] = sorted(df["result_run_dir"].dropna().astype(str).unique().tolist()) if "result_run_dir" in df.columns else []
        summary["available_models"] = sorted(df["model"].dropna().astype(str).unique().tolist()) if "model" in df.columns else []
        summary["available_modes"] = sorted(df["mode"].dropna().astype(str).unique().tolist()) if "mode" in df.columns else []
        summary["available_experiments"] = sorted(df["experiment"].dropna().astype(str).unique().tolist()) if "experiment" in df.columns else []
        summary["group_columns_used"] = group_cols

        if {"result_run_dir", "phase", "mode", "experiment", "model"}.issubset(df.columns):
            counts = (
                df.groupby(["result_run_dir", "phase", "mode", "experiment", "model"], dropna=False)
                .size()
                .reset_index(name="n_rows")
            )
            counts.to_csv(output_dir / "final_robustness_global_counts.csv", index=False)

        if {"phase", "mode", "experiment", "model", "MAE", "RMSE", "R2", "sMAPE"}.issubset(df.columns):
            agg_cols = ["MAE", "RMSE", "R2", "sMAPE"]
            stability_cols = [c for c in ["phase", "mode", "experiment", "model"] if c in df.columns]
            stability = (
                df.groupby(stability_cols, dropna=False)[agg_cols]
                .agg(["mean", "std", "min", "max", "count"])
            )
            stability.columns = ["_".join(col).strip("_") for col in stability.columns]
            stability = stability.reset_index()
            stability.to_csv(output_dir / "final_robustness_metric_stability.csv", index=False)

    with open(output_dir / "final_robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[WRITE] {output_dir / 'final_robustness_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect final robustness modeling results.")
    parser.add_argument("--outputs-root", type=Path, default=Path("~/outputs/batteries_modeling_v3").expanduser())
    parser.add_argument("--output-dir", type=Path, default=Path("~/outputs/batteries_modeling_v3/final_robustness_summary").expanduser())
    args = parser.parse_args()
    args.outputs_root = args.outputs_root.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] outputs_root={args.outputs_root}")
    print(f"[INFO] output_dir={args.output_dir}")

    counts = collect_master_tables(args.outputs_root, args.output_dir)
    cv_count = collect_cv_metrics(args.outputs_root, args.output_dir)
    make_status_summary(args.output_dir)

    print("[DONE] Collection finished.")
    print("[COUNTS]", counts)
    print("[CV_ROWS]", cv_count)


if __name__ == "__main__":
    main()
