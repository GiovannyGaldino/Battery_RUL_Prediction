#!/usr/bin/env python3
"""Produce a simple completion report for the final consolidated campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check final campaign completion.")
    p.add_argument("--summary-dir", type=Path, default=Path("~/outputs/batteries_modeling_v3/modeling_results_final_campaign_v1/summary").expanduser())
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = args.summary_dir.expanduser()
    summary_dir.mkdir(parents=True, exist_ok=True)
    global_path = summary_dir / "final_campaign_metrics_global.csv"
    cv_path = summary_dir / "final_campaign_cv_metrics.csv"

    expected = {
        "full_matrix_seed42_global": 2 * 13 * 6 * 1,
        "full_matrix_seed42_per_dataset": 2 * 8 * 13 * 6,
        "full_matrix_seed42_lodo": 2 * 8 * 13 * 6,
        "repeated_split_seed13_global_per_dataset": 2 * 9 * 13 * 6,
        "repeated_split_seed73_global_per_dataset": 2 * 9 * 13 * 6,
        "internal_groupkfold_final_test_rows": 2 * 13 * 6,
        "internal_groupkfold_cv_fold_rows": 2 * 13 * 6 * 3,
    }

    rows = []
    if global_path.exists():
        g = pd.read_csv(global_path)
        rows.append({"item": "all_global_metric_rows", "expected": sum(v for k, v in expected.items() if k != "internal_groupkfold_cv_fold_rows"), "observed": len(g)})
        if not g.empty:
            for comp, part in g.groupby("campaign_component", dropna=False):
                rows.append({"item": f"component_{comp}_global_metric_rows", "expected": None, "observed": len(part)})
            for cols in [["campaign_component", "random_state"], ["campaign_component", "random_state", "phase"], ["campaign_component", "random_state", "phase", "mode"]]:
                if all(c in g.columns for c in cols):
                    tab = g.groupby(cols, dropna=False).size().reset_index(name="observed")
                    tab.to_csv(summary_dir / ("status_by_" + "_".join(cols) + ".csv"), index=False)
    else:
        rows.append({"item": "all_global_metric_rows", "expected": sum(v for k, v in expected.items() if k != "internal_groupkfold_cv_fold_rows"), "observed": 0})

    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        rows.append({"item": "internal_groupkfold_cv_fold_rows", "expected": expected["internal_groupkfold_cv_fold_rows"], "observed": len(cv)})
    else:
        rows.append({"item": "internal_groupkfold_cv_fold_rows", "expected": expected["internal_groupkfold_cv_fold_rows"], "observed": 0})

    status = pd.DataFrame(rows)
    status["complete"] = status.apply(lambda r: None if pd.isna(r["expected"]) else int(r["observed"]) >= int(r["expected"]), axis=1)
    status.to_csv(summary_dir / "final_campaign_completion_status.csv", index=False)

    with open(summary_dir / "final_campaign_expected_counts.json", "w", encoding="utf-8") as f:
        json.dump(expected, f, indent=2, ensure_ascii=False)

    print(status.to_string(index=False))
    print(f"[DONE] Status written to {summary_dir / 'final_campaign_completion_status.csv'}")


if __name__ == "__main__":
    main()
