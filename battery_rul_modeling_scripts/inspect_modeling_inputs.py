#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from battery_rul_modeling.feature_groups import candidate_numeric_features, select_experiment_features
from battery_rul_modeling.io_utils import filter_valid_target, load_phase_xy


def parse_args():
    p = argparse.ArgumentParser(description="Inspeciona os parquet de modelagem antes do treinamento.")
    p.add_argument("--input-root", default="~/outputs/batteries_modeling_v3/modeling_ready")
    p.add_argument("--target", default="target_rul80_steps_clipped")
    p.add_argument("--phases", nargs="+", default=["charge", "discharge"])
    p.add_argument("--experiments", nargs="+", default=["all_features", "no_curves", "global_only", "capacity_only", "curve_only", "no_temperature", "no_energy", "no_stage_gap"])
    return p.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    for phase in args.phases:
        print("\n" + "=" * 80)
        print(f"Fase: {phase}")
        X, y, meta, resolved_target = load_phase_xy(input_root, phase, args.target)
        X, y, meta = filter_valid_target(X, y, meta, drop_zero_rul=False)
        print(f"X shape: {X.shape}")
        print(f"y target: {resolved_target} | y count: {len(y)} | y min/median/max: {y.min():.3f} / {y.median():.3f} / {y.max():.3f}")
        print(f"Datasets: {meta['dataset'].nunique()} | Batteries: {meta['battery_id'].nunique()}")
        print("Rows by dataset:")
        print(meta.groupby('dataset').size().sort_values(ascending=False).to_string())
        features = candidate_numeric_features(X)
        print(f"Candidate numeric features after leakage/id filtering: {len(features)}")
        for exp in args.experiments:
            try:
                exp_features = select_experiment_features(features, exp)
                print(f"  {exp}: {len(exp_features)} features")
            except Exception as exc:
                print(f"  {exp}: erro ({exc})")


if __name__ == "__main__":
    main()
