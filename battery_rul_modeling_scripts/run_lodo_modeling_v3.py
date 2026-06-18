#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from battery_rul_modeling.comparison import build_comparison_tables
from battery_rul_modeling.config import ModelingConfig
from battery_rul_modeling.io_utils import ensure_dir, filter_valid_target, load_phase_xy, write_json
from battery_rul_modeling.splits import make_lodo_split
from battery_rul_modeling.training import train_phase


def parse_args():
    parser = argparse.ArgumentParser(description="Executa validação leave-one-dataset-out por fase.")
    parser.add_argument("--input-root", default="~/outputs/batteries_modeling_v3/modeling_ready")
    parser.add_argument("--output-root", default="~/outputs/batteries_modeling_v3/modeling_results_lodo")
    parser.add_argument("--target", default="target_rul80_steps_clipped")
    parser.add_argument("--phases", nargs="+", default=["charge", "discharge"], choices=["charge", "discharge"])
    parser.add_argument("--heldout-datasets", nargs="+", default=None, help="Datasets a deixar fora. Se omitido, usa todos.")
    parser.add_argument("--models", nargs="+", default=["median", "elasticnet", "random_forest", "extra_trees", "hist_gradient_boosting", "mlp"])
    parser.add_argument("--experiments", nargs="+", default=["all_features"])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-missing-frac", type=float, default=0.60)
    parser.add_argument("--min-variance", type=float, default=1e-12)
    parser.add_argument("--mape-min-true", type=float, default=1.0)
    parser.add_argument("--drop-zero-rul", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dataset-balanced-weights", action="store_true")
    parser.add_argument("--allow-cycle-number-as-feature", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_output = Path(args.output_root).expanduser().resolve()
    input_root = Path(args.input_root).expanduser().resolve()
    ensure_dir(base_output)

    # Discover datasets from requested phases.
    phase_data = {}
    datasets = set()
    for phase in args.phases:
        X, y, meta, resolved_target = load_phase_xy(input_root, phase, args.target)
        X, y, meta = filter_valid_target(X, y, meta, drop_zero_rul=args.drop_zero_rul)
        phase_data[phase] = (X, y, meta, resolved_target)
        datasets.update(meta["dataset"].astype(str).unique().tolist())

    heldouts = args.heldout_datasets if args.heldout_datasets else sorted(datasets)
    for heldout in heldouts:
        print("\n" + "#" * 90)
        print(f"[LODO] Dataset em teste: {heldout}")
        out = base_output / f"heldout_{heldout}"
        cfg = ModelingConfig(
            input_root=input_root,
            output_root=out,
            target=args.target,
            phases=args.phases,
            models=args.models,
            experiments=args.experiments,
            random_seed=args.random_seed,
            test_size=0.0,
            val_size=args.val_size,
            cv_folds=args.cv_folds,
            n_iter=args.n_iter,
            n_jobs=args.n_jobs,
            max_missing_frac=args.max_missing_frac,
            min_variance=args.min_variance,
            mape_min_true=args.mape_min_true,
            drop_zero_rul=args.drop_zero_rul,
            quick=args.quick,
            dataset_balanced_weights=args.dataset_balanced_weights,
            allow_cycle_number_as_feature=args.allow_cycle_number_as_feature,
        )
        ensure_dir(cfg.output_root)
        write_json(cfg.output_root / "logs" / "run_config.json", {**cfg.__dict__, "heldout_dataset": heldout})

        for phase, (X, y, meta, resolved_target) in phase_data.items():
            if heldout not in set(meta["dataset"].astype(str)):
                print(f"[WARN] {heldout} não aparece na fase {phase}; pulando.")
                continue
            split = make_lodo_split(meta, heldout_dataset=heldout, random_seed=args.random_seed, val_size=args.val_size)
            phase_tables = ensure_dir(cfg.output_root / phase / "tables")
            split.summary.to_csv(phase_tables / "split_summary_by_dataset_lodo.csv", index=False)
            split.battery_table.to_csv(phase_tables / "split_batteries_lodo.csv", index=False)
            train_phase(X, y, meta, split.split_labels, phase, cfg)
        build_comparison_tables(cfg.output_root)

    print("\n[OK] LODO concluído. Resultados em:")
    print(base_output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.", file=sys.stderr)
        sys.exit(130)
