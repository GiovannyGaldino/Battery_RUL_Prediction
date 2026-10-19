#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from battery_rul_modeling.comparison import build_comparison_tables
from battery_rul_modeling.config import ModelingConfig
from battery_rul_modeling.io_utils import ensure_dir, filter_valid_target, load_phase_xy, write_json
from battery_rul_modeling.splits import make_group_split, subsample_rows_by_battery
from battery_rul_modeling.training import train_phase


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina modelos de RUL por ramo de carga/descarga usando parquet já preparado."
    )
    parser.add_argument("--input-root", default="~/outputs/batteries_modeling_v3/modeling_ready")
    parser.add_argument("--output-root", default="~/outputs/batteries_modeling_v3/modeling_results")
    parser.add_argument("--target", default="target_rul80_steps_clipped")
    parser.add_argument("--phases", nargs="+", default=["charge", "discharge"], choices=["charge", "discharge"])
    parser.add_argument("--models", nargs="+", default=["median", "elasticnet", "random_forest", "extra_trees", "hist_gradient_boosting", "mlp"])
    parser.add_argument("--experiments", nargs="+", default=["all_features"])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-missing-frac", type=float, default=0.60)
    parser.add_argument("--min-variance", type=float, default=1e-12)
    parser.add_argument("--mape-min-true", type=float, default=1.0)
    parser.add_argument("--drop-zero-rul", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Reduz a busca de hiperparâmetros para teste inicial.")
    parser.add_argument("--dataset-balanced-weights", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Amostragem opcional para teste rápido.")
    parser.add_argument("--max-test-rows", type=int, default=None, help="Amostragem opcional do conjunto de teste para debug.")
    parser.add_argument("--allow-cycle-number-as-feature", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ModelingConfig(
        input_root=Path(args.input_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        target=args.target,
        phases=args.phases,
        models=args.models,
        experiments=args.experiments,
        random_seed=args.random_seed,
        test_size=args.test_size,
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
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        allow_cycle_number_as_feature=args.allow_cycle_number_as_feature,
    )
    ensure_dir(cfg.output_root)
    write_json(cfg.output_root / "logs" / "run_config.json", cfg.__dict__)

    print("Battery RUL modeling v3")
    print(f"Input root:  {cfg.input_root}")
    print(f"Output root: {cfg.output_root}")
    print(f"Target:      {cfg.target}")
    print(f"Phases:      {cfg.phases}")
    print(f"Models:      {cfg.models}")
    print(f"Experiments: {cfg.experiments}")

    for phase in cfg.phases:
        print("\n" + "=" * 80)
        print(f"[INFO] Carregando fase: {phase}")
        X_raw, y, meta, resolved_target = load_phase_xy(cfg.input_root, phase, cfg.target)
        X_raw, y, meta = filter_valid_target(X_raw, y, meta, drop_zero_rul=cfg.drop_zero_rul)
        print(f"[INFO] {phase}: {len(X_raw)} amostras após filtro de alvo. Target resolvido: {resolved_target}")
        print(f"[INFO] {phase}: {meta['battery_id'].nunique()} baterias | {meta['dataset'].nunique()} datasets")

        split = make_group_split(meta, random_seed=cfg.random_seed, test_size=cfg.test_size, val_size=cfg.val_size)
        phase_tables = ensure_dir(cfg.output_root / phase / "tables")
        split.summary.to_csv(phase_tables / "split_summary_by_dataset.csv", index=False)
        split.battery_table.to_csv(phase_tables / "split_batteries.csv", index=False)
        meta.assign(split=split.split_labels, target=y).to_csv(phase_tables / "modeling_rows_with_split.csv", index=False)

        # Optional row subsampling for debugging. Split labels must be recomputed after subsampling.
        if cfg.max_train_rows is not None:
            train_like = split.split_labels.isin(["train", "val"])
            X_trainlike, y_trainlike, meta_trainlike = subsample_rows_by_battery(
                X_raw.loc[train_like].reset_index(drop=True),
                y.loc[train_like].reset_index(drop=True),
                meta.loc[train_like].reset_index(drop=True),
                cfg.max_train_rows,
                cfg.random_seed,
            )
            X_testlike, y_testlike, meta_testlike = X_raw.loc[~train_like].reset_index(drop=True), y.loc[~train_like].reset_index(drop=True), meta.loc[~train_like].reset_index(drop=True)
            if cfg.max_test_rows is not None:
                X_testlike, y_testlike, meta_testlike = subsample_rows_by_battery(
                    X_testlike, y_testlike, meta_testlike, cfg.max_test_rows, cfg.random_seed
                )
            X_raw = pd.concat([X_trainlike, X_testlike], ignore_index=True)
            y = pd.concat([y_trainlike, y_testlike], ignore_index=True)
            meta = pd.concat([meta_trainlike, meta_testlike], ignore_index=True)
            split = make_group_split(meta, random_seed=cfg.random_seed, test_size=cfg.test_size, val_size=cfg.val_size)
            split.summary.to_csv(phase_tables / "split_summary_by_dataset_subsampled.csv", index=False)

        train_phase(X_raw, y, meta, split.split_labels, phase, cfg)

    build_comparison_tables(cfg.output_root)
    print("\n[OK] Modelagem concluída. Resultados em:")
    print(cfg.output_root)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.", file=sys.stderr)
        sys.exit(130)
