from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from .feature_groups import candidate_numeric_features, select_experiment_features
from .io_utils import ensure_dir, write_json, write_text
from .metrics import add_error_columns, grouped_metrics, make_normalized_rul_bins, make_rul_bins, regression_metrics
from .models import dataset_balanced_sample_weights, make_pipeline, param_distributions, supports_sample_weight
from .plots import plot_bar_metric, plot_pred_vs_true, plot_residuals

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def neg_rmse(y_true, y_pred):
    return -float(np.sqrt(mean_squared_error(y_true, y_pred)))

RMSE_SCORER = make_scorer(neg_rmse, greater_is_better=True)


def _fit_search_or_default(
    model_name: str,
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    cfg,
    meta_train: pd.DataFrame,
):
    params = param_distributions(model_name, quick=cfg.quick)
    fit_params = {}
    if cfg.dataset_balanced_weights and supports_sample_weight(pipeline):
        fit_params["model__sample_weight"] = dataset_balanced_sample_weights(meta_train["dataset"])

    unique_groups = pd.Series(groups_train).nunique()
    if model_name == "median" or not params or unique_groups < 2:
        pipeline.fit(X_train, y_train, **fit_params)
        return pipeline, {}, None

    n_splits = min(cfg.cv_folds, unique_groups)
    if n_splits < 2:
        pipeline.fit(X_train, y_train, **fit_params)
        return pipeline, {}, None

    cv = GroupKFold(n_splits=n_splits)
    n_iter = min(cfg.n_iter, int(np.prod([len(v) for v in params.values()]))) if params else 0
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=max(1, n_iter),
        scoring=RMSE_SCORER,
        cv=cv,
        random_state=cfg.random_seed,
        n_jobs=cfg.n_jobs,
        verbose=1,
        refit=True,
        error_score=np.nan,
        return_train_score=True,
    )
    search.fit(X_train, y_train, groups=groups_train, **fit_params)
    return search.best_estimator_, search.best_params_, search


def _refit_final(best_estimator, best_params: Dict, X_fit: pd.DataFrame, y_fit: pd.Series, cfg, meta_fit: pd.DataFrame):
    final_model = clone(best_estimator)
    if best_params:
        final_model.set_params(**best_params)
    fit_params = {}
    if cfg.dataset_balanced_weights and supports_sample_weight(final_model):
        fit_params["model__sample_weight"] = dataset_balanced_sample_weights(meta_fit["dataset"])
    final_model.fit(X_fit, y_fit, **fit_params)
    return final_model


def _prediction_frame(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    phase: str,
    experiment: str,
    model_name: str,
    split: str,
) -> pd.DataFrame:
    out = meta.reset_index(drop=True).copy()
    out["y_true"] = np.asarray(y_true, dtype=float)
    out["y_pred"] = np.asarray(y_pred, dtype=float)
    out["phase"] = out.get("phase", phase)
    out["phase_model"] = phase
    out["experiment"] = experiment
    out["model"] = model_name
    out["split"] = split
    out = add_error_columns(out)
    out["rul_bin_abs"] = make_rul_bins(out)
    out["rul_bin_norm"] = make_normalized_rul_bins(out)
    return out


def _save_metrics_and_plots(pred_df: pd.DataFrame, base_dir: Path, split: str, cfg) -> Dict[str, pd.DataFrame]:
    phase = pred_df["phase_model"].iloc[0]
    model = pred_df["model"].iloc[0]
    experiment = pred_df["experiment"].iloc[0]

    metric_row = {
        "phase": phase,
        "experiment": experiment,
        "model": model,
        "split": split,
    }
    metric_row.update(regression_metrics(pred_df["y_true"], pred_df["y_pred"], meta=pred_df, mape_min_true=cfg.mape_min_true))
    global_metrics = pd.DataFrame([metric_row])

    dataset_metrics = grouped_metrics(pred_df, ["phase_model", "experiment", "model", "split", "dataset"], cfg.mape_min_true)
    battery_metrics = grouped_metrics(pred_df, ["phase_model", "experiment", "model", "split", "dataset", "battery_id"], cfg.mape_min_true)
    rul_abs_metrics = grouped_metrics(pred_df, ["phase_model", "experiment", "model", "split", "rul_bin_abs"], cfg.mape_min_true)
    rul_norm_metrics = grouped_metrics(pred_df, ["phase_model", "experiment", "model", "split", "rul_bin_norm"], cfg.mape_min_true)

    metrics_dir = ensure_dir(base_dir / phase / "metrics" / experiment)
    plots_dir = ensure_dir(base_dir / phase / "plots" / experiment / model / split)

    global_metrics.to_csv(metrics_dir / f"{model}_{split}_global_metrics.csv", index=False)
    dataset_metrics.to_csv(metrics_dir / f"{model}_{split}_dataset_metrics.csv", index=False)
    battery_metrics.to_csv(metrics_dir / f"{model}_{split}_battery_metrics.csv", index=False)
    rul_abs_metrics.to_csv(metrics_dir / f"{model}_{split}_rul_abs_bin_metrics.csv", index=False)
    rul_norm_metrics.to_csv(metrics_dir / f"{model}_{split}_rul_norm_bin_metrics.csv", index=False)

    plot_pred_vs_true(pred_df, plots_dir / "pred_vs_true.png", f"{phase} | {experiment} | {model} | {split}")
    plot_residuals(pred_df, plots_dir / "residuals_vs_rul.png", f"{phase} | {experiment} | {model} | {split}")
    plot_bar_metric(dataset_metrics, "dataset", "mae", plots_dir / "mae_by_dataset.png", "MAE por dataset")
    plot_bar_metric(battery_metrics, "battery_id", "mae", plots_dir / "mae_by_battery_top30.png", "MAE por bateria - top 30")
    plot_bar_metric(rul_abs_metrics, "rul_bin_abs", "mae", plots_dir / "mae_by_abs_rul_bin.png", "MAE por faixa absoluta de RUL")
    plot_bar_metric(rul_norm_metrics, "rul_bin_norm", "mae", plots_dir / "mae_by_norm_rul_bin.png", "MAE por faixa normalizada de RUL")

    return {
        "global": global_metrics,
        "dataset": dataset_metrics,
        "battery": battery_metrics,
        "rul_abs": rul_abs_metrics,
        "rul_norm": rul_norm_metrics,
    }


def train_phase(
    X_raw: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    split_labels: pd.Series,
    phase: str,
    cfg,
) -> Dict[str, List[pd.DataFrame]]:
    output_root = ensure_dir(cfg.output_root)
    all_numeric = candidate_numeric_features(X_raw, allow_cycle_number_as_feature=cfg.allow_cycle_number_as_feature)
    if not all_numeric:
        raise ValueError(f"Nenhuma feature numérica candidata encontrada para a fase {phase}.")

    aggregate = {"global": [], "dataset": [], "battery": [], "rul_abs": [], "rul_norm": []}

    for experiment in cfg.experiments:
        feature_cols = select_experiment_features(all_numeric, experiment)
        if not feature_cols:
            print(f"[WARN] Experimento {experiment} sem features na fase {phase}; pulando.")
            continue
        tables_dir = ensure_dir(output_root / phase / "tables" / experiment)
        write_text(tables_dir / "feature_columns_input.txt", feature_cols)

        X = X_raw[feature_cols].copy()
        train_mask = split_labels.eq("train").to_numpy()
        val_mask = split_labels.eq("val").to_numpy()
        test_mask = split_labels.eq("test").to_numpy()
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            raise ValueError(f"Split inválido para {phase}: treino={train_mask.sum()}, teste={test_mask.sum()}.")

        X_train, y_train, meta_train = X.loc[train_mask], y.loc[train_mask], meta.loc[train_mask]
        X_val, y_val, meta_val = X.loc[val_mask], y.loc[val_mask], meta.loc[val_mask]
        X_test, y_test, meta_test = X.loc[test_mask], y.loc[test_mask], meta.loc[test_mask]

        X_fit = pd.concat([X_train, X_val], axis=0) if len(X_val) else X_train
        y_fit = pd.concat([y_train, y_val], axis=0) if len(y_val) else y_train
        meta_fit = pd.concat([meta_train, meta_val], axis=0) if len(meta_val) else meta_train

        for model_name in cfg.models:
            start = time.time()
            print(f"\n[INFO] Fase={phase} | experimento={experiment} | modelo={model_name}")
            model_dir = ensure_dir(output_root / phase / "models" / experiment)
            pred_dir = ensure_dir(output_root / phase / "predictions" / experiment)
            log_dir = ensure_dir(output_root / "logs")

            pipeline = make_pipeline(model_name, cfg.max_missing_frac, cfg.min_variance, cfg.random_seed)
            try:
                best_estimator, best_params, search = _fit_search_or_default(
                    model_name, pipeline, X_train, y_train, meta_train["battery_id"], cfg, meta_train
                )

                # Validation uses the estimator fitted only on the training subset.
                if len(X_val):
                    val_pred = best_estimator.predict(X_val)
                    val_df = _prediction_frame(y_val, val_pred, meta_val, phase, experiment, model_name, "val")
                    val_df.to_csv(pred_dir / f"{model_name}_val_predictions.csv", index=False)
                    metrics = _save_metrics_and_plots(val_df, output_root, "val", cfg)
                    for k, v in metrics.items():
                        aggregate[k].append(v)

                final_model = _refit_final(best_estimator, best_params, X_fit, y_fit, cfg, meta_fit)
                test_pred = final_model.predict(X_test)
                test_df = _prediction_frame(y_test, test_pred, meta_test, phase, experiment, model_name, "test")
                test_df.to_csv(pred_dir / f"{model_name}_test_predictions.csv", index=False)
                metrics = _save_metrics_and_plots(test_df, output_root, "test", cfg)
                for k, v in metrics.items():
                    aggregate[k].append(v)

                joblib.dump(final_model, model_dir / f"{model_name}.joblib")

                cleaner_report = None
                try:
                    cleaner_report = final_model.named_steps["cleaner"].drop_report_
                    cleaner_report.to_csv(tables_dir / f"{model_name}_feature_cleaning_report.csv", index=False)
                    write_text(tables_dir / f"{model_name}_feature_columns_after_cleaning.txt", final_model.named_steps["cleaner"].feature_names_out_)
                except Exception:
                    pass

                if search is not None:
                    cv_results = pd.DataFrame(search.cv_results_)
                    cv_results.to_csv(tables_dir / f"{model_name}_cv_results.csv", index=False)

                elapsed = time.time() - start
                write_json(log_dir / f"{phase}_{experiment}_{model_name}_run.json", {
                    "phase": phase,
                    "experiment": experiment,
                    "model": model_name,
                    "best_params": best_params,
                    "n_input_features": len(feature_cols),
                    "n_train_rows": int(len(X_train)),
                    "n_val_rows": int(len(X_val)),
                    "n_test_rows": int(len(X_test)),
                    "elapsed_seconds": elapsed,
                    "status": "ok",
                })
                print(f"[OK] Concluído em {elapsed:.1f} s")
            except Exception as exc:
                elapsed = time.time() - start
                write_json(log_dir / f"{phase}_{experiment}_{model_name}_ERROR.json", {
                    "phase": phase,
                    "experiment": experiment,
                    "model": model_name,
                    "elapsed_seconds": elapsed,
                    "status": "error",
                    "error": repr(exc),
                })
                print(f"[ERRO] {phase}/{experiment}/{model_name}: {exc}")
                continue

    for kind, frames in aggregate.items():
        if frames:
            df = pd.concat(frames, ignore_index=True)
            out_name = {
                "global": "global_metrics_all.csv",
                "dataset": "dataset_metrics_all.csv",
                "battery": "battery_metrics_all.csv",
                "rul_abs": "rul_abs_bin_metrics_all.csv",
                "rul_norm": "rul_norm_bin_metrics_all.csv",
            }[kind]
            ensure_dir(output_root / phase / "metrics").joinpath(out_name).write_text(df.to_csv(index=False), encoding="utf-8")
    return aggregate
