# Battery RUL Modeling – Full HPC Pipeline

This package contains a complete modeling script for lithium-ion battery RUL prediction using the already prepared files:

```text
~/outputs/batteries_modeling_v3/modeling_ready/charge_X.parquet
~/outputs/batteries_modeling_v3/modeling_ready/charge_y.parquet
~/outputs/batteries_modeling_v3/modeling_ready/charge_id.parquet
~/outputs/batteries_modeling_v3/modeling_ready/discharge_X.parquet
~/outputs/batteries_modeling_v3/modeling_ready/discharge_y.parquet
~/outputs/batteries_modeling_v3/modeling_ready/discharge_id.parquet
```

## What the script does

The script runs:

1. Global mixed-dataset models with dataset-balanced sample weights.
2. Per-dataset models, split by battery inside each dataset.
3. Optional LODO models, where one dataset is completely held out.
4. Optional ablation experiments:
   - `all_features`
   - `no_curves`
   - `global_stats_only`
   - `curves_only`
   - `capacity_only`
   - `energy_added`
   - `no_temperature`
   - `no_time_sampling`
   - `no_stage_gap`
   - `resolution_p003`
   - `resolution_p009`
   - `resolution_p015`
   - `resolution_p030`

For each case, it computes and saves:

- Global metrics
- Metrics by dataset
- Metrics by battery
- Metrics by RUL range
- Predictions with metadata
- Trained model
- Optional plots

## Output structure

Default output root:

```text
~/outputs/batteries_modeling_v3/modeling_results_full
```

Example:

```text
modeling_results_full/
  all_metrics_global.csv
  all_metrics_dataset.csv
  all_metrics_battery.csv
  all_metrics_rul_bin.csv
  all_metrics_macro_by_dataset.csv
  charge/
    global/
      all_datasets/
        all_features/
          random_forest/
            model.joblib
            predictions.csv
            metrics_global.csv
            metrics_by_dataset.csv
            metrics_by_battery.csv
            metrics_by_rul_bin.csv
            plots/
```

## Recommended first run

Use this first to validate the full pipeline:

```bash
cd ~/projects/Batteries/battery_rul_modeling_full_hpc
bash run_modeling_full_hpc.sh
```

This runs only:

- charge and discharge
- global + per_dataset
- all_features
- all models

## Complete run with ablation and LODO

This may be expensive:

```bash
bash run_modeling_complete_hpc.sh
```

## Useful custom commands

Run only tree models:

```bash
python run_modeling_full_hpc.py \
  --modes global per_dataset \
  --experiments all_features \
  --models median random_forest extra_trees hist_gradient_boosting \
  --n-jobs 1 \
  --model-n-jobs -1 \
  --make-plots
```

Run all ablations only for global models:

```bash
python run_modeling_full_hpc.py \
  --modes global \
  --experiments all \
  --models median random_forest hist_gradient_boosting \
  --n-jobs 1 \
  --model-n-jobs -1
```

Run LODO only:

```bash
python run_modeling_full_hpc.py \
  --modes lodo \
  --experiments all_features \
  --models median random_forest hist_gradient_boosting \
  --n-jobs 1 \
  --model-n-jobs -1
```

## Notes on parallelism

For HPC use:

- `--model-n-jobs -1` lets RandomForest and ExtraTrees use all available cores internally.
- `--n-jobs 1` avoids oversubscription.
- If you want to train models in parallel, use for example `--n-jobs 3 --model-n-jobs 4`, depending on the available cores.

Using `--n-jobs len(models)` together with `--model-n-jobs -1` can oversubscribe the machine.
