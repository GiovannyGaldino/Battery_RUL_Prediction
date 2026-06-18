# Final robustness modeling package

This package adapts the current full HPC modeling pipeline to run a final robustness layer for the lithium-ion battery RUL article.

The goal is not to reopen the whole modeling study with expensive hyperparameter search. Instead, the package adds targeted experiments that improve methodological robustness:

1. Elastic Net in the complete matrix.
2. Compact MLP in a reduced matrix.
3. Repeated random seeds for a reduced global/per-dataset matrix.
4. Internal GroupKFold validation with 3 folds for selected global configurations.
5. No SVR/KNN/GPR, due to high cost and limited methodological gain at this scale.

## Main script

```bash
run_modeling_final_robustness_hpc.py
```

It is based on the previous `run_modeling_full_hpc.py`, with these changes:

- Compact MLP changed to `(64, 32)` hidden units.
- Elastic Net max iterations increased to `10000`.
- Metrics now store `random_state`, `cv_folds_requested`, `sample_weighting`, and `missing_threshold`.
- Default output root changed to `modeling_results_final_robustness`.

## Recommended execution

Copy this folder to the HPC path:

```bash
~/projects/Batteries/battery_rul_modeling_final_robustness_hpc
```

Then run:

```bash
cd ~/projects/Batteries/battery_rul_modeling_final_robustness_hpc
bash launch_final_robustness_nohup.sh
```

Monitor with:

```bash
tail -f ~/outputs/batteries_modeling_v3/modeling_final_robustness_all.log
```

Check process:

```bash
pgrep -af run_modeling_final_robustness_hpc.py
```

## Individual scripts

### 1. Elastic Net in complete matrix

```bash
bash run_final_01_elasticnet_full_matrix.sh
```

Expected combinations:

```text
2 phases × 13 experiments × (1 global + 8 per_dataset + 8 lodo) × 1 model = 442 fits
```

Output:

```text
~/outputs/batteries_modeling_v3/modeling_results_final_elasticnet_full
```

### 2. Compact MLP in reduced matrix

```bash
bash run_final_02_mlp_reduced_matrix.sh
```

Reduced experiments:

```text
all_features
no_time_sampling
capacity_only
curves_only
resolution_p015
```

Expected combinations:

```text
2 phases × 5 experiments × (1 global + 8 per_dataset + 8 lodo) × 1 model = 170 fits
```

Output:

```text
~/outputs/batteries_modeling_v3/modeling_results_final_mlp_reduced
```

### 3. Repeated seeds

```bash
bash run_final_03_repeated_seeds_reduced.sh
```

Seeds:

```text
13, 42, 73
```

Modes:

```text
global, per_dataset
```

Experiments:

```text
all_features, no_time_sampling
```

Models:

```text
median, elasticnet, random_forest, extra_trees, hist_gradient_boosting, mlp
```

Expected combinations:

```text
3 seeds × 2 phases × 2 experiments × (1 global + 8 per_dataset) × 6 models = 648 fits
```

Outputs:

```text
~/outputs/batteries_modeling_v3/modeling_results_final_repeated_seed_13
~/outputs/batteries_modeling_v3/modeling_results_final_repeated_seed_42
~/outputs/batteries_modeling_v3/modeling_results_final_repeated_seed_73
```

### 4. Internal GroupKFold validation

```bash
bash run_final_04_internal_cv_global.sh
```

Modes:

```text
global
```

Experiments:

```text
all_features, no_time_sampling
```

Models:

```text
elasticnet, random_forest, extra_trees, hist_gradient_boosting, mlp
```

CV:

```text
GroupKFold by battery, 3 folds, applied only inside the training set
```

Expected final fits:

```text
2 phases × 2 experiments × 5 models = 20 final fits
```

Each fit also generates `cv_metrics.csv` with 3 internal folds when possible.

Output:

```text
~/outputs/batteries_modeling_v3/modeling_results_final_internal_cv_global
```

## Summary collection

After the runs, collect results with:

```bash
python -u collect_final_robustness_results.py \
  --outputs-root ~/outputs/batteries_modeling_v3 \
  --output-dir ~/outputs/batteries_modeling_v3/final_robustness_summary
```

Main summary outputs:

```text
final_robustness_global.csv
final_robustness_dataset.csv
final_robustness_battery.csv
final_robustness_rul_bin.csv
final_robustness_macro_by_dataset.csv
final_robustness_cv_metrics.csv
final_robustness_metric_stability.csv
final_robustness_summary.json
```

## Methodological interpretation

These runs support the following manuscript claims:

- A regularized linear baseline was evaluated in the complete experimental matrix.
- A compact neural-network baseline was evaluated in a reduced but representative matrix.
- Split sensitivity was assessed by repeated battery-level splits using multiple random seeds.
- Internal validation was performed with grouped cross-validation by battery for selected global configurations.
- The final comparison remains centered on feature representation, charge/discharge phase, feature-family ablation, dataset imbalance, and cross-dataset generalization, not exhaustive model optimization.
