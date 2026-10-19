# Final Consolidated Experimental Campaign — Battery RUL Modeling

This package defines a single final experimental campaign for the Energies manuscript. It is intended to replace the previous partial/modeling runs as the official source of final results.

## 1. Conceptual design

The final campaign is not a post-hoc correction. It is a complete, predefined model-feature-validation matrix with robustness diagnostics.

### Data branches

- `charge`
- `discharge`

The branches are modeled independently under the same protocol.

### Target

- `target_rul80_steps_clipped`

### Datasets expected in the prepared modeling files

- `CALCE`
- `HUST`
- `MATR`
- `NRW`
- `OXFORD`
- `PCOE`
- `RWTH`
- `XJTU`

### Feature experiments

The campaign evaluates all 13 feature configurations:

1. `all_features`
2. `no_curves`
3. `global_stats_only`
4. `curves_only`
5. `capacity_only`
6. `energy_added`
7. `no_temperature`
8. `no_time_sampling`
9. `no_stage_gap`
10. `resolution_p003`
11. `resolution_p009`
12. `resolution_p015`
13. `resolution_p030`

### Model families

Six model families are evaluated with fixed reproducible configurations:

1. `median`
2. `elasticnet`
3. `random_forest`
4. `extra_trees`
5. `hist_gradient_boosting`
6. `mlp`

No exhaustive grid search or randomized hyperparameter search is performed. The focus is comparability across phases, datasets, feature families, and validation regimes.

## 2. Validation protocol

The final campaign has three predefined components.

### Component A — Full matrix

Reference seed: `42`.

Modes:

- `global`: mixed-dataset battery-level holdout split;
- `per_dataset`: independent battery-level holdout split inside each dataset;
- `lodo`: leave-one-dataset-out validation.

Expected final-test model configurations:

```text
2 phases × 13 experiments × 6 models × (1 global + 8 per_dataset + 8 lodo) = 2652
```

### Component B — Repeated split stability

Additional seeds: `13` and `73`.

Modes:

- `global`
- `per_dataset`

The reference seed `42` for these modes is already available in Component A. LODO is not repeated because the test domain is fixed by the held-out dataset rather than by a random battery split.

Expected additional final-test configurations:

```text
2 additional seeds × 2 phases × 13 experiments × 6 models × (1 global + 8 per_dataset) = 2808
```

### Component C — Internal GroupKFold

Reference seed: `42`.

Mode:

- `global`

CV:

- 3-fold `GroupKFold` by `battery_id` inside the training set.

Expected final-test configurations:

```text
2 phases × 13 experiments × 6 models = 156
```

Expected internal CV rows:

```text
2 phases × 13 experiments × 6 models × 3 folds = 468
```

## 3. HPC execution strategy

The campaign is sharded to use a 32-core node safely.

Default configuration:

```text
MAX_PARALLEL=2
N_JOBS=2
MODEL_N_JOBS=8
```

Approximate upper bound:

```text
2 shards × 2 model-level jobs × 8 tree-ensemble cores = 32 cores
```

Each shard writes to a separate output directory, so parallel shards do not overwrite each other's master CSV files.

## 4. Installation on the HPC

Upload this package to the HPC/JupyterLab home directory and run:

```bash
mkdir -p ~/projects/Batteries

tar -xzf battery_rul_modeling_final_campaign_hpc.tar.gz -C ~/projects/Batteries

cd ~/projects/Batteries/battery_rul_modeling_final_campaign_hpc
chmod +x *.sh *.py
```

## 5. Launch

```bash
cd ~/projects/Batteries/battery_rul_modeling_final_campaign_hpc
bash launch_final_campaign_nohup.sh
```

Monitor:

```bash
tail -f ~/outputs/batteries_modeling_v3/final_campaign_logs/final_campaign_master.log
```

Detailed nohup log:

```bash
tail -f ~/outputs/batteries_modeling_v3/final_campaign_logs/final_campaign_nohup.log
```

List active Python processes:

```bash
pgrep -af run_modeling_final_campaign_hpc.py
```

CPU usage:

```bash
ps -eo pid,stat,etime,%cpu,%mem,cmd | grep run_modeling_final_campaign_hpc.py | grep -v grep
```

## 6. Output structure

Main output root:

```text
~/outputs/batteries_modeling_v3/modeling_results_final_campaign_v1
```

Logs:

```text
~/outputs/batteries_modeling_v3/final_campaign_logs
```

Shard outputs:

```text
modeling_results_final_campaign_v1/shards/<component>/seed_<seed>/<phase>_<mode>/...
```

Consolidated summary:

```text
modeling_results_final_campaign_v1/summary
```

Important summary files:

```text
final_campaign_metrics_global.csv
final_campaign_metrics_dataset.csv
final_campaign_metrics_battery.csv
final_campaign_metrics_rul_bin.csv
final_campaign_metrics_macro_by_dataset.csv
final_campaign_cv_metrics.csv
final_campaign_repeated_seed_stability.csv
final_campaign_best_by_case.csv
final_campaign_completion_status.csv
```

## 7. Stopping the campaign

```bash
cd ~/projects/Batteries/battery_rul_modeling_final_campaign_hpc
bash stop_final_campaign.sh
```

## 8. Clean restart

Only run this if you intentionally want to delete all final-campaign outputs and start again:

```bash
cd ~/projects/Batteries/battery_rul_modeling_final_campaign_hpc
bash clean_final_campaign_outputs.sh
```

## 9. Methodological wording for the article

Use the final campaign as the only official modeling protocol. Do not describe earlier partial runs.

Recommended section title:

```text
Machine-Learning Evaluation Protocol and Robustness Analysis
```

Recommended wording:

```text
The final experimental campaign was defined as a single consolidated model-feature-validation matrix. Charge and discharge branches were modeled independently using the same feature configurations, model families, and validation protocols. Six model families were evaluated with fixed reproducible configurations: a median baseline, Elastic Net, Random Forest, Extra Trees, HistGradientBoosting, and a compact multilayer perceptron. The objective was not exhaustive hyperparameter optimization, but a controlled comparison of charge- and discharge-derived feature representations across heterogeneous public datasets.

Three validation regimes were used: mixed-dataset grouped validation, intra-dataset grouped validation, and leave-one-dataset-out validation. In all cases, splitting was performed by battery to avoid leakage between cycles from the same cell. To assess the sensitivity of random battery-level splits, the global and intra-dataset protocols were repeated with three random seeds. In addition, the global mixed-dataset protocol was evaluated with an internal three-fold GroupKFold validation by battery as a robustness diagnostic within the training data. This internal validation was not used for hyperparameter optimization.
```
