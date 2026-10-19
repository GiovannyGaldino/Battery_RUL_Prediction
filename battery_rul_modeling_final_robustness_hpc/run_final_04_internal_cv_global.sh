#!/bin/bash
set -euo pipefail

cd ~/projects/Batteries/battery_rul_modeling_final_robustness_hpc

# GroupKFold interno para configurações globais.
# Dois modelos em paralelo; RF/ExtraTrees usam até 16 threads cada.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u run_modeling_final_robustness_hpc.py \
  --input-root ~/outputs/batteries_modeling_v3/modeling_ready \
  --output-root ~/outputs/batteries_modeling_v3/modeling_results_final_internal_cv_global \
  --phases charge discharge \
  --modes global \
  --experiments all_features no_time_sampling \
  --models elasticnet random_forest extra_trees hist_gradient_boosting mlp \
  --random-state 42 \
  --test-size 0.2 \
  --missing-threshold 0.95 \
  --sample-weighting dataset_inverse \
  --min-batteries-per-dataset 3 \
  --n-jobs 2 \
  --model-n-jobs 16 \
  --cv-folds 3 \
  --skip-existing
