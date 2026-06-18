#!/bin/bash
set -euo pipefail

cd ~/projects/Batteries/battery_rul_modeling_final_robustness_hpc

# ElasticNet não usa model_n_jobs de forma relevante.
# Mantemos BLAS controlado para evitar oversubscription.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python -u run_modeling_final_robustness_hpc.py \
  --input-root ~/outputs/batteries_modeling_v3/modeling_ready \
  --output-root ~/outputs/batteries_modeling_v3/modeling_results_final_elasticnet_full \
  --phases charge discharge \
  --modes global per_dataset lodo \
  --experiments all \
  --models elasticnet \
  --random-state 42 \
  --test-size 0.2 \
  --missing-threshold 0.95 \
  --sample-weighting dataset_inverse \
  --min-batteries-per-dataset 3 \
  --n-jobs 1 \
  --model-n-jobs 1 \
  --cv-folds 0 \
  --skip-existing
