#!/bin/bash
set -euo pipefail

cd ~/projects/Batteries/battery_rul_modeling_final_robustness_hpc

# MLP usa operações matriciais; deixamos algumas threads BLAS.
# Como há apenas um modelo por caso, --n-jobs maior não ajuda.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

python -u run_modeling_final_robustness_hpc.py \
  --input-root ~/outputs/batteries_modeling_v3/modeling_ready \
  --output-root ~/outputs/batteries_modeling_v3/modeling_results_final_mlp_reduced \
  --phases charge discharge \
  --modes global per_dataset lodo \
  --experiments all_features no_time_sampling capacity_only curves_only resolution_p015 \
  --models mlp \
  --random-state 42 \
  --test-size 0.2 \
  --missing-threshold 0.95 \
  --sample-weighting dataset_inverse \
  --min-batteries-per-dataset 3 \
  --n-jobs 1 \
  --model-n-jobs 1 \
  --cv-folds 0 \
  --skip-existing
