#!/bin/bash
set -euo pipefail

cd ~/projects/Batteries/battery_rul_modeling_final_robustness_hpc

# Para RF/ExtraTrees:
# n_jobs=2 x model_n_jobs=16 = até 32 núcleos.
# BLAS limitado para evitar competição com os ensembles.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SEEDS=(13 42 73)

for SEED in "${SEEDS[@]}"; do
  echo "================================================================================"
  echo "[REPEATED-SEED] random_state=${SEED}"
  echo "================================================================================"

  python -u run_modeling_final_robustness_hpc.py \
    --input-root ~/outputs/batteries_modeling_v3/modeling_ready \
    --output-root ~/outputs/batteries_modeling_v3/modeling_results_final_repeated_seed_${SEED} \
    --phases charge discharge \
    --modes global per_dataset \
    --experiments all_features no_time_sampling \
    --models median elasticnet random_forest extra_trees hist_gradient_boosting mlp \
    --random-state ${SEED} \
    --test-size 0.2 \
    --missing-threshold 0.95 \
    --sample-weighting dataset_inverse \
    --min-batteries-per-dataset 3 \
    --n-jobs 2 \
    --model-n-jobs 16 \
    --cv-folds 0 \
    --skip-existing

done
