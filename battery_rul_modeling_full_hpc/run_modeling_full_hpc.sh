#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python run_modeling_full_hpc.py \
  --input-root "$HOME/outputs/batteries_modeling_v3/modeling_ready" \
  --output-root "$HOME/outputs/batteries_modeling_v3/modeling_results_full" \
  --phases charge discharge \
  --models median elasticnet random_forest extra_trees hist_gradient_boosting mlp \
  --modes global per_dataset \
  --experiments all_features \
  --sample-weighting dataset_inverse \
  --n-jobs 1 \
  --model-n-jobs -1 \
  --make-plots \
  "$@"
