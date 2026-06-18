#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Full run including global, per-dataset, LODO, and all ablation experiments.
# This can be computationally expensive. Adjust --models, --n-jobs, and --model-n-jobs as needed.
python run_modeling_full_hpc.py \
  --input-root "$HOME/outputs/batteries_modeling_v3/modeling_ready" \
  --output-root "$HOME/outputs/batteries_modeling_v3/modeling_results_full" \
  --phases charge discharge \
  --models median elasticnet random_forest extra_trees hist_gradient_boosting mlp \
  --modes global per_dataset lodo \
  --experiments all \
  --sample-weighting dataset_inverse \
  --n-jobs 1 \
  --model-n-jobs -1 \
  --make-plots \
  "$@"
