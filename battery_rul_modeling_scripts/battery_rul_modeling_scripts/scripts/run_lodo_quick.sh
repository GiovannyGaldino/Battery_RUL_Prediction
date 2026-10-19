#!/usr/bin/env bash
set -euo pipefail
python run_lodo_modeling_v3.py \
  --input-root ~/outputs/batteries_modeling_v3/modeling_ready \
  --output-root ~/outputs/batteries_modeling_v3/modeling_results_lodo \
  --target target_rul80_steps_clipped \
  --phases charge discharge \
  --models median elasticnet random_forest extra_trees hist_gradient_boosting mlp \
  --experiments all_features \
  --quick \
  --n-iter 4 \
  --cv-folds 3 \
  --n-jobs -1
