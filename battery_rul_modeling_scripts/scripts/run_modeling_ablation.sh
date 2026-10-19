#!/usr/bin/env bash
set -euo pipefail
python run_modeling_v3.py \
  --input-root ~/outputs/batteries_modeling_v3/modeling_ready \
  --output-root ~/outputs/batteries_modeling_v3/modeling_results \
  --target target_rul80_steps_clipped \
  --phases charge discharge \
  --models median elasticnet random_forest extra_trees hist_gradient_boosting mlp \
  --experiments all_features no_curves global_only capacity_only curve_only no_temperature no_energy no_stage_gap resolution_p003 resolution_p009 resolution_p015 resolution_p030 \
  --n-iter 12 \
  --cv-folds 3 \
  --n-jobs -1
