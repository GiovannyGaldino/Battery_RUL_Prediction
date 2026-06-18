#!/usr/bin/env bash
set -euo pipefail
python inspect_modeling_inputs.py \
  --input-root ~/outputs/batteries_modeling_v3/modeling_ready \
  --target target_rul80_steps_clipped
