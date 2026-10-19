#!/usr/bin/env bash
# Braco B: limiar 0.85, config pre-especificada (all_features) + no_causal_trend, 6 modelos, 3 modos.
# Repartido em 4 processos (2 experimentos x 2 fases), MODEL_N_JOBS threads cada; --skip-existing.
# Uso: bash 07b_threshold85.sh
set -euo pipefail; source "$(dirname "$0")/env.sh"
[ -f "$DATA_V3/modeling_ready_t85/charge_X.parquet" ] || { echo "ABORTADO: falta modeling_ready_t85 (03_build_matrix.sh)"; exit 1; }
LOGS="$OUT_V3/logs_t85"; mkdir -p "$LOGS"
for E in all_features no_causal_trend; do for P in charge discharge; do
  nohup python "$PATCHED/run_modeling_final_campaign_hpc.py" --input-root "$DATA_V3/modeling_ready_t85" \
      --output-root "$OUT_V3/arm_t85" --target target_rul85_steps_clipped --campaign-component threshold_85 \
      --phases "$P" --modes global lodo per_dataset --experiments "$E" \
      --random-state $SEED_SPLIT --model-random-state $SEED_MODEL --elasticnet-cv \
      --no-save-predictions --skip-existing --n-jobs 1 --model-n-jobs "$MODEL_N_JOBS" > "$LOGS/t85_${E}_${P}.log" 2>&1 &
  sleep 1
done; done
echo "braco B lancado: 4 processos x $MODEL_N_JOBS threads (alvo 360 ajustes). logs: $LOGS/t85_*.log"
