#!/usr/bin/env bash
# Braco R: 20 splits repetidos (pooled + intradominio), config pre-especificada,
# 6 modelos, semente de MODELO fixa e semente de SPLIT variando. Uso: bash 07r_repeated.sh [paralelismo=4]
set -euo pipefail; source "$(dirname "$0")/env.sh"
P="${1:-4}"; RUN="$PATCHED/run_modeling_final_campaign_hpc.py"; LOGS="$OUT_V3/logs_repeated"; mkdir -p "$LOGS"
J="$OUT_V3/repeated_jobs.txt"; : > "$J"
for s in $(seq 1 20); do
  echo "python $RUN --input-root $DATA_V3/modeling_ready_t80 --output-root $OUT_V3/repeated/split_s$s --target target_rul80_steps_clipped --campaign-component repeated_split --phases charge discharge --modes global per_dataset --experiments all_features --random-state $s --model-random-state $SEED_MODEL --elasticnet-cv --no-save-predictions --skip-existing --n-jobs 1 --model-n-jobs $MODEL_N_JOBS > $LOGS/split_s$s.log 2>&1" >> "$J"
done
nohup bash -c "xargs -P $P -I{} bash -c '{}' < $J" > /dev/null 2>&1 &
echo "20 sementes lancadas com paralelismo $P (cada uma: 6 modelos x 8 escopos x 2 fases = 96 ajustes)."
