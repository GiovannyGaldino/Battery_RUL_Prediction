#!/usr/bin/env bash
# Nucleo, braco A (RUL80, features causais): 14 experimentos x 6 modelos x 3 modos.
# Fatiado por experimento como o run_campanha.sh de v2. Uso: bash 06_campaign.sh [n_shards]
set -euo pipefail; source "$(dirname "$0")/env.sh"
[ -d "$OUT_V3/quick_test" ] || { echo "ABORTADO: rode 05_quick_test.sh antes e confira o portao 3"; exit 1; }
N="${1:-14}"; LOGS="$OUT_V3/logs_core"; mkdir -p "$LOGS"
QUOTA=$(awk '$1!="max"{printf "%d", $1/$2} $1=="max"{print "sem cota"}' /sys/fs/cgroup/cpu.max 2>/dev/null || nproc)
echo "shards: $N | threads por modelo: $MODEL_N_JOBS | total ~$((N*MODEL_N_JOBS)) | nproc $(nproc) | cota cgroup: $QUOTA nucleos"
EXPS=(all_features no_curves global_stats_only curves_only capacity_only energy_added no_temperature \
      no_time_sampling no_stage_gap resolution_p003 resolution_p009 resolution_p015 resolution_p030 no_causal_trend)
for ((s=0; s<N; s++)); do
  lista=""; for ((i=s; i<${#EXPS[@]}; i+=N)); do lista="$lista ${EXPS[$i]}"; done; [ -z "$lista" ] && continue
  nohup python "$PATCHED/run_modeling_final_campaign_hpc.py" --input-root "$DATA_V3/modeling_ready_t80" \
      --output-root "$OUT_V3/core_t80" --target target_rul80_steps_clipped --campaign-component full_matrix \
      --phases charge discharge --modes global lodo per_dataset --experiments $lista \
      --random-state $SEED_SPLIT --model-random-state $SEED_MODEL --elasticnet-cv \
      --no-save-predictions --skip-existing --n-jobs 1 --model-n-jobs "$MODEL_N_JOBS" \
      > "$LOGS/shard_$(printf '%02d' $s).log" 2>&1 &
  sleep 2
done
echo "lancado. Acompanhe: find $OUT_V3/core_t80 -name metrics_global.csv | wc -l   (alvo: 14 exps x 6 modelos x 15 escopos x 2 fases = 2520)"
