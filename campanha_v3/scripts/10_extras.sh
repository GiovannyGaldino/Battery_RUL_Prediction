#!/usr/bin/env bash
# Extras citados no corpo: importancia por permutacao e curvas de aprendizado (matriz v3).
# Rodar depois do nucleo. Uso: bash 10_extras.sh   (usa MODEL_N_JOBS; 8 e um bom valor com a maquina livre)
set -euo pipefail; source "$(dirname "$0")/env.sh"
[ -d "$OUT_V3/core_t80" ] || { echo "ABORTADO: sem nucleo"; exit 1; }
cd "$NEW"
python permutation_importance.py --input-root "$DATA_V3/modeling_ready_t80" --out "$OUT_V3/extras/importance" --n-jobs "$MODEL_N_JOBS"
python learning_curves.py --input-root "$DATA_V3/modeling_ready_t80" --out "$OUT_V3/extras/learning_curves" --n-jobs "$MODEL_N_JOBS"
echo "extras prontos em $OUT_V3/extras"
