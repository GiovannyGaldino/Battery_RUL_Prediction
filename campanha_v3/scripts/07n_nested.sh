#!/usr/bin/env bash
# Braco N: selecao nested source-only. Gera 42 jobs e os roda com xargs -P.
# Uso: bash 07n_nested.sh [paralelismo=6]
set -euo pipefail; source "$(dirname "$0")/env.sh"
P="${1:-6}"; RUN="$PATCHED/run_modeling_final_campaign_hpc.py"
python "$NEW/nested_selection.py" plan --output-root "$OUT_V3/nested" --runner "$RUN" \
    --input-root "$DATA_V3/modeling_ready_t80" --experiments all --models random_forest \
    --random-state $SEED_SPLIT --model-random-state $SEED_MODEL --model-n-jobs "$MODEL_N_JOBS" \
    --jobs-file "$OUT_V3/nested_jobs.txt"
mkdir -p "$OUT_V3/logs_nested"
nohup bash -c "xargs -P $P -I{} bash -c '{} >> $OUT_V3/logs_nested/nested.log 2>&1' < $OUT_V3/nested_jobs.txt" > /dev/null 2>&1 &
echo "42 jobs lancados com paralelismo $P. Depois:"
echo "  python $NEW/nested_selection.py select --output-root $OUT_V3/nested --core-root $OUT_V3/core_t80"
