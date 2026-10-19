#!/usr/bin/env bash
# Mede UM ajuste do ElasticNetCV no maior escopo (carga, pooled, all_features) antes de
# relancar a campanha. Saida em $OUT_V3/timing_enet (fora do nucleo; pode ser apagada).
# Decisao: abaixo de ~15 min, relancar 06/07b; acima disso, reduzir l1_ratio ou alphas
# no patch_03 e registrar no CHANGELOG. Nao decidir por suposicao.
set -euo pipefail; source "$(dirname "$0")/env.sh"
OUT="$OUT_V3/timing_enet"; rm -rf "$OUT"; mkdir -p "$OUT"
echo "inicio: $(date '+%H:%M:%S') | threads: $MODEL_N_JOBS"
SECONDS=0
python "$PATCHED/run_modeling_final_campaign_hpc.py" --input-root "$DATA_V3/modeling_ready_t80" \
    --output-root "$OUT" --target target_rul80_steps_clipped --campaign-component timing \
    --phases charge --models elasticnet --modes global --experiments all_features \
    --random-state $SEED_SPLIT --model-random-state $SEED_MODEL --elasticnet-cv --no-save-predictions \
    --n-jobs 1 --model-n-jobs "$MODEL_N_JOBS" > "$OUT/timing.log" 2>&1
echo "fim: $(date '+%H:%M:%S') | duracao: $((SECONDS/60)) min $((SECONDS%60)) s"
python - <<PY
import pandas as pd, glob
f = glob.glob("$OUT/**/metrics_global.csv", recursive=True)[0]
d = pd.read_csv(f); print(d.iloc[0].to_string())
PY
echo "avisos de convergencia: $(grep -c ConvergenceWarning "$OUT/timing.log" || true)"
grep -c Traceback "$OUT/timing.log" | sed 's/^/tracebacks no log: /' || true
tail -3 "$OUT/timing.log" | cut -c1-160
