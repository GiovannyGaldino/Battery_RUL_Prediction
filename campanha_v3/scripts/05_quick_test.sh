#!/usr/bin/env bash
# PORTAO 3: 32 ajustes (RF, 2 fases, pooled + 7 LODO, dois experimentos).
#   no_causal_trend  = tendencia removida  -> reproduz o diagnostico de vazamento:
#                      pooled nMAE ~0.112 (carga) e ~0.118 (descarga)
#   all_features     = tendencia causal    -> deve ficar ENTRE 0.104 (v2, com
#                      vazamento) e o valor de no_causal_trend. Abaixo de 0.104 e
#                      vazamento novo; acima de no_causal_trend, as causais atrapalham.
set -euo pipefail; source "$(dirname "$0")/env.sh"
[ -f "$OUT_V3/smoke/logs/shard_00.log" ] || { echo "ABORTADO: rode 04_gate.sh, 04b_preflight.sh e 05a_smoke.sh antes"; exit 1; }
python "$PATCHED/run_modeling_final_campaign_hpc.py" --input-root "$DATA_V3/modeling_ready_t80" \
    --output-root "$OUT_V3/quick_test" --target target_rul80_steps_clipped --campaign-component quick_test \
    --phases charge discharge --models random_forest --modes global lodo --experiments all_features no_causal_trend \
    --random-state $SEED_SPLIT --model-random-state $SEED_MODEL --no-save-predictions --skip-existing \
    --n-jobs 1 --model-n-jobs "$MODEL_N_JOBS"
python - <<PY
import pandas as pd, glob
d = pd.concat(pd.read_csv(f) for f in glob.glob("$OUT_V3/quick_test/**/metrics_global.csv", recursive=True))
g = d[d["mode"]=="global"][["phase","experiment","nMAE","nMAE_clip0","R2","n_features"]].sort_values(["phase","experiment"])
print(g.to_string(index=False))
print("\nreferencias (matriz com curvas em todos os dominios): v2 com vazamento 0.104 / 0.102")
print("esperado: no_causal_trend acima de v2 (sem o vazamento); all_features entre v2 e no_causal_trend\n")
nf = d[d["mode"]=="global"].groupby("experiment")["n_features"].first()
print("n_features pooled:", nf.to_dict(), " (all_features deve ficar perto de 341)\n")
l = d[d["mode"]=="lodo"].groupby(["phase","experiment"])["nMAE"].median().rename("LODO nMAE mediana")
print(l.to_string()); print("\nreferencia LODO mediana v2 (com vazamento): 0.900 / 0.678 (RF, all_features)")
PY
