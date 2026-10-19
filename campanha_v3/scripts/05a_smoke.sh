#!/usr/bin/env bash
# FUMACA: o runner de verdade, todos os experimentos x modos x fases, so com o modelo
# `median` (custo zero de treino). Exercita carga, selecao, filtro, pesos, particoes e
# escrita. Alvo: 14 exps x 15 escopos x 2 fases = 420 metrics_global.csv e zero [ERROR].
# Minutos com 14 shards. Uso: bash 05a_smoke.sh [n_shards]
set -euo pipefail; source "$(dirname "$0")/env.sh"
[ -f "$OUT_V3/gate/preflight_t80_charge.csv" ] || { echo "ABORTADO: rode 04b_preflight.sh antes"; exit 1; }
N="${1:-14}"; OUT="$OUT_V3/smoke"; rm -rf "$OUT"; mkdir -p "$OUT/logs"
EXPS=(all_features no_curves global_stats_only curves_only capacity_only energy_added no_temperature \
      no_time_sampling no_stage_gap resolution_p003 resolution_p009 resolution_p015 resolution_p030 no_causal_trend)
for ((s=0; s<N; s++)); do
  lista=""; for ((i=s; i<${#EXPS[@]}; i+=N)); do lista="$lista ${EXPS[$i]}"; done; [ -z "$lista" ] && continue
  python "$PATCHED/run_modeling_final_campaign_hpc.py" --input-root "$DATA_V3/modeling_ready_t80" \
      --output-root "$OUT" --target target_rul80_steps_clipped --campaign-component smoke \
      --phases charge discharge --modes global lodo per_dataset --experiments $lista --models median \
      --random-state $SEED_SPLIT --model-random-state $SEED_MODEL --no-save-predictions \
      --n-jobs 1 --model-n-jobs 1 > "$OUT/logs/shard_$(printf '%02d' $s).log" 2>&1 &
done
wait
feitos=$(find "$OUT" -name metrics_global.csv | wc -l); erros=$(cat "$OUT"/logs/*.log | grep -c "\[ERROR\]" || true)
echo "fumaca: $feitos de 420 casos escritos; $erros linhas [ERROR]"
grep -h "\[ERROR\]" "$OUT"/logs/*.log | head -20 || true
python - <<PY
import pandas as pd, glob
d = pd.concat(pd.read_csv(f) for f in glob.glob("$OUT/**/metrics_global.csv", recursive=True))
piv = d[d.phase=="charge"].pivot_table(index="experiment", columns="scope", values="n_features", aggfunc="first")
cols = ["all_datasets"] + [c for c in piv.columns if c.startswith("heldout_")] + [c for c in piv.columns if not c.startswith("heldout_") and c != "all_datasets"]
print("\nn_features (carga) por experimento x escopo, medido pelo runner:"); print(piv[cols].to_string())
p = pd.read_csv("$OUT_V3/gate/preflight_t80_charge.csv")
m = d[d.phase=="charge"].merge(p, on=["experiment","scope"], suffixes=("_runner","_prevoo"))
dif = m[m.n_features_runner != m.n_features_prevoo]
print(f"\ncasos em que o runner e o pre-voo divergem em n_features: {len(dif)} (pooled e por dominio usam 80% das celulas no runner; divergencias pequenas ali sao esperadas)")
if len(dif): print(dif[["experiment","scope","n_features_runner","n_features_prevoo"]].head(20).to_string(index=False))
PY
[ "$feitos" -eq 420 ] && [ "$erros" -eq 0 ] && echo "FUMACA OK" || { echo "FUMACA FALHOU"; exit 1; }
