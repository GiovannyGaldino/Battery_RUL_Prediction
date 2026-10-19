#!/usr/bin/env bash
# Conferencia no meio da campanha: compara os ajustes ja gravados com o pre-voo.
# Rode depois de ~1 h e a qualquer momento. Se sinalizar, mate a campanha e investigue.
set -euo pipefail; source "$(dirname "$0")/env.sh"
python - <<PY
import pandas as pd, glob, sys, subprocess
raiz = "$OUT_V3/core_t80"; flags = []
fs = glob.glob(f"{raiz}/*/*/*/*/*/metrics_global.csv")
if not fs: print("nenhum ajuste gravado ainda"); sys.exit(0)
d = pd.concat(pd.read_csv(f) for f in fs)
print(f"ajustes gravados: {len(d)} | experimentos: {d.experiment.nunique()} | modelos: {sorted(d.model.unique())}")
# 1. n_features de cada ajuste contra o pre-voo (mesma fase/experimento/escopo)
pf = pd.concat(pd.read_csv(f"$OUT_V3/gate/preflight_t80_{p}.csv").assign(phase=p) for p in ["charge","discharge"])
m = d.merge(pf[["phase","experiment","scope","n_features","n_after_missing"]], on=["phase","experiment","scope"], suffixes=("","_prevoo"))
m["dif"] = (m.n_features - m.n_features_prevoo).abs()
ruim = m[m.dif > 3]   # constantes no subconjunto de 80% explicam diferencas de 1 a 3
print(f"n_features vs pre-voo: {len(m)} comparados, {len(ruim)} com diferenca > 3")
if len(ruim): flags.append("n_features divergente do pre-voo"); print(ruim[["phase","mode","scope","experiment","model","n_features","n_features_prevoo"]].head(15).to_string(index=False))
# 2. curves_only fora do MATR
co = d[(d.experiment=="curves_only") & (d["mode"]!="global")]
print(f"curves_only fora do pooled: {len(co)} ajustes em {sorted(co.scope.unique())}")
if len(co) and not any(s != "MATR" and s != "heldout_MATR" for s in co.scope.unique()): flags.append("curves_only so em escopos com MATR")
# 3. previsoes negativas e NaN
neg = d[d.share_pred_negative > 0.05][["phase","mode","scope","experiment","model","share_pred_negative"]]
print(f"ajustes com mais de 5% de previsoes negativas: {len(neg)} (esperado: alguns do elasticnet e mlp)")
if d.nMAE.isna().any(): flags.append("nMAE NaN em algum ajuste")
# 4. pooled all_features RF contra o teste rapido
q = d[(d["mode"]=="global") & (d.experiment=="all_features") & (d.model=="random_forest")]
for _, s in q.iterrows():
    ref = {"charge": 0.1115, "discharge": 0.1175}[s.phase]
    print(f"pooled RF all_features {s.phase}: nMAE {s.nMAE:.4f} (teste rapido {ref})")
    if abs(s.nMAE - ref) > 1e-3: flags.append(f"pooled RF {s.phase} difere do teste rapido")
print("\\nSINALIZACOES:" if flags else "\\nsem sinalizacoes ate aqui"); [print("  -", f) for f in flags]
PY
erros=$(cat "$OUT_V3"/logs_core/*.log "$OUT_V3"/logs_t85/*.log 2>/dev/null | grep -c "\[ERROR\]\|Traceback" || true)
echo "linhas [ERROR]/Traceback nos logs: $erros"
