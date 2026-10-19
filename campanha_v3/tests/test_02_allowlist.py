#!/usr/bin/env python3
"""A lista de inclusao contra as 366 colunas REAIS da matriz v2 (colunas_e_grupos.csv)."""
import sys, pathlib, importlib.util, pandas as pd
RAIZ = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("mb", RAIZ/"patched"/"build_modeling_from_features_v3.py")
mb = importlib.util.module_from_spec(spec); sys.modules["mb"] = mb; spec.loader.exec_module(mb)
cols = pd.read_csv(RAIZ/"tests"/"data"/"colunas_e_grupos.csv").iloc[:,0].astype(str).tolist()
T = "target_rul80_steps_clipped"
VAZ = ["capacity_trend_slope_per_step","capacity_trend_spearman","capacity_positive_step_rate",
       "capacity_negative_step_rate","capacity_median_abs_step_ah","source_row_id"]
falhas = []
def check(c, m):
    print(("  ok   " if c else "  FALHA")+" "+m)
    if not c: falhas.append(m)
print(f"== {len(cols)} colunas reais da matriz v2 ==")
leg = {c: mb.should_remove_column(c, T, "legacy_denylist") for c in cols}
alw = {c: mb.should_remove_column(c, T, "allowlist") for c in cols}
for v in VAZ:
    if v in cols:
        check(leg[v][0] is False or v=="source_row_id", f"legacy deixa passar {v}: {leg[v]}")
        check(alw[v][0] is True, f"allowlist REJEITA {v}: {alw[v]}")
k_leg = [c for c in cols if not leg[c][0]]; k_alw = [c for c in cols if not alw[c][0]]
print(f"\n  legacy mantem {len(k_leg)} | allowlist mantem {len(k_alw)}")
perd = sorted(set(k_leg)-set(k_alw)); ganh = sorted(set(k_alw)-set(k_leg))
print(f"  rejeitadas so pela allowlist ({len(perd)}):")
for c in perd: print(f"     - {c:<58} {alw[c][1]}")
print(f"  aceitas so pela allowlist ({len(ganh)}): {ganh}")
check(set(perd) >= set(v for v in VAZ if v in cols), "todas as seis contaminadas estao entre as rejeitadas")
# nada intra-ciclo legitimo foi perdido por engano: as perdas devem ser so as seis
extra = [c for c in perd if c not in VAZ and c != "__index_level_0__"]  # indice do pandas, nao e feature
check(len(extra) == 0, f"nenhuma coluna legitima perdida (extras: {extra})")
print(f"\n{'TODOS OS TESTES PASSARAM' if not falhas else str(len(falhas))+' FALHA(S)'}"); sys.exit(1 if falhas else 0)
