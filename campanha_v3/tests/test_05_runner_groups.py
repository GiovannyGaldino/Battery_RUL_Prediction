#!/usr/bin/env python3
"""feature_group v3 contra as 366 colunas reais: stage_gap deixa de ser vazio; legacy reproduz v2."""
import sys, re, pathlib, pandas as pd
RAIZ = pathlib.Path(__file__).resolve().parent.parent
src = (RAIZ/"patched"/"run_modeling_final_campaign_hpc.py").read_text()
# extrai so feature_group + a flag, sem importar sklearn (indisponivel aqui)
ns = {}
blk = src[src.index("LEGACY_FEATURE_GROUPS = False"): src.index("def has_resolution(")]
exec(blk, ns)
cols = pd.read_csv(RAIZ/"tests"/"data"/"colunas_e_grupos.csv")
cols.columns = ["column", "group_v2"]
falhas = []
def check(c, m):
    print(("  ok   " if c else "  FALHA")+" "+m)
    if not c: falhas.append(m)
ns["LEGACY_FEATURE_GROUPS"] = True
leg = {c: ns["feature_group"](c) for c in cols.column}
ns["LEGACY_FEATURE_GROUPS"] = False
v3 = {c: ns["feature_group"](c) for c in cols.column}
dif = [(c, g, leg[c]) for c, g in zip(cols.column, cols.group_v2) if leg[c] != g]
check(len(dif) == 0, f"legacy reproduz grupos_de_features v2 nas 366 colunas (divergencias: {dif[:5]})")
sg = sorted(c for c in cols.column if v3[c] == "stage_gap")
print(f"  stage_gap v3 ({len(sg)}): {sg}")
check(len(sg) >= 10, "stage_gap deixa de ser vazio (>=10 colunas)")
mov = {c: (leg[c], v3[c]) for c in cols.column if leg[c] != v3[c]}
check(all(v == "stage_gap" for _, v in mov.values()), "as unicas mudancas sao entradas em stage_gap")
print("  movidas:", {k: v[0] for k, v in mov.items()})
print(f"\n{'TODOS OS TESTES PASSARAM' if not falhas else str(len(falhas))+' FALHA(S)'}"); sys.exit(1 if falhas else 0)
