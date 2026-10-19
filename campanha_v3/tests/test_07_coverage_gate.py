#!/usr/bin/env python3
"""O portao de cobertura fecha quando uma familia chega vazia num dominio (o caso das curvas)."""
import sys, pathlib, importlib.util
import numpy as np, pandas as pd
RAIZ = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("cg", RAIZ/"new"/"coverage_gate.py")
cg = importlib.util.module_from_spec(spec); sys.modules["cg"] = cg; spec.loader.exec_module(cg)
rng = np.random.default_rng(0)
n = 300; ds = pd.Series(["MATR"]*100 + ["CALCE"]*100 + ["HUST"]*100)
X = pd.DataFrame({f"VQ_p003_{i:03d}": rng.normal(size=n) for i in range(10)} |
                 {f"voltage_mean_{i}": rng.normal(size=n) for i in range(6)} |
                 {"energy_sign_change_count": rng.integers(0, 3, n).astype(float)})
falhas = []
def check(c, m):
    print(("  ok   " if c else "  FALHA") + " " + m)
    if not c: falhas.append(m)
t = cg.audit(X, ds); check(not t["flag"].any(), "matriz completa passa")
check(cg.family("VQ_p003_001") == "vq" and cg.family("causal_capacity_x") == "causal", "familia = prefixo")
Xb = X.copy(); Xb.loc[ds != "MATR", [c for c in X.columns if c.startswith("VQ_")]] = np.nan
t = cg.audit(Xb, ds); bad = t[t["flag"]]
check(set(bad["dataset"]) == {"CALCE", "HUST"} and set(bad["family"]) == {"vq"}, "curvas vazias fora do MATR fecham o portao")
check(not t[(t["family"] == "voltage")]["flag"].any(), "familia preenchida nao e sinalizada")
Xc = X.copy(); Xc.loc[ds == "HUST", "energy_sign_change_count"] = np.nan
t = cg.audit(Xc, ds); check(not t["flag"].any(), "familia com menos de min_cols colunas nao fecha o portao")
Xd = X.copy(); Xd.loc[ds == "HUST", [c for c in X.columns if c.startswith("VQ_")][:4]] = np.nan
t = cg.audit(Xd, ds); check(not t["flag"].any(), "cobertura parcial (60%) acima do limiar passa")
print(f"\n{'TODOS OS TESTES PASSARAM' if not falhas else str(len(falhas))+' FALHA(S)'}"); sys.exit(1 if falhas else 0)
