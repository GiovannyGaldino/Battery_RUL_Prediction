#!/usr/bin/env python3
"""O pre-voo pega: familia vazia fora de um dominio, grupo vazio, experimentos duplicados."""
import sys, pathlib, importlib.util, tempfile, subprocess
import numpy as np, pandas as pd
RAIZ = pathlib.Path(__file__).resolve().parent.parent
try:
    import sklearn  # o runner importa sklearn
except ImportError:
    print("scikit-learn ausente: teste pulado (roda no HPC)"); sys.exit(0)
spec = importlib.util.spec_from_file_location("pf", RAIZ/"new"/"preflight.py")
pf = importlib.util.module_from_spec(spec); sys.modules["pf"] = pf; spec.loader.exec_module(pf)
r = pf.load_runner(RAIZ/"patched"/"run_modeling_final_campaign_hpc.py")
rng = np.random.default_rng(0)
ds = ["CALCE","HUST","MATR","NRW","OXFORD","PCOE","XJTU"]
meta = pd.DataFrame({"dataset": np.repeat(ds, 60), "battery_id": [f"{d}_{i%6}" for d in ds for i in range(60)]})
n = len(meta)
def mk(prefixes, k):
    return {f"{p}_{i:03d}": rng.normal(size=n) for p in prefixes for i in range(k)}
X = pd.DataFrame(mk(["VQ_p003","QV_p009"], 5) | mk(["capacity_mean","energy_total","temperature_max","voltage_mean",
        "current_mean","power_mean","time_step","stage_gap_dur","causal_capacity"], 3))
falhas = []
def check(c, m):
    print(("  ok   " if c else "  FALHA")+" "+m)
    if not c: falhas.append(m)
exps = r.expand_experiments(["all"])
tab, cs = pf.n_features_table(r, X, meta, exps, 0.95)
check(len(tab) == len(exps) * 15, "14 experimentos x 15 escopos")
check((tab["n_features"] > 0).all(), "matriz completa: features em todos os escopos")
Xb = X.copy(); Xb.loc[meta["dataset"] != "MATR", [c for c in X.columns if c.startswith(("VQ_","QV_"))]] = np.nan
tab, _ = pf.n_features_table(r, Xb, meta, exps, 0.95)
z = tab[(tab.experiment == "curves_only") & (tab.n_features == 0)]
check(len(z) == 7 and "heldout_MATR" in set(z.scope), "curvas so no MATR: curves_only sem features em 7 escopos (heldout_MATR + 6 por dominio)")
af = tab[(tab.experiment == "all_features")].set_index("scope")["n_features"]
check(af["heldout_MATR"] < 0.8 * af["all_datasets"], "curvas so no MATR: all_features cai abaixo de 80% do pooled em heldout_MATR")
check(cs["no_stage_gap"] != cs["all_features"], "no_stage_gap difere de all_features (grupo stage_gap nao vazio)")
print(f"\n{'TODOS OS TESTES PASSARAM' if not falhas else str(len(falhas))+' FALHA(S)'}"); sys.exit(1 if falhas else 0)
