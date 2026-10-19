#!/usr/bin/env python3
"""As features causais NUNCA usam o futuro; o re-join preserva linhas e renomeia o escopo de celula."""
import sys, pathlib, importlib.util, tempfile
import numpy as np, pandas as pd
RAIZ = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rj", RAIZ/"new"/"rejoin_targets.py")
rj = importlib.util.module_from_spec(spec); sys.modules["rj"] = rj; spec.loader.exec_module(rj)
falhas = []
def check(c, m):
    print(("  ok   " if c else "  FALHA")+" "+m)
    if not c: falhas.append(m)

print("== causalidade: prefixo k nao depende de cap[k+1:] ==")
rng = np.random.default_rng(0)
cap = 1.0 - 0.0005*np.arange(300) + rng.normal(0, 0.002, 300)
f_all = rj.causal_trend_features(cap)
for k in [3, 10, 57, 150, 299]:
    f_pre = rj.causal_trend_features(cap[:k+1])
    for nome in ["slope", "pearson", "pos_rate", "neg_rate", "med_abs_step"]:
        a, b = f_all[nome][k], f_pre[nome][k]
        check(np.isclose(a, b, equal_nan=True), f"{nome}[{k}] igual com ou sem o futuro ({a:.6g} vs {b:.6g})")
# mudar o futuro nao muda o passado
cap2 = cap.copy(); cap2[200:] += 5.0
f2 = rj.causal_trend_features(cap2)
check(np.allclose(f_all["slope"][:200], f2["slope"][:200], equal_nan=True), "alterar cap[200:] nao muda slope[:200]")
check(not np.allclose(f_all["slope"][200:], f2["slope"][200:], equal_nan=True), "mas muda slope[200:] (sanidade)")
check(np.all(np.isnan(f_all["slope"][:2])) and np.isfinite(f_all["slope"][2]), "NaN antes de min_history=3")
# forma fechada = polyfit no prefixo completo
s_poly = np.polyfit(np.arange(300), cap, 1)[0]
check(np.isclose(f_all["slope"][-1], s_poly, rtol=1e-8), f"slope final = polyfit ({f_all['slope'][-1]:.6g} vs {s_poly:.6g})")
d = np.diff(cap); check(np.isclose(f_all["pos_rate"][-1], (d>0).mean()), "pos_rate final = fracao de passos positivos")
check(np.isclose(f_all["med_abs_step"][-1], np.median(np.abs(d))), "med_abs_step final = mediana global")
# NaN no meio nao quebra
cap3 = cap.copy(); cap3[50:55] = np.nan
f3 = rj.causal_trend_features(cap3)
check(np.isfinite(f3["slope"][100]), "NaN no meio: slope continua definido depois")

print("\n== re-join em disco ==")
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    n = 40
    base = pd.DataFrame({"source_file_key": "syn/A.parquet", "source_row_id": np.arange(n),
                         "dataset": "SYN", "battery_id": "A", "phase_folder": "charge", "cycle_index": np.arange(n)})
    feat1 = base.assign(feature_level="phase", voltage_mean=rng.normal(3.7, .01, n),
                        capacity_trend_slope_per_step=-0.001,           # escopo de celula (v2)
                        target_rul80_steps_clipped=np.arange(n)[::-1].astype(float),   # alvo VELHO
                        capacity_trend_spearman=-0.99)
    feat2 = feat1.assign(feature_level="stage_1")                     # segundo nivel: mesma chave
    feat = pd.concat([feat1, feat2], ignore_index=True)
    # simula o defeito v2: source_row_id gravado como int16 com volta (inutil para juncao)
    feat["source_row_id"] = ((feat["source_row_id"] + 32760) % 65536 - 32768).astype("int16")
    (td/"charge").mkdir(); feat.to_parquet(td/"charge"/"A.features.parquet", index=False)
    capv = np.linspace(1.0, 0.6, n)
    tg = base.assign(target_capacity_ah=capv, target_rul80_steps_clipped=np.arange(n)[::-1].astype(float)*2,
                     target_rul85_steps_clipped=np.arange(n)[::-1].astype(float)*2-5,
                     capacity_trend_slope_per_step=-0.0103, capacity_trend_spearman=-1.0,
                     capacity_positive_step_rate=0.0, capacity_negative_step_rate=1.0, capacity_median_abs_step_ah=0.0103)
    out = td/"out"
    # segundo dataset na tabela de alvos, para checar que o escopo da unicidade e por dataset/fase
    tg2 = tg.assign(dataset="OUTRO", source_file_key="syn/B.parquet")
    tgp = rj.prepare_targets(pd.concat([tg, tg2], ignore_index=True), 3)
    r = rj.rejoin_one(td/"charge"/"A.features.parquet", tgp, out/"charge"/"A.features.parquet")
    m = pd.read_parquet(out/"charge"/"A.features.parquet")
    check(r["rows_in"] == r["rows_out"] == 2*n and r["source_rows"] == n, f"linhas preservadas com 2 niveis por registro ({r})")
    ph = m[m.feature_level=="phase"].sort_values("cycle_index"); st = m[m.feature_level=="stage_1"].sort_values("cycle_index")
    check(np.allclose(ph.causal_capacity_trend_slope_per_step.to_numpy(), st.causal_capacity_trend_slope_per_step.to_numpy(), equal_nan=True), "causal_ identica nos dois niveis do mesmo registro")
    check("cell_capacity_trend_slope_per_step" in m and "capacity_trend_slope_per_step" not in m, "escopo de celula renomeado para cell_")
    check("causal_capacity_trend_slope_per_step" in m and m["causal_capacity_trend_slope_per_step"].iloc[-1] < 0, "causal_ adicionada e negativa (degradacao)")
    check((ph["target_rul80_steps_clipped"].to_numpy() == np.arange(n)[::-1]*2).all(), "alvo VELHO substituido pelo NOVO")
    check("target_rul85_steps_clipped" in m, "alvo secundario presente")
    check("voltage_mean" in m, "feature intra-ciclo preservada")
    check((ph["source_row_id"].to_numpy() == np.arange(n)).all() and str(ph["source_row_id"].dtype) == "int64", "source_row_id corrompido (int16) substituido pelo correto dos alvos")
    # linha sem alvo -> aborta
    try:
        sem = tgp[~((tgp.dataset == "SYN") & (tgp.cycle_index == 5))]      # tira UM ciclo do SYN
        rj.rejoin_one(td/"charge"/"A.features.parquet", sem, out/"x.parquet"); check(False, "deveria abortar com linha sem alvo")
    except RuntimeError as e:
        check("sem alvo" in str(e), f"aborta com linha sem alvo: {e}")
print(f"\n{'TODOS OS TESTES PASSARAM' if not falhas else str(len(falhas))+' FALHA(S)'}"); sys.exit(1 if falhas else 0)
