#!/usr/bin/env python3
"""Agregador: das saidas do runner (metrics_*.csv) as tabelas do corpo do manuscrito.

Entradas (todas arvores escritas por run_modeling_final_campaign_hpc.py):
  --core      nucleo (core_t80): 14 experimentos x 6 modelos x {global, lodo, per_dataset} x 2 fases
  --repeated  braco R (repeated/split_s*): all_features, 6 modelos, {global, per_dataset}, N sementes  [opcional]
  --nested    CSV de nested_selection.py select (regimes source-only)                                [opcional]
Saidas em --out:
  T3_hierarchy.csv, T4_selection.csv, T5_paired.csv, T6_per_domain.csv,
  S_ablations.csv, S_models.csv, S_negative_transfer.csv, S_within_partitions.csv,
  fig3_data.csv, resumo.json, README_tabelas.md
Configuracao pre-especificada: --model random_forest --experiment all_features.
Piso ingenuo: modelo `median` (mediana ponderada do treino) no mesmo escopo.
Transferencia negativa: nMAE do modelo acima do nMAE do `median` no mesmo escopo LODO.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

PHASES = ["charge", "discharge"]


def load_tree(root: Path, name: str = "metrics_global.csv") -> pd.DataFrame:
    fs = sorted(root.rglob(name))
    if not fs:
        raise SystemExit(f"nenhum {name} em {root}")
    d = pd.concat((pd.read_csv(f).assign(_file=str(f)) for f in fs), ignore_index=True)
    return d


def naive_lodo(core: pd.DataFrame) -> pd.DataFrame:
    """nMAE do preditor `median` por (phase, scope) LODO; independe do experimento (usa all_features)."""
    m = core[(core["mode"] == "lodo") & (core.model == "median") & (core.experiment == "all_features")]
    return m.set_index(["phase", "scope"])[["nMAE", "R2"]].rename(columns={"nMAE": "naive_nMAE", "R2": "naive_R2"})


def pre_specified(core: pd.DataFrame, model: str, experiment: str) -> pd.DataFrame:
    return core[(core.model == model) & (core.experiment == experiment)].copy()


# ------------------------------------------------------------------ T3 e T6
def within_domain(core: pd.DataFrame, repeated: pd.DataFrame | None, model: str, experiment: str):
    """Por dominio: media (e desvio) de nMAE e R2 sobre as particoes repetidas; sem braco R, 1 particao."""
    r = None
    if repeated is not None and len(repeated):
        r = repeated[(repeated["mode"] == "per_dataset") & (repeated.model == model) & (repeated.experiment == experiment)]
        n_scopes = core[(core["mode"] == "per_dataset") & (core.model == model) & (core.experiment == experiment)].groupby(["phase", "scope"]).ngroups
        ok = len(r) and r.groupby(["phase", "scope"]).ngroups == n_scopes and r.groupby(["phase", "scope"]).random_state.nunique().min() >= 2
        if not ok:
            print(f"[AVISO] braco R incompleto para {model}/{experiment} ({len(r)} ajustes intra-dominio): usando 1 particao do nucleo")
            r = None
    if r is not None:
        g = r.groupby(["phase", "scope"]).agg(nMAE=("nMAE", "mean"), nMAE_sd=("nMAE", "std"), R2=("R2", "mean"),
                                              R2_sd=("R2", "std"), n_partitions=("random_state", "nunique")).reset_index()
        src = f"{int(g.n_partitions.min())} a {int(g.n_partitions.max())} particoes (braco R)" if g.n_partitions.min() != g.n_partitions.max() else f"{int(g.n_partitions.max())} particoes (braco R)"
    else:
        p = pre_specified(core, model, experiment); p = p[p["mode"] == "per_dataset"]
        g = p.groupby(["phase", "scope"]).agg(nMAE=("nMAE", "first"), R2=("R2", "first")).reset_index()
        g["nMAE_sd"] = np.nan; g["R2_sd"] = np.nan; g["n_partitions"] = 1
        src = "1 particao (nucleo; braco R ausente)"
    return g.rename(columns={"scope": "dataset"}), src


def pooled_repeated(repeated, model, experiment):
    """Pooled sobre as particoes do braco R: ponderado (metrics_global) e macro (metrics_by_dataset), media e DP entre sementes."""
    if repeated is None or not len(repeated):
        return None
    r = repeated[(repeated["mode"] == "global") & (repeated.model == model) & (repeated.experiment == experiment)]
    if r.groupby("phase").random_state.nunique().min() < 2:
        return None
    rows = []
    for _, f in r.iterrows():
        bd = pd.read_csv(Path(f._file).with_name("metrics_by_dataset.csv"))
        rows.append({"phase": f.phase, "random_state": f.random_state, "weighted_nMAE": f.nMAE, "weighted_R2": f.R2,
                     "macro_nMAE": bd.nMAE.mean(), "macro_R2": bd.R2.mean(), "macro_nMAE_median_domain": bd.nMAE.median(), "n_domains": len(bd)})
    d = pd.DataFrame(rows)
    return d.groupby("phase").agg(n_partitions=("random_state", "nunique"), weighted_nMAE=("weighted_nMAE", "mean"), weighted_nMAE_sd=("weighted_nMAE", "std"),
                                  weighted_R2=("weighted_R2", "mean"), macro_nMAE=("macro_nMAE", "mean"), macro_nMAE_sd=("macro_nMAE", "std"),
                                  macro_R2=("macro_R2", "mean"), macro_R2_sd=("macro_R2", "std"))


def distinct_test_cells(repeated, model, experiment):
    """Por dominio e fase: celulas de teste distintas ao longo das particoes intra-dominio do braco R (sensibilidade a particao, nao IC)."""
    if repeated is None or not len(repeated):
        return None
    r = repeated[(repeated["mode"] == "per_dataset") & (repeated.model == model) & (repeated.experiment == experiment)]
    rows = []
    for (ph, sc), g in r.groupby(["phase", "scope"]):
        cells = set()
        for f in g._file:
            bb = pd.read_csv(Path(f).with_name("metrics_by_battery.csv")); cells |= set(bb.battery_id)
        rows.append({"phase": ph, "dataset": sc, "n_partitions": g.random_state.nunique(), "n_distinct_test_cells": len(cells),
                     "n_test_cells_per_partition": int(g.n_test_batteries.iloc[0]), "n_cells_domain": int(g.n_test_batteries.iloc[0] + g.n_train_batteries.iloc[0])})
    return pd.DataFrame(rows)


def t3_hierarchy(core, repeated, core_root, model, experiment):
    p = pre_specified(core, model, experiment)
    pr = pooled_repeated(repeated, model, experiment)
    rows = []
    for ph in PHASES:
        g = p[(p.phase == ph) & (p["mode"] == "global")].iloc[0]
        # macro do pooled: metrics_by_dataset do mesmo ajuste
        bd = pd.read_csv(Path(g._file).with_name("metrics_by_dataset.csv"))
        wd, src = within_domain(core, repeated, model, experiment)
        wd = wd[wd.phase == ph]
        lo = p[(p.phase == ph) & (p["mode"] == "lodo")]
        nv = core[(core.phase == ph) & (core["mode"] == "lodo") & (core.model == "median") & (core.experiment == experiment)]
        rows += [
            {"phase": ph, "protocol": "Pooled (all domains in training)", "summary": "Sample-weighted", "nMAE": g.nMAE, "R2": g.R2, "n": int(g.n_test_batteries)},
            {"phase": ph, "protocol": "Pooled (all domains in training)", "summary": "Macro-average over 7 domains", "nMAE": bd.nMAE.mean(), "R2": bd.R2.mean(), "n": len(bd)},]
        if pr is not None:
            q = pr.loc[ph]
            rows += [
                {"phase": ph, "protocol": f"Pooled, {int(q.n_partitions)} partitions", "summary": "Sample-weighted, mean (sd)", "nMAE": q.weighted_nMAE, "R2": q.weighted_R2, "n": int(q.n_partitions), "nMAE_sd": q.weighted_nMAE_sd},
                {"phase": ph, "protocol": f"Pooled, {int(q.n_partitions)} partitions", "summary": "Macro-average, mean (sd)", "nMAE": q.macro_nMAE, "R2": q.macro_R2, "n": int(q.n_partitions), "nMAE_sd": q.macro_nMAE_sd},]
        rows += [
            {"phase": ph, "protocol": f"Within-domain ({src})", "summary": "Macro-average over 7 domains", "nMAE": wd.nMAE.mean(), "R2": wd.R2.mean(), "n": len(wd)},
            {"phase": ph, "protocol": "Cross-domain, LODO", "summary": "Macro-average over 7 domains", "nMAE": lo.nMAE.mean(), "R2": lo.R2.mean(), "n": len(lo)},
            {"phase": ph, "protocol": "Cross-domain, LODO", "summary": "Median over 7 domains", "nMAE": lo.nMAE.median(), "R2": lo.R2.median(), "n": len(lo)},
            {"phase": ph, "protocol": "Naive median predictor, LODO", "summary": "Median over 7 domains", "nMAE": nv.nMAE.median(), "R2": nv.R2.median(), "n": len(nv)},
        ]
    t = pd.DataFrame(rows)
    wide = t.pivot_table(index=["protocol", "summary"], columns="phase", values=["nMAE", "R2"], aggfunc="first", sort=False)
    wide.columns = [f"{ph}_{m}" for m, ph in wide.columns]
    order = [r for r in dict.fromkeys(zip(t.protocol, t.summary))]
    wide = wide.loc[order][["charge_nMAE", "charge_R2", "discharge_nMAE", "discharge_R2"]].reset_index()
    return wide, t


def t6_per_domain(core, repeated, model, experiment):
    p = pre_specified(core, model, experiment)
    wd, src = within_domain(core, repeated, model, experiment)
    lo = p[p["mode"] == "lodo"].copy(); lo["dataset"] = lo.scope.str.replace("heldout_", "", regex=False)
    nv = naive_lodo(core).reset_index(); nv["dataset"] = nv.scope.str.replace("heldout_", "", regex=False)
    t = wd.merge(lo[["phase", "dataset", "nMAE", "R2", "MAE", "y_mean", "n_test_batteries"]].rename(
        columns={"nMAE": "lodo_nMAE", "R2": "lodo_R2", "MAE": "lodo_MAE_cycles", "y_mean": "lodo_rul_mean_cycles", "n_test_batteries": "n_cells"}), on=["phase", "dataset"])
    t = t.merge(nv[["phase", "dataset", "naive_nMAE"]], on=["phase", "dataset"])
    t["ratio_lodo_within"] = t.lodo_nMAE / t.nMAE
    t["lodo_over_naive"] = t.lodo_nMAE / t.naive_nMAE
    t["negative_transfer"] = t.lodo_over_naive > 1
    t = t.rename(columns={"nMAE": "within_nMAE", "R2": "within_R2", "nMAE_sd": "within_nMAE_sd", "R2_sd": "within_R2_sd"})
    t["within_source"] = src
    return t.sort_values(["phase", "dataset"]).reset_index(drop=True)


# ------------------------------------------------------------------ T4
def t4_selection(core, nested: pd.DataFrame | None, model, experiment, space: str):
    lo = core[core["mode"] == "lodo"].copy()
    if space == "experiments":
        cand = lo[lo.model == model]
    else:
        cand = lo[lo.model != "median"]
    nv = naive_lodo(core)
    scopes = sorted(set(zip(lo.phase, lo.scope)))
    oracle = cand.groupby(["phase", "scope"]).nMAE.min()
    regimes = {}
    ps = pre_specified(core, model, experiment).set_index(["phase", "scope"]).nMAE
    regimes["Pre-specified (%s, %s)" % (model, experiment)] = ps
    if nested is not None and len(nested):
        nn = nested[(nested.model == model) & (nested.criterion == "mean")]
        complete = len(nn) == len(scopes) and (nn.inner_folds >= 6).all() and nn.outer_found.all()
        if not complete:
            print(f"[AVISO] braco N incompleto ({len(nn)} de {len(scopes)} escopos com 6 dobras internas): linhas source-only omitidas")
            nested = None
    if nested is not None and len(nested):
        for crit in ["mean", "median"]:
            s = nested[(nested.criterion == crit) & (nested.model == model)].copy()
            s["scope"] = "heldout_" + s.target
            regimes[f"Source-only, {crit} over the six remaining domains"] = s.set_index(["phase", "scope"]).outer_lodo_nMAE
    rows = [{"regime": "Target-supervised", "target_info": "Full (best configuration on the held-out domain)",
             "median_optimism_pct": 0.0, "negative_transfer_scopes": int(sum(oracle[s] > nv.naive_nMAE[s] for s in scopes)),
             "n_scopes": len(scopes), "candidate_space": space, "n_candidates": int(cand.groupby(["phase", "scope"]).size().max())}]
    for name, ser in regimes.items():
        opt, neg, n = [], 0, 0
        for s in scopes:
            if s in ser.index and not np.isnan(ser[s]):
                opt.append((ser[s] - oracle[s]) / oracle[s] * 100); neg += int(ser[s] > nv.naive_nMAE[s]); n += 1
        rows.append({"regime": name, "target_info": "None", "median_optimism_pct": float(np.median(opt)) if opt else np.nan,
                     "negative_transfer_scopes": neg, "n_scopes": n, "candidate_space": space, "n_candidates": rows[0]["n_candidates"]})
    per_scope = pd.DataFrame({"oracle_nMAE": oracle, "naive_nMAE": nv.naive_nMAE, **{k: v for k, v in regimes.items()}})
    per_scope = per_scope.loc[[s for s in scopes if s in per_scope.index]]   # so escopos LODO
    per_scope["oracle_config"] = cand.loc[cand.groupby(["phase", "scope"]).nMAE.idxmin()].set_index(["phase", "scope"]).apply(lambda r: f"{r.experiment}/{r.model}", axis=1)
    return pd.DataFrame(rows), per_scope.reset_index()


# ------------------------------------------------------------------ T5
def t5_paired(core, model, experiment, n_boot=10000, seed=0):
    from scipy import stats
    p = pre_specified(core, model, experiment); p = p[p["mode"] == "lodo"]
    bb = pd.concat(pd.read_csv(Path(f).with_name("metrics_by_battery.csv")) for f in p._file)
    bb = bb[bb.scope.str.replace("heldout_", "", regex=False) == bb.dataset]   # cada celula no seu heldout
    w = bb.pivot_table(index=["dataset", "battery_id"], columns="phase", values="nMAE", aggfunc="first").dropna()
    w["diff"] = w["charge"] - w["discharge"]    # positivo favorece a descarga
    rows, pvals = [], []
    for ds, g in w.groupby(level="dataset"):
        d = g["diff"].to_numpy()
        pv = stats.wilcoxon(d, alternative="two-sided").pvalue if len(d) >= 2 and np.any(d != 0) else np.nan
        rows.append({"dataset": ds, "cells_in_both": len(d), "median_diff_charge_minus_discharge": float(np.median(d)), "wilcoxon_p": pv})
        pvals.append(pv)
    t = pd.DataFrame(rows)
    # Holm
    order = np.argsort([1 if np.isnan(x) else x for x in pvals]); m = len(pvals); adj = [np.nan] * m; running = 0
    for rank, i in enumerate(order):
        if np.isnan(pvals[i]): continue
        running = max(running, (m - rank) * pvals[i]); adj[i] = min(1.0, running)
    t["holm_p"] = adj
    t["favored_phase"] = np.where(t.holm_p < 0.05, np.where(t.median_diff_charge_minus_discharge > 0, "Discharge", "Charge"), "Not significant")
    d = w["diff"].to_numpy(); rng = np.random.default_rng(seed)
    boots = np.array([np.median(rng.choice(d, len(d), replace=True)) for _ in range(n_boot)])
    pv_all = stats.wilcoxon(d, alternative="two-sided").pvalue
    dom_med = t.median_diff_charge_minus_discharge.to_numpy()
    sign_p = stats.binomtest(int((dom_med > 0).sum()), int((dom_med != 0).sum()), 0.5).pvalue if (dom_med != 0).sum() else np.nan
    t.loc[len(t)] = {"dataset": "All domains (cell-weighted)", "cells_in_both": len(d), "median_diff_charge_minus_discharge": float(np.median(d)),
                     "wilcoxon_p": pv_all, "holm_p": np.nan, "favored_phase": f"95% bootstrap CI [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]"}
    t.loc[len(t)] = {"dataset": "All domains (domain-equal)", "cells_in_both": len(dom_med), "median_diff_charge_minus_discharge": float(np.median(dom_med)),
                     "wilcoxon_p": sign_p, "holm_p": np.nan, "favored_phase": "sign test over domain medians"}
    return t, w.reset_index()


# ------------------------------------------------------------------ suplementares
def s_ablations(core, model):
    lo = core[core["mode"] == "lodo"]; nv = naive_lodo(core)
    rows = []
    for e, g in lo.groupby("experiment"):
        rf = g[g.model == model]; pooled = core[(core["mode"] == "global") & (core.experiment == e) & (core.model == model)]
        rows.append({"experiment": e, "n_features_pooled_charge": int(pooled[pooled.phase == "charge"].n_features.iloc[0]) if len(pooled) else np.nan,
                     "lodo_median_nMAE_all_models": g.nMAE.median(), "lodo_median_nMAE_%s" % model: rf.nMAE.median(),
                     "lodo_scopes_evaluated_%s" % model: len(rf),
                     "negative_transfer_scopes_%s" % model: int(sum(r.nMAE > nv.naive_nMAE[(r.phase, r.scope)] for _, r in rf.iterrows())),
                     "pooled_nMAE_charge_%s" % model: pooled[pooled.phase == "charge"].nMAE.iloc[0] if len(pooled) else np.nan,
                     "pooled_nMAE_discharge_%s" % model: pooled[pooled.phase == "discharge"].nMAE.iloc[0] if len(pooled) else np.nan})
    return pd.DataFrame(rows).sort_values("lodo_median_nMAE_%s" % model).reset_index(drop=True)


def s_ablations_by_phase(core, model):
    """Por experimento e fase: nMAE pooled e mediana LODO do estimador pre-especificado (dados da Figura 5)."""
    rows = []
    for (e, ph), g in core[core.model == model].groupby(["experiment", "phase"]):
        rows.append({"experiment": e, "phase": ph, "pooled_nMAE": g[g["mode"] == "global"].nMAE.iloc[0],
                     "lodo_median_nMAE": g[g["mode"] == "lodo"].nMAE.median(), "lodo_macro_nMAE": g[g["mode"] == "lodo"].nMAE.mean(),
                     "n_features_pooled": int(g[g["mode"] == "global"].n_features.iloc[0])})
    return pd.DataFrame(rows)


def s_models(core, experiment):
    lo = core[(core["mode"] == "lodo") & (core.experiment == experiment)]; nv = naive_lodo(core)
    rows = []
    for m, g in lo.groupby("model"):
        pooled = core[(core["mode"] == "global") & (core.experiment == experiment) & (core.model == m)].set_index("phase")
        rows.append({"model": m, "pooled_nMAE_charge": pooled.nMAE.get("charge"), "pooled_R2_charge": pooled.R2.get("charge"),
                     "pooled_nMAE_discharge": pooled.nMAE.get("discharge"), "pooled_R2_discharge": pooled.R2.get("discharge"),
                     "lodo_median_nMAE_charge": g[g.phase == "charge"].nMAE.median(), "lodo_median_nMAE_discharge": g[g.phase == "discharge"].nMAE.median(),
                     "negative_transfer_scopes": int(sum(r.nMAE > nv.naive_nMAE[(r.phase, r.scope)] for _, r in g.iterrows())),
                     "share_pred_negative_max": g.share_pred_negative.max()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", type=Path, required=True); ap.add_argument("--repeated", type=Path)
    ap.add_argument("--nested", type=Path, help="CSV de nested_selection.py select")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="random_forest"); ap.add_argument("--experiment", default="all_features")
    ap.add_argument("--selection-space", default="experiments", choices=["experiments", "experiments_x_models"])
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    core = load_tree(a.core)
    print(f"nucleo: {len(core)} ajustes, {core.experiment.nunique()} experimentos, {core.model.nunique()} modelos")
    repeated = load_tree(a.repeated) if a.repeated and a.repeated.exists() and list(a.repeated.rglob("metrics_global.csv")) else None
    print(f"braco R: {len(repeated) if repeated is not None else 0} ajustes")
    nested = pd.read_csv(a.nested) if a.nested and a.nested.exists() else None
    t3, t3_long = t3_hierarchy(core, repeated, a.core, a.model, a.experiment); t3.to_csv(a.out / "T3_hierarchy.csv", index=False)
    t3_long.to_csv(a.out / "fig3_data.csv", index=False)
    t4, t4_scope = t4_selection(core, nested, a.model, a.experiment, a.selection_space)
    t4.to_csv(a.out / "T4_selection.csv", index=False); t4_scope.to_csv(a.out / "T4_per_scope.csv", index=False)
    t5, t5_cells = t5_paired(core, a.model, a.experiment); t5.to_csv(a.out / "T5_paired.csv", index=False); t5_cells.to_csv(a.out / "T5_cells.csv", index=False)
    t6 = t6_per_domain(core, repeated, a.model, a.experiment); t6.to_csv(a.out / "T6_per_domain.csv", index=False)
    sa = s_ablations(core, a.model); sa.to_csv(a.out / "S_ablations.csv", index=False)
    sm = s_models(core, a.experiment); sm.to_csv(a.out / "S_models.csv", index=False)
    s_ablations_by_phase(core, a.model).to_csv(a.out / "S_ablations_by_phase.csv", index=False)
    t6[["phase", "dataset", "lodo_nMAE", "naive_nMAE", "lodo_over_naive", "negative_transfer"]].to_csv(a.out / "S_negative_transfer.csv", index=False)
    if repeated is not None:
        wd, _ = within_domain(core, repeated, a.model, a.experiment); wd.to_csv(a.out / "S_within_partitions.csv", index=False)
        pr = pooled_repeated(repeated, a.model, a.experiment)
        if pr is not None: pr.reset_index().to_csv(a.out / "S_pooled_partitions.csv", index=False); print("\nS_pooled_partitions\n" + pr.round(3).to_string())
        dc = distinct_test_cells(repeated, a.model, a.experiment)
        if dc is not None: dc.to_csv(a.out / "S_distinct_test_cells.csv", index=False); print("\nS_distinct_test_cells\n" + dc.to_string(index=False))
    g = t3.set_index(["protocol", "summary"])
    resumo = {
        "pre_specified": {"model": a.model, "experiment": a.experiment},
        "n_fits_core": int(len(core)), "n_experiments": int(core.experiment.nunique()), "n_models": int(core.model.nunique()),
        "pooled_weighted_nMAE": {ph: float(g.loc[("Pooled (all domains in training)", "Sample-weighted"), f"{ph}_nMAE"]) for ph in PHASES},
        "pooled_weighted_R2": {ph: float(g.loc[("Pooled (all domains in training)", "Sample-weighted"), f"{ph}_R2"]) for ph in PHASES},
        "lodo_median_nMAE": {ph: float(g.loc[("Cross-domain, LODO", "Median over 7 domains"), f"{ph}_nMAE"]) for ph in PHASES},
        "lodo_macro_nMAE": {ph: float(g.loc[("Cross-domain, LODO", "Macro-average over 7 domains"), f"{ph}_nMAE"]) for ph in PHASES},
        "naive_lodo_median_nMAE": {ph: float(g.loc[("Naive median predictor, LODO", "Median over 7 domains"), f"{ph}_nMAE"]) for ph in PHASES},
        "negative_transfer_pre_specified": {"count": int(t6.negative_transfer.sum()), "of": int(len(t6)),
                                            "scopes": [f"{r.dataset} {r.phase}" for _, r in t6[t6.negative_transfer].iterrows()]},
        "ratio_lodo_within_median": {ph: float(t6[t6.phase == ph].ratio_lodo_within.median()) for ph in PHASES},
        "within_source": t6.within_source.iloc[0],
        "best_ablation_lodo": sa.iloc[0].experiment, "worst_ablation_lodo": sa.iloc[-1].experiment,
        "paired_all_cells": t5.iloc[-2].to_dict(),
        "selection_regimes": t4.to_dict(orient="records"),
    }
    (a.out / "resumo.json").write_text(json.dumps(resumo, indent=2, default=str))
    (a.out / "README_tabelas.md").write_text(
        "# Tabelas geradas por new/aggregate.py\n\n"
        f"Fonte: `{a.core}`" + (f", `{a.repeated}`" if repeated is not None else "") + (f", `{a.nested}`" if nested is not None else "") + "\n\n"
        "T3_hierarchy: Tabela 3 (hierarquia; pooled ponderado e macro, intra-dominio, LODO macro e mediana, piso ingenuo).\n"
        "T4_selection: Tabela 4 (regimes de selecao; otimismo mediano e transferencia negativa); T4_per_scope: por escopo.\n"
        "T5_paired: Tabela 5 (pareado carga x descarga sob LODO; Wilcoxon, Holm, bootstrap); T5_cells: por celula.\n"
        "T6_per_domain: Tabela 6 (intra-dominio, LODO, razao, LODO/ingenuo).\n"
        "S_ablations, S_models, S_negative_transfer, S_within_partitions: suplementares. fig3_data: dados da Figura 3.\n"
        "resumo.json: numeros soltos citados no texto.\n")
    pd.set_option("display.width", 200)
    print("\nT3\n" + t3.round(3).to_string(index=False)); print("\nT4\n" + t4.round(1).to_string(index=False))
    print("\nT5\n" + t5.round(4).to_string(index=False)); print("\nT6\n" + t6.round(3).drop(columns=["within_source"]).to_string(index=False))
    print("\nS_models\n" + sm.round(3).to_string(index=False)); print("\nS_ablations\n" + sa.round(3).to_string(index=False))
    print(f"\ngravado em {a.out}")


if __name__ == "__main__":
    main()
