#!/usr/bin/env python3
"""Pre-voo da campanha: reproduz, sem treinar, o que o runner fara com cada matriz.

Roda em minutos e responde, para cada alvo e fase:
  1. estrutura: X, y e id alinhados; y finito e > 0; conjunto de dominios esperado;
     celulas e linhas por dominio;
  2. grupos de features (feature_group do runner): tamanho de cada grupo, fracao em
     `other`, grupo vazio (o caso stage_gap da v2);
  3. n_features por experimento x escopo, com o MESMO select_experiment_features e o
     MESMO filtro de ausencia do runner, para pooled, 7 LODO e 7 por dominio. Sinaliza
     zero features e qualquer escopo abaixo de --min-ratio x pooled (o caso das curvas
     de 2026-09-03: 138 de 341 em heldout_MATR);
  4. experimentos com o mesmo conjunto de colunas (duplicatas como no_stage_gap ==
     all_features na v2).
Qualquer sinalizacao fecha o portao (exit 1). Escreve preflight_<tag>.csv/.md em --out.
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("runner", path)
    r = importlib.util.module_from_spec(spec); sys.modules["runner"] = r; spec.loader.exec_module(r)
    return r


def n_features_table(r, X, meta, experiments, missing_threshold):
    datasets = sorted(meta["dataset"].unique())
    scopes = [("global", "all_datasets", np.ones(len(X), bool))]
    scopes += [("lodo", f"heldout_{d}", (meta["dataset"] != d).to_numpy()) for d in datasets]
    scopes += [("per_dataset", d, (meta["dataset"] == d).to_numpy()) for d in datasets]
    rows, colsets = [], {}
    for e in experiments:
        sel = r.select_experiment_features(X.columns, e)
        colsets[e] = frozenset(sel)
        for mode, scope, mask in scopes:
            # os dois filtros de prepare_feature_columns, separados: ausencia e constancia
            Xt = X.loc[mask, sel]
            miss = Xt.isna().mean(axis=0); keep1 = miss[miss <= missing_threshold].index.tolist()
            if keep1:
                nun = Xt[keep1].nunique(dropna=True); keep2 = nun[nun > 1].index.tolist()
            else:
                keep2 = []
            try:
                kept = r.prepare_feature_columns(Xt, sel, missing_threshold); n, err = len(kept), ""
            except ValueError as ex:
                n, err = 0, str(ex)
            assert n == len(keep2), "replica dos filtros divergiu do runner"
            rows.append({"experiment": e, "mode": mode, "scope": scope, "n_selected": len(sel),
                         "n_after_missing": len(keep1), "n_constant_dropped": len(keep1) - len(keep2),
                         "n_features": n, "n_train_rows": int(mask.sum()),
                         "n_train_cells": int(meta.loc[mask, "battery_id"].nunique()), "error": err})
    return pd.DataFrame(rows), colsets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--phase", required=True, choices=["charge", "discharge"])
    ap.add_argument("--target", required=True)
    ap.add_argument("--runner", type=Path, default=RAIZ / "patched" / "run_modeling_final_campaign_hpc.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", default="CALCE,HUST,MATR,NRW,OXFORD,PCOE,XJTU")
    ap.add_argument("--missing-threshold", type=float, default=0.95)
    ap.add_argument("--min-ratio", type=float, default=0.8, help="n_features do escopo / pooled abaixo disto sinaliza")
    ap.add_argument("--max-other", type=float, default=0.10, help="fracao maxima de colunas no grupo 'other'")
    ap.add_argument("--min-cells", type=int, default=3, help="celulas minimas por dominio para a particao intra-dominio existir")
    ap.add_argument("--known-missing", type=Path, default=RAIZ / "docs" / "known_missing_by_domain.csv",
                    help="CSV (dataset,column,reason): colunas declaradas como ausentes por propriedade dos dados; "
                         "descontadas do calculo da razao, e so elas")
    a = ap.parse_args()
    r = load_runner(a.runner)
    flags = []
    X, y, meta = r.load_phase(a.input_root, a.phase, a.target)
    T = f"[PREVOO {a.tag}]"
    print(f"{T} apos load_phase: {X.shape[0]} linhas x {X.shape[1]} features, {meta['battery_id'].nunique()} celulas")

    # 1. estrutura
    exp_ds = set(a.datasets.split(","))
    got = set(meta["dataset"].unique())
    if got != exp_ds: flags.append(f"dominios: esperado {sorted(exp_ds)}, encontrado {sorted(got)}")
    yv = pd.to_numeric(y, errors="coerce")
    if not (np.isfinite(yv).all() and (yv > 0).all()): flags.append("y com valores nao finitos ou <= 0 apos load_phase")
    per_ds = meta.groupby("dataset").agg(rows=("battery_id", "size"), cells=("battery_id", "nunique"))
    per_ds["y_mean"] = yv.groupby(meta["dataset"].values).mean().round(1)
    per_ds["y_max"] = yv.groupby(meta["dataset"].values).max()
    print(f"{T} por dominio:\n{per_ds.to_string()}")
    # Menos de 3 celulas inviabiliza a particao intra-dominio (2 treino + 1 teste); entre 3 e 4 e
    # aviso: OXFORD tem 4 celulas na descarga (6 na carga), como na v2 e no manuscrito.
    tiny = per_ds[per_ds["cells"] < a.min_cells]
    if len(tiny): flags.append(f"dominios com menos de {a.min_cells} celulas: {list(tiny.index)}")
    small = per_ds[(per_ds["cells"] >= a.min_cells) & (per_ds["cells"] < 5)]
    if len(small): print(f"{T} AVISO: dominios com menos de 5 celulas (particao 80/20 fica com 1 celula de teste): {list(small.index)}")

    # 2. grupos
    groups = pd.Series({c: r.feature_group(c) for c in X.columns})
    gc = groups.value_counts()
    print(f"{T} grupos de features:\n{gc.to_string()}")
    other = gc.get("other", 0) / len(groups)
    if other > a.max_other: flags.append(f"grupo 'other' com {other:.1%} das colunas (> {a.max_other:.0%})")
    for g in ("curves", "capacity", "energy", "temperature", "time_sampling", "stage_gap", "voltage", "current", "power"):
        if gc.get(g, 0) == 0: flags.append(f"grupo '{g}' vazio")
    n_causal = sum(c.lower().startswith("causal_") for c in X.columns)
    print(f"{T} colunas causal_*: {n_causal}")
    if n_causal == 0: flags.append("nenhuma coluna causal_* (no_causal_trend seria igual a all_features)")

    # 3. n_features por experimento x escopo
    known = pd.DataFrame(columns=["dataset", "column", "reason"])
    if a.known_missing.exists():
        known = pd.read_csv(a.known_missing)
        print(f"{T} colunas declaradas como ausentes por dominio: {len(known)} (de {a.known_missing.name})")
    exps = r.expand_experiments(["all"])
    tab, colsets = n_features_table(r, X, meta, exps, a.missing_threshold)
    # colunas >95% ausentes por dominio: explicam as sinalizacoes e alimentam o CSV de declaradas
    gone_by_ds = {}
    for d in sorted(meta["dataset"].unique()):
        miss = X[(meta["dataset"] == d).to_numpy()].isna().mean()
        gone_by_ds[d] = sorted(miss[miss > a.missing_threshold].index)
    gone_tab = pd.DataFrame([(d, c, r.feature_group(c)) for d, cs in gone_by_ds.items() for c in cs],
                            columns=["dataset", "column", "group"])
    declared = set(zip(known["dataset"], known["column"]))
    gone_tab["declared"] = [(d, c) in declared for d, c in zip(gone_tab["dataset"], gone_tab["column"])]
    if len(gone_tab):
        print(f"{T} colunas >{a.missing_threshold:.0%} ausentes por dominio (total / declaradas):")
        print(gone_tab.groupby("dataset").agg(total=("column", "size"), declaradas=("declared", "sum")).to_string())
    undeclared = gone_tab[~gone_tab["declared"]]
    pooled = tab[tab["mode"] == "global"].set_index("experiment")["n_after_missing"]
    # razao do filtro de AUSENCIA apenas: e ele que denuncia uma familia vazia num dominio.
    # Constantes dentro de um dominio (protocolo fixo) sao esperadas e ficam so reportadas.
    tab["ratio_vs_pooled"] = tab.apply(lambda s: s["n_after_missing"] / pooled[s["experiment"]] if pooled[s["experiment"]] else np.nan, axis=1)
    const = tab[tab["experiment"] == "all_features"].set_index("scope")["n_constant_dropped"]
    print(f"{T} constantes removidas por escopo (all_features): {const.to_dict()}")
    lodo_const = tab[(tab["mode"] == "lodo") & (tab["n_constant_dropped"] > 0.10 * tab["n_after_missing"])]
    for _, s in lodo_const.iterrows():
        flags.append(f"{s['experiment']} em lodo/{s['scope']}: {s['n_constant_dropped']} constantes removidas com seis dominios no treino")
    piv = tab.pivot(index="experiment", columns="scope", values="n_features")
    cols = ["all_datasets"] + [c for c in piv.columns if c.startswith("heldout_")] + [c for c in piv.columns if not c.startswith("heldout_") and c != "all_datasets"]
    print(f"{T} n_features por experimento x escopo (apos filtro de ausencia {a.missing_threshold}):\n{piv[cols].to_string()}")
    zero = tab[tab["n_features"] == 0]
    for _, s in zero.iterrows(): flags.append(f"{s['experiment']} em {s['mode']}/{s['scope']}: sem features ({s['error']})")
    low = tab[(tab["n_features"] > 0) & (tab["ratio_vs_pooled"] < a.min_ratio)]
    for _, s in low.iterrows():
        sel = colsets[s["experiment"]]
        dec = [c for c in known[known["dataset"] == s["scope"]]["column"] if c in sel] if s["mode"] == "per_dataset" else []
        adj = pooled[s["experiment"]] - len(dec)
        ratio_adj = s["n_after_missing"] / adj if adj else np.nan
        und = undeclared[(undeclared["dataset"] == s["scope"]) & (undeclared["column"].isin(sel))]
        if ratio_adj >= a.min_ratio and len(und) == 0:
            print(f"{T} {s['experiment']} em {s['scope']}: {s['n_after_missing']} de {pooled[s['experiment']]} apos ausencia, "
                  f"{len(dec)} colunas declaradas ausentes; razao ajustada {ratio_adj:.2f}: ok")
            continue
        fams = und.groupby("group")["column"].apply(
            lambda c: f"{len(c)} ({', '.join(sorted(c)[:6])}{'...' if len(c) > 6 else ''})").to_dict()
        flags.append(f"{s['experiment']} em {s['mode']}/{s['scope']}: {s['n_after_missing']} features apos o filtro de ausencia, "
                     f"{s['ratio_vs_pooled']:.2f} do pooled ({pooled[s['experiment']]}); nao declaradas: {fams}")
    a.out.mkdir(parents=True, exist_ok=True)
    gone_tab.to_csv(a.out / f"missing_by_domain_{a.tag}.csv", index=False)

    # 4. duplicatas
    seen = {}
    for e, cs in colsets.items():
        if cs in seen: flags.append(f"experimentos com o mesmo conjunto de colunas: {seen[cs]} == {e}")
        else: seen[cs] = e

    a.out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.out / f"preflight_{a.tag}.csv", index=False)
    with open(a.out / f"preflight_{a.tag}.md", "w") as f:
        f.write(f"# Pre-voo {a.tag}\n\n{X.shape[0]} linhas x {X.shape[1]} features\n\n## Por dominio\n\n```\n{per_ds.to_string()}\n```\n\n")
        f.write(f"## Grupos\n\n```\n{gc.to_string()}\n```\n\n## n_features\n\n```\n{piv[cols].to_string()}\n```\n\n")
        f.write("## Sinalizacoes\n\n" + ("\n".join(f"* {x}" for x in flags) if flags else "nenhuma") + "\n")
    if flags:
        print(f"{T} PORTAO FECHADO, {len(flags)} sinalizacao(oes):"); [print("   -", x) for x in flags]; return 1
    print(f"{T} ok: nenhuma sinalizacao")
    return 0


if __name__ == "__main__":
    sys.exit(main())
