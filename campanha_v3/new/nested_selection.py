#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nested_selection.py  (campanha_v3, novo)  --  selecao source-only NESTED

Em v2 a "selecao source-only" escolhia a configuracao para o alvo T usando os
escores LODO dos outros seis dominios; mas cada um desses escores veio de um
modelo cujo treino INCLUIA T. Era um limite otimista, nao uma selecao limpa.

Aqui T e retirado de todo treino usado na selecao:

  para cada alvo T:
     para cada validacao interna V != T:
         treina em D \\ {T, V}, avalia em V           (--exclude-datasets T,
                                                        --lodo-heldouts V)
     c* = argmin_c  media_V  nMAE(c; V)               (ou mediana, reportadas ambas)
     resultado nested(T) = escore LODO EXTERNO de c* em T (da campanha nucleo)

Dois modos:
  plan   : gera um arquivo de jobs (uma linha de comando por (T, V)) para o HPC
  select : le os metrics_global.csv das rodadas internas e da campanha nucleo,
           faz o argmin e grava nested_selection_<phase>.csv

Custo: 7 alvos x 6 dobras internas = 42 rodadas do runner, cada uma com os
experimentos pedidos e (por default) so o modelo pre-especificado.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

DATASETS_DEFAULT = ["CALCE", "HUST", "MATR", "NRW", "OXFORD", "PCOE", "XJTU"]


def plan(a):
    linhas = []
    for T in a.datasets:
        for V in a.datasets:
            if V == T:
                continue
            out = a.output_root / f"target_{T}" / f"inner_{V}"
            cmd = (f"python {a.runner} --input-root {a.input_root} --output-root {out} "
                   f"--target {a.target} --campaign-component nested_inner "
                   f"--phases charge discharge --models {' '.join(a.models)} "
                   f"--modes lodo --lodo-heldouts {V} --exclude-datasets {T} "
                   f"--experiments {' '.join(a.experiments)} "
                   f"--random-state {a.random_state} --model-random-state {a.model_random_state} "
                   f"--sample-weighting dataset_inverse --no-save-predictions --skip-existing "
                   f"--n-jobs 1 --model-n-jobs {a.model_n_jobs}"
                   + (" --elasticnet-cv" if a.elasticnet_cv else ""))
            linhas.append(cmd)
    a.jobs_file.parent.mkdir(parents=True, exist_ok=True)
    a.jobs_file.write_text("\n".join(linhas) + "\n")
    print(f"{len(linhas)} jobs em {a.jobs_file}  (rode com: xargs -P <n> -I{{}} bash -c '{{}}' < {a.jobs_file})")


def _load_metrics(root: Path) -> pd.DataFrame:
    fs = list(root.rglob("metrics_global.csv"))
    if not fs:
        raise SystemExit(f"nenhum metrics_global.csv em {root}")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def select(a):
    inner = _load_metrics(a.output_root)
    core = _load_metrics(a.core_root)
    metric = a.metric
    inner["target_dataset"] = inner["scope"].str.replace("heldout_", "", regex=False)  # V
    # o alvo T vem do caminho: .../target_T/inner_V/...
    # recuperamos T pelo output_root gravado no run_info? nao existe; usamos a estrutura de pastas
    rows = []
    for f in a.output_root.rglob("metrics_global.csv"):
        parts = f.relative_to(a.output_root).parts
        T = parts[0].replace("target_", ""); V = parts[1].replace("inner_", "")
        d = pd.read_csv(f); d["T"] = T; d["V"] = V; rows.append(d)
    inner = pd.concat(rows, ignore_index=True)
    inner = inner[inner.model.isin(a.models)]
    core = core[(core["mode"] == "lodo") & core.model.isin(a.models)]
    core["T"] = core["scope"].str.replace("heldout_", "", regex=False)
    out = []
    for (phase, T, model), g in inner.groupby(["phase", "T", "model"]):
        agg = g.groupby("experiment")[metric].agg(["mean", "median", "count"]).reset_index()
        for crit in ["mean", "median"]:
            best = agg.sort_values(crit).iloc[0]
            ext = core[(core.phase == phase) & (core["T"] == T) & (core.model == model) & (core.experiment == best.experiment)]
            out.append({"phase": phase, "target": T, "model": model, "criterion": crit,
                        "selected_experiment": best.experiment,
                        f"inner_{crit}_{metric}": float(best[crit]), "inner_folds": int(best["count"]),
                        f"outer_lodo_{metric}": float(ext[metric].iloc[0]) if len(ext) else float("nan"),
                        "outer_found": bool(len(ext))})
    res = pd.DataFrame(out)
    a.output_root.mkdir(parents=True, exist_ok=True)
    p = a.output_root / f"nested_selection_{metric}.csv"; res.to_csv(p, index=False)
    print(res.to_string(index=False)); print(f"\ngravado: {p}")
    faltam = res[~res.outer_found]
    if len(faltam):
        print(f"\nATENCAO: {len(faltam)} selecoes sem escore externo na campanha nucleo (experimento ausente?)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); s = sub.add_parser("select")
    for x in (p, s):
        x.add_argument("--output-root", type=Path, required=True, help="raiz das rodadas internas (nested/)")
        x.add_argument("--datasets", nargs="+", default=DATASETS_DEFAULT)
        x.add_argument("--models", nargs="+", default=["random_forest"])
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--target", default="target_rul80_steps_clipped")
    p.add_argument("--experiments", nargs="+", default=["all"])
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--model-random-state", type=int, default=42)
    p.add_argument("--model-n-jobs", type=int, default=8)
    p.add_argument("--elasticnet-cv", action="store_true")
    p.add_argument("--jobs-file", type=Path, default=Path("nested_jobs.txt"))
    s.add_argument("--core-root", type=Path, required=True, help="raiz da campanha nucleo (LODO externo)")
    s.add_argument("--metric", default="nMAE", choices=["nMAE", "nMAE_clip0", "MAE"])
    a = ap.parse_args()
    for k in ("output_root", "runner", "input_root", "core_root", "jobs_file"):
        if hasattr(a, k) and getattr(a, k) is not None:
            setattr(a, k, getattr(a, k).expanduser())
    (plan if a.cmd == "plan" else select)(a)


if __name__ == "__main__":
    main()
