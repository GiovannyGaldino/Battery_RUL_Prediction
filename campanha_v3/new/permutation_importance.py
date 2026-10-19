#!/usr/bin/env python3
"""Importancia por permutacao (v3), replica do importancias.py da v2 sobre a matriz v3.

RF + all_features, pooled (particao estratificada por bateria, seed 42) e por dominio
(particao por bateria dentro do dominio), permutacao sobre o TESTE, 5 repeticoes,
neg-MAE, mais importancia por impureza. Participacao por familia (feature_group do
runner) com duas convencoes: soma bruta (v2) e soma com valores negativos truncados em zero.
Saidas: importances.csv (por feature), importances_by_group.csv (por familia).
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from extras_common import load_runner, load_phase_experiment
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", default="target_rul80_steps_clipped"); ap.add_argument("--experiment", default="all_features")
    ap.add_argument("--model", default="random_forest"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-seed", type=int, default=4242); ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-repeats", type=int, default=5); ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--min-cells", type=int, default=3)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    r = load_runner(); rows = []
    for phase in ["charge", "discharge"]:
        X, y, meta = load_phase_experiment(r, a.input_root, phase, a.target, a.experiment)
        print(f"[{phase}] X={X.shape} celulas={meta.battery_id.nunique()}")
        scen = [("global", "all_datasets")] + [("per_dataset", d) for d in sorted(meta.dataset.astype(str).unique())]
        for mode, scope in scen:
            if mode == "global":
                tr, te = r.split_global_by_battery_stratified(meta, a.test_size, a.seed)
            else:
                sel = (meta.dataset.astype(str) == scope).to_numpy(); sub = meta[sel].reset_index(drop=True)
                if sub.battery_id.nunique() < a.min_cells: print(f"   {scope}: <{a.min_cells} celulas, pulado"); continue
                trs, tes = r.split_within_dataset_by_battery(sub, a.test_size, a.seed)
                idx = np.where(sel)[0]; tr = np.zeros(len(meta), bool); te = np.zeros(len(meta), bool); tr[idx[trs]] = True; te[idx[tes]] = True
            cols = r.prepare_feature_columns(X[tr], X.columns, 0.95)
            est, sc = r.make_model_specs(a.model_seed, a.n_jobs)[a.model]
            pipe = r.build_pipeline(est, sc)
            w = r.compute_sample_weights(meta[tr].reset_index(drop=True), "dataset_inverse") if mode == "global" else None
            try:
                pipe.fit(X.loc[tr, cols], y[tr], **({"regressor__sample_weight": w} if w is not None else {}))
            except TypeError:
                pipe.fit(X.loc[tr, cols], y[tr])
            pi = permutation_importance(pipe, X.loc[te, cols], y[te], n_repeats=a.n_repeats, random_state=a.seed,
                                        n_jobs=a.n_jobs, scoring="neg_mean_absolute_error")
            imp = getattr(pipe.steps[-1][1], "feature_importances_", np.full(len(cols), np.nan))
            for c, p, s, i in zip(cols, pi.importances_mean, pi.importances_std, imp):
                rows.append({"phase": phase, "mode": mode, "scope": scope, "feature": c, "group": r.feature_group(c),
                             "perm_importance": float(p), "perm_importance_sd": float(s), "impurity_importance": float(i),
                             "n_train_cells": int(meta.loc[tr, "battery_id"].nunique()), "n_test_cells": int(meta.loc[te, "battery_id"].nunique())})
            print(f"   {mode}/{scope}: {len(cols)} features, {int(tr.sum())} treino, {int(te.sum())} teste")
    d = pd.DataFrame(rows); d.to_csv(a.out / "importances.csv", index=False)
    g = d.assign(perm_clip=d.perm_importance.clip(lower=0)).groupby(["phase", "mode", "scope", "group"]).agg(
        perm_sum=("perm_importance", "sum"), perm_sum_clip=("perm_clip", "sum"), impurity_sum=("impurity_importance", "sum"), n_features=("feature", "size")).reset_index()
    for c in ["perm_sum", "perm_sum_clip", "impurity_sum"]:
        g[c.replace("sum", "share")] = g.groupby(["phase", "mode", "scope"])[c].transform(lambda s: 100 * s / s.sum())
    g.to_csv(a.out / "importances_by_group.csv", index=False)
    print("\nparticipacao por familia, pooled (soma bruta como na v2 / truncada em zero):")
    piv = g[g["mode"] == "global"].pivot_table(index="group", columns="phase", values=["perm_share", "perm_share_clip"]).round(1)
    print(piv.to_string()); print(f"\ngravado em {a.out}")


if __name__ == "__main__":
    main()
