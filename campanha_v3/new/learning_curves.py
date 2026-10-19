#!/usr/bin/env python3
"""Curvas de aprendizado por dominio (v3), mesmo desenho da v2 (curva_aprendizado.csv):

por dominio e fase, particoes por bateria com test_size em {0.2, 0.4, 0.6, 0.8}
(treino com 80/60/40/20 % das celulas), 3 sementes de particao, modelos `median`
e RF (all_features), semente de modelo fixa. Saida: learning_curves.csv com
n_train_batt, n_test_batt, MAE, nMAE, R2, RMSE por linha, e um resumo por dominio
(media sobre sementes) em learning_curves_summary.csv.
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from extras_common import load_runner, load_phase_experiment
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", default="target_rul80_steps_clipped"); ap.add_argument("--experiment", default="all_features")
    ap.add_argument("--models", nargs="+", default=["median", "random_forest"])
    ap.add_argument("--test-sizes", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    ap.add_argument("--model-seed", type=int, default=4242); ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--min-cells", type=int, default=3)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    r = load_runner(); rows = []
    for phase in ["charge", "discharge"]:
        X, y, meta = load_phase_experiment(r, a.input_root, phase, a.target, a.experiment)
        for ds in sorted(meta.dataset.astype(str).unique()):
            sel = (meta.dataset.astype(str) == ds).to_numpy(); sub = meta[sel].reset_index(drop=True)
            Xs, ys = X[sel].reset_index(drop=True), y[sel].reset_index(drop=True)
            if sub.battery_id.nunique() < a.min_cells: print(f"[{phase}] {ds}: <{a.min_cells} celulas, pulado"); continue
            for ts in a.test_sizes:
                for seed in a.seeds:
                    tr, te = r.split_within_dataset_by_battery(sub, ts, seed)
                    if tr.sum() == 0 or te.sum() == 0: continue
                    cols = r.prepare_feature_columns(Xs[tr], Xs.columns, 0.95)
                    for mname in a.models:
                        est, sc = r.make_model_specs(a.model_seed, a.n_jobs)[mname]
                        pipe = r.build_pipeline(est, sc); pipe.fit(Xs.loc[tr, cols], ys[tr])
                        m = r.regression_metrics(ys[te], pipe.predict(Xs.loc[te, cols]))
                        rows.append({"phase": phase, "dataset": ds, "test_size": ts, "seed": seed, "model": mname,
                                     "n_train_batt": int(sub.loc[tr, "battery_id"].nunique()), "n_test_batt": int(sub.loc[te, "battery_id"].nunique()),
                                     "n_train_samples": int(tr.sum()), "n_test_samples": int(te.sum()), "n_features": len(cols),
                                     **{k: m[k] for k in ["MAE", "nMAE", "R2", "RMSE", "nMAE_clip0"] if k in m}})
            print(f"[{phase}] {ds}: {len(a.test_sizes) * len(a.seeds) * len(a.models)} ajustes")
    d = pd.DataFrame(rows); d.to_csv(a.out / "learning_curves.csv", index=False)
    s = d.groupby(["phase", "dataset", "model", "test_size"]).agg(n_train_batt=("n_train_batt", "mean"), nMAE=("nMAE", "mean"),
        nMAE_sd=("nMAE", "std"), R2=("R2", "mean"), n_seeds=("seed", "nunique")).reset_index().sort_values(["phase", "dataset", "model", "n_train_batt"])
    # incremento final: variacao relativa do nMAE entre a segunda maior e a maior fracao de treino
    rf = s[s.model == "random_forest"]
    inc = []
    for (ph, ds), g in rf.groupby(["phase", "dataset"]):
        g = g.sort_values("n_train_batt")
        if len(g) >= 2: inc.append({"phase": ph, "dataset": ds, "nMAE_min_train": g.nMAE.iloc[0], "nMAE_max_train": g.nMAE.iloc[-1],
                                    "last_increment_pct": 100 * (g.nMAE.iloc[-1] - g.nMAE.iloc[-2]) / g.nMAE.iloc[-2]})
    s.to_csv(a.out / "learning_curves_summary.csv", index=False); pd.DataFrame(inc).to_csv(a.out / "learning_curves_increments.csv", index=False)
    print(pd.DataFrame(inc).round(3).to_string(index=False)); print(f"\ngravado em {a.out}")


if __name__ == "__main__":
    main()
