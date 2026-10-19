#!/usr/bin/env python3
"""Portao de cobertura: cada familia de features precisa estar preenchida em cada dominio.

Motivo (2026-09-03): a primeira campanha v3 foi montada com `curve_features_v3`, que so
tem curvas V(Q)/Q(V) validas para o MATR; nos outros seis dominios as 204 colunas de
curva chegaram vazias e o filtro de ausencia do runner as removeu em silencio. A
auditoria de causalidade nao pega isso: ela testa o que existe, nao o que falta.

Regra: para cada (dataset, familia) com >= --min-cols colunas, a fracao media de
valores presentes deve ser >= --min-coverage. Qualquer violacao fecha o portao (exit 1).
Familia = prefixo do nome ate o primeiro '_' (vq, qv, voltage, capacity, causal, ...).
"""
import argparse, sys
from pathlib import Path
import pandas as pd


def family(col: str) -> str:
    return col.split("_", 1)[0].lower()


def coverage_table(X: pd.DataFrame, dataset: pd.Series) -> pd.DataFrame:
    fam = pd.Series({c: family(c) for c in X.columns})
    rows = []
    present = X.notna()
    for f, cols in fam.groupby(fam).groups.items():
        cols = list(cols)
        cov = present[cols].mean(axis=1).groupby(dataset.values).mean()
        for ds, v in cov.items():
            rows.append({"dataset": ds, "family": f, "n_cols": len(cols), "coverage": float(v)})
    return pd.DataFrame(rows).sort_values(["family", "dataset"]).reset_index(drop=True)


def audit(X: pd.DataFrame, dataset: pd.Series, min_coverage: float = 0.5, min_cols: int = 5) -> pd.DataFrame:
    t = coverage_table(X, dataset)
    t["flag"] = (t["n_cols"] >= min_cols) & (t["coverage"] < min_coverage)
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=Path, required=True)
    ap.add_argument("--id", type=Path, required=True, help="parquet com a coluna dataset, alinhado a --x")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--min-cols", type=int, default=5)
    a = ap.parse_args()
    X = pd.read_parquet(a.x); ids = pd.read_parquet(a.id)
    if len(X) != len(ids):
        print(f"[COBERTURA {a.tag}] X tem {len(X)} linhas e id tem {len(ids)}: desalinhado"); return 1
    t = audit(X, ids["dataset"], a.min_coverage, a.min_cols)
    a.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out / f"coverage_{a.tag}.csv", index=False)
    piv = t.pivot(index="family", columns="dataset", values="coverage").round(3)
    print(f"[COBERTURA {a.tag}] {X.shape[1]} colunas, {t['family'].nunique()} familias, {ids['dataset'].nunique()} dominios")
    print(piv.to_string())
    bad = t[t["flag"]]
    if len(bad):
        print(f"[COBERTURA {a.tag}] PORTAO FECHADO: {len(bad)} (dominio, familia) abaixo de {a.min_coverage}:")
        print(bad.to_string(index=False)); return 1
    print(f"[COBERTURA {a.tag}] ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
