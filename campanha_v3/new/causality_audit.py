#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
causality_audit.py  (campanha_v3, novo)  --  PORTAO antes de qualquer treino

Testa cada coluna de entrada do modelo por dois sinais que uma feature
intra-ciclo legitima nao pode exibir:

  A. CONSTANTE DENTRO DA CELULA e variavel entre celulas, E alem disso
     (A1) especifica da celula: numero de valores distintos >= --cell-specific
          x numero de celulas (uma estatistica da trajetoria e continua e quase
          nunca se repete entre celulas), OU
     (A2) preditiva da vida: |Spearman(constante, vida da celula)| >= --life-rho
          DENTRO de algum dataset com >= --min-cells-per-dataset celulas.
     Sem A1/A2, uma constante por celula e flag de protocolo (ex.: VQ_valid_p003,
     energy_sign_change_count): nao e vazamento temporal e vai so para o relatorio.
     Teria pego as cinco capacity_trend_* de v2 (A1 e A2).
  B. PROXY DO ORDINAL DO CICLO
     |Spearman(feature, cycle_index)| >= --ordinal-rho dentro da celula, na
     grande maioria das celulas -> indice disfarcado. Teria pego source_row_id.

Tambem reporta, sem bloquear, as colunas mais correlacionadas com o alvo dentro
da celula (C), para inspecao humana.

Entrada: ou um parquet de modelagem com metadados, ou o par X/id do
modeling_ready ({phase}_X.parquet + {phase}_id.parquet).

Sai com codigo 1 se A ou B sinalizarem qualquer coluna (a menos de --report-only).
Grava <out>/causality_audit_<tag>.csv com todas as colunas e medidas.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

GRUPO_PREF = ["dataset", "battery_id", "phase_folder", "phase"]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy(); rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def audit(X: pd.DataFrame, meta: pd.DataFrame, y: pd.Series | None,
          min_rows: int = 3, const_share: float = 0.99, ordinal_rho: float = 0.9995,
          ordinal_share: float = 0.95, max_cells: int | None = None,
          cell_specific: float = 0.5, life_rho: float = 0.5, min_cells_per_dataset: int = 5) -> pd.DataFrame:
    grp = [c for c in GRUPO_PREF if c in meta.columns]
    if "battery_id" not in grp:
        raise ValueError("meta precisa ter battery_id")
    cyc = "cycle_index" if "cycle_index" in meta.columns else ("cycle_number" if "cycle_number" in meta.columns else None)
    key = meta[grp].astype(str).agg("|".join, axis=1).to_numpy()
    cells = pd.Series(key).groupby(key).indices
    items = list(cells.items())
    if max_cells:
        items = items[:max_cells]
    cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    Xn = X[cols].to_numpy(dtype=float)
    cyc_v = pd.to_numeric(meta[cyc], errors="coerce").to_numpy(float) if cyc else None
    y_v = pd.to_numeric(y, errors="coerce").to_numpy(float) if y is not None else None

    n_cells_ok = 0
    const_hits = np.zeros(len(cols)); ord_hits = np.zeros(len(cols)); ord_n = np.zeros(len(cols))
    tgt_rho = np.zeros(len(cols)); tgt_n = np.zeros(len(cols))
    cell_val = []      # (dataset, celula) -> valor constante por coluna (NaN se nao constante)
    cell_life = []     # vida da celula: numero de linhas (ou max do alvo, se houver)
    cell_ds = []
    ds_v = meta["dataset"].astype(str).to_numpy() if "dataset" in meta.columns else np.array(["_"] * len(meta))
    for _, ii in items:
        if len(ii) < min_rows:
            continue
        n_cells_ok += 1
        sub = Xn[ii]
        row_val = np.full(len(cols), np.nan)
        # A: nunique <= 1 ignorando NaN
        for j in range(len(cols)):
            v = sub[:, j]; v = v[np.isfinite(v)]
            if len(v) >= 1 and np.unique(v).size <= 1:
                const_hits[j] += 1; row_val[j] = v[0]
        cell_val.append(row_val); cell_ds.append(ds_v[ii[0]])
        cell_life.append(float(np.nanmax(y_v[ii])) if y_v is not None else float(len(ii)))
        # B e C
        for j in range(len(cols)):
            if cyc_v is not None:
                r = _spearman(sub[:, j], cyc_v[ii])
                if np.isfinite(r):
                    ord_n[j] += 1
                    if abs(r) >= ordinal_rho:
                        ord_hits[j] += 1
            if y_v is not None:
                r = _spearman(sub[:, j], y_v[ii])
                if np.isfinite(r):
                    tgt_rho[j] += abs(r); tgt_n[j] += 1
    overall_nunique = np.array([X[c].nunique(dropna=True) for c in cols])
    # A1: especificidade da celula; A2: correlacao com a vida DENTRO de cada dataset
    CV = np.array(cell_val) if cell_val else np.zeros((0, len(cols)))
    life = np.array(cell_life); dss = np.array(cell_ds)
    spec_ratio = np.full(len(cols), np.nan); life_r = np.full(len(cols), np.nan)
    for j in range(len(cols)):
        v = CV[:, j] if len(CV) else np.array([])
        ok = np.isfinite(v)
        if ok.sum() >= 2:
            spec_ratio[j] = np.unique(v[ok]).size / max(ok.sum(), 1)
            best = np.nan
            for d in np.unique(dss[ok]):
                m = ok & (dss == d)
                if m.sum() >= min_cells_per_dataset:
                    r = _spearman(v[m], life[m])
                    if np.isfinite(r) and (not np.isfinite(best) or abs(r) > abs(best)):
                        best = r
            life_r[j] = best
    rep = pd.DataFrame({
        "column": cols,
        "cells_evaluated": n_cells_ok,
        "share_cells_constant": const_hits / max(n_cells_ok, 1),
        "overall_nunique": overall_nunique,
        "share_cells_ordinal_proxy": np.where(ord_n > 0, ord_hits / np.maximum(ord_n, 1), np.nan),
        "mean_abs_spearman_vs_target": np.where(tgt_n > 0, tgt_rho / np.maximum(tgt_n, 1), np.nan),
        "const_cell_specific_ratio": spec_ratio,
        "const_max_within_dataset_life_rho": life_r,
    })
    const_cell = (rep.share_cells_constant >= const_share) & (rep.overall_nunique > 1)
    rep["cell_constant"] = const_cell
    rep["flag_cell_scope_constant"] = const_cell & (
        (rep.const_cell_specific_ratio >= cell_specific) | (rep.const_max_within_dataset_life_rho.abs() >= life_rho))
    rep["protocol_constant_not_flagged"] = const_cell & ~rep["flag_cell_scope_constant"]
    rep["flag_cycle_ordinal_proxy"] = rep.share_cells_ordinal_proxy >= ordinal_share
    rep["flagged"] = rep.flag_cell_scope_constant | rep.flag_cycle_ordinal_proxy
    return rep.sort_values(["flagged", "mean_abs_spearman_vs_target"], ascending=[False, False]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modeling-parquet", type=Path, help="parquet com features + metadados + alvo")
    ap.add_argument("--x", type=Path); ap.add_argument("--id", type=Path); ap.add_argument("--y", type=Path)
    ap.add_argument("--target", default="target_rul80_steps_clipped")
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--tag", default="audit")
    ap.add_argument("--const-share", type=float, default=0.99)
    ap.add_argument("--ordinal-rho", type=float, default=0.9995)
    ap.add_argument("--ordinal-share", type=float, default=0.95)
    ap.add_argument("--max-cells", type=int, default=None, help="amostra de celulas, para teste rapido")
    ap.add_argument("--cell-specific", type=float, default=0.5, help="A1: distintos/celulas acima disto = estatistica da trajetoria")
    ap.add_argument("--life-rho", type=float, default=0.5, help="A2: |rho| com a vida dentro do dataset acima disto = vazamento")
    ap.add_argument("--min-cells-per-dataset", type=int, default=5)
    ap.add_argument("--report-only", action="store_true", help="nao falha, so grava o relatorio")
    a = ap.parse_args()

    if a.modeling_parquet:
        df = pd.read_parquet(a.modeling_parquet.expanduser())
        meta_cols = [c for c in GRUPO_PREF + ["cycle_index", "cycle_number"] if c in df.columns]
        meta = df[meta_cols]; y = df[a.target] if a.target in df.columns else None
        X = df.drop(columns=meta_cols + ([a.target] if a.target in df.columns else []))
    elif a.x and a.id:
        X = pd.read_parquet(a.x.expanduser()); meta = pd.read_parquet(a.id.expanduser())
        y = pd.read_parquet(a.y.expanduser()).iloc[:, 0] if a.y else None
        X = X.drop(columns=[c for c in X.columns if c in meta.columns], errors="ignore")
    else:
        sys.exit("informe --modeling-parquet ou --x/--id")

    rep = audit(X, meta, y, const_share=a.const_share, ordinal_rho=a.ordinal_rho,
                ordinal_share=a.ordinal_share, max_cells=a.max_cells,
                cell_specific=a.cell_specific, life_rho=a.life_rho, min_cells_per_dataset=a.min_cells_per_dataset)
    a.out.mkdir(parents=True, exist_ok=True)
    p = a.out / f"causality_audit_{a.tag}.csv"; rep.to_csv(p, index=False)
    fl = rep[rep.flagged]
    print(f"colunas auditadas: {len(rep)} | celulas: {int(rep.cells_evaluated.iloc[0]) if len(rep) else 0}")
    print(f"relatorio: {p}")
    pc = rep[rep.protocol_constant_not_flagged]
    if len(pc):
        print(f"\nconstantes de protocolo por celula, NAO sinalizadas ({len(pc)}; distintos/celulas e |rho| com a vida):")
        for r in pc.itertuples():
            print(f"   {r.column:<55} {r.const_cell_specific_ratio:.3f}   {abs(r.const_max_within_dataset_life_rho) if np.isfinite(r.const_max_within_dataset_life_rho) else float('nan'):.2f}")
    if len(fl):
        print(f"\nSINALIZADAS ({len(fl)}):")
        for r in fl.itertuples():
            mot = []
            if r.flag_cell_scope_constant: mot.append(f"constante na celula ({r.share_cells_constant:.0%}; distintos/celulas {r.const_cell_specific_ratio:.2f}; |rho| vida {abs(r.const_max_within_dataset_life_rho) if np.isfinite(r.const_max_within_dataset_life_rho) else float('nan'):.2f})")
            if r.flag_cycle_ordinal_proxy: mot.append(f"proxy do ordinal ({r.share_cells_ordinal_proxy:.0%})")
            print(f"   {r.column:<55} {'; '.join(mot)}")
        print("\nPORTAO FECHADO." if not a.report_only else "\n(report-only)")
        sys.exit(0 if a.report_only else 1)
    print("\nnenhuma coluna sinalizada. PORTAO ABERTO.")
    top = rep.dropna(subset=["mean_abs_spearman_vs_target"]).head(8)
    if len(top):
        print("\nmais correlacionadas com o alvo dentro da celula (informativo):")
        for r in top.itertuples(): print(f"   {r.column:<55} |rho| medio {r.mean_abs_spearman_vs_target:.3f}")


if __name__ == "__main__":
    main()
