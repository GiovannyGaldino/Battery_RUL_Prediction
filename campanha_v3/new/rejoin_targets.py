#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rejoin_targets.py  (campanha_v3, novo)

Rejunta alvos RECONSTRUIDOS aos parquets de features EXISTENTES, sem refazer a
extracao de curvas. Em v2 as colunas target_* moravam dentro de cada parquet de
features, e mudar o alvo exigia reextrair tudo (horas). Aqui a juncao e por
chave (source_file_key, source_row_id), que ambos os lados carregam.

No mesmo passo:
  * as cinco estatisticas da trajetoria inteira recebem o prefixo cell_
    (escopo de celula: NAO sao preditores validos, ficam so para auditoria);
  * sao criadas as contrapartes CAUSAIS em janela expansiva, prefixo causal_,
    calculadas so com os ciclos <= k:
        causal_capacity_trend_slope_per_step   (regressao linear, forma fechada)
        causal_capacity_trend_pearson          (Pearson; Spearman expansivo nao
                                                tem forma fechada barata, ver README)
        causal_capacity_positive_step_rate
        causal_capacity_negative_step_rate
        causal_capacity_median_abs_step_ah
    com NaN enquanto houver menos de --min-history ciclos (default 3, igual ao
    compute_trend_stats original).

Uso:
  python rejoin_targets.py --features-root  ~/data/processed/features_hierarchical_v3_fix \
                           --targets        ~/data/processed/features_v3_causal/_targets/targets_v3.parquet \
                           --output-root    ~/data/processed/features_v3_causal
Cada parquet de features tem varias linhas por registro (uma por feature_level);
a juncao e muitos-para-um pela chave SEMANTICA (dataset, battery_id, phase_folder,
cycle_index[, cycle_number]). A chave de id nao serve: nos parquets v2 com mais de
32.767 registros, source_row_id foi gravado como int16 e deu a volta.
Aborta se qualquer linha ficar sem alvo ou se a chave nao for unica nos alvos.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

CHAVE = ["source_file_key", "source_row_id"]          # so diagnostico: v2 gravou source_row_id como int16
CHAVE_SEM = ["dataset", "battery_id", "phase_folder", "cycle_index"]   # a juncao real
GRUPO = ["dataset", "battery_id", "phase_folder"]
TREND_V2 = ["capacity_trend_slope_per_step", "capacity_trend_spearman",
            "capacity_positive_step_rate", "capacity_negative_step_rate",
            "capacity_median_abs_step_ah"]


def causal_trend_features(cap: np.ndarray, min_history: int = 3) -> dict:
    """Estatisticas de tendencia em janela EXPANSIVA: a linha k usa so cap[:k+1].
    Complexidade O(n) para slope/pearson/taxas, O(n log n) para a mediana."""
    y = np.asarray(cap, dtype=float)
    n = len(y)
    out = {k: np.full(n, np.nan) for k in
           ["slope", "pearson", "pos_rate", "neg_rate", "med_abs_step"]}
    if n == 0:
        return out
    # validos ate k (NaN nao conta nem desloca o indice x)
    fin = np.isfinite(y)
    x = np.arange(n, dtype=float)
    # somas cumulativas so sobre pontos validos
    w = fin.astype(float)
    cnt = np.cumsum(w)
    sx = np.cumsum(x * w); sy = np.cumsum(np.where(fin, y, 0.0))
    sxx = np.cumsum(x * x * w); sxy = np.cumsum(np.where(fin, x * y, 0.0))
    syy = np.cumsum(np.where(fin, y * y, 0.0))
    den = cnt * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (cnt * sxy - sx * sy) / den
        varx = cnt * sxx - sx * sx
        vary = cnt * syy - sy * sy
        pear = (cnt * sxy - sx * sy) / np.sqrt(varx * vary)
    ok = cnt >= min_history
    out["slope"] = np.where(ok & np.isfinite(slope), slope, np.nan)
    out["pearson"] = np.where(ok & np.isfinite(pear), pear, np.nan)
    # passos entre validos consecutivos
    yv = y.copy()
    idx_valid = np.where(fin)[0]
    if len(idx_valid) >= 2:
        d = np.diff(y[idx_valid])                # passos entre pontos validos
        pos = np.cumsum(d > 0); neg = np.cumsum(d < 0); tot = np.arange(1, len(d) + 1)
        # o passo j fica disponivel a partir do ponto valido j+1
        pr = np.full(n, np.nan); nr = np.full(n, np.nan); ma = np.full(n, np.nan)
        med = pd.Series(np.abs(d)).expanding().median().to_numpy()
        pr_v = pos / tot; nr_v = neg / tot
        # propaga para os indices originais (valido j+1 recebe estatistica ate o passo j)
        pr[idx_valid[1:]] = pr_v; nr[idx_valid[1:]] = nr_v; ma[idx_valid[1:]] = med
        # forward-fill para linhas invalidas e para manter "ultimo valor conhecido"
        pr = pd.Series(pr).ffill().to_numpy(); nr = pd.Series(nr).ffill().to_numpy()
        ma = pd.Series(ma).ffill().to_numpy()
        out["pos_rate"] = np.where(ok, pr, np.nan)
        out["neg_rate"] = np.where(ok, nr, np.nan)
        out["med_abs_step"] = np.where(ok, ma, np.nan)
    return out


def add_causal_features(df: pd.DataFrame, cap_col: str, cycle_col: str, min_history: int) -> pd.DataFrame:
    df = df.copy()
    for c in ["causal_capacity_trend_slope_per_step", "causal_capacity_trend_pearson",
              "causal_capacity_positive_step_rate", "causal_capacity_negative_step_rate",
              "causal_capacity_median_abs_step_ah"]:
        df[c] = np.nan
    order = df.sort_values(GRUPO + [cycle_col], kind="mergesort").index
    for _, idx in df.loc[order].groupby(GRUPO, sort=False).indices.items():
        ii = order[idx]
        cap = pd.to_numeric(df.loc[ii, cap_col], errors="coerce").to_numpy(float)
        f = causal_trend_features(cap, min_history)
        df.loc[ii, "causal_capacity_trend_slope_per_step"] = f["slope"]
        df.loc[ii, "causal_capacity_trend_pearson"] = f["pearson"]
        df.loc[ii, "causal_capacity_positive_step_rate"] = f["pos_rate"]
        df.loc[ii, "causal_capacity_negative_step_rate"] = f["neg_rate"]
        df.loc[ii, "causal_capacity_median_abs_step_ah"] = f["med_abs_step"]
    return df


def _chave(df: pd.DataFrame, extra_cycle_number: bool) -> list:
    k = list(CHAVE_SEM) + (["cycle_number"] if extra_cycle_number else [])
    for c in ("dataset", "battery_id", "phase_folder"):
        df[c] = df[c].astype(str)
    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce")
    if extra_cycle_number:
        df["cycle_number"] = pd.to_numeric(df["cycle_number"], errors="coerce")
    return k


def rejoin_one(feat_path: Path, targets: pd.DataFrame, out_path: Path) -> dict:
    """Juncao MUITOS-PARA-UM pela chave semantica (dataset, battery_id, phase_folder,
    cycle_index[, cycle_number]). A chave de id (source_file_key, source_row_id) NAO
    e usada: nos parquets v2 com mais de 32.767 registros (HUST, MATR) source_row_id
    foi gravado como int16 e deu a volta. O source_row_id correto vem dos alvos."""
    feat = pd.read_parquet(feat_path)
    n_in = len(feat)
    usa_cn = "cycle_number" in feat.columns and "cycle_number" in targets.columns
    k = _chave(feat, usa_cn)
    tg = targets.copy(); _chave(tg, usa_cn)
    # restringe os alvos a este dataset/fase para validar unicidade no escopo certo
    ds, ph = feat["dataset"].iloc[0], feat["phase_folder"].iloc[0]
    tg = tg[(tg["dataset"] == ds) & (tg["phase_folder"] == ph)]
    dup = int(tg.duplicated(k).sum())
    if dup:
        raise RuntimeError(f"{feat_path.name}: chave semantica {k} nao e unica nos alvos ({dup} duplicatas)")
    # derruba tudo que os alvos fornecem (inclusive o source_row_id corrompido e o escopo de celula antigo)
    fornecidas = [c for c in tg.columns if c not in k]
    feat = feat.drop(columns=[c for c in fornecidas + TREND_V2 if c in feat.columns], errors="ignore")
    m = feat.merge(tg, on=k, how="left", validate="many_to_one")
    if len(m) != n_in:
        raise RuntimeError(f"{feat_path.name}: juncao mudou o numero de linhas ({n_in} -> {len(m)})")
    faltando = int(m["target_capacity_ah"].isna().sum())
    if faltando:
        ex = m.loc[m["target_capacity_ah"].isna(), k].drop_duplicates().head(5).values.tolist()
        raise RuntimeError(f"{feat_path.name}: {faltando} linhas sem alvo correspondente; ex.: {ex}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out_path, index=False)
    niveis = m["feature_level"].nunique() if "feature_level" in m.columns else 1
    return {"file": str(feat_path), "rows_in": n_in, "rows_out": len(m), "feature_levels": int(niveis),
            "source_rows": int(m[k].drop_duplicates().shape[0]), "join_key": "+".join(k)}


def prepare_targets(tg: pd.DataFrame, min_history: int) -> pd.DataFrame:
    """Uma linha por registro de origem: renomeia o escopo de celula e calcula as causais."""
    tg = tg.copy()
    tg["source_row_id"] = pd.to_numeric(tg["source_row_id"], errors="coerce").astype("int64")
    tg = tg.rename(columns={c: "cell_" + c for c in TREND_V2 if c in tg.columns})
    cyc = "cycle_index" if "cycle_index" in tg.columns else "cycle_number"
    return add_causal_features(tg, cap_col="target_capacity_ah", cycle_col=cyc, min_history=min_history)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-root", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True, help="_targets/targets_v3.parquet reconstruido")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--min-history", type=int, default=3)
    a = ap.parse_args()
    fr, orr = a.features_root.expanduser(), a.output_root.expanduser()
    tg = pd.read_parquet(a.targets.expanduser())
    print(f"alvos: {len(tg):,} registros; calculando cell_* e causal_* uma vez por registro...")
    tg = prepare_targets(tg, a.min_history)
    files = sorted(p for p in fr.rglob("*.features.parquet") if "_targets" not in p.parts)
    if not files:
        sys.exit(f"nenhum *.features.parquet em {fr}")
    print(f"features: {len(files)} arquivos | alvos: {len(tg):,} linhas")
    rel = []
    for f in files:
        r = rejoin_one(f, tg, orr / f.relative_to(fr))
        rel.append(r); print(f"  ok  {f.relative_to(fr)}  {r['rows_in']:,} linhas | {r['source_rows']:,} registros | {r['feature_levels']} nivel(is) | chave {r['join_key']}")
    (orr / "_targets").mkdir(parents=True, exist_ok=True)
    tg.to_parquet(orr / "_targets" / "targets_v3.parquet", index=False)
    pd.DataFrame(rel).to_csv(orr / "_targets" / "rejoin_report.csv", index=False)
    with open(orr / "_targets" / "rejoin_manifest.json", "w") as fh:
        json.dump({"features_root": str(fr), "targets": str(a.targets), "files": len(files),
                   "rows": int(sum(r["rows_in"] for r in rel)), "min_history": a.min_history,
                   "cell_scope_columns": ["cell_" + t for t in TREND_V2],
                   "causal_columns": ["causal_capacity_trend_slope_per_step", "causal_capacity_trend_pearson",
                                      "causal_capacity_positive_step_rate", "causal_capacity_negative_step_rate",
                                      "causal_capacity_median_abs_step_ah"]}, fh, indent=2)
    print(f"\npronto: {orr}")


if __name__ == "__main__":
    main()
