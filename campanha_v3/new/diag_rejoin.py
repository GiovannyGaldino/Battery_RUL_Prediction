#!/usr/bin/env python3
"""Por que uma juncao features x alvos nao casa? Compara as chaves dos dois lados.
Uso: python diag_rejoin.py <features.parquet> <targets_v3.parquet>"""
import sys, pandas as pd, numpy as np
f, t = sys.argv[1], sys.argv[2]
F = pd.read_parquet(f, columns=["source_file_key","source_row_id","dataset","battery_id","phase_folder","cycle_index","feature_level"])
T = pd.read_parquet(t, columns=["source_file_key","source_row_id","dataset","battery_id","phase_folder","cycle_index"])
ds = F.dataset.astype(str).iloc[0]; ph = F.phase_folder.astype(str).iloc[0]
Tsub = T[(T.dataset.astype(str)==ds) & (T.phase_folder.astype(str)==ph)]
print(f"FEATURES {f.split('/')[-1]}: {len(F):,} linhas | niveis {F.feature_level.unique().tolist()}")
Fp = F[F.feature_level.astype(str)=="phase"] if "feature_level" in F else F
print(f"  nivel phase: {len(Fp):,} linhas | source_row_id unicos {Fp.source_row_id.nunique():,} "
      f"min {Fp.source_row_id.min()} max {Fp.source_row_id.max()} dtype {Fp.source_row_id.dtype}")
print(f"  source_file_key: {F.source_file_key.unique().tolist()}")
print(f"  celulas {Fp.battery_id.nunique()} | cycle_index min {Fp.cycle_index.min()} max {Fp.cycle_index.max()}")
print(f"ALVOS {ds}/{ph}: {len(Tsub):,} linhas | source_row_id min {Tsub.source_row_id.min()} max {Tsub.source_row_id.max()}")
print(f"  source_file_key: {Tsub.source_file_key.unique().tolist()}")
print(f"  celulas {Tsub.battery_id.nunique()} | cycle_index min {Tsub.cycle_index.min()} max {Tsub.cycle_index.max()}")
# casamento por id
k1 = ["source_file_key","source_row_id"]
m1 = Fp.merge(Tsub[k1].assign(_ok=1), on=k1, how="left")
print(f"\ncasam por (file_key, row_id): {int(m1._ok.notna().sum()):,} de {len(Fp):,}")
# casamento por chave semantica
k2 = ["dataset","battery_id","phase_folder","cycle_index"]
for D in (Fp, Tsub):
    D["dataset"]=D["dataset"].astype(str); D["battery_id"]=D["battery_id"].astype(str); D["phase_folder"]=D["phase_folder"].astype(str)
dupT = int(Tsub.duplicated(k2).sum()); dupF = int(Fp.duplicated(k2).sum())
m2 = Fp.merge(Tsub[k2].drop_duplicates().assign(_ok=1), on=k2, how="left")
print(f"casam por (dataset, battery, phase, cycle_index): {int(m2._ok.notna().sum()):,} de {len(Fp):,} | duplicatas dessa chave: alvos {dupT}, features {dupF}")
# onde estao os nao casados por id
nc = m1[m1._ok.isna()]
if len(nc):
    print(f"\nnao casados por id: row_id min {nc.source_row_id.min()} max {nc.source_row_id.max()} | celulas {nc.battery_id.nunique()}")
    print("  primeiras 5 chaves:", nc[k1].head().values.tolist())
    print("  ha id igual no alvo com OUTRO file_key?", int(Tsub.source_row_id.isin(nc.source_row_id).sum()))
