#!/usr/bin/env python3
"""Testes do patch 01 sobre trajetorias sinteticas. Sem dados reais, sem HPC."""
import sys, pathlib, importlib.util
import numpy as np, pandas as pd
RAIZ = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ext", RAIZ / "patched" / "build_hierarchical_features_v3.py")
ext = importlib.util.module_from_spec(spec); sys.modules["ext"] = ext; spec.loader.exec_module(ext)

falhas = []
def check(cond, msg):
    print(("  ok   " if cond else "  FALHA") + " " + msg)
    if not cond: falhas.append(msg)

def eol(cap, **kw):
    base = dict(initial_capacity=1.0, threshold_ratio=0.80, rolling_window=5,
                consecutive=3, min_position=3)
    base.update(kw)
    return ext.find_eol_position(pd.Series(cap, dtype=float), **base)

print("== find_eol_position: motivos de rejeicao ==")
# 1. degrada linearmente de 1.0 a 0.6 em 40 ciclos: deve achar
lin = np.linspace(1.0, 0.6, 40)
pos, _, capv, reason, mn = eol(lin)
check(reason == "found" and np.isfinite(pos), f"trajetoria limpa -> found (pos={pos}, reason={reason})")
check(abs(mn - 0.6) < 1e-9, f"min_ratio_reached = 0.6 (obtido {mn:.3f})")

# 2. nunca cruza (para em 0.83)
pos, _, _, reason, mn = eol(np.linspace(1.0, 0.83, 40))
check(reason == "never_crossed", f"para em 0.83 -> never_crossed (obtido {reason})")
check(abs(mn - 0.83) < 1e-9, "min_ratio 0.83")

# 3. cruza e volta acima de 0.90 no fim (outlier tardio): v2 rejeita por permanencia
s = np.linspace(1.0, 0.6, 40); s[-1] = 0.95
pos, _, _, reason, _ = eol(s)
check(reason == "permanence_violated", f"outlier tardio bruto, modo v2 -> permanence_violated (obtido {reason})")
# 3b. mesma serie, permanencia na suavizada: o outlier isolado nao sobrevive a mediana movel 5
pos, _, _, reason, _ = eol(s, permanence_on_smooth=True)
check(reason == "found", f"outlier tardio, permanencia na suavizada -> found (obtido {reason})")

# 4. a suavizada so fica abaixo na ultima posicao (queda nos 3 ultimos brutos):
#    1 posicao consecutiva < 3 -> no_persistence
s = np.ones(40); s[-3:] = 0.70
pos, _, _, reason, _ = eol(s)
check(reason == "no_persistence", f"queda so no fim, consecutive=3 -> no_persistence (obtido {reason})")
# 4b. dois pontos isolados no meio: a mediana movel 5 nem registra cruzamento
s = np.ones(40); s[20:22] = 0.70
_, _, _, reason, _ = eol(s)
check(reason == "never_crossed", f"2 pontos isolados sao absorvidos pela mediana movel -> never_crossed ({reason})")

# 5. capacidade inicial invalida
pos, _, _, reason, mn = eol(lin, initial_capacity=np.nan)
check(reason == "no_initial_capacity", "initial nan -> no_initial_capacity")

# 6. permanencia escalada: limiar 0.85 com margem 0.10 -> 0.95
s = np.linspace(1.0, 0.80, 40); s[-1] = 0.93   # volta a 0.93 no fim
_, _, _, r_abs, _ = eol(s, threshold_ratio=0.85, permanence_ratio=0.90)   # v2: 0.93 > 0.90 -> rejeita
_, _, _, r_mar, _ = eol(s, threshold_ratio=0.85, permanence_ratio=0.95)   # margin: 0.93 < 0.95 -> aceita
check(r_abs == "permanence_violated" and r_mar == "found",
      f"permanencia absoluta 0.90 rejeita ({r_abs}), escalada 0.95 aceita ({r_mar})")

# 7. reproducao v2: assinatura antiga via defaults
pos, _, capv, reason, _ = eol(lin)
check(reason == "found", "defaults reproduzem v2")

print("\n== build_targets: nomes de coluna por limiar ==")
def celula(bid, cap, ds="SYN"):
    n = len(cap)
    return pd.DataFrame({
        "dataset": ds, "battery_id": bid, "phase_folder": "charge", "phase": "charge",
        "cycle_index": np.arange(n), "cycle_number": np.arange(n),
        "source_file_key": f"{ds}/{bid}", "source_row_id": np.arange(n),
        "phase_capacity_ah": cap, "nominal_capacity_ah": 1.0,
    })
df = pd.concat([celula("A", np.linspace(1.0, 0.6, 40)),
                celula("B", np.linspace(1.0, 0.805, 40)),  # so cruza 0.85, nao 0.80 (C0 estimado ~0.991)
                celula("C", np.linspace(1.0, 0.95, 40))],  # nunca
               ignore_index=True)
tg, summ = ext.build_targets(df, initial_method="early_median", initial_window=5,
                             initial_fraction=0.02, eol_rolling_window=5, eol_consecutive=3,
                             eol_min_position=3, drop_post_eol=True,
                             eol_threshold=0.80, eol_secondary_threshold=0.85,
                             permanence_mode="margin", permanence_margin=0.10,
                             permanence_on_smooth=True)
cols = set(tg.columns)
for c in ["target_rul80_steps_clipped", "target_rul85_steps_clipped",
          "target_eol80_found", "target_eol85_found",
          "target_eol80_reject_reason", "target_eol85_reject_reason",
          "target_eol80_min_ratio_reached", "target_eol85_threshold_ratio",
          "target_eol85_permanence_ratio"]:
    check(c in cols, f"coluna {c}")
check("target_rul70_steps_clipped" not in cols, "sem colunas rul70 quando secundario=0.85")
por = tg.groupby("battery_id").agg(f80=("target_eol80_found","max"), f85=("target_eol85_found","max"),
                                  r80=("target_eol80_reject_reason","first"), r85=("target_eol85_reject_reason","first"))
print(por.to_string())
check(bool(por.loc["A","f80"]) and bool(por.loc["A","f85"]), "A: achada nos dois limiares")
check((not por.loc["B","f80"]) and bool(por.loc["B","f85"]), "B: censurada a 0.80, achada a 0.85")
check(por.loc["B","r80"] == "never_crossed", f"B a 0.80 -> never_crossed ({por.loc['B','r80']})")
check((not por.loc["C","f80"]) and (not por.loc["C","f85"]), "C: censurada nos dois")
check(abs(summ.loc[summ.battery_id=="B", "eol80_min_ratio_reached"].iloc[0] - 0.812) < 0.02,
      "summary traz min_ratio_reached")
check(abs(summ.loc[summ.battery_id=="B", "eol85_permanence_ratio"].iloc[0] - 0.95) < 1e-9,
      "summary registra permanencia 0.95 para limiar 0.85 (margin)")
# celulas de A: RUL85 <= RUL80 em toda linha valida (EOL85 vem antes)
a = tg[tg.battery_id=="A"].dropna(subset=["target_rul80_steps_clipped","target_rul85_steps_clipped"])
check((a.target_rul85_steps_clipped <= a.target_rul80_steps_clipped).all(), "RUL85 <= RUL80 linha a linha")

print("\n== consumidores dos alvos com limiares 80/85 ==")
try:
    ds = ext.create_dataset_summary(tg)
    check("eol80_batteries" in ds.columns and "eol85_batteries" in ds.columns, f"create_dataset_summary com 80/85: {[c for c in ds.columns if 'eol' in c]}")
    check(int(ds.eol85_batteries.sum()) == 2 and int(ds.eol80_batteries.sum()) == 1, "contagens por limiar (A e B a 0.85; so A a 0.80)")
except Exception as e:
    check(False, f"create_dataset_summary quebrou: {e}")

print(f"\n{'TODOS OS TESTES PASSARAM' if not falhas else str(len(falhas))+' FALHA(S)'}")
sys.exit(1 if falhas else 0)
