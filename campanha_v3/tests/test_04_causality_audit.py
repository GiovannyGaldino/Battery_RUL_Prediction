#!/usr/bin/env python3
"""O portao pega o escopo de celula e o proxy do ordinal, e deixa passar features intra-ciclo."""
import sys, pathlib, importlib.util
import numpy as np, pandas as pd
RAIZ = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ca", RAIZ/"new"/"causality_audit.py")
ca = importlib.util.module_from_spec(spec); sys.modules["ca"] = ca; spec.loader.exec_module(ca)
rng = np.random.default_rng(1)
rows = []
for b in range(30):                       # 30 celulas, vidas 40..400
    n = int(rng.integers(40, 400)); life = n
    cyc = np.arange(n)
    cap = 1.0 - (0.25 * cyc / life) + rng.normal(0, 0.003, n)
    rows.append(pd.DataFrame({
        "dataset": "SYN", "battery_id": f"B{b:02d}", "phase_folder": "charge", "cycle_index": cyc,
        "target": (life - cyc).astype(float),
        # legitimas intra-ciclo
        "voltage_mean": rng.normal(3.7, 0.02, n), "current_max": rng.normal(1.0, 0.05, n),
        "capacityh_mean": cap * rng.normal(1, 0.01, n),
        "time_duration_s": rng.normal(3600, 100, n),
        # tendencia degradante legitima (capacidade), monotona mas ruidosa
        "energy_total": cap * 3.7 + rng.normal(0, 0.01, n),
        # causal em janela expansiva: muda ao longo da celula
        "causal_slope": pd.Series(cap).expanding().apply(lambda v: np.polyfit(np.arange(len(v)), v, 1)[0] if len(v) > 2 else np.nan).to_numpy(),
        # VAZAMENTOS
        "leak_cell_slope": -0.25 / life,                       # constante na celula, varia entre celulas
        "leak_cell_spearman": -0.99 + rng.normal(0, 1e-4),    # idem, ruido minusculo
        "leak_row_id": 100000 * b + cyc,                       # ordinal do ciclo
        # constante GLOBAL (nao e vazamento, so inutil): nao deve ser sinalizada como escopo de celula
        "const_global": 1.0,
        # constante de PROTOCOLO por celula: poucos valores, sem relacao com a vida -> NAO sinalizar
        "proto_flag": float(b % 3),
        # constante por celula que E a vida (n de ciclos): inteira, mas prediz a vida -> sinalizar (A2)
        "leak_n_cycles": float(life),
    }))
df = pd.concat(rows, ignore_index=True)
meta = df[["dataset", "battery_id", "phase_folder", "cycle_index"]]
X = df.drop(columns=["dataset", "battery_id", "phase_folder", "cycle_index", "target"])
rep = ca.audit(X, meta, df["target"]).set_index("column")
falhas = []
def check(c, m):
    print(("  ok   " if c else "  FALHA")+" "+m)
    if not c: falhas.append(m)
print(rep[["share_cells_constant","const_cell_specific_ratio","const_max_within_dataset_life_rho","share_cells_ordinal_proxy","flag_cell_scope_constant","flag_cycle_ordinal_proxy","flagged"]].round(3).to_string())
check(bool(rep.loc["leak_cell_slope","flag_cell_scope_constant"]), "constante exata na celula -> sinalizada")
check(bool(rep.loc["leak_cell_spearman","flag_cell_scope_constant"]), "constante na celula com ruido 1e-4 (unico valor por celula) -> sinalizada")
check(bool(rep.loc["leak_row_id","flag_cycle_ordinal_proxy"]), "ordinal do ciclo -> sinalizada como proxy")
check(not bool(rep.loc["const_global","flagged"]), "constante GLOBAL nao e sinalizada (overall_nunique=1)")
check(bool(rep.loc["proto_flag","cell_constant"]) and not bool(rep.loc["proto_flag","flagged"]), "constante de PROTOCOLO por celula (3 valores, sem relacao com a vida) NAO e sinalizada")
check(bool(rep.loc["leak_n_cycles","flag_cell_scope_constant"]), f"n de ciclos por celula (inteiro, prediz a vida) e sinalizada por A2 (|rho| {abs(rep.loc['leak_n_cycles','const_max_within_dataset_life_rho']):.2f})")
for c in ["voltage_mean","current_max","capacityh_mean","time_duration_s","energy_total","causal_slope"]:
    check(not bool(rep.loc[c,"flagged"]), f"legitima nao sinalizada: {c}")
check(rep.loc["energy_total","share_cells_ordinal_proxy"] < 0.95, "capacidade degradante ruidosa nao e proxy do ordinal a rho>=0.9995")
print(f"\n{'TODOS OS TESTES PASSARAM' if not falhas else str(len(falhas))+' FALHA(S)'}"); sys.exit(1 if falhas else 0)
