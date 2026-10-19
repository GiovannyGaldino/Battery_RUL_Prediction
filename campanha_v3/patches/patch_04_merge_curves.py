#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PATCH 04 - merge_modeling_with_curve_features_v3_fixed.py

1. main() ganha argparse (os caminhos eram fixos no codigo). Defaults iguais aos antigos.
2. As colunas de DIAGNOSTICO das curvas sao removidas na juncao. E exatamente a
   lista do sanitizador v2 (_sanitize_model_input_csv_after_run em
   build_modeling_dataframe_v1.py:1244-1258), que nunca chegou a rodar: por isso
   VQ_valid_p003..p030 e QV_valid_p003..p030 (flags booleanas constantes por
   celula) estavam no modelo v2. --keep-curve-diagnostics reproduz v2."""
import sys, pathlib, py_compile
RAIZ = pathlib.Path(__file__).resolve().parent.parent
SRC = RAIZ/"pristine"/"merge_modeling_with_curve_features_v3_fixed.py"
DST = RAIZ/"patched"/"merge_modeling_with_curve_features_v3_fixed.py"
t = SRC.read_text()
V = '''def main():
    modeling_root = Path.home() / "data/batteries/modeling_from_v3"
    curve_root = Path.home() / "data/batteries/curve_features_v3_fixed"
    output_root = Path.home() / "data/batteries/modeling_from_v3_with_curves_fixed"
'''
N = '''def main():
    import argparse
    ap = argparse.ArgumentParser(description="Merge modeling tables with curve features (v3: caminhos por CLI).")
    ap.add_argument("--modeling-root", type=Path, default=Path.home() / "data/batteries/modeling_from_v3")
    ap.add_argument("--curve-root", type=Path, default=Path.home() / "data/batteries/curve_features_v3_fixed")
    ap.add_argument("--output-root", type=Path, default=Path.home() / "data/batteries/modeling_from_v3_with_curves_fixed")
    a = ap.parse_args()
    modeling_root, curve_root, output_root = (p.expanduser() for p in (a.modeling_root, a.curve_root, a.output_root))
'''
if t.count(V) != 1: sys.exit("ABORTADO: ancora de main()")
t = t.replace(V, N, 1)

# 2. filtro de diagnosticos, com a lista do sanitizador v2
V2 = '''    curve_feature_cols = [
        c for c in curves.columns
        if c not in KEYS and (
            c.startswith("VQ_") or
            c.startswith("QV_") or
            c == "_source_curve_file"
        )
    ]
'''
N2 = '''    curve_feature_cols = [
        c for c in curves.columns
        if c not in KEYS and (
            c.startswith("VQ_") or
            c.startswith("QV_") or
            c == "_source_curve_file"
        )
    ]
    # v3: remove diagnosticos de curva. Lista identica a do sanitizador v2, que
    # nunca rodou (build_modeling_dataframe_v1.py:1244-1258).
    if not KEEP_CURVE_DIAGNOSTICS:
        import re as _re
        _exact = {"VQ_n_raw_points", "VQ_n_unique_q", "VQ_q_min", "VQ_q_max", "VQ_q_range",
                  "VQ_voltage_min", "VQ_voltage_max", "VQ_voltage_range"}
        _regex = [r"^VQ_valid_p\\d+$", r"^QV_valid_p\\d+$", r"^VQ_invalid_reason_p\\d+$",
                  r".*invalid_reason.*", r".*_reason$"]
        _drop = [c for c in curve_feature_cols
                 if c in _exact or any(_re.match(rx, c) for rx in _regex)]
        if _drop:
            print(f"curve diagnostics removed ({len(_drop)}): {sorted(_drop)}")
        curve_feature_cols = [c for c in curve_feature_cols if c not in _drop]
'''
if t.count(V2) != 1: sys.exit("ABORTADO: ancora de curve_feature_cols")
t = t.replace(V2, N2, 1)
t = t.replace("KEYS = [", "KEEP_CURVE_DIAGNOSTICS = False   # v3: definido por --keep-curve-diagnostics\n\nKEYS = [", 1)
V3 = '''    ap.add_argument("--output-root", type=Path, default=Path.home() / "data/batteries/modeling_from_v3_with_curves_fixed")
    a = ap.parse_args()'''
N3 = '''    ap.add_argument("--output-root", type=Path, default=Path.home() / "data/batteries/modeling_from_v3_with_curves_fixed")
    ap.add_argument("--keep-curve-diagnostics", action="store_true",
                    help="Mantem VQ_valid_p*, QV_valid_p* e afins (reproduz v2).")
    a = ap.parse_args()
    global KEEP_CURVE_DIAGNOSTICS
    KEEP_CURVE_DIAGNOSTICS = bool(a.keep_curve_diagnostics)'''
if t.count(V3) != 1: sys.exit("ABORTADO: ancora do argparse")
t = t.replace(V3, N3, 1)
DST.write_text(t); py_compile.compile(str(DST), doraise=True)
print("  ok  argparse em main()\n  ok  diagnosticos de curva removidos na juncao (lista do sanitizador v2)\ncompilado:", DST.name)
