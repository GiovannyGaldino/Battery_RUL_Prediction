#!/usr/bin/env python3
"""Verificacao de release: hashes das matrizes de entrada, contagem de ajustes por componente e
regeneracao das tabelas do manuscrito a partir dos resultados arquivados.

  --write   grava docs/RELEASE_CHECKSUMS.json (primeira vez, na maquina onde a campanha rodou)
  (padrao)  compara com docs/RELEASE_CHECKSUMS.json e regenera T3-T6 num diretorio temporario,
            comparando com as tabelas arquivadas em <out>/tables (tolerancia 1e-9)
Exit 1 em qualquer divergencia.
"""
import argparse, hashlib, json, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
EXPECTED = {"core_t80": 2520, "arm_t85": 360, "repeated": 1920, "nested": 1176}
TABLES = ["T3_hierarchy.csv", "T4_selection.csv", "T5_paired.csv", "T6_per_domain.csv", "S_ablations.csv", "S_models.csv"]


def sha256(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def input_hashes(data_root: Path) -> dict:
    out = {}
    for sub in ["modeling_ready_t80", "modeling_ready_t85", "features_causal"]:
        d = data_root / sub
        if not d.exists():
            continue
        for f in sorted(d.rglob("*.parquet")):
            out[str(f.relative_to(data_root))] = {"sha256": sha256(f), "bytes": f.stat().st_size}
    return out


def fit_counts(out_root: Path) -> dict:
    return {k: len(list((out_root / k).rglob("metrics_global.csv"))) if (out_root / k).exists() else 0 for k in EXPECTED}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True); ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--checksums", type=Path, default=RAIZ / "docs" / "RELEASE_CHECKSUMS.json")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(); problemas = []
    ih = input_hashes(a.data_root); fc = fit_counts(a.out_root)
    th = {t: sha256(a.out_root / "tables" / t) for t in TABLES if (a.out_root / "tables" / t).exists()}
    print(f"entradas: {len(ih)} parquets | ajustes: {fc} | tabelas arquivadas: {len(th)}")
    for k, v in EXPECTED.items():
        if fc[k] != v: problemas.append(f"{k}: {fc[k]} ajustes, esperado {v}")
    if a.write:
        a.checksums.parent.mkdir(parents=True, exist_ok=True)
        a.checksums.write_text(json.dumps({"inputs": ih, "fit_counts": fc, "tables": th}, indent=1))
        print(f"gravado {a.checksums}")
    else:
        if not a.checksums.exists(): problemas.append(f"sem {a.checksums}; rode com --write na maquina da campanha")
        else:
            ref = json.loads(a.checksums.read_text())
            for k, v in ref["inputs"].items():
                if k not in ih: problemas.append(f"entrada ausente: {k}")
                elif ih[k]["sha256"] != v["sha256"]: problemas.append(f"hash divergente: {k}")
            for k, v in ref["tables"].items():
                if th.get(k) != v: problemas.append(f"tabela arquivada divergente: {k}")
        # regenera as tabelas e compara
        with tempfile.TemporaryDirectory() as tmp:
            args = [sys.executable, str(RAIZ / "new" / "aggregate.py"), "--core", str(a.out_root / "core_t80"), "--out", tmp]
            if (a.out_root / "repeated").exists(): args += ["--repeated", str(a.out_root / "repeated")]
            if (a.out_root / "nested" / "nested_selection_nMAE.csv").exists(): args += ["--nested", str(a.out_root / "nested" / "nested_selection_nMAE.csv")]
            r = subprocess.run(args, capture_output=True, text=True)
            if r.returncode != 0: problemas.append("aggregate.py falhou: " + r.stderr[-400:])
            else:
                for t in TABLES:
                    A, B = a.out_root / "tables" / t, Path(tmp) / t
                    if not A.exists() or not B.exists(): problemas.append(f"tabela ausente: {t}"); continue
                    da, db = pd.read_csv(A), pd.read_csv(B)
                    try:
                        pd.testing.assert_frame_equal(da, db, check_exact=False, rtol=1e-9, atol=1e-9)
                    except AssertionError as e:
                        problemas.append(f"{t} regenerada difere da arquivada: {str(e)[:200]}")
    if problemas:
        print("VERIFICACAO FALHOU:"); [print("  -", p) for p in problemas]; return 1
    print("VERIFICACAO OK: entradas, contagens e tabelas conferem"); return 0


if __name__ == "__main__":
    sys.exit(main())
