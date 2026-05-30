#!/usr/bin/env python
"""Run extraction, cleaning, and/or auditing for a supported dataset adapter.

This is a thin public entry point. It does not implement adapter logic; it only
calls the dataset-specific scripts in `src/adapters` with consistent arguments.
Run it from the repository root.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATASET_SCRIPTS = {
    "calce": {"arg": "--project-root", "extract": "src/adapters/calce/extract_calce_cycles.py", "clean": "src/adapters/calce/clean_calce_cycles.py", "audit": "src/adapters/calce/audit_calce_cycles.py"},
    "hust": {"arg": "--base-dir", "extract": "src/adapters/hust/extract_hust_cycles.py", "clean": "src/adapters/hust/clean_hust_cycles.py", "audit": "src/adapters/hust/audit_hust_cycles.py"},
    "matr": {"arg": "--base-dir", "extract": "src/adapters/matr/extract_matr_cycles.py", "clean": "src/adapters/matr/clean_matr_cycles.py", "audit": "src/adapters/matr/audit_matr_cycles.py"},
    "nasa_pcoe": {"arg": "--base-dir", "extract": "src/adapters/nasa_pcoe/extract_pcoe_cycles.py", "clean": "src/adapters/nasa_pcoe/clean_pcoe_cycles.py", "audit": "src/adapters/nasa_pcoe/audit_pcoe_cycles.py"},
    "nrw": {"arg": "--base-dir", "extract": "src/adapters/nrw/extract_nrw_cycles.py", "clean": "src/adapters/nrw/clean_nrw_cycles.py", "audit": "src/adapters/nrw/audit_nrw_cycles.py"},
    "ora": {"arg": "--base-dir", "extract": "src/adapters/ora/extract_ora_cycles.py", "clean": "src/adapters/ora/clean_ora_cycles.py", "audit": "src/adapters/ora/audit_ora_cycles.py"},
    "rwth": {"arg": "--base-dir", "extract": "src/adapters/rwth/extract_rwth_cycles.py", "clean": "src/adapters/rwth/clean_rwth_cycles.py", "audit": "src/adapters/rwth/audit_rwth_cycles.py"},
    "xjtu": {"arg": "--base-dir", "extract": "src/adapters/xjtu/extract_xjtu_cycles.py", "clean": "src/adapters/xjtu/clean_xjtu_cycles.py", "audit": "src/adapters/xjtu/audit_xjtu_cycles.py"},
}


def run(script: Path, base_arg: str, base_dir: Path, extra: list[str]) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Adapter script not found: {script}")
    cmd = [sys.executable, str(script), base_arg, str(base_dir), *extra]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public dataset adapters.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SCRIPTS))
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Repository/data root. Default: current directory.")
    parser.add_argument("--stage", choices=["extract", "clean", "audit", "all"], default="all")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra arguments appended to each called script.")
    args = parser.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    spec = DATASET_SCRIPTS[args.dataset]
    stages = ["extract", "clean", "audit"] if args.stage == "all" else [args.stage]

    for stage in stages:
        run(base_dir / spec[stage], spec["arg"], base_dir, args.extra)


if __name__ == "__main__":
    main()
