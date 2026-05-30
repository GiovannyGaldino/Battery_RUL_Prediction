# -*- coding: utf-8 -*-
"""Run the complete Oxford/ORA extraction-cleaning-audit pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .ora_schema import DATASET_FOLDER, infer_default_base_dir
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from ora_schema import DATASET_FOLDER, infer_default_base_dir  # type: ignore


def run_step(script: Path, args: list[str]) -> None:
    command = [sys.executable, str(script), *args]
    print("\n" + "=" * 100)
    print("Running:", " ".join(command))
    print("=" * 100)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    default_base_dir = infer_default_base_dir()
    parser = argparse.ArgumentParser(description="Run the complete Oxford/ORA adapter pipeline.")
    parser.add_argument("--base-dir", type=str, default=str(default_base_dir), help="Project root directory.")
    parser.add_argument(
        "--mat-file",
        type=str,
        default=None,
        help="Path to Oxford_Battery_Degradation_Dataset_1.mat. If omitted, it is resolved from --base-dir.",
    )
    parser.add_argument("--round-vectors", type=int, default=None, help="Optional decimal places for vector rounding.")
    parser.add_argument("--voltage-min", type=float, default=2.70, help="Minimum voltage retained in cleaning.")
    parser.add_argument("--voltage-max", type=float, default=4.25, help="Maximum voltage retained in cleaning.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    this_dir = Path(__file__).resolve().parent
    base_dir = Path(args.base_dir)
    mat_file = Path(args.mat_file) if args.mat_file else base_dir / "data" / "raw" / "ORA" / "Oxford_Battery_Degradation_Dataset_1.mat"
    extracted_dir = base_dir / "data" / "interim" / DATASET_FOLDER / "extracted"
    clean_dir = base_dir / "data" / "interim" / DATASET_FOLDER / "clean"
    report_dir = base_dir / "data" / "reports" / DATASET_FOLDER

    common_extract_args = [
        "--base-dir", str(base_dir),
        "--mat-file", str(mat_file),
        "--out-dir", str(extracted_dir),
        "--report-dir", str(report_dir),
    ]
    if args.round_vectors is not None:
        common_extract_args += ["--round-vectors", str(args.round_vectors)]

    clean_args = [
        "--base-dir", str(base_dir),
        "--input-dir", str(extracted_dir),
        "--out-dir", str(clean_dir),
        "--report-dir", str(report_dir),
        "--voltage-min", str(args.voltage_min),
        "--voltage-max", str(args.voltage_max),
    ]
    if args.round_vectors is not None:
        clean_args += ["--round-vectors", str(args.round_vectors)]

    audit_args = [
        "--base-dir", str(base_dir),
        "--clean-dir", str(clean_dir),
        "--report-dir", str(report_dir),
    ]

    run_step(this_dir / "extract_ora_cycles.py", common_extract_args)
    run_step(this_dir / "clean_ora_cycles.py", clean_args)
    run_step(this_dir / "audit_ora_cycles.py", audit_args)

    print("\nOxford/ORA full pipeline completed.")


if __name__ == "__main__":
    main()
