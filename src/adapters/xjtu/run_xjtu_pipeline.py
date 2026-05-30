# -*- coding: utf-8 -*-
"""Run the XJTU extraction, cleaning, and audit pipeline in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from xjtu_schema import default_clean_dir, default_extracted_dir, default_raw_dir, default_report_dir


def run_step(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:", " ".join(command))
    print("=" * 80)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full XJTU adapter pipeline.")
    parser.add_argument("--raw-dir", type=str, default=None, help="Raw XJTU directory. Default: data/raw/XJTU")
    parser.add_argument("--extracted-dir", type=str, default=None, help="Extracted output directory. Default: data/interim/xjtu/extracted")
    parser.add_argument("--clean-dir", type=str, default=None, help="Clean output directory. Default: data/interim/xjtu/clean")
    parser.add_argument("--report-dir", type=str, default=None, help="Report directory. Default: data/reports/xjtu")
    parser.add_argument("--limit-files", type=int, default=None, help="Optional file limit for extraction testing.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip audit plots.")
    parser.add_argument("--parquet-chunk-size", type=int, default=5, help="Rows kept in RAM before parquet flush. Default: 5.")
    parser.add_argument("--read-batch-size", type=int, default=5, help="Rows read per Arrow batch during cleaning. Default: 5.")
    parser.add_argument("--reference-min-capacity-ah", type=float, default=1.0, help="Minimum capacity accepted for initial-capacity reference. Default: 1.0 Ah.")
    parser.add_argument("--reference-max-capacity-ah", type=float, default=2.3, help="Maximum capacity accepted for initial-capacity reference. Default: 2.3 Ah.")
    parser.add_argument("--reference-window-cycles", type=int, default=5, help="Number of first representative cycles used for median initial capacity. Default: 5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    python = sys.executable

    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else default_raw_dir().resolve()
    extracted_dir = Path(args.extracted_dir).resolve() if args.extracted_dir else default_extracted_dir().resolve()
    clean_dir = Path(args.clean_dir).resolve() if args.clean_dir else default_clean_dir().resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else default_report_dir().resolve()

    extract_cmd = [
        python,
        str(here / "extract_xjtu_cycles.py"),
        "--raw-dir", str(raw_dir),
        "--extracted-dir", str(extracted_dir),
        "--report-dir", str(report_dir),
        "--parquet-chunk-size", str(args.parquet_chunk_size),
    ]
    if args.limit_files is not None:
        extract_cmd.extend(["--limit-files", str(args.limit_files)])

    clean_cmd = [
        python,
        str(here / "clean_xjtu_cycles.py"),
        "--extracted-dir", str(extracted_dir),
        "--clean-dir", str(clean_dir),
        "--report-dir", str(report_dir),
        "--parquet-chunk-size", str(args.parquet_chunk_size),
        "--read-batch-size", str(args.read_batch_size),
        "--reference-min-capacity-ah", str(args.reference_min_capacity_ah),
        "--reference-max-capacity-ah", str(args.reference_max_capacity_ah),
        "--reference-window-cycles", str(args.reference_window_cycles),
    ]

    audit_cmd = [
        python,
        str(here / "audit_xjtu_cycles.py"),
        "--clean-dir", str(clean_dir),
        "--report-dir", str(report_dir),
    ]
    if args.skip_plots:
        audit_cmd.append("--skip-plots")

    run_step(extract_cmd)
    run_step(clean_cmd)
    run_step(audit_cmd)


if __name__ == "__main__":
    main()
