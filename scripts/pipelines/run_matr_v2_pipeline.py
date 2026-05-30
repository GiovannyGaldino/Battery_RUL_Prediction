r"""
Run MATR adapter v2 pipeline: extraction -> cleaning -> audit.

This runner separates:
    --project-dir : where the source code is located.
    --base-dir    : where the Batteries data/reports folders are located.

Typical use from the repository root:
    python scripts/pipelines/run_matr_v2_pipeline.py --base-dir .

Default raw path used by the extraction step:
    <base-dir>\data\raw\MATR\*.mat
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str]) -> None:
    print("\n" + "=" * 88)
    print("Running:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print("=" * 88)
    subprocess.run(cmd, check=True)


def default_project_dir() -> Path:
    # scripts/pipelines/run_matr_v2_pipeline.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MATR adapter v2 extraction, cleaning and audit.")
    parser.add_argument("--base-dir", type=str, required=True, help="Data/report root directory, e.g. <project-root>")
    parser.add_argument("--project-dir", type=str, default=None, help="Project source-code root. Defaults to the root that contains this runner.")
    parser.add_argument("--raw-dir", type=str, default=None, help="Optional MATR raw directory. Defaults to <base-dir>/data/raw/MATR.")
    parser.add_argument("--discharge-current-threshold-a", type=float, default=0.5)
    parser.add_argument("--confirm-discharge-points", type=int, default=1)
    parser.add_argument("--start-cycle-index", type=int, default=1)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--enable-voltage-window", action="store_true", help="Pass this to cleaning if you want voltage-window trimming. Default preserves full contiguous segments.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve() if args.project_dir else default_project_dir()
    python = sys.executable

    extract_script = project_dir / "src" / "adapters" / "matr" / "extract_matr_cycles.py"
    clean_script = project_dir / "src" / "adapters" / "matr" / "clean_matr_cycles.py"
    audit_script = project_dir / "src" / "adapters" / "matr" / "audit_matr_cycles.py"

    for path in [extract_script, clean_script, audit_script]:
        if not path.exists():
            raise FileNotFoundError(f"Script not found: {path}")

    extract_cmd = [
        python,
        str(extract_script),
        "--base-dir",
        str(base_dir),
        "--discharge-current-threshold-a",
        str(args.discharge_current_threshold_a),
        "--confirm-discharge-points",
        str(args.confirm_discharge_points),
        "--start-cycle-index",
        str(args.start_cycle_index),
        "--min-points",
        str(args.min_points),
    ]
    if args.raw_dir:
        extract_cmd.extend(["--raw-dir", str(Path(args.raw_dir).expanduser().resolve())])

    clean_cmd = [python, str(clean_script), "--base-dir", str(base_dir)]
    if args.enable_voltage_window:
        clean_cmd.append("--enable-voltage-window")

    run_step(extract_cmd)
    run_step(clean_cmd)

    if not args.skip_audit:
        run_step([python, str(audit_script), "--base-dir", str(base_dir)])

    print("\nMATR adapter v2 pipeline finished.")


if __name__ == "__main__":
    main()
