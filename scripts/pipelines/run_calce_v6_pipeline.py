#!/usr/bin/env python
"""Run the complete CALCE v6 adapter pipeline.

This helper removes the previous CALCE interim/report outputs by default and then
runs extraction, cleaning, and auditing in sequence. It does not modify raw data.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_project_root(project_root_arg: str | None) -> Path:
    if project_root_arg:
        return Path(project_root_arg).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() or (candidate / "src").exists():
            return candidate
    return current


def run_step(project_root: Path, script: Path, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(script), "--project-root", str(project_root), *extra_args]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(project_root), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CALCE v6 extraction, cleaning and auditing.")
    parser.add_argument("--project-root", default=None, help="Project root. Defaults to the current Batteries folder.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not remove old CALCE interim/report outputs before running.")
    parser.add_argument("--skip-plots", action="store_true", help="Pass --skip-plots to audit.")
    parser.add_argument("--max-cycles-per-battery", type=int, default=40, help="Maximum cycles per battery to plot during audit.")
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    adapter_dir = project_root / "src" / "adapters" / "calce"
    extract_script = adapter_dir / "extract_calce_cycles.py"
    clean_script = adapter_dir / "clean_calce_cycles.py"
    audit_script = adapter_dir / "audit_calce_cycles.py"

    for script in [extract_script, clean_script, audit_script]:
        if not script.exists():
            raise FileNotFoundError(f"Required script not found: {script}")

    if not args.keep_existing:
        for folder in [project_root / "data" / "interim" / "calce", project_root / "data" / "reports" / "calce"]:
            if folder.exists():
                print(f"[CLEAN] Removing previous output folder: {folder}")
                shutil.rmtree(folder)

    run_step(project_root, extract_script, [])
    run_step(project_root, clean_script, [])
    audit_args = ["--max-cycles-per-battery", str(args.max_cycles_per_battery)]
    if args.skip_plots:
        audit_args.append("--skip-plots")
    run_step(project_root, audit_script, audit_args)

    print("[OK] CALCE v6 pipeline completed.")
    print(f"[OK] Check: {project_root / 'data' / 'reports' / 'calce' / 'audit_report.json'}")


if __name__ == "__main__":
    main()
