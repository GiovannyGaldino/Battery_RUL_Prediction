#!/usr/bin/env python
"""Thin runner for hierarchical feature extraction v3."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Run hierarchical feature extraction v3.")
    p.add_argument("--base-dir", type=Path, default=Path("."))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fast-mode", action="store_true")
    p.add_argument("--no-stages", action="store_true")
    p.add_argument("--dataset", default=None, help="Optional dataset filter passed to the extractor.")
    p.add_argument("--phase-dirs", nargs="*", default=None, help="Optional phase folders, e.g. charge discharge.")
    args, unknown = p.parse_known_args()

    base = args.base_dir.expanduser().resolve()
    cmd = [
        sys.executable,
        str(base / "src" / "feature_extraction" / "build_hierarchical_features_v3.py"),
        "--input-root", str(base / "data" / "interim"),
        "--output-root", str(base / "data" / "processed" / "features_hierarchical_v3"),
        "--reports-root", str(base / "reports" / "feature_extraction_v3"),
        "--workers", str(args.workers),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.fast_mode:
        cmd.append("--fast-mode")
    if args.no_stages:
        cmd.append("--no-stages")
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.phase_dirs:
        cmd.extend(["--phase-dirs", *args.phase_dirs])
    cmd.extend(unknown)

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(base), check=True)


if __name__ == "__main__":
    main()
