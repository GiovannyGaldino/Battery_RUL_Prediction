#!/usr/bin/env python
"""Thin runner for building modeling dataframes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Build modeling dataframes from clean cycles and feature files.")
    p.add_argument("--base-dir", type=Path, default=Path("."))
    args, unknown = p.parse_known_args()

    base = args.base_dir.expanduser().resolve()
    cmd = [
        sys.executable,
        str(base / "src" / "modeling" / "build_modeling_dataframe_v1.py"),
        "--interim-root", str(base / "data" / "interim"),
        "--features-root", str(base / "data" / "processed" / "features_hierarchical_v3"),
        "--output-root", str(base / "data" / "processed" / "modeling_dataset_v1"),
        "--reports-root", str(base / "reports" / "modeling_dataset_v1"),
        *unknown,
    ]

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(base), check=True)


if __name__ == "__main__":
    main()
