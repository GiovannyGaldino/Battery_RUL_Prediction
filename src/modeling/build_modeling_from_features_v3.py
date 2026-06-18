from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = {
    "dataset",
    "source_file",
    "source_path",
    "battery_id",
    "battery_index",
    "cell_id_original",
    "cycle_index",
    "cycle_number",
    "source_cycle_id",
    "cycle_pair_id",
    "phase",
    "phase_folder",
    "feature_level",
    "stage_id",
    "stage_index",
    "stage_start_index",
    "stage_end_index",
}

LEAKAGE_EXACT = {
    "cycle_life",
    "remaining_cycles",
    "remaining_cycles_norm",
    "eol_criterion",
    "eol_capacity_ah",
    "initial_capacity_ah",
    "nominal_capacity_ah",
    "soh",
}

PREFIXES_TO_REMOVE = (
    "target_",
    "input_",
)

SUBSTRINGS_TO_REMOVE = (
    "rul",
    "remaining_cycles",
    "eol",
    "after_eol",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build modeling tables from features_hierarchical_v3 outputs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.home() / "data/batteries/features_hierarchical_v3",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "data/batteries/modeling_from_v3",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target_rul80_steps_clipped",
    )
    parser.add_argument(
        "--feature-level",
        type=str,
        default="phase",
    )
    parser.add_argument(
        "--min-non-null-rate",
        type=float,
        default=0.20,
        help="Drop feature columns with less than this non-null rate.",
    )
    return parser.parse_args()


def should_remove_column(col: str, target: str) -> tuple[bool, str]:
    c = col.lower()

    if col == target:
        return False, "target_y"

    if col in IDENTIFIER_COLUMNS:
        return True, "identifier_or_group_column"

    if col in LEAKAGE_EXACT:
        return True, "direct_lifetime_or_health_label"

    if col.startswith(PREFIXES_TO_REMOVE):
        return True, "target_or_input_diagnostic_prefix"

    if any(s in c for s in SUBSTRINGS_TO_REMOVE):
        return True, "target_leakage_substring"

    return False, ""


def load_feature_file(path: Path, target: str, feature_level: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if "feature_level" in df.columns:
        df = df[df["feature_level"].astype(str).str.lower() == feature_level.lower()].copy()

    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in {path}")

    df = df[df[target].notna()].copy()

    return df


def build_branch(
    files: list[Path],
    branch: str,
    target: str,
    feature_level: str,
    min_non_null_rate: float,
    output_root: Path,
) -> None:
    branch_files = [f for f in files if f.parts[-3] == branch]

    frames = []
    file_rows = []

    for f in branch_files:
        df = load_feature_file(f, target=target, feature_level=feature_level)

        if len(df) == 0:
            file_rows.append({
                "file": str(f),
                "branch": branch,
                "rows_after_filter": 0,
                "status": "empty_after_target_or_feature_level_filter",
            })
            continue

        df["_source_feature_file"] = str(f)
        frames.append(df)

        file_rows.append({
            "file": str(f),
            "branch": branch,
            "rows_after_filter": len(df),
            "status": "loaded",
        })

    if not frames:
        raise RuntimeError(f"No usable frames for branch {branch}")

    full = pd.concat(frames, ignore_index=True)

    metadata_cols = [
        c for c in [
            "dataset",
            "battery_id",
            "phase_folder",
            "phase",
            "cycle_number",
            "cycle_index",
            "feature_level",
            "_source_feature_file",
        ]
        if c in full.columns
    ]

    y = full[target]

    removed_rows = []
    candidate_cols = []

    for col in full.columns:
        remove, reason = should_remove_column(col, target=target)

        if col in metadata_cols:
            remove = False
            reason = "metadata_kept_for_evaluation"

        if col == target:
            remove = False
            reason = "target_kept_as_y"

        if remove:
            removed_rows.append({
                "branch": branch,
                "column": col,
                "reason": reason,
            })
            continue

        if col in metadata_cols or col == target:
            continue

        candidate_cols.append(col)

    numeric_cols = []
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(full[col]):
            numeric_cols.append(col)
        else:
            removed_rows.append({
                "branch": branch,
                "column": col,
                "reason": "non_numeric_feature",
            })

    kept_features = []
    for col in numeric_cols:
        non_null_rate = float(full[col].notna().mean())
        nunique = full[col].nunique(dropna=True)

        if non_null_rate < min_non_null_rate:
            removed_rows.append({
                "branch": branch,
                "column": col,
                "reason": f"low_non_null_rate_{non_null_rate:.4f}",
            })
            continue

        if nunique <= 1:
            removed_rows.append({
                "branch": branch,
                "column": col,
                "reason": "constant_or_single_value",
            })
            continue

        kept_features.append(col)

    modeling = full[metadata_cols + [target] + kept_features].copy()

    output_root.mkdir(parents=True, exist_ok=True)

    out_parquet = output_root / f"{branch}_{feature_level}_modeling.parquet"
    out_features = output_root / f"{branch}_{feature_level}_model_input_columns.csv"
    out_removed = output_root / f"{branch}_{feature_level}_removed_columns.csv"
    out_file_summary = output_root / f"{branch}_{feature_level}_source_file_summary.csv"
    out_dataset_summary = output_root / f"{branch}_{feature_level}_dataset_summary.csv"

    modeling.to_parquet(out_parquet, index=False)

    pd.DataFrame({
        "branch": branch,
        "feature_level": feature_level,
        "target": target,
        "feature_name": kept_features,
    }).to_csv(out_features, index=False)

    pd.DataFrame(removed_rows).to_csv(out_removed, index=False)
    pd.DataFrame(file_rows).to_csv(out_file_summary, index=False)

    group_cols = [c for c in ["dataset", "phase_folder", "phase"] if c in modeling.columns]
    dataset_summary = (
        modeling
        .groupby(group_cols, dropna=False)
        .agg(
            rows=(target, "size"),
            batteries=("battery_id", "nunique") if "battery_id" in modeling.columns else (target, "size"),
            target_min=(target, "min"),
            target_median=(target, "median"),
            target_mean=(target, "mean"),
            target_max=(target, "max"),
        )
        .reset_index()
    )
    dataset_summary.to_csv(out_dataset_summary, index=False)

    print("\n" + "=" * 80)
    print(f"Branch: {branch}")
    print(f"Output: {out_parquet}")
    print(f"Rows: {len(modeling)}")
    print(f"Metadata columns: {len(metadata_cols)}")
    print(f"Features kept: {len(kept_features)}")
    print(f"Target: {target}")
    print(dataset_summary.to_string(index=False))


def main() -> None:
    args = parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    files = sorted(input_root.rglob("*.features.parquet"))
    files = [f for f in files if "_targets" not in f.parts]

    print("Input root:", input_root)
    print("Output root:", output_root)
    print("Feature files found:", len(files))

    if not files:
        raise RuntimeError("No *.features.parquet files found.")

    for branch in ["charge", "discharge"]:
        build_branch(
            files=files,
            branch=branch,
            target=args.target,
            feature_level=args.feature_level,
            min_non_null_rate=args.min_non_null_rate,
            output_root=output_root,
        )


if __name__ == "__main__":
    main()
