from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET = "target_rul80_steps_clipped"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.home() / "data/batteries/modeling_from_v3",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "outputs/batteries_modeling_v3/eda",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--sample-for-spearman",
        type=int,
        default=120000,
        help="Maximum rows used for Spearman correlation. Use 0 for full dataframe.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    metadata = {
        "dataset",
        "battery_id",
        "phase_folder",
        "phase",
        "cycle_number",
        "cycle_index",
        "feature_level",
        "_source_feature_file",
        target,
    }

    cols = []
    for c in df.columns:
        if c in metadata:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def summarize_dataset(df: pd.DataFrame, target: str, outdir: Path, label: str):
    group_cols = [c for c in ["dataset", "phase_folder", "phase"] if c in df.columns]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows=(target, "size"),
            batteries=("battery_id", "nunique"),
            target_min=(target, "min"),
            target_q25=(target, lambda x: x.quantile(0.25)),
            target_median=(target, "median"),
            target_mean=(target, "mean"),
            target_q75=(target, lambda x: x.quantile(0.75)),
            target_max=(target, "max"),
            zero_target_fraction=(target, lambda x: float((x == 0).mean())),
        )
        .reset_index()
    )

    summary.to_csv(outdir / f"{label}_dataset_target_summary.csv", index=False)

    return summary


def feature_completeness(df: pd.DataFrame, features: list[str], outdir: Path, label: str):
    rows = []
    for c in features:
        s = df[c]
        rows.append(
            {
                "feature": c,
                "non_null_rate": float(s.notna().mean()),
                "nan_rate": float(s.isna().mean()),
                "nunique": int(s.nunique(dropna=True)),
                "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
                "std": float(s.std(skipna=True)) if s.notna().any() else np.nan,
                "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
                "median": float(s.median(skipna=True)) if s.notna().any() else np.nan,
                "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
            }
        )

    comp = pd.DataFrame(rows).sort_values(["nan_rate", "feature"])
    comp.to_csv(outdir / f"{label}_feature_completeness.csv", index=False)
    return comp


def corr_with_target(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    outdir: Path,
    label: str,
    sample_for_spearman: int,
):
    work = df[[target] + features].copy()

    pearson = work.corr(method="pearson", numeric_only=True)[target].drop(target)

    if sample_for_spearman and len(work) > sample_for_spearman:
        work_s = work.sample(sample_for_spearman, random_state=42)
    else:
        work_s = work

    spearman = work_s.corr(method="spearman", numeric_only=True)[target].drop(target)

    corr = pd.DataFrame(
        {
            "feature": pearson.index,
            "pearson": pearson.values,
            "spearman": spearman.reindex(pearson.index).values,
        }
    )

    corr["abs_pearson"] = corr["pearson"].abs()
    corr["abs_spearman"] = corr["spearman"].abs()
    corr["max_abs_corr"] = corr[["abs_pearson", "abs_spearman"]].max(axis=1)

    corr = corr.sort_values("max_abs_corr", ascending=False)

    corr.to_csv(outdir / f"{label}_feature_target_correlations.csv", index=False)

    return corr


def plot_top_correlations(corr: pd.DataFrame, outdir: Path, label: str, top_n: int):
    top = corr.head(top_n).copy()
    if top.empty:
        return

    # Plot Spearman absolute correlation
    plot_df = top.sort_values("abs_spearman", ascending=True)

    plt.figure(figsize=(10, max(6, 0.28 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["abs_spearman"])
    plt.xlabel("|Spearman correlation with target|")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} features by Spearman correlation - {label}")
    plt.tight_layout()
    plt.savefig(outdir / f"{label}_top{top_n}_spearman_corr.png", dpi=200)
    plt.close()

    # Plot Pearson absolute correlation
    plot_df = top.sort_values("abs_pearson", ascending=True)

    plt.figure(figsize=(10, max(6, 0.28 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["abs_pearson"])
    plt.xlabel("|Pearson correlation with target|")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} features by Pearson correlation - {label}")
    plt.tight_layout()
    plt.savefig(outdir / f"{label}_top{top_n}_pearson_corr.png", dpi=200)
    plt.close()


def plot_target_distribution(df: pd.DataFrame, target: str, outdir: Path, label: str):
    plt.figure(figsize=(9, 5))
    plt.hist(df[target].dropna(), bins=80)
    plt.xlabel(target)
    plt.ylabel("Frequency")
    plt.title(f"Target distribution - {label}")
    plt.tight_layout()
    plt.savefig(outdir / f"{label}_target_distribution.png", dpi=200)
    plt.close()

    if "dataset" in df.columns:
        zero = (
            df.assign(is_zero=(df[target] == 0).astype(float))
            .groupby("dataset", dropna=False)["is_zero"]
            .mean()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(9, 5))
        plt.bar(zero.index.astype(str), zero.values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Fraction of target = 0")
        plt.title(f"Zero-RUL fraction by dataset - {label}")
        plt.tight_layout()
        plt.savefig(outdir / f"{label}_zero_target_fraction_by_dataset.png", dpi=200)
        plt.close()


def feature_redundancy(df: pd.DataFrame, top_features: list[str], outdir: Path, label: str):
    top_features = [c for c in top_features if c in df.columns]
    if len(top_features) < 2:
        return

    corr = df[top_features].corr(method="spearman", numeric_only=True)

    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append(
                {
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "spearman_corr": corr.iloc[i, j],
                    "abs_spearman_corr": abs(corr.iloc[i, j]),
                }
            )

    pairs_df = pd.DataFrame(pairs).sort_values(
        "abs_spearman_corr", ascending=False
    )
    pairs_df.to_csv(outdir / f"{label}_top_feature_redundancy_pairs.csv", index=False)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar(label="Spearman correlation")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title(f"Redundancy among top features - {label}")
    plt.tight_layout()
    plt.savefig(outdir / f"{label}_top_feature_redundancy_heatmap.png", dpi=200)
    plt.close()


def run_one(df: pd.DataFrame, branch: str, subset_name: str, args):
    label = f"{branch}_{subset_name}"
    outdir = args.output_root / branch / subset_name
    ensure_dir(outdir)

    print("\n" + "=" * 100)
    print(label)
    print("shape:", df.shape)

    features = get_feature_columns(df, args.target)
    print("numeric feature columns:", len(features))

    summarize_dataset(df, args.target, outdir, label)
    feature_completeness(df, features, outdir, label)

    plot_target_distribution(df, args.target, outdir, label)

    corr = corr_with_target(
        df=df,
        features=features,
        target=args.target,
        outdir=outdir,
        label=label,
        sample_for_spearman=args.sample_for_spearman,
    )

    plot_top_correlations(corr, outdir, label, args.top_n)

    top_features = corr.head(min(args.top_n, 30))["feature"].tolist()
    feature_redundancy(df, top_features, outdir, label)

    print("Top correlations:")
    print(corr.head(15)[["feature", "pearson", "spearman", "max_abs_corr"]].to_string(index=False))


def main():
    args = parse_args()
    ensure_dir(args.output_root)

    files = {
        "charge": args.input_root / "charge_phase_modeling.parquet",
        "discharge": args.input_root / "discharge_phase_modeling.parquet",
    }

    global_rows = []

    for branch, path in files.items():
        print("\nReading:", path)
        df = pd.read_parquet(path)

        run_one(df, branch, "all_valid", args)

        pre = df[df[args.target] > 0].copy()
        run_one(pre, branch, "pre_eol_only", args)

        global_rows.append(
            {
                "branch": branch,
                "rows_all_valid": len(df),
                "rows_pre_eol_only": len(pre),
                "pre_eol_fraction": len(pre) / len(df) if len(df) else np.nan,
                "features_total": len(get_feature_columns(df, args.target)),
            }
        )

    pd.DataFrame(global_rows).to_csv(
        args.output_root / "eda_global_summary.csv", index=False
    )

    print("\nDone.")
    print("EDA output:", args.output_root)


if __name__ == "__main__":
    main()
