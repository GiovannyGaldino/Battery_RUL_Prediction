from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_DEFAULT = "target_rul80_steps_clipped"

METADATA_COLUMNS = {
    "dataset",
    "battery_id",
    "phase_folder",
    "phase",
    "cycle_number",
    "cycle_index",
    "feature_level",
    "_source_feature_file",
    "_source_curve_file",
}

SAMPLE_RE = re.compile(r"^(VQ|QV)_p(\d{3})_(\d{3})$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="EDA by dataset and systematic sampling redundancy analysis."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.home() / "data/batteries/modeling_from_v3_with_curves",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "outputs/batteries_modeling_v3/eda_by_dataset_curves",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=TARGET_DEFAULT,
    )

    parser.add_argument(
        "--phases",
        nargs="+",
        default=["charge", "discharge"],
        choices=["charge", "discharge"],
    )

    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["all_valid", "pre_eol_only"],
        choices=["all_valid", "pre_eol_only"],
    )

    parser.add_argument(
        "--sample-per-group",
        type=int,
        default=80000,
        help="Maximum rows per dataset/subset for correlation estimates. Use 0 for full data.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[3, 9, 15, 30],
    )

    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_input_file(input_root: Path, phase: str) -> Path:
    candidates = [
        input_root / f"{phase}_phase_modeling.parquet",
        input_root / f"{phase}_phase_modeling_with_curves.parquet",
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"No modeling file found for phase={phase} in {input_root}")


def get_numeric_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    cols = []

    for c in df.columns:
        if c == target:
            continue

        if c in METADATA_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    return cols


def get_direct_sampling_columns(df: pd.DataFrame, family: str | None = None, p: int | None = None) -> list[str]:
    cols = []

    for c in df.columns:
        m = SAMPLE_RE.match(c)
        if not m:
            continue

        fam = m.group(1)
        res = int(m.group(2))

        if family is not None and fam != family:
            continue

        if p is not None and res != p:
            continue

        cols.append(c)

    return sorted(cols, key=lambda x: (x.split("_p")[0], int(x.split("_p")[1].split("_")[0]), int(x.split("_")[-1])))


def sample_df(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if max_rows and max_rows > 0 and len(df) > max_rows:
        return df.sample(max_rows, random_state=random_state)
    return df


def summarize_target(df: pd.DataFrame, target: str, phase: str, subset: str) -> dict:
    out = {
        "phase": phase,
        "subset": subset,
        "dataset": "ALL",
        "rows": len(df),
        "batteries": df["battery_id"].nunique() if "battery_id" in df.columns else np.nan,
        "target_min": df[target].min(),
        "target_q25": df[target].quantile(0.25),
        "target_median": df[target].median(),
        "target_mean": df[target].mean(),
        "target_q75": df[target].quantile(0.75),
        "target_max": df[target].max(),
        "zero_target_fraction": float((df[target] == 0).mean()),
    }
    return out


def compute_feature_target_correlations(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    phase: str,
    subset: str,
    dataset: str,
    sample_per_group: int,
) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    work = sample_df(df[[target] + features].copy(), sample_per_group)

    usable = []
    for c in features:
        s = work[c]
        if s.notna().sum() < 5:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        usable.append(c)

    if not usable:
        return pd.DataFrame()

    work = work[[target] + usable]

    pearson = work[usable].corrwith(work[target], method="pearson")

    ranked = work.rank(method="average")
    spearman = ranked[usable].corrwith(ranked[target], method="pearson")

    out = pd.DataFrame(
        {
            "phase": phase,
            "subset": subset,
            "dataset": dataset,
            "feature": usable,
            "pearson": pearson.reindex(usable).values,
            "spearman": spearman.reindex(usable).values,
        }
    )

    out["abs_pearson"] = out["pearson"].abs()
    out["abs_spearman"] = out["spearman"].abs()
    out["max_abs_corr"] = out[["abs_pearson", "abs_spearman"]].max(axis=1)

    out["is_direct_sampling"] = out["feature"].apply(lambda x: bool(SAMPLE_RE.match(x)))
    out["curve_family"] = out["feature"].apply(lambda x: SAMPLE_RE.match(x).group(1) if SAMPLE_RE.match(x) else "")
    out["resolution"] = out["feature"].apply(lambda x: int(SAMPLE_RE.match(x).group(2)) if SAMPLE_RE.match(x) else np.nan)
    out["sample_index"] = out["feature"].apply(lambda x: int(SAMPLE_RE.match(x).group(3)) if SAMPLE_RE.match(x) else np.nan)

    out = out.sort_values("max_abs_corr", ascending=False)

    return out


def effective_dimension_from_corr(corr: pd.DataFrame) -> tuple[float, float]:
    if corr.shape[0] == 0:
        return np.nan, np.nan

    mat = corr.to_numpy(dtype=float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(mat, 1.0)

    try:
        eigvals = np.linalg.eigvalsh(mat)
        eigvals = np.clip(eigvals, 0, None)
        denom = np.sum(eigvals ** 2)

        if denom <= 0:
            return np.nan, np.nan

        eff = float((np.sum(eigvals) ** 2) / denom)
        eff_fraction = eff / len(eigvals)

        return eff, eff_fraction

    except Exception:
        return np.nan, np.nan


def redundancy_metrics_for_sampling(
    df: pd.DataFrame,
    target: str,
    phase: str,
    subset: str,
    dataset: str,
    family: str,
    p: int,
    sample_per_group: int,
) -> dict:
    cols = get_direct_sampling_columns(df, family=family, p=p)

    base = {
        "phase": phase,
        "subset": subset,
        "dataset": dataset,
        "family": family,
        "resolution": p,
        "n_sampling_features": len(cols),
        "rows": len(df),
        "sampled_rows": min(len(df), sample_per_group) if sample_per_group else len(df),
    }

    if len(cols) < 2 or len(df) < 5:
        return {
            **base,
            "valid": 0,
            "mean_abs_corr": np.nan,
            "median_abs_corr": np.nan,
            "p90_abs_corr": np.nan,
            "max_abs_corr": np.nan,
            "frac_abs_corr_ge_080": np.nan,
            "frac_abs_corr_ge_090": np.nan,
            "frac_abs_corr_ge_095": np.nan,
            "frac_abs_corr_ge_099": np.nan,
            "adjacent_abs_corr_mean": np.nan,
            "adjacent_abs_corr_min": np.nan,
            "adjacent_abs_corr_max": np.nan,
            "effective_dimension": np.nan,
            "effective_dimension_fraction": np.nan,
            "target_max_abs_spearman": np.nan,
            "target_mean_abs_spearman": np.nan,
        }

    work = sample_df(df[[target] + cols].copy(), sample_per_group)

    usable = []
    for c in cols:
        if work[c].notna().sum() >= 5 and work[c].nunique(dropna=True) > 1:
            usable.append(c)

    if len(usable) < 2:
        return {
            **base,
            "valid": 0,
            "mean_abs_corr": np.nan,
            "median_abs_corr": np.nan,
            "p90_abs_corr": np.nan,
            "max_abs_corr": np.nan,
            "frac_abs_corr_ge_080": np.nan,
            "frac_abs_corr_ge_090": np.nan,
            "frac_abs_corr_ge_095": np.nan,
            "frac_abs_corr_ge_099": np.nan,
            "adjacent_abs_corr_mean": np.nan,
            "adjacent_abs_corr_min": np.nan,
            "adjacent_abs_corr_max": np.nan,
            "effective_dimension": np.nan,
            "effective_dimension_fraction": np.nan,
            "target_max_abs_spearman": np.nan,
            "target_mean_abs_spearman": np.nan,
        }

    ranked = work[[target] + usable].rank(method="average")
    corr = ranked[usable].corr(method="pearson")

    mat = corr.to_numpy(dtype=float)
    upper = mat[np.triu_indices_from(mat, k=1)]
    upper = upper[np.isfinite(upper)]
    abs_upper = np.abs(upper)

    adjacent = []
    usable_sorted = sorted(usable, key=lambda x: int(x.split("_")[-1]))

    for a, b in zip(usable_sorted[:-1], usable_sorted[1:]):
        val = corr.loc[a, b]
        if np.isfinite(val):
            adjacent.append(abs(float(val)))

    adjacent = np.asarray(adjacent, dtype=float)

    eff_dim, eff_frac = effective_dimension_from_corr(corr)

    target_corr = ranked[usable].corrwith(ranked[target], method="pearson")
    target_abs = target_corr.abs().dropna()

    return {
        **base,
        "valid": 1,
        "n_sampling_features_usable": len(usable),
        "mean_abs_corr": float(np.mean(abs_upper)) if len(abs_upper) else np.nan,
        "median_abs_corr": float(np.median(abs_upper)) if len(abs_upper) else np.nan,
        "p90_abs_corr": float(np.quantile(abs_upper, 0.90)) if len(abs_upper) else np.nan,
        "max_abs_corr": float(np.max(abs_upper)) if len(abs_upper) else np.nan,
        "frac_abs_corr_ge_080": float(np.mean(abs_upper >= 0.80)) if len(abs_upper) else np.nan,
        "frac_abs_corr_ge_090": float(np.mean(abs_upper >= 0.90)) if len(abs_upper) else np.nan,
        "frac_abs_corr_ge_095": float(np.mean(abs_upper >= 0.95)) if len(abs_upper) else np.nan,
        "frac_abs_corr_ge_099": float(np.mean(abs_upper >= 0.99)) if len(abs_upper) else np.nan,
        "adjacent_abs_corr_mean": float(np.mean(adjacent)) if len(adjacent) else np.nan,
        "adjacent_abs_corr_min": float(np.min(adjacent)) if len(adjacent) else np.nan,
        "adjacent_abs_corr_max": float(np.max(adjacent)) if len(adjacent) else np.nan,
        "effective_dimension": eff_dim,
        "effective_dimension_fraction": eff_frac,
        "target_max_abs_spearman": float(target_abs.max()) if len(target_abs) else np.nan,
        "target_mean_abs_spearman": float(target_abs.mean()) if len(target_abs) else np.nan,
    }


def plot_redundancy_metrics(redundancy: pd.DataFrame, output_root: Path):
    plot_dir = output_root / "plots" / "sampling_redundancy"
    ensure_dir(plot_dir)

    metrics = [
        "mean_abs_corr",
        "adjacent_abs_corr_mean",
        "frac_abs_corr_ge_090",
        "effective_dimension_fraction",
        "target_max_abs_spearman",
    ]

    for phase in sorted(redundancy["phase"].unique()):
        for subset in sorted(redundancy["subset"].unique()):
            for family in sorted(redundancy["family"].unique()):
                part = redundancy[
                    (redundancy["phase"] == phase)
                    & (redundancy["subset"] == subset)
                    & (redundancy["family"] == family)
                    & (redundancy["valid"] == 1)
                ].copy()

                if part.empty:
                    continue

                for metric in metrics:
                    plt.figure(figsize=(10, 6))

                    for dataset, g in part.groupby("dataset"):
                        g = g.sort_values("resolution")
                        plt.plot(g["resolution"], g[metric], marker="o", label=dataset)

                    plt.xlabel("Systematic sampling resolution p")
                    plt.ylabel(metric)
                    plt.title(f"{metric} vs resolution | {phase} | {subset} | {family}")
                    plt.grid(True, alpha=0.3)
                    plt.legend(fontsize=8, ncol=2)
                    plt.tight_layout()

                    out = plot_dir / f"{phase}_{subset}_{family}_{metric}_vs_resolution.png"
                    plt.savefig(out, dpi=200)
                    plt.close()


def plot_top_features(top_corr: pd.DataFrame, output_root: Path, top_n: int):
    plot_dir = output_root / "plots" / "top_feature_target_correlations"
    ensure_dir(plot_dir)

    for (phase, subset, dataset), g in top_corr.groupby(["phase", "subset", "dataset"]):
        g = g.head(top_n).copy()
        if g.empty:
            continue

        g = g.sort_values("abs_spearman")

        plt.figure(figsize=(10, max(6, 0.25 * len(g))))
        plt.barh(g["feature"], g["abs_spearman"])
        plt.xlabel("|Spearman correlation with target|")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} features | {phase} | {subset} | {dataset}")
        plt.tight_layout()

        safe_dataset = str(dataset).replace("/", "_")
        out = plot_dir / f"{phase}_{subset}_{safe_dataset}_top{top_n}_spearman.png"
        plt.savefig(out, dpi=200)
        plt.close()


def main():
    args = parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    ensure_dir(output_root)
    ensure_dir(output_root / "tables")

    all_target_summaries = []
    all_corrs = []
    all_top_corrs = []
    all_sampling_corrs = []
    all_redundancy = []

    for phase in args.phases:
        path = get_input_file(input_root, phase)
        print("\n" + "=" * 100)
        print("Reading:", path)

        df_phase = pd.read_parquet(path)
        print("Shape:", df_phase.shape)

        numeric_features = get_numeric_feature_columns(df_phase, args.target)
        sampling_features = get_direct_sampling_columns(df_phase)

        print("Numeric feature columns:", len(numeric_features))
        print("Direct sampling columns:", len(sampling_features))

        datasets = ["ALL"] + sorted(df_phase["dataset"].dropna().astype(str).unique().tolist())

        for subset in args.subsets:
            if subset == "all_valid":
                df_subset = df_phase.copy()
            elif subset == "pre_eol_only":
                df_subset = df_phase[df_phase[args.target] > 0].copy()
            else:
                raise ValueError(subset)

            print("\n" + "-" * 100)
            print(f"Phase={phase} | subset={subset} | rows={len(df_subset)}")

            for dataset in datasets:
                if dataset == "ALL":
                    df_group = df_subset
                else:
                    df_group = df_subset[df_subset["dataset"].astype(str) == dataset]

                if len(df_group) == 0:
                    continue

                summary = summarize_target(df_group, args.target, phase, subset)
                summary["dataset"] = dataset
                all_target_summaries.append(summary)

                corr = compute_feature_target_correlations(
                    df=df_group,
                    features=numeric_features,
                    target=args.target,
                    phase=phase,
                    subset=subset,
                    dataset=dataset,
                    sample_per_group=args.sample_per_group,
                )

                if not corr.empty:
                    all_corrs.append(corr)

                    top = corr.head(args.top_n).copy()
                    top["rank"] = np.arange(1, len(top) + 1)
                    all_top_corrs.append(top)

                    sampling_corr = corr[corr["is_direct_sampling"]].copy()
                    all_sampling_corrs.append(sampling_corr)

                for family in ["VQ", "QV"]:
                    for p in args.resolutions:
                        red = redundancy_metrics_for_sampling(
                            df=df_group,
                            target=args.target,
                            phase=phase,
                            subset=subset,
                            dataset=dataset,
                            family=family,
                            p=p,
                            sample_per_group=args.sample_per_group,
                        )
                        all_redundancy.append(red)

                print(
                    f"{phase} | {subset} | {dataset} | rows={len(df_group)} | "
                    f"batteries={df_group['battery_id'].nunique() if 'battery_id' in df_group.columns else 'NA'}"
                )

    target_summary = pd.DataFrame(all_target_summaries)
    target_summary.to_csv(
        output_root / "tables" / "target_summary_by_dataset.csv",
        index=False,
    )

    corrs = pd.concat(all_corrs, ignore_index=True) if all_corrs else pd.DataFrame()
    corrs.to_csv(
        output_root / "tables" / "feature_target_correlations_by_dataset.csv",
        index=False,
    )

    top_corrs = pd.concat(all_top_corrs, ignore_index=True) if all_top_corrs else pd.DataFrame()
    top_corrs.to_csv(
        output_root / "tables" / "top_feature_target_correlations_by_dataset.csv",
        index=False,
    )

    sampling_corrs = pd.concat(all_sampling_corrs, ignore_index=True) if all_sampling_corrs else pd.DataFrame()
    sampling_corrs.to_csv(
        output_root / "tables" / "sampling_feature_target_correlations_by_dataset.csv",
        index=False,
    )

    redundancy = pd.DataFrame(all_redundancy)
    redundancy.to_csv(
        output_root / "tables" / "sampling_redundancy_by_dataset.csv",
        index=False,
    )

    plot_redundancy_metrics(redundancy, output_root)

    if not top_corrs.empty:
        plot_top_features(top_corrs, output_root, top_n=min(args.top_n, 40))

    print("\nDone.")
    print("Output root:", output_root)
    print("Main tables:")
    print(output_root / "tables" / "target_summary_by_dataset.csv")
    print(output_root / "tables" / "feature_target_correlations_by_dataset.csv")
    print(output_root / "tables" / "top_feature_target_correlations_by_dataset.csv")
    print(output_root / "tables" / "sampling_feature_target_correlations_by_dataset.csv")
    print(output_root / "tables" / "sampling_redundancy_by_dataset.csv")


if __name__ == "__main__":
    main()
