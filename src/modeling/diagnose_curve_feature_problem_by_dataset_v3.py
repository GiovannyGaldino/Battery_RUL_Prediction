from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_DEFAULT = "target_rul80_steps_clipped"
SAMPLE_RE = re.compile(r"^(VQ|QV)_p(\d{3})_(\d{3})$")
VALID_RE = re.compile(r"^(VQ|QV)_valid_p(\d{3})$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.home() / "data/batteries/modeling_from_v3_with_curves",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "outputs/batteries_modeling_v3/curve_problem_diagnostics",
    )
    parser.add_argument("--target", type=str, default=TARGET_DEFAULT)
    parser.add_argument("--sample-per-group", type=int, default=80000)
    parser.add_argument("--top-n", type=int, default=50)
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
    raise FileNotFoundError(f"No input file for phase={phase}")


def direct_curve_columns(df: pd.DataFrame, family: str | None = None, resolution: int | None = None) -> list[str]:
    cols = []
    for c in df.columns:
        m = SAMPLE_RE.match(c)
        if not m:
            continue

        fam = m.group(1)
        p = int(m.group(2))

        if family is not None and fam != family:
            continue
        if resolution is not None and p != resolution:
            continue

        cols.append(c)

    return sorted(cols, key=lambda x: (x.split("_")[0], int(x.split("_p")[1].split("_")[0]), int(x.split("_")[-1])))


def valid_column(family: str, resolution: int) -> str:
    return f"{family}_valid_p{resolution:03d}"


def sample_df(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows and len(df) > max_rows:
        return df.sample(max_rows, random_state=42)
    return df


def spearman_corr_with_target(df: pd.DataFrame, features: list[str], target: str, sample_per_group: int) -> pd.DataFrame:
    if len(df) < 5 or not features:
        return pd.DataFrame()

    work = sample_df(df[[target] + features].copy(), sample_per_group)

    usable = []
    for c in features:
        if c not in work.columns:
            continue
        if work[c].notna().sum() < 5:
            continue
        if work[c].nunique(dropna=True) <= 1:
            continue
        usable.append(c)

    if not usable:
        return pd.DataFrame()

    ranked = work[[target] + usable].rank(method="average")
    corr = ranked[usable].corrwith(ranked[target], method="pearson")

    out = pd.DataFrame({
        "feature": usable,
        "spearman": corr.reindex(usable).values,
    })
    out["abs_spearman"] = out["spearman"].abs()
    out = out.sort_values("abs_spearman", ascending=False)

    return out


def diagnose_group(df: pd.DataFrame, phase: str, subset: str, dataset: str, target: str, sample_per_group: int, top_n: int):
    rows_summary = []
    rows_feature = []
    rows_corr = []

    target_std = float(df[target].std()) if len(df) else np.nan
    target_nunique = int(df[target].nunique(dropna=True)) if len(df) else 0
    zero_frac = float((df[target] == 0).mean()) if len(df) else np.nan
    n_batteries = int(df["battery_id"].nunique()) if "battery_id" in df.columns else np.nan

    for family in ["VQ", "QV"]:
        for p in [3, 9, 15, 30]:
            cols = direct_curve_columns(df, family=family, resolution=p)
            vcol = valid_column(family, p)

            if not cols:
                rows_summary.append({
                    "phase": phase,
                    "subset": subset,
                    "dataset": dataset,
                    "family": family,
                    "resolution": p,
                    "rows": len(df),
                    "batteries": n_batteries,
                    "target_std": target_std,
                    "target_nunique": target_nunique,
                    "zero_target_fraction": zero_frac,
                    "n_curve_cols": 0,
                    "valid_rate": np.nan,
                    "mean_non_null_rate": np.nan,
                    "median_non_null_rate": np.nan,
                    "median_nunique": np.nan,
                    "median_std": np.nan,
                    "median_range": np.nan,
                    "max_abs_spearman": np.nan,
                    "mean_abs_spearman_top10": np.nan,
                    "n_curve_features_abs_spearman_ge_050": 0,
                    "n_curve_features_abs_spearman_ge_070": 0,
                    "n_curve_features_abs_spearman_ge_080": 0,
                    "problem_flag": "missing_curve_columns",
                })
                continue

            valid_rate = float(df[vcol].mean()) if vcol in df.columns else np.nan

            stats = []
            for c in cols:
                s = df[c]
                non_null = float(s.notna().mean())
                nunique = int(s.nunique(dropna=True))
                std = float(s.std(skipna=True)) if s.notna().any() else np.nan
                cmin = float(s.min(skipna=True)) if s.notna().any() else np.nan
                cmax = float(s.max(skipna=True)) if s.notna().any() else np.nan
                crange = cmax - cmin if np.isfinite(cmin) and np.isfinite(cmax) else np.nan

                stats.append({
                    "phase": phase,
                    "subset": subset,
                    "dataset": dataset,
                    "family": family,
                    "resolution": p,
                    "feature": c,
                    "non_null_rate": non_null,
                    "nunique": nunique,
                    "std": std,
                    "min": cmin,
                    "max": cmax,
                    "range": crange,
                })

            stats_df = pd.DataFrame(stats)
            rows_feature.append(stats_df)

            corr_df = spearman_corr_with_target(df, cols, target, sample_per_group)
            if not corr_df.empty:
                corr_df.insert(0, "resolution", p)
                corr_df.insert(0, "family", family)
                corr_df.insert(0, "dataset", dataset)
                corr_df.insert(0, "subset", subset)
                corr_df.insert(0, "phase", phase)
                rows_corr.append(corr_df)

                max_abs = float(corr_df["abs_spearman"].max())
                mean_top10 = float(corr_df.head(10)["abs_spearman"].mean())
                n_ge_050 = int((corr_df["abs_spearman"] >= 0.50).sum())
                n_ge_070 = int((corr_df["abs_spearman"] >= 0.70).sum())
                n_ge_080 = int((corr_df["abs_spearman"] >= 0.80).sum())
            else:
                max_abs = np.nan
                mean_top10 = np.nan
                n_ge_050 = 0
                n_ge_070 = 0
                n_ge_080 = 0

            median_std = float(stats_df["std"].median(skipna=True))
            median_range = float(stats_df["range"].median(skipna=True))
            median_nunique = float(stats_df["nunique"].median(skipna=True))
            mean_non_null = float(stats_df["non_null_rate"].mean())
            median_non_null = float(stats_df["non_null_rate"].median())

            flags = []

            if len(df) < 100:
                flags.append("few_rows")
            if n_batteries < 3:
                flags.append("few_batteries")
            if target_nunique < 10 or target_std == 0:
                flags.append("low_target_variability")
            if np.isfinite(zero_frac) and zero_frac > 0.80:
                flags.append("many_zero_targets")
            if np.isfinite(valid_rate) and valid_rate < 0.80:
                flags.append("low_curve_validity")
            if np.isfinite(mean_non_null) and mean_non_null < 0.80:
                flags.append("many_nan_curve_features")
            if np.isfinite(median_nunique) and median_nunique <= 3:
                flags.append("near_constant_curve_features")
            if np.isfinite(median_std) and median_std < 1e-8:
                flags.append("very_low_curve_variability")
            if not np.isfinite(max_abs) or max_abs < 0.30:
                flags.append("weak_curve_target_relation")

            problem_flag = ";".join(flags) if flags else "no_obvious_problem"

            rows_summary.append({
                "phase": phase,
                "subset": subset,
                "dataset": dataset,
                "family": family,
                "resolution": p,
                "rows": len(df),
                "batteries": n_batteries,
                "target_std": target_std,
                "target_nunique": target_nunique,
                "zero_target_fraction": zero_frac,
                "n_curve_cols": len(cols),
                "valid_rate": valid_rate,
                "mean_non_null_rate": mean_non_null,
                "median_non_null_rate": median_non_null,
                "median_nunique": median_nunique,
                "median_std": median_std,
                "median_range": median_range,
                "max_abs_spearman": max_abs,
                "mean_abs_spearman_top10": mean_top10,
                "n_curve_features_abs_spearman_ge_050": n_ge_050,
                "n_curve_features_abs_spearman_ge_070": n_ge_070,
                "n_curve_features_abs_spearman_ge_080": n_ge_080,
                "problem_flag": problem_flag,
            })

    summary = pd.DataFrame(rows_summary)
    feature_stats = pd.concat(rows_feature, ignore_index=True) if rows_feature else pd.DataFrame()
    corrs = pd.concat(rows_corr, ignore_index=True) if rows_corr else pd.DataFrame()

    top_corrs = (
        corrs.sort_values(["phase", "subset", "dataset", "family", "resolution", "abs_spearman"], ascending=[True, True, True, True, True, False])
        .groupby(["phase", "subset", "dataset", "family", "resolution"], dropna=False)
        .head(top_n)
        .copy()
        if not corrs.empty else pd.DataFrame()
    )

    return summary, feature_stats, corrs, top_corrs


def main():
    args = parse_args()

    output_root = args.output_root.expanduser().resolve()
    tables_dir = output_root / "tables"
    ensure_dir(tables_dir)

    all_summaries = []
    all_feature_stats = []
    all_corrs = []
    all_top_corrs = []

    for phase in ["charge", "discharge"]:
        path = get_input_file(args.input_root.expanduser().resolve(), phase)
        print("\n" + "=" * 100)
        print("Reading:", path)

        df = pd.read_parquet(path)
        print("shape:", df.shape)

        for subset in ["all_valid", "pre_eol_only"]:
            if subset == "all_valid":
                df_subset = df.copy()
            else:
                df_subset = df[df[args.target] > 0].copy()

            datasets = ["ALL"] + sorted(df_subset["dataset"].dropna().astype(str).unique().tolist())

            for dataset in datasets:
                if dataset == "ALL":
                    df_group = df_subset
                else:
                    df_group = df_subset[df_subset["dataset"].astype(str) == dataset]

                if df_group.empty:
                    continue

                print(f"{phase} | {subset} | {dataset} | rows={len(df_group)}")

                summary, feature_stats, corrs, top_corrs = diagnose_group(
                    df=df_group,
                    phase=phase,
                    subset=subset,
                    dataset=dataset,
                    target=args.target,
                    sample_per_group=args.sample_per_group,
                    top_n=args.top_n,
                )

                all_summaries.append(summary)
                if not feature_stats.empty:
                    all_feature_stats.append(feature_stats)
                if not corrs.empty:
                    all_corrs.append(corrs)
                if not top_corrs.empty:
                    all_top_corrs.append(top_corrs)

    summary_all = pd.concat(all_summaries, ignore_index=True)
    summary_all.to_csv(tables_dir / "curve_problem_summary_by_dataset.csv", index=False)

    feature_stats_all = pd.concat(all_feature_stats, ignore_index=True) if all_feature_stats else pd.DataFrame()
    feature_stats_all.to_csv(tables_dir / "curve_feature_variability_by_dataset.csv", index=False)

    corr_all = pd.concat(all_corrs, ignore_index=True) if all_corrs else pd.DataFrame()
    corr_all.to_csv(tables_dir / "curve_feature_target_correlations_by_dataset.csv", index=False)

    top_all = pd.concat(all_top_corrs, ignore_index=True) if all_top_corrs else pd.DataFrame()
    top_all.to_csv(tables_dir / "top_curve_feature_correlations_by_dataset.csv", index=False)

    print("\nDone.")
    print("Saved:")
    print(tables_dir / "curve_problem_summary_by_dataset.csv")
    print(tables_dir / "curve_feature_variability_by_dataset.csv")
    print(tables_dir / "curve_feature_target_correlations_by_dataset.csv")
    print(tables_dir / "top_curve_feature_correlations_by_dataset.csv")


if __name__ == "__main__":
    main()
