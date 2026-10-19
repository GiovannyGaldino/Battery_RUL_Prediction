from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


TABLE_ROOT = Path.home() / "outputs/batteries_modeling_v3/eda_by_dataset_curves_fixed/tables"
OUT_ROOT = Path.home() / "outputs/batteries_modeling_v3/article_figures_eda_fixed"

DATASET_ORDER = ["CALCE", "HUST", "MATR", "NRW", "OXFORD", "PCOE", "RWTH", "XJTU"]
PHASE_ORDER = ["charge", "discharge"]
SUBSET_ORDER = ["all_valid", "pre_eol_only"]
FAMILY_ORDER = ["VQ", "QV"]
RESOLUTION_ORDER = [3, 9, 15, 30]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, name: str) -> None:
    ensure_dir(OUT_ROOT)
    png = OUT_ROOT / f"{name}.png"
    pdf = OUT_ROOT / f"{name}.pdf"
    svg = OUT_ROOT / f"{name}.svg"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {svg}")


def wrap_label(label: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False))


def phase_label(phase: str) -> str:
    return {"charge": "Charge", "discharge": "Discharge"}.get(phase, str(phase))


def subset_label(subset: str) -> str:
    return {
        "all_valid": "All valid",
        "pre_eol_only": "Pre-EOL only",
    }.get(subset, str(subset))


def feature_family(feature: str) -> str:
    if feature.startswith("VQ_p"):
        return "V(Q) direct"
    if feature.startswith("QV_p"):
        return "Q(V) direct"
    if feature.startswith("VQ_"):
        return "V(Q) statistic"
    if feature.startswith("QV_"):
        return "Q(V) statistic"
    if feature.startswith("capacityh"):
        return "Capacity"
    if feature.startswith("energy"):
        return "Energy"
    if feature.startswith("voltage"):
        return "Voltage"
    if feature.startswith("current"):
        return "Current"
    if feature.startswith("temperature"):
        return "Temperature"
    if feature.startswith("power"):
        return "Power"
    if feature.startswith("integral"):
        return "Integral"
    return "Other"


def add_bar_labels(ax, fmt="{:.0f}", fontsize=8) -> None:
    for container in ax.containers:
        labels = []
        for v in container.datavalues:
            if np.isnan(v) or v == 0:
                labels.append("")
            else:
                labels.append(fmt.format(v))
        ax.bar_label(container, labels=labels, fontsize=fontsize, padding=2)


def figure_1_sample_composition() -> None:
    """
    Figure 1:
    Stacked bar chart showing number of samples by phase, subset and dataset.
    """
    target_summary = pd.read_csv(TABLE_ROOT / "target_summary_by_dataset.csv")

    df = target_summary[
        (target_summary["dataset"] != "ALL")
        & (target_summary["phase"].isin(PHASE_ORDER))
        & (target_summary["subset"].isin(SUBSET_ORDER))
    ].copy()

    df["panel"] = df["phase"].map(phase_label) + "\n" + df["subset"].map(subset_label)

    panel_order = [
        "Charge\nAll valid",
        "Charge\nPre-EOL only",
        "Discharge\nAll valid",
        "Discharge\nPre-EOL only",
    ]

    pivot = (
        df.pivot_table(
            index="panel",
            columns="dataset",
            values="rows",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(panel_order)
        .reindex(columns=DATASET_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10, 5.8))

    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))

    for dataset in DATASET_ORDER:
        values = pivot[dataset].values
        ax.bar(x, values, bottom=bottom, label=dataset)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=10)
    ax.set_ylabel("Number of samples")
    ax.set_title("Modeling sample composition by phase, subset, and dataset")
    ax.legend(
        title="Dataset",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
    )

    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, "fig1_sample_composition_by_phase_subset_dataset")


def figure_2_top_feature_correlations(top_n: int = 12) -> None:
    """
    Figure 2:
    Two panels with top absolute Spearman correlations for charge and discharge,
    using aggregated ALL / pre_eol_only.
    """
    corr = pd.read_csv(TABLE_ROOT / "feature_target_correlations_by_dataset.csv")

    df = corr[
        (corr["subset"] == "pre_eol_only")
        & (corr["dataset"] == "ALL")
        & (corr["phase"].isin(PHASE_ORDER))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=False)

    for ax, phase in zip(axes, PHASE_ORDER):
        part = (
            df[df["phase"] == phase]
            .sort_values("abs_spearman", ascending=False)
            .head(top_n)
            .copy()
        )

        part = part.sort_values("abs_spearman", ascending=True)
        colors_key = part["feature"].map(feature_family)

        # If seaborn is available, use a palette by feature family.
        if HAS_SEABORN:
            palette = dict(
                zip(
                    sorted(colors_key.unique()),
                    sns.color_palette("tab10", n_colors=len(colors_key.unique())),
                )
            )
            colors = colors_key.map(palette)
        else:
            colors = None

        ax.barh(part["feature"].map(lambda x: wrap_label(x, 24)), part["abs_spearman"], color=colors)
        ax.set_title(f"{phase_label(phase)} / Pre-EOL")
        ax.set_xlabel("|Spearman correlation with RUL|")
        ax.set_xlim(0, max(1.0, part["abs_spearman"].max() * 1.08))
        ax.grid(axis="x", alpha=0.25)

        for y, v in enumerate(part["abs_spearman"]):
            ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8)

    # Shared legend by feature type
    if HAS_SEABORN:
        handles = []
        labels = []
        for fam, color in palette.items():
            handles.append(plt.Line2D([0], [0], marker="s", linestyle="", color=color))
            labels.append(fam)
        fig.legend(handles, labels, title="Feature group", loc="lower center", ncol=4, fontsize=8)

    fig.suptitle("Top feature–target associations in the aggregated pre-EOL subset", y=1.02)
    fig.tight_layout(rect=[0, 0.08 if HAS_SEABORN else 0, 1, 1])
    savefig(fig, "fig2_top_spearman_features_charge_discharge_pre_eol")


def figure_3_dataset_curve_heatmap() -> None:
    """
    Figure 3:
    Heatmap of maximum absolute Spearman correlation between direct curve samples
    and pre-EOL RUL for each dataset, phase, and curve family.
    """
    sampling_corr = pd.read_csv(TABLE_ROOT / "sampling_feature_target_correlations_by_dataset.csv")

    df = sampling_corr[
        (sampling_corr["subset"] == "pre_eol_only")
        & (sampling_corr["dataset"] != "ALL")
        & (sampling_corr["curve_family"].isin(FAMILY_ORDER))
        & (sampling_corr["phase"].isin(PHASE_ORDER))
    ].copy()

    # Direct sampling only; this table should already contain direct sampling rows,
    # but keep this guard for safety.
    if "is_direct_sampling" in df.columns:
        df = df[df["is_direct_sampling"] == True].copy()

    grouped = (
        df.groupby(["dataset", "phase", "curve_family"], dropna=False)["abs_spearman"]
        .max()
        .reset_index()
    )

    grouped["column"] = grouped["phase"].map(phase_label) + " " + grouped["curve_family"].map(
        {"VQ": "V(Q)", "QV": "Q(V)"}
    )

    col_order = [
        "Charge V(Q)",
        "Charge Q(V)",
        "Discharge V(Q)",
        "Discharge Q(V)",
    ]

    heat = (
        grouped.pivot_table(
            index="dataset",
            columns="column",
            values="abs_spearman",
            aggfunc="max",
        )
        .reindex(DATASET_ORDER)
        .reindex(columns=col_order)
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    if HAS_SEABORN:
        sns.heatmap(
            heat,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Maximum |Spearman| with RUL"},
            ax=ax,
        )
    else:
        im = ax.imshow(heat.values, vmin=0, vmax=1, aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Maximum |Spearman| with RUL")

        ax.set_xticks(np.arange(len(heat.columns)))
        ax.set_xticklabels(heat.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(heat.index)))
        ax.set_yticklabels(heat.index)

        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                val = heat.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    ax.set_title("Dataset-level informativeness of direct curve samples")
    ax.set_xlabel("Phase and curve family")
    ax.set_ylabel("Dataset")

    fig.tight_layout()
    savefig(fig, "fig3_dataset_curve_family_heatmap_pre_eol")


def figure_4_redundancy_vs_resolution() -> None:
    """
    Figure 4:
    Four-panel plot showing redundancy and informativeness vs sampling resolution.
    Uses aggregated ALL / pre_eol_only.
    """
    red = pd.read_csv(TABLE_ROOT / "sampling_redundancy_by_dataset.csv")

    df = red[
        (red["subset"] == "pre_eol_only")
        & (red["dataset"] == "ALL")
        & (red["family"].isin(FAMILY_ORDER))
        & (red["phase"].isin(PHASE_ORDER))
        & (red["valid"] == 1)
    ].copy()

    df["label"] = df["phase"].map(phase_label) + " " + df["family"].map(
        {"VQ": "V(Q)", "QV": "Q(V)"}
    )

    label_order = ["Charge V(Q)", "Charge Q(V)", "Discharge V(Q)", "Discharge Q(V)"]

    metrics = [
        ("mean_abs_corr", "Mean |ρ| among sampled points"),
        ("adjacent_abs_corr_mean", "Mean adjacent-point |ρ|"),
        ("effective_dimension_fraction", "Effective dimension fraction"),
        ("target_max_abs_spearman", "Maximum |Spearman| with RUL"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    axes = axes.ravel()

    for ax, (metric, ylabel) in zip(axes, metrics):
        for label in label_order:
            part = df[df["label"] == label].sort_values("resolution")
            if part.empty:
                continue
            ax.plot(
                part["resolution"],
                part[metric],
                marker="o",
                linewidth=2,
                label=label,
            )

        ax.set_xticks(RESOLUTION_ORDER)
        ax.set_xlabel("Sampling resolution p")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.25)

        if metric != "effective_dimension_fraction":
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(0, 1.02)

    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Redundancy and informativeness of systematic curve sampling", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig4_sampling_redundancy_vs_resolution_pre_eol")


def figure_5_ablation_design_table() -> None:
    """
    Optional visual table with suggested modeling ablations.
    """
    rows = [
        ["M0", "Global descriptors only", "Baseline with scalar/vector statistics"],
        ["M1", "V(Q) direct samples only", "Voltage-capacity morphology"],
        ["M2", "Q(V) direct samples only", "Capacity-voltage morphology"],
        ["M3", "V(Q) + Q(V) direct samples", "Combined curve representation"],
        ["M4", "Global + direct curve samples", "Complementarity between global and curve features"],
        ["M5", "Global + direct curve samples + curve statistics", "Full feature set"],
    ]

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Experiment", "Feature set", "Purpose"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
        cell.set_linewidth(0.5)

    ax.set_title("Feature-set ablation design for the modeling stage", pad=16)
    fig.tight_layout()
    savefig(fig, "table_feature_set_ablation_design")


def main() -> None:
    ensure_dir(OUT_ROOT)

    required = [
        "target_summary_by_dataset.csv",
        "feature_target_correlations_by_dataset.csv",
        "sampling_feature_target_correlations_by_dataset.csv",
        "sampling_redundancy_by_dataset.csv",
    ]

    missing = [f for f in required if not (TABLE_ROOT / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {TABLE_ROOT}: {missing}"
        )

    print("Input tables:", TABLE_ROOT)
    print("Output figures:", OUT_ROOT)
    print("Seaborn available:", HAS_SEABORN)

    figure_1_sample_composition()
    figure_2_top_feature_correlations(top_n=12)
    figure_3_dataset_curve_heatmap()
    figure_4_redundancy_vs_resolution()
    figure_5_ablation_design_table()

    print("\nDone.")
    print("Figures saved in:", OUT_ROOT)


if __name__ == "__main__":
    main()
