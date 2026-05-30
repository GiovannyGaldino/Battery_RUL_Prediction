# -*- coding: utf-8 -*-
"""
Generate Figure 5 for the battery RUL article.

Figure 5. Capacity evolution examples after standardization.
(a) HUST normalized capacity showing early capacity increase;
(b) representative monotonic degradation case;
(c) example with capacity anomaly or non-monotonic behavior.

Default input priority:
1) data/interim/discharge/*_discharge_clean_cycles.parquet
2) data/interim/*/clean/*_discharge_clean_cycles.parquet

Output:
reports/article_figures/figure5_capacity_evolution_examples/figure5_capacity_evolution_examples.png
reports/article_figures/figure5_capacity_evolution_examples/figure5_selected_cases.csv
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_PROJECT_ROOT = Path.cwd()
DEFAULT_DATA_ROOT = DEFAULT_PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "reports" / "article_figures" / "figure5_capacity_evolution_examples"


@dataclass
class CaseSelection:
    panel: str
    dataset: str
    battery_id: str
    reason: str
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 5: normalized capacity examples after standardization.")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--phase", choices=["charge", "discharge"], default="discharge")
    parser.add_argument("--hust-battery", default="9-4", help="Battery fragment for panel (a). Default: 9-4")
    parser.add_argument("--monotonic-battery", default="", help="Optional exact/fragment battery id for panel (b).")
    parser.add_argument("--anomaly-battery", default="", help="Optional exact/fragment battery id for panel (c).")
    parser.add_argument("--initial-window", type=int, default=5, help="Window used for initial capacity normalization.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-format", choices=["png", "pdf", "tiff", "svg"], default="png")
    parser.add_argument("--engine", choices=["auto", "fastparquet", "pyarrow"], default="auto")
    parser.add_argument("--legend-fontsize", type=float, default=11.0)
    parser.add_argument("--axis-fontsize", type=float, default=11.0)
    parser.add_argument("--tick-fontsize", type=float, default=10.0)
    parser.add_argument("--panel-label-fontsize", type=float, default=13.0)
    parser.add_argument("--ymin", type=float, default=0.6)
    parser.add_argument("--ymax", type=float, default=1.15)
    return parser.parse_args()


def choose_engine(engine: str) -> str:
    if engine in {"fastparquet", "pyarrow"}:
        return engine
    try:
        import fastparquet  # noqa: F401
        return "fastparquet"
    except Exception:
        pass
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        pass
    raise RuntimeError("Install fastparquet or pyarrow to read parquet files.")


def vector_to_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, pd.Series):
        arr = value.to_numpy()
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return np.array([], dtype=float)
        try:
            arr = np.asarray(ast.literal_eval(text))
        except Exception:
            text = text.replace(";", ",").replace("[", "").replace("]", "")
            arr = np.asarray([p for p in re.split(r"[,\s]+", text) if p])
    else:
        try:
            if pd.isna(value):
                return np.array([], dtype=float)
        except Exception:
            pass
        arr = np.asarray([value])
    try:
        out = np.asarray(arr, dtype=float).reshape(-1)
    except Exception:
        converted = []
        for item in np.asarray(arr, dtype=object).reshape(-1):
            try:
                converted.append(float(item))
            except Exception:
                continue
        out = np.asarray(converted, dtype=float)
    return out[np.isfinite(out)]


def normalize_text(text: Any) -> str:
    return str(text).lower().replace("_", "-").replace(" ", "")


def find_clean_cycle_files(data_root: Path, phase: str) -> list[Path]:
    interim = data_root / "interim"
    candidates: list[Path] = []
    global_dir = interim / phase
    if global_dir.exists():
        candidates.extend(sorted(global_dir.glob(f"*_{phase}_clean_cycles.parquet")))
        candidates.extend(sorted(global_dir.glob(f"*{phase}*clean*.parquet")))

    # Fallback to per-dataset clean folders.
    candidates.extend(sorted(interim.glob(f"*/clean/*_{phase}_clean_cycles.parquet")))
    candidates.extend(sorted(interim.glob(f"*/clean/*{phase}*clean*.parquet")))

    unique: list[Path] = []
    seen = set()
    for path in candidates:
        key = str(path).lower()
        if key not in seen and path.exists():
            unique.append(path)
            seen.add(key)
    if not unique:
        raise FileNotFoundError(f"No {phase} clean-cycle parquet files found under {interim}")
    return unique


def infer_dataset_from_path(path: Path) -> str:
    name = path.name.lower()
    for token in ["hust", "matr", "calce", "pcoe", "rwth", "xjtu", "oxford", "ora", "rw", "nrw"]:
        if token in name:
            return token.upper()
    return path.stem.split("_")[0].upper()


def load_all_cycles(data_root: Path, phase: str, engine: str) -> pd.DataFrame:
    files = find_clean_cycle_files(data_root, phase)
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_parquet(path, engine=engine)
        except Exception as exc:
            print(f"Warning: could not read {path}: {exc}")
            continue
        if df.empty:
            continue
        if "dataset" not in df.columns:
            df["dataset"] = infer_dataset_from_path(path)
        if "phase" not in df.columns:
            df["phase"] = phase
        if "source_parquet" not in df.columns:
            df["source_parquet"] = str(path)
        if "feature_level" in df.columns:
            df = df[df["feature_level"].astype(str).str.lower().eq("phase")].copy()
        if "phase" in df.columns:
            df = df[df["phase"].astype(str).str.lower().eq(phase)].copy()
        frames.append(df)
    if not frames:
        raise RuntimeError("No readable, non-empty clean-cycle files were found.")
    out = pd.concat(frames, ignore_index=True, sort=False)
    required = {"battery_id", "cycle_number"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    return out


def scalar_float(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return np.nan
    try:
        value = row[col]
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = vector_to_float_array(value)
            return float(arr[-1]) if len(arr) else np.nan
        return float(value)
    except Exception:
        return np.nan


def get_phase_capacity(row: pd.Series) -> float:
    for col in ["phase_capacity_ah", "target_capacity_ah", "input_phase_capacity_ah", "QDischarge", "QCharge"]:
        val = scalar_float(row, col)
        if np.isfinite(val) and val > 0:
            return val
    arr = vector_to_float_array(row.get("capacity_ah", None))
    if len(arr):
        val = float(arr[-1])
        if np.isfinite(val) and val > 0:
            return val
    return np.nan


def battery_series(df: pd.DataFrame, dataset: str, battery_id: str, initial_window: int) -> pd.DataFrame:
    work = df[(df["dataset"].astype(str) == str(dataset)) & (df["battery_id"].astype(str) == str(battery_id))].copy()
    work["cycle_number"] = pd.to_numeric(work["cycle_number"], errors="coerce")
    work = work[np.isfinite(work["cycle_number"])].sort_values("cycle_number")
    if work.empty:
        raise ValueError(f"No rows for {dataset} / {battery_id}")
    work["capacity_ah_selected"] = work.apply(get_phase_capacity, axis=1).astype(float)
    work = work[np.isfinite(work["capacity_ah_selected"]) & (work["capacity_ah_selected"] > 0)].copy()
    if work.empty:
        raise ValueError(f"No valid capacity values for {dataset} / {battery_id}")
    n0 = min(max(initial_window, 1), len(work))
    initial_capacity = float(work["capacity_ah_selected"].iloc[:n0].median())
    if not np.isfinite(initial_capacity) or initial_capacity <= 0:
        initial_capacity = float(work["capacity_ah_selected"].iloc[0])
    work["capacity_norm"] = work["capacity_ah_selected"] / initial_capacity
    return work


def all_battery_stats(df: pd.DataFrame, initial_window: int) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "battery_id"]
    for (dataset, battery_id), group in df.groupby(group_cols, dropna=False):
        try:
            s = battery_series(df, str(dataset), str(battery_id), initial_window)
        except Exception:
            continue
        if len(s) < 10:
            continue
        y = s["capacity_norm"].to_numpy(dtype=float)
        x = s["cycle_number"].to_numpy(dtype=float)
        dy = np.diff(y)
        early_n = min(max(5, len(y) // 10), len(y))
        early_increase = float(np.nanmax(y[:early_n]) - y[0]) if early_n > 1 else 0.0
        positive_step_rate = float(np.mean(dy > 0)) if len(dy) else np.nan
        large_jump_rate = float(np.mean(np.abs(dy) > 0.03)) if len(dy) else np.nan
        corr = float(pd.Series(x).corr(pd.Series(y), method="spearman")) if len(y) > 2 else np.nan
        slope = float(np.polyfit(x, y, 1)[0]) if len(y) > 2 and np.nanstd(x) > 0 else np.nan
        final_norm = float(y[-1])
        monotonic_score = -positive_step_rate - max(0.0, early_increase) + max(0.0, -corr if np.isfinite(corr) else 0.0)
        anomaly_score = positive_step_rate + large_jump_rate + max(0.0, early_increase) + float(np.nanstd(dy) if len(dy) else 0.0)
        rows.append({
            "dataset": str(dataset),
            "battery_id": str(battery_id),
            "n_cycles": int(len(s)),
            "first_cycle": float(x[0]),
            "last_cycle": float(x[-1]),
            "initial_capacity_ah": float(s["capacity_ah_selected"].iloc[:min(initial_window, len(s))].median()),
            "final_capacity_norm": final_norm,
            "early_capacity_increase": early_increase,
            "positive_step_rate": positive_step_rate,
            "large_jump_rate": large_jump_rate,
            "spearman_cycle_capacity": corr,
            "linear_slope_per_cycle": slope,
            "monotonic_score": monotonic_score,
            "anomaly_score": anomaly_score,
        })
    return pd.DataFrame(rows)


def select_by_fragment(df: pd.DataFrame, fragment: str, dataset_filter: str | None = None) -> tuple[str, str]:
    target = normalize_text(fragment)
    work = df.copy()
    if dataset_filter:
        work = work[work["dataset"].astype(str).str.lower().eq(dataset_filter.lower())]
    mask = work["battery_id"].map(lambda x: target in normalize_text(x))
    found = work.loc[mask, ["dataset", "battery_id"]].drop_duplicates()
    if found.empty:
        available = work[["dataset", "battery_id"]].drop_duplicates().head(30).to_dict("records")
        raise ValueError(f"No battery matching fragment '{fragment}'. First available: {available}")
    row = found.iloc[0]
    return str(row["dataset"]), str(row["battery_id"])


def choose_cases(df: pd.DataFrame, stats: pd.DataFrame, args: argparse.Namespace) -> list[CaseSelection]:
    hust_dataset, hust_battery = select_by_fragment(df, args.hust_battery, dataset_filter="HUST")

    if args.monotonic_battery.strip():
        mono_dataset, mono_battery = select_by_fragment(df, args.monotonic_battery)
        mono_reason = "user-selected representative monotonic case"
        mono_score = np.nan
    else:
        candidates = stats.copy()
        # Prefer clearly decreasing trajectories with few positive steps and enough length.
        candidates = candidates[
            (candidates["n_cycles"] >= 30) &
            (candidates["final_capacity_norm"] < 0.95) &
            (candidates["positive_step_rate"] <= 0.20) &
            (candidates["spearman_cycle_capacity"] < -0.75)
        ].copy()
        if candidates.empty:
            candidates = stats[stats["n_cycles"] >= 30].copy()
        candidates = candidates.sort_values(["positive_step_rate", "spearman_cycle_capacity", "n_cycles"], ascending=[True, True, False])
        row = candidates.iloc[0]
        mono_dataset, mono_battery = str(row["dataset"]), str(row["battery_id"])
        mono_reason = "automatic: low positive-step rate and strong decreasing trend"
        mono_score = float(row.get("monotonic_score", np.nan))

    if args.anomaly_battery.strip():
        anom_dataset, anom_battery = select_by_fragment(df, args.anomaly_battery)
        anom_reason = "user-selected anomaly/non-monotonic case"
        anom_score = np.nan
    else:
        candidates = stats.copy()
        # Avoid selecting the same as panel (a) unless unavoidable.
        candidates = candidates[~((candidates["dataset"].astype(str) == hust_dataset) & (candidates["battery_id"].astype(str) == hust_battery))]
        candidates = candidates[candidates["n_cycles"] >= 30].copy()
        candidates = candidates.sort_values(["anomaly_score", "positive_step_rate", "early_capacity_increase"], ascending=False)
        row = candidates.iloc[0]
        anom_dataset, anom_battery = str(row["dataset"]), str(row["battery_id"])
        anom_reason = "automatic: high positive-step/jump/early-increase score"
        anom_score = float(row.get("anomaly_score", np.nan))

    return [
        CaseSelection("a", hust_dataset, hust_battery, "HUST normalized capacity with early capacity increase", np.nan),
        CaseSelection("b", mono_dataset, mono_battery, mono_reason, mono_score),
        CaseSelection("c", anom_dataset, anom_battery, anom_reason, anom_score),
    ]


def plot_cases(df: pd.DataFrame, cases: list[CaseSelection], args: argparse.Namespace, output_path: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), sharey=True, constrained_layout=True)
    panel_titles = {
        "a": "Early capacity increase",
        "b": "Monotonic degradation",
        "c": "Non-monotonic behavior",
    }
    info_rows = []

    for ax, case in zip(axes, cases):
        s = battery_series(df, case.dataset, case.battery_id, args.initial_window)
        ax.plot(s["cycle_number"], s["capacity_norm"], linewidth=1.8, label=f"{case.dataset} {case.battery_id}")
        ax.axhline(0.8, linestyle="--", linewidth=1.0, alpha=0.7, label="80% EOL threshold")
        ax.text(0.02, 0.95, f"({case.panel})", transform=ax.transAxes, fontsize=args.panel_label_fontsize,
                fontweight="bold", va="top", ha="left")
        # Subtitle inside the axes, not as a top figure title.
        ax.text(0.02, 0.86, panel_titles[case.panel], transform=ax.transAxes, fontsize=args.axis_fontsize,
                va="top", ha="left")
        ax.set_xlabel("Cycle number", fontsize=args.axis_fontsize)
        ax.tick_params(axis="both", labelsize=args.tick_fontsize)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(args.ymin, args.ymax)
        ax.legend(frameon=False, fontsize=args.legend_fontsize, loc="best")

        y = s["capacity_norm"].to_numpy(dtype=float)
        dy = np.diff(y)
        info_rows.append({
            "panel": case.panel,
            "dataset": case.dataset,
            "battery_id": case.battery_id,
            "reason": case.reason,
            "score": case.score,
            "n_cycles": int(len(s)),
            "first_cycle": float(s["cycle_number"].iloc[0]),
            "last_cycle": float(s["cycle_number"].iloc[-1]),
            "initial_capacity_ah": float(s["capacity_ah_selected"].iloc[:min(args.initial_window, len(s))].median()),
            "max_capacity_norm": float(np.nanmax(y)),
            "final_capacity_norm": float(y[-1]),
            "early_capacity_increase": float(np.nanmax(y[:min(max(5, len(y)//10), len(y))]) - y[0]),
            "positive_step_rate": float(np.mean(dy > 0)) if len(dy) else np.nan,
            "large_jump_rate_abs_gt_0p03": float(np.mean(np.abs(dy) > 0.03)) if len(dy) else np.nan,
        })

    axes[0].set_ylabel("Normalized capacity", fontsize=args.axis_fontsize)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(info_rows)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    engine = choose_engine(args.engine)
    df = load_all_cycles(data_root, args.phase, engine)
    stats = all_battery_stats(df, args.initial_window)
    if stats.empty:
        raise RuntimeError("Could not compute battery-level capacity statistics.")

    cases = choose_cases(df, stats, args)
    output_path = output_root / f"figure5_capacity_evolution_examples.{args.fig_format}"
    selected_info = plot_cases(df, cases, args, output_path)

    selected_info_path = output_root / "figure5_selected_cases.csv"
    stats_path = output_root / "figure5_candidate_stats.csv"
    selected_info.to_csv(selected_info_path, index=False, encoding="utf-8-sig")
    stats.sort_values("anomaly_score", ascending=False).to_csv(stats_path, index=False, encoding="utf-8-sig")

    print("Figure 5 generated.")
    print(f"Output figure: {output_path}")
    print(f"Selected cases: {selected_info_path}")
    print(f"Candidate statistics: {stats_path}")
    for row in selected_info.itertuples(index=False):
        print(f"Panel ({row.panel}): {row.dataset} {row.battery_id} | cycles {int(round(row.first_cycle))}-{int(round(row.last_cycle))} | {row.reason}")


if __name__ == "__main__":
    main()
