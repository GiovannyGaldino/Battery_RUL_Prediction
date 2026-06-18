from pathlib import Path
import json
import pandas as pd

campaign_root = Path.home() / "outputs/batteries_modeling_v3/modeling_results_final_campaign_v1"
shards_root = campaign_root / "shards"
summary_dir = campaign_root / "summary_turbo_recursive"
summary_dir.mkdir(parents=True, exist_ok=True)

metric_files = [
    "metrics_global.csv",
    "metrics_dataset.csv",
    "metrics_battery.csv",
    "metrics_rul_bin.csv",
    "metrics_macro_by_dataset.csv",
]

experiments = {
    "all_features",
    "no_curves",
    "global_stats_only",
    "curves_only",
    "capacity_only",
    "energy_added",
    "no_temperature",
    "no_time_sampling",
    "no_stage_gap",
    "resolution_p003",
    "resolution_p009",
    "resolution_p015",
    "resolution_p030",
}

models = {
    "median",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "mlp",
}

def infer_metadata(path: Path):
    parts = list(path.parts)

    meta = {
        "source_file": str(path),
        "campaign_component": None,
        "random_state": None,
        "phase": None,
        "mode": None,
        "scope": None,
        "experiment": None,
        "model": None,
        "metric_file": path.name,
    }

    if "shards" in parts:
        i = parts.index("shards")
        if len(parts) > i + 1:
            meta["campaign_component"] = parts[i + 1]
        if len(parts) > i + 2 and parts[i + 2].startswith("seed_"):
            meta["random_state"] = parts[i + 2].replace("seed_", "")

    for phase in ["charge", "discharge"]:
        if phase in parts:
            meta["phase"] = phase
            break

    for mode in ["global", "per_dataset", "lodo"]:
        if mode in parts:
            meta["mode"] = mode
            break

    for p in parts:
        if p in experiments:
            meta["experiment"] = p
        elif p in models:
            meta["model"] = p
        elif p.startswith("heldout_") or p == "all_datasets":
            meta["scope"] = p

    return meta

def collect_file(filename):
    dfs = []

    for f in shards_root.rglob(filename):
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] erro lendo {f}: {e}")
            continue

        meta = infer_metadata(f)

        for k, v in meta.items():
            if k not in df.columns:
                df[k] = v
            else:
                df[f"path_{k}"] = v

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

summary = {}

for filename in metric_files:
    df = collect_file(filename)
    out = summary_dir / f"turbo_recursive_{filename}"
    df.to_csv(out, index=False)
    summary[filename] = {"rows": int(len(df)), "path": str(out)}
    print(f"[OK] {filename}: {len(df)} rows")

cv = collect_file("cv_metrics.csv")
cv_out = summary_dir / "turbo_recursive_cv_metrics.csv"
cv.to_csv(cv_out, index=False)
summary["cv_metrics.csv"] = {"rows": int(len(cv)), "path": str(cv_out)}
print(f"[OK] cv_metrics.csv: {len(cv)} rows")

global_df = collect_file("metrics_global.csv")

expected = {
    "metrics_global_rows": 5616,
    "cv_metrics_rows": 468,
    "full_matrix_metrics_global": 2652,
    "repeated_split_metrics_global": 2808,
    "internal_groupkfold_metrics_global": 156,
}

component_counts = {}
if not global_df.empty and "campaign_component" in global_df.columns:
    component_counts = global_df["campaign_component"].value_counts(dropna=False).to_dict()

status_rows = [
    {
        "item": "metrics_global_rows",
        "expected": expected["metrics_global_rows"],
        "observed": int(len(global_df)),
        "complete": int(len(global_df)) == expected["metrics_global_rows"],
    },
    {
        "item": "cv_metrics_rows",
        "expected": expected["cv_metrics_rows"],
        "observed": int(len(cv)),
        "complete": int(len(cv)) == expected["cv_metrics_rows"],
    },
    {
        "item": "full_matrix_metrics_global",
        "expected": expected["full_matrix_metrics_global"],
        "observed": int(component_counts.get("full_matrix", 0)),
        "complete": int(component_counts.get("full_matrix", 0)) == expected["full_matrix_metrics_global"],
    },
    {
        "item": "repeated_split_metrics_global",
        "expected": expected["repeated_split_metrics_global"],
        "observed": int(component_counts.get("repeated_split", 0)),
        "complete": int(component_counts.get("repeated_split", 0)) == expected["repeated_split_metrics_global"],
    },
    {
        "item": "internal_groupkfold_metrics_global",
        "expected": expected["internal_groupkfold_metrics_global"],
        "observed": int(component_counts.get("internal_groupkfold", 0)),
        "complete": int(component_counts.get("internal_groupkfold", 0)) == expected["internal_groupkfold_metrics_global"],
    },
]

overall = all(row["complete"] for row in status_rows)
status_rows.append({
    "item": "overall",
    "expected": None,
    "observed": None,
    "complete": overall,
})

status_df = pd.DataFrame(status_rows)
status_df.to_csv(summary_dir / "turbo_recursive_completion_status.csv", index=False)

with open(summary_dir / "turbo_recursive_collection_summary.json", "w") as fp:
    json.dump(
        {
            "summary_dir": str(summary_dir),
            "files": summary,
            "expected": expected,
            "component_counts_metrics_global": component_counts,
            "overall_complete": overall,
        },
        fp,
        indent=2,
    )

print()
print("=== COMPONENT COUNTS metrics_global.csv ===")
print(component_counts)

print()
print("=== STATUS ===")
print(status_df.to_string(index=False))

print()
print(f"[DONE] Summary written to: {summary_dir}")
