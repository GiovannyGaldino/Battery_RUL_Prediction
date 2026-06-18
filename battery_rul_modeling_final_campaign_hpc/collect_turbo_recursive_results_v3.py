from pathlib import Path
import json
import pandas as pd

campaign_root = Path.home() / "outputs/batteries_modeling_v3/modeling_results_final_campaign_v1"
shards_root = campaign_root / "shards"
summary_dir = campaign_root / "summary_turbo_recursive_v3"
summary_dir.mkdir(parents=True, exist_ok=True)

files_to_collect = {
    "metrics_global.csv": "turbo_metrics_global.csv",
    "metrics_by_dataset.csv": "turbo_metrics_by_dataset.csv",
    "metrics_by_battery.csv": "turbo_metrics_by_battery.csv",
    "metrics_by_rul_bin.csv": "turbo_metrics_by_rul_bin.csv",
    "cv_metrics.csv": "turbo_cv_metrics.csv",
}

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
        elif p.startswith("heldout_") or p == "all_datasets" or p in {"CALCE", "HUST", "MATR", "NRW", "OXFORD", "PCOE", "RWTH", "XJTU"}:
            meta["scope"] = p

    return meta

def collect_file(filename):
    dfs = []
    file_count = 0

    for f in shards_root.rglob(filename):
        file_count += 1
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

    if dfs:
        out = pd.concat(dfs, ignore_index=True)
    else:
        out = pd.DataFrame()

    return out, file_count

summary = {}

for input_name, output_name in files_to_collect.items():
    df, file_count = collect_file(input_name)
    out_path = summary_dir / output_name
    df.to_csv(out_path, index=False)

    summary[input_name] = {
        "output_name": output_name,
        "file_count": int(file_count),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "path": str(out_path),
    }

    print(f"[OK] {input_name}: files={file_count}, rows={len(df)}, output={output_name}")

# Status esperado por quantidade de arquivos
expected_files = {
    "metrics_global.csv": 5616,
    "metrics_by_dataset.csv": 5616,
    "metrics_by_battery.csv": 5616,
    "metrics_by_rul_bin.csv": 5616,
    "cv_metrics.csv": 156,
}

status_rows = []

for filename, expected in expected_files.items():
    observed = summary.get(filename, {}).get("file_count", 0)
    status_rows.append({
        "item": filename,
        "expected_files": expected,
        "observed_files": observed,
        "complete": observed == expected,
        "rows": summary.get(filename, {}).get("rows", 0),
    })

# Status adicional para linhas CV
cv_rows = summary.get("cv_metrics.csv", {}).get("rows", 0)
status_rows.append({
    "item": "cv_metrics_rows",
    "expected_files": 468,
    "observed_files": cv_rows,
    "complete": cv_rows == 468,
    "rows": cv_rows,
})

overall = all(row["complete"] for row in status_rows)
status_rows.append({
    "item": "overall",
    "expected_files": None,
    "observed_files": None,
    "complete": overall,
    "rows": None,
})

status_df = pd.DataFrame(status_rows)
status_df.to_csv(summary_dir / "turbo_recursive_v3_completion_status.csv", index=False)

with open(summary_dir / "turbo_recursive_v3_collection_summary.json", "w") as fp:
    json.dump(
        {
            "summary_dir": str(summary_dir),
            "files": summary,
            "overall_complete": overall,
        },
        fp,
        indent=2,
    )

print()
print("=== STATUS ===")
print(status_df.to_string(index=False))
print()
print(f"[DONE] Summary written to: {summary_dir}")
