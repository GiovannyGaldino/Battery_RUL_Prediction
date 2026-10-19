from pathlib import Path
import ast
import numpy as np
import pandas as pd

def to_arr(x):
    if x is None:
        return np.array([])
    if isinstance(x, np.ndarray):
        a = x
    elif isinstance(x, (list, tuple)):
        a = np.asarray(x)
    elif isinstance(x, str):
        try:
            a = np.asarray(ast.literal_eval(x))
        except Exception:
            return np.array([])
    else:
        try:
            a = np.asarray(x)
        except Exception:
            return np.array([])
    try:
        return a.astype(float).ravel()
    except Exception:
        return np.array([])

root = Path.home() / "data/batteries/interim"
cols = ["voltage_v", "capacity_ah", "time_s", "current_a", "phase_capacity_ah"]

rows = []

for f in sorted(root.rglob("*.parquet")):
    df = pd.read_parquet(f)
    available = [c for c in cols if c in df.columns]
    sample = df[available].head(200).copy()

    row = {
        "file": str(f.relative_to(root)),
        "rows": len(df),
        "columns": len(df.columns),
        "has_voltage_v": "voltage_v" in df.columns,
        "has_capacity_ah": "capacity_ah" in df.columns,
        "has_time_s": "time_s" in df.columns,
        "has_current_a": "current_a" in df.columns,
        "has_phase_capacity_ah": "phase_capacity_ah" in df.columns,
    }

    for c in ["voltage_v", "capacity_ah", "time_s", "current_a"]:
        if c in sample.columns:
            lens = sample[c].apply(lambda x: len(to_arr(x)))
            ranges = sample[c].apply(lambda x: np.nan if len(to_arr(x)) < 2 else np.nanmax(to_arr(x)) - np.nanmin(to_arr(x)))
            row[f"{c}_median_len"] = float(lens.median())
            row[f"{c}_max_len"] = int(lens.max())
            row[f"{c}_valid_vector_rate"] = float((lens >= 3).mean())
            row[f"{c}_positive_range_rate"] = float((ranges > 0).mean())
        else:
            row[f"{c}_median_len"] = np.nan
            row[f"{c}_max_len"] = np.nan
            row[f"{c}_valid_vector_rate"] = np.nan
            row[f"{c}_positive_range_rate"] = np.nan

    rows.append(row)

out = pd.DataFrame(rows)
print(out.to_string(index=False))

out_path = Path.home() / "outputs/batteries_modeling_v3/interim_curve_input_diagnostic.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
print("\nSaved:", out_path)
