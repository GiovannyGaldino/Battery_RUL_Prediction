from __future__ import annotations

from pathlib import Path
import pandas as pd


KEEP_CURVE_DIAGNOSTICS = False   # v3: definido por --keep-curve-diagnostics

KEYS = ["dataset", "battery_id", "phase_folder", "phase", "cycle_index", "cycle_number"]


def load_curve_branch(curve_root: Path, phase: str) -> pd.DataFrame:
    files = sorted((curve_root / phase).rglob("*.curve_features.parquet"))
    if not files:
        raise RuntimeError(f"No curve feature files found for phase={phase}")

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df["_source_curve_file"] = str(f)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def merge_branch(modeling_root: Path, curve_root: Path, output_root: Path, phase: str) -> None:
    modeling_path = modeling_root / f"{phase}_phase_modeling.parquet"
    modeling = pd.read_parquet(modeling_path)

    curves = load_curve_branch(curve_root, phase)

    missing_model_keys = [k for k in KEYS if k not in modeling.columns]
    missing_curve_keys = [k for k in KEYS if k not in curves.columns]

    if missing_model_keys:
        raise RuntimeError(f"Missing keys in modeling {phase}: {missing_model_keys}")
    if missing_curve_keys:
        raise RuntimeError(f"Missing keys in curves {phase}: {missing_curve_keys}")

    dup_model = int(modeling.duplicated(KEYS).sum())
    dup_curves = int(curves.duplicated(KEYS).sum())

    print("\n" + "=" * 100)
    print(f"Phase: {phase}")
    print("modeling rows:", len(modeling))
    print("curve rows:", len(curves))
    print("modeling duplicated keys:", dup_model)
    print("curve duplicated keys:", dup_curves)

    if dup_model > 0:
        raise RuntimeError(f"Duplicated keys in modeling {phase}")
    if dup_curves > 0:
        raise RuntimeError(f"Duplicated keys in curve features {phase}")

    curve_feature_cols = [
        c for c in curves.columns
        if c not in KEYS and (
            c.startswith("VQ_") or
            c.startswith("QV_") or
            c == "_source_curve_file"
        )
    ]
    # v3: remove diagnosticos de curva. Lista identica a do sanitizador v2, que
    # nunca rodou (build_modeling_dataframe_v1.py:1244-1258).
    if not KEEP_CURVE_DIAGNOSTICS:
        import re as _re
        _exact = {"VQ_n_raw_points", "VQ_n_unique_q", "VQ_q_min", "VQ_q_max", "VQ_q_range",
                  "VQ_voltage_min", "VQ_voltage_max", "VQ_voltage_range"}
        _regex = [r"^VQ_valid_p\d+$", r"^QV_valid_p\d+$", r"^VQ_invalid_reason_p\d+$",
                  r".*invalid_reason.*", r".*_reason$"]
        _drop = [c for c in curve_feature_cols
                 if c in _exact or any(_re.match(rx, c) for rx in _regex)]
        if _drop:
            print(f"curve diagnostics removed ({len(_drop)}): {sorted(_drop)}")
        curve_feature_cols = [c for c in curve_feature_cols if c not in _drop]

    curves_reduced = curves[KEYS + curve_feature_cols].copy()

    merged = modeling.merge(
        curves_reduced,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    audit = (
        merged
        .groupby(["dataset", "phase_folder", "_merge"], dropna=False)
        .size()
        .reset_index(name="rows")
    )

    match_rate = float((merged["_merge"] == "both").mean())

    print("match rate:", match_rate)
    print(audit.to_string(index=False))

    output_root.mkdir(parents=True, exist_ok=True)

    audit_path = output_root / f"{phase}_merge_audit_by_dataset.csv"
    out_path = output_root / f"{phase}_phase_modeling_with_curves.parquet"

    audit.to_csv(audit_path, index=False)

    merged = merged.drop(columns=["_merge"])
    merged.to_parquet(out_path, index=False)

    feature_cols = [c for c in merged.columns if c.startswith("VQ_") or c.startswith("QV_")]
    pd.DataFrame({"feature_name": feature_cols}).to_csv(
        output_root / f"{phase}_curve_feature_columns.csv",
        index=False,
    )

    print("saved:", out_path)
    print("shape:", merged.shape)
    print("curve feature cols:", len(feature_cols))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Merge modeling tables with curve features (v3: caminhos por CLI).")
    ap.add_argument("--modeling-root", type=Path, default=Path.home() / "data/batteries/modeling_from_v3")
    ap.add_argument("--curve-root", type=Path, default=Path.home() / "data/batteries/curve_features_v3_fixed")
    ap.add_argument("--output-root", type=Path, default=Path.home() / "data/batteries/modeling_from_v3_with_curves_fixed")
    ap.add_argument("--keep-curve-diagnostics", action="store_true",
                    help="Mantem VQ_valid_p*, QV_valid_p* e afins (reproduz v2).")
    a = ap.parse_args()
    global KEEP_CURVE_DIAGNOSTICS
    KEEP_CURVE_DIAGNOSTICS = bool(a.keep_curve_diagnostics)
    modeling_root, curve_root, output_root = (p.expanduser() for p in (a.modeling_root, a.curve_root, a.output_root))

    for phase in ["charge", "discharge"]:
        merge_branch(modeling_root, curve_root, output_root, phase)


if __name__ == "__main__":
    main()
