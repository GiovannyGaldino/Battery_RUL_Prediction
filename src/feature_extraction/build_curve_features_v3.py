from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


KEY_COLUMNS = [
    "dataset",
    "battery_id",
    "phase_folder",
    "phase",
    "cycle_index",
    "cycle_number",
]

OPTIONAL_ID_COLUMNS = [
    "source_file",
    "source_path",
    "source_cycle_id",
    "cycle_pair_id",
]

VECTOR_COLUMNS = [
    "voltage_v",
    "capacity_ah",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract systematic V(Q) and Q(V) curve features from clean interim "
            "battery cycle files."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.home() / "data/batteries/interim",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "data/batteries/curve_features_v3",
    )

    parser.add_argument(
        "--phases",
        nargs="+",
        default=["charge", "discharge"],
        choices=["charge", "discharge"],
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset filename prefixes, e.g. matr hust calce.",
    )

    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[3, 9, 15, 30],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke testing.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs even if .done exists.",
    )

    return parser.parse_args()


def count_parquet_rows(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        try:
            return int(len(pd.read_parquet(path, columns=[])))
        except Exception:
            return None


def vector_to_array(value) -> np.ndarray:
    """
    Robust converter for vector cells stored in parquet.

    Handles:
    - Python lists/tuples
    - numpy arrays
    - pandas Series
    - pyarrow scalar/list-like objects with .as_py()
    - strings representing lists
    - object arrays containing nested arrays/lists
    """
    if value is None:
        return np.array([], dtype=float)

    # pyarrow scalar/list scalar
    if hasattr(value, "as_py"):
        try:
            value = value.as_py()
        except Exception:
            return np.array([], dtype=float)

    # pandas Series
    if isinstance(value, pd.Series):
        value = value.to_numpy()

    # bytes
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return np.array([], dtype=float)

    # string representation of a list
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return np.array([], dtype=float)

        try:
            parsed = ast.literal_eval(value)
            return vector_to_array(parsed)
        except Exception:
            # fallback for comma/space separated numeric strings
            try:
                arr = np.fromstring(value.replace("[", "").replace("]", ""), sep=",")
                if arr.size == 0:
                    arr = np.fromstring(value.replace("[", "").replace("]", ""), sep=" ")
                return arr.astype(float).ravel()
            except Exception:
                return np.array([], dtype=float)

    # standard array conversion
    try:
        arr = np.asarray(value, dtype=object)
    except Exception:
        return np.array([], dtype=float)

    # unwrap 0-D object arrays
    safety = 0
    while isinstance(arr, np.ndarray) and arr.ndim == 0 and safety < 5:
        safety += 1
        try:
            item = arr.item()
        except Exception:
            break

        if hasattr(item, "as_py"):
            try:
                item = item.as_py()
            except Exception:
                return np.array([], dtype=float)

        if isinstance(item, str):
            return vector_to_array(item)

        if isinstance(item, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(item, dtype=object)
            continue

        try:
            return np.asarray([item], dtype=float)
        except Exception:
            return np.array([], dtype=float)

    # flatten object arrays that may contain nested arrays/lists
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        flat = []

        def collect(obj):
            if hasattr(obj, "as_py"):
                try:
                    obj = obj.as_py()
                except Exception:
                    return

            if isinstance(obj, pd.Series):
                obj = obj.to_numpy()

            if isinstance(obj, np.ndarray):
                if obj.ndim == 0:
                    try:
                        collect(obj.item())
                    except Exception:
                        return
                else:
                    for item in obj:
                        collect(item)
                return

            if isinstance(obj, (list, tuple)):
                for item in obj:
                    collect(item)
                return

            flat.append(obj)

        collect(arr)

        try:
            arr = np.asarray(flat, dtype=float).ravel()
        except Exception:
            return np.array([], dtype=float)

        return arr[np.isfinite(arr)]

    try:
        arr = arr.astype(float).ravel()
    except Exception:
        return np.array([], dtype=float)

    return arr[np.isfinite(arr)]

def aggregate_duplicate_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    unique_x, start_idx, counts = np.unique(
        xs,
        return_index=True,
        return_counts=True,
    )

    sum_y = np.add.reduceat(ys, start_idx)
    mean_y = sum_y / counts

    return unique_x, mean_y


def empty_curve_features(prefix: str, p: int) -> dict:
    out = {f"{prefix}_valid_p{p:03d}": 0}

    for i in range(1, p + 1):
        out[f"{prefix}_p{p:03d}_{i:03d}"] = np.nan

    stat_names = [
        "x_min",
        "x_max",
        "x_range",
        "y_min",
        "y_max",
        "y_range",
        "area",
        "slope_mean",
        "slope_std",
        "slope_min",
        "slope_max",
        "slope_max_abs",
        "inflection_count",
    ]

    for name in stat_names:
        out[f"{prefix}_{name}_p{p:03d}"] = np.nan

    return out


def curve_features(x_raw, y_raw, prefix: str, resolutions: list[int]) -> dict:
    x = vector_to_array(x_raw)
    y = vector_to_array(y_raw)

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    out = {}

    for p in resolutions:
        if len(x) < 3 or len(np.unique(x)) < 2:
            out.update(empty_curve_features(prefix, p))
            continue

        xu, yu = aggregate_duplicate_x(x, y)

        if len(xu) < 2 or not np.isfinite(xu).all() or not np.isfinite(yu).all():
            out.update(empty_curve_features(prefix, p))
            continue

        x_min = float(np.min(xu))
        x_max = float(np.max(xu))
        x_range = x_max - x_min

        if not np.isfinite(x_range) or x_range <= 0:
            out.update(empty_curve_features(prefix, p))
            continue

        grid = np.linspace(x_min, x_max, p)
        samples = np.interp(grid, xu, yu)

        tmp = {f"{prefix}_valid_p{p:03d}": 1}

        for i, value in enumerate(samples, start=1):
            tmp[f"{prefix}_p{p:03d}_{i:03d}"] = float(value)

        dx = np.diff(grid)
        dy = np.diff(samples)

        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = dy / dx

        slopes = slopes[np.isfinite(slopes)]

        if len(samples) >= 3:
            d2 = np.diff(samples, n=2)
            signs = np.sign(d2[np.abs(d2) > 1e-12])
            inflections = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
        else:
            inflections = 0

        area_value = float(np.trapezoid(samples, grid))

        tmp.update(
            {
                f"{prefix}_x_min_p{p:03d}": x_min,
                f"{prefix}_x_max_p{p:03d}": x_max,
                f"{prefix}_x_range_p{p:03d}": x_range,
                f"{prefix}_y_min_p{p:03d}": float(np.min(samples)),
                f"{prefix}_y_max_p{p:03d}": float(np.max(samples)),
                f"{prefix}_y_range_p{p:03d}": float(np.max(samples) - np.min(samples)),
                f"{prefix}_area_p{p:03d}": area_value,
                f"{prefix}_slope_mean_p{p:03d}": float(np.mean(slopes)) if len(slopes) else np.nan,
                f"{prefix}_slope_std_p{p:03d}": float(np.std(slopes)) if len(slopes) else np.nan,
                f"{prefix}_slope_min_p{p:03d}": float(np.min(slopes)) if len(slopes) else np.nan,
                f"{prefix}_slope_max_p{p:03d}": float(np.max(slopes)) if len(slopes) else np.nan,
                f"{prefix}_slope_max_abs_p{p:03d}": float(np.max(np.abs(slopes))) if len(slopes) else np.nan,
                f"{prefix}_inflection_count_p{p:03d}": inflections,
            }
        )

        out.update(tmp)

    return out


def read_parquet_batches(path: Path, columns: list[str], batch_size: int):
    """
    Robust reader for parquet files with vector/list columns.

    Uses the top-level Arrow schema names through schema_arrow.names.
    The previous use of ParquetFile(path).schema.names may miss nested/list
    columns such as voltage_v and capacity_ah.
    """
    requested_cols = list(dict.fromkeys(columns))

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)

        read_cols = [c for c in requested_cols if c in available]

        if not read_cols:
            raise RuntimeError(f"No requested columns found in {path}")

        df = pd.read_parquet(path, columns=read_cols)

    except Exception as exc:
        print(f"[WARNING] Selective parquet read failed for {path.name}: {exc}")
        print("[WARNING] Falling back to full pandas read.")

        df_full = pd.read_parquet(path)
        read_cols = [c for c in requested_cols if c in df_full.columns]

        if not read_cols:
            raise RuntimeError(f"No requested columns found in {path}")

        df = df_full[read_cols].copy()

    for start_idx in range(0, len(df), batch_size):
        yield df.iloc[start_idx:start_idx + batch_size].copy()

def get_input_files(
    input_root: Path,
    phases: list[str],
    datasets: list[str] | None,
) -> list[Path]:
    files = []

    dataset_filter = None
    if datasets:
        dataset_filter = {d.lower() for d in datasets}

    for phase in phases:
        phase_dir = input_root / phase

        if not phase_dir.exists():
            print(f"[WARNING] Phase directory not found: {phase_dir}")
            continue

        for path in sorted(phase_dir.glob("*.parquet")):
            if dataset_filter:
                prefix = path.name.split("_")[0].lower()
                if prefix not in dataset_filter:
                    continue

            files.append(path)

    return files


def process_file(
    path: Path,
    output_root: Path,
    resolutions: list[int],
    batch_size: int,
    max_rows: int | None,
    force: bool,
):
    phase_folder = path.parent.name.lower()
    dataset_folder = path.name.split("_")[0].lower()

    out_dir = output_root / phase_folder / dataset_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / path.name.replace(".parquet", ".curve_features.parquet")
    done_path = Path(str(out_path) + ".done")

    if done_path.exists() and out_path.exists() and not force:
        print(f"[SKIP] {phase_folder}/{dataset_folder}: {path.name}")
        return

    total_rows_est = count_parquet_rows(path)

    if max_rows is not None:
        total_rows_est = (
            min(total_rows_est, max_rows)
            if total_rows_est is not None
            else max_rows
        )

    print("\n" + "=" * 100)
    print(f"[PROCESS] {phase_folder}/{dataset_folder}")
    print(f"Source: {path}")
    print(f"Output: {out_path}")
    print(f"Estimated rows: {total_rows_est}")
    print(f"Resolutions: {resolutions}")

    wanted_cols = KEY_COLUMNS + OPTIONAL_ID_COLUMNS + VECTOR_COLUMNS

    rows = []
    total = 0

    pbar = tqdm(
        total=total_rows_est,
        desc=f"{phase_folder}/{dataset_folder}",
        unit="row",
        dynamic_ncols=True,
    )

    try:
        for batch_df in read_parquet_batches(path, wanted_cols, batch_size=batch_size):
            if batch_df.empty:
                continue

            if "phase_folder" not in batch_df.columns:
                batch_df["phase_folder"] = phase_folder

            if "phase" not in batch_df.columns:
                batch_df["phase"] = phase_folder

            if "dataset" not in batch_df.columns:
                batch_df["dataset"] = dataset_folder.upper()

            records = batch_df.to_dict("records")

            for rec in records:
                if max_rows is not None and total >= max_rows:
                    break

                out = {}

                for col in KEY_COLUMNS + OPTIONAL_ID_COLUMNS:
                    if col in rec:
                        out[col] = rec[col]

                # V(Q): x = capacity_ah, y = voltage_v
                out.update(
                    curve_features(
                        rec.get("capacity_ah"),
                        rec.get("voltage_v"),
                        prefix="VQ",
                        resolutions=resolutions,
                    )
                )

                # Q(V): x = voltage_v, y = capacity_ah
                out.update(
                    curve_features(
                        rec.get("voltage_v"),
                        rec.get("capacity_ah"),
                        prefix="QV",
                        resolutions=resolutions,
                    )
                )

                rows.append(out)
                total += 1
                pbar.update(1)

            if max_rows is not None and total >= max_rows:
                break

    finally:
        pbar.close()

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(out_path, index=False)

    done_path.write_text(
        f"source={path}\n"
        f"rows={len(df_out)}\n"
        f"columns={len(df_out.columns)}\n"
        f"resolutions={resolutions}\n",
        encoding="utf-8",
    )

    print(f"[DONE] {phase_folder}/{dataset_folder}")
    print(f"Rows: {len(df_out)}")
    print(f"Columns: {len(df_out.columns)}")


def main():
    args = parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    files = get_input_files(input_root, args.phases, args.datasets)

    print("Input root:", input_root)
    print("Output root:", output_root)
    print("Files found:", len(files))
    print("Resolutions:", args.resolutions)
    print("Batch size:", args.batch_size)

    if not files:
        raise RuntimeError("No input files found.")

    for path in tqdm(files, desc="Input files", unit="file", dynamic_ncols=True):
        process_file(
            path=path,
            output_root=output_root,
            resolutions=args.resolutions,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
            force=args.force,
        )

    print("\nAll curve-feature extraction jobs finished.")


if __name__ == "__main__":
    main()
