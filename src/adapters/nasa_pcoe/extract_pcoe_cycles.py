"""
Extract NASA PCoE charge and discharge cycles into branch-specific parquet files.

Outputs
-------
data/interim/pcoe/extracted/pcoe_charge_extracted_cycles.parquet
data/interim/pcoe/extracted/pcoe_discharge_extracted_cycles.parquet
data/reports/pcoe/extraction_report.json

The extraction stage is intentionally conservative: it separates charge and
discharge phases, preserves traceability metadata, and records warnings instead of
applying final exclusions. Physical trimming, EOL assignment, validity flags, and
schema finalization are performed by clean_pcoe_cycles.py.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

try:
    from .pcoe_schema import (
        ADAPTER_VERSION,
        DATASET_NAME,
        DEFAULT_BASE_DIR,
        EXTRACTED_COLUMNS,
    )
except ImportError:  # Allows direct execution from the adapter folder.
    from pcoe_schema import (  # type: ignore
        ADAPTER_VERSION,
        DATASET_NAME,
        DEFAULT_BASE_DIR,
        EXTRACTED_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract NASA PCoE charge and discharge cycles from MATLAB files."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Project root containing data/raw, data/interim, and data/reports.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing NASA PCoE .mat files. Default: <base-dir>/data/raw/PCOE.",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=None,
        help="Interim directory. Default: <base-dir>/data/interim/pcoe.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Report directory. Default: <base-dir>/data/reports/pcoe.",
    )
    parser.add_argument(
        "--lowercase-output-dir",
        action="store_true",
        default=True,
        help="Use lowercase pcoe in output directories. Enabled by default.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    base_dir = args.base_dir
    raw_dir = args.raw_dir or base_dir / "data" / "raw" / "PCOE"
    interim_dir = args.interim_dir or base_dir / "data" / "interim" / "pcoe"
    reports_dir = args.reports_dir or base_dir / "data" / "reports" / "pcoe"
    extracted_dir = interim_dir / "extracted"

    for path in [extracted_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "raw_dir": raw_dir,
        "interim_dir": interim_dir,
        "extracted_dir": extracted_dir,
        "reports_dir": reports_dir,
        "charge_file": extracted_dir / "pcoe_charge_extracted_cycles.parquet",
        "discharge_file": extracted_dir / "pcoe_discharge_extracted_cycles.parquet",
        "report_file": reports_dir / "extraction_report.json",
    }


# ---------------------------------------------------------------------------
# MATLAB and value utilities
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_battery_id(file_stem: str) -> str:
    """Convert NASA PCoE file names to project battery IDs."""
    match = re.match(r"B0*(\d+)(p\d+)?$", file_stem.strip(), flags=re.IGNORECASE)
    if match:
        return f"PCOE{int(match.group(1))}"
    return f"PCOE_{file_stem.strip()}"


def extract_battery_index(file_stem: str) -> int:
    match = re.search(r"B0*(\d+)", file_stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def extract_file_part_number(file_stem: str) -> int:
    match = re.search(r"p(\d+)$", file_stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def pcoe_file_sort_key(path: Path) -> tuple[int, int, str]:
    return (extract_battery_index(path.stem), extract_file_part_number(path.stem), path.stem)


def safe_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        array = np.asarray(value).squeeze()
        if array.size == 0:
            return ""
        if array.dtype.kind in {"U", "S"}:
            return str(array.item() if array.size == 1 else array).strip()
        if array.size == 1:
            item = array.item()
            if isinstance(item, bytes):
                return item.decode(errors="ignore").strip()
            return str(item).strip()
        return str(array).strip()
    except Exception:
        return str(value).strip()


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        array = np.asarray(value).squeeze()
        if array.size == 0:
            return default
        return float(array.item() if array.size == 1 else array.flat[0])
    except Exception:
        return default


def as_float_list(value: Any) -> list[float]:
    try:
        array = np.asarray(value, dtype=float).squeeze()
        if array.size == 0:
            return []
        array = np.atleast_1d(array).astype(float)
        return array.tolist()
    except Exception:
        return []


def get_struct_field(obj: Any, field: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, field):
        return getattr(obj, field)
    if isinstance(obj, np.ndarray) and obj.dtype.names and field in obj.dtype.names:
        return obj[field]
    return default


def unwrap_matlab_vector(obj: Any) -> list[Any]:
    array = np.asarray(obj).squeeze()
    if array.ndim == 0:
        try:
            return [array.item()]
        except Exception:
            return [array]
    return list(array.flat)


def choose_matlab_variable(mat: dict[str, Any], mat_path: Path) -> str:
    candidates = [key for key in mat.keys() if not key.startswith("__")]
    if not candidates:
        raise ValueError(f"No MATLAB variable found in {mat_path.name}.")

    preferred = re.sub(r"p\d+$", "", mat_path.stem, flags=re.IGNORECASE)
    if preferred in mat:
        return preferred
    if mat_path.stem in mat:
        return mat_path.stem
    return candidates[0]


def relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------


def extract_phase_vectors(data_obj: Any) -> dict[str, list[float]]:
    """Extract signal vectors from a NASA PCoE cycle data structure."""
    return {
        "time_s": as_float_list(get_struct_field(data_obj, "Time", None)),
        "voltage_v": as_float_list(get_struct_field(data_obj, "Voltage_measured", None)),
        "current_a": as_float_list(get_struct_field(data_obj, "Current_measured", None)),
        "temperature_c": as_float_list(get_struct_field(data_obj, "Temperature_measured", None)),
        "voltage_charge_v": as_float_list(get_struct_field(data_obj, "Voltage_charge", None)),
        "current_charge_a": as_float_list(get_struct_field(data_obj, "Current_charge", None)),
        "voltage_load_v": as_float_list(get_struct_field(data_obj, "Voltage_load", None)),
        "current_load_a": as_float_list(get_struct_field(data_obj, "Current_load", None)),
    }


def build_extracted_row(
    *,
    mat_path: Path,
    base_dir: Path,
    battery_id: str,
    battery_index: int,
    source_cycle_index: int,
    cycle_pair_number: int,
    phase_cycle_number: int,
    phase: str,
    cycle_type: str,
    cycle_obj: Any,
    data_obj: Any,
) -> dict[str, Any]:
    vectors = extract_phase_vectors(data_obj)
    n_points_raw = max(
        len(vectors["time_s"]),
        len(vectors["voltage_v"]),
        len(vectors["current_a"]),
        len(vectors["temperature_c"]),
    )

    warnings: list[str] = []
    if not vectors["time_s"]:
        warnings.append("missing_time_vector")
    if not vectors["voltage_v"]:
        warnings.append("missing_voltage_vector")
    if not vectors["current_a"]:
        warnings.append("missing_current_vector")
    if not vectors["temperature_c"]:
        warnings.append("missing_temperature_vector")

    ambient_temperature_c = safe_float(
        get_struct_field(cycle_obj, "ambient_temperature", np.nan), default=np.nan
    )
    raw_capacity_summary_ah = safe_float(
        get_struct_field(data_obj, "Capacity", np.nan), default=np.nan
    )
    matlab_test_time = safe_string(get_struct_field(cycle_obj, "time", ""))

    source_cycle_id = f"{mat_path.stem}:cycle_{source_cycle_index:05d}:{phase}"
    cycle_pair_id = f"{battery_id}_pair_{cycle_pair_number:05d}"

    policy_readable = (
        f"NASA PCoE {phase}; ambient_temperature_c={ambient_temperature_c}; "
        f"source_cycle_index={source_cycle_index}; source_file={mat_path.name}"
    )

    row = {
        "dataset": DATASET_NAME,
        "source_file": mat_path.name,
        "source_path": relative_path(mat_path, base_dir),
        "battery_id": battery_id,
        "battery_index": int(battery_index),
        "cell_id_original": re.sub(r"p\d+$", "", mat_path.stem, flags=re.IGNORECASE),
        "source_cycle_index": int(source_cycle_index),
        "source_cycle_id": source_cycle_id,
        "cycle_pair_id": cycle_pair_id,
        "cycle_pair_number": int(cycle_pair_number),
        "cycle_number": int(phase_cycle_number),
        "phase": phase,
        "ambient_temperature_c": ambient_temperature_c,
        "matlab_cycle_type": cycle_type,
        "matlab_test_time": matlab_test_time,
        "raw_capacity_summary_ah": raw_capacity_summary_ah,
        "time_s": vectors["time_s"],
        "voltage_v": vectors["voltage_v"],
        "current_a": vectors["current_a"],
        "temperature_c": vectors["temperature_c"],
        "voltage_charge_v": vectors["voltage_charge_v"],
        "current_charge_a": vectors["current_charge_a"],
        "voltage_load_v": vectors["voltage_load_v"],
        "current_load_a": vectors["current_load_a"],
        "n_points_raw": int(n_points_raw),
        "extraction_warnings": warnings,
        "policy_readable": policy_readable,
        "barcode": mat_path.stem,
        "adapter_version": ADAPTER_VERSION,
        "created_at": utc_now(),
    }
    return row


def extract_one_file(mat_path: Path, base_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    variable_name = choose_matlab_variable(mat, mat_path)
    battery_obj = mat[variable_name]
    cycles_obj = get_struct_field(battery_obj, "cycle", None)
    if cycles_obj is None:
        raise ValueError(f"Field 'cycle' not found in {mat_path.name}.")

    cycles = unwrap_matlab_vector(cycles_obj)
    battery_id = normalize_battery_id(mat_path.stem)
    battery_index = extract_battery_index(mat_path.stem)

    rows: list[dict[str, Any]] = []
    counts = {
        "charge": 0,
        "discharge": 0,
        "impedance": 0,
        "ignored": 0,
        "missing_data": 0,
    }

    cycle_pair_number = 0
    last_pair_has_charge = False
    phase_counters = {"charge": 0, "discharge": 0}

    for source_cycle_index, cycle_obj in enumerate(cycles, start=1):
        cycle_type = safe_string(get_struct_field(cycle_obj, "type", "")).lower()
        if cycle_type not in {"charge", "discharge", "impedance"}:
            counts["ignored"] += 1
            continue
        if cycle_type == "impedance":
            counts["impedance"] += 1
            continue

        phase = cycle_type
        counts[phase] += 1

        if phase == "charge":
            cycle_pair_number += 1
            last_pair_has_charge = True
        elif phase == "discharge" and not last_pair_has_charge:
            cycle_pair_number += 1
        elif phase == "discharge" and last_pair_has_charge:
            last_pair_has_charge = False

        phase_counters[phase] += 1

        data_obj = get_struct_field(cycle_obj, "data", None)
        if data_obj is None:
            counts["missing_data"] += 1
            continue

        rows.append(
            build_extracted_row(
                mat_path=mat_path,
                base_dir=base_dir,
                battery_id=battery_id,
                battery_index=battery_index,
                source_cycle_index=source_cycle_index,
                cycle_pair_number=cycle_pair_number,
                phase_cycle_number=phase_counters[phase],
                phase=phase,
                cycle_type=cycle_type,
                cycle_obj=cycle_obj,
                data_obj=data_obj,
            )
        )

    report = {
        "source_file": mat_path.name,
        "matlab_variable": variable_name,
        "battery_id": battery_id,
        "battery_index": battery_index,
        "file_part_number": extract_file_part_number(mat_path.stem),
        "n_cycles_total": int(len(cycles)),
        "n_charge_cycles_found": int(counts["charge"]),
        "n_discharge_cycles_found": int(counts["discharge"]),
        "n_impedance_cycles_found": int(counts["impedance"]),
        "n_ignored_cycles": int(counts["ignored"]),
        "n_missing_data_cycles": int(counts["missing_data"]),
        "n_rows_extracted": int(len(rows)),
    }
    return rows, report


def recompute_global_cycle_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Renumber cycles globally for batteries split across p1/p2 files."""
    if df.empty:
        return df

    df = df.copy()
    df["_file_part_number"] = df["source_file"].map(lambda name: extract_file_part_number(Path(str(name)).stem))
    df = df.sort_values(
        ["battery_index", "battery_id", "_file_part_number", "source_file", "source_cycle_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Recompute phase-specific cycle numbers.
    df["cycle_number"] = (
        df.groupby(["battery_id", "phase"]).cumcount().astype(int) + 1
    )

    # Recompute pair numbers across split files.
    pair_numbers: list[int] = []
    for _, group in df.groupby("battery_id", sort=False):
        pair_counter = 0
        last_pair_has_charge = False
        for _, row in group.iterrows():
            phase = row["phase"]
            if phase == "charge":
                pair_counter += 1
                last_pair_has_charge = True
            elif phase == "discharge" and not last_pair_has_charge:
                pair_counter += 1
            elif phase == "discharge" and last_pair_has_charge:
                last_pair_has_charge = False
            pair_numbers.append(pair_counter)

    df["cycle_pair_number"] = pair_numbers
    df["cycle_pair_id"] = df.apply(
        lambda row: f"{row['battery_id']}_pair_{int(row['cycle_pair_number']):05d}", axis=1
    )
    df["source_cycle_id"] = df.apply(
        lambda row: f"{Path(str(row['source_file'])).stem}:cycle_{int(row['source_cycle_index']):05d}:{row['phase']}",
        axis=1,
    )

    return df.drop(columns=["_file_part_number"])


def empty_extracted_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EXTRACTED_COLUMNS)


def save_branch(df_all: pd.DataFrame, phase: str, output_file: Path) -> int:
    branch = df_all[df_all["phase"] == phase].copy() if not df_all.empty else empty_extracted_frame()
    for column in EXTRACTED_COLUMNS:
        if column not in branch.columns:
            branch[column] = None
    branch = branch[EXTRACTED_COLUMNS]
    branch.to_parquet(output_file, index=False)
    return int(len(branch))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    raw_dir = paths["raw_dir"]

    mat_files = sorted(raw_dir.glob("*.mat"), key=pcoe_file_sort_key)
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {raw_dir}")

    all_rows: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []
    failed_files: list[dict[str, str]] = []

    for mat_path in tqdm(mat_files, desc="Extracting NASA PCoE files", unit="file"):
        try:
            rows, report = extract_one_file(mat_path, paths["base_dir"])
            all_rows.extend(rows)
            file_reports.append(report)
        except Exception as exc:
            failed_files.append({"source_file": mat_path.name, "error": str(exc)})

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all = recompute_global_cycle_numbers(df_all)
    else:
        df_all = empty_extracted_frame()

    n_charge = save_branch(df_all, "charge", paths["charge_file"])
    n_discharge = save_branch(df_all, "discharge", paths["discharge_file"])

    report = {
        "adapter_version": ADAPTER_VERSION,
        "created_at": utc_now(),
        "raw_dir": str(raw_dir),
        "charge_output_file": str(paths["charge_file"]),
        "discharge_output_file": str(paths["discharge_file"]),
        "n_files_found": int(len(mat_files)),
        "n_files_failed": int(len(failed_files)),
        "n_rows_total": int(len(df_all)),
        "n_charge_rows": n_charge,
        "n_discharge_rows": n_discharge,
        "n_batteries": int(df_all["battery_id"].nunique()) if not df_all.empty else 0,
        "batteries": sorted(df_all["battery_id"].unique().tolist()) if not df_all.empty else [],
        "file_reports": file_reports,
        "failed_files": failed_files,
    }

    with open(paths["report_file"], "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("Extraction completed.")
    print(f"Charge cycles:    {n_charge} -> {paths['charge_file']}")
    print(f"Discharge cycles: {n_discharge} -> {paths['discharge_file']}")
    print(f"Report: {paths['report_file']}")


if __name__ == "__main__":
    main()
