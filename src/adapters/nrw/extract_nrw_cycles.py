"""
Extract NASA Random Walk (NRW) reference charge and discharge cycles.

This script converts raw RW*.mat files into phase-specific parquet files for the
standard lithium-ion battery cycle schema used by the project. The extractor is
intentionally conservative for the NASA Random Walk dataset: it keeps only steps
whose comment is exactly reference charge or reference discharge and skips rest,
random-walk, pulsed, and ambiguous characterization segments.

Default project layout expected by this script:
    Batteries/data/raw/NRW/RW1.mat
    Batteries/data/interim/nrw/extracted/nrw_charge_extracted_cycles.parquet
    Batteries/data/interim/nrw/extracted/nrw_discharge_extracted_cycles.parquet

Run from any directory:
    python src/adapters/nrw/extract_nrw_cycles.py

Optional:
    python src/adapters/nrw/extract_nrw_cycles.py --base-dir "D:/.../Batteries"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable


DATASET_CODE = "NRW"
DATASET_FOLDER = "nrw"
ADAPTER_VERSION = "nrw_extractor_v1.1"

DEFAULT_BASE_DIR = Path.cwd()

MIN_POINTS_EXTRACTED = 10
NOMINAL_CAPACITY_AH = 2.1
NOMINAL_VOLTAGE_V = 3.7

REFERENCE_CHARGE_EXACT_COMMENTS = {
    "reference charge",
    "reference charging",
    "reference cycle charge",
}
REFERENCE_DISCHARGE_EXACT_COMMENTS = {
    "reference discharge",
    "reference discharging",
    "reference cycle discharge",
}
REFERENCE_POWER_DISCHARGE_EXACT_COMMENTS = {
    "reference power discharge",
}

STANDARD_CYCLE_COLUMNS = [
    # Identification
    "dataset", "source_file", "source_path", "battery_id", "battery_index",
    "cell_id_original", "cycle_index", "cycle_number", "source_cycle_id",
    "cycle_pair_id", "phase",

    # Lifetime and target
    "cycle_life", "remaining_cycles", "remaining_cycles_norm", "eol_criterion",
    "eol_capacity_ah", "initial_capacity_ah", "nominal_capacity_ah", "soh",

    # Vector signals
    "time_s", "voltage_v", "current_a", "temperature_c", "capacity_ah",
    "energy_wh", "power_w",

    # Scalar values derived from signals
    "n_points", "duration_s", "sample_period_median_s", "voltage_min_v",
    "voltage_max_v", "voltage_start_v", "voltage_end_v", "current_min_a",
    "current_max_a", "current_mean_a", "temperature_min_c",
    "temperature_max_c", "temperature_mean_c", "phase_capacity_ah",
    "phase_energy_wh",

    # Signal source metadata
    "voltage_source", "current_source", "temperature_source", "capacity_source",
    "energy_source",

    # Experimental protocol
    "chemistry", "form_factor", "manufacturer", "nominal_voltage_v",
    "ambient_temperature_c", "charge_protocol", "discharge_protocol",
    "c_rate_charge", "c_rate_discharge", "dod", "policy_readable", "barcode",

    # Quality and traceability
    "is_valid_cycle", "is_model_usable", "exclude_reason", "cleaning_warnings",
    "n_removed_points", "trim_start_index", "trim_end_index", "has_voltage",
    "has_current", "has_temperature", "has_capacity", "has_energy",
    "adapter_version", "cleaning_version", "created_at",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def matlab_to_str(value: Any) -> str:
    """Convert MATLAB strings/chars loaded by scipy.io.loadmat to Python str."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode(errors="ignore").strip()

    try:
        arr = np.asarray(value)
    except Exception:
        return str(value).strip()

    if arr.size == 0:
        return ""

    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.astype(str).ravel()).strip()

    # MATLAB char arrays are sometimes loaded as uint16/uint8 code points.
    if arr.dtype.kind in {"u", "i"} and arr.ndim >= 1:
        flat = arr.ravel()
        if flat.size > 1 and np.all((flat >= 0) & (flat < 256)):
            try:
                return "".join(chr(int(v)) for v in flat if int(v) != 0).strip()
            except Exception:
                pass

    try:
        return str(arr.item()).strip()
    except Exception:
        return str(value).strip()


def safe_float_array(value: Any) -> np.ndarray:
    """Convert a MATLAB vector-like object to a 1D float array."""
    if value is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(value, dtype=float).squeeze()
        arr = np.atleast_1d(arr).astype(float)
        return arr.ravel()
    except Exception:
        return np.array([], dtype=float)


def get_struct_field(obj: Any, field: str) -> Any:
    """Access a field from a scipy MATLAB struct, dict, or numpy void."""
    if obj is None:
        return None

    if hasattr(obj, field):
        return getattr(obj, field)

    if isinstance(obj, dict) and field in obj:
        return obj[field]

    try:
        return obj[field]
    except Exception:
        return None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def identify_reference_phase(
    comment: str,
    step_type: str,
    *,
    include_reference_power_discharge: bool = False,
) -> str | None:
    """Return 'charge', 'discharge', or None for non-reference/ambiguous steps.

    The NASA Random Walk files include comments such as ``rest post reference
    charge`` and ``rest prior reference discharge``. Version 1.0 used substring
    matching and therefore selected those rest steps. Version 1.1 uses exact
    normalized comments to avoid extracting rest, pulse, and ambiguous
    characterization steps as charge/discharge cycles.
    """
    comment_text = normalize_text(comment)
    type_text = normalize_text(step_type)

    if comment_text in REFERENCE_CHARGE_EXACT_COMMENTS:
        return "charge"
    if comment_text in REFERENCE_DISCHARGE_EXACT_COMMENTS:
        return "discharge"

    if include_reference_power_discharge and comment_text in REFERENCE_POWER_DISCHARGE_EXACT_COMMENTS:
        # Optional because this stage is not paired one-to-one with the standard
        # reference charge cycles in all NRW files. Keep disabled for the main
        # RUL pipeline unless the article explicitly analyzes this protocol.
        return "discharge"

    # Fallback only for unusually encoded MATLAB comments that are exactly
    # "reference" and whose step type identifies the phase. Do not use substring
    # matching because it captures rest stages containing the word reference.
    if comment_text == "reference":
        if type_text in {"d", "discharge"}:
            return "discharge"
        if type_text in {"c", "charge"}:
            return "charge"

    return None


def nominal_temperature_for_battery(battery_index: int) -> float:
    """
    Nominal NASA Random Walk ambient group temperature.

    The legacy project convention treats RW21-RW28 as elevated-temperature cells
    and RW1-RW20 as room-temperature cells. Measured temperature is preserved when
    available; this value is only fallback metadata.
    """
    if battery_index >= 21:
        return 40.0
    return 25.0


def align_raw_vectors(
    time_s: Any,
    voltage_v: Any,
    current_a: Any,
    temperature_c: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    """Minimally align raw vectors without applying final cleaning decisions."""
    warnings: list[str] = []

    time = safe_float_array(time_s)
    voltage = safe_float_array(voltage_v)
    current = safe_float_array(current_a)
    temperature = safe_float_array(temperature_c)

    n = min(len(time), len(voltage), len(current))
    if n == 0:
        return None

    time = time[:n]
    voltage = voltage[:n]
    current = current[:n]

    if len(temperature) >= n:
        temperature = temperature[:n]
    else:
        warnings.append("temperature_vector_missing_or_short")
        temperature = np.full(n, np.nan, dtype=float)

    mask = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current)
    if not mask.all():
        warnings.append(f"removed_nonfinite_core_samples={int((~mask).sum())}")

    time = time[mask]
    voltage = voltage[mask]
    current = current[mask]
    temperature = temperature[mask]

    if len(time) < MIN_POINTS_EXTRACTED:
        return None

    order = np.argsort(time)
    time = time[order]
    voltage = voltage[order]
    current = current[order]
    temperature = temperature[order]

    _, unique_idx = np.unique(time, return_index=True)
    if len(unique_idx) != len(time):
        warnings.append(f"removed_duplicate_time_samples={len(time) - len(unique_idx)}")
    unique_idx = np.sort(unique_idx)

    time = time[unique_idx]
    voltage = voltage[unique_idx]
    current = current[unique_idx]
    temperature = temperature[unique_idx]

    if len(time) < MIN_POINTS_EXTRACTED:
        return None

    time = time - time[0]
    return time, voltage, current, temperature, warnings


def enforce_current_sign(current_a: np.ndarray, phase: str) -> np.ndarray:
    """Apply project convention: charge positive, discharge negative."""
    if phase == "charge":
        return np.abs(current_a.astype(float))
    if phase == "discharge":
        return -np.abs(current_a.astype(float))
    raise ValueError(f"Unsupported phase: {phase}")


def cumulative_trapezoid_integral(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with output length equal to input length."""
    if len(time_s) == 0:
        return np.array([], dtype=float)
    if len(time_s) == 1:
        return np.array([0.0], dtype=float)

    dt = np.diff(time_s)
    mid_values = 0.5 * (values[1:] + values[:-1])
    increments = mid_values * dt
    increments = np.where(np.isfinite(increments), increments, 0.0)
    return np.concatenate([[0.0], np.cumsum(increments)])


def calculate_capacity_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    """Accumulated phase capacity in Ah using abs(current)."""
    capacity = cumulative_trapezoid_integral(time_s, np.abs(current_a)) / 3600.0
    capacity = np.maximum.accumulate(np.where(np.isfinite(capacity), capacity, 0.0))
    if capacity.size:
        capacity[0] = 0.0
    return capacity


def calculate_energy_wh(time_s: np.ndarray, voltage_v: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    """Accumulated absolute phase energy in Wh; signed instantaneous power is stored separately."""
    abs_power = np.abs(voltage_v * current_a)
    energy = cumulative_trapezoid_integral(time_s, abs_power) / 3600.0
    energy = np.maximum.accumulate(np.where(np.isfinite(energy), energy, 0.0))
    if energy.size:
        energy[0] = 0.0
    return energy


def finite_nanmin(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else math.nan


def finite_nanmax(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else math.nan


def finite_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else math.nan


def build_row(
    *,
    mat_path: Path,
    base_dir: Path,
    battery_name: str,
    battery_index: int,
    phase: str,
    phase_cycle_index: int,
    source_step_index: int,
    source_comment: str,
    source_type: str,
    source_date: str,
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    temperature_c: np.ndarray,
    extraction_warnings: list[str],
) -> dict[str, Any]:
    signed_current = enforce_current_sign(current_a, phase)
    capacity_ah = calculate_capacity_ah(time_s, signed_current)
    energy_wh = calculate_energy_wh(time_s, voltage_v, signed_current)
    power_w = voltage_v * signed_current

    n_points = int(len(time_s))
    sample_period = float(np.nanmedian(np.diff(time_s))) if n_points > 1 else math.nan
    phase_capacity = float(capacity_ah[-1]) if capacity_ah.size else math.nan
    phase_energy = float(energy_wh[-1]) if energy_wh.size else math.nan
    temp_source = "measured" if np.isfinite(temperature_c).any() else "unknown_nan"

    try:
        source_path = str(mat_path.relative_to(base_dir))
    except Exception:
        source_path = str(mat_path)

    source_cycle_id = f"{battery_name}_step_{source_step_index:05d}"
    cycle_pair_id = f"{DATASET_CODE}_{battery_name}_ref_{phase_cycle_index:04d}"
    created_at = utc_now_iso()
    ambient_temp = nominal_temperature_for_battery(battery_index)

    return {
        "dataset": DATASET_CODE,
        "source_file": mat_path.name,
        "source_path": source_path,
        "battery_id": f"{DATASET_CODE}_{battery_name}",
        "battery_index": int(battery_index),
        "cell_id_original": battery_name,
        "cycle_index": int(phase_cycle_index),
        "cycle_number": float(phase_cycle_index),
        "source_cycle_id": source_cycle_id,
        "cycle_pair_id": cycle_pair_id,
        "phase": phase,

        "cycle_life": math.nan,
        "remaining_cycles": math.nan,
        "remaining_cycles_norm": math.nan,
        "eol_criterion": "not_assigned_at_extraction",
        "eol_capacity_ah": math.nan,
        "initial_capacity_ah": math.nan,
        "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "soh": math.nan,

        "time_s": time_s.astype(float).tolist(),
        "voltage_v": voltage_v.astype(float).tolist(),
        "current_a": signed_current.astype(float).tolist(),
        "temperature_c": temperature_c.astype(float).tolist(),
        "capacity_ah": capacity_ah.astype(float).tolist(),
        "energy_wh": energy_wh.astype(float).tolist(),
        "power_w": power_w.astype(float).tolist(),

        "n_points": n_points,
        "duration_s": float(time_s[-1] - time_s[0]) if n_points > 1 else math.nan,
        "sample_period_median_s": sample_period,
        "voltage_min_v": finite_nanmin(voltage_v),
        "voltage_max_v": finite_nanmax(voltage_v),
        "voltage_start_v": float(voltage_v[0]) if n_points else math.nan,
        "voltage_end_v": float(voltage_v[-1]) if n_points else math.nan,
        "current_min_a": finite_nanmin(signed_current),
        "current_max_a": finite_nanmax(signed_current),
        "current_mean_a": finite_nanmean(signed_current),
        "temperature_min_c": finite_nanmin(temperature_c),
        "temperature_max_c": finite_nanmax(temperature_c),
        "temperature_mean_c": finite_nanmean(temperature_c),
        "phase_capacity_ah": phase_capacity,
        "phase_energy_wh": phase_energy,

        "voltage_source": "measured",
        "current_source": "measured",
        "temperature_source": temp_source,
        "capacity_source": "integrated_current",
        "energy_source": "integrated_power",

        "chemistry": "LCO",
        "form_factor": "18650",
        "manufacturer": "unknown",
        "nominal_voltage_v": NOMINAL_VOLTAGE_V,
        "ambient_temperature_c": ambient_temp,
        "charge_protocol": "NASA Random Walk reference charge when available; random-walk charge steps skipped",
        "discharge_protocol": "NASA Random Walk reference discharge when available; random-walk discharge steps skipped",
        "c_rate_charge": math.nan,
        "c_rate_discharge": math.nan,
        "dod": math.nan,
        "policy_readable": f"comment={source_comment}; type={source_type}; date={source_date}",
        "barcode": "",

        "is_valid_cycle": False,
        "is_model_usable": False,
        "exclude_reason": "not_cleaned_yet",
        "cleaning_warnings": extraction_warnings,
        "n_removed_points": 0,
        "trim_start_index": 0,
        "trim_end_index": max(n_points - 1, 0),
        "has_voltage": bool(np.isfinite(voltage_v).any()),
        "has_current": bool(np.isfinite(signed_current).any()),
        "has_temperature": bool(np.isfinite(temperature_c).any()),
        "has_capacity": bool(np.isfinite(capacity_ah).any() and phase_capacity > 0),
        "has_energy": bool(np.isfinite(energy_wh).any() and phase_energy > 0),
        "adapter_version": ADAPTER_VERSION,
        "cleaning_version": "not_cleaned_yet",
        "created_at": created_at,
    }


def empty_standard_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)


def reorder_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in STANDARD_CYCLE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[STANDARD_CYCLE_COLUMNS].copy()


def extract_one_file(
    mat_path: Path,
    base_dir: Path,
    *,
    include_reference_power_discharge: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    battery_name = mat_path.stem
    match = re.search(r"(\d+)$", battery_name)
    battery_index = int(match.group(1)) if match else -1

    report = {
        "source_file": mat_path.name,
        "battery_id": f"{DATASET_CODE}_{battery_name}",
        "status": "ok",
        "n_steps": 0,
        "n_reference_charge": 0,
        "n_reference_discharge": 0,
        "n_skipped_non_reference": 0,
        "n_failed_steps": 0,
        "comments_seen": {},
        "warnings": [],
    }

    try:
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        report["status"] = "failed_read"
        report["warnings"].append(str(exc))
        return empty_standard_frame(), empty_standard_frame(), report

    if "data" not in mat:
        report["status"] = "missing_data_field"
        return empty_standard_frame(), empty_standard_frame(), report

    data = mat["data"]
    steps = get_struct_field(data, "step")
    if steps is None:
        report["status"] = "missing_data_step_field"
        return empty_standard_frame(), empty_standard_frame(), report

    steps = np.atleast_1d(steps)
    report["n_steps"] = int(len(steps))

    rows_by_phase: dict[str, list[dict[str, Any]]] = {"charge": [], "discharge": []}
    phase_counts = {"charge": 0, "discharge": 0}

    for step_index, step in enumerate(steps, start=1):
        comment = matlab_to_str(get_struct_field(step, "comment"))
        step_type = matlab_to_str(get_struct_field(step, "type"))
        date = matlab_to_str(get_struct_field(step, "date"))

        normalized_comment = normalize_text(comment or step_type or "unknown")
        report["comments_seen"][normalized_comment] = report["comments_seen"].get(normalized_comment, 0) + 1

        phase = identify_reference_phase(
            comment=comment,
            step_type=step_type,
            include_reference_power_discharge=include_reference_power_discharge,
        )
        if phase is None:
            report["n_skipped_non_reference"] += 1
            continue

        time_candidate = get_struct_field(step, "relativeTime")
        if time_candidate is None or safe_float_array(time_candidate).size == 0:
            time_candidate = get_struct_field(step, "time")

        aligned = align_raw_vectors(
            time_s=time_candidate,
            voltage_v=get_struct_field(step, "voltage"),
            current_a=get_struct_field(step, "current"),
            temperature_c=get_struct_field(step, "temperature"),
        )

        if aligned is None:
            report["n_failed_steps"] += 1
            report["warnings"].append(f"step_{step_index:05d}_{phase}_too_short_or_missing_core_vectors")
            continue

        time_s, voltage_v, current_a, temperature_c, warnings = aligned
        phase_counts[phase] += 1

        row = build_row(
            mat_path=mat_path,
            base_dir=base_dir,
            battery_name=battery_name,
            battery_index=battery_index,
            phase=phase,
            phase_cycle_index=phase_counts[phase],
            source_step_index=step_index,
            source_comment=comment,
            source_type=step_type,
            source_date=date,
            time_s=time_s,
            voltage_v=voltage_v,
            current_a=current_a,
            temperature_c=temperature_c,
            extraction_warnings=warnings,
        )
        rows_by_phase[phase].append(row)

    report["n_reference_charge"] = int(phase_counts["charge"])
    report["n_reference_discharge"] = int(phase_counts["discharge"])

    charge_df = reorder_standard_columns(pd.DataFrame(rows_by_phase["charge"])) if rows_by_phase["charge"] else empty_standard_frame()
    discharge_df = reorder_standard_columns(pd.DataFrame(rows_by_phase["discharge"])) if rows_by_phase["discharge"] else empty_standard_frame()

    return charge_df, discharge_df, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract NASA Random Walk reference cycles into standard parquet files.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Project root directory containing data/ and src/.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory containing RW*.mat files. Defaults to <base-dir>/data/raw/NRW.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <base-dir>/data/interim/nrw/extracted.")
    parser.add_argument("--reports-dir", type=Path, default=None, help="Report directory. Defaults to <base-dir>/data/reports/nrw.")
    parser.add_argument(
        "--include-reference-power-discharge",
        action="store_true",
        help=(
            "Also extract steps whose comment is exactly 'reference power discharge'. "
            "Disabled by default because these steps are not standard reference "
            "capacity-discharge cycles and are not always paired with charge cycles."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    raw_dir = args.raw_dir or (base_dir / "data" / "raw" / "NRW")
    output_dir = args.output_dir or (base_dir / "data" / "interim" / DATASET_FOLDER / "extracted")
    reports_dir = args.reports_dir or (base_dir / "data" / "reports" / DATASET_FOLDER)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    charge_out = output_dir / f"{DATASET_FOLDER}_charge_extracted_cycles.parquet"
    discharge_out = output_dir / f"{DATASET_FOLDER}_discharge_extracted_cycles.parquet"
    report_out = reports_dir / "extraction_report.json"
    summary_out = reports_dir / "extraction_summary.csv"

    files = sorted(
        raw_dir.glob("RW*.mat"),
        key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)) if re.search(r"(\d+)$", p.stem) else 999999,
    )
    if not files:
        raise FileNotFoundError(f"No RW*.mat files found in: {raw_dir}")

    all_charge: list[pd.DataFrame] = []
    all_discharge: list[pd.DataFrame] = []
    file_reports: list[dict[str, Any]] = []

    print(f"Found {len(files)} raw NASA Random Walk files in: {raw_dir}")
    for mat_path in tqdm(files, desc="Extracting NRW files", unit="file"):
        charge_df, discharge_df, report = extract_one_file(
            mat_path,
            base_dir=base_dir,
            include_reference_power_discharge=args.include_reference_power_discharge,
        )
        file_reports.append(report)
        if not charge_df.empty:
            all_charge.append(charge_df)
        if not discharge_df.empty:
            all_discharge.append(discharge_df)

    charge_all = reorder_standard_columns(pd.concat(all_charge, ignore_index=True)) if all_charge else empty_standard_frame()
    discharge_all = reorder_standard_columns(pd.concat(all_discharge, ignore_index=True)) if all_discharge else empty_standard_frame()

    if not charge_all.empty:
        charge_all = charge_all.sort_values(["battery_index", "cycle_index"]).reset_index(drop=True)
    if not discharge_all.empty:
        discharge_all = discharge_all.sort_values(["battery_index", "cycle_index"]).reset_index(drop=True)

    charge_all.to_parquet(charge_out, index=False)
    discharge_all.to_parquet(discharge_out, index=False)

    summary_df = pd.DataFrame(file_reports)
    summary_df.drop(columns=["comments_seen", "warnings"], errors="ignore").to_csv(summary_out, index=False, encoding="utf-8-sig")

    report_payload = {
        "dataset": DATASET_CODE,
        "adapter_version": ADAPTER_VERSION,
        "created_at": utc_now_iso(),
        "raw_dir": str(raw_dir),
        "charge_output": str(charge_out),
        "discharge_output": str(discharge_out),
        "n_files": len(files),
        "n_charge_rows": int(len(charge_all)),
        "n_discharge_rows": int(len(discharge_all)),
        "n_batteries_charge": int(charge_all["battery_id"].nunique()) if not charge_all.empty else 0,
        "n_batteries_discharge": int(discharge_all["battery_id"].nunique()) if not discharge_all.empty else 0,
        "include_reference_power_discharge": bool(args.include_reference_power_discharge),
        "reference_phase_detection": "exact_comment_match_v1.1",
        "file_reports": file_reports,
    }
    with report_out.open("w", encoding="utf-8") as fp:
        json.dump(report_payload, fp, indent=2, ensure_ascii=False)

    print("\nExtraction completed.")
    print(f"Charge cycles:    {len(charge_all):6d} -> {charge_out}")
    print(f"Discharge cycles: {len(discharge_all):6d} -> {discharge_out}")
    print(f"Report: {report_out}")
    print(f"Summary: {summary_out}")


if __name__ == "__main__":
    main()
