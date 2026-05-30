# -*- coding: utf-8 -*-
r"""
Extract Oxford/ORA battery characterization cycles into separate charge and
 discharge branches.

Expected raw input example:
    D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\raw\ORA\Oxford_Battery_Degradation_Dataset_1.mat

Main outputs:
    data/interim/ora/extracted/ora_charge_extracted_cycles.parquet
    data/interim/ora/extracted/ora_discharge_extracted_cycles.parquet
    data/reports/ora/extraction_report.json
    data/reports/ora/extraction_summary.csv

Oxford-specific assumptions used by this adapter:
    - C1ch is the primary charge characterization phase.
    - C1dc is the primary discharge characterization phase.
    - The source file provides q, v, t, and T vectors for each phase.
    - Current is reconstructed as +0.740 A for C1ch and -0.740 A for C1dc.
    - The nominal chamber temperature is 40 °C; measured T is used when present.

This script preserves questionable cycles whenever possible and records extraction
warnings instead of applying final model-readiness filtering. Final validity and
model-usability decisions are performed in clean_ora_cycles.py.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab import mat_struct

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

try:
    from .ora_schema import (
        ADAPTER_VERSION,
        AMBIENT_TEMPERATURE_C,
        C1_CURRENT_A,
        DATASET_FOLDER,
        DATASET_NAME,
        EOL_CAPACITY_AH,
        NOMINAL_CAPACITY_AH,
        PRIMARY_CHARGE_PHASE,
        PRIMARY_DISCHARGE_PHASE,
        SIGNAL_KEYS,
        STANDARD_CYCLE_COLUMNS,
        align_vectors,
        empty_standard_dataframe,
        ensure_dir,
        ensure_python_scalars,
        infer_default_base_dir,
        integrate_energy_from_power,
        make_relative_path,
        median_sample_period,
        normalize_capacity_to_ah,
        normalize_time_seconds,
        now_iso,
        phase_to_current,
        reorder_standard_columns,
        safe_array,
        to_float_list,
        vector_stats,
        write_json,
    )
except ImportError:  # pragma: no cover - direct execution fallback
    sys.path.append(str(Path(__file__).resolve().parent))
    from ora_schema import (  # type: ignore
        ADAPTER_VERSION,
        AMBIENT_TEMPERATURE_C,
        C1_CURRENT_A,
        DATASET_FOLDER,
        DATASET_NAME,
        EOL_CAPACITY_AH,
        NOMINAL_CAPACITY_AH,
        PRIMARY_CHARGE_PHASE,
        PRIMARY_DISCHARGE_PHASE,
        SIGNAL_KEYS,
        STANDARD_CYCLE_COLUMNS,
        align_vectors,
        empty_standard_dataframe,
        ensure_dir,
        ensure_python_scalars,
        infer_default_base_dir,
        integrate_energy_from_power,
        make_relative_path,
        median_sample_period,
        normalize_capacity_to_ah,
        normalize_time_seconds,
        now_iso,
        phase_to_current,
        reorder_standard_columns,
        safe_array,
        to_float_list,
        vector_stats,
        write_json,
    )


def fieldnames(obj: Any) -> list[str]:
    if isinstance(obj, mat_struct):
        return list(getattr(obj, "_fieldnames", []) or [])
    if hasattr(obj, "_fieldnames"):
        return list(getattr(obj, "_fieldnames", []) or [])
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        return list(obj.dtype.names)
    return []


def get_field(obj: Any, field: str) -> Any:
    if isinstance(obj, mat_struct):
        return getattr(obj, field)
    if hasattr(obj, field):
        return getattr(obj, field)
    if isinstance(obj, np.ndarray) and obj.dtype.names and field in obj.dtype.names:
        return obj[field]
    raise KeyError(field)


def has_field(obj: Any, field: str) -> bool:
    return field in fieldnames(obj)


def get_field_case_insensitive(obj: Any, field: str) -> Any:
    names = fieldnames(obj)
    if field in names:
        return get_field(obj, field)
    lower_map = {name.lower(): name for name in names}
    if field.lower() in lower_map:
        return get_field(obj, lower_map[field.lower()])
    raise KeyError(field)


def infer_number_from_name(name: str) -> int | None:
    match = re.search(r"(\d+)", str(name))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def sort_key_by_embedded_number(name: str) -> int:
    number = infer_number_from_name(name)
    return int(number) if number is not None else 10**12


def get_cell_variables(mat_data: dict[str, Any]) -> list[tuple[str, int, Any]]:
    cells: list[tuple[str, int, Any]] = []
    for key, value in mat_data.items():
        if key.startswith("__"):
            continue
        if re.fullmatch(r"Cell\d+", key, flags=re.IGNORECASE):
            cell_number = infer_number_from_name(key) or (len(cells) + 1)
            cells.append((key, int(cell_number), value))
    return sorted(cells, key=lambda item: item[1])


def extract_phase_vectors(phase_obj: Any, source_phase_name: str) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    """Extract and align t, v, q, and T vectors from an Oxford phase object."""
    warnings: list[str] = []

    try:
        raw_vectors: dict[str, np.ndarray] = {}
        missing_keys: list[str] = []
        for key in SIGNAL_KEYS:
            try:
                raw_vectors[key] = safe_array(get_field_case_insensitive(phase_obj, key), keep_nan=True)
            except Exception:
                missing_keys.append(key)

        if "t" in missing_keys or "v" in missing_keys or "q" in missing_keys:
            warnings.append(f"missing_required_phase_keys:{','.join(missing_keys)}")
            return None, warnings

        time_s = normalize_time_seconds(raw_vectors["t"])
        voltage_v = safe_array(raw_vectors["v"], keep_nan=True)
        capacity_ah = normalize_capacity_to_ah(raw_vectors["q"])

        if "T" in missing_keys or raw_vectors.get("T", np.array([])).size == 0:
            temperature_c = np.full(min(len(time_s), len(voltage_v), len(capacity_ah)), AMBIENT_TEMPERATURE_C, dtype=float)
            warnings.append("temperature_missing_nominal_40C_used")
            aligned = align_vectors(
                {"time_s": time_s, "voltage_v": voltage_v, "capacity_ah": capacity_ah, "temperature_c": temperature_c},
            )
        else:
            temperature_c = safe_array(raw_vectors["T"], keep_nan=True)
            aligned = align_vectors(
                {"time_s": time_s, "voltage_v": voltage_v, "capacity_ah": capacity_ah, "temperature_c": temperature_c},
            )

        if not aligned:
            warnings.append("empty_aligned_vectors")
            return None, warnings

        n_points = min(len(vector) for vector in aligned.values())
        if n_points < 5:
            warnings.append("too_few_points_after_alignment")
            return None, warnings

        current_a = phase_to_current(source_phase_name, n_points)
        result = {
            "time_s": aligned["time_s"][:n_points],
            "voltage_v": aligned["voltage_v"][:n_points],
            "current_a": current_a,
            "temperature_c": aligned["temperature_c"][:n_points],
            "capacity_ah": aligned["capacity_ah"][:n_points],
        }
        return result, warnings

    except Exception as exc:
        warnings.append(f"phase_extraction_error:{type(exc).__name__}:{exc}")
        return None, warnings


def phase_name_to_standard_phase(source_phase_name: str) -> str:
    if source_phase_name.lower().endswith("ch"):
        return "charge"
    if source_phase_name.lower().endswith("dc"):
        return "discharge"
    raise ValueError(f"Cannot infer standard phase from source phase: {source_phase_name}")


def build_extracted_row(
    *,
    base_dir: Path,
    mat_path: Path,
    cell_name: str,
    cell_number: int,
    cycle_field: str,
    cycle_index: int,
    cycle_life_proxy: float,
    source_phase_name: str,
    phase_vectors: dict[str, np.ndarray],
    extraction_warnings: list[str],
    round_vectors: int | None,
) -> dict[str, Any]:
    phase = phase_name_to_standard_phase(source_phase_name)
    cycle_number = infer_number_from_name(cycle_field)
    if cycle_number is None:
        cycle_number = cycle_index

    battery_id = f"{DATASET_NAME}_Cell{cell_number:02d}"
    cycle_pair_id = f"{battery_id}_cycle_{int(cycle_number):05d}"

    time_s = safe_array(phase_vectors["time_s"], keep_nan=True)
    voltage_v = safe_array(phase_vectors["voltage_v"], keep_nan=True)
    current_a = safe_array(phase_vectors["current_a"], keep_nan=True)
    temperature_c = safe_array(phase_vectors["temperature_c"], keep_nan=True)
    capacity_ah = safe_array(phase_vectors["capacity_ah"], keep_nan=True)
    power_w = voltage_v * current_a
    energy_wh = integrate_energy_from_power(power_w, time_s)

    voltage_stats = vector_stats(voltage_v)
    current_stats = vector_stats(current_a)
    temperature_stats = vector_stats(temperature_c)

    duration_s = float(time_s[-1] - time_s[0]) if time_s.size >= 2 else np.nan
    phase_capacity_ah = float(capacity_ah[-1]) if capacity_ah.size else np.nan
    phase_energy_wh = float(energy_wh[-1]) if energy_wh.size else np.nan
    remaining_cycles = float(cycle_life_proxy - float(cycle_number)) if np.isfinite(cycle_life_proxy) else np.nan

    temperature_is_measured = "temperature_missing_nominal_40C_used" not in extraction_warnings

    row = {
        "dataset": DATASET_NAME,
        "source_file": mat_path.name,
        "source_path": make_relative_path(mat_path, base_dir),
        "battery_id": battery_id,
        "battery_index": int(cell_number),
        "cell_id_original": cell_name,
        "cycle_index": int(cycle_index),
        "cycle_number": float(cycle_number),
        "source_cycle_id": f"{cycle_field}/{source_phase_name}",
        "cycle_pair_id": cycle_pair_id,
        "phase": phase,
        "cycle_life": float(cycle_life_proxy),
        "remaining_cycles": remaining_cycles,
        "remaining_cycles_norm": remaining_cycles / cycle_life_proxy if np.isfinite(cycle_life_proxy) and cycle_life_proxy > 0 else np.nan,
        "eol_criterion": "proxy_last_available_characterization_cycle_before_cleaning",
        "eol_capacity_ah": EOL_CAPACITY_AH,
        "initial_capacity_ah": np.nan,
        "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "soh": np.nan,
        "time_s": to_float_list(time_s, decimals=round_vectors),
        "voltage_v": to_float_list(voltage_v, decimals=round_vectors),
        "current_a": to_float_list(current_a, decimals=round_vectors),
        "temperature_c": to_float_list(temperature_c, decimals=round_vectors),
        "capacity_ah": to_float_list(capacity_ah, decimals=round_vectors),
        "energy_wh": to_float_list(energy_wh, decimals=round_vectors),
        "power_w": to_float_list(power_w, decimals=round_vectors),
        "n_points": int(len(time_s)),
        "duration_s": duration_s,
        "sample_period_median_s": median_sample_period(time_s),
        "voltage_min_v": voltage_stats["min"],
        "voltage_max_v": voltage_stats["max"],
        "voltage_start_v": voltage_stats["start"],
        "voltage_end_v": voltage_stats["end"],
        "current_min_a": current_stats["min"],
        "current_max_a": current_stats["max"],
        "current_mean_a": current_stats["mean"],
        "temperature_min_c": temperature_stats["min"],
        "temperature_max_c": temperature_stats["max"],
        "temperature_mean_c": temperature_stats["mean"],
        "phase_capacity_ah": phase_capacity_ah,
        "phase_energy_wh": phase_energy_wh,
        "voltage_source": "measured",
        "current_source": "reconstructed",
        "temperature_source": "measured" if temperature_is_measured else "nominal_constant",
        "capacity_source": "measured",
        "energy_source": "integrated_power",
        "chemistry": "LCO",
        "form_factor": "pouch",
        "manufacturer": "Kokam",
        "nominal_voltage_v": np.nan,
        "ambient_temperature_c": AMBIENT_TEMPERATURE_C,
        "charge_protocol": "C1ch characterization at +0.740 A; current reconstructed from Oxford dataset description",
        "discharge_protocol": "C1dc characterization at -0.740 A; current reconstructed from Oxford dataset description",
        "c_rate_charge": 1.0 if phase == "charge" else np.nan,
        "c_rate_discharge": 1.0 if phase == "discharge" else np.nan,
        "dod": np.nan,
        "policy_readable": (
            "Oxford Battery Degradation Dataset 1; C1 characterization cycles; "
            f"source_phase={source_phase_name}; nominal capacity 0.740 Ah; chamber temperature 40 C"
        ),
        "barcode": "Kokam_SLPB533459H4_740mAh",
        "is_valid_cycle": False,
        "is_model_usable": False,
        "exclude_reason": "not_evaluated_in_extraction_stage",
        "cleaning_warnings": extraction_warnings,
        "n_removed_points": 0,
        "trim_start_index": 0,
        "trim_end_index": int(max(len(time_s) - 1, 0)),
        "has_voltage": True,
        "has_current": True,
        "has_temperature": temperature_is_measured,
        "has_capacity": True,
        "has_energy": True,
        "adapter_version": ADAPTER_VERSION,
        "cleaning_version": "not_cleaned",
        "created_at": now_iso(),
    }
    return ensure_python_scalars(row)


def extract_ora_cycles(
    *,
    mat_path: Path,
    base_dir: Path,
    round_vectors: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not mat_path.exists():
        raise FileNotFoundError(f"Raw Oxford MAT file not found: {mat_path}")

    mat_data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    cells = get_cell_variables(mat_data)
    if not cells:
        available_keys = [key for key in mat_data.keys() if not key.startswith("__")]
        raise RuntimeError(f"No CellN variables were found. Available variables: {available_keys}")

    charge_rows: list[dict[str, Any]] = []
    discharge_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []

    for cell_name, cell_number, cell_obj in tqdm(cells, desc="Cells", unit="cell"):
        cell_fields = fieldnames(cell_obj)
        cycle_fields = [field for field in cell_fields if field.lower().startswith("cyc")]
        cycle_fields = sorted(cycle_fields, key=sort_key_by_embedded_number)

        cycle_numbers = [infer_number_from_name(field) for field in cycle_fields if infer_number_from_name(field) is not None]
        cycle_life_proxy = float(max(cycle_numbers)) if cycle_numbers else np.nan

        extracted_charge = 0
        extracted_discharge = 0
        skipped = 0

        iterator = tqdm(cycle_fields, desc=f"{cell_name} cycles", unit="cycle", leave=False)
        for local_index, cycle_field in enumerate(iterator, start=1):
            try:
                cycle_obj = get_field(cell_obj, cycle_field)
                for source_phase_name in (PRIMARY_CHARGE_PHASE, PRIMARY_DISCHARGE_PHASE):
                    if not has_field(cycle_obj, source_phase_name):
                        errors.append({
                            "cell": cell_name,
                            "cycle": cycle_field,
                            "source_phase": source_phase_name,
                            "error": "phase_missing",
                        })
                        skipped += 1
                        continue

                    phase_obj = get_field_case_insensitive(cycle_obj, source_phase_name)
                    phase_vectors, warnings = extract_phase_vectors(phase_obj, source_phase_name)
                    if phase_vectors is None:
                        errors.append({
                            "cell": cell_name,
                            "cycle": cycle_field,
                            "source_phase": source_phase_name,
                            "error": "phase_vector_extraction_failed",
                            "warnings": warnings,
                        })
                        skipped += 1
                        continue

                    row = build_extracted_row(
                        base_dir=base_dir,
                        mat_path=mat_path,
                        cell_name=cell_name,
                        cell_number=cell_number,
                        cycle_field=cycle_field,
                        cycle_index=local_index,
                        cycle_life_proxy=cycle_life_proxy,
                        source_phase_name=source_phase_name,
                        phase_vectors=phase_vectors,
                        extraction_warnings=warnings,
                        round_vectors=round_vectors,
                    )

                    if row["phase"] == "charge":
                        charge_rows.append(row)
                        extracted_charge += 1
                    elif row["phase"] == "discharge":
                        discharge_rows.append(row)
                        extracted_discharge += 1

            except Exception as exc:
                errors.append({
                    "cell": cell_name,
                    "cycle": cycle_field,
                    "source_phase": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                skipped += 1

        cell_summaries.append({
            "cell_name": cell_name,
            "battery_id": f"{DATASET_NAME}_Cell{cell_number:02d}",
            "battery_index": cell_number,
            "n_cycle_fields": len(cycle_fields),
            "n_charge_rows_extracted": extracted_charge,
            "n_discharge_rows_extracted": extracted_discharge,
            "n_skipped_phase_segments": skipped,
            "cycle_life_proxy": cycle_life_proxy,
            "first_cycle_field": cycle_fields[0] if cycle_fields else None,
            "last_cycle_field": cycle_fields[-1] if cycle_fields else None,
        })

    charge_df = reorder_standard_columns(pd.DataFrame(charge_rows)) if charge_rows else empty_standard_dataframe()
    discharge_df = reorder_standard_columns(pd.DataFrame(discharge_rows)) if discharge_rows else empty_standard_dataframe()

    if not charge_df.empty:
        charge_df = charge_df.sort_values(["battery_id", "cycle_number", "cycle_index"]).reset_index(drop=True)
    if not discharge_df.empty:
        discharge_df = discharge_df.sort_values(["battery_id", "cycle_number", "cycle_index"]).reset_index(drop=True)

    report = {
        "dataset": DATASET_NAME,
        "adapter_version": ADAPTER_VERSION,
        "source_file": str(mat_path),
        "source_path": make_relative_path(mat_path, base_dir),
        "file_size_mb": round(mat_path.stat().st_size / (1024 ** 2), 3),
        "n_cells_found": len(cells),
        "cells_found": [cell[0] for cell in cells],
        "n_charge_rows_extracted": int(len(charge_df)),
        "n_discharge_rows_extracted": int(len(discharge_df)),
        "n_errors": int(len(errors)),
        "cell_summaries": cell_summaries,
        "errors": errors[:1000],
        "notes": [
            "C1ch and C1dc were extracted as independent phase-specific rows.",
            "The clean stage recomputes EOL, targets, validity, and model-usability flags.",
            f"Current was reconstructed using +/-{C1_CURRENT_A:.3f} A for C1 characterization phases.",
            f"Nominal capacity is {NOMINAL_CAPACITY_AH:.3f} Ah and nominal EOL is {EOL_CAPACITY_AH:.3f} Ah.",
            "Raw Oxford time vectors are converted from fractional days to seconds.",
        "tqdm progress bars are enabled for cell and cycle loops.",
        ],
    }

    return charge_df, discharge_df, report


def parse_args() -> argparse.Namespace:
    default_base_dir = infer_default_base_dir()
    parser = argparse.ArgumentParser(description="Extract Oxford/ORA cycles into separate charge and discharge parquet files.")
    parser.add_argument("--base-dir", type=str, default=str(default_base_dir), help="Project root directory.")
    parser.add_argument(
        "--mat-file",
        type=str,
        default=str(default_base_dir / "data" / "raw" / "ORA" / "Oxford_Battery_Degradation_Dataset_1.mat"),
        help="Path to Oxford_Battery_Degradation_Dataset_1.mat.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(default_base_dir / "data" / "interim" / DATASET_FOLDER / "extracted"),
        help="Directory where extracted parquet files will be saved.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=str(default_base_dir / "data" / "reports" / DATASET_FOLDER),
        help="Directory where extraction reports will be saved.",
    )
    parser.add_argument("--round-vectors", type=int, default=None, help="Optional decimal places for vector rounding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    mat_path = Path(args.mat_file)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)

    ensure_dir(out_dir)
    ensure_dir(report_dir)

    charge_df, discharge_df, report = extract_ora_cycles(
        mat_path=mat_path,
        base_dir=base_dir,
        round_vectors=args.round_vectors,
    )

    charge_file = out_dir / "ora_charge_extracted_cycles.parquet"
    discharge_file = out_dir / "ora_discharge_extracted_cycles.parquet"
    charge_df.to_parquet(charge_file, index=False)
    discharge_df.to_parquet(discharge_file, index=False)

    summary_rows = []
    for phase, df in [("charge", charge_df), ("discharge", discharge_df)]:
        if df.empty:
            summary_rows.append({"phase": phase, "n_rows": 0, "n_batteries": 0, "n_model_usable_placeholder": 0})
        else:
            summary_rows.append({"phase": phase, "n_rows": int(len(df)), "n_batteries": int(df["battery_id"].nunique()), "n_model_usable_placeholder": 0})
    pd.DataFrame(summary_rows).to_csv(report_dir / "extraction_summary.csv", index=False, encoding="utf-8-sig")
    write_json(report_dir / "extraction_report.json", report)

    print("\nOxford/ORA extraction completed.")
    print(f"Charge extracted file:    {charge_file}")
    print(f"Discharge extracted file: {discharge_file}")
    print(f"Extraction report:        {report_dir / 'extraction_report.json'}")
    print(f"Charge rows: {len(charge_df)} | Discharge rows: {len(discharge_df)}")


if __name__ == "__main__":
    main()
