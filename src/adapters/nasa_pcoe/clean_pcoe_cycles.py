"""
Clean NASA PCoE extracted charge and discharge cycles into the official schema.

Outputs
-------
data/interim/pcoe/clean/pcoe_charge_clean_cycles.parquet
data/interim/pcoe/clean/pcoe_discharge_clean_cycles.parquet
data/interim/pcoe/clean/pcoe_charge_rejected_cycles.csv
data/interim/pcoe/clean/pcoe_discharge_rejected_cycles.csv
data/reports/pcoe/cleaning_report.json

Main conventions
----------------
- Charge current is positive.
- Discharge current is negative.
- capacity_ah is non-negative accumulated phase capacity.
- cycle_life is estimated from the discharge branch by default, using an
  EOL threshold of the robust initial discharge capacity, using a persistence check to avoid false EOL caused by isolated anomalous cycles.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

try:
    from .pcoe_schema import (
        ADAPTER_VERSION,
        CLEANING_VERSION,
        DATASET_NAME,
        DEFAULT_BASE_DIR,
        STANDARD_CYCLE_COLUMNS,
    )
except ImportError:  # Allows direct execution from the adapter folder.
    from pcoe_schema import (  # type: ignore
        ADAPTER_VERSION,
        CLEANING_VERSION,
        DATASET_NAME,
        DEFAULT_BASE_DIR,
        STANDARD_CYCLE_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Parameters and constants
# ---------------------------------------------------------------------------

NOMINAL_CAPACITY_AH = 2.0
CHARGE_PROTOCOL = "CC-CV charge at 1.5 A until 4.2 V with 20 mA cutoff when available in source metadata"
DISCHARGE_PROTOCOL = "NASA PCoE discharge protocol; current and cutoff voltage depend on the battery group"

# Conservative voltage bounds for NASA PCoE 18650 cells. Values outside this
# interval are treated as instrumentation/transient edges and removed before
# computing capacity and scalar statistics.
VOLTAGE_HARD_MIN_V = 1.5
VOLTAGE_HARD_MAX_V = 4.35
MIN_POINTS = 20
MIN_QV_VOLTAGE_RANGE_V = 0.05
MIN_UNIQUE_VOLTAGE_POINTS = 5
POST_CUTOFF_REBOUND_MIN_RISE_V = 0.03
POST_CUTOFF_REBOUND_MIN_POINTS = 3
POST_CHARGE_DROP_MIN_V = 0.03
POST_CHARGE_DROP_MIN_POINTS = 3
DEFAULT_MIN_INITIAL_CAPACITY_AH = 0.20
DEFAULT_MAX_INITIAL_CAPACITY_AH = 2.30
DEFAULT_EOL_TAIL_FRACTION = 0.80
DEFAULT_MIN_EOL_CYCLE = 10
DEFAULT_MIN_MODEL_CAPACITY_AH = 0.20
DEFAULT_MIN_MODEL_CAPACITY_RATIO = 0.50


# ---------------------------------------------------------------------------
# CLI and paths
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean NASA PCoE extracted cycles into the standard clean-cycle schema."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Project root containing data/interim and data/reports.",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=None,
        help="PCoE interim directory. Default: <base-dir>/data/interim/pcoe.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="PCoE report directory. Default: <base-dir>/data/reports/pcoe.",
    )
    parser.add_argument(
        "--eol-ratio",
        type=float,
        default=0.80,
        help="End-of-life capacity ratio relative to robust initial discharge capacity.",
    )
    parser.add_argument(
        "--eol-min-consecutive",
        type=int,
        default=2,
        help="Minimum consecutive cycles below threshold to declare EOL.",
    )
    parser.add_argument(
        "--initial-capacity-cycles",
        type=int,
        default=5,
        help="Number of early cycles used to estimate robust initial capacity.",
    )
    parser.add_argument(
        "--min-initial-capacity-ah",
        type=float,
        default=DEFAULT_MIN_INITIAL_CAPACITY_AH,
        help="Minimum plausible early-cycle capacity used for initial-capacity estimation.",
    )
    parser.add_argument(
        "--max-initial-capacity-ah",
        type=float,
        default=DEFAULT_MAX_INITIAL_CAPACITY_AH,
        help="Maximum plausible early-cycle capacity used for initial-capacity estimation.",
    )
    parser.add_argument(
        "--eol-tail-fraction",
        type=float,
        default=DEFAULT_EOL_TAIL_FRACTION,
        help="Minimum fraction of the post-candidate tail that must remain below the EOL threshold.",
    )
    parser.add_argument(
        "--min-eol-cycle",
        type=float,
        default=DEFAULT_MIN_EOL_CYCLE,
        help="Do not declare EOL before this cycle number.",
    )
    parser.add_argument(
        "--min-model-capacity-ah",
        type=float,
        default=DEFAULT_MIN_MODEL_CAPACITY_AH,
        help="Minimum absolute phase capacity required for model usability. The cycle remains in the clean file but is not model-usable below this value.",
    )
    parser.add_argument(
        "--min-model-capacity-ratio",
        type=float,
        default=DEFAULT_MIN_MODEL_CAPACITY_RATIO,
        help="Minimum phase_capacity_ah / initial_capacity_ah required for model usability. This removes partial or corrupted cycles before feature extraction.",
    )
    parser.add_argument(
        "--allow-phase-eol-fallback",
        action="store_true",
        help="Allow charge- or phase-derived EOL fallback when discharge-derived EOL is unavailable. Disabled by default for PCoE.",
    )
    parser.add_argument(
        "--keep-post-eol",
        action="store_true",
        help="Keep cycles after EOL as valid but not model-usable. Default behavior also keeps them with flags.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    base_dir = args.base_dir
    interim_dir = args.interim_dir or base_dir / "data" / "interim" / "pcoe"
    reports_dir = args.reports_dir or base_dir / "data" / "reports" / "pcoe"
    extracted_dir = interim_dir / "extracted"
    clean_dir = interim_dir / "clean"

    for path in [clean_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "interim_dir": interim_dir,
        "extracted_dir": extracted_dir,
        "clean_dir": clean_dir,
        "reports_dir": reports_dir,
        "charge_extracted_file": extracted_dir / "pcoe_charge_extracted_cycles.parquet",
        "discharge_extracted_file": extracted_dir / "pcoe_discharge_extracted_cycles.parquet",
        "charge_clean_file": clean_dir / "pcoe_charge_clean_cycles.parquet",
        "discharge_clean_file": clean_dir / "pcoe_discharge_clean_cycles.parquet",
        "charge_rejected_file": clean_dir / "pcoe_charge_rejected_cycles.csv",
        "discharge_rejected_file": clean_dir / "pcoe_discharge_rejected_cycles.csv",
        "report_file": reports_dir / "cleaning_report.json",
    }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def as_array(value: Any, *, allow_nan: bool = False) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value.astype(float, copy=False).reshape(-1)
    elif isinstance(value, list):
        array = np.asarray(value, dtype=float).reshape(-1)
    elif pd.isna(value):
        array = np.asarray([], dtype=float)
    else:
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            array = np.asarray([], dtype=float)

    if not allow_nan:
        array = array[np.isfinite(array)]
    return array


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        array = np.asarray(value).squeeze()
        if array.size == 0:
            return default
        out = float(array.item() if array.size == 1 else array.flat[0])
        return out if np.isfinite(out) else default
    except Exception:
        return default


def finite_or_nan(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def vector_to_list(array: np.ndarray) -> list[float]:
    return [float(x) if np.isfinite(x) else np.nan for x in np.asarray(array, dtype=float).reshape(-1)]


def cumulative_trapezoid_abs(y: np.ndarray, x: np.ndarray, scale: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = min(len(y), len(x))
    out = np.zeros(n, dtype=float)
    if n < 2:
        return out
    for idx in range(1, n):
        delta_x = x[idx] - x[idx - 1]
        if np.isfinite(delta_x) and delta_x > 0:
            out[idx] = out[idx - 1] + 0.5 * (abs(y[idx]) + abs(y[idx - 1])) * delta_x / scale
        else:
            out[idx] = out[idx - 1]
    return out


def time_is_strictly_increasing(time_s: np.ndarray) -> bool:
    return len(time_s) >= 2 and bool(np.all(np.diff(time_s) > 0))


def align_and_sort_vectors(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    temperature_c: np.ndarray,
    ambient_temperature_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, bool, list[str], np.ndarray] | None:
    warnings: list[str] = []

    if len(time_s) == 0 or len(voltage_v) == 0 or len(current_a) == 0:
        return None

    n = min(len(time_s), len(voltage_v), len(current_a))
    time_s = time_s[:n]
    voltage_v = voltage_v[:n]
    current_a = current_a[:n]

    if len(temperature_c) >= n:
        temperature_c = temperature_c[:n]
        temperature_source = "measured"
        has_temperature = True
    elif len(temperature_c) > 0:
        m = min(n, len(temperature_c))
        time_s = time_s[:m]
        voltage_v = voltage_v[:m]
        current_a = current_a[:m]
        temperature_c = temperature_c[:m]
        n = m
        temperature_source = "measured"
        has_temperature = True
        warnings.append("temperature_vector_shorter_than_main_signals")
    elif np.isfinite(ambient_temperature_c):
        temperature_c = np.full(n, ambient_temperature_c, dtype=float)
        temperature_source = "nominal_constant"
        has_temperature = False
        warnings.append("temperature_filled_from_ambient_temperature")
    else:
        temperature_c = np.full(n, np.nan, dtype=float)
        temperature_source = "unknown_nan"
        has_temperature = False
        warnings.append("temperature_filled_with_nan")

    finite_mask = np.isfinite(time_s) & np.isfinite(voltage_v) & np.isfinite(current_a)
    # Temperature NaN is allowed when unavailable.
    if not finite_mask.any():
        return None

    original_indices = np.arange(n)[finite_mask]
    time_s = time_s[finite_mask]
    voltage_v = voltage_v[finite_mask]
    current_a = current_a[finite_mask]
    temperature_c = temperature_c[finite_mask]

    order = np.argsort(time_s, kind="mergesort")
    time_s = time_s[order]
    voltage_v = voltage_v[order]
    current_a = current_a[order]
    temperature_c = temperature_c[order]
    original_indices = original_indices[order]

    unique_times, unique_positions = np.unique(time_s, return_index=True)
    unique_positions = np.sort(unique_positions)
    if len(unique_positions) < len(time_s):
        warnings.append("duplicated_time_samples_removed")

    time_s = time_s[unique_positions]
    voltage_v = voltage_v[unique_positions]
    current_a = current_a[unique_positions]
    temperature_c = temperature_c[unique_positions]
    original_indices = original_indices[unique_positions]

    time_s = time_s - time_s[0]

    return (
        time_s,
        voltage_v,
        current_a,
        temperature_c,
        temperature_source,
        has_temperature,
        warnings,
        original_indices,
    )


def trim_nonphysical_edges(
    *,
    phase: str,
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    temperature_c: np.ndarray,
    original_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Trim voltage outliers and obvious non-physical terminal edges.

    NASA PCoE files can contain isolated low/high voltage spikes or a short
    terminal segment after the useful electrochemical phase. Discharge cycles
    may contain voltage rebound after the cutoff voltage. Charge cycles may
    contain a terminal voltage drop after the high-voltage plateau. Both edges
    corrupt Q(V) interpolation and early EOL estimation if kept.
    """
    warnings: list[str] = []

    hard_mask = (
        np.isfinite(voltage_v)
        & (voltage_v >= VOLTAGE_HARD_MIN_V)
        & (voltage_v <= VOLTAGE_HARD_MAX_V)
    )
    if hard_mask.sum() < len(voltage_v):
        warnings.append("voltage_hard_bounds_removed")

    if hard_mask.sum() == 0:
        return time_s[:0], voltage_v[:0], current_a[:0], temperature_c[:0], original_indices[:0], warnings

    keep_positions = np.where(hard_mask)[0]
    start_pos = int(keep_positions[0])
    end_pos = int(keep_positions[-1])

    local_voltage = voltage_v[start_pos : end_pos + 1]

    if phase == "discharge" and end_pos > start_pos:
        # Keep the useful segment until the minimum-voltage cutoff when a
        # post-cutoff rebound is detected.
        if len(local_voltage) >= POST_CUTOFF_REBOUND_MIN_POINTS + 2:
            min_local_idx = int(np.nanargmin(local_voltage))
            post_min = local_voltage[min_local_idx:]
            if len(post_min) >= POST_CUTOFF_REBOUND_MIN_POINTS:
                rebound_rise = float(np.nanmax(post_min) - post_min[0])
                if rebound_rise >= POST_CUTOFF_REBOUND_MIN_RISE_V:
                    end_pos = start_pos + min_local_idx
                    warnings.append("post_cutoff_voltage_rebound_trimmed")

    elif phase == "charge" and end_pos > start_pos:
        # Keep the charge segment up to the high-voltage plateau. If the signal
        # drops after the maximum voltage, remove the terminal non-charge edge.
        if len(local_voltage) >= POST_CHARGE_DROP_MIN_POINTS + 2:
            max_local_idx = int(np.nanargmax(local_voltage))
            post_peak = local_voltage[max_local_idx:]
            if len(post_peak) >= POST_CHARGE_DROP_MIN_POINTS:
                drop_after_peak = float(post_peak[0] - np.nanmin(post_peak))
                if drop_after_peak >= POST_CHARGE_DROP_MIN_V:
                    below = np.where(post_peak <= post_peak[0] - POST_CHARGE_DROP_MIN_V)[0]
                    if len(below) > 0:
                        # Retain samples before the first terminal drop.
                        end_pos = start_pos + max_local_idx + max(0, int(below[0]) - 1)
                    else:
                        end_pos = start_pos + max_local_idx
                    warnings.append("post_charge_voltage_drop_trimmed")

    time_s = time_s[start_pos : end_pos + 1]
    voltage_v = voltage_v[start_pos : end_pos + 1]
    current_a = current_a[start_pos : end_pos + 1]
    temperature_c = temperature_c[start_pos : end_pos + 1]
    original_indices = original_indices[start_pos : end_pos + 1]

    # Apply the voltage bounds again after terminal trimming as a fail-safe.
    if len(voltage_v) > 0:
        mask = (
            np.isfinite(voltage_v)
            & (voltage_v >= VOLTAGE_HARD_MIN_V)
            & (voltage_v <= VOLTAGE_HARD_MAX_V)
        )
        if mask.sum() < len(voltage_v):
            warnings.append("voltage_hard_bounds_removed_after_terminal_trim")
        time_s = time_s[mask]
        voltage_v = voltage_v[mask]
        current_a = current_a[mask]
        temperature_c = temperature_c[mask]
        original_indices = original_indices[mask]

    if len(time_s) > 0:
        time_s = time_s - time_s[0]

    return time_s, voltage_v, current_a, temperature_c, original_indices, warnings

def can_interpolate_qv_curve(voltage_v: np.ndarray, capacity_ah: np.ndarray) -> bool:
    if len(voltage_v) < MIN_POINTS or len(capacity_ah) < MIN_POINTS:
        return False
    if not np.isfinite(voltage_v).any() or not np.isfinite(capacity_ah).any():
        return False
    voltage_range = float(np.nanmax(voltage_v) - np.nanmin(voltage_v))
    if voltage_range < MIN_QV_VOLTAGE_RANGE_V:
        return False
    unique_voltage = np.unique(np.round(voltage_v[np.isfinite(voltage_v)], decimals=4))
    return len(unique_voltage) >= MIN_UNIQUE_VOLTAGE_POINTS


def summarize_temperature(temperature_c: np.ndarray) -> tuple[float, float, float]:
    finite = temperature_c[np.isfinite(temperature_c)]
    if len(finite) == 0:
        return np.nan, np.nan, np.nan
    return float(np.min(finite)), float(np.max(finite)), float(np.mean(finite))


def invalid_reason(
    *,
    n_points: int,
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    phase_capacity_ah: float,
    duration_s: float,
) -> str:
    if n_points < MIN_POINTS:
        return "insufficient_points"
    if not time_is_strictly_increasing(time_s):
        return "invalid_or_non_increasing_time"
    if len(voltage_v) == 0 or not np.isfinite(voltage_v).any():
        return "missing_voltage"
    if len(current_a) == 0 or not np.isfinite(current_a).any():
        return "missing_current_and_capacity"
    if not np.isfinite(phase_capacity_ah) or phase_capacity_ah <= 0:
        return "invalid_or_zero_capacity"
    if not np.isfinite(duration_s) or duration_s <= 0:
        return "invalid_duration"
    if float(np.nanmax(voltage_v) - np.nanmin(voltage_v)) <= 0:
        return "invalid_voltage_range"
    return ""


# ---------------------------------------------------------------------------
# Row cleaning
# ---------------------------------------------------------------------------


def clean_one_row(row: pd.Series) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    phase = str(row.get("phase", "")).lower().strip()
    if phase not in {"charge", "discharge"}:
        rejected = rejected_from_row(row, "invalid_phase")
        return None, rejected

    ambient_temperature_c = safe_float(row.get("ambient_temperature_c", np.nan))

    aligned = align_and_sort_vectors(
        time_s=as_array(row.get("time_s", [])),
        voltage_v=as_array(row.get("voltage_v", [])),
        current_a=as_array(row.get("current_a", [])),
        temperature_c=as_array(row.get("temperature_c", []), allow_nan=True),
        ambient_temperature_c=ambient_temperature_c,
    )
    if aligned is None:
        rejected = rejected_from_row(row, "missing_required_signal_vectors")
        return None, rejected

    (
        time_s,
        voltage_v,
        current_a,
        temperature_c,
        temperature_source,
        has_temperature,
        warnings,
        original_indices,
    ) = aligned

    time_s, voltage_v, current_a, temperature_c, original_indices, trim_warnings = trim_nonphysical_edges(
        phase=phase,
        time_s=time_s,
        voltage_v=voltage_v,
        current_a=current_a,
        temperature_c=temperature_c,
        original_indices=original_indices,
    )
    warnings.extend(trim_warnings)

    n_removed_points = int(max(0, int(row.get("n_points_raw", len(as_array(row.get("time_s", []))))) - len(time_s)))
    trim_start_index = int(original_indices[0]) if len(original_indices) else -1
    trim_end_index = int(original_indices[-1]) if len(original_indices) else -1

    if len(time_s) == 0:
        rejected = rejected_from_row(row, "empty_cycle_after_trimming")
        return None, rejected

    # Current sign convention.
    if phase == "charge":
        current_a = np.abs(current_a)
    else:
        current_a = -np.abs(current_a)

    capacity_ah = cumulative_trapezoid_abs(current_a, time_s, scale=3600.0)
    capacity_source = "integrated_current"

    raw_capacity_summary_ah = safe_float(row.get("raw_capacity_summary_ah", np.nan))
    if phase == "discharge" and np.isfinite(raw_capacity_summary_ah) and raw_capacity_summary_ah > 0:
        if len(capacity_ah) > 0 and capacity_ah[-1] > 0:
            scale_factor = raw_capacity_summary_ah / capacity_ah[-1]
            if np.isfinite(scale_factor) and scale_factor > 0:
                capacity_ah = capacity_ah * scale_factor
                warnings.append("integrated_capacity_scaled_to_source_summary_capacity")
        else:
            capacity_ah = np.linspace(0.0, raw_capacity_summary_ah, len(time_s))
            capacity_source = "summary_value_only"
            warnings.append("capacity_vector_reconstructed_from_summary_capacity_only")

    power_w = voltage_v * current_a
    energy_wh = cumulative_trapezoid_abs(power_w, time_s, scale=3600.0)

    n_points = int(len(time_s))
    duration_s = float(time_s[-1]) if n_points else np.nan
    sample_period_median_s = float(np.median(np.diff(time_s))) if n_points >= 2 else np.nan
    temperature_min_c, temperature_max_c, temperature_mean_c = summarize_temperature(temperature_c)
    phase_capacity_ah = float(capacity_ah[-1]) if len(capacity_ah) else np.nan
    phase_energy_wh = float(energy_wh[-1]) if len(energy_wh) else np.nan

    reason = invalid_reason(
        n_points=n_points,
        time_s=time_s,
        voltage_v=voltage_v,
        current_a=current_a,
        phase_capacity_ah=phase_capacity_ah,
        duration_s=duration_s,
    )
    is_valid_cycle = reason == ""

    clean_row: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "source_file": str(row.get("source_file", "")),
        "source_path": str(row.get("source_path", "")),
        "battery_id": str(row.get("battery_id", "")),
        "battery_index": int(safe_float(row.get("battery_index", -1), -1)),
        "cell_id_original": str(row.get("cell_id_original", "")),
        "cycle_index": np.nan,  # Reassigned after all rows are cleaned.
        "cycle_number": float(safe_float(row.get("cycle_number", np.nan))),
        "source_cycle_id": str(row.get("source_cycle_id", "")),
        "cycle_pair_id": str(row.get("cycle_pair_id", "")),
        "phase": phase,
        "cycle_life": np.nan,
        "remaining_cycles": np.nan,
        "remaining_cycles_norm": np.nan,
        "eol_criterion": "80pct_initial_discharge_capacity",
        "eol_capacity_ah": np.nan,
        "initial_capacity_ah": np.nan,
        "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "soh": np.nan,
        "time_s": vector_to_list(time_s),
        "voltage_v": vector_to_list(voltage_v),
        "current_a": vector_to_list(current_a),
        "temperature_c": vector_to_list(temperature_c),
        "capacity_ah": vector_to_list(capacity_ah),
        "energy_wh": vector_to_list(energy_wh),
        "power_w": vector_to_list(power_w),
        "n_points": n_points,
        "duration_s": duration_s,
        "sample_period_median_s": sample_period_median_s,
        "voltage_min_v": float(np.nanmin(voltage_v)) if n_points else np.nan,
        "voltage_max_v": float(np.nanmax(voltage_v)) if n_points else np.nan,
        "voltage_start_v": float(voltage_v[0]) if n_points else np.nan,
        "voltage_end_v": float(voltage_v[-1]) if n_points else np.nan,
        "current_min_a": float(np.nanmin(current_a)) if n_points else np.nan,
        "current_max_a": float(np.nanmax(current_a)) if n_points else np.nan,
        "current_mean_a": float(np.nanmean(current_a)) if n_points else np.nan,
        "temperature_min_c": temperature_min_c,
        "temperature_max_c": temperature_max_c,
        "temperature_mean_c": temperature_mean_c,
        "phase_capacity_ah": phase_capacity_ah,
        "phase_energy_wh": phase_energy_wh,
        "voltage_source": "measured",
        "current_source": "measured",
        "temperature_source": temperature_source,
        "capacity_source": capacity_source,
        "energy_source": "integrated_power",
        "chemistry": "unknown",
        "form_factor": "18650",
        "manufacturer": "unknown",
        "nominal_voltage_v": np.nan,
        "ambient_temperature_c": ambient_temperature_c,
        "charge_protocol": CHARGE_PROTOCOL,
        "discharge_protocol": DISCHARGE_PROTOCOL,
        "c_rate_charge": 0.75,
        "c_rate_discharge": np.nan,
        "dod": np.nan,
        "policy_readable": str(row.get("policy_readable", "")),
        "barcode": str(row.get("barcode", "")),
        "is_valid_cycle": bool(is_valid_cycle),
        "is_model_usable": False,
        "exclude_reason": reason,
        "cleaning_warnings": warnings,
        "n_removed_points": n_removed_points,
        "trim_start_index": trim_start_index,
        "trim_end_index": trim_end_index,
        "has_voltage": bool(n_points > 0 and np.isfinite(voltage_v).any()),
        "has_current": bool(n_points > 0 and np.isfinite(current_a).any()),
        "has_temperature": bool(has_temperature),
        "has_capacity": bool(np.isfinite(phase_capacity_ah) and phase_capacity_ah > 0),
        "has_energy": bool(np.isfinite(phase_energy_wh) and phase_energy_wh > 0),
        "adapter_version": str(row.get("adapter_version", ADAPTER_VERSION)),
        "cleaning_version": CLEANING_VERSION,
        "created_at": utc_now(),
    }

    if not is_valid_cycle:
        rejected = rejected_from_clean_row(clean_row)
        return None, rejected

    return clean_row, None


def rejected_from_row(row: pd.Series, reason: str) -> dict[str, Any]:
    return {
        "dataset": DATASET_NAME,
        "source_file": str(row.get("source_file", "")),
        "source_path": str(row.get("source_path", "")),
        "battery_id": str(row.get("battery_id", "")),
        "battery_index": safe_float(row.get("battery_index", np.nan)),
        "cell_id_original": str(row.get("cell_id_original", "")),
        "cycle_number": safe_float(row.get("cycle_number", np.nan)),
        "source_cycle_id": str(row.get("source_cycle_id", "")),
        "cycle_pair_id": str(row.get("cycle_pair_id", "")),
        "phase": str(row.get("phase", "")),
        "exclude_reason": reason,
        "n_points_raw": safe_float(row.get("n_points_raw", np.nan)),
        "created_at": utc_now(),
    }


def rejected_from_clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", DATASET_NAME),
        "source_file": row.get("source_file", ""),
        "source_path": row.get("source_path", ""),
        "battery_id": row.get("battery_id", ""),
        "battery_index": row.get("battery_index", np.nan),
        "cell_id_original": row.get("cell_id_original", ""),
        "cycle_number": row.get("cycle_number", np.nan),
        "source_cycle_id": row.get("source_cycle_id", ""),
        "cycle_pair_id": row.get("cycle_pair_id", ""),
        "phase": row.get("phase", ""),
        "exclude_reason": row.get("exclude_reason", ""),
        "n_points": row.get("n_points", np.nan),
        "phase_capacity_ah": row.get("phase_capacity_ah", np.nan),
        "voltage_min_v": row.get("voltage_min_v", np.nan),
        "voltage_max_v": row.get("voltage_max_v", np.nan),
        "created_at": utc_now(),
    }


# ---------------------------------------------------------------------------
# EOL and model-usability assignment
# ---------------------------------------------------------------------------


def robust_initial_capacity(
    group: pd.DataFrame,
    n_cycles: int,
    *,
    min_capacity_ah: float = DEFAULT_MIN_INITIAL_CAPACITY_AH,
    max_capacity_ah: float = DEFAULT_MAX_INITIAL_CAPACITY_AH,
) -> float:
    """Estimate initial capacity using early physically plausible cycles."""
    early = group.sort_values("cycle_number").head(max(1, n_cycles * 2))
    values = pd.to_numeric(early["phase_capacity_ah"], errors="coerce")
    values = values[np.isfinite(values) & (values > 0)]
    plausible = values[(values >= min_capacity_ah) & (values <= max_capacity_ah)]
    if len(plausible) >= max(1, min(n_cycles, 3)):
        values = plausible.head(max(1, n_cycles))
    else:
        values = values.head(max(1, n_cycles))
    if len(values) == 0:
        return np.nan
    return float(np.median(values))


def find_cycle_life(
    group: pd.DataFrame,
    initial_capacity_ah: float,
    eol_ratio: float,
    min_consecutive: int,
    *,
    min_eol_cycle: float = DEFAULT_MIN_EOL_CYCLE,
    eol_tail_fraction: float = DEFAULT_EOL_TAIL_FRACTION,
) -> tuple[float, float]:
    """Find a persistent EOL crossing.

    The previous v1 logic returned the first short run below threshold. This can
    falsely mark EOL at cycle 1 when early anomalous cycles occur and the capacity
    subsequently recovers. Here a candidate EOL must: (i) occur after a minimum
    cycle number, (ii) contain the requested consecutive below-threshold samples,
    and (iii) remain below threshold for most of the remaining tail.
    """
    if not np.isfinite(initial_capacity_ah) or initial_capacity_ah <= 0:
        return np.nan, np.nan

    threshold = float(initial_capacity_ah * eol_ratio)
    ordered = group.sort_values("cycle_number").copy()
    capacity = pd.to_numeric(ordered["phase_capacity_ah"], errors="coerce").to_numpy(dtype=float)
    cycle_numbers = pd.to_numeric(ordered["cycle_number"], errors="coerce").to_numpy(dtype=float)

    finite = np.isfinite(capacity) & np.isfinite(cycle_numbers)
    capacity = capacity[finite]
    cycle_numbers = cycle_numbers[finite]
    if len(capacity) == 0:
        return np.nan, threshold

    below = capacity <= threshold
    min_consecutive = max(1, int(min_consecutive))
    eol_tail_fraction = float(np.clip(eol_tail_fraction, 0.0, 1.0))

    for i in range(0, len(capacity) - min_consecutive + 1):
        candidate_cycle = cycle_numbers[i]
        if candidate_cycle < min_eol_cycle:
            continue
        if not np.all(below[i : i + min_consecutive]):
            continue
        tail = below[i:]
        if len(tail) == 0:
            continue
        tail_fraction = float(np.mean(tail))
        # Allow end-of-series crossings even when the tail is short.
        near_series_end = len(tail) <= max(min_consecutive + 1, 3)
        if tail_fraction >= eol_tail_fraction or near_series_end:
            return float(candidate_cycle), threshold

    return np.nan, threshold


def assign_lifetime_targets(
    df: pd.DataFrame,
    *,
    eol_ratio: float,
    eol_min_consecutive: int,
    initial_capacity_cycles: int,
    min_initial_capacity_ah: float = DEFAULT_MIN_INITIAL_CAPACITY_AH,
    max_initial_capacity_ah: float = DEFAULT_MAX_INITIAL_CAPACITY_AH,
    eol_tail_fraction: float = DEFAULT_EOL_TAIL_FRACTION,
    min_eol_cycle: float = DEFAULT_MIN_EOL_CYCLE,
    min_model_capacity_ah: float = DEFAULT_MIN_MODEL_CAPACITY_AH,
    min_model_capacity_ratio: float = DEFAULT_MIN_MODEL_CAPACITY_RATIO,
    allow_phase_eol_fallback: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    eol_criterion = f"{int(round(eol_ratio * 100))}pct_initial_discharge_capacity_persistent"

    battery_target_info: dict[str, dict[str, float]] = {}
    discharge = df[df["phase"] == "discharge"].copy()
    for battery_id, group in discharge.groupby("battery_id"):
        initial = robust_initial_capacity(
            group,
            initial_capacity_cycles,
            min_capacity_ah=min_initial_capacity_ah,
            max_capacity_ah=max_initial_capacity_ah,
        )
        cycle_life, threshold = find_cycle_life(
            group,
            initial,
            eol_ratio,
            eol_min_consecutive,
            min_eol_cycle=min_eol_cycle,
            eol_tail_fraction=eol_tail_fraction,
        )
        battery_target_info[str(battery_id)] = {
            "target_initial_capacity_ah": initial,
            "cycle_life": cycle_life,
            "eol_capacity_ah": threshold,
        }

    # Assign phase-specific initial capacity for SOH only. The RUL target remains
    # battery-level and is derived from the discharge branch unless explicitly
    # requested otherwise.
    for (battery_id, phase), group in df.groupby(["battery_id", "phase"]):
        phase_initial = robust_initial_capacity(
            group,
            initial_capacity_cycles,
            min_capacity_ah=min_initial_capacity_ah,
            max_capacity_ah=max_initial_capacity_ah,
        )
        mask = (df["battery_id"] == battery_id) & (df["phase"] == phase)
        df.loc[mask, "initial_capacity_ah"] = phase_initial
        if np.isfinite(phase_initial) and phase_initial > 0:
            df.loc[mask, "soh"] = df.loc[mask, "phase_capacity_ah"] / phase_initial

    # Assign battery-level EOL from discharge branch to both charge and discharge.
    for battery_id, info in battery_target_info.items():
        mask = df["battery_id"] == battery_id
        df.loc[mask, "cycle_life"] = info["cycle_life"]
        df.loc[mask, "eol_capacity_ah"] = info["eol_capacity_ah"]
        df.loc[mask, "eol_criterion"] = eol_criterion

    # Optional fallback retained for exploratory runs only. It is disabled by
    # default because it can fabricate apparent EOL from charge-cycle artifacts.
    if allow_phase_eol_fallback:
        for (battery_id, phase), group in df.groupby(["battery_id", "phase"]):
            current_life = safe_float(df.loc[df["battery_id"] == battery_id, "cycle_life"].iloc[0], np.nan)
            if np.isfinite(current_life):
                continue
            phase_initial = robust_initial_capacity(
                group,
                initial_capacity_cycles,
                min_capacity_ah=min_initial_capacity_ah,
                max_capacity_ah=max_initial_capacity_ah,
            )
            phase_life, phase_threshold = find_cycle_life(
                group,
                phase_initial,
                eol_ratio,
                eol_min_consecutive,
                min_eol_cycle=min_eol_cycle,
                eol_tail_fraction=eol_tail_fraction,
            )
            if np.isfinite(phase_life):
                mask = df["battery_id"] == battery_id
                df.loc[mask, "cycle_life"] = phase_life
                df.loc[mask, "eol_capacity_ah"] = phase_threshold
                df.loc[mask, "eol_criterion"] = f"{int(round(eol_ratio * 100))}pct_initial_{phase}_capacity_persistent_fallback"
                df.loc[mask, "cleaning_warnings"] = df.loc[mask, "cleaning_warnings"].apply(
                    lambda warnings: list(warnings) + ["cycle_life_fallback_to_phase_capacity"]
                )

    df["remaining_cycles"] = df["cycle_life"] - df["cycle_number"]
    df["remaining_cycles_norm"] = df["remaining_cycles"] / df["cycle_life"]

    usable_flags = []
    reasons = []
    for _, row in df.iterrows():
        if not bool(row["is_valid_cycle"]):
            usable_flags.append(False)
            reasons.append(str(row["exclude_reason"]))
            continue
        if not np.isfinite(row["cycle_life"]):
            usable_flags.append(False)
            reasons.append("unknown_cycle_life")
            continue
        if not np.isfinite(row["remaining_cycles"]) or row["remaining_cycles"] <= 0:
            usable_flags.append(False)
            reasons.append("non_positive_remaining_cycles")
            continue

        phase_capacity = safe_float(row.get("phase_capacity_ah", np.nan), np.nan)
        initial_capacity = safe_float(row.get("initial_capacity_ah", np.nan), np.nan)
        eol_capacity = safe_float(row.get("eol_capacity_ah", np.nan), np.nan)
        cycle_number = safe_float(row.get("cycle_number", np.nan), np.nan)
        cycle_life = safe_float(row.get("cycle_life", np.nan), np.nan)

        if not np.isfinite(phase_capacity) or phase_capacity < min_model_capacity_ah:
            usable_flags.append(False)
            reasons.append("abnormally_low_phase_capacity")
            continue
        if np.isfinite(initial_capacity) and initial_capacity > 0:
            capacity_ratio = phase_capacity / initial_capacity
            if np.isfinite(capacity_ratio) and capacity_ratio < min_model_capacity_ratio:
                usable_flags.append(False)
                reasons.append("low_phase_capacity_ratio")
                continue
        if (
            str(row.get("phase", "")).lower() == "discharge"
            and np.isfinite(eol_capacity)
            and np.isfinite(cycle_number)
            and np.isfinite(cycle_life)
            and cycle_number < cycle_life
            and phase_capacity < eol_capacity
        ):
            usable_flags.append(False)
            reasons.append("discharge_capacity_below_eol_before_cycle_life")
            continue
        voltage = as_array(row["voltage_v"])
        capacity = as_array(row["capacity_ah"])
        if not can_interpolate_qv_curve(voltage, capacity):
            usable_flags.append(False)
            reasons.append("insufficient_qv_voltage_range")
            continue
        usable_flags.append(True)
        reasons.append("")

    df["is_model_usable"] = usable_flags
    df["exclude_reason"] = reasons
    return df

def assign_cycle_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["battery_index", "battery_id", "phase", "cycle_number"], kind="mergesort").copy()
    df["cycle_index"] = df.groupby(["battery_id", "phase"]).cumcount().astype(int) + 1
    return df


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    for column in STANDARD_CYCLE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[STANDARD_CYCLE_COLUMNS]


def load_extracted(paths: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for phase, file_path in [
        ("charge", paths["charge_extracted_file"]),
        ("discharge", paths["discharge_extracted_file"]),
    ]:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing extracted {phase} file: {file_path}")
        df = pd.read_parquet(file_path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_outputs(df: pd.DataFrame, rejected: pd.DataFrame, paths: dict[str, Path]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for phase in ["charge", "discharge"]:
        phase_df = df[df["phase"] == phase].copy() if not df.empty else pd.DataFrame(columns=STANDARD_CYCLE_COLUMNS)
        phase_df = assign_cycle_index(phase_df)
        phase_df = enforce_schema(phase_df)

        clean_file = paths[f"{phase}_clean_file"]
        phase_df.to_parquet(clean_file, index=False)

        phase_rejected = rejected[rejected["phase"] == phase].copy() if not rejected.empty else pd.DataFrame()
        not_usable = phase_df[~phase_df["is_model_usable"]].copy()
        not_usable_compact = not_usable[
            [
                "dataset", "source_file", "source_path", "battery_id", "battery_index",
                "cell_id_original", "cycle_number", "source_cycle_id", "cycle_pair_id",
                "phase", "exclude_reason", "n_points", "phase_capacity_ah",
                "voltage_min_v", "voltage_max_v", "created_at",
            ]
        ]
        rejected_full = pd.concat([phase_rejected, not_usable_compact], ignore_index=True, sort=False)
        rejected_file = paths[f"{phase}_rejected_file"]
        rejected_full.to_csv(rejected_file, index=False, encoding="utf-8")

        summary[phase] = {
            "clean_file": str(clean_file),
            "rejected_file": str(rejected_file),
            "n_clean_rows": int(len(phase_df)),
            "n_valid_cycles": int(phase_df["is_valid_cycle"].sum()) if not phase_df.empty else 0,
            "n_model_usable_cycles": int(phase_df["is_model_usable"].sum()) if not phase_df.empty else 0,
            "n_rejected_rows": int(len(rejected_full)),
            "exclude_reason_counts": rejected_full["exclude_reason"].value_counts(dropna=False).to_dict()
            if not rejected_full.empty and "exclude_reason" in rejected_full.columns else {},
        }

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    extracted = load_extracted(paths)
    if extracted.empty:
        raise RuntimeError("No extracted cycles were found to clean.")

    clean_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for _, row in tqdm(extracted.iterrows(), total=len(extracted), desc="Cleaning PCoE cycles", unit="cycle"):
        clean_row, rejected_row = clean_one_row(row)
        if clean_row is not None:
            clean_rows.append(clean_row)
        if rejected_row is not None:
            rejected_rows.append(rejected_row)

    clean_df = pd.DataFrame(clean_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    if clean_df.empty:
        raise RuntimeError("No valid cycles remained after cleaning. Check extracted files and raw signals.")

    clean_df = assign_lifetime_targets(
        clean_df,
        eol_ratio=args.eol_ratio,
        eol_min_consecutive=args.eol_min_consecutive,
        initial_capacity_cycles=args.initial_capacity_cycles,
        min_initial_capacity_ah=args.min_initial_capacity_ah,
        max_initial_capacity_ah=args.max_initial_capacity_ah,
        eol_tail_fraction=args.eol_tail_fraction,
        min_eol_cycle=args.min_eol_cycle,
        min_model_capacity_ah=args.min_model_capacity_ah,
        min_model_capacity_ratio=args.min_model_capacity_ratio,
        allow_phase_eol_fallback=args.allow_phase_eol_fallback,
    )

    output_summary = save_outputs(clean_df, rejected_df, paths)

    report = {
        "cleaning_version": CLEANING_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "created_at": utc_now(),
        "eol_ratio": args.eol_ratio,
        "eol_min_consecutive": args.eol_min_consecutive,
        "initial_capacity_cycles": args.initial_capacity_cycles,
        "min_initial_capacity_ah": args.min_initial_capacity_ah,
        "max_initial_capacity_ah": args.max_initial_capacity_ah,
        "eol_tail_fraction": args.eol_tail_fraction,
        "min_eol_cycle": args.min_eol_cycle,
        "min_model_capacity_ah": args.min_model_capacity_ah,
        "min_model_capacity_ratio": args.min_model_capacity_ratio,
        "allow_phase_eol_fallback": args.allow_phase_eol_fallback,
        "n_extracted_rows": int(len(extracted)),
        "n_valid_clean_rows": int(len(clean_df)),
        "n_initial_rejected_rows": int(len(rejected_df)),
        "outputs": output_summary,
        "battery_summary": clean_df.groupby(["battery_id", "phase"]).agg(
            n_cycles=("cycle_number", "count"),
            n_model_usable=("is_model_usable", "sum"),
            cycle_life=("cycle_life", "max"),
            initial_capacity_ah=("initial_capacity_ah", "max"),
            eol_capacity_ah=("eol_capacity_ah", "max"),
            min_capacity_ah=("phase_capacity_ah", "min"),
            max_capacity_ah=("phase_capacity_ah", "max"),
        ).reset_index().to_dict(orient="records"),
    }

    with open(paths["report_file"], "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("Cleaning completed.")
    print(f"Charge clean file:    {paths['charge_clean_file']}")
    print(f"Discharge clean file: {paths['discharge_clean_file']}")
    print(f"Report: {paths['report_file']}")


if __name__ == "__main__":
    main()
