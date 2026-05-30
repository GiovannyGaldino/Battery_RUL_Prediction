"""
Shared schema definitions for the NASA PCoE adapter.

The clean-cycle files produced by this adapter must follow exactly the
STANDARD_CYCLE_COLUMNS list defined here.
"""

from __future__ import annotations

from pathlib import Path

DATASET_NAME = "PCOE"
ADAPTER_VERSION = "pcoe_adapter_v1.2"
CLEANING_VERSION = "pcoe_cleaning_v1.2"

DEFAULT_BASE_DIR = Path.cwd()

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

EXTRACTED_COLUMNS = [
    "dataset", "source_file", "source_path", "battery_id", "battery_index",
    "cell_id_original", "source_cycle_index", "source_cycle_id", "cycle_pair_id",
    "cycle_pair_number", "cycle_number", "phase", "ambient_temperature_c",
    "matlab_cycle_type", "matlab_test_time", "raw_capacity_summary_ah",
    "time_s", "voltage_v", "current_a", "temperature_c", "voltage_charge_v",
    "current_charge_a", "voltage_load_v", "current_load_a", "n_points_raw",
    "extraction_warnings", "policy_readable", "barcode", "adapter_version",
    "created_at",
]


def validate_clean_schema(columns: list[str]) -> None:
    """Validate clean-cycle columns against the official schema."""
    missing = [column for column in STANDARD_CYCLE_COLUMNS if column not in columns]
    extra = [column for column in columns if column not in STANDARD_CYCLE_COLUMNS]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected columns found: {extra}")
