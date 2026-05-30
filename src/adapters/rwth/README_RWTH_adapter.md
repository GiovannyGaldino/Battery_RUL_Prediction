# RWTH Adapter - Standard Dual Branch Pipeline

This folder contains the reconstructed RWTH adapter in English, following the project standard schema for independent charge and discharge branches.

## Scripts

1. `extract_rwth_cycles.py`
   - Reads `data/raw/RWTH/*.zip`.
   - Extracts defensible active-current segments.
   - Separates charge and discharge using the voltage trend.
   - Writes:
     - `data/interim/rwth/extracted/rwth_charge_extracted_cycles.parquet`
     - `data/interim/rwth/extracted/rwth_discharge_extracted_cycles.parquet`
     - `data/reports/rwth/extraction/rwth_extraction_report.csv`
     - `data/reports/rwth/extraction/rwth_extraction_summary.json`

2. `clean_rwth_cycles.py`
   - Reads the extracted parquet files.
   - Applies the standard current convention: positive for charge, negative for discharge.
   - Reconstructs capacity and energy when required.
   - Trims non-physical discharge rebound and short charge pre-dips.
   - Writes:
     - `data/interim/rwth/clean/rwth_charge_clean_cycles.parquet`
     - `data/interim/rwth/clean/rwth_discharge_clean_cycles.parquet`
     - `data/interim/rwth/clean/rwth_charge_rejected_cycles.csv`
     - `data/interim/rwth/clean/rwth_discharge_rejected_cycles.csv`

3. `audit_rwth_cycles.py`
   - Validates the official schema.
   - Summarizes valid, invalid, and model-usable cycles.
   - Writes:
     - `data/reports/rwth/audit_report.json`
     - `data/reports/rwth/audit_summary.csv`
     - `data/reports/rwth/rwth_audit_anomalies.csv`
     - visual plots under `data/reports/rwth/plots/`

## Run order

```bash
python src/adapters/rwth/extract_rwth_cycles.py
python src/adapters/rwth/clean_rwth_cycles.py
python src/adapters/rwth/audit_rwth_cycles.py
```

For quick tests:

```bash
python src/adapters/rwth/extract_rwth_cycles.py --max-zips 20
python src/adapters/rwth/audit_rwth_cycles.py --skip-plots
```

## Dependencies

```bash
pip install pandas numpy tqdm matplotlib pyarrow
```

`pyarrow` or `fastparquet` is required because the project standard uses parquet outputs.

## Notes specific to RWTH

- The RWTH raw dataset is ZIP-based; each ZIP may contain tabular files with German column names.
- EIS files are skipped.
- The adapter does not force ambiguous active-current regions into the cycle database.
- Partial aging segments are preserved but are not used to infer end of life. EOL inference is based only on full reference cycles when they cross the configured threshold.
- If EOL is not reached by the available full reference cycles, cycles remain valid for auditing but not model-usable because `cycle_life` is unknown.


## Version 1.1 fix

This version fixes a pandas/numpy read-only array issue observed during extraction (`ValueError: assignment destination is read-only`) by explicitly copying numeric arrays before temperature sanitization.


## Version 1.2 note

`clean_rwth_cycles.py` now assigns cycle order using the chronological timestamp embedded in RWTH ZIP file names, instead of lexicographic file-name order. This prevents EOL/RPT files from being sorted before aging files and avoids artificially small `cycle_life` values in the discharge branch.


## Version v1.3

- Fixed missing `import re` in `clean_rwth_cycles.py`, required by RWTH filename timestamp and numeric-token parsing.

## Version v1.4

- Changed the default RWTH target assignment mode to `end_of_sequence`.
- This is necessary because the RWTH discharge branch is dominated by partial variable-DoD aging segments; their local `phase_capacity_ah` is not directly comparable to full-cycle reference capacity and should not be used as a strict 80% EOL trigger.
- The default target now represents remaining observed chronological segments until the end of the available RWTH sequence for each battery and phase.
- The strict capacity-based EOL mode is still available:

```bash
python src/adapters/rwth/clean_rwth_cycles.py --target-mode capacity_eol
```

- A guarded fallback mode is also available:

```bash
python src/adapters/rwth/clean_rwth_cycles.py --target-mode hybrid
```

Use `capacity_eol` only when the analysis is restricted to full reference cycles. Use the default `end_of_sequence` for the standardized dual charge/discharge branch used before feature extraction.
