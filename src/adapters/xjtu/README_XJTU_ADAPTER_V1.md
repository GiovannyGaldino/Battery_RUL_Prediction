# XJTU adapter v1.3

This folder contains the reconstructed XJTU adapter in English.

Version 1.3 keeps the bounded-memory streaming workflow and corrects the initial-capacity/SOH assignment. The cleaner now uses a two-pass streaming strategy: first it computes a robust initial-capacity reference for each battery and phase from valid representative cycles; then it writes cleaned rows using those references without loading the full dataset into memory.

## Files

- `xjtu_schema.py`: shared standard schema, paths, and utilities.
- `extract_xjtu_cycles.py`: extracts charge and discharge cycle segments from raw `.mat` files.
- `clean_xjtu_cycles.py`: cleans, standardizes, validates, computes robust initial capacity/SOH, and writes the official clean-cycle files.
- `audit_xjtu_cycles.py`: audits the clean files and creates JSON/CSV reports and Q(V)/V(Q) plots.
- `run_xjtu_pipeline.py`: optional wrapper to run extraction, cleaning, and audit in sequence.

## Expected input layout

```text
Batteries/data/raw/XJTU/Batch-1/2C_battery-1.mat
Batteries/data/raw/XJTU/Batch-2/*.mat
...
Batteries/data/raw/XJTU/Batch-6/*.mat
```

## Main command

```bash
python src/adapters/xjtu/run_xjtu_pipeline.py
```

Lowest-memory mode:

```bash
python src/adapters/xjtu/run_xjtu_pipeline.py --parquet-chunk-size 1 --read-batch-size 1
```

Only rerun cleaning and auditing after a successful extraction:

```bash
python src/adapters/xjtu/clean_xjtu_cycles.py --read-batch-size 1 --parquet-chunk-size 1
python src/adapters/xjtu/audit_xjtu_cycles.py
```

## Robust initial-capacity policy

The first versions could use short invalid discharge segments as the initial-capacity reference, producing artificial SOH values. Version 1.3 avoids this by computing:

```text
initial_capacity_ah = median of the first N valid representative cycles
```

Default reference policy:

```text
N = 5
reference_min_capacity_ah = 1.0
reference_max_capacity_ah = 2.3
```

These values are configurable:

```bash
python src/adapters/xjtu/clean_xjtu_cycles.py ^
  --reference-min-capacity-ah 1.0 ^
  --reference-max-capacity-ah 2.3 ^
  --reference-window-cycles 5
```

## Outputs

```text
data/interim/xjtu/extracted/xjtu_charge_extracted_cycles.parquet
data/interim/xjtu/extracted/xjtu_discharge_extracted_cycles.parquet
data/interim/xjtu/clean/xjtu_charge_clean_cycles.parquet
data/interim/xjtu/clean/xjtu_discharge_clean_cycles.parquet
data/interim/xjtu/clean/xjtu_charge_rejected_cycles.csv
data/interim/xjtu/clean/xjtu_discharge_rejected_cycles.csv
data/reports/xjtu/extraction_report.json
data/reports/xjtu/cleaning_report.json
data/reports/xjtu/audit_report.json
data/reports/xjtu/audit_summary.csv
data/reports/xjtu/anomaly_cycles.csv
data/reports/xjtu/plots_qv_voltage_y/*.png
data/reports/xjtu/plots_vq_capacity_y/*.png
```

## Dependencies

```bash
pip install numpy pandas scipy pyarrow matplotlib tqdm
```
