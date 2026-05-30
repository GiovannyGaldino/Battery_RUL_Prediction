# HUST Standard Adapter — Version 1

This folder contains the HUST adapter reconstructed for the lithium-ion battery RUL project.

The adapter follows the standard workflow:

```text
src/adapters/hust/extract_hust_cycles.py
src/adapters/hust/clean_hust_cycles.py
src/adapters/hust/audit_hust_cycles.py
```

A shared helper file is included:

```text
src/adapters/hust/hust_schema.py
```

## Main changes in version 1

- The previous generated adapter version was discarded.
- The adapter was rebuilt as a new version 1.
- `tqdm` progress bars were added to the main execution loops:
  - raw pickle files;
  - batteries;
  - cycles;
  - cleaning rows;
  - vector checks;
  - anomaly scanning;
  - plot generation.
- If `tqdm` is not installed, the scripts still run using an internal fallback without progress bars.
- Charge and discharge branches are extracted, cleaned, and audited separately.
- Clean files follow the official standard schema and preserve the same columns for both phases.
- Feature extraction is intentionally not performed in this adapter stage.

## Expected raw data location

The default raw HUST path is:

```text
D:/USUÁRIOS/Marcellus/OneDrive/Doutorado/BATERIAS - Ressubmissão/Batteries/data/raw/HUST
```

Example raw files:

```text
data/raw/HUST/1-1.pkl
data/raw/HUST/1-2.pkl
```

## Installation requirements

Minimum packages:

```bash
pip install pandas numpy pyarrow matplotlib tqdm
```

`pyarrow` or another parquet engine is required by pandas to save parquet files.

## Execution order

From the project root or from any location, run:

```bash
python src/adapters/hust/extract_hust_cycles.py --base-dir "D:/USUÁRIOS/Marcellus/OneDrive/Doutorado/BATERIAS - Ressubmissão/Batteries"

python src/adapters/hust/clean_hust_cycles.py --base-dir "D:/USUÁRIOS/Marcellus/OneDrive/Doutorado/BATERIAS - Ressubmissão/Batteries" --temperature-c 30

python src/adapters/hust/audit_hust_cycles.py --base-dir "D:/USUÁRIOS/Marcellus/OneDrive/Doutorado/BATERIAS - Ressubmissão/Batteries"
```

## Output files

Extraction:

```text
data/interim/hust/extracted/hust_charge_extracted_cycles.parquet
data/interim/hust/extracted/hust_discharge_extracted_cycles.parquet
data/reports/hust/extraction_report.json
```

Cleaning:

```text
data/interim/hust/clean/hust_charge_clean_cycles.parquet
data/interim/hust/clean/hust_discharge_clean_cycles.parquet
data/interim/hust/clean/hust_charge_rejected_cycles.csv
data/interim/hust/clean/hust_discharge_rejected_cycles.csv
data/reports/hust/cleaning_report.json
```

Audit:

```text
data/reports/hust/audit_report.json
data/reports/hust/audit_summary.csv
data/reports/hust/anomaly_list.csv
data/reports/hust/plots/charge/*.png
data/reports/hust/plots/discharge/*.png
```

## Notes

- The scripts use English names, comments, and `snake_case`.
- Current sign convention:
  - charge current is positive;
  - discharge current is negative.
- The generic `capacity_ah` column is used for both phases.
- Temperature is stored in `temperature_c`; when measured temperature is unavailable, a constant nominal vector is created and documented with `temperature_source = nominal_constant`.
- Invalid or non-model-usable cycles are not silently discarded; they are recorded in rejected-cycle CSV files and reports.
