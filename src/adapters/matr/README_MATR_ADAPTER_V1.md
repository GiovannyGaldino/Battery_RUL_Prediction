# MATR Adapter — Version 1

This package replaces the first MATR adapter reconstruction. It is written in English, uses `tqdm` progress bars, and follows the project standard workflow:

```text
src/adapters/matr/extract_matr_cycles.py
src/adapters/matr/clean_matr_cycles.py
src/adapters/matr/audit_matr_cycles.py
```

The adapter assumes the MATR raw files are stored under:

```text
D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\raw\MATR
```

Example file:

```text
D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\raw\MATR\2017-05-12_batchdata_updated_struct_errorcorrect.mat
```

## Dependencies

Install the following packages in the environment used for the battery project:

```powershell
pip install numpy pandas h5py pyarrow matplotlib tqdm
```

`pyarrow` is needed for parquet output.

## Installation

Copy the `matr` folder into:

```text
D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\src\adapters\matr
```

## Execution

Run the commands from the project root:

```powershell
cd "D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries"

python src\adapters\matr\extract_matr_cycles.py --base-dir .
python src\adapters\matr\clean_matr_cycles.py --base-dir .
python src\adapters\matr\audit_matr_cycles.py --base-dir .
```

The scripts also contain the full project path as the default base directory. Therefore, they can also be executed without `--base-dir` if the project remains in the same Windows path.

## Main outputs

Extraction:

```text
data/interim/matr/extracted/matr_charge_extracted_cycles.parquet
data/interim/matr/extracted/matr_discharge_extracted_cycles.parquet
data/interim/matr/extracted/matr_extraction_report.csv
data/interim/matr/extracted/matr_extraction_report.json
```

Cleaning:

```text
data/interim/matr/clean/matr_charge_clean_cycles.parquet
data/interim/matr/clean/matr_discharge_clean_cycles.parquet
data/interim/matr/clean/matr_charge_rejected_cycles.csv
data/interim/matr/clean/matr_discharge_rejected_cycles.csv
data/interim/matr/clean/matr_cleaning_report.json
```

Audit:

```text
data/reports/matr/audit_report.json
data/reports/matr/audit_summary.csv
data/reports/matr/anomaly_list.csv
data/reports/matr/plots/charge/...
data/reports/matr/plots/discharge/...
```

## Important parameters

Disable the voltage-window trimming:

```powershell
python src\adapters\matr\clean_matr_cycles.py --base-dir . --disable-voltage-window
```

Use all cycles in audit plots instead of plotting one every 10 cycles:

```powershell
python src\adapters\matr\audit_matr_cycles.py --base-dir . --plot-every-n-cycles 1
```

Plot only model-usable cycles:

```powershell
python src\adapters\matr\audit_matr_cycles.py --base-dir . --plot-only-model-usable
```

## Design notes

- Charge and discharge are extracted into separate files.
- Both clean files follow the same official schema.
- `capacity_ah` is generic: it represents accumulated charge capacity for charge cycles and accumulated discharge capacity for discharge cycles.
- Current sign convention is standardized: positive for charge and negative for discharge.
- Missing temperature is filled with a nominal 30 °C constant by default for MATR, while `has_temperature=False` and `temperature_source='nominal_constant'` are recorded.
- Rejected cycles are written to CSV with explicit exclusion reasons.
- The audit script validates schema, summarizes cycle counts, identifies anomalies, and creates Q(V)/V(Q) plots with progress bars.
