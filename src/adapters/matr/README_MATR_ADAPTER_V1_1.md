# MATR Adapter v1.1

This package replaces the previous MATR adapter version 1. It keeps the same official project workflow but updates the cleaning and audit stages to improve capacity/SoH handling and to preserve anomaly evidence for the manuscript.

## Main changes relative to v1

1. `clean_matr_cycles.py` now estimates `initial_capacity_ah` using a robust median of the first valid cycles instead of the first available cycle.
2. The robust estimate uses the first `N` valid phase capacities, removes IQR outliers, and stores the estimation metadata.
3. `soh` is recomputed from the robust initial capacity.
4. Cycles with abnormal SOH are explicitly flagged and, by default, excluded from model training when `soh <= 0.05` or `soh > 1.20`.
5. Detailed anomaly CSV files are written for article tables and methodological reporting.
6. The audit script aggregates cleaning anomalies and audit anomalies and creates a compact article-ready anomaly summary.
7. `tqdm` progress bars are preserved in extraction, cleaning, target finalization, anomaly generation, auditing, and plotting.

## Expected project location

Copy the `matr/` folder to:

```text
D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\src\adapters\matr
```

The default raw MATR directory remains:

```text
D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\raw\MATR
```

Example raw file:

```text
D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\raw\MATR\2017-05-12_batchdata_updated_struct_errorcorrect.mat
```

## Execution

Run from the project root:

```powershell
cd "D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries"

python src\adapters\matr\extract_matr_cycles.py --base-dir .
python src\adapters\matr\clean_matr_cycles.py --base-dir .
python src\adapters\matr\audit_matr_cycles.py --base-dir .
```

For faster audit without plots:

```powershell
python src\adapters\matr\audit_matr_cycles.py --base-dir . --no-plots
```

## New cleaning outputs

```text
data/interim/matr/clean/matr_charge_clean_cycles.parquet
data/interim/matr/clean/matr_discharge_clean_cycles.parquet

data/interim/matr/clean/matr_charge_rejected_cycles.csv
data/interim/matr/clean/matr_discharge_rejected_cycles.csv

data/interim/matr/clean/matr_charge_cycle_anomalies.csv
data/interim/matr/clean/matr_discharge_cycle_anomalies.csv
data/interim/matr/clean/matr_cycle_anomalies_all.csv

data/interim/matr/clean/matr_charge_initial_capacity_estimates.csv
data/interim/matr/clean/matr_discharge_initial_capacity_estimates.csv
data/interim/matr/clean/matr_initial_capacity_estimates_all.csv

data/interim/matr/clean/matr_cleaning_report.json
data/interim/matr/clean/matr_cleaning_anomaly_report.json
```

## New audit outputs

```text
data/reports/matr/audit_report.json
data/reports/matr/audit_summary.csv

data/reports/matr/anomaly_list.csv
data/reports/matr/audit_anomaly_list.csv
data/reports/matr/cleaning_anomaly_list.csv
data/reports/matr/article_anomaly_summary.csv
```

Use `article_anomaly_summary.csv` in the manuscript to report anomaly categories by phase, origin, group, type, severity, affected batteries, affected cycles, and model-impacting records.

## Important parameters

Robust initial capacity:

```powershell
--initial-capacity-n-cycles 10
--initial-capacity-iqr-multiplier 1.5
--initial-capacity-min-abs-ah 0.05
```

SOH thresholds:

```powershell
--min-soh-model-usable 0.05
--max-soh-model-usable 1.20
--min-soh-warning 0.50
--max-soh-warning 1.10
```

Voltage window:

```powershell
--charge-v-min 2.10 --charge-v-max 3.55
--discharge-v-min 2.10 --discharge-v-max 3.55
```

Disable voltage trimming only for diagnostic tests:

```powershell
python src\adapters\matr\clean_matr_cycles.py --base-dir . --disable-voltage-window
```

## Dependencies

```powershell
pip install numpy pandas h5py pyarrow matplotlib tqdm
```

## Notes

The clean parquet files still follow the official `STANDARD_CYCLE_COLUMNS` exactly. Anomaly details are intentionally stored in separate CSV/JSON report files so the clean schema remains compatible with the global merge and feature extraction steps.
