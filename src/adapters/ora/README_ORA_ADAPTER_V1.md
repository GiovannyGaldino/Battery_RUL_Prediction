# Oxford/ORA Adapter v1

This folder contains the reconstructed Oxford/ORA adapter using English names, English comments, `snake_case`, separated charge/discharge branches, and `tqdm` progress bars.

## Files

```text
src/adapters/ora/ora_schema.py
src/adapters/ora/extract_ora_cycles.py
src/adapters/ora/clean_ora_cycles.py
src/adapters/ora/audit_ora_cycles.py
src/adapters/ora/run_ora_pipeline.py
```

## Expected raw file

```text
D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\raw\ORA\Oxford_Battery_Degradation_Dataset_1.mat
```

## Pipeline outputs

```text
data/interim/ora/extracted/ora_charge_extracted_cycles.parquet
data/interim/ora/extracted/ora_discharge_extracted_cycles.parquet

data/interim/ora/clean/ora_charge_clean_cycles.parquet
data/interim/ora/clean/ora_discharge_clean_cycles.parquet

data/interim/ora/clean/ora_charge_rejected_cycles.csv
data/interim/ora/clean/ora_discharge_rejected_cycles.csv

data/reports/ora/extraction_report.json
data/reports/ora/cleaning_report.json
data/reports/ora/audit_report.json
data/reports/ora/audit_summary.csv
data/reports/ora/anomaly_list.csv
data/reports/ora/plots/
```

## Run the complete pipeline

From the project root:

```bash
python src/adapters/ora/run_ora_pipeline.py
```

With explicit paths:

```bash
python src/adapters/ora/run_ora_pipeline.py ^
  --base-dir "D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries" ^
  --mat-file "D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\raw\ORA\Oxford_Battery_Degradation_Dataset_1.mat" ^
  --round-vectors 8
```

## Run step by step

```bash
python src/adapters/ora/extract_ora_cycles.py
python src/adapters/ora/clean_ora_cycles.py
python src/adapters/ora/audit_ora_cycles.py
```

## Main methodological changes from the legacy ORA scripts

1. `C1ch` and `C1dc` are no longer stored in a single row. They are extracted as independent rows and saved in separate charge and discharge files.
2. The clean files follow the same `STANDARD_CYCLE_COLUMNS` order for both phases.
3. `capacity_ah` is the generic accumulated phase-capacity vector. For charge rows it means accumulated charge capacity. For discharge rows it means accumulated discharge capacity.
4. Current sign follows the project convention: positive for charge and negative for discharge.
5. EOL is recomputed during cleaning using the nominal Oxford cell capacity: `0.8 × 0.740 Ah = 0.592 Ah`.
6. Invalid or non-model-usable cycles are not silently discarded. They are flagged in the clean parquet and summarized in rejected-cycle CSV files.
7. The audit script validates schema, summarizes signal availability, summarizes targets, creates anomaly lists, and plots `V vs Q` and `Q vs V` curves by battery and phase.

## Dependencies

```bash
pip install numpy pandas scipy pyarrow matplotlib tqdm
```
