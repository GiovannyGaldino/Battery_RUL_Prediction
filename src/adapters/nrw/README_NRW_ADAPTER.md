# NASA Random Walk (NRW) Standard Adapter

This adapter reconstructs the legacy NASA Random Walk scripts into the current project schema.

## Files

Copy the folder below into your project:

```text
src/adapters/nrw/
├─ extract_nrw_cycles.py
├─ clean_nrw_cycles.py
├─ audit_nrw_cycles.py
└─ __init__.py
```

## Expected raw data

```text
Batteries/data/raw/NRW/RW1.mat
Batteries/data/raw/NRW/RW2.mat
...
```

## Run order

```bash
python src/adapters/nrw/extract_nrw_cycles.py
python src/adapters/nrw/clean_nrw_cycles.py
python src/adapters/nrw/audit_nrw_cycles.py
```

If you are not running from the default project path, pass `--base-dir`:

```bash
python src/adapters/nrw/extract_nrw_cycles.py --base-dir "D:/USUÁRIOS/Marcellus/OneDrive/Doutorado/BATERIAS - Ressubmissão/Batteries"
python src/adapters/nrw/clean_nrw_cycles.py --base-dir "D:/USUÁRIOS/Marcellus/OneDrive/Doutorado/BATERIAS - Ressubmissão/Batteries"
python src/adapters/nrw/audit_nrw_cycles.py --base-dir "D:/USUÁRIOS/Marcellus/OneDrive/Doutorado/BATERIAS - Ressubmissão/Batteries"
```

## Outputs

```text
data/interim/nrw/extracted/nrw_charge_extracted_cycles.parquet
data/interim/nrw/extracted/nrw_discharge_extracted_cycles.parquet

data/interim/nrw/clean/nrw_charge_clean_cycles.parquet
data/interim/nrw/clean/nrw_discharge_clean_cycles.parquet

data/interim/nrw/clean/nrw_charge_rejected_cycles.csv
data/interim/nrw/clean/nrw_discharge_rejected_cycles.csv

data/reports/nrw/extraction_report.json
data/reports/nrw/cleaning_report.json
data/reports/nrw/audit_report.json
data/reports/nrw/audit_summary.csv
data/reports/nrw/audit_anomalies.csv
```

## Notes

- The extractor is conservative: it keeps only reference charge and reference discharge steps.
- Random-walk operating steps are skipped because they are continuous/ambiguous segments and should not be forced into the cycle database.
- Current sign convention follows the project schema: positive current for charge and negative current for discharge.
- The default EOL criterion in the cleaner is 70% of the initial reference discharge capacity.
- If you want to reproduce the legacy behavior of using the last observed cycle when EOL is not reached, use:

```bash
python src/adapters/nrw/clean_nrw_cycles.py --eol-policy last_observed
```


## Version 1.1 extraction note

The extractor now uses exact normalized comment matching for NASA Random Walk reference cycles.
It extracts only comments equal to `reference charge` and `reference discharge` by default.
This avoids selecting rest stages such as `rest post reference charge` and `rest prior reference discharge`, which appeared in the v1.0 audit as invalid/too-short cycles.

The optional flag below can be used only for an exploratory protocol-specific study:

```bash
python src/adapters/nrw/extract_nrw_cycles.py --include-reference-power-discharge
```

For the main charge/discharge RUL branch, keep this flag disabled.
