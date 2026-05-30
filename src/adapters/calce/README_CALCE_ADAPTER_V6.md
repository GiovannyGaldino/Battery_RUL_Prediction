# CALCE Adapter v6.0.0

This package reconstructs the CALCE adapter for the dual charge/discharge RUL pipeline.
It does not convert TXT files to Excel; it reads the converted `.xls`/`.xlsx` files directly.

## Main design decisions

- One source workbook/sheet contributes at most one dominant charge segment and one dominant discharge segment.
- The electrochemical cycle number is reconstructed from chronological file order inside each cell.
- Internal CALCE cycler counters such as `Cycle_Index` are ignored for model cycle numbering because they can create thousands of pseudo-cycles.
- Voltage, current, capacity, and energy units are normalized defensively (`mV -> V`, `mA -> A`, `mAh -> Ah`, `mWh -> Wh`).
- Excel sheets with metadata above the table are handled by scanning likely header rows.
- Charge and discharge branches are stored separately, using the same official schema.
- The cleaner applies explicit physical filters to remove incomplete or suspicious segments.

## Files

```text
src/adapters/calce/extract_calce_cycles.py
src/adapters/calce/clean_calce_cycles.py
src/adapters/calce/audit_calce_cycles.py
scripts/diagnostics/diagnose_calce_segments.py
scripts/pipelines/run_calce_v6_pipeline.py
```

## Expected raw data layout

```text
data/raw/CALCE/<cell_id>/<workbook>.xlsx
```

Example:

```text
data/raw/CALCE/CS2_8/CS2_8_2_24_10.xlsx
```

## Python dependencies

The project environment must include at least:

```text
pandas
numpy
matplotlib
openpyxl
pyarrow or fastparquet
```

`pyarrow` or `fastparquet` is required because the adapter writes `.parquet` files.

## One-command execution

From the project root (`Batteries`):

```powershell
python scripts/pipelines/run_calce_v6_pipeline.py --project-root .
```

This removes previous CALCE interim/report outputs before running. To keep existing outputs:

```powershell
python scripts/pipelines/run_calce_v6_pipeline.py --project-root . --keep-existing
```

## Step-by-step execution

```powershell
python src/adapters/calce/extract_calce_cycles.py --project-root .
python src/adapters/calce/clean_calce_cycles.py --project-root .
python src/adapters/calce/audit_calce_cycles.py --project-root .
```

## Diagnostic script

The diagnostic script is optional and does not modify official Parquet files:

```powershell
python scripts/diagnostics/diagnose_calce_segments.py --project-root .
```

It writes diagnostic CSV and JSON files under:

```text
data/reports/calce/diagnostics/
```

## Gate criteria after audit

The CALCE branch should only be accepted for merge if:

- both charge and discharge clean files have valid schema;
- `cycle_max` is compatible with chronological source records, not thousands of pseudo-cycles;
- most model-usable cycles have non-negative `remaining_cycles`;
- charge current is positive and discharge current is negative;
- `capacity_ah` is non-negative and accumulated within each phase;
- charge end voltage is near the high-voltage region;
- discharge end voltage is near the cutoff region;
- anomaly count is compatible with known rejected partial segments.

## Known CALCE caveats

The CALCE source files can contain many internal current-based segments, rest periods, partial cycles, and diagnostic pulses. Some cells also have short experimental histories. Therefore, this adapter treats CALCE as a partially heterogeneous source and keeps detailed rejection reasons for every unusable cycle.
