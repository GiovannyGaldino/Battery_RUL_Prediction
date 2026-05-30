# Battery RUL Prediction — Public Code Repository

This repository contains the reproducible code used to adapt public lithium-ion battery aging datasets, extract charge/discharge features, audit processed data, and build modeling tables for Remaining Useful Life (RUL) prediction.

The repository intentionally does **not** store raw, interim, processed, modeling, report, or model artifact files. Data should be stored outside Git and referenced through command-line arguments or local configuration files.

## Repository scope

Active code is organized into four groups:

- `src/adapters/`: dataset-specific extraction, cleaning, and auditing scripts;
- `src/feature_extraction/`: vector-aware hierarchical feature extraction v3;
- `src/modeling/`: modeling dataframe construction;
- `src/validation/`: optional quality-control and audit utilities.

Convenience scripts are stored in:

- `scripts/`: reproducible public entry points;
- `scripts/figures/`: scripts used to generate manuscript figures;
- `tools/diagnostics/`: optional debugging utilities that are not required for the main pipeline.

## Data policy

Do not commit datasets to this repository. Keep local data in a separate folder such as:

```text
<project-root>/data/raw
<project-root>/data/interim
<project-root>/data/processed
<project-root>/reports
<project-root>/models
```

For a public release, publish only versioned data packages, for example in Google Drive, Zenodo, Figshare, or OSF.

## Quick start

Create an environment and install the required packages:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `configs/paths.example.yaml` to `configs/paths.local.yaml` and edit the paths for your machine. The scripts can also be run without YAML files by passing paths explicitly as command-line arguments.

## Expected raw-data layout

Place the original downloaded files under `data/raw`:

```text
data/raw/
  CALCE/
  HUST/
  MATR/
  PCOE/
  NRW/
  ORA/
  RWTH/
  XJTU/
```

See `docs/raw_data_layout.md` for dataset-specific file patterns.

## Main workflow

1. Run dataset adapters: extraction, cleaning, and audit.
2. Copy or merge clean cycle files into `data/interim/charge` and `data/interim/discharge`.
3. Run hierarchical feature extraction v3.
4. Inspect/audit feature files.
5. Build modeling dataframes.

Detailed commands are available in `docs/running_pipeline.md`.

## Minimal example

```bash
python scripts/run_adapter.py --dataset matr --base-dir . --stage all
python scripts/run_feature_extraction_v3.py --base-dir . --workers 2 --resume --fast-mode --no-stages
python scripts/run_modeling_dataframe.py --base-dir .
```

## Notes for publication

Legacy v2 feature-extraction scripts and bytecode/cache files were removed from the public package. Development-only diagnostics were moved to `tools/diagnostics` so the public workflow remains focused and reproducible.
