# Cleanup report

## Main changes

- Removed `__pycache__` and compiled Python bytecode.
- Removed legacy v2 feature-extraction scripts from the active public package.
- Moved development diagnostics from `scripts/diagnostics` to `tools/diagnostics`.
- Moved manuscript figure scripts from `scripts/article` to `scripts/figures`.
- Replaced machine-specific default paths with relative defaults and command-line path arguments.
- Added root `README.md`, `requirements.txt`, configuration examples, and pipeline documentation.

## Public entry points

- `scripts/run_adapter.py`
- `scripts/run_feature_extraction_v3.py`
- `scripts/run_modeling_dataframe.py`
