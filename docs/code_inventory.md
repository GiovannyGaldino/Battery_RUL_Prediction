# Code inventory

## Active reproducibility path

| Path | Purpose |
|---|---|
| `src/adapters/` | Dataset-specific extraction, cleaning, and audit scripts. |
| `src/feature_extraction/build_hierarchical_features_v3.py` | Main vector-aware feature extraction pipeline. |
| `src/feature_extraction/check_feature_extraction_state_v3.py` | State checker for resumable feature extraction. |
| `src/feature_extraction/monitor_feature_extraction_v3.py` | Optional live monitor for long feature-extraction runs. |
| `src/feature_extraction/inspect_features_v3_outputs.py` | Post-extraction inspection and sanity checks. |
| `src/modeling/build_modeling_dataframe_v1.py` | Builds charge/discharge modeling dataframes from clean cycles and feature files. |
| `src/validation/` | Optional data and feature auditing utilities. |
| `scripts/run_adapter.py` | Thin public runner for dataset adapters. |
| `scripts/run_feature_extraction_v3.py` | Thin public runner for feature extraction v3. |
| `scripts/run_modeling_dataframe.py` | Thin public runner for modeling-dataset generation. |
| `scripts/figures/` | Manuscript figure-generation scripts. |
| `tools/diagnostics/` | Optional development diagnostics; not part of the main workflow. |

## Removed from the public package

The original package contained bytecode caches (`__pycache__`, `*.pyc`) and legacy v2 feature-extraction scripts. These files were removed because they are not required to reproduce the current article pipeline and can confuse public users.
