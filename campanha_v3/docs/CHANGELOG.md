# CHANGELOG — campanha_v3 relative to the v2 code that ran the manuscript campaign

All line numbers refer to the **pristine v2 files** in `pristine/`.

## patch_01 — `build_hierarchical_features_v3.py` (target construction)

* `find_eol_position` (681–757) rewritten. Returns `(pos, cycle, capacity, reject_reason, min_ratio_reached)`.
  Reasons: `found`, `no_initial_capacity`, `empty_series`, `never_crossed`, `no_persistence`, `permanence_violated`.
  New parameters `permanence_ratio` (default `EOL_PERMANENCE_RATIO` = 0.90) and `permanence_on_smooth` (default False). **Defaults reproduce v2.**
* `build_targets` (868–1057): new parameters `eol_threshold` (0.80), `eol_secondary_threshold` (0.70), `permanence_mode` (`absolute`|`margin`), `permanence_ratio`, `permanence_margin` (0.10), `permanence_on_smooth`.
  Column names are derived from the thresholds (`eol80/rul80`, `eol85/rul85`, …). With defaults the names are identical to v2.
  New per-row columns: `target_eolNN_reject_reason`, `target_eolNN_min_ratio_reached`, `target_eolNN_threshold_ratio`, `target_eolNN_permanence_ratio`. Same in `capacity_life_summary_v3.csv`.
* `build_base_record` (1485–1487): copies every `target_*` column plus `TARGET_PASSTHROUGH_COLUMNS` instead of a fixed name list.
* argparse (1957–1970): added `--eol-threshold`, `--eol-secondary-threshold`, `--eol-permanence-mode`, `--eol-permanence-ratio`, `--eol-permanence-margin`, `--eol-permanence-on`. **Removed** the dead flag `--eol-allow-single-point-fallback` (1969; never read) and the unreachable block (747).
* Not changed: `compute_initial_capacity` (early median of max(5, 2 % of n) cycles), the XJTU-only comparable-cycles filter (626–650), `compute_trend_stats` (759–808, still computed and written, now renamed at re-join).

## patch_02 — `build_modeling_from_features_v3.py` (modeling matrix)

* New `should_remove_column_allowlist`: keep only columns whose prefix is in `ALLOWED_PREFIXES` (`voltage_ current_ temperature_ power_ energy_ capacityh_ time_ integral_ causal_ vq_ qv_ eq_`) or whose name is in `ALLOWED_EXACT` (four intra-cycle `capacity_*` descriptors); reject `cell_ target_ input_ source_` prefixes and identifiers. Default: reject.
* `--column-policy {allowlist, legacy_denylist}`; default `allowlist`. `legacy_denylist` is the v2 function unchanged (87–105).
* Writes `<branch>_<level>_column_policy.json` with the lists used.
* Verified against the 366 real v2 matrix columns (`tests/test_02_allowlist.py`): the allow-list rejects exactly the five trend descriptors, `source_row_id` and the pandas index artefact `__index_level_0__`, and nothing else.

## patch_03 — `run_modeling_final_campaign_hpc.py` (campaign runner)

* `feature_group` (261–283): `stage_gap` tested before the signal families. Ten gap/jump columns move into `stage_gap`; `--legacy-feature-groups` reproduces v2 (verified on the 366 real columns).
* `regression_metrics` (428–460): adds `MAE_clip0 RMSE_clip0 R2_clip0 nMAE_clip0 nRMSE_clip0 n_pred_negative share_pred_negative`. Existing columns untouched.
* `make_model_specs` (370–417): `elasticnet_cv` option → `ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=20 (sklearn-generated path, eps=1e-3), cv=5, max_iter=5000, tol=1e-3, n_jobs=model_n_jobs)`.
  * History. The first launch (2026-09-03) used `n_alphas=30`, which the scikit-learn on the HPC (≥ 1.9) no longer accepts; all shards died at start. The second launch used a fixed grid `alphas=logspace(-4,0,30)`. With the RUL target in the thousands of cycles, `alpha_max` (the smallest alpha that zeroes every coefficient) is orders of magnitude above 1, so the whole grid sat in the near-unregularised corner: no convergence in 50,000 iterations, single-threaded (`n_jobs=1`), more than an hour per fit, and the machine's load fell from ~200 to ~7 while fourteen shards sat in coordinate descent. Thirty-seven fits in 6.7 h, all in the pooled scope. That launch was stopped. The third launch uses the data-scaled path with warm starts, which is what `ElasticNetCV` is designed for; `05b_time_elasticnet.sh` measures one fit on the largest scope before relaunching. Results of the other five models from the second launch are kept (`--skip-existing`); every `elasticnet/` directory from that launch is deleted by `limpa_parciais.sh --elasticnet` before the relaunch, so no ElasticNet result mixes the two specifications.
  * The container's cgroup CPU quota is 32 cores (`cpu.max` = `3200000 100000`), not the 128 that `nproc` reports. `env.sh` now defaults to `MODEL_N_JOBS=2` (14 shards + arm B = 30 threads). The third launch was stopped after 25 min for this reason and relaunched as the fourth with the caps; completed fits were kept.
  * `scripts/env.sh` caps `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` at `MODEL_N_JOBS`. Without the cap every process opened one BLAS/OpenMP thread per core (ElasticNetCV workers, MLP, HistGradientBoosting) and the load reached 614 on 128 cores with fifteen shards. Throughput only; results are unaffected.
  * ElasticNet emits `ConvergenceWarning` on some path points; the fit completes and the warning does not invalidate the result. `n_pred_negative` in `metrics_global.csv` is the diagnostic that matters for a poor linear fit.
* argparse: `--model-random-state` (default = `--random-state`), `--elasticnet-cv`, `--exclude-datasets`, `--legacy-feature-groups`.
* `main`: `--exclude-datasets` filters X/y/meta right after `load_phase`; `run_config.json` records `package: campanha_v3`.
* New experiment `no_causal_trend` (all features minus `causal_*`), appended to the `all` list.
* Not changed: `MedianRegressor` (weighted median), `compute_sample_weights`, splits, nMAE definition (`MAE / mean(|y_true|)`).

## patch_04 — `merge_modeling_with_curve_features_v3_fixed.py`

* `main()` gets argparse (`--modeling-root --curve-root --output-root`) with the previous hard-coded paths as defaults.
* **Curve diagnostics are removed at merge time**, using the exact list of the v2 post-run sanitizer (`build_modeling_dataframe_v1.py:1244-1258`), which never executed. Against the real 226 curve columns this removes the eight boolean validity flags `VQ_valid_p{003,009,015,030}` and `QV_valid_p{003,009,015,030}`, which were constant per cell and **present in the v2 model**. Found by the causality gate on the first real run (2026-09-03). `--keep-curve-diagnostics` reproduces v2.

## unchanged — `prepare_modeling_data_with_id.py`

* Copied as is. It already preserves NaN for in-fold median imputation and aligns its exclusion rules with the matrix builder (author's August patch).

## new/rejoin_targets.py

* Joins rebuilt targets to the existing feature parquets on `(source_file_key, source_row_id)`. The feature parquets carry several rows per source record (one per `feature_level`: phase and stages), so the join is **many-to-one**; the causal descriptors are computed once per source record on the targets table and broadcast to every level. Aborts on any unmatched row or row-count change.
* **The join key is semantic** (`dataset, battery_id, phase_folder, cycle_index[, cycle_number]`), the same key `merge_modeling_with_curve_features_v3_fixed.py` uses. The id key `(source_file_key, source_row_id)` cannot be used: in the v2 feature parquets with more than 32,767 records (HUST, MATR) `source_row_id` was written as **int16** and wrapped around (HUST charge: 146,122 rows, 65,536 distinct ids in −32768..32767). This did not affect v2 results (the runner drops the column) but it is a data-integrity defect of the published v2 parquets; v3 rewrites `source_row_id` from the targets table as int64. Diagnosed with `new/diag_rejoin.py`.
* Renames the five whole-trajectory descriptors to `cell_*`.
* Adds `causal_capacity_trend_slope_per_step`, `causal_capacity_trend_pearson`, `causal_capacity_positive_step_rate`, `causal_capacity_negative_step_rate`, `causal_capacity_median_abs_step_ah`, each computed on the prefix ≤ k (NaN for fewer than `--min-history` = 3 valid cycles). `tests/test_03_rejoin.py` proves prefix independence from the future.

## new/causality_audit.py

* Rule A: a column constant within ≥ 99 % of cells and variable across cells is flagged **only if** it is cell-specific (distinct values ≥ 0.5 × cells; a whole-trajectory statistic is continuous and rarely repeats) **or** predictive of cell life (|Spearman| ≥ 0.5 against the cell's life within some dataset with ≥ 5 cells). Per-cell constants failing both are protocol constants (e.g. an integer sign-change count under a fixed protocol) and are listed in the report without failing the gate. Rule B: |Spearman| ≥ 0.9995 against the cycle index in ≥ 95 % of cells. Exit code 1 on any flag.
* First real run (2026-09-03, t80 charge, 364 columns, 234 cells): flagged the eight `*_valid_p*` flags (now removed by patch_04) and `energy_sign_change_count` (a protocol constant; no longer flagged under the refined rule A).

## new/coverage_gate.py (added 2026-09-03, after the first full v3 campaign was discarded)

* **Why.** `scripts/env.sh` pointed the curve merge at `~/data/batteries/curve_features_v3`, while the v2 merge script defaults to `curve_features_v3_fixed`. The two folders have the same rows and the same 218 curve columns, but in `curve_features_v3` the V(Q)/Q(V) values are non-null **only for MATR** (coverage 0.0 in the other seven datasets); `_fixed` has 0.945–1.0 everywhere. The merge audit reports `both` for every row in both cases, because the join matches; the values are simply empty. The runner's missingness filter (0.95) then silently dropped the 204 curve columns from every training set without MATR: `all_features` had 341 features in the pooled scope, 138 in `heldout_MATR`, 108–135 in the six non-MATR per-dataset scopes, and `curves_only` failed with "No features left after missingness filter" in those 14 scopes (84 missing fits of 2,520). Diagnosed from `n_features` in `metrics_global.csv` and confirmed against the v2 result for `per_dataset/CALCE/curves_only` (204 features, R² 0.32), which proved the curves existed in v2.
* **Consequence.** The first complete v3 campaign (2,436 core fits and 147 arm-B fits, 2026-09-03) was discarded; the directory is kept as `~/outputs/campanha_v3_curvas_so_matr` for the record and nothing from it enters the manuscript. Four experiments (`no_curves`, `global_stats_only`, `capacity_only`, `energy_added`) never see curve columns and would be numerically identical; they were re-run anyway so that the published tree comes from one build.
* **Gate.** `coverage_gate.py` runs first in `04_gate.sh`: for every (dataset, feature family) with ≥ 5 columns, the mean fraction of present values must be ≥ 0.5; any violation exits 1. Family = name prefix up to the first underscore. `tests/test_07_coverage_gate.py` reproduces the failure mode. The causality audit could not catch this: it tests the columns that exist, not the ones that arrive empty.
* `03_build_matrix.sh` gained a `merge` mode (curve merge + prepare only, base matrix untouched), aborts if the curve folder is missing, and writes `CURVE_SOURCE.txt` next to each `modeling_ready_*`.

## new/preflight.py and scripts/05a_smoke.sh (added 2026-09-03)

* **Why.** Every defect found on 2026-09-03 (curves present only for MATR; the empty `stage_gap` group inherited from v2; a model specification the installed scikit-learn rejects) would have cost a full campaign to discover. Two cheap checks now sit between the gates and the campaign.
* `preflight.py` (GATE 2b, minutes): imports the runner and reproduces, without training, `load_phase`, `feature_group`, `select_experiment_features` and `prepare_feature_columns` for all 14 experiments × 15 scopes (pooled, 7 LODO, 7 per-dataset), both phases, both targets. Flags: any scope with zero features; any scope below `--min-ratio` (0.8) of the experiment's pooled feature count; empty feature groups; more than 10 % of columns in `other`; two experiments selecting the same column set; unexpected dataset set; domains with fewer than 5 cells; non-finite or non-positive targets. Writes `preflight_<tag>.csv/.md` to the gate folder. `tests/test_08_preflight.py` reproduces the curves-only-in-MATR failure on synthetic data.
* `05a_smoke.sh` (SMOKE, minutes with 14 shards): runs the actual runner over every experiment × mode × phase with `--models median` only, so every code path (loading, selection, filter, weights, splits, metrics, output tree) executes at zero training cost. Passes only with 420 `metrics_global.csv` and no `[ERROR]` line; prints the runner's own `n_features` table next to the preflight's.
* `05_quick_test.sh` now refuses to run before the smoke test; `06_campaign.sh` still requires the quick test.
* First real run of the preflight (t80 charge, 2026-09-03) flagged `no_curves` in per-dataset HUST (108 of 138) and OXFORD (110 of 138). Diagnosis: only 2 and 1 columns fall to the missingness filter (`temperature_corr_vs_time`, `temperature_slope_vs_time` in HUST and CALCE; `current_corr_vs_time` in OXFORD: trend descriptors undefined when the sensor does not vary within the cycle); the remaining 28 and 27 are removed by the runner's constant-feature filter, because under a fixed protocol many descriptors are constant within one domain. The preflight now separates the two filters: the 0.8 ratio applies to the missingness filter only (the one that exposes an empty family), constants are reported per scope and flagged only in a LODO scope (more than 10 % of the columns constant with six domains in training). The three genuinely missing columns are declared in `docs/known_missing_by_domain.csv` with the reason, and only declared columns are discounted.
* Second preflight run flagged OXFORD in discharge for having fewer than 5 cells. It has 4 (6 in charge), as in v2 and in the manuscript; the 80/20 battery split leaves one test cell. The hard limit is now 3 cells (`--min-cells`), below which no intra-domain split exists; 3 or 4 cells is a printed warning.

## new/aggregate.py and scripts/08_aggregate.sh (added 2026-09-04)

* Builds the manuscript's Tables 3–6 directly from the runner outputs, with the columns the tables use: T3 (protocol hierarchy: pooled sample-weighted and macro, within-domain macro, LODO macro and median, naive median predictor), T4 (selection regimes: median selection-induced optimism over the fourteen LODO scopes relative to the target-supervised oracle, negative-transfer count; the source-only rows come from `nested_selection.py select` when present), T5 (paired charge × discharge over cells evaluated in both branches under LODO: per-domain Wilcoxon, Holm correction, 10,000-sample bootstrap CI of the cell-weighted median, domain-equal median with a sign test), T6 (per-domain within, LODO, ratio, LODO / naive). Also `S_ablations`, `S_models`, `S_negative_transfer`, `S_within_partitions`, `fig3_data.csv` and `resumo.json` with the numbers quoted in the text.
* Within-domain values are means over the repeated partitions of arm R when it exists (labelled with the number of partitions), otherwise the single core partition, labelled as such.
* Negative transfer is defined once, in code: model nMAE above the `median` model's nMAE in the same LODO scope.
* The candidate space for the oracle and for selection is `--selection-space experiments` (14 experiments, random forest) by default, matching the nested arm; `experiments_x_models` widens it to all non-naive models.

## new/permutation_importance.py, new/learning_curves.py, scripts/10_extras.sh (added 2026-09-05)

* Two analyses quoted in the manuscript body were computed in v2 on the leaky matrix (`importancias.py`, `curva_aprendizado.csv`) and are recomputed on the v3 matrix with the same design. Permutation importance: RF + all_features, pooled (stratified battery split, seed 42) and per domain, permutation on the test split, 5 repeats, negative MAE, plus impurity importance; family shares reported both as the raw sum (v2 convention) and with negative importances clipped at zero. Learning curves: per domain and phase, battery-level splits with test sizes 0.2/0.4/0.6/0.8 (training on 80/60/40/20 % of the cells), partition seeds 101–103, `median` and RF, model seed 4242; `learning_curves_increments.csv` gives the relative change of nMAE between the two largest training fractions. Note that `feature_group` places the `causal_*` descriptors in the capacity family.
* The internal GroupKFold arm of v2 (Supplementary Data S5) is not reproduced; it is cited only in the supplement and is dropped from it.

## Release verification and CI (added 2026-09-06, after external review)

* `new/verify_release.py` / `scripts/verify_release.sh`: SHA256 of every parquet under `modeling_ready_t80`, `modeling_ready_t85` and `features_causal` (the archived inputs the campaign is reproducible from), expected fit counts per component (2,520 / 360 / 1,920 / 1,176), SHA256 of the archived manuscript tables, and regeneration of T3–T6 and the supplementary tables from the archived results with comparison at 1e-9. `--write` records `docs/RELEASE_CHECKSUMS.json` once, on the campaign machine.
* `.github/workflows/tests.yml` runs the eight synthetic test suites on every push; `run_tests.sh` now exits non-zero if any suite fails.
* Reproducibility claim to use in the manuscript: the modeling campaign is reproducible from the archived processed feature matrices; the raw-to-feature extraction is documented but not re-executed by this package.
* `aggregate.py`: pooled protocol also reported over the arm-R partitions (sample-weighted and macro-average, mean and sd; `S_pooled_partitions.csv`); number of distinct test cells per domain across the partitions (`S_distinct_test_cells.csv`), to be reported as partition sensitivity rather than a confidence interval where the domain has fewer than five cells; T6 carries the LODO MAE in cycles, the mean RUL of the scope and the number of cells, so extreme nMAE values are shown with their scale.

## new/nested_selection.py

* `plan`: emits 42 runner jobs (7 targets × 6 inner validation domains) with `--exclude-datasets T --lodo-heldouts V`.
* `select`: argmin over inner folds (mean and median) per target, then reports the outer LODO score of the selected configuration from the core campaign.
