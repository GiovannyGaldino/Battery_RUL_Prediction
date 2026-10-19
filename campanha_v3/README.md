# campanha_v3 — corrected, auditable re-run of the RUL modeling campaign

This package re-runs the modeling campaign of the manuscript with the defects
found in the code audit of 2026-09-02 removed, **without re-extracting curve
features**. It is designed to be published alongside the data (Zenodo) so that
a reader can see exactly what changed relative to the v2 code and reproduce
every step.

## What is in the box

```
pristine/   the v2 source files exactly as they ran (SHA256SUMS included)
patches/    one Python script per patch; each asserts its anchors and refuses
            to run on a file it does not recognise, plus the resulting .diff
patched/    the v2 files after the patches (what v3 actually runs)
new/        modules that did not exist in v2
tests/      synthetic-data tests; run without any real data
scripts/    the pipeline, in order, with gates
docs/       CHANGELOG and the audit that motivated each change
```

Nothing from v2 is overwritten: v3 reads the v2 feature parquets and writes
only to new directories (`~/data/processed/v3`, `~/outputs/campanha_v3`).

## Defects removed (see docs/CHANGELOG.md for file:line references)

| # | Defect in v2 | Fix in v3 |
|---|---|---|
| 1 | Five capacity-trend descriptors computed over each cell's **whole trajectory** were used as features at every cycle (temporal leakage). | Renamed `cell_*` (never predictors) and replaced by **causal** expanding-window counterparts `causal_*` that use only cycles ≤ k (`new/rejoin_targets.py`). |
| 2 | Feature selection was a **deny-list by name**; anything not anticipated entered the model. | **Allow-list by intra-cycle family**, default reject (`patch_02`). |
| 3 | No automated check that a feature is causal. | **Causality audit gate** flags cell-constant columns that are cell-specific or predict cell life, and cycle-ordinal proxies; the campaign refuses to train if anything is flagged (`new/causality_audit.py`). |
| 3b | Eight curve validity flags `VQ_valid_p*`/`QV_valid_p*` (per-cell constants) were in the v2 model because the post-run sanitizer never ran. | Removed at curve-merge time with the sanitizer's own list (`patch_04`). |
| 4 | EOL detection rejected cells with no record of *why*; 51 of 349 "censored" cell-phases had in fact crossed 80%. | `find_eol_position` returns a **reject reason** and the minimum SOH reached; both are written per row and per cell (`patch_01`). |
| 5 | Permanence rule fixed at 0.90·C₀ on **raw** values; one late outlier voided a valid EOL; threshold 0.80 hard-coded. | Threshold and permanence are CLI parameters; permanence can scale with the threshold and be checked on the smoothed series. **Defaults reproduce v2 exactly.** |
| 6 | `no_stage_gap` ≡ `all_features` and `global_stats_only` ≡ `no_curves` because the `stage_gap` group was tested last and was empty. | `stage_gap` tested first; `--legacy-feature-groups` reproduces v2 (`patch_03`). |
| 7 | Predictions never truncated at RUL ≥ 0. | Every metrics file also carries `*_clip0` columns and the count of negative predictions; raw columns unchanged. |
| 8 | One seed for both the split and the estimators. | `--model-random-state` separate from `--random-state`. |
| 9 | `ElasticNet(alpha=0.01)` fixed. | `--elasticnet-cv` fits `ElasticNetCV` inside the training set. |
| 10 | "Source-only" configuration selection used scores from models whose training set included the target. | `new/nested_selection.py`: the target leaves **every** training set used for selection (`--exclude-datasets`). |
| 11 | Dead CLI flag `--eol-allow-single-point-fallback`. | Removed together with its unreachable block. |
| 12 | Merge script had hard-coded paths. | argparse with the old defaults (`patch_04`). |

## Pipeline

Run **one script at a time** and read its gate before the next; each script
refuses to start if the previous step's output is missing. `00` first runs
`scripts/check_deps.py` (fastparquet, scikit-learn, tqdm, pyarrow, ...).

```
bash scripts/00_apply_patches.sh    # verifies v2 SHA256, applies patches, runs tests
bash scripts/01_rebuild_targets.sh  # targets only (--dry-run): thresholds 0.80 and 0.85   [GATE 1]
bash scripts/02_rejoin.sh           # joins new targets to the EXISTING feature parquets; adds causal_*
bash scripts/03_build_matrix.sh     # allow-list matrix, curve merge, modeling_ready for both targets
bash scripts/04_gate.sh             # coverage per family/domain, then causality audit        [GATE 2]
bash scripts/04b_preflight.sh       # n_features per experiment x scope without training       [GATE 2b]
bash scripts/05a_smoke.sh           # the real runner, all cases, median model only: 420 cases  [SMOKE]
bash scripts/05_quick_test.sh       # 16 fits; pooled nMAE expected ≈0.112 / 0.118          [GATE 3]
bash scripts/05b_time_elasticnet.sh # one ElasticNetCV fit on the largest scope; decide before launching
bash scripts/06_campaign.sh         # core: 14 experiments × 6 models × 15 scopes × 2 phases
bash scripts/07b_threshold85.sh     # arm B: threshold 0.85, 4 processes (independent of 06)
bash scripts/09_pos_nucleo.sh       # after the core: relaunch arm B sharded + arm R at 4 threads each
bash scripts/07n_nested.sh [P]      # arm N: nested source-only selection, 42 jobs, xargs -P
bash scripts/07r_repeated.sh [P]    # arm R: 20 repeated splits
bash scripts/status.sh              # progress of every component
bash scripts/08_aggregate.sh        # manuscript Tables 3-6, supplementary CSVs, resumo.json
bash scripts/10_extras.sh           # permutation importance + per-domain learning curves (cited in the body)
bash scripts/verify_release.sh --write   # once, on the campaign machine: hashes of inputs, fit counts, table hashes
bash scripts/verify_release.sh      # anywhere: verify hashes and counts, regenerate T3-T6 and compare
bash scripts/retoma.sh              # after a container restart: relaunch what is incomplete
```

Curve source: `env.sh` points the curve merge at `curve_features_v3_fixed`, the
folder the v2 campaign used and the only one with V(Q)/Q(V) curves for every
dataset. The sibling `curve_features_v3` has curves for MATR only; a first v3
campaign built on it was discarded on 2026-09-03 (see CHANGELOG, coverage gate).

Machine note (measured 2026-09-03): the JupyterHub container reports 128 cores
(`nproc`) but its cgroup quota is 32 (`/sys/fs/cgroup/cpu.max` = `3200000 100000`).
`env.sh` therefore defaults to `MODEL_N_JOBS=2` and caps BLAS/OpenMP threads at the
same value; fourteen shards plus arm B use 30 threads. Asking for more only adds
throttling: with 112 threads the load reached 614 while about 20 cores did work.

```bash
bash scripts/limpa_parciais.sh      # drops incomplete model dirs (add --elasticnet after a spec change)
```

GATE 1 measured counts (HPC, 2026-09-03): 469 cell-phases reach EOL80 with the
corrected rules (v2: 465) and 691 reach EOL85. Reject reasons at 0.80:
never_crossed 330, no_persistence 9, permanence_violated 6. Note: "crossed" is
decided on the 5-cycle rolling median, so a single-cycle dip below the threshold
does not count; the audit's earlier estimate (516/740) used the raw minimum and
overstated the recoverable cells.
GATE 3 measured (HPC, 2026-09-03, random forest): pooled nMAE `all_features`
0.108 / 0.126 and `no_causal_trend` 0.127 / 0.170 (v2 with leakage: 0.104 /
0.102). LODO median `all_features` 1.74 / 21.3 against 0.900 / 0.678 in v2: the
leaked trend descriptors were carrying each cell's fade rate across domains, and
without them the short-lived domains (CALCE, NRW, Oxford, PCoE) are essentially
unpredictable zero-shot. The pre-specified configuration remains random forest
with `all_features` (declared before the core campaign); `no_causal_trend` is an
ablation.

## The three feature states the campaign compares

| state | where | meaning |
|---|---|---|
| leaky | v2 results (`~/outputs/campanha_v2`) | what the manuscript reports today |
| removed | v3 experiment `no_causal_trend` | trend information deleted |
| causal | v3 experiment `all_features` | trend information a deployed model would legitimately have |

## Notes a reviewer will ask about

* **Spearman → Pearson.** The v2 descriptor `capacity_trend_spearman` has no
  cheap exact expanding-window form; its causal counterpart is
  `causal_capacity_trend_pearson` (closed form, O(n)). The other four are exact
  expanding versions of the v2 statistics.
* **ElasticNetCV inner CV is row-level** (cv=5, within the training set only).
  It never sees the test fold; battery grouping inside that inner CV is not
  enforced. Declared, not tuned per case.
* **Sample weighting** (`dataset_inverse`) is unchanged from v2 and documented:
  training is domain-balanced, metrics are record-weighted.
* **Defaults reproduce v2** in `patch_01` and `patch_03`; the v3 behaviour is
  what `scripts/` pass explicitly. In `patch_02` the default is the allow-list,
  because the v2 default is the leaky one; `--column-policy legacy_denylist`
  reproduces it.

## Requirements

Python ≥ 3.10, numpy, pandas, pyarrow, fastparquet, scikit-learn ≥ 1.4, tqdm,
joblib, matplotlib (see `requirements.txt`). The tests need only numpy, pandas,
tqdm and fastparquet.

## Provenance

`pristine/SHA256SUMS` is the fingerprint of the v2 files that were audited and
that `00_apply_patches.sh` checks against the HPC before applying anything.
The v2 code archived on Zenodo (v1.0.0) differs from the code that ran the v2
campaign (`build_hierarchical_features_v3.py`: 70,212 vs 75,386 bytes); this
package is built on the code that ran.


## Release 1.1.0 (Zenodo)

This is the package that produced the results reported in the manuscript (core matrix 2,520 fits; threshold arm 360; repeated partitions 1,920; nested selection 1,176). `docs/RELEASE_CHECKSUMS.json` holds the SHA256 of the archived feature matrices (`modeling_ready_t80`, `modeling_ready_t85`, `features_causal`) and of the result tables; `docs/pip_freeze.txt` is the frozen environment of the HPC container (Python 3.13, scikit-learn 1.9.0). `bash scripts/verify_release.sh` recomputes both and regenerates the tables from the archived metrics. The matrices and the metrics are deposited in the companion data record.
