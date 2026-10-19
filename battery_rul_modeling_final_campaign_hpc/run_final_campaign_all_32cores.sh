#!/usr/bin/env bash
set -euo pipefail

# Final consolidated experimental campaign for the Energies manuscript.
# This script intentionally ignores previous runs and writes to a new final-campaign root.
# Designed for a 32-core HPC node by running independent shards safely.

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/Batteries/battery_rul_modeling_final_campaign_hpc}"
INPUT_ROOT="${INPUT_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_ready}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_results_final_campaign_v1}"
LOG_DIR="${LOG_DIR:-$HOME/outputs/batteries_modeling_v3/final_campaign_logs}"

# Parallelism policy for a 32-core node.
# Two concurrent shards, each with two model-level jobs and eight cores per tree ensemble.
# Approximate upper bound: 2 shards x 2 model jobs x 8 tree cores = 32 cores.
MAX_PARALLEL="${MAX_PARALLEL:-2}"
N_JOBS="${N_JOBS:-2}"
MODEL_N_JOBS="${MODEL_N_JOBS:-8}"

TEST_SIZE="${TEST_SIZE:-0.2}"
MISSING_THRESHOLD="${MISSING_THRESHOLD:-0.95}"
SAMPLE_WEIGHTING="${SAMPLE_WEIGHTING:-dataset_inverse}"
MIN_BATTERIES_PER_DATASET="${MIN_BATTERIES_PER_DATASET:-3}"
TARGET="${TARGET:-target_rul80_steps_clipped}"

# Control numerical libraries to avoid oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$CAMPAIGN_ROOT" "$LOG_DIR"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="$PROJECT_DIR/run_modeling_final_campaign_hpc.py"
COMMON_ARGS=(
  --input-root "$INPUT_ROOT"
  --target "$TARGET"
  --experiments all
  --models median elasticnet random_forest extra_trees hist_gradient_boosting mlp
  --test-size "$TEST_SIZE"
  --missing-threshold "$MISSING_THRESHOLD"
  --sample-weighting "$SAMPLE_WEIGHTING"
  --min-batteries-per-dataset "$MIN_BATTERIES_PER_DATASET"
  --n-jobs "$N_JOBS"
  --model-n-jobs "$MODEL_N_JOBS"
  --parallel-backend threads
  --skip-existing
)

# Set SAVE_MODELS=1 only if you really need model.joblib for every fit.
# The default is metrics + predictions + configs, which is enough for analysis and much lighter.
if [[ "${SAVE_MODELS:-0}" == "1" ]]; then
  COMMON_ARGS+=(--save-models)
fi

# Set SAVE_PREDICTIONS=0 to reduce disk usage. Metrics are still generated.
if [[ "${SAVE_PREDICTIONS:-1}" == "0" ]]; then
  COMMON_ARGS+=(--no-save-predictions)
fi

# Set MAKE_PLOTS=1 only for final selected cases. Plotting every case is expensive.
if [[ "${MAKE_PLOTS:-0}" == "1" ]]; then
  COMMON_ARGS+=(--make-plots)
fi

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    sleep 20
  done
}

run_shard() {
  local component="$1"
  local seed="$2"
  local phase="$3"
  local mode="$4"
  local cv_folds="$5"
  local out_root="$CAMPAIGN_ROOT/shards/${component}/seed_${seed}/${phase}_${mode}"
  local log_file="$LOG_DIR/${component}_seed_${seed}_${phase}_${mode}.log"

  mkdir -p "$out_root"
  echo "[$(date '+%F %T')] START component=${component} seed=${seed} phase=${phase} mode=${mode} cv=${cv_folds}" | tee -a "$LOG_DIR/final_campaign_master.log"

  "$PYTHON_BIN" -u "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --output-root "$out_root" \
    --campaign-component "$component" \
    --random-state "$seed" \
    --phases "$phase" \
    --modes "$mode" \
    --cv-folds "$cv_folds" \
    > "$log_file" 2>&1

  echo "[$(date '+%F %T')] END   component=${component} seed=${seed} phase=${phase} mode=${mode} cv=${cv_folds}" | tee -a "$LOG_DIR/final_campaign_master.log"
}

# -----------------------------------------------------------------------------
# 1) Full reference matrix, seed 42, all phases, all modes, all feature ablations,
#    all six model families.
# -----------------------------------------------------------------------------
for phase in charge discharge; do
  for mode in global per_dataset lodo; do
    wait_for_slot
    run_shard "full_matrix" 42 "$phase" "$mode" 0 &
  done
done
wait

# -----------------------------------------------------------------------------
# 2) Repeated grouped splits for split-stability analysis.
#    The reference seed 42 is already included in the full matrix. Here we add
#    seeds 13 and 73 for modes whose holdout depends on random battery splitting.
#    LODO is not repeated because the test domain is determined by the held-out dataset.
# -----------------------------------------------------------------------------
for seed in 13 73; do
  for phase in charge discharge; do
    for mode in global per_dataset; do
      wait_for_slot
      run_shard "repeated_split" "$seed" "$phase" "$mode" 0 &
    done
  done
done
wait

# -----------------------------------------------------------------------------
# 3) Internal GroupKFold by battery for the global mixed-dataset setting.
#    This is a robustness diagnostic, not hyperparameter optimization.
# -----------------------------------------------------------------------------
for phase in charge discharge; do
  wait_for_slot
  run_shard "internal_groupkfold" 42 "$phase" "global" 3 &
done
wait

# -----------------------------------------------------------------------------
# 4) Consolidation and status report.
# -----------------------------------------------------------------------------
"$PYTHON_BIN" -u "$PROJECT_DIR/collect_final_campaign_results.py" \
  --campaign-root "$CAMPAIGN_ROOT" \
  --output-dir "$CAMPAIGN_ROOT/summary" \
  > "$LOG_DIR/collect_final_campaign_results.log" 2>&1

"$PYTHON_BIN" -u "$PROJECT_DIR/check_final_campaign_status.py" \
  --summary-dir "$CAMPAIGN_ROOT/summary" \
  > "$LOG_DIR/check_final_campaign_status.log" 2>&1

echo "[$(date '+%F %T')] FINAL CAMPAIGN FINISHED" | tee -a "$LOG_DIR/final_campaign_master.log"
echo "Results: $CAMPAIGN_ROOT"
echo "Summary: $CAMPAIGN_ROOT/summary"
echo "Logs:    $LOG_DIR"
