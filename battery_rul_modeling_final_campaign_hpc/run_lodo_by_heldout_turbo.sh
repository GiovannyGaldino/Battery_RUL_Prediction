#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/Batteries/battery_rul_modeling_final_campaign_hpc}"
INPUT_ROOT="${INPUT_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_ready}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_results_final_campaign_v1}"
LOG_DIR="${LOG_DIR:-$HOME/outputs/batteries_modeling_v3/final_campaign_logs}"

MAX_HELDOUT_PARALLEL="${MAX_HELDOUT_PARALLEL:-16}"
N_JOBS="${N_JOBS:-6}"
MODEL_N_JOBS="${MODEL_N_JOBS:-1}"

TARGET="${TARGET:-target_rul80_steps_clipped}"
TEST_SIZE="${TEST_SIZE:-0.2}"
MISSING_THRESHOLD="${MISSING_THRESHOLD:-0.95}"
SAMPLE_WEIGHTING="${SAMPLE_WEIGHTING:-dataset_inverse}"
MIN_BATTERIES_PER_DATASET="${MIN_BATTERIES_PER_DATASET:-3}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NO_MASTER_TABLES=1

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="$PROJECT_DIR/run_modeling_final_campaign_hpc.py"
MASTER_LOG="$LOG_DIR/lodo_byheldout_turbo_master.log"

PHASES=(charge discharge)
HELDOUTS=(CALCE HUST MATR NRW OXFORD PCOE RWTH XJTU)

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_HELDOUT_PARALLEL )); do
    sleep 10
  done
}

run_one() {
  local phase="$1"
  local heldout="$2"
  local out_root="$CAMPAIGN_ROOT/shards/full_matrix/seed_42/${phase}_lodo"
  local log_file="$LOG_DIR/full_matrix_seed_42_${phase}_lodo_${heldout}_turbo.log"

  echo "[$(date '+%F %T')] START phase=${phase} heldout=${heldout}" | tee -a "$MASTER_LOG"

  "$PYTHON_BIN" -u "$SCRIPT" \
    --input-root "$INPUT_ROOT" \
    --target "$TARGET" \
    --experiments all \
    --models median elasticnet random_forest extra_trees hist_gradient_boosting mlp \
    --test-size "$TEST_SIZE" \
    --missing-threshold "$MISSING_THRESHOLD" \
    --sample-weighting "$SAMPLE_WEIGHTING" \
    --min-batteries-per-dataset "$MIN_BATTERIES_PER_DATASET" \
    --n-jobs "$N_JOBS" \
    --model-n-jobs "$MODEL_N_JOBS" \
    --parallel-backend threads \
    --skip-existing \
    --output-root "$out_root" \
    --campaign-component full_matrix \
    --random-state 42 \
    --phases "$phase" \
    --modes lodo \
    --lodo-heldouts "$heldout" \
    --cv-folds 0 \
    >> "$log_file" 2>&1

  echo "[$(date '+%F %T')] END   phase=${phase} heldout=${heldout}" | tee -a "$MASTER_LOG"
}

echo "[$(date '+%F %T')] LODO BY-HELDOUT TURBO STARTED max_parallel=${MAX_HELDOUT_PARALLEL} n_jobs=${N_JOBS} model_n_jobs=${MODEL_N_JOBS}" | tee -a "$MASTER_LOG"

for phase in "${PHASES[@]}"; do
  for heldout in "${HELDOUTS[@]}"; do
    wait_for_slot
    run_one "$phase" "$heldout" &
  done
done

wait

echo "[$(date '+%F %T')] LODO BY-HELDOUT TURBO FINISHED" | tee -a "$MASTER_LOG"
