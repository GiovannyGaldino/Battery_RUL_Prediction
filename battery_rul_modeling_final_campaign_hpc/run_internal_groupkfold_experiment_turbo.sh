#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/Batteries/battery_rul_modeling_final_campaign_hpc}"
INPUT_ROOT="${INPUT_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_ready}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_results_final_campaign_v1}"
LOG_DIR="${LOG_DIR:-$HOME/outputs/batteries_modeling_v3/final_campaign_logs}"

MAX_TASK_PARALLEL="${MAX_TASK_PARALLEL:-16}"
N_JOBS="${N_JOBS:-6}"
MODEL_N_JOBS="${MODEL_N_JOBS:-1}"

TARGET="${TARGET:-target_rul80_steps_clipped}"
TEST_SIZE="${TEST_SIZE:-0.2}"
MISSING_THRESHOLD="${MISSING_THRESHOLD:-0.95}"
SAMPLE_WEIGHTING="${SAMPLE_WEIGHTING:-dataset_inverse}"
MIN_BATTERIES_PER_DATASET="${MIN_BATTERIES_PER_DATASET:-3}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NO_MASTER_TABLES=1

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="$PROJECT_DIR/run_modeling_final_campaign_hpc.py"
MASTER_LOG="$LOG_DIR/internal_groupkfold_experiment_turbo_master.log"

PHASES=(charge discharge)
EXPERIMENTS=(
  all_features
  no_curves
  global_stats_only
  curves_only
  capacity_only
  energy_added
  no_temperature
  no_time_sampling
  no_stage_gap
  resolution_p003
  resolution_p009
  resolution_p015
  resolution_p030
)

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_TASK_PARALLEL )); do
    sleep 5
  done
}

run_one() {
  local phase="$1"
  local experiment="$2"

  local out_root="$CAMPAIGN_ROOT/shards/internal_groupkfold/seed_42/${phase}_global"
  local log_file="$LOG_DIR/internal_groupkfold_seed_42_${phase}_global_${experiment}_turbo.log"

  echo "[$(date '+%F %T')] START phase=${phase} experiment=${experiment}" | tee -a "$MASTER_LOG"

  "$PYTHON_BIN" -u "$SCRIPT" \
    --input-root "$INPUT_ROOT" \
    --target "$TARGET" \
    --experiments "$experiment" \
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
    --campaign-component internal_groupkfold \
    --random-state 42 \
    --phases "$phase" \
    --modes global \
    --cv-folds 3 \
    >> "$log_file" 2>&1

  echo "[$(date '+%F %T')] END   phase=${phase} experiment=${experiment}" | tee -a "$MASTER_LOG"
}

echo "[$(date '+%F %T')] INTERNAL GROUPKFOLD EXPERIMENT TURBO STARTED max_parallel=${MAX_TASK_PARALLEL} n_jobs=${N_JOBS} model_n_jobs=${MODEL_N_JOBS}" | tee -a "$MASTER_LOG"

for phase in "${PHASES[@]}"; do
  for experiment in "${EXPERIMENTS[@]}"; do
    wait_for_slot
    run_one "$phase" "$experiment" &
  done
done

wait

echo "[$(date '+%F %T')] INTERNAL GROUPKFOLD EXPERIMENT TURBO FINISHED" | tee -a "$MASTER_LOG"

FINAL_MASTER="$LOG_DIR/final_campaign_master.log"

for phase in "${PHASES[@]}"; do
  root="$CAMPAIGN_ROOT/shards/internal_groupkfold/seed_42/${phase}_global"
  count=$(find "$root" -name metrics_global.csv 2>/dev/null | wc -l)

  echo "[$(date '+%F %T')] CHECK phase=${phase} mode=global cv=3: ${count}/78" | tee -a "$MASTER_LOG"

  if [ "$count" -ge 78 ]; then
    grep -q "RESUME END   component=internal_groupkfold seed=42 phase=${phase} mode=global cv=3" "$FINAL_MASTER" || \
      echo "[$(date '+%F %T')] RESUME END   component=internal_groupkfold seed=42 phase=${phase} mode=global cv=3" >> "$FINAL_MASTER"
  else
    echo "[$(date '+%F %T')] WARNING: internal_groupkfold incomplete for phase=${phase}" | tee -a "$MASTER_LOG"
  fi
done
