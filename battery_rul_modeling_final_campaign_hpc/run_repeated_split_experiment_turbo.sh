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
MASTER_LOG="$LOG_DIR/repeated_split_experiment_turbo_master.log"

SEEDS=(13 73)
PHASES=(charge discharge)
MODES=(global per_dataset)
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
  local seed="$1"
  local phase="$2"
  local mode="$3"
  local experiment="$4"

  local out_root="$CAMPAIGN_ROOT/shards/repeated_split/seed_${seed}/${phase}_${mode}"
  local log_file="$LOG_DIR/repeated_split_seed_${seed}_${phase}_${mode}_${experiment}_turbo.log"

  echo "[$(date '+%F %T')] START seed=${seed} phase=${phase} mode=${mode} experiment=${experiment}" | tee -a "$MASTER_LOG"

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
    --campaign-component repeated_split \
    --random-state "$seed" \
    --phases "$phase" \
    --modes "$mode" \
    --cv-folds 0 \
    >> "$log_file" 2>&1

  echo "[$(date '+%F %T')] END   seed=${seed} phase=${phase} mode=${mode} experiment=${experiment}" | tee -a "$MASTER_LOG"
}

echo "[$(date '+%F %T')] REPEATED SPLIT EXPERIMENT TURBO STARTED max_parallel=${MAX_TASK_PARALLEL} n_jobs=${N_JOBS} model_n_jobs=${MODEL_N_JOBS}" | tee -a "$MASTER_LOG"

for seed in "${SEEDS[@]}"; do
  for phase in "${PHASES[@]}"; do
    for mode in "${MODES[@]}"; do
      for experiment in "${EXPERIMENTS[@]}"; do
        wait_for_slot
        run_one "$seed" "$phase" "$mode" "$experiment" &
      done
    done
  done
done

wait

echo "[$(date '+%F %T')] REPEATED SPLIT EXPERIMENT TURBO FINISHED" | tee -a "$MASTER_LOG"

FINAL_MASTER="$LOG_DIR/final_campaign_master.log"

all_ok=1

for seed in "${SEEDS[@]}"; do
  for phase in "${PHASES[@]}"; do
    for mode in "${MODES[@]}"; do
      root="$CAMPAIGN_ROOT/shards/repeated_split/seed_${seed}/${phase}_${mode}"

      count=$(find "$root" -name metrics_global.csv 2>/dev/null | wc -l)

      if [ "$mode" = "global" ]; then
        expected=78
      else
        expected=624
      fi

      echo "[$(date '+%F %T')] CHECK seed=${seed} phase=${phase} mode=${mode}: ${count}/${expected}" | tee -a "$MASTER_LOG"

      if [ "$count" -ge "$expected" ]; then
        grep -q "RESUME END   component=repeated_split seed=${seed} phase=${phase} mode=${mode} cv=0" "$FINAL_MASTER" || \
          echo "[$(date '+%F %T')] RESUME END   component=repeated_split seed=${seed} phase=${phase} mode=${mode} cv=0" >> "$FINAL_MASTER"
      else
        all_ok=0
      fi
    done
  done
done

if [ "$all_ok" -eq 1 ]; then
  echo "[$(date '+%F %T')] REPEATED SPLIT COMPLETE" | tee -a "$MASTER_LOG"
else
  echo "[$(date '+%F %T')] WARNING: repeated_split still incomplete" | tee -a "$MASTER_LOG"
fi
