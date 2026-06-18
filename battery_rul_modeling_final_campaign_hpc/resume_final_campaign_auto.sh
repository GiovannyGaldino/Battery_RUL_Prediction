#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/Batteries/battery_rul_modeling_final_campaign_hpc}"
INPUT_ROOT="${INPUT_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_ready}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_results_final_campaign_v1}"
LOG_DIR="${LOG_DIR:-$HOME/outputs/batteries_modeling_v3/final_campaign_logs}"

MAX_PARALLEL="${MAX_PARALLEL:-2}"
N_JOBS="${N_JOBS:-2}"
MODEL_N_JOBS="${MODEL_N_JOBS:-8}"

TEST_SIZE="${TEST_SIZE:-0.2}"
MISSING_THRESHOLD="${MISSING_THRESHOLD:-0.95}"
SAMPLE_WEIGHTING="${SAMPLE_WEIGHTING:-dataset_inverse}"
MIN_BATTERIES_PER_DATASET="${MIN_BATTERIES_PER_DATASET:-3}"
TARGET="${TARGET:-target_rul80_steps_clipped}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$CAMPAIGN_ROOT" "$LOG_DIR"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="$PROJECT_DIR/run_modeling_final_campaign_hpc.py"
MASTER_LOG="$LOG_DIR/final_campaign_master.log"

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

is_completed() {
  local component="$1"
  local seed="$2"
  local phase="$3"
  local mode="$4"
  local cv_folds="$5"

  grep -E "END|RESUME END" "$MASTER_LOG" 2>/dev/null | grep -q "component=${component} seed=${seed} phase=${phase} mode=${mode} cv=${cv_folds}"
}

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

  if is_completed "$component" "$seed" "$phase" "$mode" "$cv_folds"; then
    echo "[$(date '+%F %T')] SKIP completed component=${component} seed=${seed} phase=${phase} mode=${mode} cv=${cv_folds}" | tee -a "$MASTER_LOG"
    return 0
  fi

  echo "[$(date '+%F %T')] RESTART incomplete component=${component} seed=${seed} phase=${phase} mode=${mode} cv=${cv_folds}" | tee -a "$MASTER_LOG"

  rm -rf "$out_root"
  rm -f "$log_file"
  mkdir -p "$out_root"

  echo "[$(date '+%F %T')] RESUME START component=${component} seed=${seed} phase=${phase} mode=${mode} cv=${cv_folds}" | tee -a "$MASTER_LOG"

  "$PYTHON_BIN" -u "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --output-root "$out_root" \
    --campaign-component "$component" \
    --random-state "$seed" \
    --phases "$phase" \
    --modes "$mode" \
    --cv-folds "$cv_folds" \
    > "$log_file" 2>&1

  echo "[$(date '+%F %T')] RESUME END   component=${component} seed=${seed} phase=${phase} mode=${mode} cv=${cv_folds}" | tee -a "$MASTER_LOG"
}

echo "[$(date '+%F %T')] AUTO SAFE RESUME STARTED" | tee -a "$MASTER_LOG"

# 1) Matriz final completa, seed 42.
for phase in charge discharge; do
  for mode in global per_dataset lodo; do
    wait_for_slot
    run_shard "full_matrix" 42 "$phase" "$mode" 0 &
  done
done
wait

# 2) Repetição por sementes para global/per_dataset.
# A seed 42 da matriz principal já cobre o caso base; aqui retomamos seeds 13 e 73.
for seed in 13 73; do
  for phase in charge discharge; do
    for mode in global per_dataset; do
      wait_for_slot
      run_shard "repeated_split" "$seed" "$phase" "$mode" 0 &
    done
  done
done
wait

# 3) GroupKFold interno no modo global.
for phase in charge discharge; do
  wait_for_slot
  run_shard "internal_groupkfold" 42 "$phase" "global" 3 &
done
wait

# 4) Consolidação final.
echo "[$(date '+%F %T')] COLLECT START" | tee -a "$MASTER_LOG"

if [ -f "$PROJECT_DIR/collect_final_campaign_results.py" ]; then
  "$PYTHON_BIN" -u "$PROJECT_DIR/collect_final_campaign_results.py" \
    --campaign-root "$CAMPAIGN_ROOT" \
    --output-dir "$CAMPAIGN_ROOT/summary" \
    > "$LOG_DIR/collect_final_campaign_results.log" 2>&1
else
  echo "collect_final_campaign_results.py não encontrado." | tee -a "$MASTER_LOG"
  exit 1
fi

if [ -f "$PROJECT_DIR/check_final_campaign_status.py" ]; then
  "$PYTHON_BIN" -u "$PROJECT_DIR/check_final_campaign_status.py" \
    --summary-dir "$CAMPAIGN_ROOT/summary" \
    > "$LOG_DIR/check_final_campaign_status.log" 2>&1 || true
fi

echo "[$(date '+%F %T')] COLLECT END" | tee -a "$MASTER_LOG"
echo "[$(date '+%F %T')] FINAL CAMPAIGN FINISHED AFTER AUTO SAFE RESUME" | tee -a "$MASTER_LOG"

echo "Results: $CAMPAIGN_ROOT"
echo "Summary: $CAMPAIGN_ROOT/summary"
echo "Logs:    $LOG_DIR"
