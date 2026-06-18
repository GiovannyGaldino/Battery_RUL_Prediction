#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/Batteries/battery_rul_modeling_final_campaign_hpc}"
INPUT_ROOT="${INPUT_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_ready}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_results_final_campaign_v1}"
LOG_DIR="${LOG_DIR:-$HOME/outputs/batteries_modeling_v3/final_campaign_logs}"

MAX_MODEL_TASK_PARALLEL="${MAX_MODEL_TASK_PARALLEL:-48}"
MODEL_N_JOBS="${MODEL_N_JOBS:-2}"

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
MASTER_LOG="$LOG_DIR/finish_missing_lodo_model_turbo_master.log"
MISSING_TSV="$LOG_DIR/missing_lodo_model_tasks.tsv"

echo "[$(date '+%F %T')] FINISH MISSING LODO MODEL TURBO STARTED" | tee -a "$MASTER_LOG"

echo "[INFO] Aguardando processos LODO antigos terminarem, se existirem..."
while pgrep -af "rerun_missing_lodo_turbo|run_lodo_byheldout_experiment_turbo|run_lodo_by_heldout_turbo|resume_final_campaign_auto_soft|run_modeling_final_campaign_hpc.py" | grep -v watch | grep -v finish_missing_lodo_model_turbo >/dev/null; do
  pgrep -af "rerun_missing_lodo_turbo|run_lodo_byheldout_experiment_turbo|run_lodo_by_heldout_turbo|resume_final_campaign_auto_soft|run_modeling_final_campaign_hpc.py" | grep -v watch | grep -v finish_missing_lodo_model_turbo || true
  sleep 60
done

make_missing_list() {
python3 - <<'PY'
from pathlib import Path

campaign_root = Path.home() / "outputs/batteries_modeling_v3/modeling_results_final_campaign_v1"
out = Path.home() / "outputs/batteries_modeling_v3/final_campaign_logs/missing_lodo_model_tasks.tsv"

phases = ["charge", "discharge"]
heldouts = ["CALCE", "HUST", "MATR", "NRW", "OXFORD", "PCOE", "RWTH", "XJTU"]
experiments = [
    "all_features",
    "no_curves",
    "global_stats_only",
    "curves_only",
    "capacity_only",
    "energy_added",
    "no_temperature",
    "no_time_sampling",
    "no_stage_gap",
    "resolution_p003",
    "resolution_p009",
    "resolution_p015",
    "resolution_p030",
]
models = ["median", "elasticnet", "random_forest", "extra_trees", "hist_gradient_boosting", "mlp"]

missing = []
charge_count = 0
discharge_count = 0

for phase in phases:
    base = campaign_root / "shards" / "full_matrix" / "seed_42" / f"{phase}_lodo" / phase / "lodo"
    for heldout in heldouts:
        scope = f"heldout_{heldout}"
        for experiment in experiments:
            for model in models:
                f = base / scope / experiment / model / "metrics_global.csv"
                if f.exists():
                    if phase == "charge":
                        charge_count += 1
                    else:
                        discharge_count += 1
                else:
                    missing.append((phase, heldout, experiment, model))

with out.open("w") as fp:
    fp.write("phase\theldout\texperiment\tmodel\n")
    for row in missing:
        fp.write("\t".join(row) + "\n")

print(f"charge_count={charge_count}")
print(f"discharge_count={discharge_count}")
print(f"total_count={charge_count + discharge_count}")
print(f"missing_count={len(missing)}")
print(f"missing_tsv={out}")
PY
}

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_MODEL_TASK_PARALLEL )); do
    sleep 5
  done
}

run_one_model() {
  local phase="$1"
  local heldout="$2"
  local experiment="$3"
  local model="$4"

  local out_root="$CAMPAIGN_ROOT/shards/full_matrix/seed_42/${phase}_lodo"
  local log_file="$LOG_DIR/rerun_missing_model_${phase}_${heldout}_${experiment}_${model}.log"

  echo "[$(date '+%F %T')] START phase=${phase} heldout=${heldout} experiment=${experiment} model=${model}" | tee -a "$MASTER_LOG"

  "$PYTHON_BIN" -u "$SCRIPT" \
    --input-root "$INPUT_ROOT" \
    --target "$TARGET" \
    --experiments "$experiment" \
    --models "$model" \
    --test-size "$TEST_SIZE" \
    --missing-threshold "$MISSING_THRESHOLD" \
    --sample-weighting "$SAMPLE_WEIGHTING" \
    --min-batteries-per-dataset "$MIN_BATTERIES_PER_DATASET" \
    --n-jobs 1 \
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

  echo "[$(date '+%F %T')] END   phase=${phase} heldout=${heldout} experiment=${experiment} model=${model}" | tee -a "$MASTER_LOG"
}

for ROUND in 1 2 3; do
  echo "[$(date '+%F %T')] ROUND ${ROUND} - checking missing LODO model files" | tee -a "$MASTER_LOG"
  make_missing_list | tee -a "$MASTER_LOG"

  MISSING_COUNT=$(tail -n +2 "$MISSING_TSV" | wc -l)

  if [ "$MISSING_COUNT" -eq 0 ]; then
    echo "[$(date '+%F %T')] LODO COMPLETE: no missing files." | tee -a "$MASTER_LOG"

    FINAL_MASTER="$LOG_DIR/final_campaign_master.log"

    grep -q "RESUME END   component=full_matrix seed=42 phase=charge mode=lodo cv=0" "$FINAL_MASTER" || \
      echo "[$(date '+%F %T')] RESUME END   component=full_matrix seed=42 phase=charge mode=lodo cv=0" >> "$FINAL_MASTER"

    grep -q "RESUME END   component=full_matrix seed=42 phase=discharge mode=lodo cv=0" "$FINAL_MASTER" || \
      echo "[$(date '+%F %T')] RESUME END   component=full_matrix seed=42 phase=discharge mode=lodo cv=0" >> "$FINAL_MASTER"

    exit 0
  fi

  echo "[$(date '+%F %T')] Rerunning missing model files: ${MISSING_COUNT}" | tee -a "$MASTER_LOG"

  while IFS=$'\t' read -r phase heldout experiment model; do
    wait_for_slot
    run_one_model "$phase" "$heldout" "$experiment" "$model" &
  done < <(tail -n +2 "$MISSING_TSV")

  wait
done

echo "[$(date '+%F %T')] FINAL CHECK AFTER 3 ROUNDS" | tee -a "$MASTER_LOG"
make_missing_list | tee -a "$MASTER_LOG"

REMAINING=$(tail -n +2 "$MISSING_TSV" | wc -l)
if [ "$REMAINING" -gt 0 ]; then
  echo "[$(date '+%F %T')] WARNING: remaining missing files after 3 rounds: $REMAINING" | tee -a "$MASTER_LOG"
  echo "[INFO] Remaining tasks:"
  cat "$MISSING_TSV"
  exit 2
fi
