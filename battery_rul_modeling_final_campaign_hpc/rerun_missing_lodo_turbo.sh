#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/Batteries/battery_rul_modeling_final_campaign_hpc}"
INPUT_ROOT="${INPUT_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_ready}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_results_final_campaign_v1}"
LOG_DIR="${LOG_DIR:-$HOME/outputs/batteries_modeling_v3/final_campaign_logs}"

MAX_TASK_PARALLEL="${MAX_TASK_PARALLEL:-24}"
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
MASTER_LOG="$LOG_DIR/rerun_missing_lodo_turbo_master.log"
MISSING_TSV="$LOG_DIR/missing_lodo_tasks.tsv"

python3 - <<'PY'
from pathlib import Path
from collections import defaultdict

campaign_root = Path.home() / "outputs/batteries_modeling_v3/modeling_results_final_campaign_v1"

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
models = [
    "median",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "mlp",
]

missing_files = []
missing_tasks = defaultdict(int)

for phase in phases:
    base = campaign_root / "shards" / "full_matrix" / "seed_42" / f"{phase}_lodo" / phase / "lodo"
    for heldout in heldouts:
        scope = f"heldout_{heldout}"
        for experiment in experiments:
            for model in models:
                f = base / scope / experiment / model / "metrics_global.csv"
                if not f.exists():
                    missing_files.append(str(f))
                    missing_tasks[(phase, heldout, experiment)] += 1

out = Path.home() / "outputs/batteries_modeling_v3/final_campaign_logs/missing_lodo_tasks.tsv"
with out.open("w") as fp:
    fp.write("phase\theldout\texperiment\tmissing_model_files\n")
    for (phase, heldout, experiment), n in sorted(missing_tasks.items()):
        fp.write(f"{phase}\t{heldout}\t{experiment}\t{n}\n")

print(f"[INFO] Missing metric files: {len(missing_files)}")
print(f"[INFO] Missing phase-heldout-experiment tasks: {len(missing_tasks)}")
print(f"[INFO] Task list: {out}")

for (phase, heldout, experiment), n in sorted(missing_tasks.items()):
    print(f"{phase:9s} {heldout:8s} {experiment:20s} missing_models={n}")
PY

if [ ! -s "$MISSING_TSV" ]; then
  echo "[INFO] Nenhuma tarefa faltante encontrada."
  exit 0
fi

TOTAL_TASKS=$(tail -n +2 "$MISSING_TSV" | wc -l)
echo "[$(date '+%F %T')] RERUN MISSING LODO STARTED tasks=${TOTAL_TASKS} max_parallel=${MAX_TASK_PARALLEL} n_jobs=${N_JOBS} model_n_jobs=${MODEL_N_JOBS}" | tee -a "$MASTER_LOG"

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_TASK_PARALLEL )); do
    sleep 5
  done
}

run_one() {
  local phase="$1"
  local heldout="$2"
  local experiment="$3"

  local out_root="$CAMPAIGN_ROOT/shards/full_matrix/seed_42/${phase}_lodo"
  local log_file="$LOG_DIR/rerun_missing_${phase}_lodo_${heldout}_${experiment}.log"

  echo "[$(date '+%F %T')] START phase=${phase} heldout=${heldout} experiment=${experiment}" | tee -a "$MASTER_LOG"

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
    --campaign-component full_matrix \
    --random-state 42 \
    --phases "$phase" \
    --modes lodo \
    --lodo-heldouts "$heldout" \
    --cv-folds 0 \
    >> "$log_file" 2>&1

  echo "[$(date '+%F %T')] END   phase=${phase} heldout=${heldout} experiment=${experiment}" | tee -a "$MASTER_LOG"
}

tail -n +2 "$MISSING_TSV" | while IFS=$'\t' read -r phase heldout experiment missing_count; do
  wait_for_slot
  run_one "$phase" "$heldout" "$experiment" &
done

wait

echo "[$(date '+%F %T')] RERUN MISSING LODO FINISHED" | tee -a "$MASTER_LOG"
