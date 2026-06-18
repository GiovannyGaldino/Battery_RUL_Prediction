#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/Batteries/battery_rul_modeling_final_campaign_hpc}"
LOG_DIR="${LOG_DIR:-$HOME/outputs/batteries_modeling_v3/final_campaign_logs}"
mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

nohup bash run_final_campaign_all_32cores.sh \
  > "$LOG_DIR/final_campaign_nohup.log" 2>&1 &

echo "Final campaign launched. PID: $!"
echo "Master log: $LOG_DIR/final_campaign_master.log"
echo "Nohup log:  $LOG_DIR/final_campaign_nohup.log"
echo "Use: tail -f $LOG_DIR/final_campaign_master.log"
