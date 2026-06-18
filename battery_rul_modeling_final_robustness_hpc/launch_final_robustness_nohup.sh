#!/bin/bash
set -euo pipefail

mkdir -p ~/outputs/batteries_modeling_v3
cd ~/projects/Batteries/battery_rul_modeling_final_robustness_hpc

nohup bash run_final_robustness_all.sh \
  > ~/outputs/batteries_modeling_v3/modeling_final_robustness_all.log 2>&1 &

echo "Final robustness run launched. PID: $!"
echo "Log: ~/outputs/batteries_modeling_v3/modeling_final_robustness_all.log"
echo "Monitor with: tail -f ~/outputs/batteries_modeling_v3/modeling_final_robustness_all.log"
