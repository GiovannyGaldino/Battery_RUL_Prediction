#!/usr/bin/env bash
set -euo pipefail

echo "Active final-campaign processes:"
pgrep -af "run_modeling_final_campaign_hpc.py|run_final_campaign_all_32cores.sh" || true

echo "Stopping final-campaign processes with SIGTERM..."
pkill -TERM -f "run_modeling_final_campaign_hpc.py" || true
pkill -TERM -f "run_final_campaign_all_32cores.sh" || true
sleep 10

echo "Remaining processes:"
pgrep -af "run_modeling_final_campaign_hpc.py|run_final_campaign_all_32cores.sh" || true

echo "If anything remains after a minute, run:"
echo "pkill -9 -f 'run_modeling_final_campaign_hpc.py|run_final_campaign_all_32cores.sh'"
