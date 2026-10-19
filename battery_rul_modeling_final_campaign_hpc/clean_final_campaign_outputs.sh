#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$HOME/outputs/batteries_modeling_v3/modeling_results_final_campaign_v1}"
LOG_DIR="${LOG_DIR:-$HOME/outputs/batteries_modeling_v3/final_campaign_logs}"

echo "This will delete:"
echo "  $CAMPAIGN_ROOT"
echo "  $LOG_DIR"
read -r -p "Type DELETE to continue: " ans
if [[ "$ans" != "DELETE" ]]; then
  echo "Canceled."
  exit 0
fi
rm -rf "$CAMPAIGN_ROOT" "$LOG_DIR"
echo "Deleted final-campaign outputs and logs."
