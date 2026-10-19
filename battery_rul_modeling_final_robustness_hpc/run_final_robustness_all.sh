#!/bin/bash
set -euo pipefail

cd ~/projects/Batteries/battery_rul_modeling_final_robustness_hpc

START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[START] Final robustness run started at ${START}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] nproc: $(nproc)"

echo ""
echo "################################################################################"
echo "# 1/4 Elastic Net in full matrix"
echo "################################################################################"
bash run_final_01_elasticnet_full_matrix.sh

echo ""
echo "################################################################################"
echo "# 2/4 Compact MLP in reduced matrix"
echo "################################################################################"
bash run_final_02_mlp_reduced_matrix.sh

echo ""
echo "################################################################################"
echo "# 3/4 Repeated seeds for reduced global/per-dataset matrix"
echo "################################################################################"
bash run_final_03_repeated_seeds_reduced.sh

echo ""
echo "################################################################################"
echo "# 4/4 Internal GroupKFold CV for selected global configurations"
echo "################################################################################"
bash run_final_04_internal_cv_global.sh

echo ""
echo "################################################################################"
echo "# Collect final robustness summaries"
echo "################################################################################"
python -u collect_final_robustness_results.py \
  --outputs-root ~/outputs/batteries_modeling_v3 \
  --output-dir ~/outputs/batteries_modeling_v3/final_robustness_summary

END=$(date '+%Y-%m-%d %H:%M:%S')
echo "[DONE] Final robustness run finished at ${END}"
