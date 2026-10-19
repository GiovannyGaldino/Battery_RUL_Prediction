#!/usr/bin/env bash
# Depois de um reinicio do conteiner: confere dependencias, limpa diretorios de modelo
# incompletos e relanca o nucleo e o braco B com --skip-existing (so o que falta roda).
# Uso: bash retoma.sh            (MODEL_N_JOBS do env.sh, 2 por padrao)
set -euo pipefail; source "$(dirname "$0")/env.sh"
python "$V3/scripts/check_deps.py"
[ -d "$OUT_V3/quick_test" ] || { echo "ABORTADO: nao ha quick_test; a campanha ainda nao foi liberada (rode 05)"; exit 1; }
vivos=$(pgrep -fc run_modeling_final_campaign_hpc.py || true)
[ "$vivos" -eq 0 ] || { echo "ABORTADO: ja ha $vivos processos do runner vivos"; exit 1; }
bash "$V3/scripts/limpa_parciais.sh"
echo "retomando com MODEL_N_JOBS=$MODEL_N_JOBS (OMP/BLAS=$OMP_NUM_THREADS)"
bash "$V3/scripts/06_campaign.sh" 14
bash "$V3/scripts/07b_threshold85.sh"
sleep 30; bash "$V3/scripts/status.sh" | head -3
