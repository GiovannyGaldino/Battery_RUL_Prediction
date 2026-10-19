#!/usr/bin/env bash
# Depois do nucleo completo: mata o braco B de 1 processo (se ainda vivo), relanca o braco B
# em 4 processos e o braco R com paralelismo 4; cada processo com 4 threads (32 na cota).
# O braco N (07n) vai depois, quando B e R terminarem: bash 07n_nested.sh 8 com MODEL_N_JOBS=4.
set -euo pipefail; source "$(dirname "$0")/env.sh"
feitos=$(find "$OUT_V3/core_t80" -name metrics_global.csv | wc -l)
[ "$feitos" -ge 2520 ] || { echo "ABORTADO: nucleo com $feitos de 2520"; exit 1; }
pkill -f run_modeling_final_campaign_hpc.py || true; sleep 5
bash "$V3/scripts/limpa_parciais.sh"
export MODEL_N_JOBS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
bash "$V3/scripts/07b_threshold85.sh"
bash "$V3/scripts/07r_repeated.sh" 4
sleep 60; bash "$V3/scripts/status.sh" | head -5; uptime
