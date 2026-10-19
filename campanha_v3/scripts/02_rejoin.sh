#!/usr/bin/env bash
# Rejunta os alvos novos aos parquets de features v2 (sem reextrair) e cria as
# features causais (causal_*) e o escopo de celula (cell_*).
set -euo pipefail; source "$(dirname "$0")/env.sh"
[ -f "$DATA_V3/targets/_targets/targets_v3.parquet" ] || { echo "ABORTADO: rode 01_rebuild_targets.sh antes (falta targets_v3.parquet)"; exit 1; }
python "$NEW/rejoin_targets.py" --features-root "$FEAT_V2" \
    --targets "$DATA_V3/targets/_targets/targets_v3.parquet" \
    --output-root "$DATA_V3/features_causal" --min-history 3
