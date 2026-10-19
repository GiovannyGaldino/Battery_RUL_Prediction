#!/usr/bin/env bash
# Matriz de modelagem com LISTA DE INCLUSAO, para os dois alvos.
# Uso: bash 03_build_matrix.sh          (matriz base + merge de curvas + prepare)
#      bash 03_build_matrix.sh merge    (so merge de curvas + prepare; a matriz base ja existe)
set -euo pipefail; source "$(dirname "$0")/env.sh"
[ -f "$DATA_V3/features_causal/_targets/rejoin_manifest.json" ] || { echo "ABORTADO: rode 02_rejoin.sh antes"; exit 1; }
[ -d "$CURVES/charge" ] || { echo "ABORTADO: pasta de curvas nao encontrada: $CURVES"; exit 1; }
MODO="${1:-all}"
echo "curvas: $CURVES | modo: $MODO"
for T in 80 85; do
  M="$DATA_V3/modeling_t$T"; MC="$DATA_V3/modeling_t${T}_with_curves"; R="$DATA_V3/modeling_ready_t$T"
  if [ "$MODO" = "all" ]; then
    python "$PATCHED/build_modeling_from_features_v3.py" --input-root "$DATA_V3/features_causal" \
        --output-root "$M" --target "target_rul${T}_steps_clipped" --column-policy allowlist
  else
    [ -f "$M/charge_phase_modeling.parquet" ] || { echo "ABORTADO: matriz base ausente em $M; rode sem 'merge'"; exit 1; }
  fi
  rm -rf "$MC" "$R"
  python "$PATCHED/merge_modeling_with_curve_features_v3_fixed.py" \
      --modeling-root "$M" --curve-root "$CURVES" --output-root "$MC"
  python "$PATCHED/prepare_modeling_data_with_id.py" --features-root "$MC" --output-root "$R" \
      --target "target_rul${T}_steps_clipped"
  echo "curve_root usado: $CURVES" > "$R/CURVE_SOURCE.txt"
  echo "pronto: $R"
done
