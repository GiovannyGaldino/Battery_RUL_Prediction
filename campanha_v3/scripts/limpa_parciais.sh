#!/usr/bin/env bash
# Remove diretorios de modelo incompletos (sem metrics_by_rul_bin.csv, o ultimo arquivo
# escrito) em core_t80 e arm_t85, para que --skip-existing os recompute.
# Com --elasticnet, remove tambem TODOS os diretorios elasticnet/ (mudanca de especificacao).
set -euo pipefail; source "$(dirname "$0")/env.sh"
n=0
for d in $(find "$OUT_V3/core_t80" "$OUT_V3/arm_t85" -mindepth 5 -maxdepth 5 -type d 2>/dev/null); do
  if [ -f "$d/metrics_global.csv" ] && [ ! -f "$d/metrics_by_rul_bin.csv" ]; then rm -rf "$d"; n=$((n+1)); fi
done
echo "parciais removidos: $n"
if [ "${1:-}" = "--elasticnet" ]; then
  m=$(find "$OUT_V3/core_t80" "$OUT_V3/arm_t85" -type d -name elasticnet 2>/dev/null | wc -l)
  find "$OUT_V3/core_t80" "$OUT_V3/arm_t85" -type d -name elasticnet -prune -exec rm -rf {} + 2>/dev/null || true
  echo "diretorios elasticnet removidos: $m"
fi
