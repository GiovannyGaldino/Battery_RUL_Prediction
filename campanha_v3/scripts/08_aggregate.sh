#!/usr/bin/env bash
# Tabelas do manuscrito a partir do nucleo (+ braco R e selecao aninhada, se existirem).
# Uso: bash 08_aggregate.sh [--force]   (sem --force exige o nucleo completo: 2520 ajustes)
set -euo pipefail; source "$(dirname "$0")/env.sh"
feitos=$(find "$OUT_V3/core_t80" -name metrics_global.csv 2>/dev/null | wc -l)
if [ "$feitos" -lt 2520 ] && [ "${1:-}" != "--force" ]; then
  echo "ABORTADO: nucleo com $feitos de 2520 ajustes; use --force para tabelas parciais"; exit 1; fi
ARGS=(--core "$OUT_V3/core_t80" --out "$OUT_V3/tables")
nr=$(find "$OUT_V3/repeated" -name metrics_global.csv 2>/dev/null | wc -l)
if [ "$nr" -ge 1920 ]; then ARGS+=(--repeated "$OUT_V3/repeated"); else echo "braco R com $nr de 1920 ajustes: intra-dominio com 1 particao do nucleo"; fi
[ -f "$OUT_V3/nested/nested_selection_nMAE.csv" ] && ARGS+=(--nested "$OUT_V3/nested/nested_selection_nMAE.csv")
python "$NEW/aggregate.py" "${ARGS[@]}"
if [ -d "$OUT_V3/arm_t85" ] && [ "$(find "$OUT_V3/arm_t85" -name metrics_global.csv | wc -l)" -ge 360 ]; then
  echo; echo "== braco B (limiar 0.85), so as tabelas que nao dependem de ablacoes"
  python "$NEW/aggregate.py" --core "$OUT_V3/arm_t85" --out "$OUT_V3/tables_t85" | tail -3 || true
fi
