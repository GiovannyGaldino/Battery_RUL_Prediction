#!/usr/bin/env bash
# PORTAO 2: (a) cobertura por familia e dominio; (b) auditoria de causalidade.
# Qualquer sinalizacao fecha o portao: NAO treine.
set -euo pipefail; source "$(dirname "$0")/env.sh"
[ -f "$DATA_V3/modeling_ready_t80/charge_X.parquet" ] || { echo "ABORTADO: rode 03_build_matrix.sh antes"; exit 1; }
for T in 80 85; do for P in charge discharge; do
  R="$DATA_V3/modeling_ready_t$T"
  python "$NEW/coverage_gate.py" --x "$R/${P}_X.parquet" --id "$R/${P}_id.parquet" \
      --out "$OUT_V3/gate" --tag "t${T}_${P}"
done; done
for T in 80 85; do for P in charge discharge; do
  R="$DATA_V3/modeling_ready_t$T"
  python "$NEW/causality_audit.py" --x "$R/${P}_X.parquet" --id "$R/${P}_id.parquet" --y "$R/${P}_y.parquet" \
      --out "$OUT_V3/gate" --tag "t${T}_${P}"
done; done
echo "PORTAO 2 ABERTO nos quatro conjuntos (cobertura e causalidade)."
