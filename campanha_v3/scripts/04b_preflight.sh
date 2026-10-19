#!/usr/bin/env bash
# PORTAO 2b (pre-voo): sem treinar, reproduz a selecao de features e o filtro de
# ausencia do runner em cada experimento x escopo, para os dois alvos e as duas fases.
# Minutos. Qualquer sinalizacao fecha o portao: NAO lance 05/06.
set -euo pipefail; source "$(dirname "$0")/env.sh"
[ -f "$OUT_V3/gate/coverage_t80_charge.csv" ] || { echo "ABORTADO: rode 04_gate.sh antes"; exit 1; }
for T in 80 85; do for P in charge discharge; do
  python "$NEW/preflight.py" --input-root "$DATA_V3/modeling_ready_t$T" --phase "$P" \
      --target "target_rul${T}_steps_clipped" --out "$OUT_V3/gate" --tag "t${T}_${P}" \
      --datasets "$DATASETS_X"
done; done
echo "PORTAO 2b ABERTO: pre-voo sem sinalizacoes nos quatro conjuntos."
