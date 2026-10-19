#!/usr/bin/env bash
# Confere hashes das matrizes de entrada, contagem de ajustes e regenera T3-T6 contra as arquivadas.
# Uso: bash verify_release.sh [--write]   (--write so na maquina onde a campanha rodou, uma vez)
set -euo pipefail; source "$(dirname "$0")/env.sh"
python "$NEW/verify_release.py" --data-root "$DATA_V3" --out-root "$OUT_V3" "$@"
