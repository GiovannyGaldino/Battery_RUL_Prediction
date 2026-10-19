#!/usr/bin/env bash
# Verifica que o codigo v2 no HPC e o que foi auditado, e aplica os patches.
set -euo pipefail; source "$(dirname "$0")/env.sh"
echo "== dependencias =="
python "$V3/scripts/check_deps.py" || exit 1
echo "== 0. conferindo que o codigo v2 do HPC e o auditado (SHA256) =="
for f in build_hierarchical_features_v3.py:src/feature_extraction \
         build_modeling_from_features_v3.py:src/modeling \
         run_modeling_final_campaign_hpc.py:battery_rul_modeling_final_campaign_hpc \
         merge_modeling_with_curve_features_v3_fixed.py:src/modeling \
         prepare_modeling_data_with_id.py:src/validation; do
  n="${f%%:*}"; d="${f##*:}"
  esperado=$(grep " $n\$" "$V3/pristine/SHA256SUMS" | cut -d' ' -f1)
  atual=$(sha256sum "$REPO/$d/$n" | cut -d' ' -f1)
  if [ "$esperado" != "$atual" ]; then
    echo "  DIVERGE: $d/$n"; echo "    auditado: $esperado"; echo "    no HPC : $atual"
    echo "  O codigo no HPC nao e o que foi auditado. Pare e compare antes de seguir."; exit 1
  fi
  echo "  ok  $d/$n"
done
echo "== 1. aplicando patches (pristine -> patched) =="
for p in "$V3"/patches/patch_0*.py; do python "$p"; done
cp "$V3/pristine/prepare_modeling_data_with_id.py" "$V3/patched/"
echo "== 2. testes locais (sem dados) =="
bash "$V3/scripts/run_tests.sh"
echo "PATCHES APLICADOS E TESTADOS."
