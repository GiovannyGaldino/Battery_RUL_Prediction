#!/usr/bin/env bash
# Reconstroi SO os alvos (--dry-run, minutos). Dois limiares numa rodada:
# primario 0.80 (colunas eol80/rul80) e secundario 0.85 (eol85/rul85).
# Permanencia escalada (limiar + 0.10) e checada na serie suavizada.
set -euo pipefail; source "$(dirname "$0")/env.sh"
OUT="$DATA_V3/targets"; REP="$DATA_V3/reports_targets"; mkdir -p "$OUT" "$REP"
python "$PATCHED/build_hierarchical_features_v3.py" --dry-run \
    --input-root "$INTERIM" --output-root "$OUT" --reports-root "$REP" --dataset "$DATASETS" \
    --initial-window 5 --initial-fraction 0.02 \
    --eol-rolling-window 5 --eol-consecutive 3 --eol-min-position 3 \
    --eol-threshold 0.80 --eol-secondary-threshold 0.85 \
    --eol-permanence-mode margin --eol-permanence-margin 0.10 --eol-permanence-on smooth
echo "== PORTAO 1: contagem de celulas por limiar (medido em 2026-09-03: 469 a 0.80, 691 a 0.85; motivos: never_crossed 330/108, no_persistence 9/12, permanence_violated 6/3) =="
python - <<PY
import pandas as pd
s = pd.read_csv("$OUT/_targets/capacity_life_summary_v3.csv", encoding="utf-8-sig")
print("pares celula-fase:", len(s))
for k in ["eol80", "eol85"]:
    print(f"  {k}_found: {int(s[k+'_found'].sum())}   motivos:", s[k+"_reject_reason"].value_counts().to_dict())
print(s.groupby(["dataset","phase_folder"])[["eol80_found","eol85_found"]].sum().astype(int).to_string())
PY
