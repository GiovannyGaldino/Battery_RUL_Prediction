#!/usr/bin/env bash
# Progresso de cada componente da campanha v3.
source "$(dirname "$0")/env.sh"
c() { find "$1" -name metrics_global.csv 2>/dev/null | wc -l; }
printf "%-12s %6s / %-6s  %s\n" componente feito alvo raiz
printf "%-12s %6s / %-6s  %s\n" nucleo_t80  "$(c "$OUT_V3/core_t80")"  2520 "$OUT_V3/core_t80"
printf "%-12s %6s / %-6s  %s\n" braco_t85   "$(c "$OUT_V3/arm_t85")"   360  "$OUT_V3/arm_t85"
printf "%-12s %6s / %-6s  %s\n" nested      "$(c "$OUT_V3/nested")"    1176 "$OUT_V3/nested"
printf "%-12s %6s / %-6s  %s\n" repetidos   "$(c "$OUT_V3/repeated")"  1920 "$OUT_V3/repeated"
echo; echo "processos do runner vivos: $(pgrep -f run_modeling_final_campaign | wc -l)"
for f in "$OUT_V3"/logs_core/shard_*.log; do
  [ -f "$f" ] || continue
  n=$(grep -ac '^\[DONE\] .* | ' "$f" 2>/dev/null; true)
  ult=$(grep -a '^\[CASE\]' "$f" 2>/dev/null | tail -1 | cut -c1-100)
  printf "  %-10s %4s  %s\n" "$(basename "$f" .log)" "$n" "$ult"
done
