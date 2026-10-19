#!/usr/bin/env bash
set -euo pipefail; V3="${V3:-$(cd "$(dirname "$0")/.." && pwd)}"
falhas=0; for t in "$V3"/tests/test_0*.py; do echo "--- $(basename "$t")"; python "$t" | tail -1 || falhas=$((falhas+1)); done; [ "$falhas" -eq 0 ] && echo "todas as suites passaram" || { echo "$falhas suite(s) falharam"; exit 1; }
