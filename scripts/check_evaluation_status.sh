#!/usr/bin/env bash
# Quick status for evaluation runs (read-only).
# Usage:
#   ./scripts/check_evaluation_status.sh
#   ./scripts/check_evaluation_status.sh results/full_run

set -euo pipefail

RESULTS_DIR="${1:-results/full_run}"

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "No results directory: $RESULTS_DIR"
  exit 1
fi

echo "=== Evaluation status: $RESULTS_DIR ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo

# Running processes?
if pgrep -fl "run_evaluation.py" >/dev/null 2>&1; then
  echo "--- Active run_evaluation.py processes ---"
  pgrep -fl "run_evaluation.py" || true
  echo
else
  echo "--- No run_evaluation.py process found ---"
  echo
fi

echo "--- Summary JSON files ---"
shopt -s nullglob
summaries=("$RESULTS_DIR"/*_summary.json)
if ((${#summaries[@]} == 0)); then
  echo "(none yet)"
else
  for f in "${summaries[@]}"; do
    python3 -c "
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.load(p.open())
print(
    f\"{p.name}: {d.get('model')} {d.get('dataset')} {d.get('condition')} | \"
    f\"valid={d.get('n_valid')}/{d.get('n_total')} acc={d.get('accuracy_percent', 0):.1f}%\"
)
" "$f"
  done
fi
echo

echo "--- Results CSV progress (rows written) ---"
csvs=("$RESULTS_DIR"/*_results.csv)
if ((${#csvs[@]} == 0)); then
  echo "(none yet)"
else
  for f in "${csvs[@]}"; do
    n=$(($(wc -l <"$f") - 1))
    echo "$f: $n trials"
  done
fi
echo

echo "--- Latest log tail (if any) ---"
logs=("$RESULTS_DIR"/*.log)
if ((${#logs[@]} == 0)); then
  echo "(no .log files)"
else
  latest=$(ls -t "${logs[@]}" | head -1)
  echo "File: $latest"
  tail -5 "$latest"
fi
