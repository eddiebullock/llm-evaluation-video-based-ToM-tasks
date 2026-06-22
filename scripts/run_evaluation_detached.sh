#!/usr/bin/env bash
# Run evaluation fully detached (macOS + Linux). Safe to `tail -f` in another terminal.
#
# Usage:
#   ./scripts/run_evaluation_detached.sh \
#     --model gemini-3-flash --dataset mindreading --condition audio_only \
#     --trials_file data/trial_definitions/mindreading_emotions_test.json \
#     --data_dir "/path/to/MindReading/Emotions" \
#     --cache_dir cache/full_run \
#     --results_dir results/full_run \
#     --seed 42 \
#     --log results/full_run/mindreading_audio_only.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log)
      LOG="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$LOG" ]]; then
  echo "Pass --log <path> for output (e.g. results/full_run/my_run.log)"
  exit 2
fi

mkdir -p "$(dirname "$LOG")"
LOG_ABS="$(cd "$(dirname "$LOG")" && pwd)/$(basename "$LOG")"
RUNNER="$REPO_ROOT/experiments/run_evaluation.py"

echo "Starting detached evaluation..." >&2
echo "  log: $LOG_ABS" >&2
echo "  monitor in another terminal: tail -f $LOG_ABS" >&2

PID="$(
  PYTHONUNBUFFERED=1 python3 -u - "$RUNNER" "$LOG_ABS" "${ARGS[@]}" <<'PY'
import os
import subprocess
import sys

runner = sys.argv[1]
log_path = sys.argv[2]
args = sys.argv[3:]

with open(log_path, "ab", buffering=0) as log_f:
    proc = subprocess.Popen(
        [sys.executable, "-u", runner, *args],
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
        start_new_session=True,
    )
print(proc.pid, flush=True)
PY
)"

echo "PID=$PID"
sleep 3
if kill -0 "$PID" 2>/dev/null; then
  STAT="$(ps -p "$PID" -o stat= 2>/dev/null | tr -d ' ' || echo "?")"
  if [[ "$STAT" == *T* ]]; then
    echo "WARNING: process looks stopped (stat=$STAT). Try tail from a different terminal tab." >&2
  else
    echo "Running (stat=$STAT)." >&2
  fi
  if [[ -s "$LOG_ABS" ]]; then
    echo "--- log tail ---" >&2
    tail -5 "$LOG_ABS" >&2
  else
    echo "Log empty so far (normal for a few seconds)." >&2
  fi
else
  echo "Process exited immediately; check log:" >&2
  cat "$LOG_ABS" 2>/dev/null || true
  exit 1
fi
