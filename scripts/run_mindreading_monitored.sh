#!/usr/bin/env bash
# Mindreading: smoke-test all 3 conditions, then run full evaluation sequentially.
#
# Good practice: one condition at a time (same API key, shared cache, easier logs).
# Do NOT run video_only + audio_only + multimodal in parallel unless you accept rate limits.
#
# Usage (from repo root):
#   export MR_DATA_DIR="/path/to/MindReading/Emotions"
#   ./scripts/run_mindreading_monitored.sh
#
# Smoke only (no full run):
#   ./scripts/run_mindreading_monitored.sh --smoke-only
#
# Skip smoke (full run only):
#   ./scripts/run_mindreading_monitored.sh --skip-smoke
#
# Check progress anytime (another terminal):
#   ./scripts/check_evaluation_status.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-gemini-3-flash}"
SEED="${SEED:-42}"
MR_DATA_DIR="${MR_DATA_DIR:-data/mindreading}"
TRIALS_FULL="${TRIALS_FULL:-data/trial_definitions/mindreading_emotions_test.json}"
TRIALS_SMOKE="${TRIALS_SMOKE:-data/trial_definitions/smoke_mindreading_test_20.json}"
CACHE_DIR="${CACHE_DIR:-cache/full_run}"
RESULTS_DIR="${RESULTS_DIR:-results/full_run}"
SMOKE_N="${SMOKE_N:-20}"

RUNNER="experiments/run_evaluation.py"
# run_evaluation.py adds .deps to sys.path; PYTHONPATH optional.
export PYTHONPATH=".:${PYTHONPATH:-}"

SMOKE_ONLY=0
SKIP_SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --smoke-only) SMOKE_ONLY=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    *) echo "Unknown arg: $arg"; exit 2 ;;
  esac
done

mkdir -p "$CACHE_DIR" "$RESULTS_DIR"

run_one() {
  local condition="$1"
  local trials_file="$2"
  local max_trials="${3:-}"
  local tag="$4"
  local log="$RESULTS_DIR/mindreading_${condition}_${MODEL}_${tag}.log"

  echo ""
  echo "========== $tag | condition=$condition | trials=$trials_file =========="
  local -a cmd=(
    python "$RUNNER"
    --model "$MODEL"
    --dataset mindreading
    --condition "$condition"
    --trials_file "$trials_file"
    --data_dir "$MR_DATA_DIR"
    --cache_dir "$CACHE_DIR"
    --results_dir "$RESULTS_DIR"
    --seed "$SEED"
  )
  if [[ -n "$max_trials" ]]; then
    cmd+=(--max_trials "$max_trials")
  fi

  echo "Log: $log"
  local pid
  pid="$(
    ./scripts/run_evaluation_detached.sh \
      --log "$log" \
      "${cmd[@]:2}" 2>&1 | sed -n 's/^PID=//p' | tail -1
  )"
  if [[ -z "$pid" ]]; then
    echo "Failed to start detached run; see $log"
    return 1
  fi
  echo "Waiting for PID $pid ..."
  wait "$pid" || true
  echo "--- $tag $condition done ---"
  ./scripts/check_evaluation_status.sh "$RESULTS_DIR"
}

if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  echo ">>> Phase 1: SMOKE ($SMOKE_N trials × 3 conditions) — expect ~5–15 minutes total"
  for cond in video_only audio_only multimodal; do
    run_one "$cond" "$TRIALS_SMOKE" "$SMOKE_N" "smoke"
  done
  echo ""
  echo ">>> Smoke finished. Inspect summaries above."
  echo "    Decode errors on some .mov files in logs are normal."
  if [[ "$SMOKE_ONLY" -eq 1 ]]; then
    echo ">>> --smoke-only set; exiting before full run."
    exit 0
  fi
  echo ">>> Starting full run in 10s (Ctrl+C to abort)..."
  sleep 10
fi

echo ">>> Phase 2: FULL (~583 processable trials × 3 conditions) — expect many hours"
echo ">>> Monitor in another terminal:"
echo "    ./scripts/check_evaluation_status.sh $RESULTS_DIR"
echo "    tail -f $RESULTS_DIR/mindreading_video_only_${MODEL}_full.log"

for cond in video_only audio_only multimodal; do
  run_one "$cond" "$TRIALS_FULL" "" "full"
done

echo ""
echo ">>> All Mindreading conditions complete."
