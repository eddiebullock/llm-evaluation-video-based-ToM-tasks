#!/usr/bin/env bash
# Rerun Mindreading video_only for all main-study models after T->V video resolution fix.
# Run from repo root. Uses cache: V-direct trials hit cache; newly resolved T trials call API.
#
#   ./scripts/run_mindreading_video_only_rerun.sh
#   ./scripts/run_mindreading_video_only_rerun.sh --model gemini-3-flash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR_MR="${DATA_DIR_MR:-data/mindreading}"
TRIALS_MR="${TRIALS_MR:-data/trial_definitions/mindreading_emotions_test.json}"
CACHE_DIR="${CACHE_DIR:-cache/full_run}"
RESULTS_DIR="${RESULTS_DIR:-results/full_run}"
SEED="${SEED:-42}"

MODEL=""
ALL=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; ALL=false; shift 2 ;;
    --all) ALL=true; shift ;;
  esac
done

if [[ "$ALL" == true ]]; then
  MODELS=(gemini-3-flash gpt-5 gpt-5-mini claude-opus-4-5)
else
  MODELS=("$MODEL")
fi

RUNNER="$REPO_ROOT/scripts/run_evaluation_detached.sh"

for m in "${MODELS[@]}"; do
  echo ""
  echo "======== Mindreading video_only | $m ========"
  "$RUNNER" \
    --model "$m" \
    --dataset mindreading \
    --condition video_only \
    --trials_file "$TRIALS_MR" \
    --data_dir "$DATA_DIR_MR" \
    --cache_dir "$CACHE_DIR" \
    --results_dir "$RESULTS_DIR" \
    --seed "$SEED" \
    --log "$RESULTS_DIR/mindreading_video_only_${m}.log"

  echo "Waiting for Mindreading run to finish..."
  while pgrep -f "run_evaluation.py.*--model $m.*mindreading" >/dev/null 2>&1; do
    sleep 60
  done
done

echo ""
echo "Done. Run: python analysis/run_study_analysis.py --results-dir results/full_run"
