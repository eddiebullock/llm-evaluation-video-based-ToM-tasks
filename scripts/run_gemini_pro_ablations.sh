#!/usr/bin/env bash
# DEPRECATED for the main study: Gemini 3 Pro omitted (see analysis/study_config.py).
# Archived for optional reruns only; analysis excludes gemini-3-pro by default.
#
# Usage (foreground):
#   ./scripts/run_gemini_pro_ablations.sh
#
# Usage (detached; monitor in another terminal):
#   ./scripts/run_gemini_pro_ablations.sh --detached
#   tail -f results/full_run/gemini-3-pro_suite.log
#
# Override data roots if needed:
#   DATA_DIR_EU=... DATA_DIR_MR=... ./scripts/run_gemini_pro_ablations.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="gemini-3-pro"
SEED=42
RUNNER="$REPO_ROOT/experiments/run_evaluation.py"
CACHE_DIR="${CACHE_DIR:-cache/full_run}"
RESULTS_DIR="${RESULTS_DIR:-results/full_run}"
TRIALS_EU="${TRIALS_EU:-data/trial_definitions/eu_emotion_test_final.json}"
TRIALS_MR="${TRIALS_MR:-data/trial_definitions/mindreading_emotions_test.json}"

DATA_DIR_EU="${DATA_DIR_EU:-data/eu_emotion}"
DATA_DIR_MR="${DATA_DIR_MR:-data/mindreading}"

DETACHED=0
if [[ "${1:-}" == "--detached" ]]; then
  DETACHED=1
fi

_run_one() {
  local dataset="$1"
  local condition="$2"
  local trials_file="$3"
  local data_dir="$4"
  local log_name="$5"

  local log_path="$RESULTS_DIR/$log_name"
  mkdir -p "$(dirname "$log_path")"

  echo ""
  echo "========== $(date -Iseconds) START $MODEL $dataset $condition =========="
  echo "  trials: $trials_file"
  echo "  data_dir: $data_dir"
  echo "  log: $log_path"

  PYTHONUNBUFFERED=1 python3 -u "$RUNNER" \
    --model "$MODEL" \
    --dataset "$dataset" \
    --condition "$condition" \
    --trials_file "$trials_file" \
    --data_dir "$data_dir" \
    --cache_dir "$CACHE_DIR" \
    --results_dir "$RESULTS_DIR" \
    --seed "$SEED" \
    2>&1 | tee -a "$log_path"

  echo "========== $(date -Iseconds) DONE $MODEL $dataset $condition =========="
}

_run_suite() {
  echo "Gemini 3 Pro ablation suite"
  echo "  repo: $REPO_ROOT"
  echo "  cache: $CACHE_DIR"
  echo "  results: $RESULTS_DIR"
  df -h "$REPO_ROOT" | tail -1 || true

  # EU (118 trials each)
  _run_one eu_emotion video_only "$TRIALS_EU" "$DATA_DIR_EU" "eu_video_only_${MODEL}.log"
  _run_one eu_emotion audio_only "$TRIALS_EU" "$DATA_DIR_EU" "eu_audio_only_${MODEL}.log"
  _run_one eu_emotion multimodal "$TRIALS_EU" "$DATA_DIR_EU" "eu_multimodal_${MODEL}.log"

  # Mindreading (1263 trials each; ~23 videos may fail to decode)
  _run_one mindreading video_only "$TRIALS_MR" "$DATA_DIR_MR" "mindreading_video_only_${MODEL}.log"
  _run_one mindreading audio_only "$TRIALS_MR" "$DATA_DIR_MR" "mindreading_audio_only_${MODEL}.log"
  _run_one mindreading multimodal "$TRIALS_MR" "$DATA_DIR_MR" "mindreading_multimodal_${MODEL}.log"

  echo ""
  echo "All six $MODEL runs finished at $(date -Iseconds)."
  ls -1 "$RESULTS_DIR"/${MODEL}_*_summary.json 2>/dev/null || true
}

if [[ "$DETACHED" -eq 1 ]]; then
  SUITE_LOG="$RESULTS_DIR/gemini-3-pro_suite.log"
  mkdir -p "$(dirname "$SUITE_LOG")"
  echo "Launching detached suite -> $SUITE_LOG"
  nohup bash "$0" >>"$SUITE_LOG" 2>&1 &
  echo "PID=$!"
  echo "Monitor: tail -f $SUITE_LOG"
  exit 0
fi

_run_suite
