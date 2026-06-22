from __future__ import annotations

"""
Models included in the main study analyses.

Gemini 3 Pro was omitted after pilot evaluation: MR video-only accuracy was
effectively identical to Gemini 3 Flash, while full modality ablations would
require many additional API days under Tier-1 rate limits. Archived Pro outputs
under results/ remain on disk but are excluded from analysis by default.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Models evaluated on video-only for both datasets.
VIDEO_ONLY_MODELS = [
    "gemini-3-flash",
    "gpt-5",
    "gpt-5-mini",
    "claude-opus-4-5",
]

# Gemini model used for audio-capable modality ablations (EU + Mindreading).
GEMINI_AUDIO_MODEL = "gemini-3-flash"

# Summaries/CSVs for these models are skipped in analysis scripts.
EXCLUDED_MODELS = frozenset({"gemini-3-pro"})

# Repository root (parent of analysis/).
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Legacy runs from earlier evaluation pipelines (optional import).
# Imported to results/legacy_import/ for cross-model video-only tables.
LEGACY_RESULT_SOURCES: List[Dict[str, str]] = [
    {
        "legacy_dir": "results/eu_emotion_gpt5",
        "model": "gpt-5",
        "dataset": "eu_emotion",
        "condition": "video_only",
    },
    {
        "legacy_dir": "results/eu_emotion_gpt5_mini",
        "model": "gpt-5-mini",
        "dataset": "eu_emotion",
        "condition": "video_only",
    },
    {
        "legacy_dir": "results/eu_emotion_opus_4_5",
        "model": "claude-opus-4-5",
        "dataset": "eu_emotion",
        "condition": "video_only",
    },
    {
        "legacy_dir": "results/mindreading_gpt5",
        "model": "gpt-5",
        "dataset": "mindreading",
        "condition": "video_only",
    },
    {
        "legacy_dir": "results/mindreading_gpt5_mini_video_only",
        "model": "gpt-5-mini",
        "dataset": "mindreading",
        "condition": "video_only",
    },
    {
        "legacy_dir": "results/mindreading_opus_4_5",
        "model": "claude-opus-4-5",
        "dataset": "mindreading",
        "condition": "video_only",
    },
    {
        "legacy_dir": "results/mindreading_gemini3_flash_video_only",
        "model": "gemini-3-flash",
        "dataset": "mindreading",
        "condition": "video_only",
    },
]

# For five-model video-only MR comparisons, prefer full_run reruns (consistent trial coverage).
VIDEO_ONLY_LEGACY_KEYS: frozenset[ResultKey] = frozenset()

ResultKey = Tuple[str, str, str]  # (model, dataset, condition)
