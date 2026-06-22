from __future__ import annotations

"""
Convenience runner for modality ablations.

This script shells out to `experiments/run_evaluation.py` so it works with the repo's
non-package `experiments/` directory layout.

EU-Emotions and Mindreading stimuli live under different roots; pass both `--data-dir-eu`
and `--data-dir-mr` (or the legacy single `--data-dir` only if both datasets share one root).

Examples:
  python run_ablation_suite.py \\
    --data-dir-eu "/path/to/EU_emotions" \\
    --data-dir-mr "/path/to/MindReading/Emotions" \\
    --trials-eu "../data/trial_definitions/eu_emotion_test_final.json" \\
    --trials-eu-audio "../data/trial_definitions/eu_emotion_test_final.json" \\
    --trials-mr "../data/trial_definitions/mindreading_emotions_test.json"
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _evaluation_cmd(
    *,
    runner: Path,
    model: str,
    dataset: str,
    condition: str,
    trials_file: str,
    data_dir: str,
    cache_dir: str,
    results_dir: str,
    seed: int,
) -> List[str]:
    return [
        sys.executable,
        str(runner),
        "--model",
        model,
        "--dataset",
        dataset,
        "--condition",
        condition,
        "--trials_file",
        trials_file,
        "--data_dir",
        data_dir,
        "--cache_dir",
        cache_dir,
        "--results_dir",
        results_dir,
        "--seed",
        str(seed),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Run modality ablations for EU-Emotions and Mindreading.")
    p.add_argument(
        "--data-dir",
        default=None,
        help="Legacy: single base directory for both datasets (only if both trial JSONs resolve under it).",
    )
    p.add_argument("--data-dir-eu", default=None, help="EU-Emotions root (contains emotions N/ and EU Emotion - UK Voices/).")
    p.add_argument("--data-dir-mr", default=None, help="Mindreading root (contains item folders, not .../Emotions/Audio/).")
    p.add_argument("--cache-dir", default="cache/", help="Cache directory.")
    p.add_argument("--results-dir", default="results/", help="Results directory.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    p.add_argument(
        "--trials-eu",
        required=True,
        help="EU trial JSON (118 face trials in eu_emotion_test_final.json; foils generated if missing).",
    )
    p.add_argument(
        "--trials-eu-audio",
        required=True,
        help="EU trial JSON for audio_only (same 118 file: UK Voices paired by label; foils generated if missing).",
    )
    p.add_argument("--trials-mr", required=True, help="Mindreading trial JSON (with candidate_labels).")

    args = p.parse_args()

    data_dir_eu: Optional[str] = args.data_dir_eu or args.data_dir
    data_dir_mr: Optional[str] = args.data_dir_mr or args.data_dir
    if not data_dir_eu or not data_dir_mr:
        p.error("Provide --data-dir-eu and --data-dir-mr (or legacy --data-dir for both).")

    repo_root = Path(__file__).resolve().parent
    runner = repo_root / "experiments" / "run_evaluation.py"
    if not runner.exists():
        raise FileNotFoundError(f"Missing runner: {runner}")

    # EU: video-only for all study models; audio/multimodal for Gemini 3 Flash only.
    from analysis.study_config import GEMINI_AUDIO_MODEL, VIDEO_ONLY_MODELS

    for m in VIDEO_ONLY_MODELS:
        _run(
            _evaluation_cmd(
                runner=runner,
                model=m,
                dataset="eu_emotion",
                condition="video_only",
                trials_file=args.trials_eu,
                data_dir=data_dir_eu,
                cache_dir=args.cache_dir,
                results_dir=args.results_dir,
                seed=args.seed,
            )
        )

    # EU multimodal: face trials + UK Voices pairing via eu_audio_resolver (emotion label).
    for m in [GEMINI_AUDIO_MODEL]:
        _run(
            _evaluation_cmd(
                runner=runner,
                model=m,
                dataset="eu_emotion",
                condition="multimodal",
                trials_file=args.trials_eu,
                data_dir=data_dir_eu,
                cache_dir=args.cache_dir,
                results_dir=args.results_dir,
                seed=args.seed,
            )
        )

    for m in [GEMINI_AUDIO_MODEL]:
        _run(
            _evaluation_cmd(
                runner=runner,
                model=m,
                dataset="eu_emotion",
                condition="audio_only",
                trials_file=args.trials_eu_audio,
                data_dir=data_dir_eu,
                cache_dir=args.cache_dir,
                results_dir=args.results_dir,
                seed=args.seed,
            )
        )

    # Mindreading: Gemini 3 Flash modality ablations; item-folder audio (not Emotions/Audio/).
    for cond in ["video_only", "audio_only", "multimodal"]:
        for m in [GEMINI_AUDIO_MODEL]:
            _run(
                _evaluation_cmd(
                    runner=runner,
                    model=m,
                    dataset="mindreading",
                    condition=cond,
                    trials_file=args.trials_mr,
                    data_dir=data_dir_mr,
                    cache_dir=args.cache_dir,
                    results_dir=args.results_dir,
                    seed=args.seed,
                )
            )


if __name__ == "__main__":
    main()
