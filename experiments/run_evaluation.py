from __future__ import annotations

"""
Main entrypoint for running the LLM mental state recognition evaluation.

This script loads pre-generated four-alternative forced-choice trials from JSON, runs
each trial through a specified LLM API wrapper, and writes:
- Full per-trial results to CSV
- Summary statistics to JSON
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure `publication_repo/` and bundled `.deps` are on sys.path before third-party imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_LOCAL_DEPS = _REPO_ROOT / ".deps"
if _LOCAL_DEPS.exists() and str(_LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DEPS))

import pandas as pd
from tqdm import tqdm

from prompts import get_prompt

from models.llm_wrapper import LLMWrapper
from eu_audio_resolver import build_audio_mapping_audit as build_eu_audio_audit
from eu_audio_resolver import resolve_eu_multimodal_audio, save_audio_mapping_audit as save_eu_audio_audit
from mindreading_audio_resolver import build_audio_mapping_audit as build_mr_audio_audit
from mindreading_audio_resolver import (
    extract_audio_from_video,
    has_audio_stream,
    resolve_item_folder_audio,
    resolve_mr_v_video_from_t_stimulus,
    save_audio_mapping_audit as save_mr_audio_audit,
)
from trial_foils import (
    build_emotion_pool_from_trials,
    load_eu_emotion_label_pool,
    resolve_candidate_labels,
)

logger = logging.getLogger(__name__)

_EU_EMOTION_LABELS_FILE = _REPO_ROOT.parent / "data" / "eu_emotion_states_list.txt"


MODEL_CHOICES = ["gemini-3-pro", "gemini-3-flash", "gpt-5", "gpt-5-mini", "claude-opus-4-5"]
DATASET_CHOICES = ["eu_emotion", "mindreading"]
CONDITION_CHOICES = ["video_only", "audio_only", "multimodal"]


def _resolve_path(path_str: Optional[str], data_dir: Path) -> Optional[str]:
    """
    Resolve trial paths relative to `data_dir` when they are not absolute.

    Args:
        path_str: Path string from trial JSON (may be None).
        data_dir: Base directory for relative trial paths.

    Returns:
        Absolute path string, or None if input is None.
    """
    if path_str is None:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((data_dir / p).resolve())


def _load_trials(trials_file: str) -> List[Dict[str, Any]]:
    """
    Load trial definitions from a JSON file.

    Args:
        trials_file: Path to JSON file containing a list of trial dicts.

    Returns:
        List of trial dicts.
    """
    p = Path(trials_file)
    if not p.exists():
        raise FileNotFoundError(f"trials_file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "trials" in data and isinstance(data["trials"], list):
        return data["trials"]
    if isinstance(data, list):
        return data
    raise ValueError("trials_file JSON must contain a list of trials or an object with a 'trials' list")


def _is_audio_path(path_str: str) -> bool:
    ext = Path(path_str).suffix.lower()
    if ext in {".wav", ".mp3", ".m4a", ".ogg", ".aif", ".aiff"}:
        return True
    # Mindreading sometimes stores audio as audio-only .mov with a 'T' marker in the filename.
    # Example: 0700803Z6Tcalming.mov
    if ext == ".mov":
        name = Path(path_str).name
        # Heuristic: contains 'T' marker and does not contain 'V' marker.
        if "T" in name and "V" not in name:
            return True
    return False


def _is_video_path(path_str: str) -> bool:
    ext = Path(path_str).suffix.lower()
    if ext in {".mp4", ".m4v", ".avi", ".webm"}:
        return True
    if ext == ".mov":
        name = Path(path_str).name
        # Mindreading video clips typically include a 'V' marker.
        if "V" in name:
            return True
    return False


def run_evaluation(
    *,
    model_name: str,
    dataset_name: str,
    condition: str,
    trials_file: str,
    data_dir: str,
    cache_dir: str,
    results_dir: str,
    seed: int,
    max_trials: Optional[int] = None,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Run evaluation for a single model and dataset.

    Args:
        model_name: Model identifier (must be one of MODEL_CHOICES).
        dataset_name: Dataset identifier (must be one of DATASET_CHOICES).
        trials_file: JSON path containing trial definitions.
        data_dir: Base directory for resolving relative video/audio paths.
        cache_dir: Cache directory for API responses.
        results_dir: Directory to write results.
        seed: Random seed (passed through to the wrapper cache key via prompt, and for reproducibility metadata).

    Returns:
        (results_csv_path, summary_json_path, summary_dict)
    """
    trials = _load_trials(trials_file)
    if max_trials is not None:
        if max_trials <= 0:
            raise ValueError("max_trials must be > 0")
        trials = trials[: int(max_trials)]
    base_data_dir = Path(data_dir).resolve()

    if dataset_name == "eu_emotion":
        emotion_pool = load_eu_emotion_label_pool(_EU_EMOTION_LABELS_FILE)
    else:
        emotion_pool = build_emotion_pool_from_trials(trials)

    prompt_template = get_prompt(model_name, condition=condition)
    wrapper = LLMWrapper(model_name=model_name, cache_dir=cache_dir, prompt_template=prompt_template)

    rows: List[Dict[str, Any]] = []
    n_total = len(trials)
    n_unparseable = 0

    # Optional: create an audit of audio resolution.
    if condition in {"audio_only", "multimodal"} and model_name.startswith("gemini"):
        try:
            audit_path = Path(results_dir) / f"{model_name}_{dataset_name}_{condition}_audio_mapping_audit.json"
            if dataset_name == "mindreading":
                audit = build_mr_audio_audit(trials, base_data_dir=base_data_dir, max_audit_rows=None)
                save_mr_audio_audit(audit, audit_path)
            elif dataset_name == "eu_emotion":
                audit = build_eu_audio_audit(
                    trials, base_data_dir=base_data_dir, seed=seed, max_audit_rows=None
                )
                save_eu_audio_audit(audit, audit_path)
            else:
                audit = []
            if audit:
                logger.info("Saved audio mapping audit to %s", str(audit_path))
        except Exception:
            logger.exception("Failed to build/save audio mapping audit (continuing)")

    for trial_idx, trial in enumerate(
        tqdm(trials, desc=f"Evaluating {model_name} on {dataset_name}", unit="trial")
    ):
        try:
            trial_id = trial.get("trial_id")
            correct_label = trial.get("correct_label") or trial.get("emotion")
            if not correct_label:
                raise ValueError(f"Missing correct_label/emotion (trial_id={trial_id})")
            candidate_labels = resolve_candidate_labels(
                trial, emotion_pool, seed=seed, trial_index=trial_idx
            )

            # Trials can provide either 'video_path' or 'stimulus_path'. Many existing trial JSONs
            # in this repo use 'stimulus_path'.
            stimulus_path = trial.get("video_path") or trial.get("stimulus_path")
            if stimulus_path is None:
                raise ValueError(f"Missing 'video_path'/'stimulus_path' for trial_id={trial_id}")

            resolved_stimulus = _resolve_path(str(stimulus_path), base_data_dir)

            # Determine condition inputs (video/audio paths may be None depending on condition).
            video_path: Optional[str] = None
            audio_path: Optional[str] = None

            # If the stimulus itself is audio (e.g., EU voice trials), treat it as audio.
            if _is_audio_path(resolved_stimulus):
                audio_path = resolved_stimulus
            elif _is_video_path(resolved_stimulus):
                video_path = resolved_stimulus
            else:
                # Default to video to preserve existing behavior; the wrapper will fail if it can't open.
                video_path = resolved_stimulus

            # Bring in explicit audio_path if present in trial JSON (rare for these trial defs).
            audio_path_raw = trial.get("audio_path")
            if audio_path_raw:
                audio_path = _resolve_path(audio_path_raw, base_data_dir)

            # Keep a copy of the resolved video path for audio resolution, even if the evaluation
            # condition later sets `video_path=None` (e.g., audio_only).
            video_path_for_audio_resolution = video_path

            # Apply condition gates.
            if condition == "video_only":
                audio_path = None
            elif condition == "audio_only":
                video_path = None
                # if a trial is video-based and audio is missing, we try to resolve Mindreading item-folder audio below.
            elif condition == "multimodal":
                # Keep both if available.
                pass
            else:
                raise ValueError(f"Unknown condition: {condition}")

            # Mindreading T-marker stimuli point at audio-only .mov files. For video_only and
            # multimodal, resolve the paired V video clip so all models see the same face video.
            if (
                dataset_name == "mindreading"
                and condition in {"video_only", "multimodal"}
                and video_path is None
                and _is_audio_path(resolved_stimulus)
            ):
                paired_v = resolve_mr_v_video_from_t_stimulus(Path(resolved_stimulus))
                if paired_v is not None:
                    video_path = str(paired_v)
                    trial["video_resolution_rule"] = "t_stimulus_paired_v"
                elif condition == "video_only":
                    logger.warning(
                        "No paired V video for T stimulus %s (trial_id=%s); skipping API call",
                        resolved_stimulus,
                        trial_id,
                    )

            # Only Gemini models are evaluated with audio input in this pipeline.
            if not model_name.startswith("gemini"):
                audio_path = None

            # Mindreading audio resolution (item-folder audio, not leakage directory).
            if dataset_name == "mindreading" and model_name.startswith("gemini") and condition in {"audio_only", "multimodal"}:
                # If audio isn't already specified, resolve from the same item folder as the video.
                if audio_path is None:
                    if video_path_for_audio_resolution is None:
                        # We cannot resolve item-folder audio without a video path to locate the folder.
                        raise ValueError(f"Mindreading {condition} requires a video path to resolve item-folder audio (trial_id={trial_id})")
                    ap, rule = resolve_item_folder_audio(Path(video_path_for_audio_resolution))
                    audio_path = str(ap) if ap else None
                    trial["audio_resolution_rule"] = rule

                # Gemini expects wav/mp3 — never send raw .mov (API rejects application/octet-stream).
                if audio_path is not None:
                    apath = Path(audio_path)
                    if apath.suffix.lower() == ".mov":
                        try:
                            extracted = extract_audio_from_video(apath)
                            if extracted is not None and extracted.exists():
                                audio_path = str(extracted)
                                trial["audio_resolution_rule"] = (
                                    str(trial.get("audio_resolution_rule", "")) + "+extracted_wav"
                                ).strip("+")
                            else:
                                logger.warning(
                                    "Could not extract wav from %s (trial_id=%s); skipping API call",
                                    audio_path,
                                    trial_id,
                                )
                                audio_path = None
                        except Exception:
                            logger.exception("Failed to extract wav from %s", audio_path)
                            audio_path = None

            # EU audio/multimodal: pair UK Voices (or sidecar) by emotion label.
            # audio_only clears video_path above; resolve from video_path_for_audio_resolution.
            if dataset_name == "eu_emotion" and model_name.startswith("gemini") and condition in {
                "audio_only",
                "multimodal",
            }:
                eu_video_for_audio = video_path if condition == "multimodal" else video_path_for_audio_resolution
                if audio_path is None and eu_video_for_audio is not None:
                    ap, rule = resolve_eu_multimodal_audio(
                        Path(eu_video_for_audio),
                        emotion_label=correct_label,
                        base_data_dir=base_data_dir,
                        trial_id=str(trial_id),
                        seed=seed,
                    )
                    audio_path = str(ap) if ap else None
                    trial["audio_resolution_rule"] = rule

            # Hard guard against leakage directory (even if a trial mistakenly supplies it).
            if audio_path is not None and "/Emotions/Audio/" in audio_path.replace("\\", "/"):
                raise ValueError(f"Refusing to use leakage audio path: {audio_path}")

            out = wrapper.classify(video_path=video_path, audio_path=audio_path, candidate_labels=candidate_labels)
            predicted_label = out.get("predicted_label")
            raw_response = out.get("raw_response", "")
            cached = bool(out.get("cached", False))

            is_correct: Optional[bool]
            if predicted_label is None:
                is_correct = None
                n_unparseable += 1
            else:
                is_correct = bool(str(predicted_label).casefold() == str(correct_label).casefold())

            rows.append(
                {
                    "trial_id": trial_id,
                    "dataset": dataset_name,
                    "model": model_name,
                    "condition": condition,
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "correct_label": correct_label,
                    "predicted_label": predicted_label,
                    "is_correct": is_correct,
                    "cached": cached,
                    "raw_response": raw_response,
                }
            )
        except Exception as e:
            if "daily request quota exceeded" in str(e).casefold():
                logger.error(
                    "Stopping evaluation: Gemini daily quota exceeded (trial_id=%s). "
                    "Rerun the same command after RPD resets; cached trials will be skipped.",
                    trial.get("trial_id"),
                )
                raise
            logger.exception("Failed processing trial: %s", trial.get("trial_id"))
            n_unparseable += 1
            rows.append(
                {
                    "trial_id": trial.get("trial_id"),
                    "dataset": dataset_name,
                    "model": model_name,
                    "condition": condition,
                    "video_path": trial.get("video_path"),
                    "audio_path": trial.get("audio_path"),
                    "correct_label": trial.get("correct_label"),
                    "predicted_label": None,
                    "is_correct": None,
                    "cached": False,
                    "raw_response": "",
                }
            )

    df = pd.DataFrame(rows)
    n_valid = int(df["predicted_label"].notna().sum()) if "predicted_label" in df.columns else 0

    if n_valid > 0:
        accuracy = float(df.loc[df["predicted_label"].notna(), "is_correct"].mean())
    else:
        accuracy = 0.0

    summary: Dict[str, Any] = {
        "model": model_name,
        "dataset": dataset_name,
        "condition": condition,
        "seed": seed,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_unparseable": int(n_total - n_valid),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
    }

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / f"{model_name}_{dataset_name}_{condition}_results.csv"
    summary_json = out_dir / f"{model_name}_{dataset_name}_{condition}_summary.json"

    df.to_csv(results_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return results_csv, summary_json, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM mental state recognition evaluation.")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES, help="Model name.")
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES, help="Dataset name.")
    parser.add_argument("--condition", required=True, choices=CONDITION_CHOICES, help="Input condition.")
    parser.add_argument("--trials_file", required=True, help="Path to JSON file of pre-generated trials.")
    parser.add_argument("--data_dir", required=True, help="Path to directory containing video/audio files.")
    parser.add_argument("--cache_dir", default="cache/", help="Path to cache directory.")
    parser.add_argument("--results_dir", default="results/", help="Path to results directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max_trials",
        type=int,
        default=None,
        help="Optional cap on number of trials (first N in JSON). Useful for smoke checks.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args()

    results_csv, summary_json, summary = run_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        condition=args.condition,
        trials_file=args.trials_file,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        seed=args.seed,
        max_trials=args.max_trials,
    )

    logger.info("Saved results CSV to %s", str(results_csv))
    logger.info("Saved summary JSON to %s", str(summary_json))
    print(
        f"Completed evaluation: model={summary['model']} dataset={summary['dataset']} "
        f"condition={summary['condition']} "
        f"n_total={summary['n_total']} n_valid={summary['n_valid']} "
        f"n_unparseable={summary['n_unparseable']} accuracy={summary['accuracy_percent']:.2f}%"
    )


if __name__ == "__main__":
    main()

