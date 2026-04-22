from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from prompts import get_prompt

from models.llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)


MODEL_CHOICES = ["gemini-3-pro", "gemini-3-flash", "gpt-5", "gpt-5-mini", "claude-opus-4-5"]
DATASET_CHOICES = ["eu_emotion", "mindreading"]


def _resolve_path(path_str: Optional[str], data_dir: Path) -> Optional[str]:
    if path_str is None:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((data_dir / p).resolve())


def _load_trials(trials_file: str) -> List[Dict[str, Any]]:
    p = Path(trials_file)
    if not p.exists():
        raise FileNotFoundError(f"trials_file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("trials_file JSON must contain a list of trial objects")
    return data


def run_evaluation(
    *,
    model_name: str,
    dataset_name: str,
    trials_file: str,
    data_dir: str,
    cache_dir: str,
    results_dir: str,
    seed: int,
) -> Tuple[Path, Path, Dict[str, Any]]:
    trials = _load_trials(trials_file)
    base_data_dir = Path(data_dir).resolve()

    prompt_template = get_prompt(model_name)
    wrapper = LLMWrapper(model_name=model_name, cache_dir=cache_dir, prompt_template=prompt_template)

    rows: List[Dict[str, Any]] = []
    n_total = len(trials)
    n_unparseable = 0

    for trial in tqdm(trials, desc=f"Evaluating {model_name} on {dataset_name}", unit="trial"):
        try:
            trial_id = trial.get("trial_id")
            correct_label = trial["correct_label"]
            candidate_labels = trial["candidate_labels"]
            if not isinstance(candidate_labels, list) or len(candidate_labels) != 4:
                raise ValueError(f"candidate_labels must be a list of 4 labels (trial_id={trial_id})")

            video_path = _resolve_path(trial["video_path"], base_data_dir)
            audio_path_raw = trial.get("audio_path")
            audio_path = _resolve_path(audio_path_raw, base_data_dir)

            if not model_name.startswith("gemini"):
                audio_path = None

            out = wrapper.classify(video_path=video_path, audio_path=audio_path, candidate_labels=candidate_labels)
            predicted_label = out.get("predicted_label")
            raw_response = out.get("raw_response", "")
            cached = bool(out.get("cached", False))

            is_correct: Optional[bool]
            if predicted_label is None:
                is_correct = None
                n_unparseable += 1
            else:
                is_correct = bool(predicted_label == correct_label)

            rows.append(
                {
                    "trial_id": trial_id,
                    "dataset": dataset_name,
                    "model": model_name,
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "correct_label": correct_label,
                    "predicted_label": predicted_label,
                    "is_correct": is_correct,
                    "cached": cached,
                    "raw_response": raw_response,
                }
            )
        except Exception:
            logger.exception("Failed processing trial: %s", trial.get("trial_id"))
            n_unparseable += 1
            rows.append(
                {
                    "trial_id": trial.get("trial_id"),
                    "dataset": dataset_name,
                    "model": model_name,
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
        "seed": seed,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_unparseable": int(n_total - n_valid),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
    }

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / f"{model_name}_{dataset_name}_results.csv"
    summary_json = out_dir / f"{model_name}_{dataset_name}_summary.json"

    df.to_csv(results_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return results_csv, summary_json, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM mental state recognition evaluation.")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES, help="Model name.")
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES, help="Dataset name.")
    parser.add_argument("--trials_file", required=True, help="Path to JSON file of pre-generated trials.")
    parser.add_argument("--data_dir", required=True, help="Path to directory containing video/audio files.")
    parser.add_argument("--cache_dir", default="cache/", help="Path to cache directory.")
    parser.add_argument("--results_dir", default="results/", help="Path to results directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args()

    results_csv, summary_json, summary = run_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        trials_file=args.trials_file,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        seed=args.seed,
    )

    logger.info("Saved results CSV to %s", str(results_csv))
    logger.info("Saved summary JSON to %s", str(summary_json))
    print(
        f"Completed evaluation: model={summary['model']} dataset={summary['dataset']} "
        f"n_total={summary['n_total']} n_valid={summary['n_valid']} "
        f"n_unparseable={summary['n_unparseable']} accuracy={summary['accuracy_percent']:.2f}%"
    )


if __name__ == "__main__":
    main()
