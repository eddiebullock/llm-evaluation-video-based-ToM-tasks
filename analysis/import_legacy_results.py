from __future__ import annotations

"""
Import pre-publication_repo evaluation outputs into publication_repo format.

Reads legacy ``predictions.json`` + ``summary.json`` under mr_ts_play/results/
and writes ``{model}_{dataset}_{condition}_results.csv`` and ``*_summary.json``
to publication_repo/results/legacy_import/.

Usage (from publication_repo/):
  python analysis/import_legacy_results.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.study_config import LEGACY_RESULT_SOURCES, WORKSPACE_ROOT

logger = logging.getLogger(__name__)


def _rows_from_predictions(
    predictions: List[Dict[str, Any]],
    *,
    model: str,
    dataset: str,
    condition: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in predictions:
        predicted = item.get("predicted_label")
        rows.append(
            {
                "trial_id": item.get("trial_id"),
                "dataset": dataset,
                "model": model,
                "condition": condition,
                "video_path": item.get("video_path"),
                "audio_path": item.get("audio_path"),
                "correct_label": item.get("correct_label"),
                "predicted_label": predicted,
                "is_correct": item.get("is_correct") if predicted is not None else None,
                "cached": False,
                "raw_response": item.get("reasoning") or "",
                "source": "legacy_import",
            }
        )
    return rows


def import_legacy_run(
    *,
    legacy_dir: Path,
    model: str,
    dataset: str,
    condition: str,
    output_dir: Path,
    seed: int = 42,
) -> Path:
    predictions_path = legacy_dir / "predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions.json: {predictions_path}")

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError(f"Expected list in {predictions_path}")

    rows = _rows_from_predictions(predictions, model=model, dataset=dataset, condition=condition)
    df = pd.DataFrame(rows)
    n_valid = int(df["predicted_label"].notna().sum())
    n_total = len(df)
    if n_valid > 0:
        accuracy = float(df.loc[df["predicted_label"].notna(), "is_correct"].mean())
    else:
        accuracy = 0.0

    summary: Dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "condition": condition,
        "seed": seed,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_unparseable": int(n_total - n_valid),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "source": "legacy_import",
        "legacy_dir": str(legacy_dir),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model}_{dataset}_{condition}"
    csv_path = output_dir / f"{stem}_results.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    df.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Imported %s | %s | %s: %.2f%% (%d/%d) -> %s",
        model,
        dataset,
        condition,
        accuracy * 100,
        int(round(accuracy * n_valid)),
        n_valid,
        csv_path.name,
    )
    return summary_path


def import_all_legacy(output_dir: Optional[Path] = None) -> List[Path]:
    out = output_dir or (_REPO_ROOT / "results" / "legacy_import")
    written: List[Path] = []
    for entry in LEGACY_RESULT_SOURCES:
        legacy_dir = WORKSPACE_ROOT / entry["legacy_dir"]
        if not legacy_dir.exists():
            raise FileNotFoundError(f"Legacy results not found: {legacy_dir}")
        written.append(
            import_legacy_run(
                legacy_dir=legacy_dir,
                model=entry["model"],
                dataset=entry["dataset"],
                condition=entry["condition"],
                output_dir=out,
            )
        )
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    paths = import_all_legacy()
    print(f"Imported {len(paths)} legacy runs to publication_repo/results/legacy_import/")


if __name__ == "__main__":
    main()
