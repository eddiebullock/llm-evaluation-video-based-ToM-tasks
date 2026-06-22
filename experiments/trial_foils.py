from __future__ import annotations

"""
Deterministic 4-AFC foil generation for trial JSONs that only specify correct_label.

Matches the legacy EU multimodal experiment: correct label + 3 random foils, shuffled
per trial using a stable hash of trial_id (not Python's salted hash()).
"""

import hashlib
import random
from pathlib import Path
from typing import List, Optional, Sequence


def _trial_rng(trial_id: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{trial_id}|{seed}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def load_eu_emotion_label_pool(labels_file: Path) -> List[str]:
    if not labels_file.is_file():
        raise FileNotFoundError(f"EU emotion label list not found: {labels_file}")
    labels: List[str] = []
    for line in labels_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            labels.append(s)
    if len(labels) < 4:
        raise ValueError(f"EU emotion label pool too small ({len(labels)} labels): {labels_file}")
    return labels


def build_emotion_pool_from_trials(trials: Sequence[dict]) -> List[str]:
    pool = set()
    for t in trials:
        for key in ("correct_label", "emotion"):
            val = t.get(key)
            if val:
                pool.add(str(val).strip())
    return sorted(pool, key=str.casefold)


def generate_candidate_labels(
    correct_label: str,
    emotion_pool: Sequence[str],
    *,
    trial_id: str,
    seed: int,
    n_options: int = 4,
) -> List[str]:
    """
    Return n_options labels: correct_label + (n_options-1) foils, shuffled deterministically.
    """
    if n_options < 2:
        raise ValueError("n_options must be >= 2")

    correct = correct_label.strip()
    correct_cf = correct.casefold()

    pool = sorted({str(e).strip() for e in emotion_pool if str(e).strip()}, key=str.casefold)
    if not any(e.casefold() == correct_cf for e in pool):
        pool = sorted(set(pool) | {correct}, key=str.casefold)

    others = [e for e in pool if e.casefold() != correct_cf]
    n_foils = n_options - 1
    if len(others) < n_foils:
        raise ValueError(
            f"Not enough foil labels for trial_id={trial_id!r} (need {n_foils}, have {len(others)})"
        )

    rng = _trial_rng(str(trial_id), seed)
    foils = rng.sample(others, n_foils)
    labels = [correct] + foils
    rng.shuffle(labels)
    return labels


def resolve_candidate_labels(
    trial: dict,
    emotion_pool: Sequence[str],
    *,
    seed: int,
    trial_index: Optional[int] = None,
) -> List[str]:
    """
    Use pre-specified candidate_labels when valid; otherwise generate foils.
    """
    existing = trial.get("candidate_labels")
    if isinstance(existing, list) and len(existing) == 4:
        return list(existing)

    correct_label = trial.get("correct_label") or trial.get("emotion")
    if not correct_label:
        raise ValueError(
            f"Trial missing correct_label/emotion and candidate_labels (trial_id={trial.get('trial_id')!r})"
        )

    trial_id = trial.get("trial_id")
    if trial_id is None:
        trial_id = f"trial_{trial_index if trial_index is not None else 0}"

    generated = generate_candidate_labels(
        str(correct_label),
        emotion_pool,
        trial_id=str(trial_id),
        seed=seed,
    )
    trial["candidate_labels"] = generated
    trial["candidate_labels_generated"] = True
    return generated
