from __future__ import annotations

"""
Basic vs complex mental-state classification for per-state analyses.

Basic emotions follow six Ekman-style categories: happiness, sadness, fear,
anger, disgust, and surprise. EU-Emotions uses base labels plus low-intensity
variants. Mindreading uses exact label matches only (no synonym expansion).
"""

from typing import FrozenSet, Set

# EU-Emotions: base + low-intensity variants (case-insensitive match at runtime).
EU_BASIC_LABELS: FrozenSet[str] = frozenset(
    {
        "happy",
        "happy low intensity",
        "sad",
        "sad low intensity",
        "afraid",
        "afraid low intensity",
        "angry",
        "angry low intensity",
        "disgusted",
        "disgusted low intensity",
        "surprised",
        "surprised low intensity",
    }
)

# Mindreading: exact canonical words only — no synonyms (e.g. not fearful, terrified).
MR_BASIC_LABELS: FrozenSet[str] = frozenset(
    {
        "happy",
        "sad",
        "afraid",
        "angry",
        "disgusted",
        "surprised",
    }
)


def _norm(label: str) -> str:
    return (label or "").strip().casefold()


def is_eu_basic(label: str) -> bool:
    return _norm(label) in EU_BASIC_LABELS


def is_eu_complex(label: str) -> bool:
    n = _norm(label)
    return n != "neutral" and n not in EU_BASIC_LABELS


def is_mr_basic(label: str) -> bool:
    return _norm(label) in MR_BASIC_LABELS


def is_mr_complex(label: str) -> bool:
    return not is_mr_basic(label)
