from __future__ import annotations

import json
from pathlib import Path

from experiments.trial_foils import (
    generate_candidate_labels,
    load_eu_emotion_label_pool,
    resolve_candidate_labels,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EU_LABELS = _REPO_ROOT / "data" / "eu_emotion_states_list.txt"
_TEST_FINAL = _REPO_ROOT / "data" / "trial_definitions" / "eu_emotion_test_final.json"


def test_generate_candidate_labels_deterministic():
    pool = load_eu_emotion_label_pool(_EU_LABELS)
    a = generate_candidate_labels("joking", pool, trial_id="train_10", seed=42)
    b = generate_candidate_labels("joking", pool, trial_id="train_10", seed=42)
    assert a == b
    assert len(a) == 4
    assert any(x.casefold() == "joking" for x in a)


def test_resolve_skips_when_present():
    trial = {
        "trial_id": "t1",
        "correct_label": "sad",
        "candidate_labels": ["sad", "happy", "afraid", "bored"],
    }
    out = resolve_candidate_labels(trial, ["sad", "happy"], seed=42)
    assert out == ["sad", "happy", "afraid", "bored"]
    assert "candidate_labels_generated" not in trial


def test_resolve_generates_for_test_final_format():
    data = json.loads(_TEST_FINAL.read_text(encoding="utf-8"))
    trials = data["trials"]
    pool = load_eu_emotion_label_pool(_EU_LABELS)
    t0 = dict(trials[0])
    assert "candidate_labels" not in t0
    labels = resolve_candidate_labels(t0, pool, seed=42, trial_index=0)
    assert len(labels) == 4
    assert t0.get("candidate_labels_generated") is True
    correct = (t0.get("correct_label") or "").casefold()
    assert sum(1 for x in labels if x.casefold() == correct) == 1
