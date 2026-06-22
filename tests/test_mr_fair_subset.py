import pandas as pd

from analysis.load_results import apply_mr_fair_video_subset
from analysis.mr_fair_subset import (
    fair_mr_video_trial_ids,
    filter_mr_video_only_fair,
    has_evaluated_video,
)


def _sample_df() -> pd.DataFrame:
    rows = []
    for model in ("gemini-3-flash", "gpt-5"):
        for tid, vp, correct in [
            ("t1", "/v/a.mov", True),
            ("t2", "/v/b.mov", False),
            ("t3", "", None),
        ]:
            rows.append(
                {
                    "trial_id": tid,
                    "dataset": "mindreading",
                    "condition": "video_only",
                    "model": model,
                    "video_path": vp,
                    "is_correct": correct if correct is not None else None,
                    "predicted_label": "x" if correct is not None else None,
                }
            )
    rows.append(
        {
            "trial_id": "eu1",
            "dataset": "eu_emotion",
            "condition": "video_only",
            "model": "gpt-5",
            "video_path": "/v/eu.mov",
            "is_correct": True,
            "predicted_label": "happy",
        }
    )
    return pd.DataFrame(rows)


def test_has_evaluated_video() -> None:
    assert has_evaluated_video(pd.Series({"video_path": "/x.mov"}))
    assert not has_evaluated_video(pd.Series({"video_path": ""}))
    assert not has_evaluated_video(pd.Series({"video_path": None}))


def test_fair_mr_video_intersection() -> None:
    df = _sample_df()
    ids = fair_mr_video_trial_ids(df, ["gemini-3-flash", "gpt-5"])
    assert ids == {"t1", "t2"}


def test_filter_mr_video_only_fair_drops_empty_video() -> None:
    df = _sample_df()
    out = filter_mr_video_only_fair(df)
    mr = out[(out["dataset"] == "mindreading") & (out["condition"] == "video_only")]
    assert set(mr["trial_id"]) == {"t1", "t2"}
    assert len(out[out["trial_id"] == "eu1"]) == 1


def test_apply_mr_fair_video_subset_overwrites_counts() -> None:
    df = _sample_df()
    results = {
        "gemini-3-flash": {"mindreading": {"video_only": (99, 100)}},
        "gpt-5": {"mindreading": {"video_only": (50, 100)}},
    }
    patched, ids, note = apply_mr_fair_video_subset(results, df, models=["gemini-3-flash", "gpt-5"])
    assert ids == {"t1", "t2"}
    assert patched["gpt-5"]["mindreading"]["video_only"] == (1, 2)
    assert patched["gemini-3-flash"]["mindreading"]["video_only"] == (1, 2)
    assert "n=2" in note
