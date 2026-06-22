from __future__ import annotations

"""
Mindreading video-only fair subset for cross-model comparisons.

657 trials in the Mindreading battery point at T-marker audio-only .mov files.
Non-Gemini models skip these in video_only (no video_path). Gemini Flash previously
received text-only prompts on those trials. For comparable video-only results, restrict
MR video_only analysis to trials where a video file was actually evaluated
(non-empty video_path in the results CSV).

Cross-model tables use the intersection of valid, video-evaluated trial_ids across
all models in the comparison set.
"""

from typing import Iterable, List, Set, Tuple

import pandas as pd

from .study_config import VIDEO_ONLY_MODELS

Counts = Tuple[int, int]  # (n_correct, n_total)


def has_evaluated_video(row: pd.Series) -> bool:
    vp = row.get("video_path")
    if pd.isna(vp):
        return False
    return bool(str(vp).strip())


def filter_mr_video_only_fair(results_df: pd.DataFrame) -> pd.DataFrame:
    """Drop MR video_only rows where no video was sent to the model."""
    if "dataset" not in results_df.columns or "condition" not in results_df.columns:
        return results_df
    is_mr_vo = (results_df["dataset"] == "mindreading") & (results_df["condition"] == "video_only")
    fair_mr = results_df[is_mr_vo & results_df.apply(has_evaluated_video, axis=1)]
    return pd.concat([results_df[~is_mr_vo], fair_mr], ignore_index=True)


def fair_mr_video_trial_ids(
    results_df: pd.DataFrame,
    models: Iterable[str] | None = None,
) -> Set[str]:
    """
    Trial IDs valid for all requested models on MR video_only with evaluated video.
    """
    model_list = list(models if models is not None else VIDEO_ONLY_MODELS)
    mr = results_df[
        (results_df["dataset"] == "mindreading")
        & (results_df["condition"] == "video_only")
        & (results_df["model"].isin(model_list))
    ].copy()
    mr = mr[mr["is_correct"].notna() & mr.apply(has_evaluated_video, axis=1)]

    id_sets: List[Set[str]] = []
    for model in model_list:
        ids = set(mr.loc[mr["model"] == model, "trial_id"].astype(str))
        id_sets.append(ids)
    if not id_sets:
        return set()
    common = id_sets[0].copy()
    for s in id_sets[1:]:
        common &= s
    return common


def mr_video_only_counts_on_trials(
    results_df: pd.DataFrame,
    model: str,
    trial_ids: Set[str],
) -> Counts:
    sub = results_df[
        (results_df["model"] == model)
        & (results_df["dataset"] == "mindreading")
        & (results_df["condition"] == "video_only")
        & (results_df["trial_id"].astype(str).isin(trial_ids))
    ]
    valid = sub[sub["is_correct"].notna()]
    n_total = int(valid.shape[0])
    n_correct = int(valid["is_correct"].astype(bool).sum())
    return n_correct, n_total


def mr_fair_subset_note(trial_ids: Set[str]) -> str:
    return (
        "Mindreading video_only cross-model comparisons use the intersection of trials "
        f"where each model received a video stimulus (n={len(trial_ids)}). "
        "T-marker audio-only trials without video input are excluded."
    )
