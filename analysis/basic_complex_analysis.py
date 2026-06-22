from __future__ import annotations

"""Compute basic vs complex accuracy summaries from per-trial results."""

from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .basic_complex_mapping import is_eu_basic, is_eu_complex, is_mr_basic, is_mr_complex

Counts = Tuple[int, int, float]  # n_correct, n_total, accuracy


def _accuracy_subset(df: pd.DataFrame, predicate) -> Counts:
    sub = df[df["is_correct"].notna()].copy()
    if sub.empty:
        return 0, 0, 0.0
    labels = sub["mental_state"] if "mental_state" in sub.columns else sub["correct_label"]
    mask = labels.astype(str).map(lambda s: predicate(s))
    sub = sub[mask]
    n_total = int(sub.shape[0])
    if n_total == 0:
        return 0, 0, 0.0
    n_correct = int(sub["is_correct"].astype(bool).sum())
    return n_correct, n_total, n_correct / n_total


def basic_complex_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-model basic vs complex accuracy for EU and Mindreading video_only rows.
    """
    rows: List[Dict[str, object]] = []
    vo = results_df[results_df["condition"] == "video_only"].copy()
    if "mental_state" not in vo.columns and "correct_label" in vo.columns:
        vo = vo.rename(columns={"correct_label": "mental_state"})

    for model in sorted(vo["model"].unique()):
        mdf = vo[vo["model"] == model]
        eu = mdf[mdf["dataset"] == "eu_emotion"]
        mr = mdf[mdf["dataset"] == "mindreading"]
        for dataset, sub, basic_fn, complex_fn in (
            ("eu_emotion", eu, is_eu_basic, is_eu_complex),
            ("mindreading", mr, is_mr_basic, is_mr_complex),
        ):
            if sub.empty:
                continue
            for category, fn in (("basic", basic_fn), ("complex", complex_fn)):
                nc, nt, acc = _accuracy_subset(sub, fn)
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "category": category,
                        "n_correct": nc,
                        "n_total": nt,
                        "accuracy": acc,
                        "accuracy_percent": acc * 100.0,
                    }
                )
    return pd.DataFrame(rows)
