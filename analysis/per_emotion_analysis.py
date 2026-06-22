from __future__ import annotations

"""
Per-mental-state (per-emotion) accuracy analysis.

This script computes accuracy for each mental state for each model, and provides
cross-model summaries (mean/SD/min/max) to identify hardest/easiest states.
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

LOW_INTENSITY_TOKEN = "low intensity"


def compute_per_emotion_accuracy(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-mental-state accuracy for each model.

    Args:
        results_df: pandas DataFrame with columns:
            - model
            - mental_state
            - is_correct
            - predicted_label

        Rows with is_correct == None/NaN are treated as unparseable and excluded
        from accuracy computations.

    Returns:
        DataFrame with columns:
            - mental_state
            - model
            - n_trials
            - n_correct
            - accuracy
    """
    required = {"model", "mental_state", "is_correct", "predicted_label"}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df missing required columns: {sorted(missing)}")

    df = results_df.copy()
    df = df[df["is_correct"].notna()]
    df["is_correct"] = df["is_correct"].astype(bool)

    grouped = df.groupby(["mental_state", "model"], as_index=False)
    out = grouped.agg(
        n_trials=("is_correct", "size"),
        n_correct=("is_correct", "sum"),
    )
    out["accuracy"] = out["n_correct"] / out["n_trials"]
    return out[["mental_state", "model", "n_trials", "n_correct", "accuracy"]]


def _normalize_state(s: str) -> str:
    return (s or "").strip().casefold()


def remove_neutral(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove 'neutral' rows from a results dataframe (case-insensitive).
    """
    if "mental_state" in results_df.columns:
        col = "mental_state"
    elif "correct_label" in results_df.columns:
        col = "correct_label"
    else:
        return results_df
    return results_df[results_df[col].astype(str).map(_normalize_state) != "neutral"].copy()


def intensity_summary(per_emotion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise accuracy for high vs low intensity variants (EU-Emotions).

    Treat labels containing 'low intensity' as low-intensity. Base label is the label
    with the 'low intensity' suffix removed.

    Returns a per-model summary with:
      - n_pairs
      - mean_high_accuracy
      - mean_low_accuracy
      - mean_difference (high - low)
    """
    req = {"mental_state", "model", "accuracy"}
    missing = req - set(per_emotion_df.columns)
    if missing:
        raise ValueError(f"per_emotion_df missing required columns: {sorted(missing)}")

    df = per_emotion_df.copy()
    df["ms_cf"] = df["mental_state"].astype(str).map(_normalize_state)
    df["is_low"] = df["ms_cf"].str.contains(LOW_INTENSITY_TOKEN, regex=False)
    df["base"] = df["ms_cf"].str.replace(f" {LOW_INTENSITY_TOKEN}", "", regex=False).str.strip()

    high = df[~df["is_low"]][["model", "base", "accuracy"]].rename(columns={"accuracy": "high_accuracy"})
    low = df[df["is_low"]][["model", "base", "accuracy"]].rename(columns={"accuracy": "low_accuracy"})
    merged = high.merge(low, on=["model", "base"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["model", "n_pairs", "mean_high_accuracy", "mean_low_accuracy", "mean_difference"])

    merged["difference"] = merged["high_accuracy"] - merged["low_accuracy"]
    out = (
        merged.groupby("model", as_index=False)
        .agg(
            n_pairs=("base", "size"),
            mean_high_accuracy=("high_accuracy", "mean"),
            mean_low_accuracy=("low_accuracy", "mean"),
            mean_difference=("difference", "mean"),
        )
        .sort_values("mean_difference", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    return out


def compute_cross_model_summary(per_emotion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-model summary statistics per mental state.

    Args:
        per_emotion_df: DataFrame from `compute_per_emotion_accuracy`, with columns:
            - mental_state
            - model
            - accuracy

    Returns:
        Summary DataFrame with columns:
            - mental_state
            - mean_accuracy
            - sd_accuracy
            - min_accuracy
            - max_accuracy

        Sorted by mean_accuracy ascending (hardest first).
    """
    required = {"mental_state", "model", "accuracy"}
    missing = required - set(per_emotion_df.columns)
    if missing:
        raise ValueError(f"per_emotion_df missing required columns: {sorted(missing)}")

    g = per_emotion_df.groupby("mental_state", as_index=False)["accuracy"]
    summary = g.agg(
        mean_accuracy="mean",
        sd_accuracy="std",
        min_accuracy="min",
        max_accuracy="max",
    )
    summary["sd_accuracy"] = summary["sd_accuracy"].fillna(0.0)
    summary = summary.sort_values("mean_accuracy", ascending=True, kind="mergesort").reset_index(drop=True)
    return summary


def identify_perfect_and_zero(
    summary_df: pd.DataFrame, threshold_perfect: float = 1.0, threshold_zero: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify mental states that are perfect (all models at 100%) or universally failed (all models at 0%).

    Args:
        summary_df: Output of `compute_cross_model_summary`.
        threshold_perfect: Threshold for perfect recognition (default 1.0).
        threshold_zero: Threshold for universal failure (default 0.0).

    Returns:
        (perfect_df, zero_df) DataFrames, each a subset of summary_df.
    """
    required = {"mental_state", "min_accuracy", "max_accuracy"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df missing required columns: {sorted(missing)}")

    perfect = summary_df[summary_df["min_accuracy"] >= threshold_perfect].copy()
    zero = summary_df[summary_df["max_accuracy"] <= threshold_zero].copy()
    return perfect.reset_index(drop=True), zero.reset_index(drop=True)


def _load_results_csvs(results_dir: str, exclude_models: Optional[Iterable[str]] = None) -> pd.DataFrame:
    from .load_results import load_results_csvs

    return load_results_csvs(results_dir, exclude_models=exclude_models)


def generate_report_from_dataframe(results_df: pd.DataFrame, output_path: str) -> None:
    """
    Generate per-mental-state report from a combined results DataFrame.
    """
    if "mental_state" not in results_df.columns:
        if "correct_label" in results_df.columns:
            results_df = results_df.rename(columns={"correct_label": "mental_state"})
        else:
            raise ValueError("Results must include 'mental_state' or 'correct_label' column.")

    results_df = remove_neutral(results_df)

    per_emotion = compute_per_emotion_accuracy(results_df)
    summary = compute_cross_model_summary(per_emotion)
    perfect_df, zero_df = identify_perfect_and_zero(summary)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_emotion.to_csv(out, index=False)

    print("\n=== Per-mental-state report ===")
    print(f"Perfectly recognised states (all models 100%): {len(perfect_df)}")
    print(f"Universally failed states (all models 0%):   {len(zero_df)}")

    top5 = summary.sort_values("mean_accuracy", ascending=False).head(5)
    bottom5 = summary.sort_values("mean_accuracy", ascending=True).head(5)

    print("\nTop 5 mental states by mean accuracy:")
    for _, r in top5.iterrows():
        print(f"- {r['mental_state']}: {r['mean_accuracy']*100:.2f}% (sd={r['sd_accuracy']*100:.2f}%)")

    print("\nBottom 5 mental states by mean accuracy:")
    for _, r in bottom5.iterrows():
        print(f"- {r['mental_state']}: {r['mean_accuracy']*100:.2f}% (sd={r['sd_accuracy']*100:.2f}%)")

    try:
        intens = intensity_summary(per_emotion)
        if not intens.empty:
            print("\n=== High vs low intensity summary (by model; neutral excluded) ===")
            for _, r in intens.iterrows():
                print(
                    f"- {r['model']}: pairs={int(r['n_pairs'])} "
                    f"high={r['mean_high_accuracy']*100:.2f}% "
                    f"low={r['mean_low_accuracy']*100:.2f}% "
                    f"diff={r['mean_difference']*100:.2f}pp"
                )
    except Exception as e:
        logger.warning("Failed intensity summary: %s", str(e))


def generate_report(results_dir: str, output_path: str) -> None:
    """
    Generate a per-emotion analysis report from model result CSVs.

    This function:
    - Loads all `*_results.csv` files from results_dir
    - Combines them into a single DataFrame
    - Computes per-mental-state accuracy per model
    - Computes cross-model summary stats
    - Identifies perfectly recognized and universally failed mental states
    - Saves the full per-emotion breakdown to output_path as CSV
    - Prints paper-style summary statistics:
        - number of perfectly recognised states
        - number of universally failed states
        - top 5 and bottom 5 mental states by mean accuracy

    Args:
        results_dir: Directory containing per-model result CSVs.
        output_path: File path to save per-emotion breakdown CSV.
    """
    results_df = _load_results_csvs(results_dir)
    generate_report_from_dataframe(results_df, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Example:
    # python analysis/per_emotion_analysis.py
    try:
        generate_report(results_dir="results/", output_path="results/per_emotion_breakdown.csv")
    except Exception as e:
        print("Demo failed (expected if results/ is empty):", str(e))

