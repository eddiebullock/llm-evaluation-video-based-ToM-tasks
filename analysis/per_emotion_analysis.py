from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def compute_per_emotion_accuracy(results_df: pd.DataFrame) -> pd.DataFrame:
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


def compute_cross_model_summary(per_emotion_df: pd.DataFrame) -> pd.DataFrame:
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
    required = {"mental_state", "min_accuracy", "max_accuracy"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df missing required columns: {sorted(missing)}")

    perfect = summary_df[summary_df["min_accuracy"] >= threshold_perfect].copy()
    zero = summary_df[summary_df["max_accuracy"] <= threshold_zero].copy()
    return perfect.reset_index(drop=True), zero.reset_index(drop=True)


def _load_results_csvs(results_dir: str) -> pd.DataFrame:
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"results_dir does not exist: {results_path}")

    csvs = sorted(results_path.glob("*_results.csv"))
    if not csvs:
        raise FileNotFoundError(f"No '*_results.csv' files found in: {results_path}")

    frames: List[pd.DataFrame] = []
    for p in csvs:
        df = pd.read_csv(p)
        df["source_file"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def generate_report(results_dir: str, output_path: str) -> None:
    results_df = _load_results_csvs(results_dir)

    if "mental_state" not in results_df.columns:
        if "correct_label" in results_df.columns:
            results_df = results_df.rename(columns={"correct_label": "mental_state"})
        else:
            raise ValueError("Results CSVs must include 'mental_state' or 'correct_label' column.")

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        generate_report(results_dir="results/", output_path="results/per_emotion_breakdown.csv")
    except Exception as e:
        print("Demo failed (expected if results/ is empty):", str(e))