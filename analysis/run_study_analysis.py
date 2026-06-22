from __future__ import annotations

"""
Run publication analyses on completed evaluation outputs.

Loads summary JSON + per-trial CSVs from results/full_run (excludes gemini-3-pro
by default). Writes:
  - analysis_outputs/statistical_analysis.json
  - analysis_outputs/per_emotion_breakdown.csv

Usage (from repository root):
  python analysis/run_study_analysis.py --results-dir results/full_run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.basic_complex_analysis import basic_complex_summary
from analysis.load_results import load_results_for_analysis, summarize_loaded_results
from analysis.per_emotion_analysis import generate_report_from_dataframe, intensity_summary, compute_per_emotion_accuracy, remove_neutral
from analysis.statistical_analysis import run_all_analyses
from analysis.study_config import EXCLUDED_MODELS

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run statistical and per-emotion analyses on study results.")
    parser.add_argument(
        "--results-dir",
        default=str(_REPO_ROOT / "results" / "full_run"),
        help="Directory with *_summary.json and *_results.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "analysis_outputs"),
        help="Directory for analysis artifacts.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, results_df, analysis_meta = load_results_for_analysis(results_dir)
    print("\n=== Loaded summaries (gemini-3-pro excluded) ===")
    print(summarize_loaded_results(results))
    print(f"\n=== {analysis_meta['mr_video_only_fair_subset']['note']} ===")

    stats_out = run_all_analyses(results)
    stats_out["analysis_metadata"] = analysis_meta
    stats_path = output_dir / "statistical_analysis.json"

    def _json_default(obj: object) -> object:
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    stats_path.write_text(json.dumps(stats_out, indent=2, default=_json_default), encoding="utf-8")
    logger.info("Wrote %s", stats_path)

    per_emotion_path = output_dir / "per_emotion_breakdown.csv"
    generate_report_from_dataframe(results_df, str(per_emotion_path))

    results_df_neutral = remove_neutral(results_df)
    if "mental_state" not in results_df_neutral.columns and "correct_label" in results_df_neutral.columns:
        results_df_neutral = results_df_neutral.rename(columns={"correct_label": "mental_state"})
    per_emotion = compute_per_emotion_accuracy(results_df_neutral)
    intensity_path = output_dir / "intensity_summary.csv"
    intens = intensity_summary(per_emotion)
    if not intens.empty:
        intens.to_csv(intensity_path, index=False)
        logger.info("Wrote %s", intensity_path)

    basic_complex_path = output_dir / "basic_complex_summary.csv"
    bc = basic_complex_summary(results_df_neutral)
    if not bc.empty:
        bc.to_csv(basic_complex_path, index=False)
        logger.info("Wrote %s", basic_complex_path)

    summary_md = _build_results_section_summary(stats_out, analysis_meta)
    summary_path = output_dir / "results_section_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    logger.info("Wrote %s", summary_path)

    print(f"\nAnalysis outputs written to: {output_dir}")
    print(f"Excluded models: {sorted(EXCLUDED_MODELS)}")


def _build_results_section_summary(stats_out: dict, analysis_meta: dict) -> str:
    """Markdown tables and narrative hooks for the manuscript Results section."""
    lines = [
        "# Results section — analysis summary",
        "",
        "Generated from `analysis/run_study_analysis.py`.",
        "",
        "## Mindreading video-only (fair cross-model subset)",
        "",
        analysis_meta.get("mr_video_only_fair_subset", {}).get("note", ""),
        "",
        "| Model | Accuracy | n | Wilson 95% CI |",
        "|-------|----------|---|---------------|",
    ]

    per_model = stats_out.get("per_model_dataset", {})
    for model in sorted(per_model):
        mr = per_model[model].get("mindreading", {}).get("video_only")
        if not mr:
            continue
        acc = mr["accuracy"] * 100
        n = mr["n_total"]
        lo, hi = mr["wilson_ci_95"]
        lines.append(f"| {model} | {acc:.2f}% | {n} | [{lo*100:.2f}%, {hi*100:.2f}%] |")

    lines.extend(["", "## EU-Emotions (all conditions)", "", "| Model | Condition | Accuracy | n |", "|-------|-----------|----------|---|"])
    for model in sorted(per_model):
        eu = per_model[model].get("eu_emotion", {})
        for cond in sorted(eu):
            row = eu[cond]
            lines.append(
                f"| {model} | {cond} | {row['accuracy']*100:.2f}% | {row['n_total']} |"
            )

    lines.extend(["", "## Gemini Flash modality ablations", ""])
    mod = stats_out.get("comparisons", {}).get("modality_ablations_gemini_flash", [])
    if mod:
        lines.append("| Dataset | A | B | Acc A | Acc B | z | p (raw) |")
        lines.append("|---------|---|---|-------|-------|---|---------|")
        for row in mod:
            lines.append(
                f"| {row['dataset']} | {row['condition_a']} | {row['condition_b']} | "
                f"{row['accuracy_a']*100:.2f}% | {row['accuracy_b']*100:.2f}% | "
                f"{row['z']:.2f} | {row['p_value_raw']:.4g} |"
            )

    lines.extend(["", "## vs human benchmark (EU)", ""])
    hb = stats_out.get("comparisons", {}).get("vs_human_benchmark", {})
    for model in sorted(hb):
        eu = hb[model].get("eu_emotion", {})
        for cond, comp in eu.items():
            if not comp:
                continue
            b = comp["benchmark"]
            lines.append(
                f"- **{model}** {cond}: z={comp['z']:.2f}, p={comp['p_value_raw']:.4g}, "
                f"human={b['accuracy']*100:.2f}% (n={b['n']})"
            )

    fisher = stats_out.get("comparisons", {}).get("pairwise_models_fisher", {})
    if fisher:
        lines.extend(["", "## Pairwise model comparisons (Fisher exact, Bonferroni)", ""])
        for key, rows in fisher.items():
            lines.append(f"### {key}")
            for row in rows:
                p = row.get("p_value_bonferroni", row.get("p_value_raw"))
                lines.append(
                    f"- {row['model_a']} vs {row['model_b']}: OR={row['odds_ratio']:.3f}, "
                    f"p_bonf={p:.4g}, h={row['cohen_h']:.2f}"
                )

    lines.extend([
        "",
        "## Manuscript reminders",
        "",
        "- Remove autistic / Golan benchmark comparisons.",
        "- EU human benchmarks: O'Reilly facial expression 63% (n=1231); Lassalle audio 45.19% (n=427); note 6-AFC vs 4-AFC and modality-match caveats.",
        "- Exclude Neutral from per-mental-state summaries.",
        "- Report MR audio/multimodal Flash results with spoken-label confound caveat.",
        "- Basic vs complex: EU = six categories + low-intensity variants vs other non-neutral states; MR = exact label match only (no synonyms).",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
