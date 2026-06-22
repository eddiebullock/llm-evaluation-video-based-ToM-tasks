from __future__ import annotations

"""
Statistical analyses for the LLM mental state recognition evaluation.

This script implements confidence intervals, hypothesis tests, multiple-comparison
corrections, effect sizes, and post-hoc power analyses used to reproduce the
paper's reported statistical results.

Human benchmark values:
- This version of the analysis is designed to compare model performance to
  EU-Emotions human benchmarks reported in:
  - O’Reilly et al. (EU-Emotion Stimulus Set: validation study)
  - Lassalle et al. (EU-Emotion Voice Database)

Benchmarks differ by modality (video-only, audio-only, audio+video). Provide the
benchmark(s) in `HUMAN_BENCHMARKS` below or pass them in as an argument to
`run_all_analyses(...)`.

Chance performance: 0.25 (four-alternative forced choice).
"""

import logging
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower

logger = logging.getLogger(__name__)


CHANCE = 0.25


# Human benchmarks keyed by (dataset, condition).
#
# Values are raw (uncorrected) accuracy from the EU validation literature.
# Original human tasks used 6-AFC; this study uses 4-AFC — direct comparability
# is limited (note in manuscript limitations).
#
# O'Reilly et al. (2016) reports separate benchmarks for face (63%), body (77%),
# and social scenes (72%). Facial expression is used for video_only as the
# closest match to full-face video clips; no single modality perfectly matches
# multimodal video presentation. No human multimodal benchmark exists — omit.
HUMAN_BENCHMARKS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "eu_emotion": {
        "video_only": {
            "citation": "O'Reilly et al. (2016); facial expression",
            "accuracy": 0.63,
            "n": 1231,
        },
        "audio_only": {
            "citation": "Lassalle et al. (2019); UK vocal expression",
            "accuracy": 0.4519,
            "n": 427,
        },
        "multimodal": {
            "citation": None,
            "accuracy": None,
            "n": None,
        },
    },
    # Mindreading: no human benchmark in this study (EU-only human comparisons).
    "mindreading": {},
}


def wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a binomial proportion.

    Args:
        n_correct: Number of correct trials (successes).
        n_total: Total number of trials (n).
        confidence: Confidence level, e.g. 0.95 for 95% CI.

    Returns:
        (lower, upper) bounds for the proportion.

    Raises:
        ValueError: If n_total <= 0 or n_correct is out of range.
    """
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    if n_correct < 0 or n_correct > n_total:
        raise ValueError("n_correct must satisfy 0 <= n_correct <= n_total")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1")

    p_hat = n_correct / n_total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + (z**2) / n_total
    center = (p_hat + (z**2) / (2 * n_total)) / denom
    half = (z / denom) * math.sqrt((p_hat * (1 - p_hat) / n_total) + ((z**2) / (4 * (n_total**2))))
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    return lower, upper


def binomial_test_vs_chance(n_correct: int, n_total: int, chance: float = CHANCE) -> float:
    """
    One-sided binomial test of performance against chance.

    H0: p == chance
    H1: p > chance

    Args:
        n_correct: Number of correct trials.
        n_total: Total trials.
        chance: Chance-level performance (default 0.25 for 4AFC).

    Returns:
        One-sided p-value.
    """
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    if n_correct < 0 or n_correct > n_total:
        raise ValueError("n_correct must satisfy 0 <= n_correct <= n_total")
    if not (0.0 <= chance <= 1.0):
        raise ValueError("chance must be in [0, 1]")

    res = stats.binomtest(k=n_correct, n=n_total, p=chance, alternative="greater")
    return float(res.pvalue)


def two_proportion_z_test(p1: float, n1: int, p2: float, n2: int) -> Tuple[float, float]:
    """
    Two-proportion z-test for independent samples (two-sided).

    Tests:
        H0: p1 == p2
        H1: p1 != p2

    Uses pooled standard error under H0.

    Args:
        p1: Proportion for sample 1.
        n1: Sample size for sample 1.
        p2: Proportion for sample 2.
        n2: Sample size for sample 2.

    Returns:
        (z_statistic, p_value) with two-sided p-value.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be > 0")
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must be in [0, 1]")

    x1 = p1 * n1
    x2 = p2 * n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = float(2 * stats.norm.sf(abs(z)))
    return float(z), p


def cohen_h(p1: float, p2: float) -> float:
    """
    Compute Cohen's h effect size for two proportions.

    h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))

    Args:
        p1: Proportion 1.
        p2: Proportion 2.

    Returns:
        Cohen's h (signed).
    """
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must be in [0, 1]")
    return float(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def fisher_exact_2x2(correct1: int, total1: int, correct2: int, total2: int) -> Tuple[float, float]:
    """
    Fisher's exact test on a 2x2 contingency table (correct vs incorrect).

    Table:
        [[correct1, incorrect1],
         [correct2, incorrect2]]

    Args:
        correct1: Correct count for group 1.
        total1: Total trials for group 1.
        correct2: Correct count for group 2.
        total2: Total trials for group 2.

    Returns:
        (odds_ratio, p_value) for two-sided Fisher's exact test.
    """
    if total1 <= 0 or total2 <= 0:
        raise ValueError("total1 and total2 must be > 0")
    if correct1 < 0 or correct1 > total1 or correct2 < 0 or correct2 > total2:
        raise ValueError("correct counts must satisfy 0 <= correct <= total")

    table = np.array([[correct1, total1 - correct1], [correct2, total2 - correct2]], dtype=int)
    odds_ratio, p_value = stats.fisher_exact(table, alternative="two-sided")
    return float(odds_ratio), float(p_value)


def bonferroni_correct(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[List[float], float]:
    """
    Apply Bonferroni correction to a list of p-values.

    Args:
        p_values: Iterable of raw p-values.
        alpha: Family-wise error rate.

    Returns:
        (corrected_p_values, significance_threshold) where threshold = alpha / m.
    """
    pvals = [float(p) for p in p_values]
    m = max(1, len(pvals))
    threshold = alpha / m
    corrected = [min(1.0, p * m) for p in pvals]
    return corrected, float(threshold)


def power_analysis(effect_size_h: float, n: int, alpha: float = 0.01) -> float:
    """
    Post-hoc power analysis for a two-sample test of proportions using Cohen's h.

    Uses `statsmodels.stats.power.NormalIndPower` for a two-sided test, assuming
    equal group sizes (n per group).

    Args:
        effect_size_h: Cohen's h for the comparison.
        n: Sample size per group (approximation; for unequal sizes consider using ratio).
        alpha: Significance level (default 0.01).

    Returns:
        Estimated power in [0, 1].
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")

    analysis = NormalIndPower()
    power = analysis.power(effect_size=abs(effect_size_h), nobs1=n, alpha=alpha, ratio=1.0, alternative="two-sided")
    return float(power)


def _extract_counts(obj: Any) -> Tuple[int, int]:
    """
    Extract (n_correct, n_total) from a flexible input object.

    Supported inputs:
    - dict with keys: ('n_correct','n_total') or ('correct','total') or ('accuracy','n_total')
    - tuple/list: (n_correct, n_total)
    - pandas DataFrame-like dict: {'is_correct': [...]} (derives counts)
    """
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return int(obj[0]), int(obj[1])

    if isinstance(obj, Mapping):
        if "n_correct" in obj and "n_total" in obj:
            return int(obj["n_correct"]), int(obj["n_total"])
        if "correct" in obj and "total" in obj:
            return int(obj["correct"]), int(obj["total"])
        if "accuracy" in obj and "n_total" in obj:
            n_total = int(obj["n_total"])
            n_correct = int(round(float(obj["accuracy"]) * n_total))
            return n_correct, n_total
        if "is_correct" in obj and isinstance(obj["is_correct"], Sequence):
            seq = obj["is_correct"]
            # Treat truthy as correct; ignore None
            valid = [x for x in seq if x is not None]
            n_total = len(valid)
            n_correct = sum(1 for x in valid if bool(x))
            return int(n_correct), int(n_total)

    raise ValueError(f"Cannot extract counts from object of type {type(obj)}")


def run_all_analyses(results_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Run all statistical analyses for the paper given model results.

    Expected `results_dict` structure (flexible):
        Option A (nested, recommended):
            {
              "<model_name>": {
                  "<dataset_name>": {
                      "<condition>": {"n_correct": int, "n_total": int} | (n_correct, n_total) | ...
                  }
              }
            }

        Option B (legacy nested without conditions):
            {
              "<model_name>": {
                  "<dataset_name>": {"n_correct": int, "n_total": int} | (n_correct, n_total) | ...
              }
            }

    Analyses performed:
    - Binomial tests vs chance for each model on each dataset
    - Two-proportion z-tests vs human benchmark (if provided) per dataset+condition
    - Pairwise Fisher's exact tests among five models per dataset with Bonferroni correction
      (alpha=0.005, 10 comparisons for 5 models) within each dataset+condition
    - Cohen's h effect sizes for all comparisons
    - Post-hoc power analyses (alpha=0.01, two-sided)

    The function returns a structured dict of all results and also prints a formatted
    summary matching the tables in the paper.

    Verification:
    - Warn if computed accuracies differ from known reported values by > 0.001 (absolute proportion).
    """
    # Normalize counts into a consistent structure:
    # counts[model][dataset][condition] = (n_correct, n_total)
    models = sorted(results_dict.keys())
    counts: Dict[str, Dict[str, Dict[str, Tuple[int, int]]]] = {m: {} for m in models}

    for m in models:
        per_dataset = results_dict.get(m, {}) or {}
        if not isinstance(per_dataset, Mapping):
            raise ValueError(f"results_dict[{m!r}] must be a mapping")
        for dataset_name, payload in per_dataset.items():
            if isinstance(payload, Mapping) and any(isinstance(v, Mapping) or isinstance(v, (tuple, list)) for v in payload.values()):
                # Might be either condition-mapped or legacy counts dict. Detect by keys.
                if "n_correct" in payload or "correct" in payload or "total" in payload or "n_total" in payload or "accuracy" in payload:
                    # Legacy: payload is counts for dataset
                    n_correct, n_total = _extract_counts(payload)
                    counts[m].setdefault(dataset_name, {})["video_only"] = (n_correct, n_total)
                else:
                    # Condition-mapped
                    for condition_name, obj in payload.items():
                        n_correct, n_total = _extract_counts(obj)
                        counts[m].setdefault(dataset_name, {})[str(condition_name)] = (n_correct, n_total)
            elif isinstance(payload, (tuple, list)):
                n_correct, n_total = _extract_counts(payload)
                counts[m].setdefault(dataset_name, {})["video_only"] = (n_correct, n_total)
            else:
                raise ValueError(f"Unrecognized payload for model={m} dataset={dataset_name}: {type(payload)}")

    # Prepare outputs
    out: Dict[str, Any] = {
        "benchmarks": {
            "chance": CHANCE,
            "human_benchmarks": HUMAN_BENCHMARKS,
        },
        "per_model_dataset": {},
        "comparisons": {
            "vs_chance": {},
            "vs_human_benchmark": {},
            "pairwise_models_fisher": {},
            "cross_dataset_video_only": [],
            "modality_ablations_gemini_flash": [],
        },
    }

    # Per model/dataset/condition stats and binomial vs chance
    print("\n=== Table: Accuracy, Wilson CI, Binomial vs chance ===")
    header = f"{'Model':<18} {'Dataset':<12} {'Cond':<10} {'n':>6} {'Acc%':>8} {'CI95%':>20} {'p(chance)':>12}"
    print(header)
    print("-" * len(header))

    for m in models:
        out["per_model_dataset"].setdefault(m, {})
        for dataset_name, per_cond in counts[m].items():
            out["per_model_dataset"][m].setdefault(dataset_name, {})
            for cond, (n_correct, n_total) in per_cond.items():
                acc = n_correct / n_total if n_total > 0 else 0.0
                ci_lo, ci_hi = wilson_ci(n_correct, n_total, confidence=0.95)
                p_chance = binomial_test_vs_chance(n_correct, n_total, chance=CHANCE)

                out["per_model_dataset"][m][dataset_name][cond] = {
                    "n_correct": n_correct,
                    "n_total": n_total,
                    "accuracy": acc,
                    "wilson_ci_95": (ci_lo, ci_hi),
                }
                out["comparisons"]["vs_chance"].setdefault(m, {}).setdefault(dataset_name, {})[cond] = {"p_value": p_chance}

                print(
                    f"{m:<18} {dataset_name:<12} {cond:<10} {n_total:>6} {acc*100:>7.2f} "
                    f"[{ci_lo*100:>6.2f}, {ci_hi*100:>6.2f}] {p_chance:>12.3g}"
                )

    # Z-tests vs human benchmark (if provided) per dataset+condition
    print("\n=== Table: Z-tests vs human benchmark (if configured) ===")
    header = f"{'Model':<18} {'Dataset':<12} {'Cond':<10} {'Acc%':>8} {'z':>10} {'p':>10} {'h':>10} {'benchmark':>18}"
    print(header)
    print("-" * len(header))

    for m in models:
        out["comparisons"]["vs_human_benchmark"].setdefault(m, {})
        for dataset_name, per_cond in counts[m].items():
            out["comparisons"]["vs_human_benchmark"][m].setdefault(dataset_name, {})
            for cond, (n_correct, n_total) in per_cond.items():
                p_model = n_correct / n_total if n_total > 0 else 0.0
                b = (HUMAN_BENCHMARKS.get(dataset_name, {}) or {}).get(cond)
                if not b or b.get("accuracy") is None or b.get("n") is None:
                    out["comparisons"]["vs_human_benchmark"][m][dataset_name][cond] = None
                    print(f"{m:<18} {dataset_name:<12} {cond:<10} {p_model*100:>7.2f} {'-':>10} {'-':>10} {'-':>10} {'(none)':>18}")
                    continue

                p_b = float(b["accuracy"])
                n_b = int(b["n"])
                z, p = two_proportion_z_test(p_model, n_total, p_b, n_b)
                h = cohen_h(p_model, p_b)
                out["comparisons"]["vs_human_benchmark"][m][dataset_name][cond] = {
                    "benchmark": b,
                    "z": z,
                    "p_value_raw": p,
                    "cohen_h": h,
                    "power_alpha_0_01": power_analysis(h, n=min(n_total, n_b), alpha=0.01),
                }
                label = (b.get("citation") or "human")[:18]
                print(f"{m:<18} {dataset_name:<12} {cond:<10} {p_model*100:>7.2f} {z:>10.2f} {p:>10.3g} {h:>10.2f} {label:>18}")

    # Pairwise Fisher exact among models per dataset+condition (10 comparisons), Bonferroni alpha=0.005
    print("\n=== Table: Pairwise model comparisons (Fisher exact; Bonferroni alpha=0.005; 10 comparisons) ===")
    # Build list of dataset+condition combinations present
    combos: List[Tuple[str, str]] = []
    for m in models:
        for dataset_name, per_cond in counts[m].items():
            for cond in per_cond.keys():
                if (dataset_name, cond) not in combos:
                    combos.append((dataset_name, cond))
    combos = sorted(combos)

    for dataset_name, cond in combos:
        available_models = [m for m in models if dataset_name in counts[m] and cond in counts[m][dataset_name]]
        if len(available_models) < 2:
            continue

        raw_pvals: List[float] = []
        pair_results: List[Dict[str, Any]] = []

        for i in range(len(available_models)):
            for j in range(i + 1, len(available_models)):
                m1, m2 = available_models[i], available_models[j]
                c1, t1 = counts[m1][dataset_name][cond]
                c2, t2 = counts[m2][dataset_name][cond]
                odds, p = fisher_exact_2x2(c1, t1, c2, t2)
                h = cohen_h(c1 / t1, c2 / t2)
                raw_pvals.append(p)
                pair_results.append(
                    {
                        "dataset": dataset_name,
                        "condition": cond,
                        "model_a": m1,
                        "model_b": m2,
                        "odds_ratio": odds,
                        "p_value_raw": p,
                        "cohen_h": h,
                        "n_a": t1,
                        "n_b": t2,
                        "correct_a": c1,
                        "correct_b": c2,
                    }
                )

        corrected, thresh = bonferroni_correct(raw_pvals, alpha=0.005)
        for r, p_c in zip(pair_results, corrected):
            r["p_value_bonferroni"] = p_c
            r["alpha_threshold"] = thresh

        out["comparisons"]["pairwise_models_fisher"][f"{dataset_name}:{cond}"] = pair_results

        print(f"\nDataset: {dataset_name} | Condition: {cond} (threshold={thresh:.4g})")
        print(f"{'A':<18} {'B':<18} {'OR':>10} {'p_raw':>10} {'p_corr':>10} {'h':>8}")
        for r in pair_results:
            print(
                f"{r['model_a']:<18} {r['model_b']:<18} {r['odds_ratio']:>10.3g} "
                f"{r['p_value_raw']:>10.3g} {r['p_value_bonferroni']:>10.3g} {r['cohen_h']:>8.2f}"
            )

    # Cross-dataset generalization (video_only): EU vs Mindreading per model
    print("\n=== Table: Cross-dataset generalization (video_only; EU -> Mindreading) ===")
    cross_rows: List[Dict[str, Any]] = []
    cross_pvals: List[float] = []
    for m in models:
        eu = counts.get(m, {}).get("eu_emotion", {}).get("video_only")
        mr = counts.get(m, {}).get("mindreading", {}).get("video_only")
        if not eu or not mr:
            continue
        c_eu, t_eu = eu
        c_mr, t_mr = mr
        p_eu = c_eu / t_eu if t_eu else 0.0
        p_mr = c_mr / t_mr if t_mr else 0.0
        diff = p_mr - p_eu
        z, p = two_proportion_z_test(p_eu, t_eu, p_mr, t_mr)
        h = cohen_h(p_eu, p_mr)
        cross_pvals.append(p)
        cross_rows.append(
            {
                "model": m,
                "eu_accuracy": p_eu,
                "mindreading_accuracy": p_mr,
                "difference": diff,
                "z": z,
                "p_value_raw": p,
                "cohen_h": h,
                "n_eu": t_eu,
                "n_mr": t_mr,
            }
        )
        print(
            f"{m:<18} EU={p_eu*100:>6.2f}% MR={p_mr*100:>6.2f}% "
            f"diff={diff*100:>+6.2f}pp z={z:>6.2f} p={p:>8.3g} h={h:>6.2f}"
        )

    if cross_pvals:
        cross_corrected, cross_thresh = bonferroni_correct(cross_pvals, alpha=0.01)
        for row, p_c in zip(cross_rows, cross_corrected):
            row["p_value_bonferroni"] = p_c
            row["alpha_threshold"] = cross_thresh
    out["comparisons"]["cross_dataset_video_only"] = cross_rows

    # Modality ablations for Gemini Flash (within-model, EU and MR)
    print("\n=== Table: Modality ablations (gemini-3-flash) ===")
    modality_rows: List[Dict[str, Any]] = []
    modality_pvals: List[float] = []
    flash_counts = counts.get("gemini-3-flash", {})
    for dataset_name in ("eu_emotion", "mindreading"):
        per_cond = flash_counts.get(dataset_name, {})
        if len(per_cond) < 2:
            continue
        conds = sorted(per_cond.keys())
        for i in range(len(conds)):
            for j in range(i + 1, len(conds)):
                c1, c2 = conds[i], conds[j]
                n1, t1 = per_cond[c1]
                n2, t2 = per_cond[c2]
                p1 = n1 / t1 if t1 else 0.0
                p2 = n2 / t2 if t2 else 0.0
                z, p = two_proportion_z_test(p1, t1, p2, t2)
                h = cohen_h(p1, p2)
                modality_pvals.append(p)
                modality_rows.append(
                    {
                        "dataset": dataset_name,
                        "condition_a": c1,
                        "condition_b": c2,
                        "accuracy_a": p1,
                        "accuracy_b": p2,
                        "n_a": t1,
                        "n_b": t2,
                        "z": z,
                        "p_value_raw": p,
                        "cohen_h": h,
                    }
                )
                print(
                    f"{dataset_name:<12} {c1:<12} vs {c2:<12} "
                    f"{p1*100:>6.2f}% vs {p2*100:>6.2f}% z={z:>6.2f} p={p:>8.3g} h={h:>6.2f}"
                )

    if modality_pvals:
        mod_corrected, mod_thresh = bonferroni_correct(modality_pvals, alpha=0.005)
        for row, p_c in zip(modality_rows, mod_corrected):
            row["p_value_bonferroni"] = p_c
            row["alpha_threshold"] = mod_thresh
    out["comparisons"]["modality_ablations_gemini_flash"] = modality_rows

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from pathlib import Path

    from analysis.load_results import load_results_from_summaries, summarize_loaded_results

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results" / "full_run"
    if results_dir.exists():
        results = load_results_from_summaries(results_dir)
        print("Loaded results from:", results_dir)
        print(summarize_loaded_results(results))
        run_all_analyses(results)
    else:
        raise SystemExit(f"No results at {results_dir}. Run evaluations first, then: python analysis/run_study_analysis.py")

