from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower

logger = logging.getLogger(__name__)


NON_AUTISTIC_MEAN = 0.8629
NON_AUTISTIC_N = 17
AUTISTIC_MEAN = 0.6805
AUTISTIC_N = 21
CHANCE = 0.25


def wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95) -> Tuple[float, float]:
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
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    if n_correct < 0 or n_correct > n_total:
        raise ValueError("n_correct must satisfy 0 <= n_correct <= n_total")
    if not (0.0 <= chance <= 1.0):
        raise ValueError("chance must be in [0, 1]")

    res = stats.binomtest(k=n_correct, n=n_total, p=chance, alternative="greater")
    return float(res.pvalue)


def two_proportion_z_test(p1: float, n1: int, p2: float, n2: int) -> Tuple[float, float]:
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
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must be in [0, 1]")
    return float(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def fisher_exact_2x2(correct1: int, total1: int, correct2: int, total2: int) -> Tuple[float, float]:
    if total1 <= 0 or total2 <= 0:
        raise ValueError("total1 and total2 must be > 0")
    if correct1 < 0 or correct1 > total1 or correct2 < 0 or correct2 > total2:
        raise ValueError("correct counts must satisfy 0 <= correct <= total")

    table = np.array([[correct1, total1 - correct1], [correct2, total2 - correct2]], dtype=int)
    odds_ratio, p_value = stats.fisher_exact(table, alternative="two-sided")
    return float(odds_ratio), float(p_value)


def bonferroni_correct(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[List[float], float]:
    pvals = [float(p) for p in p_values]
    m = max(1, len(pvals))
    threshold = alpha / m
    corrected = [min(1.0, p * m) for p in pvals]
    return corrected, float(threshold)


def power_analysis(effect_size_h: float, n: int, alpha: float = 0.01) -> float:
    if n <= 0:
        raise ValueError("n must be > 0")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")

    analysis = NormalIndPower()
    power = analysis.power(effect_size=abs(effect_size_h), nobs1=n, alpha=alpha, ratio=1.0, alternative="two-sided")
    return float(power)


def _extract_counts(obj: Any) -> Tuple[int, int]:
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
            valid = [x for x in seq if x is not None]
            n_total = len(valid)
            n_correct = sum(1 for x in valid if bool(x))
            return int(n_correct), int(n_total)

    raise ValueError(f"Cannot extract counts from object of type {type(obj)}")


def run_all_analyses(results_dict: Mapping[str, Any]) -> Dict[str, Any]:
    known = {
        ("gemini-3-pro", "eu_emotion"): (0.8205, 117),
        ("gemini-3-flash", "eu_emotion"): (0.7712, 118),
        ("gpt-5", "eu_emotion"): (0.7522, 113),
        ("gpt-5-mini", "eu_emotion"): (0.7203, 118),
        ("claude-opus-4-5", "eu_emotion"): (0.7297, 111),
        ("gemini-3-flash", "mindreading"): (0.6724, 583),
        ("gpt-5", "mindreading"): (0.6573, 572),
        ("gemini-3-pro", "mindreading"): (0.6449, 583),
        ("gpt-5-mini", "mindreading"): (0.6261, 583),
        ("claude-opus-4-5", "mindreading"): (0.5592, 583),
    }

    models = sorted(results_dict.keys())
    datasets = ["eu_emotion", "mindreading"]
    counts: Dict[str, Dict[str, Tuple[int, int]]] = {m: {} for m in models}

    for m in models:
        for d in datasets:
            if d not in results_dict[m]:
                continue
            n_correct, n_total = _extract_counts(results_dict[m][d])
            counts[m][d] = (n_correct, n_total)

            acc = n_correct / n_total if n_total > 0 else 0.0
            if (m, d) in known:
                exp_acc, exp_n = known[(m, d)]
                if n_total == exp_n and abs(acc - exp_acc) > 0.001:
                    logger.warning(
                        "Accuracy mismatch for %s %s: computed=%.4f expected=%.4f (n=%s)",
                        m,
                        d,
                        acc,
                        exp_acc,
                        n_total,
                    )

    out: Dict[str, Any] = {
        "benchmarks": {
            "chance": CHANCE,
            "non_autistic": {"mean": NON_AUTISTIC_MEAN, "n": NON_AUTISTIC_N},
            "autistic": {"mean": AUTISTIC_MEAN, "n": AUTISTIC_N},
        },
        "per_model_dataset": {},
        "comparisons": {
            "vs_chance": {},
            "vs_non_autistic": {},
            "vs_autistic": {},
            "pairwise_models_fisher": {},
            "cross_dataset_ztests": {},
        },
    }

    print("\n=== Table: Accuracy, Wilson CI, Binomial vs chance ===")
    header = f"{'Model':<18} {'Dataset':<12} {'n':>6} {'Acc%':>8} {'CI95%':>20} {'p(chance)':>12}"
    print(header)
    print("-" * len(header))

    for m in models:
        out["per_model_dataset"].setdefault(m, {})
        for d in datasets:
            if d not in counts[m]:
                continue
            n_correct, n_total = counts[m][d]
            acc = n_correct / n_total
            ci_lo, ci_hi = wilson_ci(n_correct, n_total, confidence=0.95)
            p_chance = binomial_test_vs_chance(n_correct, n_total, chance=CHANCE)

            out["per_model_dataset"][m][d] = {
                "n_correct": n_correct,
                "n_total": n_total,
                "accuracy": acc,
                "wilson_ci_95": (ci_lo, ci_hi),
            }
            out["comparisons"]["vs_chance"].setdefault(m, {})[d] = {"p_value": p_chance}

            print(
                f"{m:<18} {d:<12} {n_total:>6} {acc*100:>7.2f} "
                f"[{ci_lo*100:>6.2f}, {ci_hi*100:>6.2f}] {p_chance:>12.3g}"
            )

    print("\n=== Table: Z-tests vs human benchmarks (Bonferroni alpha=0.025; 2 comparisons) ===")
    header = (
        f"{'Model':<18} {'Dataset':<12} {'Acc%':>8} "
        f"{'z(non)':>10} {'p(non)c':>10} {'h(non)':>10} "
        f"{'z(aut)':>10} {'p(aut)c':>10} {'h(aut)':>10}"
    )
    print(header)
    print("-" * len(header))

    for m in models:
        out["comparisons"]["vs_non_autistic"].setdefault(m, {})
        out["comparisons"]["vs_autistic"].setdefault(m, {})
        for d in datasets:
            if d not in counts[m]:
                continue
            n_correct, n_total = counts[m][d]
            p_model = n_correct / n_total

            z_non, p_non = two_proportion_z_test(p_model, n_total, NON_AUTISTIC_MEAN, NON_AUTISTIC_N)
            z_aut, p_aut = two_proportion_z_test(p_model, n_total, AUTISTIC_MEAN, AUTISTIC_N)

            corrected, thresh = bonferroni_correct([p_non, p_aut], alpha=0.025)
            p_non_c, p_aut_c = corrected

            h_non = cohen_h(p_model, NON_AUTISTIC_MEAN)
            h_aut = cohen_h(p_model, AUTISTIC_MEAN)

            out["comparisons"]["vs_non_autistic"][m][d] = {
                "z": z_non,
                "p_value_raw": p_non,
                "p_value_bonferroni": p_non_c,
                "alpha_threshold": thresh,
                "cohen_h": h_non,
                "power_alpha_0_01": power_analysis(h_non, n=min(n_total, NON_AUTISTIC_N), alpha=0.01),
            }
            out["comparisons"]["vs_autistic"][m][d] = {
                "z": z_aut,
                "p_value_raw": p_aut,
                "p_value_bonferroni": p_aut_c,
                "alpha_threshold": thresh,
                "cohen_h": h_aut,
                "power_alpha_0_01": power_analysis(h_aut, n=min(n_total, AUTISTIC_N), alpha=0.01),
            }

            print(
                f"{m:<18} {d:<12} {p_model*100:>7.2f} "
                f"{z_non:>10.2f} {p_non_c:>10.3g} {h_non:>10.2f} "
                f"{z_aut:>10.2f} {p_aut_c:>10.3g} {h_aut:>10.2f}"
            )

    print("\n=== Table: Pairwise model comparisons (Fisher exact; Bonferroni alpha=0.005; 10 comparisons) ===")
    for d in datasets:
        available_models = [m for m in models if d in counts[m]]
        if len(available_models) < 2:
            continue

        pairs: List[Tuple[str, str]] = []
        raw_pvals: List[float] = []
        pair_results: List[Dict[str, Any]] = []

        for i in range(len(available_models)):
            for j in range(i + 1, len(available_models)):
                m1, m2 = available_models[i], available_models[j]
                c1, t1 = counts[m1][d]
                c2, t2 = counts[m2][d]
                odds, p = fisher_exact_2x2(c1, t1, c2, t2)
                h = cohen_h(c1 / t1, c2 / t2)
                pairs.append((m1, m2))
                raw_pvals.append(p)
                pair_results.append(
                    {
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

        out["comparisons"]["pairwise_models_fisher"][d] = pair_results

        print(f"\nDataset: {d} (threshold={thresh:.4g})")
        print(f"{'A':<18} {'B':<18} {'OR':>10} {'p_raw':>10} {'p_corr':>10} {'h':>8}")
        for r in pair_results:
            print(
                f"{r['model_a']:<18} {r['model_b']:<18} {r['odds_ratio']:>10.3g} "
                f"{r['p_value_raw']:>10.3g} {r['p_value_bonferroni']:>10.3g} {r['cohen_h']:>8.2f}"
            )

    print("\n=== Table: Cross-dataset comparisons per model (Z-test; Bonferroni alpha=0.010; 5 comparisons) ===")
    raw_pvals_cd: List[float] = []
    cd_models: List[str] = []
    cd_results: List[Dict[str, Any]] = []

    for m in models:
        if "eu_emotion" not in counts[m] or "mindreading" not in counts[m]:
            continue
        c1, t1 = counts[m]["eu_emotion"]
        c2, t2 = counts[m]["mindreading"]
        p1 = c1 / t1
        p2 = c2 / t2
        z, p = two_proportion_z_test(p1, t1, p2, t2)
        h = cohen_h(p1, p2)
        cd_models.append(m)
        raw_pvals_cd.append(p)
        cd_results.append(
            {
                "model": m,
                "z": z,
                "p_value_raw": p,
                "cohen_h": h,
                "power_alpha_0_01": power_analysis(h, n=min(t1, t2), alpha=0.01),
                "eu_emotion": {"n_correct": c1, "n_total": t1, "accuracy": p1},
                "mindreading": {"n_correct": c2, "n_total": t2, "accuracy": p2},
            }
        )

    corrected, thresh = bonferroni_correct(raw_pvals_cd, alpha=0.010)
    for r, p_c in zip(cd_results, corrected):
        r["p_value_bonferroni"] = p_c
        r["alpha_threshold"] = thresh

    out["comparisons"]["cross_dataset_ztests"] = cd_results

    print(f"{'Model':<18} {'z':>10} {'p_corr':>10} {'h':>8} {'EU%':>8} {'MR%':>8}")
    for r in cd_results:
        print(
            f"{r['model']:<18} {r['z']:>10.2f} {r['p_value_bonferroni']:>10.3g} {r['cohen_h']:>8.2f} "
            f"{r['eu_emotion']['accuracy']*100:>7.2f} {r['mindreading']['accuracy']*100:>7.2f}"
        )

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    example_results = {
        "gemini-3-pro": {"eu_emotion": (96, 117), "mindreading": (376, 583)},
        "gemini-3-flash": {"eu_emotion": (91, 118), "mindreading": (392, 583)},
        "gpt-5": {"eu_emotion": (85, 113), "mindreading": (376, 572)},
        "gpt-5-mini": {"eu_emotion": (85, 118), "mindreading": (365, 583)},
        "claude-opus-4-5": {"eu_emotion": (81, 111), "mindreading": (326, 583)},
    }

    run_all_analyses(example_results)