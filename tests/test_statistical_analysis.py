"""
ensuring reported statistics are computed correctly
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the publication_repo root is on sys.path for imports when running pytest.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistical_analysis import (  # noqa: E402
    binomial_test_vs_chance,
    cohen_h,
    two_proportion_z_test,
    wilson_ci,
)


def test_wilson_ci_gemini_3_pro_eu_emotion():
    lo, hi = wilson_ci(96, 117)
    np.testing.assert_allclose([lo, hi], [0.7411, 0.8795], rtol=0.01)


def test_wilson_ci_gemini_3_pro_mindreading():
    # Gemini 3 Pro Mindreading is 376/583 = 0.6449
    lo, hi = wilson_ci(376, 583)
    np.testing.assert_allclose([lo, hi], [0.6053, 0.6827], rtol=0.01)


@pytest.mark.parametrize(
    "p1,p2,expected",
    [
        (0.8205, 0.8629, -0.11639621),
        (0.8205, 0.6805, 0.32646044),
        (0.5592, 0.8629, -0.69351824),
        (0.5592, 0.6805, -0.25066159),
    ],
)
def test_cohen_h(p1, p2, expected):
    h = cohen_h(p1, p2)
    np.testing.assert_allclose(h, expected, rtol=0.01)


def test_two_proportion_z_test_vs_non_autistic_benchmark_close():
    # The paper-style benchmark comparison uses an effective benchmark n (not participant count).
    z, p = two_proportion_z_test(0.8205, 117, 0.8629, 76)
    np.testing.assert_allclose(z, -0.77954506, rtol=0.01)
    np.testing.assert_allclose(p, 0.43565871, rtol=0.01)


def test_two_proportion_z_test_large_difference_p_lt_0_001():
    z, p = two_proportion_z_test(0.5592, 583, 0.8629, 82)
    np.testing.assert_allclose(z, -5.24897007, rtol=0.01)
    assert p < 0.001


def test_two_proportion_z_test_vs_autistic_benchmark():
    z, p = two_proportion_z_test(0.5592, 583, 0.6805, 99)
    np.testing.assert_allclose(z, -2.26, rtol=0.01)
    np.testing.assert_allclose(p, 0.024, rtol=0.01)


def test_binomial_test_vs_chance_eu_emotion():
    p = binomial_test_vs_chance(96, 117, 0.25)
    assert p < 0.001


def test_binomial_test_vs_chance_mindreading():
    p = binomial_test_vs_chance(326, 583, 0.25)
    assert p < 0.001
