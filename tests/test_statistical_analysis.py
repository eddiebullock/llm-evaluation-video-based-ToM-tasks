from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistical_analysis import (  # noqa: E402
    binomial_test_vs_chance,
    cohen_h,
    two_proportion_z_test,
    wilson_ci,
)


def test_wilson_ci_gemini_3_flash_eu_video():
    lo, hi = wilson_ci(88, 118)
    np.testing.assert_allclose([lo, hi], [0.6603, 0.8157], rtol=0.01)


def test_wilson_ci_gpt_5_mindreading_fair_subset():
    lo, hi = wilson_ci(380, 581)
    np.testing.assert_allclose([lo, hi], [0.6145, 0.6916], rtol=0.01)


@pytest.mark.parametrize(
    "p1,p2,expected",
    [
        (0.7458, 0.63, 0.2508),
        (0.7373, 0.63, 0.2315),
        (0.6540, 0.5749, 0.1680),
    ],
)
def test_cohen_h(p1, p2, expected):
    h = cohen_h(p1, p2)
    np.testing.assert_allclose(h, expected, rtol=0.05)


def test_two_proportion_z_test_flash_vs_oreilly_facial():
    z, p = two_proportion_z_test(88 / 118, 118, 0.63, 1231)
    np.testing.assert_allclose(z, 2.5028, rtol=0.01)
    np.testing.assert_allclose(p, 0.0123, rtol=0.05)


def test_two_proportion_z_test_gpt5_vs_oreilly_facial():
    z, p = two_proportion_z_test(87 / 118, 118, 0.63, 1231)
    np.testing.assert_allclose(z, 2.3185, rtol=0.01)
    np.testing.assert_allclose(p, 0.0204, rtol=0.05)


def test_two_proportion_z_test_flash_audio_vs_lassalle():
    z, p = two_proportion_z_test(69 / 118, 118, 0.4519, 427)
    np.testing.assert_allclose(z, 2.5566, rtol=0.01)
    np.testing.assert_allclose(p, 0.0106, rtol=0.05)


def test_binomial_test_vs_chance_eu_emotion():
    p = binomial_test_vs_chance(88, 118, 0.25)
    assert p < 0.001


def test_binomial_test_vs_chance_mindreading_fair_subset():
    p = binomial_test_vs_chance(380, 581, 0.25)
    assert p < 0.001
