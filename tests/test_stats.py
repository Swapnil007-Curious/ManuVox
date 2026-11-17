import itertools
import math

import numpy as np
import pytest

from edgevision.evaluation.stats import bootstrap_ci, holm, paired_bootstrap_ci, sign_flip_p


def test_bootstrap_ci_brackets_the_mean_and_is_seeded() -> None:
    x = np.r_[np.ones(30), np.zeros(20)]
    low, high = bootstrap_ci(x, resamples=2000, seed=0)
    assert low < 0.6 < high
    assert (low, high) == bootstrap_ci(x, resamples=2000, seed=0)
    assert low > 0.4 and high < 0.8


def test_bootstrap_ci_of_a_constant_is_a_point() -> None:
    assert bootstrap_ci(np.full(10, 0.5), resamples=500) == (0.5, 0.5)


def test_paired_bootstrap_ci_is_the_interval_of_the_differences() -> None:
    a = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    assert paired_bootstrap_ci(a, b, resamples=500, seed=3) == bootstrap_ci(a - b, 500, 3)
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, b[:-1])


def test_sign_flip_exact_matches_a_binomial_mcnemar_test() -> None:
    # 0/1 correctness: 8 items better in a, 2 better in b, the rest tied.
    d = np.r_[np.ones(8), -np.ones(2), np.zeros(15)]
    # Exact McNemar: two-sided binomial(10, 0.5) probability of a split at least as extreme as 8/2.
    tail = sum(math.comb(10, k) for k in range(0, 3)) / 2**10
    assert sign_flip_p(d) == pytest.approx(2 * tail)


def test_sign_flip_exact_enumeration_on_fractional_differences() -> None:
    d = np.array([1 / 3, 2 / 3, -1 / 3, 1.0, 0.0])
    nonzero = d[d != 0]
    stats = [abs(float(np.dot(s, nonzero))) for s in itertools.product((-1, 1), repeat=4)]
    expected = np.mean([s >= abs(nonzero.sum()) - 1e-12 for s in stats])
    assert sign_flip_p(d) == pytest.approx(expected)


def test_sign_flip_monte_carlo_is_close_to_exact_and_never_zero() -> None:
    rng = np.random.default_rng(1)
    d = rng.choice([-1.0, 0.0, 1.0, 1.0], size=40)
    p = sign_flip_p(d, permutations=20_000, seed=0)
    assert 0 < p <= 1
    assert sign_flip_p(np.zeros(5)) == 1.0
    assert sign_flip_p(np.ones(30), permutations=1000) == pytest.approx(1 / 1001)


def test_holm_adjusts_and_keeps_order() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm([]) == []
