"""Small, dependency-free statistics for the paper's comparisons (numpy only).

* percentile bootstrap intervals of a mean, and of a paired mean difference;
* the two-sided paired sign-flip permutation test of a mean difference (exact when there are
  few non-zero differences, Monte Carlo otherwise), which is valid for the fractional per-clip
  scores of averaged browser runs as well as for 0/1 correctness, where it reduces to an exact
  McNemar-type test on the discordant items;
* Holm's step-down adjustment of a family of p-values (controls the family-wise error rate
  under any dependence between the tests).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

BOOTSTRAP_RESAMPLES = 10_000
PERMUTATIONS = 100_000
EXACT_MAX_NONZERO = 20
SEED = 0
# Largest number of matrix cells drawn at once: resampling or sign-flipping many items (e.g. 16k
# test sequences) happens in row chunks; small inputs (clips) are drawn in one go.
CHUNK_CELLS = 4_000_000


def _as_1d(values: npt.ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"expected a non-empty 1-D array, got shape {array.shape}")
    return array


def bootstrap_ci(
    values: npt.ArrayLike,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean of ``values`` (items resampled with
    replacement)."""
    x = _as_1d(values)
    rng = np.random.default_rng(seed)
    rows = max(1, CHUNK_CELLS // x.size)
    means = np.concatenate(
        [
            x[rng.integers(0, x.size, size=(min(rows, resamples - start), x.size))].mean(axis=1)
            for start in range(0, resamples, rows)
        ]
    )
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def paired_bootstrap_ci(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval of mean(a - b), resampling the paired items."""
    x, y = _as_1d(a), _as_1d(b)
    if x.shape != y.shape:
        raise ValueError(f"paired arrays differ in shape: {x.shape} vs {y.shape}")
    return bootstrap_ci(x - y, resamples, seed, level)


def sign_flip_p(
    differences: npt.ArrayLike,
    permutations: int = PERMUTATIONS,
    seed: int = SEED,
    exact_max_nonzero: int = EXACT_MAX_NONZERO,
) -> float:
    """Two-sided p-value of H0: the paired differences are symmetric about 0.

    The statistic is |sum of differences|; zero differences do not change it under any sign
    flip and are dropped. With at most ``exact_max_nonzero`` non-zero differences every sign
    pattern is enumerated; otherwise ``permutations`` random patterns are drawn and the p-value
    is (1 + count) / (1 + permutations), which is never zero.
    """
    d = _as_1d(differences)
    d = d[np.abs(d) > 1e-12]
    if d.size == 0:
        return 1.0
    observed = abs(float(d.sum()))
    tolerance = 1e-9 * max(1.0, observed)
    if d.size <= exact_max_nonzero:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=d.size)))
        stats = np.abs(signs @ d)
        return float(np.mean(stats >= observed - tolerance))
    rng = np.random.default_rng(seed)
    count = 0
    remaining = permutations
    while remaining > 0:
        batch = min(remaining, 20_000, max(1, CHUNK_CELLS // d.size))
        signs = rng.choice((-1.0, 1.0), size=(batch, d.size))
        count += int(np.sum(np.abs(signs @ d) >= observed - tolerance))
        remaining -= batch
    return float((1 + count) / (1 + permutations))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values (same order as the input), monotone and capped at 1."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return []
    order = np.argsort(p, kind="stable")
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * float(p[index]))
        adjusted[index] = min(1.0, running)
    return [float(v) for v in adjusted]
