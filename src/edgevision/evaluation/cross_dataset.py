"""Metrics for a model evaluated on another dataset (ASL Citizen) and paired model comparisons.

``summarize`` reports top-1/top-3 with percentile bootstrap intervals over videos and, as a
sensitivity analysis, over signers (videos of one signer are not independent); calibration (ECE
before and after the model's temperature); the decision rule at the shipped threshold; and
accuracy per sign and per mapping rule. ``paired`` compares two models on the same items with a
paired bootstrap interval and a sign-flip test; ``holm_family`` adjusts a family of them.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from edgevision.evaluation import stats
from edgevision.web_export import expected_calibration_error, softmax

ECE_RESAMPLES = 1000


def topk_correct(
    logits: npt.NDArray[np.floating[Any]], y: npt.NDArray[np.int64], k: int
) -> npt.NDArray[np.bool_]:
    """Per item: is the label among the k highest logits (ties broken by lower index)?"""
    order = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    hits: npt.NDArray[np.bool_] = (order == y[:, None]).any(axis=1)
    return hits


def cluster_bootstrap_ci(
    values: npt.ArrayLike,
    clusters: npt.ArrayLike,
    resamples: int = stats.BOOTSTRAP_RESAMPLES,
    seed: int = stats.SEED,
) -> tuple[float, float]:
    """Percentile interval of the item mean when whole clusters are resampled."""
    labels = np.asarray(clusters)
    _, cluster_index = np.unique(labels, return_inverse=True)
    sums = np.bincount(cluster_index, weights=np.asarray(values, dtype=np.float64))
    counts = np.bincount(cluster_index).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, sums.size, size=(resamples, sums.size))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _ece_ci(
    probabilities: npt.NDArray[np.float64], y: npt.NDArray[np.int64]
) -> tuple[float, float]:
    rng = np.random.default_rng(stats.SEED)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    values = []
    for _ in range(ECE_RESAMPLES):
        index = rng.integers(0, y.size, y.size)
        values.append(expected_calibration_error(confidence[index], correct[index]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def summarize(
    logits: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.int64],
    *,
    signers: Sequence[str] | npt.NDArray[Any],
    signs: Sequence[str],
    rules: Sequence[str],
    labels: Sequence[str],
    temperature: float,
    threshold: float,
    excluded: int = 0,
) -> dict[str, Any]:
    """Everything the cross-dataset report needs; ``excluded`` counts videos preprocessing left
    out (too few hand frames), reported and also scored as errors in ``top1_all_videos``."""
    n = int(y.size)
    if n == 0:
        raise ValueError("no items to evaluate")
    top1, top3 = topk_correct(logits, y, 1), topk_correct(logits, y, 3)
    raw = softmax(logits.astype(np.float64))
    scaled = softmax(logits.astype(np.float64) / temperature)
    confidence = scaled.max(axis=1)
    accepted = confidence >= threshold
    signer_array = np.asarray(signers)
    per_signer = [float(top1[signer_array == s].mean()) for s in np.unique(signer_array)]
    result: dict[str, Any] = {
        "videos": n,
        "excluded_videos": excluded,
        "signers": len(per_signer),
        "signs": len(set(signs)),
        "top1": float(top1.mean()),
        "top1_ci": stats.bootstrap_ci(top1),
        "top1_ci_signer": cluster_bootstrap_ci(top1, signer_array),
        "top3": float(top3.mean()),
        "top3_ci": stats.bootstrap_ci(top3),
        "top3_ci_signer": cluster_bootstrap_ci(top3, signer_array),
        "top1_all_videos": float(top1.sum() / (n + excluded)),
        "top1_by_signer_min": min(per_signer),
        "top1_by_signer_max": max(per_signer),
        "ece_raw": expected_calibration_error(raw.max(axis=1), raw.argmax(axis=1) == y),
        "ece_raw_ci": _ece_ci(raw, y),
        "ece": expected_calibration_error(confidence, scaled.argmax(axis=1) == y),
        "ece_ci": _ece_ci(scaled, y),
        "temperature": temperature,
        "threshold": threshold,
        "coverage": float(accepted.mean()),
        "selective_accuracy": float(top1[accepted].mean()) if accepted.any() else None,
        "wrong_rejected": float((~accepted[~top1]).mean()) if (~top1).any() else None,
        "correct_rejected": float((~accepted[top1]).mean()) if top1.any() else None,
    }
    rule_array = np.asarray(rules)
    result["by_rule"] = {
        rule: {
            "videos": int((rule_array == rule).sum()),
            "top1": float(top1[rule_array == rule].mean()),
            "top3": float(top3[rule_array == rule].mean()),
        }
        for rule in sorted(set(rule_array.tolist()))
    }
    sign_array = np.asarray(signs)
    predicted = np.asarray(labels)[logits.argmax(axis=1)]
    per_sign = []
    for sign in sorted(set(sign_array.tolist())):
        mask = sign_array == sign
        wrong = predicted[mask & ~top1]
        values, counts = np.unique(wrong, return_counts=True)
        confusions = sorted(zip(values.tolist(), counts.tolist(), strict=True), key=lambda c: -c[1])
        per_sign.append(
            {
                "sign": sign,
                "videos": int(mask.sum()),
                "top1": float(top1[mask].mean()),
                "top3": float(top3[mask].mean()),
                "rules": "+".join(sorted(set(rule_array[mask].tolist()))),
                "confused_with": "; ".join(f"{label} ({count})" for label, count in confusions[:3]),
            }
        )
    result["per_sign"] = per_sign
    return result


def paired(
    correct_a: npt.ArrayLike, correct_b: npt.ArrayLike, *, clusters: npt.ArrayLike | None = None
) -> dict[str, Any]:
    """b - a on the same items: the difference, its paired bootstrap interval, items better and
    worse, the sign-flip p value and, with ``clusters``, the cluster-level interval."""
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired comparisons need two 1-D arrays over the same items")
    difference = b - a
    result: dict[str, Any] = {
        "items": int(a.size),
        "a": float(a.mean()),
        "b": float(b.mean()),
        "diff": float(difference.mean()),
        # paired_bootstrap_ci(x, y) is the interval of mean(x - y): pass (b, a) for b - a.
        "ci": stats.paired_bootstrap_ci(b, a),
        "better": int((difference > 0).sum()),
        "worse": int((difference < 0).sum()),
        "p": stats.sign_flip_p(difference),
    }
    if clusters is not None:
        result["ci_cluster"] = cluster_bootstrap_ci(difference, clusters)
    return result


def holm_family(comparisons: Mapping[str, dict[str, Any]], alpha: float = 0.05) -> None:
    """Add p_holm and significant (Holm-adjusted p < alpha) to each comparison in place."""
    names = list(comparisons)
    adjusted = stats.holm([comparisons[name]["p"] for name in names])
    for name, value in zip(names, adjusted, strict=True):
        comparisons[name]["p_holm"] = value
        comparisons[name]["significant"] = bool(value < alpha)
