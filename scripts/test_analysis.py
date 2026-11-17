"""Paired model comparisons and test-set calibration from saved test predictions.

    python scripts/final_eval.py ... --predictions-out results/final_selection/predictions \
        --onnx-model web/public/model/model.onnx --onnx-run d_hybrid_v2feat_v2recipe/seed_0
    python scripts/test_analysis.py            # results/test_analysis/*

Reads the per-sequence test logits that final_eval.py saved (git-ignored .npz files; this script
never loads the test split itself) and writes, under results/test_analysis/:

* model_comparisons.csv: the planned comparisons on the signer-independent test split. A
  sequence's score is its top-1 correctness averaged over the experiment's seeds; the difference
  of means has a paired bootstrap interval over sequences and a two-sided sign-flip permutation
  p-value, Holm-adjusted over the family below. Sequences of one signer are not independent and
  there are only four test signers, so each row also gives the difference per test signer.
* calibration.json, reliability.csv, coverage_curve.csv, per_class.csv: the shipped web model
  (fp16 ONNX) on test, with the temperature and threshold that export_web.py chose on val.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from edgevision.evaluation.stats import holm, paired_bootstrap_ci, sign_flip_p
from edgevision.web_export import ECE_BINS, expected_calibration_error, nll, softmax

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results" / "final_selection" / "predictions"
OUT = ROOT / "results" / "test_analysis"
META = ROOT / "web" / "public" / "model" / "meta.json"
ALPHA = 0.05

# The family, fixed before this script first ran: the two v1 architecture contrasts against
# the v1 CNN, the three single steps of the v2 ablation, and the shipped architecture against
# the v1 CNN.
COMPARISONS = [
    ("gru_signer_independent", "cnn_signer_independent", "v1: GRU vs CNN"),
    ("transformer_signer_independent", "cnn_signer_independent", "v1: transformer vs CNN"),
    ("b_cnn_v2feat_v1recipe", "cnn_signer_independent", "v2 features (CNN, v1 recipe)"),
    ("c_cnn_v2feat_v2recipe", "b_cnn_v2feat_v1recipe", "v2 recipe (CNN, v2 features)"),
    (
        "d_hybrid_v2feat_v2recipe",
        "c_cnn_v2feat_v2recipe",
        "hybrid architecture (v2 features, v2 recipe)",
    ),
    ("d_hybrid_v2feat_v2recipe", "cnn_signer_independent", "shipped hybrid vs v1 CNN"),
]


def load(path: Path) -> dict[str, npt.NDArray[Any]]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def topk_hits(logits: npt.NDArray[Any], y: npt.NDArray[Any], k: int) -> npt.NDArray[np.float64]:
    top = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    return (top == y[:, None]).any(axis=1).astype(np.float64)


def experiment_scores(experiment: str) -> dict[str, Any]:
    """Seed-averaged per-sequence top-1 and top-3 correctness, aligned by sequence_id."""
    files = sorted((PREDICTIONS / experiment).glob("seed_*.npz"))
    if not files:
        raise FileNotFoundError(f"no saved predictions for {experiment} under {PREDICTIONS}")
    runs = [load(f) for f in files]
    order = np.argsort(runs[0]["sequence_id"])
    ids = runs[0]["sequence_id"][order]
    top1, top3 = [], []
    for run in runs:
        o = np.argsort(run["sequence_id"])
        if not np.array_equal(run["sequence_id"][o], ids):
            raise ValueError(f"{experiment}: seeds cover different test sequences")
        top1.append(topk_hits(run["logits"][o], run["y"][o], 1))
        top3.append(topk_hits(run["logits"][o], run["y"][o], 3))
    return {
        "ids": ids,
        "participant": runs[0]["participant_id"][order],
        "top1": np.mean(top1, axis=0),
        "top3": np.mean(top3, axis=0),
        "seeds": len(runs),
    }


def comparisons() -> list[dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    rows = []
    for a, b, label in COMPARISONS:
        for name in (a, b):
            if name not in cache:
                cache[name] = experiment_scores(name)
        sa, sb = cache[a], cache[b]
        if not np.array_equal(sa["ids"], sb["ids"]):
            raise ValueError(f"{a} and {b} cover different test sequences")
        diff = sa["top1"] - sb["top1"]
        low, high = paired_bootstrap_ci(sa["top1"], sb["top1"])
        signers = {
            int(p): float(diff[sa["participant"] == p].mean()) for p in np.unique(sa["participant"])
        }
        rows.append(
            {
                "comparison": label,
                "a": a,
                "b": b,
                "metric": "top1",
                "n_sequences": int(diff.size),
                "seeds_a": sa["seeds"],
                "seeds_b": sb["seeds"],
                "mean_a": float(sa["top1"].mean()),
                "mean_b": float(sb["top1"].mean()),
                "diff": float(diff.mean()),
                "ci_low": low,
                "ci_high": high,
                "p": sign_flip_p(diff),
                "top3_diff": float((sa["top3"] - sb["top3"]).mean()),
                "signer_diffs": ";".join(f"{p}:{v:+.4f}" for p, v in signers.items()),
                "signers_positive": sum(v > 0 for v in signers.values()),
                "signers": len(signers),
            }
        )
    for row, adjusted in zip(rows, holm([r["p"] for r in rows]), strict=True):
        row["p_holm"] = adjusted
        row["significant"] = adjusted < ALPHA
    return rows


def reliability(
    probs: npt.NDArray[Any], correct: npt.NDArray[Any], kind: str
) -> list[dict[str, Any]]:
    """The equal-width bins of expected_calibration_error, with counts, confidence and accuracy."""
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    which = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, ECE_BINS - 1)
    rows = []
    for b in range(ECE_BINS):
        mask = which == b
        rows.append(
            {
                "probabilities": kind,
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": int(mask.sum()),
                "mean_confidence": float(probs[mask].mean()) if mask.any() else float("nan"),
                "accuracy": float(correct[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return rows


def ece_intervals(
    conf_raw: npt.NDArray[Any],
    right_raw: npt.NDArray[Any],
    conf: npt.NDArray[Any],
    right: npt.NDArray[Any],
) -> dict[str, float]:
    """Percentile bootstrap intervals (over sequences, 1000 resamples, seed 0) of the test ECE
    before and after temperature scaling; sequences of one signer are not independent, so the
    intervals are optimistic."""
    rng = np.random.default_rng(0)
    before, after = [], []
    for _ in range(1000):
        index = rng.integers(0, conf.size, conf.size)
        before.append(expected_calibration_error(conf_raw[index], right_raw[index]))
        after.append(expected_calibration_error(conf[index], right[index]))
    low0, high0 = np.quantile(before, [0.025, 0.975])
    low1, high1 = np.quantile(after, [0.025, 0.975])
    return {
        "test_ece_before_ci": [float(low0), float(high0)],
        "test_ece_after_ci": [float(low1), float(high1)],
    }


def calibration() -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    meta = json.loads(META.read_text(encoding="utf-8"))
    cal = meta["calibration"]
    temperature, threshold = float(cal["temperature"]), float(cal["threshold"])
    data = load(PREDICTIONS / "web_model.npz")
    logits, y = data["logits"].astype(np.float64), data["y"]
    raw = softmax(logits)
    scaled = softmax(logits / temperature)
    conf_raw, conf = raw.max(axis=1), scaled.max(axis=1)
    correct = scaled.argmax(axis=1) == y  # temperature does not change the argmax
    accepted = conf >= threshold
    summary = {
        "model": "web/public/model/model.onnx",
        "model_sha256": meta["model_sha256"],
        "variant": meta["variant"],
        "n_sequences": int(y.size),
        "test_top1": float(correct.mean()),
        "test_top3": float(topk_hits(logits, y, 3).mean()),
        "temperature_from_val": temperature,
        "threshold_from_val": threshold,
        "ece_bins": ECE_BINS,
        "test_nll_before": nll(logits, y, 1.0),
        "test_nll_after": nll(logits, y, temperature),
        "test_ece_before": expected_calibration_error(conf_raw, raw.argmax(axis=1) == y),
        "test_ece_after": expected_calibration_error(conf, correct),
        "test_coverage_at_threshold": float(accepted.mean()),
        "test_selective_accuracy_at_threshold": float(correct[accepted].mean()),
        "test_net_score_at_threshold": float(
            (2 * (correct & accepted).sum() - accepted.sum()) / y.size
        ),
        "test_correct_rejected_at_threshold": float((correct & ~accepted).sum() / correct.sum()),
        "test_wrong_rejected_at_threshold": float((~correct & ~accepted).sum() / (~correct).sum()),
        **ece_intervals(conf_raw, raw.argmax(axis=1) == y, conf, correct),
        "val": {
            k: cal[k]
            for k in (
                "nll_before",
                "nll_after",
                "ece_before",
                "ece_after",
                "coverage",
                "selective_accuracy",
                "net_score",
                "val_top1",
                "val_top3",
                "val_sequences",
            )
        },
    }
    bins = reliability(conf_raw, raw.argmax(axis=1) == y, "raw") + reliability(
        conf, correct, "temperature_scaled"
    )
    order = np.argsort(-conf, kind="stable")
    hits = np.cumsum(correct[order])
    curve = []
    for coverage in np.linspace(0.01, 1.0, 100):
        k = max(1, round(float(coverage) * y.size))
        curve.append(
            {
                "coverage": float(k / y.size),
                "selective_accuracy": float(hits[k - 1] / k),
                "threshold": float(conf[order][k - 1]),
            }
        )
    labels = meta["labels"]
    per_class = []
    for index, label in enumerate(labels):
        mask = y == index
        per_class.append(
            {
                "label": label,
                "n": int(mask.sum()),
                "top1": float(correct[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return summary, bins, curve, per_class


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = comparisons()
    write_csv(OUT / "model_comparisons.csv", rows)
    for r in rows:
        print(
            f"{r['comparison']}: {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
            f"p_holm {r['p_holm']:.2g} signers +{r['signers_positive']}/{r['signers']}"
        )
    summary, bins, curve, per_class = calibration()
    (OUT / "calibration.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT / "reliability.csv", bins)
    write_csv(OUT / "coverage_curve.csv", curve)
    write_csv(OUT / "per_class.csv", per_class)
    print(
        f"web model on test: top-1 {summary['test_top1']:.4f}, "
        f"ECE {summary['test_ece_before']:.4f} -> "
        f"{summary['test_ece_after']:.4f}, coverage {summary['test_coverage_at_threshold']:.3f} at "
        f"selective accuracy {summary['test_selective_accuracy_at_threshold']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
