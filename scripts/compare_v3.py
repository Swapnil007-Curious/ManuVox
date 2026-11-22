"""v2 vs v3, side by side, on the three test sets, with paired tests in one Holm family.

    python scripts/compare_v3.py

Inputs, each produced once:
- asl-signs signer-independent test: results/final_selection/predictions/web_model.npz (v2) and
  results/v3_eval/web_model.npz (v3; scripts/final_eval.py --skip-runs --onnx-model
  web/models/v3-citizen/model.onnx --onnx-run d_hybrid_v2feat_v2recipe/seed_0
  --predictions-out results/v3_eval).
- ASL Citizen test: results/asl_citizen/predictions/{v2,v3}_test.npz (scripts/citizen_eval.py).
- test_clips_v2: results/live_gap_v3 (scripts/live_gap.py and web/scripts/live-gap.mjs with
  experiments/live_gap_v3.json), configs v2_final and v3_final, browser and Python.

The family is top-1 on each test set and clip pipeline (four comparisons), Holm-adjusted; top-3
is reported with intervals but not tested. Items are paired by sequence id or clip. Writes
results/v2_vs_v3.json, .md and .csv.
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from edgevision.evaluation import stats
from edgevision.evaluation.cross_dataset import holm_family, paired, topk_correct

REPO = Path(__file__).resolve().parents[1]


def load_live_gap() -> Any:
    spec = importlib.util.spec_from_file_location("live_gap", REPO / "scripts" / "live_gap.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_gap"] = module
    spec.loader.exec_module(module)
    return module


def aligned(first: Path, second: Path) -> dict[str, np.ndarray]:
    """Both prediction files over the same items, in the first file's order."""
    with np.load(first) as a, np.load(second) as b:
        a_ids, b_ids = a["sequence_id"], b["sequence_id"]
        if set(a_ids.tolist()) != set(b_ids.tolist()):
            raise ValueError(f"{first} and {second} hold different items")
        order = {int(s): i for i, s in enumerate(b_ids)}
        index = np.array([order[int(s)] for s in a_ids])
        if not np.array_equal(a["y"], b["y"][index]):
            raise ValueError(f"labels differ between {first} and {second}")
        clusters = a["participant_id"] if "participant_id" in a.files else None
        return {
            "y": a["y"].astype(np.int64),
            "a": a["logits"],
            "b": b["logits"][index],
            "clusters": clusters,
        }


def split_comparison(first: Path, second: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = aligned(first, second)
    out = []
    for k in (1, 3):
        a = topk_correct(data["a"], data["y"], k)
        b = topk_correct(data["b"], data["y"], k)
        result = paired(a, b, clusters=data["clusters"])
        result["a_ci"], result["b_ci"] = stats.bootstrap_ci(a), stats.bootstrap_ci(b)
        if data["clusters"] is not None:
            # Each signer's own difference (b - a): with few signers, these say more than an
            # interval from resampling them.
            groups = data["clusters"]
            result["diff_by_cluster"] = {
                str(int(c)): float(b[groups == c].mean() - a[groups == c].mean())
                for c in np.unique(groups)
            }
        out.append(result)
    return out[0], out[1]


def clip_comparison(
    live_gap: Any, results: Path, pipeline: str, clips_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((clips_dir / "clips.json").read_text("utf-8"))
    clips = [Path(entry["y4m"]).stem for entry in manifest["clips"]]
    sign_of = {Path(entry["y4m"]).stem: entry["label"] for entry in manifest["clips"]}
    scores = {}
    for name in ("v2_final", "v3_final"):
        runs = [r for r in live_gap.runs_for(results, pipeline, name) if r.get("complete", True)]
        if not runs:
            raise FileNotFoundError(f"no {pipeline} runs of {name} in {results}")
        outcomes = [live_gap.clip_outcomes(run) for run in runs]
        scores[name] = {
            k: np.array([np.mean([o[c][k] for o in outcomes]) for c in clips])
            for k in ("top1", "top3")
        }
    out = []
    for k in ("top1", "top3"):
        a, b = scores["v2_final"][k], scores["v3_final"][k]
        result = paired(a, b, clusters=[sign_of[c] for c in clips])
        result["a_ci"], result["b_ci"] = stats.bootstrap_ci(a), stats.bootstrap_ci(b)
        result["runs"] = len(live_gap.runs_for(results, pipeline, "v3_final"))
        out.append(result)
    return out[0], out[1]


def pct(value: float) -> str:
    return f"{100 * value:.1f}"


def ci(pair: tuple[float, float]) -> str:
    return f"[{100 * pair[0]:.1f}, {100 * pair[1]:.1f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--v2-kaggle", type=Path, default=Path("results/final_selection/predictions/web_model.npz")
    )
    parser.add_argument("--v3-kaggle", type=Path, default=Path("results/v3_eval/web_model.npz"))
    parser.add_argument("--citizen-dir", type=Path, default=Path("results/asl_citizen/predictions"))
    parser.add_argument("--live-results", type=Path, default=Path("results/live_gap_v3"))
    parser.add_argument("--clips-dir", type=Path, default=Path("data/live_clips_v2"))
    parser.add_argument("--out", type=Path, default=Path("results/v2_vs_v3"))
    args = parser.parse_args(argv)

    top1: dict[str, dict[str, Any]] = {}
    top3: dict[str, dict[str, Any]] = {}
    top1["asl-signs test"], top3["asl-signs test"] = split_comparison(
        args.v2_kaggle, args.v3_kaggle
    )
    top1["ASL Citizen test"], top3["ASL Citizen test"] = split_comparison(
        args.citizen_dir / "v2_test.npz", args.citizen_dir / "v3_test.npz"
    )
    live_gap = load_live_gap()
    for pipeline, title in (("web", "test_clips_v2, browser"), ("python", "test_clips_v2, Python")):
        top1[title], top3[title] = clip_comparison(
            live_gap, args.live_results, pipeline, args.clips_dir
        )
    holm_family(top1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": "top-1, v3 - v2, Holm over the four comparisons; top-3 untested",
        "top1": top1,
        "top3": top3,
    }
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
    lines = [
        "# v2 vs v3",
        "",
        "Top-1 and top-3 (%) with 95% percentile bootstrap intervals; the difference is v3 - v2 "
        "on the same items, with its paired interval, the items each model got right alone, the "
        "sign-flip p value and the Holm-adjusted p over the four top-1 comparisons. Written by "
        "scripts/compare_v3.py.",
        "",
        "| Test set | Items | v2 top-1 | v3 top-1 | Difference | v3 better / worse | p (Holm) | "
        "v2 top-3 | v3 top-3 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for name, one in top1.items():
        three = top3[name]
        verdict = "significant" if one["significant"] else "not significant"
        lines.append(
            f"| {name} | {one['items']} | {pct(one['a'])} {ci(one['a_ci'])} | "
            f"{pct(one['b'])} {ci(one['b_ci'])} | {pct(one['diff'])} {ci(one['ci'])} | "
            f"{one['better']} / {one['worse']} | {one['p_holm']:.3f} ({verdict}) | "
            f"{pct(three['a'])} | {pct(three['b'])} |"
        )
        rows.append(
            {
                "test_set": name,
                "items": one["items"],
                "v2_top1": one["a"],
                "v3_top1": one["b"],
                "diff_top1": one["diff"],
                "diff_ci_low": one["ci"][0],
                "diff_ci_high": one["ci"][1],
                "p": one["p"],
                "p_holm": one["p_holm"],
                "significant": one["significant"],
                "v2_top3": three["a"],
                "v3_top3": three["b"],
            }
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", "utf-8")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
