"""Summarise the Teach Dex record-then-test evaluation (web/tests/e2e/teach-eval.spec.ts).

    python scripts/teach_report.py

Reads results/teach_dex/web/run*.outcomes.json and writes results/teach_dex.json and .md. This is
a separate, personalised result: it is never pooled with the main live-gap or test-split numbers.

Unit: the 25 take-2 clips (one signer), each scored as the mean over runs of its clip top-1.
Conditions on the same segments: the model alone, Teach Dex with all 25 signs taught from take 1
(what the app showed), and a control with 13 signs taught. Comparisons against the model alone:
all-taught (the effect) and the control's untaught signs (harm to signs nobody taught), paired
sign-flip tests, Holm-adjusted over these two; the control's taught signs are reported without a
test. Intervals are percentile bootstraps over clips (10,000 resamples, seed 0).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from edgevision.evaluation import stats
from edgevision.evaluation.cross_dataset import holm_family, paired


def pct(value: float) -> str:
    return f"{100 * value:.0f}"


def ci(pair: tuple[float, float]) -> str:
    return f"[{100 * pair[0]:.0f}, {100 * pair[1]:.0f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, default=Path("results/teach_dex/web"))
    parser.add_argument("--out", type=Path, default=Path("results/teach_dex"))
    args = parser.parse_args(argv)
    runs = [
        json.loads(path.read_text("utf-8")) for path in sorted(args.runs.glob("run*.outcomes.json"))
    ]
    if not runs:
        raise SystemExit(f"no run*.outcomes.json in {args.runs}")
    clips = [clip["clip"] for clip in runs[0]["clips"]]
    by_run = [{clip["clip"]: clip for clip in run["clips"]} for run in runs]
    half_taught = np.array([by_run[0][c]["half_taught"] for c in clips])

    def score(condition: str, k: str = "top1") -> np.ndarray:
        return np.array([np.mean([run[c][condition][k] for run in by_run]) for c in clips])

    base, mine, half = score("base"), score("personalized"), score("half")
    base3, mine3 = score("base", "top3"), score("personalized", "top3")
    comparisons: dict[str, dict[str, Any]] = {
        "all 25 signs taught vs the model alone": paired(base, mine),
        "13 taught: the 12 untaught signs vs the model alone": paired(
            base[~half_taught], half[~half_taught]
        ),
    }
    holm_family(comparisons)
    descriptive = paired(base[half_taught], half[half_taught])
    teaching = [run["teaching"] for run in runs]
    examples = [sum(min(v["examples"], 3) for v in t.values()) for t in teaching]
    untaught = [sorted(label for label, v in t.items() if v["examples"] == 0) for t in teaching]
    badged = [sum(c["badged_segments"] for c in run["clips"]) for run in runs]
    segments = [sum(c["segments"] for c in run["clips"]) for run in runs]
    summary = {
        "unit": "take-2 clip (n = 25, one signer); clip score = mean over runs",
        "runs": len(runs),
        "model_sha256": runs[0]["model_sha256"],
        "params": runs[0]["params"],
        "store_modes": sorted({mode for run in runs for mode in run["store_modes"] if mode}),
        "examples_recorded_per_run": examples,
        "signs_without_examples_per_run": untaught,
        "badged_segments_per_run": badged,
        "segments_per_run": segments,
        "parity_max_abs_diff": max(run["parity_max_abs_diff"] for run in runs),
        "top1": {
            "model": float(base.mean()),
            "model_ci": stats.bootstrap_ci(base),
            "teach_dex": float(mine.mean()),
            "teach_dex_ci": stats.bootstrap_ci(mine),
            "control_13_taught": float(half.mean()),
        },
        "top3": {"model": float(base3.mean()), "teach_dex": float(mine3.mean())},
        "comparisons": comparisons,
        "control_taught_signs": descriptive,
        "half_taught_signs": runs[0]["half_taught_signs"],
        "per_clip": [
            {"clip": c, "model": float(b), "teach_dex": float(m), "control": float(h)}
            for c, b, m, h in zip(clips, base, mine, half, strict=True)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    first = comparisons["all 25 signs taught vs the model alone"]
    harm = comparisons["13 taught: the 12 untaught signs vs the model alone"]
    lines = [
        "# Teach Dex: personalised result (record-then-test)",
        "",
        "Separate from every main result. One signer: take 1 of test_clips_v2 taught each sign "
        "through the app's Teach panel, up to three examples from repeated playback of the one "
        "take-1 clip. Each take-1 clip had a "
        f"{runs[0].get('lead_in_s', 3.5):g} s still lead-in (its first frame repeated) so the "
        "panel's countdown did not cover the recorded sign, and the listening window spanned the "
        "clip; each example is the first segment after the countdown. Take 2 was then recognised "
        "with the examples on, through the final live pipeline (installed Chrome, GPU tracker), "
        f"{len(runs)} runs. Blend weight and offset come from the validation split "
        "(scripts/calibrate_teach.py), not from these clips. Written by scripts/teach_report.py.",
        "",
        "| Clip top-1 on take 2 (%) | Value | 95% CI |",
        "|---|---|---|",
        f"| Model alone | {pct(base.mean())} | {ci(stats.bootstrap_ci(base))} |",
        f"| Teach Dex, all 25 signs taught | {pct(mine.mean())} | {ci(stats.bootstrap_ci(mine))} |",
        f"| Control, 13 signs taught | {pct(half.mean())} | {ci(stats.bootstrap_ci(half))} |",
        "",
        "| Comparison (paired over clips) | Difference (points) | 95% CI | Better / worse | "
        "p (Holm) |",
        "|---|---|---|---|---|",
    ]
    for name, result in comparisons.items():
        verdict = "significant" if result["significant"] else "not significant"
        lines.append(
            f"| {name} | {100 * result['diff']:+.0f} | {ci(result['ci'])} | "
            f"{result['better']} / {result['worse']} | {result['p_holm']:.3f} ({verdict}) |"
        )
    lines.append(
        f"| 13 taught: the 13 taught signs vs the model alone (no test) | "
        f"{100 * descriptive['diff']:+.0f} | {ci(descriptive['ci'])} | "
        f"{descriptive['better']} / {descriptive['worse']} | |"
    )
    lines += [
        "",
        f"Top-3: model {pct(base3.mean())}%, Teach Dex {pct(mine3.mean())}%. Examples recorded per "
        f"run: {examples} of 75; signs left without examples: {untaught}. Segments badged "
        f'"Personalized for you": {badged} of {segments} per run. The Node recomputation '
        f"matched the app's decisions to {summary['parity_max_abs_diff']:.1e}.",
        "",
        "Limits: one signer, taught and tested on two takes recorded in one sitting, the test "
        "vocabulary equal to the taught signs (the all-taught condition is the best case), and "
        f"25 clips. The effect is {100 * first['diff']:+.0f} points; the untaught-sign change "
        f"in the control is {100 * harm['diff']:+.0f} points.",
        "",
    ]
    args.out.with_suffix(".md").write_text("\n".join(lines), "utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
