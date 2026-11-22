"""Write the README's results table from the result files (nothing typed by hand).

    python scripts/readme_results.py           # rewrite the table between the README markers
    python scripts/readme_results.py --check   # exit 1 if the README's table is out of date

Sources: results/v2_vs_v3.json (scripts/compare_v3.py), results/asl_citizen/{v2,v3}_test.json
(scripts/citizen_eval.py), results/teach_dex.json and results/teach_dex_v3.json
(scripts/teach_report.py) and results/perf/v2_vs_v3_classifier_latency.json
(scripts/compare_latency.py).
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START = "<!-- results:start (scripts/readme_results.py) -->"
END = "<!-- results:end -->"
EFFECT = "all 25 signs taught vs the model alone"


def load(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def pct(value: float, decimals: int = 1) -> str:
    return f"{100 * value:.{decimals}f}%"


def ci(pair: list[float], decimals: int = 1) -> str:
    return f"[{100 * pair[0]:.{decimals}f}, {100 * pair[1]:.{decimals}f}]"


def holm(p: float) -> str:
    return "Holm p < 0.001" if p < 0.001 else f"Holm p = {p:.3f}"


def table() -> str:
    cmp_ = load("results/v2_vs_v3.json")["top1"]
    lat = load("results/perf/v2_vs_v3_classifier_latency.json")["models"]
    teach = {"v2": load("results/teach_dex.json"), "v3": load("results/teach_dex_v3.json")}
    citizen = load("results/asl_citizen/v2_test.json")
    with (ROOT / "results/tables/selection/test_per_participant.csv").open(encoding="utf-8") as f:
        isl_signers = len(
            {
                r["participant_id"]
                for r in csv.DictReader(f)
                if r["experiment"] == "d_hybrid_v2feat_v2recipe"
            }
        )
    # v3's own clip top-1 on the take-2 clips in the live-gap runs, the Teach Dex baseline's check.
    from live_gap import clip_outcomes  # scripts/live_gap.py

    take2 = []
    for run in sorted((ROOT / "results/live_gap_v3/web").glob("v3_final*.json")):
        outcomes = clip_outcomes(load(str(run.relative_to(ROOT))))
        hits = [o["top1"] for clip, o in outcomes.items() if clip.endswith("_2")]
        take2.append(sum(hits) / len(hits))
    v3_take2 = sum(take2) / len(take2)
    clips = cmp_["test_clips_v2, browser"]["items"]
    runs = cmp_["test_clips_v2, browser"]["runs"]

    def split_row(label: str, key: str, source: str) -> str:
        r = cmp_[key]
        verdict = "significant" if r["significant"] else "not significant"
        return (
            f"| {label} | {r['items']:,} | {pct(r['a'])} {ci(r['a_ci'])} | "
            f"{pct(r['b'])} {ci(r['b_ci'])} | {100 * r['diff']:+.1f} pts, {holm(r['p_holm'])} "
            f"({verdict}) | {source} |"
        )

    def teach_cell(name: str) -> str:
        t = teach[name]
        c = t["comparisons"][EFFECT]
        verdict = "significant" if c["significant"] else "not significant"
        return (
            f"{pct(t['top1']['model'], 0)} → {pct(t['top1']['teach_dex'], 0)} "
            f"({100 * c['diff']:+.0f} pts {ci(c['ci'], 0)}, {holm(c['p_holm'])}, {verdict})"
        )

    def segments(name: str) -> str:
        return ", ".join(str(n) for n in teach[name]["segments_per_run"])

    def latency(name: str) -> str:
        one = lat[name]["configs"]["wasmx1"]
        return (
            f"{one['median_p50_ms']:.1f} / {one['median_p95_ms']:.1f} ms, "
            f"{lat[name]['bytes'] / 1e6:.1f} MB"
        )

    rows = [
        "| Test set | n | v2 (public default) | v3 (research only) | v3 - v2 | Source |",
        "|---|---|---|---|---|---|",
        split_row(
            f"Google ISLR signer-independent test ({isl_signers} unseen signers), top-1",
            "asl-signs test",
            "`results/v2_vs_v3.md`",
        ),
        split_row(
            f"ASL Citizen official test ({citizen['signs']} mapped signs, {citizen['signers']} "
            "unseen signers), top-1 ¹",
            "ASL Citizen test",
            "`results/asl_citizen/`",
        ),
        split_row(
            f"Live webcam clips, browser ({clips} clips, one signer), clip top-1 ²",
            "test_clips_v2, browser",
            "`results/v2_vs_v3.md`",
        ),
        split_row(
            "Live webcam clips, Python + Holistic, clip top-1",
            "test_clips_v2, Python",
            "`results/v2_vs_v3.md`",
        ),
    ]
    extra = [
        "",
        "| Separate results | v2 | v3 | Source |",
        "|---|---|---|---|",
        "| Teach Dex, personalised (take-2 clips, model alone → up to 3 examples per sign, "
        f"replays of take 1) ³ | {teach_cell('v2')} | compromised run (see ³): "
        f"{teach_cell('v3')} | `results/teach_dex.md`, `results/teach_dex_v3.md` |",
        f"| Classifier in Chrome on an i3-1115G4 (wasm, 1 thread), p50 / p95, file size | "
        f"{latency('v2')} | {latency('v3')} | `results/perf/v2_vs_v3_classifier_latency.json` |",
        "",
        "Each test set was evaluated once per model; the four top-1 comparisons are one Holm "
        "family (`scripts/compare_v3.py`). ¹ v3 trained on ASL Citizen's training signers, so its "
        "ASL Citizen score is in-domain and v2's is cross-dataset; the label mapping is "
        f"AI-assisted and unconfirmed by a signer. ² Browser scores are means over {runs} runs. "
        "³ One signer, taught with take 1 and tested on take 2 of the same sitting; never pooled "
        "with the rows above. **The v3 Teach Dex run is compromised:** its model-alone score on "
        f"take 2 ({pct(teach['v3']['top1']['model'], 0)}) is far below v3's own "
        f"{pct(v3_take2, 0)} on the same clips in the live-gap runs, and its classified segments "
        f"per run fell ({segments('v3')}, against {segments('v2')} with v2; the harness logs no "
        "frame rate), so its gain says nothing about teaching v3 and the run needs repeating. "
        "v3 is trained on ASL Citizen (research licence) and is not deployed.",
    ]
    return "\n".join([START, *rows, *extra, END])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print(f"README.md lacks the markers {START!r} ... {END!r}", file=sys.stderr)
        return 1
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    updated = head + table() + tail
    if args.check:
        if updated != text:
            print("README results table is out of date: run scripts/readme_results.py")
            return 1
        print("README results table is up to date")
        return 0
    README.write_text(updated, encoding="utf-8", newline="\n")
    print("README results table written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
