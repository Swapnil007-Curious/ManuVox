"""Live-gap evaluation on the recorded test clips.

    python scripts/live_gap.py python   # Python Holistic pipeline, every config it can run
    python scripts/live_gap.py report   # results/live_gap.csv and results/live_gap.md

`python` reads data/live_clips/<clip>.holistic.npz (scripts/extract_holistic.py) and writes
results/live_gap/python/<config>.json. The browser runs come from web/scripts/live-gap.mjs
(results/live_gap/web/<config>.json). `report` scores every run it finds with the rule in
experiments/live_gap.json and writes one row per (pipeline, config). The clips are never used to
choose a setting or threshold: every value comes from features_spec.json and meta.json, which
are derived from the train and val splits.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from edgevision.evaluation import stats
from edgevision.live.gap import TOP_K, load_configs, run_python, score_run
from edgevision.web_export import expected_calibration_error

REPO = Path(__file__).resolve().parents[1]
PIPELINES = {"python": "Python + Holistic", "web": "Browser + Tasks"}


def run_all(args: argparse.Namespace) -> int:
    configs = load_configs(args.configs)
    out = args.results / "python"
    out.mkdir(parents=True, exist_ok=True)
    for config in configs["configs"]:
        if args.only and config["name"] not in args.only:
            continue
        if config["mirrored"]:
            print(f"{config['name']}: mirrored detection needs the browser pipeline; skipped")
            continue
        if config.get("build") == "head":
            print(f"{config['name']}: the shipped build runs in the browser only; skipped")
            continue
        run = run_python(configs, config, REPO / configs["clips_dir"], REPO)
        (out / f"{config['name']}.json").write_text(json.dumps(run) + "\n", encoding="utf-8")
        score = score_run(run)
        print(
            f"{config['name']}: clip top-1 {score['clip_top1']:.2f} "
            f"top-3 {score['clip_top3']:.2f}; "
            f"{score['segments']} segments, top-1 {score['segment_top1']:.2f} top-3 "
            f"{score['segment_top3']:.2f}, coverage {score['coverage']:.2f}",
            flush=True,
        )
    return 0


def fmt(value: Any, digits: int = 2) -> str:
    if isinstance(value, float):
        return "-" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def runs_for(results: Path, pipeline: str, name: str) -> list[dict[str, Any]]:
    """The run file of a config, plus repeats named <config>.r<N>.json."""
    folder = results / pipeline
    paths = sorted(folder.glob(f"{name}.json")) + sorted(folder.glob(f"{name}.r*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def segment_fps(run: dict[str, Any]) -> float:
    """Frames per second inside segments: the rate at which frames reached the segmenter."""
    frames = seconds = 0.0
    for clip in run["clips"]:
        for segment in clip["segments"]:
            duration = (segment["end_ms"] - segment["start_ms"]) / 1000.0
            if segment.get("frames") and duration > 0:
                frames += segment["frames"] - 1
                seconds += duration
    return frames / seconds if seconds else float("nan")


def mean_score(runs: list[dict[str, Any]], ambiguous: list[str]) -> dict[str, Any]:
    """Scores averaged over repeats: all clips, and the clips whose file is not a duplicate."""
    scores = [score_run(run) for run in runs]
    clean = [score_run(run, exclude=ambiguous) for run in runs]
    keys = ("clip_top1", "clip_top3", "segment_top1", "segment_top3", "coverage")
    merged: dict[str, Any] = {key: sum(s[key] for s in scores) / len(scores) for key in keys}
    merged["selective_accuracy"] = sum(s["selective_accuracy"] for s in scores) / len(scores)
    merged["clean_clip_top1"] = sum(s["clip_top1"] for s in clean) / len(clean)
    merged["clean_clip_top3"] = sum(s["clip_top3"] for s in clean) / len(clean)
    merged["clean_clips"] = clean[0]["clips"]
    for key in ("clips", "segments", "discarded"):
        merged[key] = sum(s[key] for s in scores) / len(scores)
    merged["fps"] = sum(segment_fps(run) for run in runs) / len(runs)
    merged["runs"] = len(scores)
    merged["per_clip"] = scores[0]["per_clip"]
    return merged


COLUMNS = (
    "pipeline",
    "config",
    "group",
    "title",
    "clips",
    "clip_top1",
    "clip_top3",
    "clean_clips",
    "clean_clip_top1",
    "clean_clip_top3",
    "segments",
    "segment_top1",
    "segment_top3",
    "coverage",
    "selective_accuracy",
    "discarded",
    "fps",
    "runs",
)
TABLE_HEADER = (
    "| Config | Change | Clip top-1 | Clip top-3 | Unambiguous top-1 | Unambiguous top-3 | "
    "Segments | Seg top-1 | Seg top-3 | Coverage | Selective acc. | fps | Runs |"
)


def setup_lines(configs: dict[str, Any], duplicates: list[list[str]], clean: int) -> list[str]:
    pairs = ", ".join(" = ".join(f"`{name}.mp4`" for name in group) for group in duplicates)
    scoring = configs["scoring"]
    return [
        "# Live-gap evaluation",
        "",
        "Generated by `scripts/live_gap.py report` from the runs under `results/live_gap/`.",
        "",
        "## Setup",
        "",
        "- Clips: the 10 recordings in `test_clips/` (one signer, file name = sign, several",
        "  repetitions per clip). They are screen recordings of the first demo's page, so",
        "  `scripts/prepare_live_clips.py` cut out the camera preview, removed the drawn landmark",
        "  overlay by inpainting and flipped the frames back to the unmirrored camera view",
        "  (640 x 480, 30 fps .y4m). Both pipelines read those same frames.",
        f"- Duplicates: {pairs or 'none'} are byte-identical files, so one clip of each pair",
        "  necessarily carries the wrong label and the best possible clip top-1 on all 10 clips is",
        f"  0.80. The unambiguous columns leave out every duplicated file (n = {clean} clips).",
        "- Python + Holistic: legacy MediaPipe Holistic 0.10.21 (the tracker asl-signs was built",
        "  with) on every frame at 30 fps, then `edgevision.live.pipeline`.",
        "- Browser + Tasks: the web app in the installed Chrome on the target laptop (i3-1115G4),",
        "  MediaPipe Tasks HandLandmarker and PoseLandmarker lite on the GPU delegate, the clip as",
        "  the fake camera (`--use-file-for-fake-video-capture`), processed in real time at the",
        "  rate the laptop sustains (fps = frames per second inside segments).",
        "- `c0_head_build` is the shipped demo itself (commit 3fbafa3, main thread); every other",
        "  config runs the current code with the settings of `experiments/live_gap.json`. The",
        "  cumulative rows add one fix at a time; the leave-one-out rows undo one fix of the final",
        "  pipeline.",
        f"- Scoring (fixed before any run): {scoring['clip']}; {scoring['segment']}.",
        f"  {scoring['accepted']}.",
        "- Nothing was tuned on these clips: the aspect constant and the segmenter thresholds come",
        "  from the train and val splits (`scripts/calibrate_live.py`), the temperature and the",
        "  confidence thresholds from val (`scripts/export_web.py`). With 10 clips (6 unambiguous)",
        "  one clip moves clip top-1 by 0.10, so differences of one or two clips are noise.",
        "",
        "## Results",
        "",
        "Clip = clip-level, Seg = segment-level (n = classified segments). Coverage and selective",
        "accuracy use the config's decision rule. Browser runs happen in real time, so frame drops",
        "differ between runs; rows with Runs > 1 are means over repeats. The Python pipeline is",
        "deterministic. `c5_segmenter` and `c6_final` differ only in the decision rule",
        "(temperature scaling does not change a segment's top-1), so in the browser their gap",
        "is run-to-run noise.",
        "",
    ]


def report(args: argparse.Namespace) -> int:
    configs = load_configs(args.configs)
    manifest = json.loads((REPO / configs["clips_dir"] / "clips.json").read_text(encoding="utf-8"))
    duplicates: list[list[str]] = list(manifest["duplicate_groups"])
    ambiguous = sorted({name for group in duplicates for name in group})
    meta = json.loads((REPO / configs["models"]["v2"] / "meta.json").read_text(encoding="utf-8"))
    labels: list[str] = meta["labels"]
    rows: list[dict[str, Any]] = []
    per_clip: dict[tuple[str, str], dict[str, Any]] = {}
    for pipeline in PIPELINES:
        for config in configs["configs"]:
            runs = runs_for(args.results, pipeline, config["name"])
            if not runs:
                continue
            score = mean_score(runs, ambiguous)
            per_clip[(pipeline, config["name"])] = score["per_clip"]
            base = {"pipeline": pipeline, "config": config["name"]}
            base |= {"group": config["group"], "title": config["title"]}
            rows.append(base | {key: score[key] for key in COLUMNS if key in score})
    if not rows:
        print(f"no runs under {args.results}", file=sys.stderr)
        return 1
    csv_lines = [",".join(COLUMNS)]
    for row in rows:
        cells = (json.dumps(row[c]) if isinstance(row[c], str) else fmt(row[c], 4) for c in COLUMNS)
        csv_lines.append(",".join(cells))
    args.csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    lines = setup_lines(configs, duplicates, len(manifest["clips"]) - len(ambiguous))
    for pipeline, title in PIPELINES.items():
        subset = [row for row in rows if row["pipeline"] == pipeline]
        if not subset:
            continue
        lines += [f"### {title}", "", TABLE_HEADER, "|" + " --- |" * 13]
        for row in subset:
            values = [
                f"`{row['config']}`",
                row["title"],
                *(fmt(row[key]) for key in ("clip_top1", "clip_top3")),
                *(fmt(row[key]) for key in ("clean_clip_top1", "clean_clip_top3")),
                fmt(row["segments"], 0),
                *(fmt(row[key]) for key in ("segment_top1", "segment_top3")),
                *(fmt(row[key]) for key in ("coverage", "selective_accuracy")),
                fmt(row["fps"], 1),
                str(row["runs"]),
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    shown = [key for key in per_clip if key[1] in ("c0_head_build", "c0_as_shipped", "c6_final")]
    if shown:
        heading = " | ".join(f"{PIPELINES[pipeline]}, `{name}`" for pipeline, name in shown)
        lines += ["## Per clip", "", f"| Clip | {heading} |", "| --- |" + " --- |" * len(shown)]
        for entry in manifest["clips"]:
            name = entry["label"]
            cells = []
            for key in shown:
                item = per_clip[key].get(name, {"segments": 0, "prediction": None})
                guess = "no segment" if item["prediction"] is None else labels[item["prediction"]]
                cells.append(f"{guess} ({item['segments']} seg.)")
            twin = " (duplicate file)" if name in ambiguous else ""
            lines.append(f"| {name}{twin} | " + " | ".join(cells) + " |")
        lines.append("")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.csv} and {args.markdown} ({len(rows)} rows)")
    return 0


def clip_outcomes(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per clip of one run: top-1 / top-3 hit (0/1), predicted label index and the top 3."""
    out = {}
    for clip in run["clips"]:
        label = int(clip["label_index"])
        classified = [
            s["probabilities"] for s in clip["segments"] if s["probabilities"] is not None
        ]
        if classified:
            mean = np.mean(np.asarray(classified, dtype=np.float64), axis=0)
            top = [int(i) for i in np.argsort(-mean, kind="stable")[:TOP_K]]
            out[clip["clip"]] = {
                "top1": float(top[0] == label),
                "top3": float(label in top),
                "prediction": top[0],
                "top3_labels": top,
            }
        else:
            out[clip["clip"]] = {"top1": 0.0, "top3": 0.0, "prediction": None, "top3_labels": []}
    return out


def wrong_rejected(run: dict[str, Any]) -> float:
    """Share of wrong classified segments (top-1 not the clip's label) below the run's threshold."""
    threshold = float(run["threshold"])
    wrong = rejected = 0
    for clip in run["clips"]:
        label = int(clip["label_index"])
        for s in clip["segments"]:
            p = s["probabilities"]
            if p is None or int(np.argmax(p)) == label:
                continue
            wrong += 1
            rejected += max(p) < threshold
    return rejected / wrong if wrong else float("nan")


def report_v2(args: argparse.Namespace, configs: dict[str, Any]) -> int:
    """The v2 report: CIs over clips, Holm-corrected comparisons, per-sign and confusion tables."""
    analysis = configs["analysis"]
    manifest = json.loads((REPO / configs["clips_dir"] / "clips.json").read_text(encoding="utf-8"))
    clips = [Path(entry["y4m"]).stem for entry in manifest["clips"]]
    label_of = {Path(entry["y4m"]).stem: entry["label"] for entry in manifest["clips"]}
    meta = json.loads((REPO / configs["models"]["v2"] / "meta.json").read_text(encoding="utf-8"))
    labels: list[str] = meta["labels"]
    per_class_path = REPO / "results" / "test_analysis" / "per_class.csv"
    test_acc: dict[str, float] = {}
    if per_class_path.exists():
        with per_class_path.open(encoding="utf-8") as handle:
            test_acc = {r["label"]: float(r["top1"]) for r in csv.DictReader(handle)}
    rows: list[dict[str, Any]] = []
    scores: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    outcomes: dict[tuple[str, str], list[dict[str, dict[str, Any]]]] = {}
    run_files: dict[tuple[str, str], list[dict[str, Any]]] = {}
    # Sensitivity analysis (not the pre-specified one): the two takes of a sign are one cluster.
    signs = sorted({label_of[c] for c in clips})
    groups = [[i for i, c in enumerate(clips) if label_of[c] == s] for s in signs]

    def by_sign(values: np.ndarray) -> np.ndarray:
        return np.array([values[g].mean() for g in groups])

    for pipeline in PIPELINES:
        for config in configs["configs"]:
            runs = [
                r
                for r in runs_for(args.results, pipeline, config["name"])
                if r.get("complete", True)
            ]
            if not runs:
                continue
            per_run = [clip_outcomes(run) for run in runs]
            missing = [c for c in clips if any(c not in o for o in per_run)]
            if missing:
                print(
                    f"{pipeline} {config['name']}: missing clips {missing}; skipped",
                    file=sys.stderr,
                )
                continue
            top1 = np.array([np.mean([o[c]["top1"] for o in per_run]) for c in clips])
            top3 = np.array([np.mean([o[c]["top3"] for o in per_run]) for c in clips])
            scores[(pipeline, config["name"])] = {"top1": top1, "top3": top3}
            outcomes[(pipeline, config["name"])] = per_run
            run_files[(pipeline, config["name"])] = runs
            seg = [score_run(run) for run in runs]
            ci1, ci3 = stats.bootstrap_ci(top1), stats.bootstrap_ci(top3)
            ci1s = stats.bootstrap_ci(by_sign(top1))
            run_top1 = [float(np.mean([o[c]["top1"] for c in clips])) for o in per_run]
            no_segment = [sum(o[c]["prediction"] is None for c in clips) for o in per_run]
            rows.append(
                {
                    "pipeline": pipeline,
                    "config": config["name"],
                    "group": config["group"],
                    "title": config["title"],
                    "base": config.get("base", ""),
                    "clips": len(clips),
                    "runs": len(runs),
                    "clip_top1": float(top1.mean()),
                    "clip_top1_ci_low": ci1[0],
                    "clip_top1_ci_high": ci1[1],
                    "clip_top3": float(top3.mean()),
                    "clip_top3_ci_low": ci3[0],
                    "clip_top3_ci_high": ci3[1],
                    "clip_top1_ci_low_sign": ci1s[0],
                    "clip_top1_ci_high_sign": ci1s[1],
                    "run_top1": run_top1,
                    "run_top1_sd": float(np.std(run_top1, ddof=1)) if len(run_top1) > 1 else 0.0,
                    "no_segment_clips": float(np.mean(no_segment)),
                    "segments_per_run": float(np.mean([s["segments"] for s in seg])),
                    "segment_top1": float(np.mean([s["segment_top1"] for s in seg])),
                    "segment_top3": float(np.mean([s["segment_top3"] for s in seg])),
                    "coverage": float(np.mean([s["coverage"] for s in seg])),
                    "selective_accuracy": float(np.nanmean([s["selective_accuracy"] for s in seg])),
                    "fps": float(np.mean([segment_fps(run) for run in runs])),
                    "wrong_rejected": float(np.nanmean([wrong_rejected(run) for run in runs])),
                    "segment_ms_median": float(
                        np.median(
                            [
                                s["end_ms"] - s["start_ms"]
                                for run in runs
                                for clip in run["clips"]
                                for s in clip["segments"]
                                if s["probabilities"] is not None
                            ]
                            or [float("nan")]
                        )
                    ),
                }
            )
    if not rows:
        print(f"no complete runs under {args.results}", file=sys.stderr)
        return 1
    comparisons: list[dict[str, Any]] = []
    for pipeline in PIPELINES:
        for metric in ("top1", "top3"):
            family = []
            for config in configs["configs"]:
                base = config.get("base")
                a, b = (pipeline, config["name"]), (pipeline, base)
                if not base or a not in scores or b not in scores:
                    continue
                d = scores[a][metric] - scores[b][metric]
                low, high = stats.paired_bootstrap_ci(scores[a][metric], scores[b][metric])
                ds = by_sign(d)
                low_s, high_s = stats.bootstrap_ci(ds)
                family.append(
                    {
                        "pipeline": pipeline,
                        "metric": metric,
                        "config": config["name"],
                        "base": base,
                        "group": config["group"],
                        "title": config["title"],
                        "diff": float(d.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "clips_better": int((d > 0).sum()),
                        "clips_worse": int((d < 0).sum()),
                        "p": stats.sign_flip_p(d),
                        "ci_low_sign": low_s,
                        "ci_high_sign": high_s,
                        "p_sign": stats.sign_flip_p(ds),
                    }
                )
            for row, adjusted in zip(family, stats.holm([r["p"] for r in family]), strict=True):
                row["p_holm"] = adjusted
                row["significant"] = adjusted < 0.05
            for row, adjusted in zip(
                family, stats.holm([r["p_sign"] for r in family]), strict=True
            ):
                row["p_holm_sign"] = adjusted
                row["significant_sign"] = adjusted < 0.05
            comparisons += family
    # Browser records in which the tracker processed fewer than half of the clip's frames (a stall
    # of the real-time pipeline). They are kept as measured, as the pre-specified scoring says; the
    # count is reported. The shipped build does not record its frame count.
    source_frames = {Path(e["y4m"]).stem: e["source_frames"] for e in manifest["clips"]}
    stalled = total_records = 0
    for (pipeline, name), runs in run_files.items():
        if pipeline != "web" or name == "c0_head_build":
            continue
        for run in runs:
            for clip in run["clips"]:
                total_records += 1
                stalled += (clip["status"].get("frames") or 0) < 0.5 * source_frames[clip["clip"]]
    # Python against browser on the same clips (descriptive; not part of the pre-specified family).
    pipeline_diffs = []
    for name in ("c4_time", "c6_final", "x_presence"):
        a, b = ("web", name), ("python", name)
        if a in scores and b in scores:
            d = scores[a]["top1"] - scores[b]["top1"]
            low_s, high_s = stats.bootstrap_ci(by_sign(d))
            pipeline_diffs.append(
                {
                    "config": name,
                    "diff": float(d.mean()),
                    "ci_low_sign": low_s,
                    "ci_high_sign": high_s,
                    "clips_web_better": int((d > 0).sum()),
                    "clips_web_worse": int((d < 0).sum()),
                }
            )
    # Calibration and abstention on the live segments (pooled over runs).
    live_calibration = []
    for (pipeline, name), runs in run_files.items():
        if name not in ("c6_final", "x_presence"):
            continue
        conf, right = [], []
        for run in runs:
            for clip in run["clips"]:
                label = int(clip["label_index"])
                for s in clip["segments"]:
                    if s["probabilities"] is not None:
                        conf.append(max(s["probabilities"]))
                        right.append(int(np.argmax(s["probabilities"])) == label)
        c, r = np.asarray(conf), np.asarray(right)
        accepted = c >= float(runs[0]["threshold"])
        live_calibration.append(
            {
                "pipeline": pipeline,
                "config": name,
                "segments": int(c.size),
                "ece": expected_calibration_error(c, r),
                "mean_confidence": float(c.mean()),
                "accuracy": float(r.mean()),
                "coverage": float(accepted.mean()),
                "selective_accuracy": float(r[accepted].mean()),
                "net_score": float((2 * (r & accepted).sum() - accepted.sum()) / c.size),
                "correct_rejected": float((r & ~accepted).sum() / max(1, r.sum())),
                "wrong_rejected": float((~r & ~accepted).sum() / max(1, (~r).sum())),
            }
        )
    per_sign: list[dict[str, Any]] = []
    confusions: list[dict[str, Any]] = []
    for pipeline in PIPELINES:
        key = (pipeline, "c6_final")
        if key not in outcomes:
            continue
        for sign in signs:
            takes = [c for c in clips if label_of[c] == sign]
            preds = [
                labels[o[c]["prediction"]] if o[c]["prediction"] is not None else "(none)"
                for c in takes
                for o in outcomes[key]
            ]
            per_sign.append(
                {
                    "pipeline": pipeline,
                    "sign": sign,
                    "top1": float(np.mean([scores[key]["top1"][clips.index(c)] for c in takes])),
                    "top3": float(np.mean([scores[key]["top3"][clips.index(c)] for c in takes])),
                    "predictions": ", ".join(
                        f"{p} x{preds.count(p)}"
                        for p in sorted(set(preds), key=lambda p: -preds.count(p))
                    ),
                    "test_top1": test_acc.get(sign, float("nan")),
                }
            )
        pairs: dict[tuple[str, str], int] = {}
        for o in outcomes[key]:
            for c in clips:
                p = o[c]["prediction"]
                guess = labels[p] if p is not None else "(no segment)"
                if guess != label_of[c]:
                    pairs[(label_of[c], guess)] = pairs.get((label_of[c], guess), 0) + 1
        runs_n = len(outcomes[key])
        for (true, guess), count in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])):
            confusions.append(
                {
                    "pipeline": pipeline,
                    "sign": true,
                    "predicted": guess,
                    "count": count,
                    "out_of": 2 * runs_n,
                }
            )
    columns = [k for k in rows[0] if not isinstance(rows[0][k], list)]  # lists: summary.json only
    lines_csv = [",".join(columns)]
    for row in rows:
        lines_csv.append(
            ",".join(
                json.dumps(row[c]) if isinstance(row[c], str) else fmt(row[c], 4) for c in columns
            )
        )
    args.csv.write_text("\n".join(lines_csv) + "\n", encoding="utf-8")
    summary = {
        "analysis": analysis,
        "clips": len(clips),
        "signs": len(signs),
        "rows": rows,
        "comparisons": comparisons,
        "stalled_records": stalled,
        "browser_records": total_records,
        "pipeline_diffs": pipeline_diffs,
        "live_calibration": live_calibration,
        "per_sign": per_sign,
        "confusions": confusions,
    }
    (args.results / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )

    def ci(row: dict[str, Any], m: str) -> str:
        return f"{fmt(row[m])} [{fmt(row[m + '_ci_low'])}, {fmt(row[m + '_ci_high'])}]"

    md = [
        "# Live-gap evaluation v2",
        "",
        "Generated by `scripts/live_gap.py report --configs experiments/live_gap_v2.json`"
        " from the runs",
        "under `results/live_gap_v2/`. The 10-clip pilot is in `results/live_gap.md`.",
        "",
        "## Setup",
        "",
        f"- Clips: {len(clips)} webcam recordings in `test_clips_v2/` "
        f"({len(signs)} signs x 2 takes, one",
        "  signer, the author), recorded unmirrored at 1280 x 720 and about 15 fps; converted with",
        "  `scripts/prepare_live_clips.py --camera` to 640 x 360 .y4m at the recording's"
        " frame rate.",
        "  Both pipelines read the same frames. No clip is a duplicate of another (SHA-256).",
        "- Python + Holistic and Browser + Tasks: as in the pilot (`results/live_gap.md`);"
        " the browser",
        "  runs in the installed Chrome on the target laptop with the GPU delegate,"
        " one browser per clip",
        "  running every config in a seeded random order (`web/scripts/live-gap.mjs --per-clip`).",
        f"- Scoring: {configs['scoring']['clip']}.",
        "- Nothing was tuned on these clips (thresholds and temperature from val). Analysis, fixed",
        "  before any run (`experiments/live_gap_v2.json`, `analysis`):",
        *[f"  - {k.replace('_', ' ')}: {v}" for k, v in analysis.items()],
        "",
        "## Results",
        "",
        "Clip top-1 and top-3 with 95% bootstrap intervals over clips; browser rows are means over",
        "runs (a clip's score is its mean over the runs). Segment metrics are means over runs.",
        "",
    ]
    for pipeline, title in PIPELINES.items():
        subset = [r for r in rows if r["pipeline"] == pipeline]
        if not subset:
            continue
        md += [
            f"### {title}",
            "",
            "| Config | Change | Clip top-1 [95% CI] | Clip top-3 [95% CI] | Segments/run"
            " | Seg top-1 | Coverage | Selective acc. | fps | Runs |",
            "|" + " --- |" * 10,
        ]
        for r in subset:
            md.append(
                f"| `{r['config']}` | {r['title']} | {ci(r, 'clip_top1')} | {ci(r, 'clip_top3')} | "
                f"{fmt(r['segments_per_run'], 1)} | {fmt(r['segment_top1'])} | "
                f"{fmt(r['coverage'])} | "
                f"{fmt(r['selective_accuracy'])} | {fmt(r['fps'], 1)} | {r['runs']} |"
            )
        md.append("")
    md += [
        "## Comparisons",
        "",
        "Each config against the config it changes (`base`); difference in clip score"
        " with a paired",
        "bootstrap interval, the clips that got better / worse, the sign-flip p-value and its Holm",
        "adjustment within the pipeline and metric (the pre-specified analysis; intervals are not",
        "adjusted for multiplicity). The last two columns are a sensitivity analysis added after",
        "review: the same test and interval with the two takes of a sign as one unit (25 signs).",
        "",
        "| Pipeline | Metric | Config | vs | Diff [95% CI] | Better / worse | p | p (Holm)"
        " | Significant | Diff CI, sign as unit | p (Holm), sign as unit |",
        "|" + " --- |" * 11,
    ]
    for c in comparisons:
        md.append(
            f"| {PIPELINES[c['pipeline']]} | {c['metric']} | `{c['config']}` | `{c['base']}` | "
            f"{c['diff']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}] | {c['clips_better']} / "
            f"{c['clips_worse']} | {c['p']:.3g} | {c['p_holm']:.3g} | "
            f"{'yes' if c['significant'] else 'no'} | "
            f"[{c['ci_low_sign']:+.2f}, {c['ci_high_sign']:+.2f}] | {c['p_holm_sign']:.3g} |"
        )
    md += [
        "",
        "## Per sign (`c6_final`)",
        "",
        "Top-1 / top-3 over the two takes (and runs); predictions with counts; the shipped model's",
        "top-1 on the same sign in the signer-independent test split"
        " (`results/test_analysis/per_class.csv`).",
        "",
        "| Pipeline | Sign | Live top-1 | Live top-3 | Predictions | Test-split top-1 |",
        "|" + " --- |" * 6,
    ]
    for s in per_sign:
        md.append(
            f"| {PIPELINES[s['pipeline']]} | {s['sign']} | {fmt(s['top1'])} | {fmt(s['top3'])} | "
            f"{s['predictions']} | {fmt(s['test_top1'])} |"
        )
    md += [
        "",
        "## Confusions (`c6_final`)",
        "",
        "Wrong clip-level predictions, counted over takes and runs.",
        "",
        "| Pipeline | Sign | Predicted as | Count |",
        "|" + " --- |" * 4,
    ]
    for c in confusions:
        md.append(
            f"| {PIPELINES[c['pipeline']]} | {c['sign']} | {c['predicted']} | "
            f"{c['count']} / {c['out_of']} |"
        )
    args.markdown.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        f"wrote {args.csv}, {args.markdown} and {args.results / 'summary.json'} ({len(rows)} rows)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("python", "report"))
    parser.add_argument("--configs", type=Path, default=REPO / "experiments" / "live_gap.json")
    parser.add_argument("--results", type=Path, default=REPO / "results" / "live_gap")
    parser.add_argument("--csv", type=Path, default=REPO / "results" / "live_gap.csv")
    parser.add_argument("--markdown", type=Path, default=REPO / "results" / "live_gap.md")
    parser.add_argument("--only", nargs="*", help="run only these configs")
    args = parser.parse_args(argv)
    if args.command == "python":
        return run_all(args)
    configs = load_configs(args.configs)
    return report_v2(args, configs) if "analysis" in configs else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
