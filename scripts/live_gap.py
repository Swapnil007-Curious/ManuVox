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
import json
import math
import sys
from pathlib import Path
from typing import Any

from edgevision.live.gap import load_configs, run_python, score_run

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
    return run_all(args) if args.command == "python" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
