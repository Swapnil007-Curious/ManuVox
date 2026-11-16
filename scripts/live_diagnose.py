"""Diagnose web-demo recordings (the "Download recordings" JSON of Record test signs mode).

Replays each recording's raw landmarks through Python preprocessing, checks that the app's
features match (max |diff| <= 1e-4), and evaluates the exported model with no aspect correction,
with the app's recorded correction and with k = r_train / r_live computed offline per recording
session. Writes results/live/diagnosis_<timestamp>.json and .md. Exits 1 when parity fails.

Keep the recordings themselves under data/live/, which git ignores.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from edgevision.live.diagnose import (
    PARITY_TEXT,
    PARITY_TOLERANCE,
    diagnose,
    load_model,
    render_markdown,
    training_reference,
)
from edgevision.live.recordings import load_recordings
from edgevision.web_export import sha256_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, action="append", required=True)
    parser.add_argument("--meta", type=Path, default=Path("web/public/model/meta.json"))
    parser.add_argument("--model", type=Path, default=Path("web/public/model/model.onnx"))
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="processed dir of the model's training data (lengths, presence, aspect_stats)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/live"))
    return parser.parse_args(argv)


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    files = [(str(path), load_recordings(path)) for path in args.recordings]
    bundle = load_model(args.meta, args.model, sha256_file(args.model))
    reference = training_reference(args.processed_dir, bundle.meta)
    report = diagnose(files, bundle, reference)
    created = datetime.now(UTC)
    report = {
        "created_at": created.isoformat(timespec="seconds"),
        "inputs": {
            "recordings": [str(path) for path in args.recordings],
            "meta": str(args.meta),
            "model": str(args.model),
            "processed_dir": str(args.processed_dir),
        },
        **report,
    }
    stamp = created.astimezone().strftime("%Y%m%d-%H%M%S")  # local time, as in the app's downloads
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"diagnosis_{stamp}.json"
    md_path = args.output_dir / f"diagnosis_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")

    parity = report["parity"]
    print(f"{report['recordings']['count']} recordings, model {report['model']['run']}")
    print(
        f"parity: {parity['checked']} segments, max |diff| {parity['max_abs_diff']} "
        f"(tolerance {PARITY_TOLERANCE}) -> {PARITY_TEXT[parity['passed']]}"
    )
    for session in report["sessions"]:
        print(
            f"session {Path(session['file']).name}#{session['session']}: r_live "
            f"{session['r_live']}, r_train {report['training']['r_train']}, k offline "
            f"{session['k_offline']:.3f}, app k {session['app_k']}"
        )
    for name, block in report["conditions"].items():
        common = ", ".join(
            f"{label} {count}" for label, count in block["prediction_distribution"][:3]
        )
        print(
            f"{name:8s} top-1 {percent(block['top1'])}, top-5 {percent(block['top5'])}, "
            f"evaluated {block['evaluated']}, most frequent: {common or 'none'}"
        )
    print(f"wrote {json_path} and {md_path}")
    return 1 if parity["passed"] is False else 0


if __name__ == "__main__":
    sys.exit(main())
