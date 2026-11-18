"""Evaluate an exported web model on the ASL Citizen official test split (mapped signs), once.

    python scripts/citizen_eval.py --landmarks data/asl_citizen/landmarks --name v2
    python scripts/citizen_eval.py --landmarks data/asl_citizen/landmarks --name v3 \\
        --model-dir web/models/v3-citizen

The landmarks are the private Kaggle dataset asl-citizen-landmarks-private, downloaded under
data/ (gitignored). The test videos are preprocessed exactly as training preprocesses asl-signs
(edgevision.data.citizen) and scored with the model's own temperature and threshold from
meta.json, both fitted on validation.

Before any test data is read, one entry is appended to results/asl_citizen/test_access_log.json.
A model (by sha256) already in the log is refused: the split is evaluated once per model, unless
--reaccess gives a reason, which is logged with the entry.

Writes results/asl_citizen/<name>_test.json and .md and <name>_test_per_sign.csv (aggregate
results, safe to commit) and results/asl_citizen/predictions/<name>_test.npz (per-video logits
for paired comparisons; *.npz is gitignored and holds no landmarks).
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from edgevision.data import citizen
from edgevision.evaluation.cross_dataset import summarize
from edgevision.evaluation.tables import append_access_log
from edgevision.layout import to_model_input
from edgevision.runtime import OrtModel

RESULTS = Path("results/asl_citizen")
ACCESS_LOG = RESULTS / "test_access_log.json"


def access_key(name: str, sha256: str) -> str:
    return f"asl-citizen-test:{name}:{sha256}"


def previous_access(sha256: str) -> list[str]:
    if not ACCESS_LOG.is_file():
        return []
    entries = json.loads(ACCESS_LOG.read_text("utf-8"))
    return [
        key
        for entry in entries
        for key in entry["runs_evaluated"]
        if key.startswith("asl-citizen-test:") and sha256 in key
    ]


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}"


def interval(pair: tuple[float, float]) -> str:
    return f"[{100 * pair[0]:.1f}, {100 * pair[1]:.1f}]"


def markdown(
    name: str, meta: dict[str, Any], result: dict[str, Any], report: dict[str, Any]
) -> str:
    lines = [
        f"# {name} on the ASL Citizen test split",
        "",
        f"Model `{meta['run']}` ({meta['variant']}, sha256 `{meta['model_sha256'][:12]}`), "
        f"{result['videos']} test videos of {result['signs']} mapped signs from "
        f"{result['signers']} signers; {result['excluded_videos']} more were left out by "
        f"preprocessing (too few hand frames) and {report['counts']['not_tracked']} could not be "
        "tracked. Written by scripts/citizen_eval.py.",
        "",
        "| Metric | Value | 95% CI (videos) | 95% CI (signers) |",
        "|---|---|---|---|",
        f"| Top-1 (%) | {percent(result['top1'])} | {interval(result['top1_ci'])} | "
        f"{interval(result['top1_ci_signer'])} |",
        f"| Top-3 (%) | {percent(result['top3'])} | {interval(result['top3_ci'])} | "
        f"{interval(result['top3_ci_signer'])} |",
        f"| Top-1, excluded videos counted wrong (%) | {percent(result['top1_all_videos'])} | | |",
        f"| ECE, raw softmax | {result['ece_raw']:.3f} | [{result['ece_raw_ci'][0]:.3f}, "
        f"{result['ece_raw_ci'][1]:.3f}] | |",
        f"| ECE, temperature {result['temperature']:.3f} (val) | {result['ece']:.3f} | "
        f"[{result['ece_ci'][0]:.3f}, {result['ece_ci'][1]:.3f}] | |",
        f"| Coverage at threshold {result['threshold']:.3f} (%) | "
        f"{percent(result['coverage'])} | | |",
        f"| Selective accuracy (%) | {percent(result['selective_accuracy'])} | | |",
        "",
        f"Per-signer top-1 ranges from {percent(result['top1_by_signer_min'])}% to "
        f"{percent(result['top1_by_signer_max'])}%.",
        "",
        "| Mapping rule | Videos | Top-1 (%) | Top-3 (%) |",
        "|---|---|---|---|",
    ]
    for rule, values in result["by_rule"].items():
        lines.append(
            f"| {rule} | {values['videos']} | {percent(values['top1'])} | "
            f"{percent(values['top3'])} |"
        )
    lines += [
        "",
        "Variant and synonym mappings are AI-assisted and unconfirmed "
        "(src/edgevision/data/citizen_labels.py).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--landmarks", type=Path, default=Path("data/asl_citizen/landmarks"))
    parser.add_argument("--model-dir", type=Path, default=Path("web/public/model"))
    parser.add_argument("--name", required=True, help="short model name for the result files")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        help="where the preprocessed test features go (default data/asl_citizen/test_<pose set>)",
    )
    parser.add_argument("--reaccess", help="reason for evaluating an already evaluated model")
    args = parser.parse_args(argv)

    meta = json.loads((args.model_dir / "meta.json").read_text("utf-8"))
    sha256 = meta["model_sha256"]
    earlier = previous_access(sha256)
    if earlier and not args.reaccess:
        print(
            f"this model was already evaluated on the ASL Citizen test split ({earlier}); "
            "pass --reaccess with a reason to evaluate it again",
            file=sys.stderr,
        )
        return 1
    key = access_key(args.name, sha256)
    if args.reaccess:
        key += f" (reaccess: {args.reaccess})"
    append_access_log(ACCESS_LOG, [key])

    pose_indices = meta["preprocessing"]["pose_indices"]
    processed = args.processed_dir or Path(f"data/asl_citizen/test_pose{len(pose_indices)}")
    report = citizen.build_processed(
        args.landmarks,
        processed,
        meta["labels"],
        pose_indices=pose_indices,
        num_frames=meta["preprocessing"]["num_frames"],
        min_hand_frames=meta["preprocessing"]["min_hand_frames"],
        splits=("test",),
    )
    manifest = pd.read_csv(processed / "manifest.csv", keep_default_na=False)
    kept = manifest.loc[~manifest["excluded"]].set_index("sequence_id")
    with np.load(processed / "features.npz", allow_pickle=False) as data:
        x = to_model_input(data["X"], data["presence"], meta["input_layout"]["num_landmarks"])
        y = data["y"].astype(np.int64)
        sequence_id = data["sequence_id"].astype(np.int64)
    rows = kept.loc[sequence_id]
    model = OrtModel(args.model_dir / meta["model_file"])
    logits = np.concatenate([model.run(x[i : i + 512]) for i in range(0, len(x), 512)])

    result = summarize(
        logits,
        y,
        signers=rows["participant_id"].astype(str).to_numpy(),
        signs=rows["sign"].tolist(),
        rules=rows["rule"].tolist(),
        labels=meta["labels"],
        temperature=meta["calibration"]["temperature"],
        threshold=meta["calibration"]["threshold"],
        excluded=int(manifest["excluded"].sum()),
    )
    per_sign = result.pop("per_sign")
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": args.name,
        "model": {key: meta[key] for key in ("run", "variant", "model_sha256", "data_source")},
        "split": "asl-citizen test (official)",
        "preprocessing": {key: report[key] for key in ("rules", "splits", "counts")},
        **result,
    }
    (RESULTS / f"{args.name}_test.json").write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
    (RESULTS / f"{args.name}_test.md").write_text(
        markdown(args.name, meta, result, report), "utf-8"
    )
    with (RESULTS / f"{args.name}_test_per_sign.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sign[0]))
        writer.writeheader()
        writer.writerows(per_sign)
    predictions = RESULTS / "predictions"
    predictions.mkdir(exist_ok=True)
    np.savez_compressed(
        predictions / f"{args.name}_test.npz",
        sequence_id=sequence_id,
        participant_id=rows["participant_id"].to_numpy(dtype=np.int64),
        y=y,
        logits=logits,
        model_sha256=np.asarray(sha256),
    )
    print(
        f"{args.name}: top-1 {percent(result['top1'])}% {interval(result['top1_ci'])}, top-3 "
        f"{percent(result['top3'])}% {interval(result['top3_ci'])} on {result['videos']} videos; "
        f"ECE {result['ece']:.3f}; results in {RESULTS}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
