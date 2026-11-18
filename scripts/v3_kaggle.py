"""Steps of kaggle/train_v3.ipynb that are not training itself.

    python scripts/v3_kaggle.py build-dataset --landmarks DIR --kaggle-processed DIR --out DIR
    python scripts/v3_kaggle.py select --runs DIR --out FILE
    python scripts/v3_kaggle.py pack --work DIR --output ZIP

build-dataset preprocesses the ASL Citizen train and val videos of the mapped signs (never test)
and joins them to the asl-signs processed dir (edgevision.data.citizen.combine). select picks the
seed with the best validation score (best_val_select: the mean of the per-source val top-1),
lowest seed on a tie. pack zips the run summaries, checkpoints, the selection and the export, and
leaves out every ASL Citizen landmark and processed feature (they may not leave the private
notebook).
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

from edgevision.data import citizen
from edgevision.train.data import label_map_from_manifest, read_manifest

# Files under the work dir that pack includes (globs relative to it). Nothing derived from ASL
# Citizen data apart from the trained weights, their metrics and the dataset counts.
PACK_GLOBS = (
    "runs/*/seed_*/summary.json",
    "runs/*/seed_*/metrics.csv",
    "runs/*/seed_*/config.yaml",
    "runs/*/seed_*/label_map.json",
    "runs/*/seed_*/splits_signer_independent.json",
    "runs/*/seed_*/best.pt",
    "v3_export/selection.json",
    "v3_export/model/*",
    "v3_export/golden/*",
    "v3_export/work/*.onnx",
    "processed_v3/preprocess_report.json",
    "processed_v3_citizen/preprocess_report.json",
    "train_v3.log",
)
NEVER_PACKED = ("features.npz", "manifest.csv", "shard_")


def build_dataset(args: argparse.Namespace) -> int:
    kaggle_dir: Path = args.kaggle_processed
    labels_by_index = label_map_from_manifest(read_manifest(kaggle_dir))
    labels = [labels_by_index[index] for index in range(len(labels_by_index))]
    report = json.loads((kaggle_dir / "preprocess_report.json").read_text("utf-8"))
    pose_indices = report["landmarks"]["pose_indices"]
    citizen_dir = args.out.with_name(args.out.name + "_citizen")
    built = citizen.build_processed(
        args.landmarks,
        citizen_dir,
        labels,
        pose_indices=pose_indices,
        rules=tuple(args.rules),
        splits=("train", "val"),  # The ASL Citizen test split is evaluated locally, once.
    )
    print(f"ASL Citizen train/val: {json.dumps(built['counts'])}")
    combined = citizen.combine(kaggle_dir, citizen_dir, args.out)
    print(f"combined split counts: {json.dumps(combined['counts'])}")
    signers = {name: len(ids) for name, ids in combined["participants"].items()}
    print(f"signers: {signers}")
    return 0


def select(args: argparse.Namespace) -> int:
    summaries = []
    for path in sorted(args.runs.glob("seed_*/summary.json")):
        summary = json.loads(path.read_text("utf-8"))
        summaries.append((summary.get("best_val_select", summary["best_val_top1"]), path.parent))
    if not summaries:
        print(f"no finished runs under {args.runs}", file=sys.stderr)
        return 1
    best = max(summaries, key=lambda item: (item[0], -int(item[1].name.split("_")[1])))
    payload = {
        "rule": "highest best_val_select (mean of asl-signs and ASL Citizen val top-1), lowest "
        "seed on a tie; validation only",
        "chosen": best[1].name,
        "scores": {path.name: score for score, path in summaries},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(best[1])
    return 0


def pack(args: argparse.Namespace) -> int:
    work: Path = args.work
    files = sorted(
        {path for pattern in PACK_GLOBS for path in work.glob(pattern) if path.is_file()}
    )
    leaked = [path for path in files if any(token in path.name for token in NEVER_PACKED)]
    if leaked:
        print(f"refusing to pack data files: {leaked}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(work).as_posix())
    print(f"wrote {args.output} ({len(files)} files)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-dataset")
    build.add_argument("--landmarks", type=Path, required=True)
    build.add_argument("--kaggle-processed", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--rules", nargs="+", default=list(citizen.RULES), choices=citizen.RULES)
    chooser = commands.add_parser("select")
    chooser.add_argument("--runs", type=Path, required=True)
    chooser.add_argument("--out", type=Path, required=True)
    packer = commands.add_parser("pack")
    packer.add_argument("--work", type=Path, required=True)
    packer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    handlers = {"build-dataset": build_dataset, "select": select, "pack": pack}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
