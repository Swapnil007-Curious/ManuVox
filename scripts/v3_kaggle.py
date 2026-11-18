"""Steps of kaggle/train_v3.ipynb that are not training itself.

    python scripts/v3_kaggle.py locate --input /kaggle/input --unpack-dir /tmp
    python scripts/v3_kaggle.py build-dataset --landmarks DIR --kaggle-processed DIR --out DIR
    python scripts/v3_kaggle.py select --runs DIR --out FILE
    python scripts/v3_kaggle.py pack --work DIR --output ZIP

locate finds the inputs wherever Kaggle mounts them and prints shell assignments (DATA_ROOT,
LANDMARKS, PREVIOUS_RUNS) for the notebook to eval; what it found goes to stderr. The landmarks
are the folder of a manifest.csv with the extraction's columns, a COMPLETE marker and shards
(a shards/ folder, or shards.zip, which it unpacks under --unpack-dir), from the dataset
asl-citizen-landmarks-private or from the extraction notebook's output added as an input;
smoke-test outputs and unfinished runs are listed and skipped.

build-dataset preprocesses the ASL Citizen train and val videos of the mapped signs (never test)
and joins them to the asl-signs processed dir (edgevision.data.citizen.combine). select picks the
seed with the best validation score (best_val_select: the mean of the per-source val top-1),
lowest seed on a tie. pack zips the run summaries, checkpoints, the selection and the export, and
leaves out every ASL Citizen landmark and processed feature (they may not leave the private
notebook).
"""

import argparse
import csv
import json
import os
import shlex
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from edgevision.data import citizen
from edgevision.train.data import label_map_from_manifest, read_manifest

# Columns of the extraction's manifest.csv that the training side reads.
LANDMARK_COLUMNS = frozenset({"video_file", "participant", "split", "label", "rule", "shard"})
DATASET_SLUG = "asl-citizen-landmarks-private"
RUN_NAME = "e_hybrid_v3citizen"
# Folders never walked: the asl-signs parquet tree (94k files) and the landmark shards.
PRUNED = frozenset({"train_landmarks", "shards", ".git", "node_modules", "__pycache__"})

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


def walk(root: Path) -> list[tuple[Path, list[str], list[str]]]:
    """os.walk below root, in sorted order, skipping PRUNED folders."""
    found = []
    for directory, folders, files in os.walk(root):
        folders[:] = sorted(name for name in folders if name not in PRUNED)
        found.append((Path(directory), folders, sorted(files)))
    return found


def landmark_candidates(input_root: Path) -> list[dict[str, Any]]:
    """Every folder below input_root whose manifest.csv has the extraction's columns."""
    candidates = []
    for directory, _, files in walk(input_root):
        if "manifest.csv" not in files:
            continue
        try:
            with (directory / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                header = set(next(csv.reader(handle), []))
        except (OSError, UnicodeDecodeError):
            continue
        if not header >= LANDMARK_COLUMNS:
            continue
        candidates.append(
            {
                "root": directory,
                "complete": (directory / "COMPLETE").is_file(),
                "smoke": (directory / "SMOKE").exists() or directory.name.endswith("-smoke"),
                "shards": (directory / "shards").is_dir(),
                "shards_zip": (directory / "shards.zip").is_file(),
            }
        )
    return candidates


def describe(candidate: dict[str, Any]) -> str:
    if candidate["smoke"]:
        return "skipped: smoke-test output, never training data"
    if not candidate["complete"]:
        return "skipped: no COMPLETE marker, the extraction did not finish"
    if not (candidate["shards"] or candidate["shards_zip"]):
        return "skipped: no shards/ folder or shards.zip"
    return "usable" + (" (shards.zip)" if not candidate["shards"] else "")


def choose_landmarks(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The first usable candidate, the private dataset before notebook outputs."""
    usable = [c for c in candidates if describe(c).startswith("usable")]
    usable.sort(key=lambda c: (DATASET_SLUG not in c["root"].as_posix(), c["root"].as_posix()))
    return usable[0] if usable else None


def materialize(candidate: dict[str, Any], unpack_dir: Path) -> Path:
    """The candidate itself, or, when it holds shards.zip only, an unpacked copy in unpack_dir
    (outside /kaggle/working, so the notebook's saved output does not hold a second copy)."""
    root: Path = candidate["root"]
    if candidate["shards"]:
        return root
    target = unpack_dir / "asl-citizen-landmarks-input"
    if target.exists():
        shutil.rmtree(target)
    (target / "shards").mkdir(parents=True)
    for path in root.iterdir():
        if path.is_file() and path.name != "shards.zip":
            shutil.copy2(path, target / path.name)
    with zipfile.ZipFile(root / "shards.zip") as archive:
        for member in archive.infolist():
            name = Path(member.filename).name  # flat, whether or not the zip has a shards/ prefix
            if member.is_dir() or not name.startswith("shard_"):
                continue
            with archive.open(member) as source, (target / "shards" / name).open("wb") as sink:
                shutil.copyfileobj(source, sink)
    return target


def find_asl_signs(input_root: Path) -> Path | None:
    """The competition folder: train.csv next to train_landmarks/."""
    for directory, _, files in walk(input_root):
        if "train.csv" in files and (directory / "train_landmarks").is_dir():
            return directory
    return None


def find_previous_runs(input_root: Path) -> list[Path]:
    return [
        directory / RUN_NAME for directory, folders, _ in walk(input_root) if RUN_NAME in folders
    ]


def note(text: str) -> None:
    print(text, file=sys.stderr)


def locate(args: argparse.Namespace) -> int:
    input_root: Path = args.input
    if not input_root.is_dir():
        note(f"no input folder {input_root}")
    asl_signs = find_asl_signs(input_root) if input_root.is_dir() else None
    note(f"asl-signs (train.csv beside train_landmarks/): {asl_signs or 'none found'}")
    candidates = landmark_candidates(input_root) if input_root.is_dir() else []
    note(f"ASL Citizen manifest.csv files found: {len(candidates)}")
    for candidate in candidates:
        note(f"  {candidate['root']}: {describe(candidate)}")
    chosen = choose_landmarks(candidates)
    landmarks = materialize(chosen, args.unpack_dir) if chosen else None
    if chosen and landmarks != chosen["root"]:
        note(f"unpacked {chosen['root'] / 'shards.zip'} into {landmarks}")
    if chosen:
        manifest = (landmarks / "manifest.csv").read_text("utf-8").splitlines()
        note(f"using {landmarks}: {len(manifest) - 1} videos in manifest.csv")
    else:
        note(
            "no usable ASL Citizen landmarks: add the dataset asl-citizen-landmarks-private or the "
            "extraction notebook's merge-run version (the one that printed COMPLETE) as an input"
        )
    previous = find_previous_runs(input_root) if input_root.is_dir() else []
    for path in previous:
        note(f"earlier runs to carry over: {path}")
    print(f"DATA_ROOT={shlex.quote(str(asl_signs or ''))}")
    print(f"LANDMARKS={shlex.quote(str(landmarks or ''))}")
    print("PREVIOUS_RUNS=(" + " ".join(shlex.quote(str(path)) for path in previous) + ")")
    return 0


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
    finder = commands.add_parser("locate")
    finder.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    finder.add_argument("--unpack-dir", type=Path, default=Path("/tmp"))
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
    handlers = {"locate": locate, "build-dataset": build_dataset, "select": select, "pack": pack}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
