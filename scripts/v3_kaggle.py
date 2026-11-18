"""Steps of kaggle/train_v3.ipynb that are not training itself.

    python scripts/v3_kaggle.py locate --input /kaggle/input --unpack-dir /tmp
    python scripts/v3_kaggle.py build-dataset --landmarks DIR --kaggle-processed DIR --out DIR
    python scripts/v3_kaggle.py select --runs DIR --out FILE
    python scripts/v3_kaggle.py pack --work DIR --output ZIP

locate finds the inputs wherever Kaggle mounts them and prints shell assignments (DATA_ROOT,
LANDMARKS, PREVIOUS_RUNS) for the notebook to eval; what it found goes to stderr. The landmarks
are the folder of a manifest.csv with the extraction's columns, from the dataset
asl-citizen-landmarks-private or from the extraction merge run's output added as an input, at
any depth. Its shard files may be in shards/, one folder further down, next to the manifest or
in a zip beside it (shards.zip), and the whole folder may itself be in a zip; when they are not
already in shards/ they are gathered (linked, copied or unzipped) under --unpack-dir. A folder
is used only when it has the COMPLETE marker, is not a smoke test and holds every shard its
manifest's tracked videos need; every folder found is listed with the reason it was used or
skipped.

build-dataset preprocesses the ASL Citizen train and val videos of the mapped signs (never test)
and joins them to the asl-signs processed dir (edgevision.data.citizen.combine). select picks the
seed with the best validation score (best_val_select: the mean of the per-source val top-1),
lowest seed on a tie. pack zips the run summaries, checkpoints, the selection and the export, and
leaves out every ASL Citizen landmark and processed feature (they may not leave the private
notebook).
"""

import argparse
import csv
import io
import json
import os
import re
import shlex
import shutil
import sys
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO

from edgevision.data import citizen
from edgevision.train.data import label_map_from_manifest, read_manifest

# Columns of the extraction's manifest.csv that the training side reads.
LANDMARK_COLUMNS = frozenset({"video_file", "participant", "split", "label", "rule", "shard"})
DATASET_SLUG = "asl-citizen-landmarks-private"
RUN_NAME = "e_hybrid_v3citizen"
# Folders never walked: the asl-signs parquet tree (94k files) and the landmark shards.
PRUNED = frozenset({"train_landmarks", "shards", ".git", "node_modules", "__pycache__"})
SHARD_NAME = re.compile(r"shard_(\d{5})\.npz")
GATHERED = "asl-citizen-landmarks-input"

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


def shard_index(name: str) -> int | None:
    match = SHARD_NAME.fullmatch(name)
    return int(match.group(1)) if match else None


def shard_file(index: int) -> str:
    return f"shard_{index:05d}.npz"


@dataclass
class Landmarks:
    """A folder holding an extraction's manifest.csv, and where its shard files are."""

    root: Path
    complete: bool
    smoke: bool
    videos: int
    needed: set[int]  # the shards holding the manifest's tracked videos
    files: dict[int, Path] = field(default_factory=dict)  # shard files on disk
    zipped: dict[int, tuple[Path, str]] = field(default_factory=dict)  # (zip, member)
    layout: str = ""

    @property
    def missing(self) -> list[int]:
        return sorted(self.needed - set(self.files) - set(self.zipped))

    def verdict(self) -> str:
        if self.smoke:
            return "skipped: smoke-test output, never training data"
        if not self.complete:
            return (
                "skipped: no COMPLETE marker, the extraction did not finish here (a part's "
                "output? add the merge run's output or the dataset instead)"
            )
        if not self.needed:
            return "skipped: the manifest lists no tracked videos"
        if self.missing:
            return (
                f"skipped: {len(self.missing)} of the {len(self.needed)} shards the manifest "
                f"needs are missing, e.g. {shard_file(self.missing[0])}"
            )
        return "usable"

    @property
    def usable(self) -> bool:
        return self.verdict() == "usable"

    @property
    def in_place(self) -> bool:
        """Every shard is already at root/shards/shard_NNNNN.npz, as the training code reads it."""
        return not self.zipped and all(
            path == self.root / "shards" / shard_file(index) for index, path in self.files.items()
        )

    def describe(self) -> str:
        return (
            f"{self.root}: {self.videos} videos, {len(self.needed)} shards needed, shard files: "
            f"{self.layout}; {self.verdict()}"
        )


def read_manifest_needs(handle: IO[str]) -> tuple[bool, int, set[int]]:
    """(has the extraction's columns, videos, shards of the tracked videos) of a manifest.csv."""
    reader = csv.DictReader(handle)
    if not set(reader.fieldnames or []) >= LANDMARK_COLUMNS:
        return False, 0, set()
    videos, needed = 0, set()
    for row in reader:
        videos += 1
        shard = (row.get("shard") or "").strip()
        if (row.get("status") or "ok") == "ok" and shard:
            needed.add(int(float(shard)))
    return True, videos, needed


def shard_locations(root: Path) -> tuple[dict[int, Path], dict[int, tuple[Path, str]], str]:
    """The shard files of a landmark folder: in shards/, one folder further down, next to the
    manifest, or in a zip beside it (shards.zip first). The first copy of each shard wins."""
    files: dict[int, Path] = {}
    zipped: dict[int, tuple[Path, str]] = {}
    layouts: list[str] = []

    def add(found: Iterable[tuple[str, Path | tuple[Path, str]]], where: str) -> None:
        added = 0
        for name, source in sorted(found, key=lambda item: item[0]):
            index = shard_index(name)
            if index is None or index in files or index in zipped:
                continue
            if isinstance(source, Path):
                files[index] = source
            else:
                zipped[index] = source
            added += 1
        if added:
            layouts.append(f"{added} in {where}")

    folder = root / "shards"
    if folder.is_dir():
        add(((path.name, path) for path in folder.glob("shard_*.npz")), "shards/")
        add(((path.name, path) for path in folder.glob("*/shard_*.npz")), "shards/*/")
    add(((path.name, path) for path in root.glob("shard_*.npz")), "the manifest's folder")
    for archive in sorted(root.glob("*.zip"), key=lambda path: (path.name != "shards.zip", path)):
        try:
            with zipfile.ZipFile(archive) as handle:
                members = [info.filename for info in handle.infolist() if not info.is_dir()]
        except (OSError, zipfile.BadZipFile):
            continue
        add(((PurePosixPath(m).name, (archive, m)) for m in members), archive.name)
    return files, zipped, ", ".join(layouts) or "none"


def landmark_candidates(input_root: Path) -> list[Landmarks]:
    """Every folder below input_root whose manifest.csv has the extraction's columns."""
    candidates = []
    for directory, _, files in walk(input_root):
        if "manifest.csv" not in files:
            continue
        try:
            with (directory / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                is_extraction, videos, needed = read_manifest_needs(handle)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if not is_extraction:
            continue
        found, zipped, layout = shard_locations(directory)
        candidates.append(
            Landmarks(
                root=directory,
                complete=(directory / "COMPLETE").is_file(),
                smoke=(directory / "SMOKE").exists() or directory.name.endswith("-smoke"),
                videos=videos,
                needed=needed,
                files=found,
                zipped=zipped,
                layout=layout,
            )
        )
    return candidates


def zips_with_a_manifest(input_root: Path) -> list[Path]:
    """Zip files below input_root holding an extraction's manifest.csv (a zipped dataset)."""
    found = []
    for directory, _, files in walk(input_root):
        for name in files:
            if not name.lower().endswith(".zip"):
                continue
            try:
                with zipfile.ZipFile(directory / name) as archive:
                    for member in archive.namelist():
                        if PurePosixPath(member).name != "manifest.csv":
                            continue
                        with archive.open(member) as raw:
                            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                            if read_manifest_needs(text)[0]:
                                found.append(directory / name)
                                break
            except (OSError, zipfile.BadZipFile, UnicodeDecodeError, ValueError):
                continue
    return found


def choose_landmarks(candidates: list[Landmarks]) -> Landmarks | None:
    """The first usable candidate, the private dataset before notebook outputs."""
    usable = [c for c in candidates if c.usable]
    usable.sort(key=lambda c: (DATASET_SLUG not in c.root.as_posix(), c.root.as_posix()))
    return usable[0] if usable else None


def link_or_copy(source: Path, target: Path) -> None:
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copyfile(source, target)


def materialize(found: Landmarks, unpack_dir: Path) -> Path:
    """found.root when its shards are already in shards/; otherwise a folder in unpack_dir
    (outside /kaggle/working, so the notebook's saved output holds no second copy) with its
    manifest and other small files copied and every shard file linked, copied or unzipped into
    shards/."""
    if found.in_place:
        return found.root
    target = unpack_dir / GATHERED
    if target.exists():
        shutil.rmtree(target)
    (target / "shards").mkdir(parents=True)
    for path in found.root.iterdir():
        if path.is_file() and path.suffix.lower() != ".zip" and shard_index(path.name) is None:
            shutil.copy2(path, target / path.name)
    for index, path in found.files.items():
        link_or_copy(path, target / "shards" / shard_file(index))
    for index, (archive_path, member) in found.zipped.items():
        sink_path = target / "shards" / shard_file(index)
        with (
            zipfile.ZipFile(archive_path) as archive,
            archive.open(member) as source,
            sink_path.open("wb") as sink,
        ):
            shutil.copyfileobj(source, sink)
    return target


def find_landmarks(input_root: Path, unpack_dir: Path) -> Path | None:
    """The usable landmark folder below input_root, gathered when needed; reports to stderr."""
    candidates = landmark_candidates(input_root) if input_root.is_dir() else []
    note(
        "ASL Citizen landmark folders (manifest.csv with the extraction's columns): "
        f"{len(candidates)}"
    )
    for candidate in candidates:
        note(f"  {candidate.describe()}")
    chosen = choose_landmarks(candidates)
    if chosen is None and input_root.is_dir():
        archives = zips_with_a_manifest(input_root)
        note(f"zip files holding such a manifest.csv: {len(archives)}")
        unpacked = []
        for number, archive_path in enumerate(archives):
            target = unpack_dir / f"asl-citizen-zipped-{number}"
            if target.exists():
                shutil.rmtree(target)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(target)
            note(f"  unpacked {archive_path} into {target}")
            for candidate in landmark_candidates(target):
                note(f"  {candidate.describe()}")
                unpacked.append(candidate)
        chosen = choose_landmarks(unpacked)
    if chosen is None:
        note(
            "no usable ASL Citizen landmarks: add the dataset asl-citizen-landmarks-private, or "
            "the output of the extraction merge run (the version whose log says COMPLETE), as an "
            "input"
        )
        return None
    landmarks = materialize(chosen, unpack_dir)
    if landmarks != chosen.root:
        note(f"gathered the shard files of {chosen.root} into {landmarks}")
    note(f"using {landmarks}: {chosen.videos} videos in manifest.csv, {len(chosen.needed)} shards")
    return landmarks


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
    landmarks = find_landmarks(input_root, args.unpack_dir)
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
