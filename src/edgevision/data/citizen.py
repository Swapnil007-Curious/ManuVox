"""ASL Citizen landmarks (kaggle/asl_citizen_merge.ipynb output) as edge-vision processed data.

The landmarks are stored in the asl-signs parquet layout, so ``coords_from_holistic`` selects the
same slots ``pivot_sequence`` does and preprocessing continues with the shared steps
(``preprocess_coords_measured``). ``build_processed`` writes a processed dir like
scripts/preprocess.py does (features.npz, manifest.csv, preprocess_report.json) plus the official
splits; ``combine`` joins it to the asl-signs processed dir for v3 training.

ASL Citizen ids are offset so they never collide with asl-signs ids: sequence ids start at
CITIZEN_SEQUENCE_OFFSET (asl-signs sequence ids are below 2**32) and participant ids at
CITIZEN_PARTICIPANT_OFFSET + the number in "P<n>". ``source_of`` tells the two apart.

The data is licensed for non-commercial research only and may not be redistributed; nothing this
module writes may be committed (docs/DATA_LICENSES.md).
"""

import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from edgevision.data.io import (
    LABEL_DTYPE,
    PARTICIPANT_DTYPE,
    SEQUENCE_DTYPE,
    aspect_stats,
    to_storage,
    write_json,
)
from edgevision.data.landmarks import N_ALL_LANDMARKS, num_selected, selection_table
from edgevision.data.preprocess import (
    AspectMeasurement,
    FloatArray,
    preprocess_coords_measured,
    snap_x,
)

CITIZEN_SEQUENCE_OFFSET = 10**10
CITIZEN_PARTICIPANT_OFFSET = 1_000_000
SOURCES = ("asl-signs", "asl-citizen")
RULES = ("exact", "variant", "synonym")
SPLIT_NAMES = ("train", "val", "test")
OFFICIAL_SPLITS_JSON = "splits_official.json"
MANIFEST_EXTRA = ("video_file", "gloss", "rule", "citizen_split")

IntArray = npt.NDArray[np.int64]


def source_of(sequence_id: npt.ArrayLike) -> npt.NDArray[np.int8]:
    """0 for asl-signs, 1 for ASL Citizen (the index into SOURCES), per sequence id."""
    return (np.asarray(sequence_id, dtype=np.int64) >= CITIZEN_SEQUENCE_OFFSET).astype(np.int8)


def participant_id(participant: str) -> int:
    """ "P12" -> CITIZEN_PARTICIPANT_OFFSET + 12."""
    match = re.fullmatch(r"P(\d+)", participant.strip())
    if match is None:
        raise ValueError(f"unexpected ASL Citizen participant id {participant!r}")
    return CITIZEN_PARTICIPANT_OFFSET + int(match.group(1))


def coords_from_holistic(
    landmarks: npt.NDArray[np.floating[Any]], pose_indices: Sequence[int]
) -> FloatArray:
    """Select (T, 543, 3) parquet-layout landmarks into (T, L, 3) float64 with snapped x, the
    array pivot_sequence makes from the same points in long format."""
    if landmarks.ndim != 3 or landmarks.shape[1:] != (N_ALL_LANDMARKS, 3):
        raise ValueError(f"expected (T, {N_ALL_LANDMARKS}, 3) landmarks, got {landmarks.shape}")
    values = landmarks.astype(np.float64)
    if np.isinf(values).any():
        raise ValueError("infinite coordinates")
    slots = selection_table(pose_indices)
    kept = np.flatnonzero(slots >= 0)
    coords = np.full((landmarks.shape[0], num_selected(pose_indices), 3), np.nan)
    coords[:, slots[kept]] = values[:, kept]
    coords[..., 0] = snap_x(coords[..., 0])
    return coords


def find_root(path: Path) -> Path:
    """The directory holding manifest.csv and shards/ at or below ``path`` (a downloaded
    Kaggle dataset may sit one or two folders down)."""
    for candidate in [path, *sorted(path.rglob("manifest.csv"))]:
        root = candidate if candidate.is_dir() else candidate.parent
        if (root / "manifest.csv").is_file() and (root / "shards").is_dir():
            return root
    raise FileNotFoundError(f"no manifest.csv with a shards/ folder under {path}")


def read_landmark_manifest(root: Path) -> pd.DataFrame:
    """manifest.csv with a stable ``sequence_id`` column: CITIZEN_SEQUENCE_OFFSET plus the row's
    position among all planned videos sorted by file name (independent of any later filter)."""
    manifest = pd.read_csv(root / "manifest.csv", keep_default_na=False, dtype=str)
    manifest = manifest.sort_values("video_file", kind="stable").reset_index(drop=True)
    manifest["sequence_id"] = CITIZEN_SEQUENCE_OFFSET + np.arange(len(manifest), dtype=np.int64)
    return manifest


def iter_landmarks(
    root: Path, rows: pd.DataFrame
) -> Iterator[tuple[pd.Series, npt.NDArray[np.float32]]]:
    """(row, (T, 543, 3) landmarks) for each row with status ok, loading each shard once."""
    usable = rows.loc[rows["status"] == "ok"]
    for shard, group in usable.groupby(usable["shard"].astype(int), sort=True):
        with np.load(root / "shards" / f"shard_{shard:05d}.npz", allow_pickle=False) as data:
            landmarks = data["landmarks"]
        for _, row in group.iterrows():
            start, length = int(row["start"]), int(row["n_frames"])
            yield row, landmarks[start : start + length]


def build_processed(
    landmarks_dir: Path,
    output_dir: Path,
    labels: Sequence[str],
    *,
    pose_indices: Sequence[int],
    num_frames: int = 32,
    min_hand_frames: int = 3,
    rules: Sequence[str] = RULES,
    splits: Sequence[str] = SPLIT_NAMES,
) -> dict[str, Any]:
    """Preprocess the mapped videos of ``splits`` into ``output_dir``; returns the report.

    ``labels`` are the model's labels in index order, so y matches the asl-signs label ids.
    """
    unknown = sorted(set(rules) - set(RULES))
    if unknown:
        raise ValueError(f"unknown mapping rules {unknown}")
    root = find_root(landmarks_dir)
    manifest = read_landmark_manifest(root)
    index = {label: position for position, label in enumerate(labels)}
    chosen = manifest.loc[
        manifest["label"].isin(index)
        & manifest["rule"].isin(rules)
        & manifest["split"].isin(splits)
    ].copy()
    not_ok = chosen.loc[chosen["status"] != "ok"]

    features, presence, records = [], [], []
    ratios, shoulder_frames, kept_frames = [], 0, 0
    for row, landmarks in iter_landmarks(root, chosen):
        result, aspect = preprocess_coords_measured(
            coords_from_holistic(landmarks, pose_indices),
            num_frames=num_frames,
            pose_indices=pose_indices,
            min_hand_frames=min_hand_frames,
        )
        outcome = to_storage(int(row["sequence_id"]), result, aspect)
        info = outcome.info
        records.append(
            {
                "sequence_id": int(row["sequence_id"]),
                "participant_id": participant_id(row["participant"]),
                "sign": row["label"],
                "label": index[row["label"]],
                "excluded": info["excluded"],
                "reason": info["reason"] or "",
                "t_raw": info["t_raw"],
                "t_kept": info["t_kept"],
                "mirrored": info["mirrored"],
                "norm_method": info["norm_method"] or "",
                "video_file": row["video_file"],
                "gloss": row["gloss"],
                "rule": row["rule"],
                "citizen_split": row["split"],
            }
        )
        if outcome.features is not None and outcome.presence is not None:
            features.append(outcome.features)
            presence.append(outcome.presence)
            ratios.append(aspect.ratios)
            shoulder_frames += aspect.shoulder_frames
            kept_frames += aspect.kept_frames

    table = pd.DataFrame.from_records(records)
    if table.empty:
        raise ValueError(f"no usable videos under {root} for splits {list(splits)}")
    kept = table.loc[~table["excluded"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    n_landmarks = num_selected(pose_indices)
    arrays: dict[str, npt.NDArray[Any]] = {
        "X": np.stack(features)
        if features
        else np.zeros((0, num_frames, n_landmarks, 2), np.float16),
        "presence": np.stack(presence) if presence else np.zeros((0, num_frames, 2), np.uint8),
        "y": kept["label"].to_numpy(dtype=LABEL_DTYPE),
        "participant_id": kept["participant_id"].to_numpy(dtype=PARTICIPANT_DTYPE),
        "sequence_id": kept["sequence_id"].to_numpy(dtype=SEQUENCE_DTYPE),
    }
    _save_npz(output_dir / "features.npz", arrays)
    table.to_csv(output_dir / "manifest.csv", index=False)

    official: dict[str, Any] = {"mode": "official", "source": SOURCES[1]}
    participants: dict[str, list[int]] = {}
    for split in SPLIT_NAMES:
        rows = kept.loc[kept["citizen_split"] == split]
        official[split] = [int(value) for value in rows["sequence_id"]]
        participants[split] = sorted(int(value) for value in rows["participant_id"].unique())
    official["participants"] = participants
    write_json(output_dir / OFFICIAL_SPLITS_JSON, official)

    measurement = AspectMeasurement(
        np.concatenate(ratios) if ratios else np.empty(0), shoulder_frames, kept_frames
    )
    reasons = table.loc[table["excluded"], "reason"].value_counts()
    report: dict[str, Any] = {
        "source": SOURCES[1],
        "landmarks_dir": str(root),
        "rules": list(rules),
        "splits": list(splits),
        "counts": {
            "planned": len(chosen),
            "not_tracked": len(not_ok),
            "total": len(table),
            "kept": len(kept),
            "excluded": len(table) - len(kept),
            "excluded_by_reason": {str(key): int(value) for key, value in reasons.items()},
            "kept_by_split": {s: len(official[s]) for s in SPLIT_NAMES},
            "signs_by_split": {
                s: int(kept.loc[kept["citizen_split"] == s, "sign"].nunique()) for s in SPLIT_NAMES
            },
        },
        "mirrored_fraction": float(kept["mirrored"].mean()) if len(kept) else 0.0,
        "config": {
            "preprocess": {
                "num_frames": num_frames,
                "use_z": False,
                "pose_indices": list(pose_indices),
                "min_hand_frames": min_hand_frames,
            }
        },
        "landmarks": {
            "num_landmarks": n_landmarks,
            "landmark_order": ["left_hand_0-20", "right_hand_0-20", "pose_indices"],
            "pose_indices": list(pose_indices),
        },
        "aspect_stats": aspect_stats(measurement),
    }
    write_json(output_dir / "preprocess_report.json", report)
    return report


def _save_npz(path: Path, arrays: dict[str, npt.NDArray[Any]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    temporary.replace(path)


def _load_arrays(processed_dir: Path) -> dict[str, npt.NDArray[Any]]:
    """The arrays of features.npz. NumPy before 2.1 had no allow_pickle argument in
    savez_compressed and stored it as a 0-d array named "allow_pickle"; such files exist, so
    that scalar is left out."""
    with np.load(processed_dir / "features.npz", allow_pickle=False) as data:
        return {
            key: data[key]
            for key in data.files
            if not (key == "allow_pickle" and data[key].ndim == 0)
        }


def combine(
    kaggle_dir: Path,
    citizen_dir: Path,
    output_dir: Path,
    *,
    kaggle_splits: str = "splits_signer_independent.json",
) -> dict[str, Any]:
    """Join the asl-signs and ASL Citizen processed dirs into one for training.

    The split file keeps each source's own split: the asl-signs signer-independent split and the
    ASL Citizen official split (whose signers are disjoint across train, val and test). It is
    named splits_signer_independent.json and records the asl-signs seed, so training with
    n_test_signers and n_val_signers set to the combined counts accepts it.
    """
    kaggle_report = json.loads((kaggle_dir / "preprocess_report.json").read_text("utf-8"))
    citizen_report = json.loads((citizen_dir / "preprocess_report.json").read_text("utf-8"))
    selection = kaggle_report["landmarks"]["pose_indices"]
    if citizen_report["landmarks"]["pose_indices"] != selection:
        raise ValueError("the two processed dirs use different pose selections")

    kaggle_manifest = pd.read_csv(kaggle_dir / "manifest.csv", keep_default_na=False)
    citizen_manifest = pd.read_csv(citizen_dir / "manifest.csv", keep_default_na=False)
    signs = dict(zip(kaggle_manifest["label"], kaggle_manifest["sign"], strict=False))
    clash = citizen_manifest.loc[citizen_manifest["label"].map(signs) != citizen_manifest["sign"]]
    if not clash.empty:
        raise ValueError(f"label ids differ between the sources, e.g. {clash.iloc[0].to_dict()}")

    first, second = _load_arrays(kaggle_dir), _load_arrays(citizen_dir)
    if np.intersect1d(first["sequence_id"], second["sequence_id"]).size:
        raise ValueError("sequence ids overlap between the sources")
    if np.intersect1d(first["participant_id"], second["participant_id"]).size:
        raise ValueError("participant ids overlap between the sources")
    arrays = {key: np.concatenate([first[key], second[key]]) for key in first}
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_npz(output_dir / "features.npz", arrays)
    manifest = pd.concat([kaggle_manifest, citizen_manifest], ignore_index=True, sort=False)
    for column in MANIFEST_EXTRA:
        manifest[column] = manifest[column].fillna("") if column in manifest else ""
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    kaggle_split = json.loads((kaggle_dir / kaggle_splits).read_text("utf-8"))
    citizen_split = json.loads((citizen_dir / OFFICIAL_SPLITS_JSON).read_text("utf-8"))
    combined: dict[str, Any] = {
        "mode": kaggle_split["mode"],
        "seed": kaggle_split["seed"],
        "sources": {
            SOURCES[0]: f"{kaggle_splits} ({kaggle_split['mode']}, seed {kaggle_split['seed']})",
            SOURCES[1]: "the official ASL Citizen splits",
        },
    }
    for name in SPLIT_NAMES:
        combined[name] = list(kaggle_split[name]) + list(citizen_split[name])
    combined["participants"] = {
        name: list(kaggle_split["participants"][name]) + list(citizen_split["participants"][name])
        for name in SPLIT_NAMES
    }
    combined["counts"] = {
        name: {SOURCES[0]: len(kaggle_split[name]), SOURCES[1]: len(citizen_split[name])}
        for name in SPLIT_NAMES
    }
    write_json(output_dir / "splits_signer_independent.json", combined)

    report = {
        **{key: kaggle_report[key] for key in ("config", "landmarks")},
        # The live aspect correction keeps the asl-signs reference ratio; ASL Citizen's is kept
        # beside it for comparison.
        "aspect_stats": kaggle_report.get("aspect_stats"),
        "aspect_stats_by_source": {
            SOURCES[0]: kaggle_report.get("aspect_stats"),
            SOURCES[1]: citizen_report.get("aspect_stats"),
        },
        "combined_from": {SOURCES[0]: str(kaggle_dir), SOURCES[1]: str(citizen_dir)},
        "counts": {
            SOURCES[0]: kaggle_report.get("counts"),
            SOURCES[1]: citizen_report.get("counts"),
        },
    }
    write_json(output_dir / "preprocess_report.json", report)
    return combined
