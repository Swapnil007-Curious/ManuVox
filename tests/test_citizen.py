"""ASL Citizen landmarks as processed data: the parquet-layout selection equals pivot_sequence,
build_processed equals preprocess_sequence, and combine keeps the two sources apart."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import SIGNS, landmark_blocks, long_frame
from edgevision.config import SplitConfig
from edgevision.data import citizen
from edgevision.data.io import to_storage
from edgevision.data.landmarks import LANDMARK_COUNTS
from edgevision.data.preprocess import pivot_sequence, preprocess_sequence
from edgevision.train.data import load_split_arrays, prepare_splits
from v2_support import V2_POSE_INDICES, build_v2_processed_dir

CASES = ("both_hands", "right_hand_only", "left_hand_only", "partial_hands", "no_hands")
# Participant -> official split; one signer per split is enough for the tests.
PARTICIPANTS = {"P1": "train", "P2": "train", "P3": "val", "P4": "test"}


def parquet_layout(blocks):
    """(T, 543, 3) float32 in the order face, left_hand, pose, right_hand."""
    return np.concatenate([blocks[name] for name in LANDMARK_COUNTS], axis=1).astype(np.float32)


def write_landmarks(root: Path, rng: np.random.Generator):
    """A miniature kaggle/asl_citizen_merge.ipynb output: manifest.csv and two shards."""
    rows, arrays, by_file = [], {0: [], 1: []}, {}
    ordinal = 0
    for participant, split in PARTICIPANTS.items():
        for sign in SIGNS:
            for repeat in range(2):
                case = CASES[ordinal % len(CASES)]
                blocks = landmark_blocks(int(rng.integers(12, 40)), rng, case)
                blocks = {k: v.astype(np.float32).astype(np.float64) for k, v in blocks.items()}
                shard = ordinal % 2
                start = sum(len(a) for a in arrays[shard])
                landmarks = parquet_layout(blocks)
                arrays[shard].append(landmarks)
                name = f"{ordinal:04d}-{sign.upper()}{repeat}.mp4"
                by_file[name] = blocks
                rows.append(
                    {
                        "video_file": name,
                        "participant": participant,
                        "split": split,
                        "gloss": sign.upper(),
                        "asl_lex_code": "X_00_000",
                        "label": sign,
                        "rule": "exact" if repeat == 0 else "variant",
                        "shard": shard,
                        "start": start,
                        "n_frames": len(landmarks),
                        "status": "ok",
                    }
                )
                ordinal += 1
    rows.append({**rows[0], "video_file": "9999-BROKEN.mp4", "status": "decode_error"})
    rows.append({**rows[1], "video_file": "9998-REVIEW.mp4", "label": "", "rule": "review"})
    (root / "shards").mkdir(parents=True)
    for shard, parts in arrays.items():
        np.savez_compressed(
            root / "shards" / f"shard_{shard:05d}.npz", landmarks=np.concatenate(parts)
        )
    pd.DataFrame(rows).sample(frac=1.0, random_state=0).to_csv(root / "manifest.csv", index=False)
    return by_file


@pytest.fixture(scope="module")
def landmarks(tmp_path_factory):
    root = tmp_path_factory.mktemp("citizen") / "dataset" / "asl-citizen-landmarks"
    return root, write_landmarks(root, np.random.default_rng(7))


def test_source_and_participant_ids():
    assert citizen.source_of([1_000_000_000, citizen.CITIZEN_SEQUENCE_OFFSET]).tolist() == [0, 1]
    assert citizen.participant_id("P12") == citizen.CITIZEN_PARTICIPANT_OFFSET + 12
    with pytest.raises(ValueError):
        citizen.participant_id("12")


@pytest.mark.parametrize("case", CASES)
def test_holistic_selection_equals_pivot_sequence(case):
    rng = np.random.default_rng(3)
    # Shards and the asl-signs parquet files both hold float32 coordinates.
    blocks = {
        k: v.astype(np.float32).astype(np.float64)
        for k, v in landmark_blocks(20, rng, case).items()
    }
    expected = pivot_sequence(long_frame(blocks), V2_POSE_INDICES)
    got = citizen.coords_from_holistic(parquet_layout(blocks).astype(np.float64), V2_POSE_INDICES)
    np.testing.assert_array_equal(got, expected)


def test_holistic_selection_checks_the_shape():
    with pytest.raises(ValueError, match="543"):
        citizen.coords_from_holistic(np.zeros((4, 75, 3)), V2_POSE_INDICES)


def test_find_root_descends_into_a_downloaded_dataset(landmarks):
    root, _ = landmarks
    assert citizen.find_root(root.parent.parent) == root


def test_build_processed_matches_the_parquet_path(landmarks, tmp_path):
    root, by_file = landmarks
    report = citizen.build_processed(root, tmp_path, SIGNS, pose_indices=V2_POSE_INDICES)
    manifest = pd.read_csv(tmp_path / "manifest.csv", keep_default_na=False)
    assert len(manifest) == len(by_file)  # The broken and the review video are left out.
    assert report["counts"]["not_tracked"] == 1
    with np.load(tmp_path / "features.npz") as data:
        arrays = {key: data[key] for key in data.files}
    kept = manifest.loc[~manifest["excluded"]].reset_index(drop=True)
    assert arrays["X"].shape == (len(kept), 32, 59, 2) and arrays["X"].dtype == np.float16
    for position, row in kept.iterrows():
        df = long_frame({k: v.astype(np.float32) for k, v in by_file[row["video_file"]].items()})
        expected = to_storage(0, preprocess_sequence(df, pose_indices=V2_POSE_INDICES))
        np.testing.assert_array_equal(arrays["X"][position], expected.features)
        np.testing.assert_array_equal(arrays["presence"][position], expected.presence)
        assert arrays["y"][position] == SIGNS.index(row["sign"])
    assert (manifest.loc[manifest["excluded"], "reason"] == "insufficient_hand_frames").all()
    assert (arrays["sequence_id"] >= citizen.CITIZEN_SEQUENCE_OFFSET).all()
    official = json.loads((tmp_path / citizen.OFFICIAL_SPLITS_JSON).read_text("utf-8"))
    assert sorted(official["test"] + official["val"] + official["train"]) == sorted(
        arrays["sequence_id"].tolist()
    )
    assert official["participants"]["val"] == [citizen.participant_id("P3")]


def test_sequence_ids_do_not_depend_on_the_filter(landmarks, tmp_path):
    root, _ = landmarks
    citizen.build_processed(root, tmp_path / "all", SIGNS, pose_indices=V2_POSE_INDICES)
    citizen.build_processed(
        root, tmp_path / "exact", SIGNS, pose_indices=V2_POSE_INDICES, rules=("exact",)
    )
    every = pd.read_csv(tmp_path / "all/manifest.csv").set_index("video_file")["sequence_id"]
    exact = pd.read_csv(tmp_path / "exact/manifest.csv").set_index("video_file")["sequence_id"]
    assert (every.loc[exact.index] == exact).all() and len(exact) < len(every)


def test_combine_keeps_each_split_and_trains_with_source_selection(landmarks, tmp_path):
    root, _ = landmarks
    kaggle = build_v2_processed_dir(tmp_path / "kaggle").root
    citizen.build_processed(root, tmp_path / "citizen", SIGNS, pose_indices=V2_POSE_INDICES)
    combined = citizen.combine(kaggle, tmp_path / "citizen", tmp_path / "combined")
    kaggle_split = json.loads((kaggle / "splits_signer_independent.json").read_text("utf-8"))
    assert combined["val"][: len(kaggle_split["val"])] == kaggle_split["val"]
    assert set(citizen.source_of(combined["val"]).tolist()) == {0, 1}
    n_test = len(combined["participants"]["test"])
    n_val = len(combined["participants"]["val"])
    split = SplitConfig(n_test_signers=n_test, n_val_signers=n_val)
    splits_path = prepare_splits(tmp_path / "combined", split, tmp_path)
    arrays = load_split_arrays(tmp_path / "combined", splits_path, ("train", "val"))
    assert len(arrays["val"].y) == len(combined["val"])

    from edgevision.train.config import load_train_config
    from edgevision.train.loop import run_training

    cfg = load_train_config(
        Path(__file__).resolve().parents[1] / "experiments/train_e_hybrid_v3citizen.yaml",
        processed_dir=tmp_path / "combined",
        output_root=tmp_path / "runs",
        epochs=1,
    ).model_copy(update={"split": split, "batch_size": 32})
    summary = run_training(cfg, seed=0, device="cpu", log=lambda _: None)
    assert summary["val_selection"] == "source_mean"
    assert set(summary["best_val_top1_by_source"]) == set(citizen.SOURCES)
    expected = np.mean(list(summary["best_val_top1_by_source"].values()))
    assert summary["best_val_select"] == pytest.approx(expected)
    metrics = pd.read_csv(tmp_path / "runs" / cfg.experiment_name / "seed_0" / "metrics.csv")
    assert {"val_select", "val_top1_asl-signs", "val_top1_asl-citizen"} <= set(metrics.columns)


def test_combine_refuses_overlapping_ids(landmarks, tmp_path):
    root, _ = landmarks
    kaggle = build_v2_processed_dir(tmp_path / "kaggle").root
    citizen.build_processed(root, tmp_path / "citizen", SIGNS, pose_indices=V2_POSE_INDICES)
    with pytest.raises(ValueError, match="overlap"):
        citizen.combine(kaggle, kaggle, tmp_path / "twice")


# Signers of the ASL Citizen official val split with videos of the planned signs (the
# extraction's manifest.csv, 2026-11-19: P13 has none of them, so 5 of the 6 official ones).
CITIZEN_VAL_SIGNERS_IN_PLAN = 5


def test_the_v3_config_split_matches_the_combined_dir():
    """The combined dir holds the asl-signs signer-independent split and the ASL Citizen train and
    val signers only (its test split is evaluated locally, once), so training accepts its split
    file only with these counts. The first Kaggle run stopped on 15 / 9."""
    from edgevision.train.config import load_train_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_train_config(root / "experiments/train_e_hybrid_v3citizen.yaml")
    asl_signs = json.loads(
        (
            root / "results/runs/cnn_signer_independent/seed_0/splits_signer_independent.json"
        ).read_text("utf-8")
    )
    assert cfg.split.mode == asl_signs["mode"] and cfg.split.seed == asl_signs["seed"]
    assert cfg.split.n_test_signers == len(asl_signs["participants"]["test"])
    assert cfg.split.n_val_signers == (
        len(asl_signs["participants"]["val"]) + CITIZEN_VAL_SIGNERS_IN_PLAN
    )


def test_combine_skips_the_allow_pickle_scalar_of_old_numpy(landmarks, tmp_path):
    """NumPy before 2.1 stored savez_compressed(..., allow_pickle=False) as an array."""
    root, _ = landmarks
    kaggle = build_v2_processed_dir(tmp_path / "kaggle").root
    with np.load(kaggle / "features.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    np.savez_compressed(kaggle / "features.npz", allow_pickle=np.asarray(False), **arrays)
    citizen.build_processed(root, tmp_path / "citizen", SIGNS, pose_indices=V2_POSE_INDICES)
    citizen.combine(kaggle, tmp_path / "citizen", tmp_path / "combined")
    with np.load(tmp_path / "combined" / "features.npz", allow_pickle=False) as data:
        assert "allow_pickle" not in data.files and set(arrays) <= set(data.files)
