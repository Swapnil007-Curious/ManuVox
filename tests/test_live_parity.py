"""The committed parity fixtures (web/tests/golden/live_parity.json) still match Python.

The fixtures store their raw inputs, so this needs no dataset: every model input is recomputed
from those inputs with the current Python pipeline and the committed model, and must agree with
the stored one within the fixture tolerance, with the same top-1. Together with
web/tests/unit/liveParity.test.ts this ties both implementations to one set of numbers.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgevision import spec
from edgevision.data.landmarks import N_HAND
from edgevision.data.preprocess import preprocess_sequence
from edgevision.live.aspect import frame_k
from edgevision.live.parity import assigned_frame
from edgevision.live.pipeline import (
    LivePipeline,
    PipelineConfig,
    preprocess_segment,
    to_model_input,
)
from edgevision.runtime import OrtModel

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "web" / "tests" / "golden" / "live_parity.json"
META = REPO / "web" / "public" / "model" / "meta.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def meta():
    return json.loads(META.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model(meta):
    return OrtModel(META.parent / meta["model_file"], intra_op_threads=1)


def rows_from_blocks(blocks, pose_indices):
    parts = []
    for kind, frames, points in (
        ("left_hand", blocks["left_hand"], list(range(N_HAND))),
        ("right_hand", blocks["right_hand"], list(range(N_HAND))),
        ("pose", blocks["pose"], list(pose_indices)),
    ):
        for frame, values in enumerate(frames):
            xy = (
                np.full((len(points), 2), np.nan)
                if values is None
                else np.asarray(values).reshape(-1, 2)
            )
            parts.append(
                pd.DataFrame(
                    {
                        "frame": frame,
                        "type": kind,
                        "landmark_index": points,
                        "x": xy[:, 0],
                        "y": xy[:, 1],
                        "z": np.where(np.isnan(xy[:, 0]), np.nan, 0.0),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def test_fixtures_belong_to_the_committed_model_and_spec(golden, meta):
    assert golden["model_sha256"] == meta["model_sha256"]
    assert len(golden["dataset_cases"]) == len(golden["live_cases"]) == 20
    assert golden["camera"]["k"] == pytest.approx(
        frame_k(golden["camera"]["width"], golden["camera"]["height"])
    )
    assert golden["preprocessing"]["pose_indices"] == list(spec.POSE_SETS["v2"])


def test_dataset_cases_match_python(golden, model):
    pre = golden["preprocessing"]
    for case in golden["dataset_cases"]:
        result = preprocess_sequence(
            rows_from_blocks(case["blocks"], pre["pose_indices"]),
            num_frames=pre["num_frames"],
            use_z=False,
            pose_indices=pre["pose_indices"],
            min_hand_frames=pre["min_hand_frames"],
        )
        assert dict(result.info) == case["info"]
        model_input = to_model_input(result)
        assert np.abs(model_input - np.asarray(case["model_input"])).max() <= golden["tolerance"]
        assert int(np.argmax(model.run(model_input[None])[0])) == int(np.argmax(case["logits"]))


def test_live_cases_match_python(golden, model):
    pre = golden["preprocessing"]
    k = frame_k(golden["camera"]["width"], golden["camera"]["height"])
    for case in golden["live_cases"]:
        frames = [assigned_frame(frame) for frame in case["frames"]]
        result, model_input = preprocess_segment(
            frames,
            k,
            pose_indices=pre["pose_indices"],
            num_frames=pre["num_frames"],
            min_hand_frames=pre["min_hand_frames"],
        )
        assert dict(result.info) == case["info"]
        assert np.abs(model_input - np.asarray(case["model_input"])).max() <= golden["tolerance"]
        assert int(np.argmax(model.run(model_input[None])[0])) == int(np.argmax(case["logits"]))


def test_stream_matches_python(golden, model):
    pre = golden["preprocessing"]
    pipeline = LivePipeline(
        PipelineConfig(
            pose_indices=tuple(pre["pose_indices"]),
            num_frames=pre["num_frames"],
            min_hand_frames=pre["min_hand_frames"],
            width=golden["camera"]["width"],
            height=golden["camera"]["height"],
        ),
        lambda x: model.run(x[None])[0],
    )
    frames = [assigned_frame(frame) for frame in golden["stream"]["frames"]]
    outcomes = []
    for index, frame in enumerate(frames):
        outcome = pipeline.push(frame)
        expected = golden["stream"]["energies"][index]
        if expected is None:
            assert np.isnan(pipeline.last_energy)
        else:
            assert pipeline.last_energy == pytest.approx(expected, rel=1e-9, abs=1e-12)
        if outcome is not None:
            outcomes.append(outcome)
    last = pipeline.flush()
    if last is not None:
        outcomes.append(last)
    assert [o.event.kind for o in outcomes] == [s["kind"] for s in golden["stream"]["segments"]]
    for outcome, expected in zip(outcomes, golden["stream"]["segments"], strict=True):
        assert outcome.event.reason == expected["reason"]
        assert len(outcome.event.frames) == expected["frames"]
        if expected["model_input"] is not None:
            assert (
                np.abs(outcome.model_input - np.asarray(expected["model_input"])).max()
                <= golden["tolerance"]
            )
            assert int(np.argmax(outcome.logits)) == int(np.argmax(expected["logits"]))
