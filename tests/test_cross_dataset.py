"""Cross-dataset metrics, the once-only ASL Citizen test evaluation, and the v3 notebook helpers."""

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import SIGNS
from edgevision.evaluation.cross_dataset import (
    cluster_bootstrap_ci,
    holm_family,
    paired,
    summarize,
    topk_correct,
)
from edgevision.export.onnx_export import export_onnx
from edgevision.models.registry import build_model
from test_citizen import write_landmarks
from v2_support import V2_LAYOUT, V2_POSE_INDICES

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_topk_correct_breaks_ties_toward_the_lower_index():
    logits = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    assert topk_correct(logits, np.array([1, 2]), 1).tolist() == [False, False]
    assert topk_correct(logits, np.array([1, 2]), 2).tolist() == [True, True]


def test_cluster_bootstrap_is_wider_than_the_item_bootstrap_for_clustered_data():
    values = np.repeat([1.0, 0.0, 1.0, 0.0, 1.0, 1.0], 50)
    clusters = np.repeat(np.arange(6), 50)
    low, high = cluster_bootstrap_ci(values, clusters)
    assert low < values.mean() < high and high - low > 0.3


def test_summarize_reports_rules_signs_and_the_decision_rule():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 200)
    logits = rng.normal(0, 1, (200, 5))
    logits[np.arange(200), y] += 2.0
    result = summarize(
        logits,
        y,
        signers=np.repeat(["a", "b", "c", "d"], 50),
        signs=[SIGNS[i] for i in y],
        rules=["exact"] * 150 + ["synonym"] * 50,
        labels=list(SIGNS),
        temperature=1.0,
        threshold=0.5,
        excluded=10,
    )
    assert result["videos"] == 200 and result["signers"] == 4
    assert result["top1"] == pytest.approx(float((logits.argmax(1) == y).mean()))
    assert result["top1_all_videos"] == pytest.approx(result["top1"] * 200 / 210)
    assert result["top1_ci"][0] <= result["top1"] <= result["top1_ci"][1]
    assert result["top3"] >= result["top1"]
    assert set(result["by_rule"]) == {"exact", "synonym"}
    assert len(result["per_sign"]) == len(set(y.tolist()))
    assert 0.0 <= result["coverage"] <= 1.0


def test_paired_and_holm():
    a = np.array([1, 0, 0, 1, 0, 0, 0, 0] * 10)
    b = np.array([1, 1, 1, 1, 0, 1, 0, 0] * 10)
    result = paired(a, b, clusters=np.repeat(np.arange(8), 10))
    assert result["diff"] == pytest.approx(0.375)
    assert result["better"] == 30 and result["worse"] == 0 and result["p"] < 0.01
    # The interval is of the same b - a difference: it contains diff and, here, excludes 0.
    low, high = result["ci"]
    assert 0.0 < low <= result["diff"] <= high
    low_c, high_c = result["ci_cluster"]
    assert low_c <= result["diff"] <= high_c
    family = {"x": result, "y": paired(a, a)}
    holm_family(family)
    assert family["x"]["significant"] and not family["y"]["significant"]
    with pytest.raises(ValueError):
        paired(a, b[:-1])


@pytest.fixture
def model_dir(tmp_path):
    torch_model = build_model("cnn", len(SIGNS), feature_set="v2", layout=V2_LAYOUT)
    directory = tmp_path / "model"
    path = export_onnx(torch_model, directory / "model.onnx")
    meta = {
        "run": "test/seed_0",
        "variant": "fp32",
        "data_source": "synthetic",
        "model_file": "model.onnx",
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "labels": list(SIGNS),
        "preprocessing": {
            "pose_indices": list(V2_POSE_INDICES),
            "num_frames": 32,
            "min_hand_frames": 3,
        },
        "input_layout": {"num_landmarks": V2_LAYOUT.num_landmarks},
        "calibration": {"temperature": 1.2, "threshold": 0.3},
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


def test_citizen_eval_logs_access_first_and_refuses_a_second_run(tmp_path, model_dir, monkeypatch):
    landmarks = tmp_path / "landmarks"
    write_landmarks(landmarks, np.random.default_rng(1))
    monkeypatch.chdir(tmp_path)
    module = load("citizen_eval")
    args = ["--landmarks", str(landmarks), "--model-dir", str(model_dir), "--name", "tiny"]
    assert module.main(args) == 0
    results = tmp_path / "results/asl_citizen"
    summary = json.loads((results / "tiny_test.json").read_text("utf-8"))
    assert summary["split"].startswith("asl-citizen test")
    assert summary["videos"] > 0 and set(summary["by_rule"]) <= {"exact", "variant"}
    assert (results / "tiny_test_per_sign.csv").is_file()
    with np.load(results / "predictions/tiny_test.npz") as data:
        assert (data["sequence_id"] >= 10**10).all() and "participant_id" in data.files
    log = json.loads((results / "test_access_log.json").read_text("utf-8"))
    assert len(log) == 1 and log[0]["runs_evaluated"][0].startswith("asl-citizen-test:tiny:")
    # Only the test signer's videos were preprocessed.
    manifest = (tmp_path / "data/asl_citizen/test_pose17/manifest.csv").read_text("utf-8")
    assert "P4" not in manifest  # participant ids are stored offset, never as P<n>
    assert module.main(args) == 1
    assert len(json.loads((results / "test_access_log.json").read_text("utf-8"))) == 1
    assert module.main([*args, "--reaccess", "checking the report layout"]) == 0
    log = json.loads((results / "test_access_log.json").read_text("utf-8"))
    assert len(log) == 2 and "reaccess: checking the report layout" in log[1]["runs_evaluated"][0]


def test_v3_select_and_pack(tmp_path):
    module = load("v3_kaggle")
    runs = tmp_path / "runs" / "e_hybrid_v3citizen"
    for seed, score in ((0, 0.61), (1, 0.64), (2, 0.64)):
        (runs / f"seed_{seed}").mkdir(parents=True)
        summary = {"best_val_select": score, "best_val_top1": 0.7}
        (runs / f"seed_{seed}" / "summary.json").write_text(json.dumps(summary), "utf-8")
        (runs / f"seed_{seed}" / "best.pt").write_bytes(b"weights")
    out = tmp_path / "v3_export" / "selection.json"
    assert module.main(["select", "--runs", str(runs), "--out", str(out)]) == 0
    assert json.loads(out.read_text("utf-8"))["chosen"] == "seed_1"  # the lower seed on a tie
    (tmp_path / "processed_v3").mkdir()
    (tmp_path / "processed_v3" / "features.npz").write_bytes(b"data")
    (tmp_path / "processed_v3" / "preprocess_report.json").write_text("{}", "utf-8")
    bundle = tmp_path / "bundle.zip"
    assert module.main(["pack", "--work", str(tmp_path), "--output", str(bundle)]) == 0
    names = zipfile.ZipFile(bundle).namelist()
    assert "runs/e_hybrid_v3citizen/seed_1/best.pt" in names
    assert "processed_v3/preprocess_report.json" in names
    assert not any("features.npz" in name for name in names)
