"""scripts/live_diagnose.py on synthetic recordings: signs whose classes differ in motion, seen
through a camera that stretches y by 2.5 (k = 0.4), recorded by an app with correction off, and
classified by a model trained on the undistorted signs."""

import json

import pytest

import conftest
from edgevision.config import SplitConfig
from edgevision.layout import V1_LAYOUT
from edgevision.live.diagnose import feature_groups
from edgevision.live.recordings import RecordingsFile, load_recordings
from edgevision.runs import find_completed_runs
from edgevision.runtime import OrtModel
from edgevision.train.config import AugmentConfig, load_train_config
from edgevision.train.loop import run_training
from edgevision.web_export import export_web
from live_support import build_motion_dataset, synthetic_recordings
from training_support import EXPERIMENTS, load_script

K_TRUE = 0.4
SPLIT = SplitConfig(n_test_signers=1, n_val_signers=1)


@pytest.fixture(scope="module")
def diagnosed(tmp_path_factory):
    root = tmp_path_factory.mktemp("live")
    data_root = build_motion_dataset(root / "asl-signs")
    processed = root / "processed"
    config = root / "preprocess.yaml"
    config.write_text(
        "preprocess:\n"
        f"  data_root: {data_root.as_posix()}\n"
        f"  output_dir: {processed.as_posix()}\n"
        "  num_workers: 1\n"
        "split:\n"
        f"  n_test_signers: {SPLIT.n_test_signers}\n"
        f"  n_val_signers: {SPLIT.n_val_signers}\n",
        encoding="utf-8",
    )
    assert conftest.load_cli_module().main(["--config", str(config), "--no-cache"]) == 0
    cfg = load_train_config(
        EXPERIMENTS / "train_cnn.yaml",
        processed_dir=processed,
        output_root=root / "runs",
        epochs=6,
    ).model_copy(
        update={
            "batch_size": 16,
            "lr": 3e-3,
            "split": SPLIT,
            "augment": AugmentConfig(affine_p=0.0, temporal_p=0.0, jitter_p=0.0),
        }
    )
    run_training(cfg, seed=0, device="cpu", log=lambda _: None)
    (run,) = find_completed_runs(root / "runs")
    result = export_web(
        run.directory,
        root / "quant",
        processed,
        out_dir=root / "model",
        golden_dir=root / "golden",
        landmark_blocks=conftest.landmark_blocks,
        long_frame=conftest.long_frame,
        data_source="synthetic",
        work_dir=root / "work",
    )
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    model = OrtModel(result.model_path, intra_op_threads=1)
    recordings = synthetic_recordings(meta, model.run, k_true=K_TRUE)
    recordings_path = root / "data" / "live" / "recordings.json"
    recordings_path.parent.mkdir(parents=True)
    recordings_path.write_text(recordings.model_dump_json(), encoding="utf-8")
    args = [
        "--recordings",
        str(recordings_path),
        "--meta",
        str(result.meta_path),
        "--model",
        str(result.model_path),
        "--processed-dir",
        str(processed),
        "--output-dir",
        str(root / "results" / "live"),
    ]
    code = load_script("live_diagnose").main(args)
    (report_path,) = (root / "results" / "live").glob("diagnosis_*.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return root, args, code, report, report_path, recordings_path, meta


def test_meta_carries_the_training_aspect_ratio(diagnosed):
    *_, meta = diagnosed
    assert meta["aspect"]["r_train"] == pytest.approx(0.4, rel=0.05)
    assert meta["aspect"]["source"].endswith("preprocess_report.json")


def test_parity_with_python_preprocessing(diagnosed):
    _, _, code, report, *_ = diagnosed
    parity = report["parity"]
    assert code == 0
    assert parity["passed"] is True and parity["checked"] == report["recordings"]["count"] == 20
    assert parity["max_abs_diff"] <= 1e-4 and parity["info_mismatches"] == []


def test_offline_calibration_recovers_k(diagnosed):
    _, _, _, report, *_ = diagnosed
    (session,) = report["sessions"]
    assert session["k_offline"] == pytest.approx(K_TRUE, rel=0.05)
    assert session["app_k"] == [1.0] and session["app_calibration_modes"] == ["off"]
    assert report["live"]["r_live"]["median"] == pytest.approx(0.4 / K_TRUE, rel=0.05)


def test_correction_restores_accuracy(diagnosed):
    _, _, _, report, *_ = diagnosed
    conditions = report["conditions"]
    none, app, offline = conditions["none"], conditions["app"], conditions["offline"]
    print(
        f"top-1 none {none['top1']:.2f}, app {app['top1']:.2f}, offline {offline['top1']:.2f}; "
        f"distinct predictions none {none['distinct_predictions']}, offline "
        f"{offline['distinct_predictions']}"
    )
    assert app["top1"] == none["top1"]  # the app recorded with correction off
    assert offline["top1"] >= 0.8
    assert offline["top1"] > none["top1"] + 0.2
    assert offline["evaluated"] == 20 and none["norm_method_fractions"] == {"shoulders": 1.0}


def test_feature_shift_shrinks_with_correction(diagnosed):
    _, _, _, report, *_ = diagnosed
    shift = {name: report["conditions"][name]["feature_shift"] for name in ("none", "offline")}
    assert set(shift["none"]) == {"coordinates", "presence", "velocity"}
    coordinates = shift["none"]["coordinates"]
    assert coordinates["mean_abs_z"] > 2 * shift["offline"]["coordinates"]["mean_abs_z"]
    assert any(item["feature"].endswith(".y") for item in coordinates["top"])


def test_reports_are_written(diagnosed):
    _, _, _, report, report_path, *_ = diagnosed
    markdown = report_path.with_suffix(".md").read_text(encoding="utf-8")
    for heading in (
        "## Parity",
        "## Accuracy by aspect correction",
        "## Feature shift",
        "## Recordings",
    ):
        assert heading in markdown
    assert report["training"]["split_source"].startswith("train split")
    assert report["live"]["shoulders_visible_fraction"] == 1.0
    assert len(report["per_recording"]) == 20


def test_parity_failure_exits_nonzero(diagnosed, tmp_path):
    _, args, _, _, _, recordings_path, _ = diagnosed
    payload = json.loads(recordings_path.read_text(encoding="utf-8"))
    payload["recordings"][3]["app_result"]["features"][5][0] += 0.01
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    load_recordings(tampered)
    changed = [*args]
    changed[changed.index("--recordings") + 1] = str(tampered)
    changed[changed.index("--output-dir") + 1] = str(tmp_path / "out")
    assert load_script("live_diagnose").main(changed) == 1
    (report_path,) = (tmp_path / "out").glob("diagnosis_*.json")
    parity = json.loads(report_path.read_text(encoding="utf-8"))["parity"]
    assert parity["passed"] is False and parity["max_abs_diff"] == pytest.approx(0.01, rel=1e-3)


def test_recordings_from_another_model_are_refused(diagnosed, tmp_path):
    _, args, *_, recordings_path, _ = diagnosed
    payload = json.loads(recordings_path.read_text(encoding="utf-8"))
    payload["app"]["model_sha256"] = "0" * 64
    other = tmp_path / "other.json"
    other.write_text(json.dumps(payload), encoding="utf-8")
    RecordingsFile.model_validate(payload)
    changed = [*args]
    changed[changed.index("--recordings") + 1] = str(other)
    with pytest.raises(ValueError, match="another model"):
        load_script("live_diagnose").main(changed)


def test_feature_groups_cover_the_feature_block():
    groups = feature_groups("v1", V1_LAYOUT)
    assert [name for name, *_ in groups] == ["coordinates", "presence", "velocity"]
    assert groups[-1][1].stop == 198 and all(
        len(names) == part.stop - part.start for _, part, names in groups
    )
