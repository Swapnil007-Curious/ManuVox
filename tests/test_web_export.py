"""Browser export: threshold selection, the variant rule and its onnxruntime-web fallback,
meta.json and golden-file schemas, aspect statistics, the val-only data access, and
consistency of the committed web/ model files."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydantic
import pytest

import conftest
from edgevision import web_export
from edgevision.data.landmarks import pose_swap_pairs
from edgevision.data.preprocess import preprocess_sequence, snap_x
from edgevision.layout import NUM_FRAMES, NUM_LANDMARKS, InputLayout, layout_dict
from edgevision.live.golden import AspectGolden, build_aspect_golden
from edgevision.live.recordings import RecordingsFile, raw_frames, replay
from edgevision.quant.pipeline import quantize_run
from edgevision.runs import CHECKPOINT_FILE, CompletedRun, find_completed_runs, run_dir
from edgevision.runtime import OrtModel
from edgevision.train.config import AugmentConfig, load_train_config
from edgevision.train.data import read_splits
from edgevision.train.loop import run_training
from edgevision.web_export import (
    FALLBACK_MIN_COVERAGE,
    MODEL_CASES,
    CandidateInput,
    Fallback,
    ModelGolden,
    PreprocessGolden,
    WasmCheck,
    WebMeta,
    aspect_settings,
    export_web,
    load_eval_split,
    make_selection,
    pack_features,
    rank_candidates,
    rows_from_json,
    select_threshold,
    sha256_file,
    softmax,
    threshold_from_logits,
)
from quant_support import FIXTURE_BATCH, FIXTURE_EPOCHS, FIXTURE_LR, FIXTURE_SPLIT
from training_support import EXPERIMENTS, REPO_ROOT, build_processed_dir, load_script

WEB_MODEL_DIR = REPO_ROOT / "web" / "public" / "model"
WEB_GOLDEN_DIR = REPO_ROOT / "web" / "tests" / "golden"
PREPROCESS_GOLDENS = ("preprocess_cases.json", "preprocess_cases_v2.json")


# ---------------------------------------------------------------------------------------------
# Threshold selection


def test_threshold_is_smallest_reaching_target():
    p = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([True, True, False, True, True, False])
    block = select_threshold(p, correct, target_precision=0.75)
    # Accuracy above t: 0.5 -> 4/6, 0.6 -> 4/5 (first to reach 0.75).
    assert block.threshold == 0.6
    assert block.coverage == pytest.approx(5 / 6)
    assert block.accuracy_above_threshold == pytest.approx(0.8)
    assert block.overall_val_top1 == pytest.approx(4 / 6)
    assert (block.method, block.warning, block.computed_on) == ("target_precision", None, "val")
    assert block.val_sequences == 6


def test_threshold_ties_accept_every_equal_probability():
    block = select_threshold(np.array([0.7, 0.7, 0.9]), np.array([True, False, True]), 0.9)
    assert block.threshold == 0.9
    assert block.coverage == pytest.approx(1 / 3)
    assert block.accuracy_above_threshold == 1.0


def test_threshold_accepts_everything_when_overall_accuracy_suffices():
    p = np.array([0.4, 0.8, 0.6])
    block = select_threshold(p, np.array([True, True, True]), 0.9)
    assert (block.threshold, block.coverage) == (0.4, 1.0)


def test_threshold_fallback_maximises_accuracy_above_min_coverage():
    # 20 samples, most confident first: wrong, right, right, then 17 wrong. No t reaches 0.9.
    p = np.linspace(0.99, 0.30, 20)
    correct = np.zeros(20, dtype=bool)
    correct[[1, 2]] = True
    block = select_threshold(p, correct, target_precision=0.9)
    # Accuracy 2/3 at coverage 0.15 is ignored (below 0.2); 2/4 at coverage exactly 0.2 wins.
    assert block.method == "fallback_max_accuracy"
    assert block.threshold == p[3]
    assert block.coverage == pytest.approx(FALLBACK_MIN_COVERAGE)
    assert block.accuracy_above_threshold == pytest.approx(0.5)
    assert block.warning is not None and "0.90" in block.warning
    assert block.fallback_min_coverage == FALLBACK_MIN_COVERAGE


def test_threshold_fallback_prefers_smaller_threshold_on_ties():
    p = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([False, True, False, True])  # accuracy 0, 1/2, 1/3, 2/4
    block = select_threshold(p, correct, target_precision=0.9)
    assert block.method == "fallback_max_accuracy"
    assert (block.threshold, block.coverage) == (0.6, 1.0)


@pytest.mark.parametrize(
    ("p", "correct", "target"),
    [
        (np.array([]), np.array([], dtype=bool), 0.9),
        (np.array([0.5, 0.6]), np.array([True]), 0.9),
        (np.array([0.5]), np.array([True]), 0.0),
        (np.array([0.5]), np.array([True]), 1.5),
    ],
)
def test_threshold_rejects_bad_input(p, correct, target):
    with pytest.raises(ValueError):
        select_threshold(p, correct, target)


def test_threshold_from_logits_uses_softmax_max_probability():
    logits = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]], dtype=np.float32)
    labels = np.array([0, 1, 0])
    probabilities = softmax(logits)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    block = threshold_from_logits(logits, labels, target_precision=0.99)
    assert block.threshold == pytest.approx(probabilities[1].max())
    assert block.coverage == pytest.approx(2 / 3)
    assert block.overall_val_top1 == pytest.approx(2 / 3)


def test_default_target_precision_is_0_80():
    assert web_export.DEFAULT_TARGET_PRECISION == 0.80


# ---------------------------------------------------------------------------------------------
# Variant rule


def inputs(**variants):
    """CandidateInput per variant from (bytes, val top-1) pairs; None marks a missing one."""
    built = {}
    for name, value in variants.items():
        if value is None:
            built[name] = CandidateInput(None, None, None, "not built")
        else:
            size, top1 = value
            built[name] = CandidateInput(Path(f"{name}.onnx"), size, top1, None)
    return built


def test_rule_picks_the_smallest_file_within_half_a_point():
    candidates, order, fp32 = rank_candidates(
        inputs(
            fp32=(1000, 0.80),
            fp16=(500, 0.798),  # 0.2 pp
            int8_dynamic=(250, 0.79),  # 1.0 pp: too far
            int8_static=(300, 0.797),  # 0.3 pp
        )
    )
    assert fp32 == 0.80
    assert order == ["int8_static", "fp16", "fp32"]
    by_name = {c.variant: c for c in candidates}
    assert [c.variant for c in candidates] == ["fp32", "fp16", "int8_dynamic", "int8_static"]
    assert not by_name["int8_dynamic"].eligible
    assert "1.00 pp below fp32" in (by_name["int8_dynamic"].note or "")
    assert by_name["fp16"].drop_pp == pytest.approx(0.2)
    assert all(c.wasm.status == "not_run" for c in candidates)


@pytest.mark.parametrize(
    ("top1", "eligible"),
    [
        (0.845, True),  # exactly 0.5 pp, despite float noise
        (0.844, False),  # 0.6 pp
        (0.86, True),  # better than fp32
    ],
)
def test_rule_limit_is_inclusive(top1, eligible):
    _, order, _ = rank_candidates(inputs(fp32=(1000, 0.85), fp16=(500, top1)))
    assert ("fp16" in order) == eligible
    assert order[-1] == "fp32"


def test_rule_skips_unavailable_variants_and_breaks_size_ties_in_variant_order():
    candidates, order, _ = rank_candidates(
        inputs(fp32=(400, 0.7), fp16=(400, 0.7), int8_dynamic=None)
    )
    assert order == ["fp32", "fp16"]
    missing = {c.variant: c for c in candidates}
    assert not missing["int8_dynamic"].available and missing["int8_dynamic"].note == "not built"
    assert not missing["int8_static"].available


def test_rule_requires_fp32():
    with pytest.raises(ValueError, match="fp32"):
        rank_candidates(inputs(fp32=None, fp16=(500, 0.8)))


def test_selection_records_fallbacks_and_whether_the_rule_was_followed():
    candidates, order, fp32 = rank_candidates(inputs(fp32=(1000, 0.8), fp16=(500, 0.8)))
    failed = WasmCheck(status="failed", detail="session creation failed")
    passed = WasmCheck(status="passed", detail="ok")
    selection = make_selection(
        candidates,
        order,
        fp32,
        "quant_report",
        chosen="fp32",
        forced=False,
        fallbacks=[Fallback(variant="fp16", reason="session creation failed")],
        checks={"fp16": failed, "fp32": passed},
    )
    assert (selection.recommended, selection.chosen, selection.follows_rule) == (
        "fp16",
        "fp32",
        True,
    )
    assert [c.wasm.status for c in selection.candidates[:2]] == ["passed", "failed"]
    forced = make_selection(
        candidates, order, fp32, "quant_report", chosen="fp32", forced=True, fallbacks=[], checks={}
    )
    assert (forced.forced, forced.follows_rule) == (True, False)


# ---------------------------------------------------------------------------------------------
# Export of a trained fixture run


@dataclass(frozen=True)
class Exported:
    run: CompletedRun
    processed: Path
    quant_root: Path
    out: Path
    golden: Path
    result: web_export.ExportResult

    def run_val_ids(self) -> list[int]:
        splits = read_splits(self.run.directory / self.run.summary["splits_file"])
        return list(splits["val"])


def train_cnn_run(root):
    processed = build_processed_dir(root / "processed", write_splits=False)
    runs_root = root / "runs"
    cfg = load_train_config(
        EXPERIMENTS / "train_cnn.yaml",
        processed_dir=processed.root,
        output_root=runs_root,
        epochs=FIXTURE_EPOCHS,
    ).model_copy(
        update={
            "batch_size": FIXTURE_BATCH,
            "lr": FIXTURE_LR,
            "split": FIXTURE_SPLIT,
            "augment": AugmentConfig(affine_p=0.0, temporal_p=0.0, jitter_p=0.0),
        }
    )
    run_training(cfg, seed=0, device="cpu", log=lambda _: None)
    (run,) = find_completed_runs(runs_root)
    return processed.root, run


def run_export(exported_root, run, processed, quant_root, **overrides):
    kwargs = {
        "out_dir": exported_root / "model",
        "golden_dir": exported_root / "golden",
        "landmark_blocks": conftest.landmark_blocks,
        "long_frame": conftest.long_frame,
        "data_source": "synthetic",
        "work_dir": exported_root / "work",
    }
    kwargs.update(overrides)
    return export_web(run.directory, quant_root, processed, **kwargs)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("web_export")
    processed, run = train_cnn_run(root)
    quant_root = root / "quant"
    summary = run.summary
    out_dir = run_dir(quant_root, summary["model"], summary["split_mode"], summary["seed"])
    report = quantize_run(
        run, processed, out_dir, variants=("fp32", "fp16", "int8_static"), log=lambda _: None
    )
    assert report["failures"] == []
    export_root = root / "export"
    result = run_export(export_root, run, processed, quant_root)
    return Exported(
        run, processed, quant_root, export_root / "model", export_root / "golden", result
    )


def test_meta_json_schema(exported):
    meta = WebMeta.model_validate_json((exported.out / "meta.json").read_text("utf-8"))
    model_path = exported.out / "model.onnx"
    assert meta == exported.result.meta
    assert (meta.model_sha256, meta.model_bytes) == (
        sha256_file(model_path),
        model_path.stat().st_size,
    )
    checkpoint = exported.run.directory / CHECKPOINT_FILE
    assert [item.model_dump() for item in meta.source_checkpoints] == [
        {"run": "cnn_signer_independent/seed_0", "sha256": sha256_file(checkpoint)}
    ]
    assert (meta.seed, meta.seeds, meta.ensemble, meta.feature_set) == (0, [0], False, "v1")
    assert meta.labels == list(conftest.SIGNS)
    assert (meta.model_name, meta.data_source) == ("cnn", "synthetic")
    assert meta.input_layout.model_dump() == layout_dict()
    assert meta.num_frames == NUM_FRAMES and len(meta.landmark_order) == NUM_LANDMARKS
    assert meta.landmark_order[0] == "left_hand_0" and meta.landmark_order[-1] == "pose_16"
    assert meta.landmark_order[21] == "right_hand_0"
    assert meta.preprocessing.min_hand_frames == 3 and meta.preprocessing.use_z is False
    assert meta.preprocessing.pose_swap_pairs == [(11, 12), (13, 14), (15, 16)]
    assert "PreprocessConfig defaults" in meta.preprocessing.source
    assert meta.aspect is None  # synthetic processed dir without a preprocess report
    assert meta.opset == 17
    assert meta.threshold.computed_on == "val"
    assert meta.threshold.target_precision == 0.80
    assert meta.threshold.val_sequences == len(exported.run_val_ids())


def test_rule_choice_and_every_candidate_are_recorded(exported):
    selection = exported.result.meta.variant_selection
    assert selection.source == "quant_report"
    assert [c.variant for c in selection.candidates] == list(web_export.WEB_VARIANTS)
    by_name = {c.variant: c for c in selection.candidates}
    assert by_name["int8_dynamic"].note == "not in quant_report.json"
    assert all(by_name[name].available for name in ("fp32", "fp16", "int8_static"))
    eligible = sorted(
        (c for c in selection.candidates if c.eligible),
        key=lambda c: (c.file_bytes, web_export.WEB_VARIANTS.index(c.variant)),
    )
    assert selection.order == [c.variant for c in eligible]
    assert exported.result.meta.variant == selection.chosen == selection.order[0]
    assert (selection.forced, selection.fallbacks, selection.follows_rule) == (False, [], True)
    sizes = {c.variant: c.file_bytes for c in selection.candidates}
    assert sizes["fp16"] < sizes["fp32"]


def test_meta_schema_rejects_inconsistent_files(exported):
    payload = json.loads((exported.out / "meta.json").read_text("utf-8"))
    other = next(v for v in web_export.WEB_VARIANTS if v != payload["variant"])
    pre = payload["preprocessing"]
    for broken in (
        {**payload, "labels": payload["labels"][:-1]},
        {**payload, "unexpected": 1},
        {**payload, "model_sha256": "not-a-hash"},
        {**payload, "num_frames": 16},
        {**payload, "variant": other},
        {**payload, "preprocessing": {**pre, "pose_swap_pairs": pre["pose_swap_pairs"][:-1]}},
        {**payload, "ensemble": True},
        {**payload, "schema_version": 1},
    ):
        with pytest.raises(pydantic.ValidationError):
            WebMeta.model_validate(broken)


def verifier(failing):
    """A wasm check that fails for the variants in ``failing`` and records every call."""
    calls = []

    def check(result):
        calls.append(result.meta.variant)
        if result.meta.variant in failing:
            return WasmCheck(status="failed", detail=f"{result.meta.variant} did not load")
        return WasmCheck(status="passed", detail="model parity: ok")

    check.calls = calls
    return check


def test_a_variant_the_browser_cannot_run_falls_back_to_the_next(exported, tmp_path):
    order = exported.result.meta.variant_selection.order
    assert len(order) >= 2, "the fixture needs two eligible variants"
    check = verifier({order[0]})
    result = run_export(
        tmp_path, exported.run, exported.processed, exported.quant_root, wasm_verifier=check
    )
    selection = result.meta.variant_selection
    assert check.calls == order[:2]
    assert result.meta.variant == selection.chosen == order[1]
    assert selection.fallbacks == [Fallback(variant=order[0], reason=f"{order[0]} did not load")]
    assert selection.follows_rule
    statuses = {c.variant: c.wasm.status for c in selection.candidates}
    assert (statuses[order[0]], statuses[order[1]]) == ("failed", "passed")
    on_disk = WebMeta.model_validate_json((tmp_path / "model" / "meta.json").read_text("utf-8"))
    assert on_disk == result.meta
    golden = ModelGolden.model_validate_json((tmp_path / "golden" / "model_cases.json").read_text())
    assert (golden.variant, golden.model_sha256) == (order[1], result.meta.model_sha256)


def test_export_fails_when_no_eligible_variant_runs_in_the_browser(exported, tmp_path):
    order = exported.result.meta.variant_selection.order
    with pytest.raises(RuntimeError, match="could not run any eligible variant"):
        run_export(
            tmp_path,
            exported.run,
            exported.processed,
            exported.quant_root,
            wasm_verifier=verifier(set(order)),
        )


def test_a_forced_variant_is_exported_and_recorded(exported, tmp_path):
    result = run_export(
        tmp_path, exported.run, exported.processed, exported.quant_root, variant="fp32"
    )
    selection = result.meta.variant_selection
    assert (result.meta.variant, selection.forced) == ("fp32", True)
    assert selection.follows_rule == (selection.recommended == "fp32")
    with pytest.raises(ValueError, match="int8_dynamic is not available"):
        run_export(
            tmp_path, exported.run, exported.processed, exported.quant_root, variant="int8_dynamic"
        )


def test_golden_files_schema(exported):
    preprocess = PreprocessGolden.model_validate_json(
        (exported.golden / "preprocess_cases.json").read_text("utf-8")
    )
    infos = [case.info for case in preprocess.cases]
    assert len(preprocess.cases) >= 12
    assert any(info.mirrored for info in infos)
    assert any(info.norm_method == "hand_fallback" and not info.excluded for info in infos)
    assert {info.reason for info in infos} >= {"insufficient_hand_frames", "degenerate_scale"}
    kept = [info for info in infos if not info.excluded]
    assert min(info.t_kept for info in kept) < NUM_FRAMES < max(info.t_kept for info in kept)
    assert any(info.t_kept < info.t_raw for info in kept)
    names = {case.name for case in preprocess.cases}
    assert {"right_only_longer", "both_hands_tie_shorter", "left_only_mirrored"} <= names

    v2 = PreprocessGolden.model_validate_json(
        (exported.golden / "preprocess_cases_v2.json").read_text("utf-8")
    )
    assert v2.config.pose_indices == list(range(17)) and len(v2.config.pose_swap_pairs) == 8
    assert [case.name for case in v2.cases] == [case.name for case in preprocess.cases]
    assert len(v2.cases[0].features[0]) == InputLayout.for_pose_indices(range(17)).frame_dim

    model = ModelGolden.model_validate_json(
        (exported.golden / "model_cases.json").read_text("utf-8")
    )
    assert len(model.cases) == MODEL_CASES
    variant = exported.result.meta.variant
    assert (model.source, model.split, model.variant, model.frame_dim) == (
        "synthetic",
        "val",
        variant,
        100,
    )
    assert model.model_sha256 == exported.result.meta.model_sha256
    assert {case.sequence_id for case in model.cases} <= set(exported.run_val_ids())
    inputs = np.array([case.input for case in model.cases], dtype=np.float32)
    logits = OrtModel(exported.out / "model.onnx", intra_op_threads=1).run(inputs)
    stored = np.array([case.logits for case in model.cases], dtype=np.float32)
    np.testing.assert_array_equal(stored, logits)


def test_golden_schema_rejects_bad_shapes(exported):
    payload = json.loads((exported.golden / "model_cases.json").read_text("utf-8"))
    short = {**payload, "cases": payload["cases"][:-1]}
    with pytest.raises(pydantic.ValidationError):
        ModelGolden.model_validate(short)
    case = payload["cases"][0]
    with pytest.raises(pydantic.ValidationError):
        ModelGolden.model_validate(
            {**payload, "cases": [{**case, "input": case["input"][:-1]}, *payload["cases"][1:]]}
        )
    with pytest.raises(pydantic.ValidationError):
        ModelGolden.model_validate({**payload, "frame_dim": 120})


def check_preprocess_golden(golden: PreprocessGolden) -> None:
    """Re-run Python preprocessing on every stored input and compare with the stored outputs."""
    config = golden.config
    assert config.pose_swap_pairs == list(pose_swap_pairs(config.pose_indices))
    width = InputLayout.for_pose_indices(config.pose_indices).frame_dim
    for case in golden.cases:
        result = preprocess_sequence(
            rows_from_json(case.rows),
            num_frames=config.num_frames,
            use_z=False,
            pose_indices=config.pose_indices,
            min_hand_frames=config.min_hand_frames,
        )
        assert dict(result.info) == case.info.model_dump(), case.name
        if result.features is None or result.hand_presence is None:
            assert case.features is None, case.name
            continue
        packed = pack_features(result.features, result.hand_presence)
        assert packed.shape == (NUM_FRAMES, width)
        np.testing.assert_array_equal(
            np.array(case.features, dtype=np.float32), packed, err_msg=case.name
        )
        np.testing.assert_array_equal(
            np.array(case.presence, dtype=np.float32), result.hand_presence, err_msg=case.name
        )
    xs = np.array([item.x for item in golden.snap_x_cases])
    np.testing.assert_array_equal(snap_x(xs), [item.snapped for item in golden.snap_x_cases])


@pytest.mark.parametrize("name", PREPROCESS_GOLDENS)
def test_golden_preprocessing_is_reproducible(exported, name):
    check_preprocess_golden(
        PreprocessGolden.model_validate_json((exported.golden / name).read_text("utf-8"))
    )


def check_recording_example(path: Path) -> None:
    """The example validates and its classified segment's features are Python's replay."""
    example = RecordingsFile.model_validate_json(path.read_text("utf-8"))
    statuses = [recording.app_result.status for recording in example.recordings]
    assert statuses == ["classified", "discarded"]
    recording = example.recordings[0]
    result = replay(
        raw_frames(recording.frames),
        recording.calibration.k,
        pose_indices=example.app.pose_indices,
        num_frames=example.app.num_frames,
        min_hand_frames=recording.min_hand_frames,
    )
    assert result.features is not None and result.hand_presence is not None
    packed = pack_features(result.features, result.hand_presence)
    np.testing.assert_array_equal(np.array(recording.app_result.features, np.float32), packed)


def test_aspect_golden_and_recording_example(exported):
    aspect = AspectGolden.model_validate_json(
        (exported.golden / "aspect_cases.json").read_text("utf-8")
    )
    assert aspect == build_aspect_golden(conftest.landmark_blocks)
    check_recording_example(exported.golden / "recording_example.json")


def test_export_reads_only_val(exported, tmp_path, monkeypatch):
    calls = []
    original = web_export.load_split_arrays

    def spy(processed_dir, splits_path, split_names, allow_test=False):
        calls.append((list(split_names), allow_test))
        return original(processed_dir, splits_path, split_names, allow_test)

    monkeypatch.setattr(web_export, "load_split_arrays", spy)
    run_export(tmp_path, exported.run, exported.processed, exported.quant_root)
    assert calls == [(["val"], False)]


@pytest.mark.parametrize("split", ["test", "train"])
def test_export_refuses_other_splits(exported, split):
    splits_path = exported.run.directory / exported.run.summary["splits_file"]
    with pytest.raises(ValueError, match="only the val split"):
        load_eval_split(exported.processed, splits_path, split)


def test_export_refuses_variants_from_another_checkpoint(exported, tmp_path):
    run_copy = tmp_path / "runs" / exported.run.directory.parent.name / exported.run.directory.name
    shutil.copytree(exported.run.directory, run_copy)
    with (run_copy / CHECKPOINT_FILE).open("ab") as handle:
        handle.write(b"\0")
    copy = CompletedRun(run_copy, exported.run.summary)
    with pytest.raises(ValueError, match="no fp32 variant"):
        run_export(tmp_path / "out", copy, exported.processed, exported.quant_root)


def test_cli_prints_the_candidates_and_the_choice(exported, tmp_path, capsys):
    code = load_script("export_web").main(
        [
            "--run",
            str(exported.run.directory),
            "--quant-root",
            str(exported.quant_root),
            "--processed-dir",
            str(exported.processed),
            "--out",
            str(tmp_path / "model"),
            "--golden-dir",
            str(tmp_path / "golden"),
            "--work-dir",
            str(tmp_path / "work"),
            "--data-source",
            "synthetic",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    for text in ("variant rule (quant_report)", "int8_dynamic", "order (smallest eligible first)"):
        assert text in output
    assert "threshold" in output and "coverage" in output and "target precision 0.80" in output
    meta = WebMeta.model_validate_json((tmp_path / "model" / "meta.json").read_text("utf-8"))
    assert f"chosen: {meta.variant}" in output


# ---------------------------------------------------------------------------------------------
# Aspect statistics


def write_report(directory: Path, report: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "preprocess_report.json").write_text(json.dumps(report), encoding="utf-8")
    return directory


def report(pose_indices, aspect=True, kept=90):
    payload = {
        "counts": {"total": 100, "kept": kept, "excluded": 100 - kept},
        "per_sign_kept": {"min": 1, "median": 2.0, "max": 3},
        "mirrored_fraction": 0.4,
        "norm_method_fractions": {"shoulders": 1.0},
        "config": {
            "preprocess": {
                "data_root": "/data/asl-signs",
                "num_frames": 32,
                "use_z": False,
                "pose_indices": list(pose_indices),
                "min_hand_frames": 3,
                "limit": None,
                "seed": 42,
            },
            "split": {"mode": "signer_independent", "seed": 42},
        },
    }
    if aspect:
        payload["aspect_stats"] = {"median": 0.4, "p10": 0.34, "p90": 0.46}
    return payload


V1 = (0, 11, 12, 13, 14, 15, 16)


def test_aspect_comes_from_the_processed_report(tmp_path):
    processed = write_report(tmp_path / "v1", report(V1))
    block = aspect_settings(processed, None, "asl-signs")
    assert block is not None and block.r_train == 0.4 and (block.k_min, block.k_max) == (0.25, 4.0)
    assert block.stats["p90"] == 0.46 and block.calibration_seconds == 2.0


def test_aspect_from_a_compatible_report_of_the_same_raw_data(tmp_path):
    processed = write_report(tmp_path / "v1", report(V1, aspect=False))
    other = write_report(tmp_path / "v2", report(range(17))) / "preprocess_report.json"
    block = aspect_settings(processed, other, "asl-signs")
    assert block is not None and block.r_train == 0.4 and "same raw data" in block.source


def test_aspect_refuses_an_incompatible_report(tmp_path):
    processed = write_report(tmp_path / "v1", report(V1, aspect=False))
    other = write_report(tmp_path / "v2", report(range(17), kept=80)) / "preprocess_report.json"
    with pytest.raises(ValueError, match="counts differs"):
        aspect_settings(processed, other, "asl-signs")
    no_nose = write_report(tmp_path / "v3", report((11, 12))) / "preprocess_report.json"
    with pytest.raises(ValueError, match=r"does not select pose \[0\]"):
        aspect_settings(processed, no_nose, "asl-signs")


def test_aspect_is_required_for_real_data_only(tmp_path):
    processed = write_report(tmp_path / "v1", report(V1, aspect=False))
    with pytest.raises(ValueError, match="has no aspect_stats"):
        aspect_settings(processed, None, "asl-signs")
    assert aspect_settings(processed, None, "synthetic") is None


# ---------------------------------------------------------------------------------------------
# Committed files in web/


@pytest.mark.parametrize("name", PREPROCESS_GOLDENS)
def test_committed_golden_preprocessing_matches_python(name):
    path = WEB_GOLDEN_DIR / name
    if not path.exists():
        pytest.skip(f"web/tests/golden/{name} not generated")
    check_preprocess_golden(PreprocessGolden.model_validate_json(path.read_text("utf-8")))


def test_committed_aspect_golden_and_recording_example_match_python():
    aspect_path = WEB_GOLDEN_DIR / "aspect_cases.json"
    if not aspect_path.exists():
        pytest.skip("web/tests/golden/aspect_cases.json not generated")
    golden = AspectGolden.model_validate_json(aspect_path.read_text("utf-8"))
    assert golden == build_aspect_golden(conftest.landmark_blocks)
    check_recording_example(WEB_GOLDEN_DIR / "recording_example.json")


def test_committed_model_files_are_consistent():
    meta_path = WEB_MODEL_DIR / "meta.json"
    if not meta_path.exists():
        pytest.skip("web/public/model/meta.json not exported")
    meta = WebMeta.model_validate_json(meta_path.read_text("utf-8"))
    assert meta.input_layout.model_dump() == layout_dict(meta.pose_indices)
    golden = ModelGolden.model_validate_json(
        (WEB_GOLDEN_DIR / "model_cases.json").read_text("utf-8")
    )
    assert (golden.model_sha256, golden.variant) == (meta.model_sha256, meta.variant)
    assert golden.num_classes == meta.num_classes
    assert golden.frame_dim == meta.input_layout.frame_dim
    example = RecordingsFile.model_validate_json(
        (WEB_GOLDEN_DIR / "recording_example.json").read_text("utf-8")
    )
    assert example.app.model_sha256 == meta.model_sha256
    model_path = WEB_MODEL_DIR / meta.model_file
    if model_path.exists():
        assert sha256_file(model_path) == meta.model_sha256
        assert model_path.stat().st_size == meta.model_bytes
