"""Evaluate every completed run on the test split, once model selection is done on val.

This is the only code path that loads the test split. Every invocation is appended to
results/final/test_access_log.json before any test data is read.

Model selection is by experiment (val_summary.csv); every table carries an experiment column.
Experiments with at least two completed seeds also get a seed ensemble (mean of the members'
logits) scored on test, in test_ensemble.csv. A run whose best.pt is absent (a summary.json
committed without its checkpoint) is listed and left out.

--processed-dir takes one or more directories; each run reads the first one whose landmark
layout matches its checkpoint, so v1 and v2 runs can be evaluated together.

With --quant-root, the ONNX variants that scripts/quantize.py wrote for each signer-independent
run with v1 features are evaluated instead of the checkpoints.

With --predictions-out DIR, nothing is re-scored and no table is rewritten: the per-sequence test
logits of every signer-independent run with a checkpoint (and, with --onnx-model, of that ONNX
file, e.g. the shipped web model, on the test split of --onnx-run) are saved to
DIR/<experiment>/seed_<s>.npz (sequence_id, participant_id, y, logits), for the paired model
comparisons and the test-set calibration of scripts/test_analysis.py. Model selection is already
done on val; these files are only read for reporting. The access is logged like every other.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from edgevision.data.io import write_json
from edgevision.evaluation.evaluators import Evaluator, OnnxEvaluator, TorchCheckpointEvaluator
from edgevision.evaluation.metrics import top_k_accuracy
from edgevision.evaluation.tables import (
    PRIMARY_SPLIT_MODE,
    append_access_log,
    ensemble_row,
    ensemble_scores,
    ensemble_table,
    per_participant_table,
    score_test_split,
    select_best_experiment,
    summarize_quant_test,
    summarize_test,
    top_confusions,
    with_experiment,
)
from edgevision.models.registry import checkpoint_layout
from edgevision.runs import (
    CHECKPOINT_FILE,
    QUANT_VARIANTS,
    CompletedRun,
    find_completed_runs,
    variant_file,
)
from edgevision.runtime import OrtModel
from edgevision.train.data import SplitArrays, load_split_arrays, matching_processed_dir
from edgevision.train.loop import resolve_device

ACCESS_LOG = "test_access_log.json"
QUANT_DIR = "quant"
# Largest allowed gap between the fp32 ONNX and torch test top-1: 0.1 percentage points.
SANITY_TOLERANCE = 0.001
MIN_ENSEMBLE_SEEDS = 2
QUANTISED_FEATURE_SETS = ("v1",)  # scripts/quantize.py builds variants of these runs only.


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--final-dir", type=Path, default=Path("results/final"))
    parser.add_argument(
        "--processed-dir",
        type=Path,
        nargs="+",
        help="override the processed_dir each run recorded; with several, each run uses the "
        "first whose landmark layout matches its checkpoint",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--quant-root",
        type=Path,
        help="evaluate the ONNX variants under this directory (scripts/quantize.py output)",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        help="save per-sequence test logits of every signer-independent run here (no tables)",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        help="with --predictions-out: also save this ONNX file's test logits "
        "(e.g. web/public/model/model.onnx)",
    )
    parser.add_argument(
        "--onnx-run",
        help="with --onnx-model: the run (<experiment>/seed_<s>) whose split file and "
        "processed dir it uses",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="with --onnx-model: save only the ONNX file's logits, not the checkpoints'",
    )
    return parser.parse_args(argv)


def save_predictions(args: argparse.Namespace, runs: Sequence[CompletedRun]) -> int:
    """Per-sequence test logits for reporting (paired comparisons, calibration); no tables."""
    runs = [
        run
        for run in runs
        if run.split_mode == PRIMARY_SPLIT_MODE and (run.directory / CHECKPOINT_FILE).is_file()
    ]
    onnx_run = None
    if args.onnx_model is not None:
        onnx_run = next((run for run in runs if run.key == args.onnx_run), None)
        if onnx_run is None:
            print(f"error: --onnx-run {args.onnx_run!r} is not a completed run", file=sys.stderr)
            return 1
    keys = [] if args.skip_runs else [run.key for run in runs]
    if args.onnx_model is not None:
        digest = hashlib.sha256(args.onnx_model.read_bytes()).hexdigest()
        keys.append(f"onnx:{args.onnx_model.as_posix()}@{digest[:12]} (split of {args.onnx_run})")
    append_access_log(args.final_dir / ACCESS_LOG, keys)
    device = resolve_device(args.device)
    loaded: dict[tuple[str, str], SplitArrays] = {}

    def write(target: Path, test: SplitArrays, logits: npt.NDArray[np.float32], name: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            sequence_id=test.sequence_id,
            participant_id=test.participant_id,
            y=test.y,
            logits=logits.astype(np.float32),
        )
        top1 = top_k_accuracy(logits, test.y, 1)
        print(f"{name}: test top-1 {top1:.4f} (n={len(test.y)}) -> {target}", flush=True)

    for run in [] if args.skip_runs else runs:
        test = load_test_split(args, run, loaded)
        evaluator = TorchCheckpointEvaluator(
            run.directory / CHECKPOINT_FILE, device, args.batch_size
        )
        # run.key is <experiment>/seed_<s>, so this writes DIR/<experiment>/seed_<s>.npz.
        write(args.predictions_out / f"{run.key}.npz", test, evaluator.predict(test.X), run.key)
    if args.onnx_model is not None and onnx_run is not None:
        test = load_test_split(args, onnx_run, loaded)
        # OrtModel, not OnnxEvaluator: the web model carries the v2 layout in its metadata, and
        # its input is the processed arrays as they are (scripts/export_web.py scores val so).
        model = OrtModel(args.onnx_model, intra_op_threads=4)
        logits = np.concatenate(
            [
                model.run(np.ascontiguousarray(test.X[i : i + args.batch_size], dtype=np.float32))
                for i in range(0, len(test.X), args.batch_size)
            ]
        )
        write(args.predictions_out / "web_model.npz", test, logits, f"onnx {args.onnx_model}")
    return 0


def run_processed_dir(args: argparse.Namespace, run: CompletedRun) -> Path:
    """The processed dir a run reads: the one it recorded, or the first --processed-dir whose
    landmark layout matches the run's checkpoint."""
    if not args.processed_dir:
        return Path(run.summary["processed_dir"])
    if len(args.processed_dir) == 1 and not (run.directory / CHECKPOINT_FILE).is_file():
        return Path(args.processed_dir[0])  # Quantised runs may come without a checkpoint.
    layout = checkpoint_layout(run.directory / CHECKPOINT_FILE)
    match = matching_processed_dir(args.processed_dir, layout)
    if match is None:
        raise ValueError(
            f"{run.key}: no --processed-dir holds features with its pose_indices "
            f"{list(layout.pose_indices)}"
        )
    return match


def load_test_split(
    args: argparse.Namespace, run: CompletedRun, cache: dict[tuple[str, str], SplitArrays]
) -> SplitArrays:
    """The run's test split, loaded once per (processed dir, split file contents)."""
    processed_dir = run_processed_dir(args, run)
    splits_path = run.directory / run.summary["splits_file"]
    key = (str(processed_dir), hashlib.sha256(splits_path.read_bytes()).hexdigest())
    if key not in cache:
        cache[key] = load_split_arrays(processed_dir, splits_path, ["test"], allow_test=True)[
            "test"
        ]
    return cache[key]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs = find_completed_runs(args.runs_root)
    if not runs:
        print(f"error: no completed runs under {args.runs_root}", file=sys.stderr)
        return 1
    val_summary_path = args.tables_dir / "val_summary.csv"
    if not val_summary_path.exists():
        print(f"error: {val_summary_path} is missing; run aggregate_val.py first", file=sys.stderr)
        return 1
    val_summary = with_experiment(pd.read_csv(val_summary_path))
    best = select_best_experiment(val_summary)
    if not any(run.experiment == best for run in runs):
        print(
            f"error: {val_summary_path} picks {best}, which has no completed run", file=sys.stderr
        )
        return 1
    if args.quant_root is not None:
        return evaluate_variants(args, runs)
    if args.predictions_out is not None:
        return save_predictions(args, runs)

    without_checkpoint = [run for run in runs if not (run.directory / CHECKPOINT_FILE).is_file()]
    for run in without_checkpoint:
        print(f"{run.key}: no {CHECKPOINT_FILE}; left out of the test evaluation", flush=True)
    runs = [run for run in runs if run not in without_checkpoint]
    evaluable = {run.experiment for run in runs}
    if not evaluable:
        print(f"error: no completed run under {args.runs_root} has a checkpoint", file=sys.stderr)
        return 1
    if best not in evaluable:
        candidates = val_summary.loc[val_summary["experiment"].isin(evaluable)]
        chosen = select_best_experiment(candidates)
        print(f"{best} has no checkpoint here; top confusions from {chosen} instead", flush=True)
        best = chosen

    append_access_log(args.final_dir / ACCESS_LOG, [run.key for run in runs])
    device = resolve_device(args.device)
    loaded: dict[tuple[str, str], SplitArrays] = {}
    records: list[dict[str, Any]] = []
    best_predictions: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = []
    best_label_map: dict[int, str] = {}
    best_run = next(run for run in runs if run.experiment == best)
    members: dict[str, list[tuple[CompletedRun, npt.NDArray[np.float32], SplitArrays]]] = {}
    for run in runs:
        summary = run.summary
        test = load_test_split(args, run, loaded)

        evaluator: Evaluator = TorchCheckpointEvaluator(
            run.directory / CHECKPOINT_FILE, device, args.batch_size
        )
        logits = evaluator.predict(test.X)
        record = {
            "model": summary["model"],
            "split_mode": summary["split_mode"],
            "seed": summary["seed"],
            "evaluator": evaluator.name,
            **score_test_split(logits, test.y, test.participant_id),
        }
        target = args.final_dir / run.key / "test_metrics.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, record)  # The run's directory names its experiment.
        records.append({**record, "experiment": run.experiment})
        members.setdefault(run.experiment, []).append((run, logits, test))
        if run.experiment == best:
            best_predictions.append((test.y, logits.argmax(axis=1)))
            best_label_map = evaluator.label_map
        print(
            f"{run.key}: test top-1 {record['test_top1']:.4f} top-5 {record['test_top5']:.4f} "
            f"macro-F1 {record['test_macro_f1']:.4f} (n={record['n_sequences']})",
            flush=True,
        )

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    summary_table = summarize_test(records)
    summary_table.to_csv(args.tables_dir / "test_fp32_summary.csv", index=False)
    per_participant_table(records).to_csv(args.tables_dir / "test_per_participant.csv", index=False)
    confusions = top_confusions(best_predictions, best_label_map)
    confusions.insert(0, "split_mode", best_run.split_mode)
    confusions.insert(0, "model", best_run.model)
    confusions.insert(0, "experiment", best)
    confusions.to_csv(args.tables_dir / "top_confusions.csv", index=False)
    print(summary_table.to_string(index=False), flush=True)
    print(
        f"top confusions from {best} ({best_run.model}, {best_run.split_mode}), "
        "chosen by mean val top-1",
        flush=True,
    )
    return write_test_ensembles(args, members)


def write_test_ensembles(
    args: argparse.Namespace,
    members: dict[str, list[tuple[CompletedRun, npt.NDArray[np.float32], SplitArrays]]],
) -> int:
    """test_ensemble.csv: the mean-logit ensemble of each experiment with at least two seeds."""
    rows = []
    for name, group in sorted(members.items()):
        if len(group) < MIN_ENSEMBLE_SEEDS:
            continue
        group = sorted(group, key=lambda item: item[0].seed)
        reference = group[0][2]
        if any(not np.array_equal(test.sequence_id, reference.sequence_id) for *_, test in group):
            print(f"error: {name}: seeds were evaluated on different test sets", file=sys.stderr)
            return 1
        scores = ensemble_scores([logits for _, logits, _ in group], reference.y)
        rows.append(ensemble_row([run for run, *_ in group], scores, "test"))
    if rows:
        table = ensemble_table(rows, "test")
        table.to_csv(args.tables_dir / "test_ensemble.csv", index=False)
        print(table.to_string(index=False), flush=True)
    return 0


def torch_test_top1(
    args: argparse.Namespace, run: CompletedRun, test: SplitArrays, device: torch.device
) -> tuple[float, str]:
    """The checkpoint's test top-1 from its test_metrics.json if present, else computed."""
    path = args.final_dir / run.key / "test_metrics.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("evaluator") != "torch_fp32" or record.get("n_sequences") != len(test.y):
            raise ValueError(f"{path} does not hold torch_fp32 metrics for this test split")
        return float(record["test_top1"]), str(path)
    evaluator = TorchCheckpointEvaluator(run.directory / CHECKPOINT_FILE, device, args.batch_size)
    return top_k_accuracy(evaluator.predict(test.X), test.y, 1), f"computed from {CHECKPOINT_FILE}"


def evaluate_variants(args: argparse.Namespace, runs: Sequence[CompletedRun]) -> int:
    """Test metrics of each ONNX variant of each signer-independent run with v1 features.

    Each run's fp32 variant must match the torch checkpoint's test top-1 within
    SANITY_TOLERANCE, or the evaluation stops with an error.
    """
    plan: list[tuple[CompletedRun, dict[str, Path]]] = []
    for run in runs:
        if run.split_mode != PRIMARY_SPLIT_MODE:
            continue
        if run.feature_set not in QUANTISED_FEATURE_SETS:
            print(f"{run.key}: feature set {run.feature_set} is not quantised; skipped", flush=True)
            continue
        directory = args.quant_root / run.key
        present = {
            variant: directory / variant_file(variant)
            for variant in QUANT_VARIANTS
            if (directory / variant_file(variant)).is_file()
        }
        if "fp32" not in present:
            print(
                f"error: {directory / variant_file('fp32')} is missing; run quantize.py first",
                file=sys.stderr,
            )
            return 1
        for variant in QUANT_VARIANTS:
            if variant not in present:
                print(f"warning: {run.key} has no {variant} variant; skipped", file=sys.stderr)
        plan.append((run, present))
    if not plan:
        print(f"error: no completed {PRIMARY_SPLIT_MODE} runs to evaluate", file=sys.stderr)
        return 1

    evaluated: list[str] = []
    for run, present in plan:
        if not (args.final_dir / run.key / "test_metrics.json").exists():
            evaluated.append(run.key)  # The checkpoint itself, for the fp32 sanity check.
        evaluated += [f"{QUANT_DIR}/{run.key}/{variant}" for variant in present]
    append_access_log(args.final_dir / ACCESS_LOG, evaluated)

    device = resolve_device(args.device)
    loaded: dict[tuple[str, str], SplitArrays] = {}
    records: list[dict[str, Any]] = []
    for run, present in plan:
        summary = run.summary
        test = load_test_split(args, run, loaded)
        evaluators = {variant: OnnxEvaluator(path) for variant, path in present.items()}
        logits = {variant: evaluator.predict(test.X) for variant, evaluator in evaluators.items()}
        try:
            torch_top1, source = torch_test_top1(args, run, test, device)
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        onnx_top1 = top_k_accuracy(logits["fp32"], test.y, 1)
        if abs(onnx_top1 - torch_top1) > SANITY_TOLERANCE:
            print(
                f"error: {run.key}: fp32 ONNX test top-1 {onnx_top1:.4f} differs from the torch "
                f"checkpoint's {torch_top1:.4f} ({source}) by more than 0.1 percentage points",
                file=sys.stderr,
            )
            return 1

        reference = logits["fp32"].argmax(axis=1)
        for variant, path in present.items():
            record: dict[str, Any] = {
                "model": summary["model"],
                "split_mode": summary["split_mode"],
                "seed": summary["seed"],
                "variant": variant,
                "evaluator": evaluators[variant].name,
                "file": str(path),
                "size_bytes": path.stat().st_size,
                **score_test_split(logits[variant], test.y, test.participant_id),
                "agreement_with_fp32": float((logits[variant].argmax(axis=1) == reference).mean()),
            }
            if variant == "fp32":
                record.update(torch_test_top1=torch_top1, torch_reference=source)
            target = (
                args.final_dir
                / QUANT_DIR
                / run.results_key
                / f"seed_{summary['seed']}"
                / f"{variant}_test_metrics.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            write_json(target, record)
            records.append({**record, "experiment": run.experiment})
            print(
                f"{run.key} {variant}: test top-1 {record['test_top1']:.4f} "
                f"top-5 {record['test_top5']:.4f} macro-F1 {record['test_macro_f1']:.4f} "
                f"agreement {record['agreement_with_fp32']:.4f} (n={record['n_sequences']})",
                flush=True,
            )

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    table = summarize_quant_test(records)
    table.to_csv(args.tables_dir / "test_quant_summary.csv", index=False)
    print(table.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
