"""Choose the model the web demo ships: signer-independent test accuracy against latency on the
target laptop (i3-1115G4, Chrome, onnxruntime-web).

    python scripts/model_selection.py export     # candidate ONNX files into web/dist/eval/
    (cd web; npm run preview)                     # another shell
    python scripts/model_selection.py bench      # web/scripts/benchmark.mjs, wasm x1, per candidate
    python scripts/model_selection.py report     # results/model_selection.csv and .md

Accuracy comes from the test tables scripts/final_eval.py already wrote
(results/tables/selection/: the mean and sd over seeds, and the seed ensembles); this script never
reads the test split. Latency is the median and p95 of 200 classifier calls on one thread in the
installed Chrome (bench.html), for the seed-0 file of each candidate (the ensemble file holds all
three seeds). The rule, fixed before measuring: the highest test top-1 among candidates whose p95
latency is under the 60 ms per-segment budget and whose file stays under the 10 MB limit for a
committed model.onnx.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import onnx

from edgevision.export.ensemble import export_ensemble_onnx
from edgevision.export.onnx_export import export_onnx
from edgevision.models.registry import load_checkpoint
from edgevision.quant.variants import convert_fp16

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"
TABLES = REPO / "results" / "tables" / "selection"
EVAL_DIR = REPO / "web" / "dist" / "eval"
BENCH_DIR = REPO / "results" / "perf" / "models"
LATENCY_BUDGET_MS = 60.0
MAX_COMMIT_BYTES = 10_000_000
SEEDS = (0, 1, 2)

# name, experiment, features, ensemble, variant
CANDIDATES: tuple[tuple[str, str, str, bool, str], ...] = (
    ("gru_v1", "gru_signer_independent", "v1", False, "fp32"),
    ("transformer_v1", "transformer_signer_independent", "v1", False, "fp32"),
    ("cnn_v1", "cnn_signer_independent", "v1", False, "fp32"),
    ("cnn_v2feat_v1recipe", "b_cnn_v2feat_v1recipe", "v2", False, "fp32"),
    ("cnn_v2feat_v2recipe", "c_cnn_v2feat_v2recipe", "v2", False, "fp32"),
    ("hybrid_v2", "d_hybrid_v2feat_v2recipe", "v2", False, "fp32"),
    ("hybrid_v2_fp16", "d_hybrid_v2feat_v2recipe", "v2", False, "fp16"),
    ("hybrid_v2_ensemble", "d_hybrid_v2feat_v2recipe", "v2", True, "fp32"),
    ("hybrid_v2_ensemble_fp16", "d_hybrid_v2feat_v2recipe", "v2", True, "fp16"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(args: argparse.Namespace) -> int:
    for name, experiment, _, ensemble, variant in CANDIDATES:
        target = args.out / name
        target.mkdir(parents=True, exist_ok=True)
        fp32 = target / "fp32.onnx"
        checkpoints = [RUNS / experiment / f"seed_{seed}" / "best.pt" for seed in SEEDS]
        if ensemble:
            export_ensemble_onnx(checkpoints, fp32)
        else:
            export_onnx(load_checkpoint(checkpoints[0]).model, fp32)
        model = target / "model.onnx"
        if variant == "fp16":
            onnx.save(convert_fp16(fp32, keep_feature_block_fp32=False), str(model))
            fp32.unlink()
        else:
            fp32.replace(model)
        graph = onnx.load(str(model)).graph
        frames, width = (dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim[1:])
        meta = {
            "model_file": "model.onnx",
            "model_sha256": sha256(model),
            "variant": variant,
            "input_layout": {
                "input_name": graph.input[0].name,
                "output_name": graph.output[0].name,
                "num_frames": frames,
                "frame_dim": width,
            },
        }
        (target / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {model.stat().st_size / 1e6:.2f} MB, input {frames} x {width}")
    return 0


def bench(args: argparse.Namespace) -> int:
    args.bench_dir.mkdir(parents=True, exist_ok=True)
    for name, *_ in CANDIDATES:
        out = args.bench_dir / f"{name}.json"
        command = [
            "node",
            "scripts/benchmark.mjs",
            "--model",
            f"eval/{name}",
            "--only",
            "wasm:1",
            "--out",
            str(out),
        ]
        subprocess.run(command, cwd=REPO / "web", check=True)
    return 0


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["experiment"]: row for row in csv.DictReader(handle)}


def pct(value: str | float | None, sd: str | float | None = None) -> str:
    if value in (None, ""):
        return "-"
    text = f"{100 * float(value):.1f}"
    return f"{text} ± {100 * float(sd):.1f}" if sd not in (None, "") else text


def one_decimal(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def report(args: argparse.Namespace) -> int:
    seeds = read_csv(TABLES / "test_fp32_summary.csv")
    ensembles = read_csv(TABLES / "test_ensemble.csv")
    rows: list[dict[str, Any]] = []
    for name, experiment, features, ensemble, variant in CANDIDATES:
        bench_file = args.bench_dir / f"{name}.json"
        result: dict[str, Any] = {}
        if bench_file.is_file():
            runs = json.loads(bench_file.read_text(encoding="utf-8"))["results"]
            result = next((r for r in runs if r["ep"] == "wasm" and r["threads"] == 1), {})
        if ensemble:
            accuracy = ensembles[experiment]
            top1, top1_sd = accuracy["test_top1"], None
            top3, top3_sd = accuracy["test_top3"], None
        else:
            accuracy = seeds[experiment]
            top1, top1_sd = accuracy["test_top1_mean"], accuracy["test_top1_std"]
            top3, top3_sd = accuracy["test_top3_mean"], accuracy["test_top3_std"]
        size = result.get("model", {}).get("bytes")
        p95 = result.get("p95")
        eligible = (
            p95 is not None
            and p95 < LATENCY_BUDGET_MS
            and size is not None
            and size < MAX_COMMIT_BYTES
        )
        rows.append(
            {
                "candidate": name,
                "experiment": experiment,
                "features": features,
                "seeds": "0-2 averaged in one file"
                if ensemble
                else "0 (accuracy: mean ± sd of 0-2)",
                "variant": variant,
                "test_top1": float(top1),
                "test_top1_sd": float(top1_sd) if top1_sd not in (None, "") else None,
                "test_top3": float(top3),
                "test_top3_sd": float(top3_sd) if top3_sd not in (None, "") else None,
                "onnx_mb": size / 1e6 if size else None,
                "latency_p50_ms": result.get("p50"),
                "latency_p95_ms": p95,
                "eligible": eligible,
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        print(
            "no candidate meets the latency and size limits (or no bench results)", file=sys.stderr
        )
        return 1
    chosen = max(eligible_rows, key=lambda row: (row["test_top1"], -row["onnx_mb"]))
    for row in rows:
        row["chosen"] = row is chosen
    columns = list(rows[0].keys())
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    sample = json.loads(
        (args.bench_dir / f"{rows[0]['candidate']}.json").read_text(encoding="utf-8")
    )
    lines = [
        "# Model selection for the web demo",
        "",
        "Generated by `scripts/model_selection.py report`.",
        "",
        "- Accuracy: signer-independent test split, from `results/tables/selection/` "
        "(`scripts/final_eval.py`); single models are the mean ± sd over seeds 0-2, ensembles "
        "average the three seeds' logits in one ONNX file.",
        "- Latency: one classifier call (input 1 x 32 x frame_dim) in Chrome "
        f"{sample['browser']} on the i3-1115G4 laptop, onnxruntime-web wasm SIMD, 1 thread, "
        "20 warm-up and 200 timed calls "
        "(`web/scripts/benchmark.mjs`); single models use the seed-0 file.",
        "- Rule (fixed before measuring): the highest test top-1 with p95 < "
        f"{LATENCY_BUDGET_MS:.0f} ms and a file under {MAX_COMMIT_BYTES / 1e6:.0f} MB (the limit "
        "for a committed model.onnx). Validation top-1 ranks the candidates the same way "
        "(`results/tables/selection/val_summary.csv`).",
        "- fp16 rows repeat the fp32 accuracy: the test split was only evaluated in fp32, and the "
        "exported fp16 file's val top-1 equals fp32's (`web/public/model/meta.json`, "
        "`variant_selection`).",
        "- Latency is the classifier alone; next to live tracking on the same laptop the shipped "
        "file took p50 18.6 / p95 27.2 ms per segment (`results/perf/tracking_gpu.json`).",
        "",
        "| Candidate | Features | Variant | Test top-1 (%) | Test top-3 (%) | ONNX (MB) | "
        "p50 (ms) | p95 (ms) | Eligible | Chosen |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['candidate']}` | {row['features']} | {row['variant']} | "
            f"{pct(row['test_top1'], row['test_top1_sd'])} | "
            f"{pct(row['test_top3'], row['test_top3_sd'])} | "
            f"{one_decimal(row['onnx_mb'])} | {one_decimal(row['latency_p50_ms'])} | "
            f"{one_decimal(row['latency_p95_ms'])} | {'yes' if row['eligible'] else 'no'} | "
            f"{'**yes**' if row['chosen'] else ''} |"
        )
    lines += [
        "",
        f"Chosen: `{chosen['candidate']}` ({chosen['experiment']}, {chosen['variant']}).",
        "",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(f"chosen {chosen['candidate']}; wrote {args.csv} and {args.markdown}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("export", "bench", "report"))
    parser.add_argument("--out", type=Path, default=EVAL_DIR)
    parser.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    parser.add_argument("--csv", type=Path, default=REPO / "results" / "model_selection.csv")
    parser.add_argument("--markdown", type=Path, default=REPO / "results" / "model_selection.md")
    args = parser.parse_args(argv)
    return {"export": export, "bench": bench, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
