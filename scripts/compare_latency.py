"""Summarise the v2 vs v3 classifier latency runs in the installed Chrome on the target laptop.

    python scripts/compare_latency.py

Reads the per-round files of web/scripts/benchmark.mjs (three interleaved rounds per model,
--model model and --model eval/v3) from results/perf/models/ and writes
results/perf/v2_vs_v3_classifier_latency.json: for each model and configuration the p50 / p95 of
every round, their medians and the largest p95. v2 and v3 share the architecture, so their files
are the same size and any latency difference is run-to-run variation.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

PERF = Path("results/perf")
ROUNDS = {
    "v2": [
        "hybrid_v2_fp16_2026-11-19.json",
        "hybrid_v2_fp16_2026-11-19.r2.json",
        "hybrid_v2_fp16_2026-11-19.r3.json",
    ],
    "v3": [
        "hybrid_v3_citizen_fp16.json",
        "hybrid_v3_citizen_fp16.r2.json",
        "hybrid_v3_citizen_fp16.r3.json",
    ],
}
DESCRIPTION = (
    "Classifier latency in installed Chrome on the target laptop (i3-1115G4), "
    "web/scripts/benchmark.mjs: 20 warm-up and 200 timed calls per configuration, three "
    "interleaved rounds per model on 2026-11-19; median over rounds of each round's p50 / p95. "
    "The app's default is wasm with 1 thread (web/src/ortBackend.ts defaultThreads)."
)


def summarise(files: list[Path]) -> dict[str, Any]:
    runs = [json.loads(path.read_text("utf-8")) for path in files]
    models = {
        (r["model"]["sha256"], r["model"]["bytes"], r["model"]["variant"])
        for run in runs
        for r in run["results"]
    }
    if len(models) != 1:
        raise ValueError(f"{files}: rounds measured different model files: {models}")
    sha256, size, variant = models.pop()
    browsers = {run["browser"] for run in runs}
    configs: dict[str, dict[str, Any]] = {}
    for run in runs:
        for r in run["results"]:
            if r.get("error"):
                raise ValueError(f"{r['ep']} x{r['threads']}: {r['error']}")
            entry = configs.setdefault(f"{r['ep']}x{r['threads']}", {"p50_ms": [], "p95_ms": []})
            entry["p50_ms"].append(round(r["p50"], 2))
            entry["p95_ms"].append(round(r["p95"], 2))
    for entry in configs.values():
        entry["median_p50_ms"] = statistics.median(entry["p50_ms"])
        entry["median_p95_ms"] = statistics.median(entry["p95_ms"])
        entry["max_p95_ms"] = max(entry["p95_ms"])
    return {
        "model_sha256": sha256,
        "bytes": size,
        "variant": variant,
        "browser": ", ".join(sorted(browsers)),
        "configs": configs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--perf", type=Path, default=PERF)
    parser.add_argument("--out", type=Path, default=PERF / "v2_vs_v3_classifier_latency.json")
    args = parser.parse_args(argv)
    summary = {
        "description": DESCRIPTION,
        "files": ROUNDS,
        "models": {
            name: summarise([args.perf / "models" / f for f in files])
            for name, files in ROUNDS.items()
        },
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    for name, model in summary["models"].items():
        one = model["configs"]["wasmx1"]
        print(
            f"{name}: {model['bytes']} bytes; wasm x1 p50 {one['median_p50_ms']} ms, "
            f"p95 {one['median_p95_ms']} ms (rounds {one['p95_ms']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
