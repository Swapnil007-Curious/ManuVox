"""Export one trained model, or the seed ensemble of its experiment, for the browser demo:
web/public/model/{model.onnx, meta.json} and the golden vectors in web/tests/golden/.

Variant rule (val only): among fp32, fp16, int8_dynamic and int8_static, the smallest file whose
val top-1 is within 0.5 percentage points of fp32. Every candidate is recorded in meta.json.
With --verify-wasm each chosen file is built into the web app and run through the browser
model-parity test (onnxruntime-web, wasm); if that fails, the next eligible variant is exported
and the reason recorded. The confidence threshold is calibrated on val (target precision 0.80);
the test split is never read.
"""

import argparse
import importlib.util
import os
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import cast

from edgevision.web_export import (
    ASL_SIGNS_CLASSES,
    DATA_SOURCES,
    DEFAULT_TARGET_PRECISION,
    DEFAULT_WORK_DIR,
    WEB_VARIANTS,
    DataSource,
    ExportResult,
    WasmCheck,
    WebVariant,
    export_web,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "conftest.py"
PREVIEW_PORT = 4173
OUTPUT_TAIL = 25


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run dir, e.g. .../seed_0")
    parser.add_argument(
        "--variant",
        choices=("auto", *WEB_VARIANTS),
        default="auto",
        help="auto applies the variant rule; a named variant overrides it (recorded as forced)",
    )
    parser.add_argument("--quant-root", type=Path, default=Path("results/quant"))
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=DEFAULT_TARGET_PRECISION)
    parser.add_argument("--out", type=Path, default=Path("web/public/model"))
    parser.add_argument("--golden-dir", type=Path, default=Path("web/tests/golden"))
    parser.add_argument(
        "--data-source",
        choices=DATA_SOURCES,
        default="asl-signs",
        help="recorded in meta.json and model_cases.json; pass synthetic for fixture data",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="export the seed ensemble of the run's experiment (every seed with best.pt)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="where variants without a quant_report.json (ensembles, v2 runs) are built",
    )
    parser.add_argument(
        "--aspect-report",
        type=Path,
        help="preprocess_report.json of another preprocessing run of the same raw data, used "
        "for aspect_stats when the processed dir's report has none",
    )
    parser.add_argument(
        "--verify-wasm",
        action="store_true",
        help="build web/ and run the browser model-parity test for each exported variant",
    )
    parser.add_argument("--web-dir", type=Path, default=REPO_ROOT / "web")
    return parser.parse_args(argv)


def load_fixtures() -> ModuleType:
    """tests/conftest.py, whose synthetic sequence builders generate the preprocessing cases."""
    spec = importlib.util.spec_from_file_location("edgevision_test_fixtures", FIXTURES)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {FIXTURES}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def _tail(text: str) -> str:
    return "\n".join(text.strip().splitlines()[-OUTPUT_TAIL:])


class PlaywrightVerifier:
    """Runs `npm run build` and the "model parity" Playwright test in web/ for an exported
    model. A failing parity test is a failed check; a build or test-runner problem raises, so
    it is never mistaken for a variant onnxruntime-web cannot run."""

    def __init__(self, web_dir: Path):
        self.web_dir = web_dir
        npm, npx = shutil.which("npm"), shutil.which("npx")
        if npm is None or npx is None:
            raise RuntimeError("--verify-wasm needs npm and npx on PATH")
        self.npm, self.npx = npm, npx

    def run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        env = {key: value for key, value in os.environ.items() if key != "BASE_PATH"}
        env["CI"] = "1"  # Playwright starts its own preview server instead of reusing one.
        return subprocess.run(
            command,
            cwd=self.web_dir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def __call__(self, result: ExportResult) -> WasmCheck:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", PREVIEW_PORT)) == 0:
                raise RuntimeError(
                    f"port {PREVIEW_PORT} is in use; stop the running preview server first"
                )
        print(f"verifying {result.meta.variant} in onnxruntime-web (npm run build, Playwright)")
        build = self.run([self.npm, "run", "build"])
        if build.returncode != 0:
            raise RuntimeError(f"npm run build failed:\n{_tail(build.stdout + build.stderr)}")
        test = self.run(
            [self.npx, "playwright", "test", "tests/e2e/parity.spec.ts", "--grep", "model parity"]
        )
        output = test.stdout + test.stderr
        summary = re.findall(r"model parity: [^\n]+", output)
        if test.returncode == 0 and re.search(r"\b1 passed\b", output):
            return WasmCheck(status="passed", detail=summary[-1] if summary else None)
        if re.search(r"\b1 failed\b", output):
            return WasmCheck(status="failed", detail=_tail(output))
        raise RuntimeError(f"the Playwright parity test did not run:\n{_tail(output)}")


def print_selection(result: ExportResult) -> None:
    meta = result.meta
    selection = meta.variant_selection
    print(
        f"run {meta.run} ({meta.model_name}, feature set {meta.feature_set}), data source: "
        f"{meta.data_source}, seeds {meta.seeds}"
    )
    print(f"variant rule ({selection.source}): {selection.rule}")
    print(f"{'variant':14s} {'bytes':>10s} {'val top-1':>10s} {'drop pp':>8s}  eligible  wasm")
    for c in selection.candidates:
        size = "-" if c.file_bytes is None else str(c.file_bytes)
        drop = "-" if c.drop_pp is None else f"{c.drop_pp:+.2f}"
        note = f"  ({c.note})" if c.note else ""
        print(
            f"{c.variant:14s} {size:>10s} {percent(c.val_top1):>10s} {drop:>8s}  "
            f"{'yes' if c.eligible else 'no':8s}  {c.wasm.status}{note}"
        )
    print(f"order (smallest eligible first): {', '.join(selection.order)}")
    for fallback in selection.fallbacks:
        print(f"fallback: {fallback.variant} failed in onnxruntime-web: {fallback.reason}")
    forced = " (forced by --variant)" if selection.forced else ""
    print(
        f"chosen: {selection.chosen}{forced}; recommended {selection.recommended}; "
        f"follows rule: {selection.follows_rule}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fixtures = load_fixtures()
    variant = None if args.variant == "auto" else cast(WebVariant, args.variant)
    verifier = PlaywrightVerifier(args.web_dir) if args.verify_wasm else None
    result = export_web(
        args.run,
        args.quant_root,
        args.processed_dir,
        out_dir=args.out,
        golden_dir=args.golden_dir,
        landmark_blocks=fixtures.landmark_blocks,
        long_frame=fixtures.long_frame,
        variant=variant,
        target_precision=args.target_precision,
        data_source=cast(DataSource, args.data_source),
        ensemble=args.ensemble,
        work_dir=args.work_dir,
        aspect_report=args.aspect_report,
        wasm_verifier=verifier,
    )
    meta = result.meta
    print_selection(result)
    threshold = meta.threshold
    print(
        f"threshold {threshold.threshold:.6f} ({threshold.method}, target precision "
        f"{threshold.target_precision:.2f}): coverage {percent(threshold.coverage)}, accuracy "
        f"above threshold {percent(threshold.accuracy_above_threshold)}, overall val top-1 "
        f"{percent(threshold.overall_val_top1)} on {threshold.val_sequences} val sequences"
    )
    if threshold.warning:
        print(f"warning: {threshold.warning}")
    if meta.aspect is None:
        print("aspect: none (synthetic data without aspect_stats); the app keeps k = 1")
    else:
        print(f"aspect: r_train {meta.aspect.r_train:.4f} from {meta.aspect.source}")
    if meta.data_source != "synthetic" and meta.num_classes != ASL_SIGNS_CLASSES:
        print(
            f"warning: {meta.num_classes} classes, but asl-signs has {ASL_SIGNS_CLASSES}; "
            "pass --data-source synthetic for fixture data"
        )
    if meta.data_source == "synthetic":
        print("warning: synthetic data; the exported model does not recognise real signs")
    print(f"wrote {result.model_path} ({meta.model_bytes} bytes, sha256 {meta.model_sha256})")
    print(f"wrote {result.meta_path}")
    print(f"wrote {', '.join(str(path) for path in result.golden_paths.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
