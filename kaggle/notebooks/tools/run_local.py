"""Execute a series' notebooks locally with Kaggle-style paths, then scrub and save them.

The notebooks read /kaggle/input and write /kaggle/working unless KAGGLE_INPUT_DIR and
KAGGLE_WORKING_DIR say otherwise; this runner sets both (the path shim), plus MANUVOX_FAST=1 with
--fast. Run it with the interpreter whose packages the kernel should use:

    <venv>/python kaggle/notebooks/tools/run_local.py manuvox --input data/kaggle_input --fast --only m1

Each notebook runs in data/kaggle_work/<folder>/ (git ignores data/). Afterwards the runner
replaces local paths in the outputs, copies cover.png next to the notebook, records the mode and
wall time in the notebook metadata ("manuvox_run") and validates the result with nbformat.
"""

import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def scrub_text(text, replacements):
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def scrub(nb, replacements):
    """Replace local paths in every output (stream text and every text-like mime bundle)."""
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if "text" in output:
                output["text"] = scrub_text(output["text"], replacements)
            for key in ("data", "metadata"):
                bundle = output.get(key, {})
                for mime, value in list(bundle.items()):
                    if isinstance(value, str) and (mime.startswith("text/") or mime.endswith("json")):
                        bundle[mime] = scrub_text(value, replacements)
            if output.get("output_type") == "error":
                output["traceback"] = [scrub_text(line, replacements) for line in output["traceback"]]


def path_variants(path, label):
    """A path in the spellings that can show up in outputs, each mapped to ``label``."""
    text = str(path.resolve())
    forms = {text, text.replace("\\", "/"), text.replace("\\", "\\\\")}
    return [(form, label) for form in sorted(forms, key=len, reverse=True)]


def run(entry, args):
    folder = ROOT / args.series / entry["folder"]
    work = (REPO / "data" / "kaggle_work" / entry["folder"]).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    os.environ["KAGGLE_INPUT_DIR"] = str(args.input.resolve())
    os.environ["KAGGLE_WORKING_DIR"] = str(work)
    os.environ["MANUVOX_FAST"] = "1" if args.fast else "0"
    nb = nbformat.read(folder / "notebook.ipynb", as_version=4)
    client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": str(work)}})
    started = time.time()
    failed = None
    try:
        client.execute()
    except CellExecutionError as error:
        failed = str(error).splitlines()[-1] if str(error) else "cell error"
    seconds = time.time() - started
    replacements = (
        path_variants(work, "/kaggle/working")
        + path_variants(args.input, "/kaggle/input")
        + path_variants(REPO, "<repo>")
        + path_variants(Path(sys.prefix), "<venv>")  # The kernel's environment, e.g. in an error from %pip.
        + path_variants(Path.home(), "~")
    )
    scrub(nb, replacements)
    nb.metadata["manuvox_run"] = {
        "mode": "fast" if args.fast else "full",
        "seconds": round(seconds),
        "device": "CPU (local test machine)",
        "python": platform.python_version(),
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "status": "failed: " + failed if failed else "ok",
    }
    nb.metadata.pop("widgets", None)
    nbformat.validate(nb)
    target = folder / "notebook.ipynb" if not failed else work / "failed.ipynb"
    with target.open("w", encoding="utf-8", newline="\n") as handle:  # LF on every OS.
        nbformat.write(nb, handle)
    cover = work / "cover.png"
    if cover.exists() and not failed:
        shutil.copyfile(cover, folder / "cover.png")
    print(f"{entry['folder']}: {'FAILED (' + failed + ')' if failed else 'ok'} in {seconds / 60:.1f} min -> {target.relative_to(REPO)}")
    return not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("series")
    parser.add_argument("--input", type=Path, required=True, help="folder that holds asl-signs/ (Kaggle's /kaggle/input)")
    parser.add_argument("--fast", action="store_true", help="set MANUVOX_FAST=1 (a few sequences per sign)")
    parser.add_argument("--only", nargs="*", help="notebook keys, e.g. m1 m3")
    parser.add_argument("--timeout", type=int, default=4 * 3600, help="seconds per cell")
    args = parser.parse_args()
    series = json.loads((ROOT / args.series / "series.json").read_text(encoding="utf-8"))
    entries = [e for e in series["notebooks"] if not args.only or e["key"] in args.only]
    results = [run(entry, args) for entry in entries]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
