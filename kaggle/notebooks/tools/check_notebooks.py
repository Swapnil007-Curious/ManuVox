"""Leak, licence and style checks for a Kaggle notebook series (and any dataset folders in it).

Errors (exit 1): local paths, the git user's email, the local account name, token-like strings,
any reference to data the series must not use, metadata that is not private or that names extra
inputs, a cover that is not 1280x720, error outputs, an invalid notebook.
Warnings: style rules for the markdown (dashes as connectors, banned words, emoji, headings in
title case, "not X but Y" phrasing).

    python kaggle/notebooks/tools/check_notebooks.py manuvox
"""

import argparse
import getpass
import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".ipynb", ".json", ".md", ".py", ".nbsrc", ".txt", ".csv"}
LEAKS = {
    "Windows path": r"[A-Za-z]:[\\/]{1,2}(?:Users|Program|Windows)",
    "MSYS path": r"/[a-z]/Users/",
    "home folder": r"/home/[a-z][\w-]+/|AppData",
    "Kaggle token": r"KGAT_[A-Za-z0-9]{8,}|\"key\"\s*:\s*\"[0-9a-f]{32}\"",
    "secret assignment": r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*['\"][^'\"]{8,}",
}
FORBIDDEN_DATA = r"(?i)citizen|v3[-_ ]?citizen|asl[-_ ]lex|groq"
BANNED_WORDS = [
    "delve", "crucial", "pivotal", "showcase", "leverage", "utilize", "utilise", "landscape", "tapestry",
    "testament", "seamless", "unlock", "game-changer", "meticulous", "furthermore", "additionally",
]
UPPER_OK = {
    "ASL", "ManuVox", "MediaPipe", "Holistic", "PyTorch", "ONNX", "CNN", "FP16", "FP32", "INT8", "Kaggle", "GitHub",
    "CPU", "GPU", "ECE", "NLL", "SHAP", "Transformer", "T4", "I", "Google", "ISLR", "RAG", "BM25", "Chrome",
}
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐✅]")


def check_text(path, text, owner_terms, errors):
    for name, pattern in LEAKS.items():
        for match in re.finditer(pattern, text):
            errors.append(f"{path}: {name}: {match.group(0)[:40]!r}")
    for term in owner_terms:
        if term and re.search(rf"(?<![\w.-]){re.escape(term)}(?![\w.-])", text, re.IGNORECASE):
            errors.append(f"{path}: contains the local account name or git email")
    for match in re.finditer(FORBIDDEN_DATA, text):
        errors.append(f"{path}: reference to data the series must not use: {match.group(0)!r}")


def check_markdown(path, cell_index, text, warnings):
    where = f"{path} cell {cell_index}"
    if "—" in text or "–" in text:
        warnings.append(f"{where}: em or en dash")
    for word in BANNED_WORDS:
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            warnings.append(f"{where}: banned word {word!r}")
    if re.search(r"\brobust\b", text, re.IGNORECASE):
        warnings.append(f"{where}: 'robust' (check it is literal)")
    if EMOJI.search(text):
        warnings.append(f"{where}: emoji")
    if re.search(r"\bnot\b[^.\n]{0,50}\bbut\b", text, re.IGNORECASE):
        warnings.append(f"{where}: possible 'not X but Y' phrasing")
    for line in text.splitlines():
        if line.startswith("#"):
            words = re.findall(r"[A-Za-z][\w+-]*", line.lstrip("#").replace("ASL Signs", "ASL"))  # The dataset's name.
            title_case = [w for w in words[1:] if w[0].isupper() and w not in UPPER_OK and not w.isupper()]
            if title_case:
                warnings.append(f"{where}: heading not in sentence case: {line.strip()!r}")


def check_folder(folder, owner_terms, errors, warnings):
    metadata_path = folder / "kernel-metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta.get("is_private") is not True:
            errors.append(f"{metadata_path}: is_private must be true")
        if meta.get("dataset_sources") or meta.get("kernel_sources") or meta.get("model_sources"):
            errors.append(f"{metadata_path}: unexpected inputs beyond the competition")
        if meta.get("competition_sources") != ["asl-signs"]:
            errors.append(f"{metadata_path}: competition_sources must be ['asl-signs']")
        cover = folder / "cover.png"
        if not cover.exists():
            errors.append(f"{folder}: cover.png missing")
        else:
            from PIL import Image

            with Image.open(cover) as image:
                if image.size != (1280, 720):
                    errors.append(f"{cover}: size {image.size}, expected (1280, 720)")
    notebook = folder / "notebook.ipynb"
    if notebook.exists():
        nb = nbformat.read(notebook, as_version=4)
        nbformat.validate(nb)
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown":
                check_markdown(notebook, index, cell.source, warnings)
            for output in cell.get("outputs", []):
                generated = output.get("data", {}).get("text/markdown")
                if generated:  # Takeaways and summaries written by the code: same style rules.
                    check_markdown(notebook, f"{index} (output)", "".join(generated), warnings)
                if output.get("output_type") == "error":
                    errors.append(f"{notebook} cell {index}: error output {output.get('ename')}")
                elif output.get("name") == "stderr":
                    warnings.append(f"{notebook} cell {index}: stderr output: {output.get('text', '')[:80]!r}")
            if cell.cell_type == "markdown":
                continue
            for line in cell.source.splitlines():
                if "#" in line and ("—" in line or "–" in line):
                    warnings.append(f"{notebook} cell {index}: dash in a code comment")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("series")
    args = parser.parse_args()
    series_dir = ROOT / args.series
    email = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True).stdout.strip()
    owner_terms = [getpass.getuser(), email]
    errors, warnings = [], []
    for path in sorted(series_dir.rglob("*")):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            if path.suffix == ".ipynb":  # Base64 images match word patterns by chance; scan everything else.
                nb = nbformat.reads(text, as_version=4)
                for cell in nb.cells:
                    for output in cell.get("outputs", []):
                        for mime in [m for m in output.get("data", {}) if m.startswith("image/")]:
                            output["data"][mime] = ""
                text = nbformat.writes(nb)
            check_text(path.relative_to(ROOT), text, owner_terms, errors)
    for folder in sorted(p for p in series_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        check_folder(folder, owner_terms, errors, warnings)
    for line in warnings:
        print("WARN ", line)
    for line in errors:
        print("ERROR", line)
    print(f"{len(errors)} errors, {len(warnings)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
