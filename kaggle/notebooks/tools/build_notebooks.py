"""Build the Kaggle notebooks of a series from their sources.

Each notebook source (<series>/_src/<name>.nbsrc) is a list of cells separated by marker lines:

    %% md              a markdown cell
    %% code            a code cell
    %% code hide       a code cell whose input Kaggle hides by default
    %% include NAME    a code cell holding <series>/_src/common/NAME.py
    %% include-hide NAME

Markdown may use {{HEADER}} (live app, repository and paper links), {{SERIES}} (links to the
other notebooks) and {{URL:key}} (one notebook's Kaggle URL). The series is described in
<series>/series.json. Writes <folder>/notebook.ipynb and <folder>/kernel-metadata.json. Outputs
of an existing notebook are kept for every code cell whose source is unchanged, so editing
markdown does not throw away an executed run.

    python kaggle/notebooks/tools/build_notebooks.py manuvox [--username NAME]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(r"^%% (md|code|code hide|include|include-hide)(?: (\S+))?\s*$")
PLACEHOLDER_USER = "YOUR-KAGGLE-USERNAME"


def slugify(title):
    """Kaggle's slug rule for plain ASCII titles: lowercase words joined by hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def notebook_url(series, entry):
    return f"https://www.kaggle.com/code/{series['username']}/{slugify(entry['kaggle_title'])}"


def parse_cells(text, common_dir):
    """Split an .nbsrc file into (kind, source, hidden) cells."""
    cells, kind, hidden, lines = [], None, False, []

    def flush():
        if kind is not None:
            source = "\n".join(lines).strip("\n")
            if source:
                cells.append((kind, source, hidden))

    for line in text.splitlines():
        match = MARKER.match(line)
        if not match:
            lines.append(line)
            continue
        flush()
        tag, name = match.groups()
        lines, hidden = [], tag.endswith("hide")
        if tag.startswith("include"):
            cells.append(("code", (common_dir / f"{name}.py").read_text(encoding="utf-8").strip("\n"), hidden))
            kind = None
        else:
            kind = "markdown" if tag == "md" else "code"
    flush()
    return cells


def render_markdown(source, series, entry):
    header = (
        f"Live app: [{series['app_url']}]({series['app_url']}) · "
        f"GitHub: [{series['repo_url']}]({series['repo_url']}) · Paper: link coming soon"
    )
    others = [e for e in series["notebooks"] if e["key"] != entry["key"]]
    listing = "\n".join(f"- [{e['number']}. {e['title']}]({notebook_url(series, e)})" for e in others)
    source = source.replace("{{HEADER}}", header).replace("{{SERIES}}", listing)
    by_key = {e["key"]: e for e in series["notebooks"]}
    return re.sub(r"\{\{URL:(\w+)\}\}", lambda m: notebook_url(series, by_key[m.group(1)]), source)


def build(series_dir, series):
    common = series_dir / "_src" / "common"
    for entry in series["notebooks"]:
        folder = series_dir / entry["folder"]
        folder.mkdir(exist_ok=True)
        cells = parse_cells((series_dir / "_src" / entry["source"]).read_text(encoding="utf-8"), common)
        path = folder / "notebook.ipynb"
        old = nbformat.read(path, as_version=4) if path.exists() else None
        old_code = {c.source: c for c in old.cells if c.cell_type == "code"} if old else {}
        nb = nbformat.v4.new_notebook()
        nb.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
        kept = 0
        for index, (kind, source, hidden) in enumerate(cells):
            if kind == "markdown":
                cell = nbformat.v4.new_markdown_cell(render_markdown(source, series, entry))
            else:
                cell = nbformat.v4.new_code_cell(source)
                if source in old_code:
                    cell.outputs = old_code[source].outputs
                    cell.execution_count = old_code[source].execution_count
                    kept += 1
                if hidden:
                    cell.metadata["_kg_hide-input"] = True
            cell.id = f"{entry['key']}-{index:02d}"
            nb.cells.append(cell)
        if old is not None and "manuvox_run" in old.metadata:
            nb.metadata["manuvox_run"] = old.metadata["manuvox_run"]
        nbformat.validate(nb)
        with path.open("w", encoding="utf-8", newline="\n") as handle:  # LF on every OS.
            nbformat.write(nb, handle)
        metadata = {
            "id": f"{series['username']}/{slugify(entry['kaggle_title'])}",
            "title": entry["kaggle_title"],
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": entry["gpu"],
            "enable_tpu": False,
            "enable_internet": entry["internet"],
            "machine_shape": "NvidiaTeslaT4" if entry["gpu"] else "",
            "dataset_sources": [],
            "competition_sources": series["competition_sources"],
            "kernel_sources": [],
            "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
        code_cells = sum(1 for k, _, _ in cells if k == "code")
        print(f"{entry['folder']}: {len(cells)} cells, outputs kept for {kept} of {code_cells} code cells")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("series", help="series folder under kaggle/notebooks, e.g. manuvox")
    parser.add_argument("--username", help="Kaggle username for the notebook ids and links")
    args = parser.parse_args()
    series_dir = ROOT / args.series
    series_path = series_dir / "series.json"
    series = json.loads(series_path.read_text(encoding="utf-8"))
    if args.username:
        series["username"] = args.username
        series_path.write_text(json.dumps(series, indent=2) + "\n", encoding="utf-8", newline="\n")
    for entry in series["notebooks"]:
        if not 5 <= len(entry["kaggle_title"]) <= 50:
            sys.exit(f"{entry['key']}: Kaggle title must be 5-50 characters: {entry['kaggle_title']!r}")
    if series["username"] == PLACEHOLDER_USER:
        print(f"note: username is the placeholder {PLACEHOLDER_USER}; rebuild with --username NAME")
    build(series_dir, series)


if __name__ == "__main__":
    main()
