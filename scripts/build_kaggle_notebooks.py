"""Build the single-cell Kaggle notebooks in kaggle/ and the code archive they run.

Every notebook takes the repository code from the private Kaggle dataset edge-vision-src only,
never from a git clone. That dataset holds edge-vision-src.tar.gz, the git archive of one commit
(--source-archive) with a SOURCE_COMMIT file. Each cell finds it under /kaggle/input, prints the
commit, checks that the archive holds the notebook sources the cell was built from (the
fingerprint of SOURCE_FILES) and stops with a clear message when it is missing or different.

    kaggle/asl_citizen_smoke.ipynb    extraction of the first 20 videos into a separate -smoke
                                      folder (never a dataset); a few minutes
    kaggle/asl_citizen_part{1,2,3}.ipynb  extraction, every third shard each (EV_PART 0/3, 1/3,
                                      2/3), run side by side
    kaggle/asl_citizen_merge.ipynb    EV_PART 0/1 with the three part outputs as inputs: copies
                                      their shards, writes manifest.csv and COMPLETE, and makes
                                      the private dataset asl-citizen-landmarks-private
    kaggle/train_v3.ipynb             the v3 recipe, kaggle/train_v3.sh

The extraction program is built in the cell, from the archive, by --script-out:
src/edgevision/data/citizen_labels.py, the 250 labels of web/public/model/meta.json and
kaggle/asl_citizen_extract.py. Kaggle's image runs Python 3.13 and mediapipe 0.10.21 (the last
release with the legacy Holistic solution the training data was made with) has wheels only up to
Python 3.12, so the extraction runs in a Python 3.11 venv made with uv, with Kaggle's PYTHONPATH
(its sitecustomize imports wrapt) kept out. Training uses Kaggle's Python when the training
dependencies import there and the same kind of venv otherwise. Every cell tees everything to a
log and always exits 0.

    python scripts/build_kaggle_notebooks.py               # write the notebooks
    python scripts/build_kaggle_notebooks.py --check       # exit 1 when a notebook is stale
    python scripts/build_kaggle_notebooks.py --source-archive artifacts/v3/upload/edge-vision-src.tar.gz
    python scripts/build_kaggle_notebooks.py --script-out extract.py   # the extraction program
    python scripts/build_kaggle_notebooks.py --fingerprint # this tree's notebook sources

This file uses only the standard library: the cells run it with Kaggle's Python.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS_SOURCE = ROOT / "src" / "edgevision" / "data" / "citizen_labels.py"
EXTRACT_BODY = ROOT / "kaggle" / "asl_citizen_extract.py"
TRAIN_BODY = ROOT / "kaggle" / "train_v3.sh"
META = ROOT / "web" / "public" / "model" / "meta.json"
KAGGLE = ROOT / "kaggle"
# What the cells run from the archive, and the files they are generated from: the archive and a
# notebook agree when these hash the same (line endings normalized).
SOURCE_FILES = (
    "scripts/build_kaggle_notebooks.py",
    "scripts/v3_kaggle.py",
    "kaggle/asl_citizen_extract.py",
    "kaggle/make_private_dataset.py",
    "kaggle/train_v3.sh",
    "src/edgevision/data/citizen_labels.py",
    "web/public/model/meta.json",
)
COMMIT_FILE = "SOURCE_COMMIT"
COMMON_MARKER = "# @common\n"
SMOKE_VIDEOS = 20
PARTS = 3
VENV_PYTHON = "3.11"  # mediapipe 0.10.21 has no wheel for Kaggle's Python 3.13
MPENV_PACKAGES = ("mediapipe==0.10.21", "opencv-python-headless", '"numpy<2"', "pandas", "pyarrow")
EXTRACT_SETTINGS = {"accelerator": "none", "isGpuEnabled": False, "isInternetEnabled": True}
TRAIN_SETTINGS = {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": True}
# Paths never allowed in the code archive: ASL Citizen data and anything derived from it,
# videos, landmark arrays, local models and credentials (docs/DATA_LICENSES.md).
FORBIDDEN = re.compile(
    r"^(data|artifacts|test_clips[^/]*)/|^results/(asl_citizen/predictions|v3_eval)/"
    r"|(^|/)web/models/|(^|/)(kaggle\.json|\.env[^/]*|credentials[^/]*|id_rsa[^/]*)$"
    r"|\.(npz|npy|parquet|mp4|webm|mov|avi|mkv|pem|p12|pfx)$",
    re.IGNORECASE,
)
FINGERPRINT_LINE = re.compile(r'^SOURCE_FINGERPRINT="([0-9a-f]*)"$', re.MULTILINE)


def normalized(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def fingerprint_of(read: Callable[[str], bytes | None]) -> str:
    """sha256 over SOURCE_FILES (path and normalized content), 16 hex digits."""
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        data = read(name)
        content = hashlib.sha256(normalized(data)).hexdigest() if data is not None else "missing"
        digest.update(f"{name}\0{content}\n".encode())
    return digest.hexdigest()[:16]


def source_fingerprint(root: Path = ROOT) -> str:
    def read(name: str) -> bytes | None:
        path = root / name
        return path.read_bytes() if path.is_file() else None

    return fingerprint_of(read)


def extraction_program() -> str:
    labels = json.loads(META.read_text(encoding="utf-8"))["labels"]
    return (
        "# Generated by scripts/build_kaggle_notebooks.py --script-out; edit the sources.\n"
        "# ---- src/edgevision/data/citizen_labels.py ----\n"
        + LABELS_SOURCE.read_text(encoding="utf-8")
        + "\n\n# The 250 labels of web/public/model/meta.json, in model order.\n"
        + f"LABELS = {json.dumps(labels)}\n\n"
        + "# ---- kaggle/asl_citizen_extract.py ----\n"
        + EXTRACT_BODY.read_text(encoding="utf-8")
    )


# Shared by every cell: paths, helpers, the uv venv and the code source. Placeholders are
# @NAME@ (no f-string, so the shell braces stay as they are).
COMMON = r"""# ---- common to the edge-vision Kaggle notebooks (scripts/build_kaggle_notebooks.py) ----
# EV_WORK, EV_INPUT and EV_TMP exist for local tests only (scripts/kaggle_local_test.py).
WORK="${EV_WORK:-/kaggle/working}"
export EV_INPUT="${EV_INPUT:-/kaggle/input}"
TMPROOT="${EV_TMP:-/tmp}"
SRC="$TMPROOT/edge-vision-src"
# The fingerprint of the sources this notebook was built from; the code archive must match it.
SOURCE_FINGERPRINT="@FINGERPRINT@"
VENV_PYTHON="@VENV_PYTHON@"
mkdir -p "$WORK" "$TMPROOT"
# clean runs a venv's Python with only the variables it needs. The Kaggle kernel exports its
# own: PYTHONPATH holds a sitecustomize that imports wrapt, and MPLBACKEND names
# matplotlib_inline, which made "import mediapipe" (it imports matplotlib.pyplot) fail in the
# venv. So the environment starts empty, keeps CLEAN_KEEP and every EV_* variable, and sets
# MPLBACKEND=Agg.
CLEAN_KEEP="HOME USER LOGNAME PATH LANG LC_ALL LC_CTYPE TZ TMPDIR TEMP TMP TERM
  LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES CUDA_HOME NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES
  OMP_NUM_THREADS HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy
  SSL_CERT_FILE SSL_CERT_DIR SYSTEMROOT WINDIR COMSPEC USERPROFILE APPDATA LOCALAPPDATA PATHEXT
  NUMBER_OF_PROCESSORS PROCESSOR_ARCHITECTURE"
clean() {
  local name keep=()
  for name in $CLEAN_KEEP $(compgen -e EV_); do
    if [ -n "${!name+x}" ]; then keep+=("$name=${!name}"); fi
  done
  env -i "${keep[@]}" MPLBACKEND=Agg PYTHONNOUSERSITE=1 "$@"
}
# Kaggle also sets UV_SYSTEM_PYTHON ("The --system flag has no effect" in uv venv's output).
uv() { env -u UV_SYSTEM_PYTHON -u VIRTUAL_ENV python -m uv "$@"; }
venv_python() { if [ -e "$1/bin/python" ]; then echo "$1/bin/python"; else echo "$1/Scripts/python.exe"; fi; }
kaggle_python() { python -c 'import platform; print(platform.python_version())' 2>&1; }
# find below the inputs without walking the asl-signs parquet tree or the landmark shards
find_inputs() { find "$EV_INPUT" \( -name train_landmarks -o -name shards \) -prune -o "$@" -print 2>/dev/null | sort; }

# make_venv DIR: an empty Python $VENV_PYTHON venv, made with uv (Kaggle's Python has no
# ensurepip, and no mediapipe 0.10.21 wheel exists for it).
make_venv() {
  echo "--- pip install uv"
  python -m uv --version 2>/dev/null || python -m pip install -q --disable-pip-version-check uv \
    || { echo "STEP FAILED: pip install uv"; return 1; }
  echo "--- uv python install $VENV_PYTHON"
  uv python install "$VENV_PYTHON" || { echo "STEP FAILED: uv python install $VENV_PYTHON"; return 1; }
  echo "--- uv venv --python $VENV_PYTHON $1"
  uv venv --clear --python "$VENV_PYTHON" "$1" || { echo "STEP FAILED: uv venv $1"; return 1; }
}

# The code comes ONLY from the private dataset edge-vision-src: edge-vision-src.tar.gz, made by
# "python scripts/build_kaggle_notebooks.py --source-archive", or that archive unpacked by Kaggle.
get_source() {
  local archives archive tree found
  rm -rf "$SRC"
  archives=$(find_inputs -type f -name 'edge-vision-src*.tar.gz')
  archive=$(echo "$archives" | head -1)
  tree=$(find_inputs -type f -name SOURCE_COMMIT | head -1)
  if [ "$(echo "$archives" | grep -c .)" -gt 1 ]; then
    echo "WARNING: several code archives are attached; using the first:"
    echo "$archives"
  fi
  if [ -n "$archive" ]; then
    echo "code: $archive"
    python -c 'import sys, tarfile; tarfile.open(sys.argv[1]).extractall(sys.argv[2], filter="data")' \
      "$archive" "$SRC" || { echo "CODE SOURCE UNREADABLE: could not unpack $archive"; return 1; }
  elif [ -n "$tree" ]; then
    echo "code: $(dirname "$tree") (the archive, unpacked by Kaggle)"
    cp -r "$(dirname "$tree")" "$SRC" \
      || { echo "CODE SOURCE UNREADABLE: could not copy $(dirname "$tree")"; return 1; }
  else
    echo "CODE SOURCE MISSING: no edge-vision-src.tar.gz under $EV_INPUT."
    echo "  These notebooks never clone the repository. Upload artifacts/v3/upload/edge-vision-src.tar.gz"
    echo "  (python scripts/build_kaggle_notebooks.py --source-archive) as a new Private dataset named"
    echo "  edge-vision-src, add it here with Add Input > Your Work > Datasets, and run again."
    return 1
  fi
  if [ ! -f "$SRC/SOURCE_COMMIT" ]; then
    echo "CODE SOURCE INVALID: no SOURCE_COMMIT in it; build the archive with --source-archive"
    return 1
  fi
  echo "code commit: $(cat "$SRC/SOURCE_COMMIT")"
  found=$(python "$SRC/scripts/build_kaggle_notebooks.py" --fingerprint 2>&1)
  if [ "$found" != "$SOURCE_FINGERPRINT" ]; then
    echo "CODE SOURCE MISMATCH: this notebook was built from notebook sources $SOURCE_FINGERPRINT,"
    echo "  the edge-vision-src input holds '$found'. Use the archive and the notebooks of one commit:"
    echo "  upload that commit's edge-vision-src.tar.gz as a new version of the dataset, or import"
    echo "  the notebooks again from the commit the archive was made from."
    return 1
  fi
  echo "code fingerprint: $found (matches this notebook)"
}
# ---- end of the common part ----
"""


def common() -> str:
    return COMMON.replace("@FINGERPRINT@", source_fingerprint()).replace(
        "@VENV_PYTHON@", VENV_PYTHON
    )


@dataclass(frozen=True)
class Extraction:
    name: str
    title: str
    note: str
    limit: int = 0
    part: str = "0/1"
    inputs: str = "Your Work > Datasets > edge-vision-src"
    dataset: bool = False


SMOKE = Extraction(
    name="asl_citizen_smoke",
    title="SMOKE TEST of the ASL Citizen landmark extraction",
    limit=SMOKE_VIDEOS,
    note=f"""# Tracks only the first {SMOKE_VIDEOS} planned videos into asl-citizen-landmarks-smoke (never the
# real shards, never a dataset) and prints the seconds per video and whether pose and hand
# landmarks were found (non-NaN) in most frames. A few minutes; the log ends with SMOKE PASS or
# SMOKE FAIL. Run it before the three parts.
""",
)


def part(number: int) -> Extraction:
    return Extraction(
        name=f"asl_citizen_part{number}",
        title=f"ASL Citizen landmark extraction, part {number} of {PARTS}",
        part=f"{number - 1}/{PARTS}",
        note=f"""# Tracks every {PARTS}rd shard (EV_PART {number - 1}/{PARTS}); run the {PARTS} part notebooks side by side, then
# asl_citizen_merge. Up to 11 hours. If the log says INCOMPLETE, add this notebook's own output
# as an input and run it again; finished shards are skipped.
""",
    )


MERGE = Extraction(
    name="asl_citizen_merge",
    title="ASL Citizen landmark extraction: merge and private dataset",
    inputs="Your Work > Datasets > edge-vision-src, and Your Work > Notebooks >\n"
    "# asl_citizen_part1, asl_citizen_part2 and asl_citizen_part3 (their finished versions)",
    dataset=True,
    note="""# Copies the shards of the three part outputs, tracks any shard still missing, writes
# manifest.csv and COMPLETE, then creates the PRIVATE dataset asl-citizen-landmarks-private
# with the KAGGLE_USERNAME and KAGGLE_KEY secrets (Add-ons > Secrets; tick both for this
# notebook), or prints the manual steps.
""",
)
EXTRACTIONS = (SMOKE, part(1), part(2), part(3), MERGE)

# The mediapipe venv, shared by the diagnostics cell and the extraction cells.
MPENV_SETUP = r"""MPENV="$TMPROOT/mpenv"
venv_versions() {
  clean "$(venv_python "$MPENV")" -c 'import platform, mediapipe, cv2, numpy; print(f"venv python {platform.python_version()}: mediapipe {mediapipe.__version__}, cv2 {cv2.__version__}, numpy {numpy.__version__}")'
}
setup_environment() {
  local py
  if [ -e "$(venv_python "$MPENV")" ] && venv_versions 2>/dev/null; then
    echo "reusing $MPENV"
    return 0
  fi
  make_venv "$MPENV" || return 1
  py=$(venv_python "$MPENV")
  echo '--- uv pip install @PACKAGES@'
  uv pip install --python "$py" @PACKAGES@ || { echo "STEP FAILED: uv pip install"; return 1; }
  # mediapipe also requires opencv-contrib-python, which writes the same cv2 package as the
  # headless build. Keep exactly one cv2: drop contrib, then restore the headless files it shared.
  if uv pip show --python "$py" opencv-contrib-python >/dev/null 2>&1; then
    OPENCV=$(uv pip show --python "$py" opencv-python-headless | sed -n 's/^Version: //p')
    echo "--- keeping only opencv-python-headless==$OPENCV"
    uv pip uninstall --python "$py" opencv-contrib-python \
      && uv pip install --python "$py" --reinstall "opencv-python-headless==$OPENCV" "numpy<2" \
      || { echo "STEP FAILED: replacing opencv-contrib-python with the headless build"; return 1; }
  fi
  echo "--- versions inside $MPENV"
  venv_versions || { echo "STEP FAILED: importing mediapipe, cv2 and numpy in $MPENV"; return 1; }
}
"""

EXTRACTION_RUN = r"""if [ "$EV_LIMIT_VIDEOS" -gt 0 ]; then
  OUT="$WORK/asl-citizen-landmarks-smoke"
else
  OUT="$WORK/asl-citizen-landmarks"
fi
export EV_OUT="$OUT"
LOG="$WORK/$NAME.log"
finish() {
  [ -n "${1:-}" ] && echo "$1" | tee -a "$LOG"
  echo "=== $(date -u '+%F %T') UTC: finished; this cell always exits 0, so read $LOG" | tee -a "$LOG"
  exit 0
}
echo "=== $(date -u '+%F %T') UTC: start $NAME (EV_PART $EV_PART, EV_LIMIT_VIDEOS $EV_LIMIT_VIDEOS); Kaggle python $(kaggle_python)" | tee -a "$LOG"
get_source 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || finish "STOPPED: no usable code source (see above); nothing was extracted."

setup_environment 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || finish "ENVIRONMENT FAILED: nothing was extracted. The full error is above and in $LOG."
PY=$(venv_python "$MPENV")

echo "--- the extraction program, built from the code source" | tee -a "$LOG"
rm -f "$WORK/$NAME.py"
python "$SRC/scripts/build_kaggle_notebooks.py" --script-out "$WORK/$NAME.py" 2>&1 | tee -a "$LOG"
[ -f "$WORK/$NAME.py" ] || finish "STOPPED: the extraction program could not be built (see above)."
clean "$PY" -u "$WORK/$NAME.py" 2>&1 | tee -a "$LOG"
echo "=== extraction exit status ${PIPESTATUS[0]}" | tee -a "$LOG"
"""

DATASET_STEP = r"""if [ -f "$OUT/COMPLETE" ]; then
  # Kaggle's own Python, where kaggle_secrets lives (not the extraction venv).
  echo "--- the private dataset" | tee -a "$LOG"
  python "$SRC/kaggle/make_private_dataset.py" "$OUT" 2>&1 | tee -a "$LOG"
else
  echo "no COMPLETE marker, so no dataset: the log above says what is missing" | tee -a "$LOG"
fi
"""


DIAGNOSTICS = r"""%%bash
# DIAGNOSTICS (edge-vision), the first cell of the smoke test. It needs no input and never stops
# the notebook: it prints the Python version, uv --version, the output of the venv creation and
# the package install (the same commands as the extraction, so the next cell reuses the venv),
# an import and Holistic test of mediapipe with Kaggle's environment and with clean(), ldd on
# mediapipe's native modules if that import fails, the free disk space and a HEAD request to the
# ASL Citizen archive. Everything also goes to diagnostics.log in the output.
set -u
@COMMON@@MPENV_SETUP@DIAG_LOG="$WORK/diagnostics.log"
URL="${EV_URL:-@URL@}"
# Variables whose values are shown; of the others only the names (they may hold tokens).
SHOWN='^(MPLBACKEND|PYTHON[A-Z_]*|LD_LIBRARY_PATH|LD_PRELOAD|UV_[A-Z_]*|PIP_[A-Z_]*|VIRTUAL_ENV|CONDA_[A-Z_]*|PATH|HOME|LANG|LC_ALL|TMPDIR)='
diagnostics() {
  echo "=== $(date -u '+%F %T') UTC: diagnostics"
  echo "--- Kaggle's Python"
  command -v python
  python -c 'import platform, sys; print(sys.version); print(platform.platform())'
  echo "--- environment variables (values only for the ones that steer Python, uv and pip)"
  env | grep -E "$SHOWN" | sort | cut -c1-300
  echo "other names: $(env | grep -vE "$SHOWN" | cut -d= -f1 | grep -E '^[A-Za-z_][A-Za-z0-9_]*$' | sort | tr '\n' ' ')"
  echo "--- uv --version"
  python -m uv --version 2>/dev/null || python -m pip install -q --disable-pip-version-check uv
  uv --version
  echo "--- the venv creation and the package install"
  setup_environment
  echo "setup_environment exit status: $?"
  local py
  py=$(venv_python "$MPENV")
  echo "--- import mediapipe with Kaggle's environment (how the extraction ran before the fix)"
  "$py" -c 'import mediapipe; print("import ok")' 2>&1 | tail -n 2
  echo "--- import mediapipe and run Holistic with clean() (how the extraction runs)"
  if clean "$py" -c 'import mediapipe as mp, numpy as np
h = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
h.process(np.zeros((360, 640, 3), np.uint8))
h.close()
print("mediapipe", mp.__version__, "import and Holistic ok")'; then
    echo "DIAGNOSTICS: mediapipe OK"
  else
    echo "DIAGNOSTICS: mediapipe FAILED; ldd of its native modules:"
    find "$MPENV" -path '*site-packages/mediapipe*' -name '*.so' | while read -r so; do
      echo "$so"
      ldd "$so" 2>&1 | sed 's/^/    /'
    done
  fi
  echo "--- free disk space"
  df -h "$WORK" "$TMPROOT" 2>&1
  echo "--- HEAD request to the ASL Citizen archive"
  python - "$URL" <<'PYEOF'
import sys, urllib.request

request = urllib.request.Request(sys.argv[1], method="HEAD")
try:
    with urllib.request.urlopen(request, timeout=60) as response:
        size = int(response.headers.get("Content-Length") or 0)
        print(f"{sys.argv[1]}: HTTP {response.status}, Content-Length {size:,} bytes "
              f"({size / 1e9:.2f} GB), Accept-Ranges {response.headers.get('Accept-Ranges')}")
except Exception as error:
    print(f"{sys.argv[1]}: HEAD FAILED, {type(error).__name__}: {error}")
PYEOF
  echo "=== $(date -u '+%F %T') UTC: diagnostics done"
}
diagnostics 2>&1 | tee -a "$DIAG_LOG"
exit 0
"""


def archive_url() -> str:
    """DEFAULT_URL of kaggle/asl_citizen_extract.py (two string literals in parentheses)."""
    match = re.search(
        r'DEFAULT_URL = \(\s*"([^"]+)"\s*"([^"]+)"\s*\)', EXTRACT_BODY.read_text("utf-8")
    )
    if not match:
        raise ValueError(f"no DEFAULT_URL in {EXTRACT_BODY}")
    return match.group(1) + match.group(2)


def mpenv_setup() -> str:
    return MPENV_SETUP.replace("@PACKAGES@", " ".join(MPENV_PACKAGES))


def diagnostics_cell() -> str:
    return (
        DIAGNOSTICS.replace("@COMMON@", common())
        .replace("@MPENV_SETUP@", mpenv_setup())
        .replace("@URL@", archive_url())
    )


def extraction_cell(spec: Extraction) -> str:
    header = f"""%%bash
# {spec.title} (edge-vision).
# Settings: Accelerator None (CPU), Internet On. Run with Save Version > Save & Run All (Commit).
# Inputs: {spec.inputs}.
{spec.note}# Licence: ASL Citizen is non-commercial research only; never redistribute it or the landmarks,
# keep this notebook and its outputs private, delete everything when the research is complete
# (docs/DATA_LICENSES.md). Generated by scripts/build_kaggle_notebooks.py; edit the sources.
set -u
export EV_LIMIT_VIDEOS={spec.limit}    # > 0: SMOKE TEST of the first N videos, into a separate -smoke folder
export EV_PART="{spec.part}"         # k/N: this run's share of the shards
NAME={spec.name}
"""
    dataset = DATASET_STEP if spec.dataset else ""
    return header + common() + mpenv_setup() + EXTRACTION_RUN + dataset + "finish\n"


def train_cell() -> str:
    body = TRAIN_BODY.read_text(encoding="utf-8")
    if body.count(COMMON_MARKER) != 1:
        raise ValueError(f"{TRAIN_BODY} must hold the line {COMMON_MARKER.strip()!r} once")
    return "%%bash\n" + body.replace(COMMON_MARKER, common())


def notebook(cells: dict[str, str], settings: dict[str, object]) -> str:
    """A notebook of bash code cells, {cell id: source}, run in order."""
    payload = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": cell_id,
                "metadata": {"trusted": True},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
            for cell_id, source in cells.items()
        ],
        "metadata": {
            "kaggle": {"language": "python", "sourceType": "notebook", **settings},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(payload, indent=1, ensure_ascii=False) + "\n"


def notebook_paths() -> dict[str, Path]:
    names = [spec.name for spec in EXTRACTIONS] + ["train_v3"]
    return {name: KAGGLE / f"{name}.ipynb" for name in names}


def built() -> dict[str, str]:
    notebooks = {}
    for spec in EXTRACTIONS:
        cells = {"diagnostics": diagnostics_cell()} if spec is SMOKE else {}
        cells["main"] = extraction_cell(spec)
        notebooks[spec.name] = notebook(cells, EXTRACT_SETTINGS)
    notebooks["train_v3"] = notebook({"main": train_cell()}, TRAIN_SETTINGS)
    return notebooks


def embedded_fingerprint(notebook_text: str) -> str | None:
    """The fingerprint in the notebook's main cell (the last one)."""
    source = "".join(json.loads(notebook_text)["cells"][-1]["source"])
    match = FINGERPRINT_LINE.search(source)
    return match.group(1) if match else None


def archive_problems(path: Path, commit: str | None = None) -> list[str]:
    """What is wrong with a code archive: no or another SOURCE_COMMIT, a forbidden path, a
    notebook missing or built from other sources than the archive holds."""
    problems = []
    with tarfile.open(path) as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}

        def read(name: str) -> bytes | None:
            member = members.get(name)
            handle = archive.extractfile(member) if member else None
            return handle.read() if handle else None

        found = read(COMMIT_FILE)
        if found is None:
            problems.append(f"no {COMMIT_FILE} file")
        elif commit and found.decode().strip() != commit:
            problems.append(f"{COMMIT_FILE} says {found.decode().strip()}, expected {commit}")
        forbidden = sorted(name for name in members if FORBIDDEN.search(name))
        if forbidden:
            problems.append(f"forbidden files: {', '.join(forbidden[:5])}")
        expected = fingerprint_of(read)
        for name in notebook_paths():
            relative = f"kaggle/{name}.ipynb"
            text = read(relative)
            if text is None:
                problems.append(f"{relative} is missing")
                continue
            embedded = embedded_fingerprint(text.decode("utf-8"))
            if embedded != expected:
                problems.append(
                    f"{relative} was built from sources {embedded}, the archive holds {expected}: "
                    "run python scripts/build_kaggle_notebooks.py and commit"
                )
    return problems


def git(*args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def source_archive(out: Path, ref: str = "HEAD", require_clean: bool = True) -> str:
    """Write the git archive of ref (with SOURCE_COMMIT) to out and check it; the commit."""
    if require_clean and ref == "HEAD":
        changed = git("status", "--porcelain", "--untracked-files=no")
        if changed:
            raise SystemExit(f"tracked files differ from HEAD; commit them first:\n{changed}")
    commit = git("rev-parse", "--verify", f"{ref}^{{commit}}")
    forbidden = [
        n for n in git("ls-tree", "-r", "--name-only", commit).splitlines() if FORBIDDEN.search(n)
    ]
    if forbidden:
        raise SystemExit(f"refusing to archive {commit}: forbidden files {forbidden[:5]}")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    git(
        "archive",
        "--format=tar.gz",
        f"--add-virtual-file={COMMIT_FILE}:{commit}",
        "-o",
        str(out),
        commit,
    )
    problems = archive_problems(out, commit)
    if problems:
        out.unlink()
        raise SystemExit("the archive was deleted:\n" + "\n".join(problems))
    return commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="fail when a notebook is stale")
    parser.add_argument("--script-out", type=Path, help="write only the extraction program here")
    parser.add_argument("--fingerprint", action="store_true", help="print the sources' fingerprint")
    parser.add_argument("--source-archive", type=Path, help="write the code archive here")
    parser.add_argument("--ref", default="HEAD", help="the commit to archive (default HEAD)")
    args = parser.parse_args(argv)
    if args.fingerprint:
        print(source_fingerprint())
        return 0
    if args.script_out:
        args.script_out.write_text(extraction_program(), encoding="utf-8", newline="\n")
        print(f"wrote {args.script_out}")
        return 0
    if args.source_archive:
        commit = source_archive(args.source_archive, args.ref)
        size = args.source_archive.stat().st_size / 1e6
        print(
            f"wrote {args.source_archive} ({size:.1f} MB): commit {commit}, notebook sources "
            f"{source_fingerprint()}; no forbidden files"
        )
        return 0
    stale = []
    for name, text in built().items():
        path = notebook_paths()[name]
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if args.check:
            if current != text:
                stale.append(path.relative_to(ROOT).as_posix())
            continue
        if current != text:
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {path.relative_to(ROOT).as_posix()}")
    known = {path.name for path in notebook_paths().values()}
    extra = sorted(p.name for p in KAGGLE.glob("*.ipynb") if p.name not in known)
    if extra:
        print(f"not built here (remove them): {', '.join(extra)}", file=sys.stderr)
        stale.extend(extra)
    if stale:
        print(
            f"stale: {', '.join(stale)}; run python scripts/build_kaggle_notebooks.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
