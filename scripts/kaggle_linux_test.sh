#!/usr/bin/env bash
# scripts/kaggle_local_test.py on Linux (WSL or a container): the closest local stand-in for a
# Kaggle notebook. uv's standalone CPython 3.13 plays Kaggle's Python (pip, no uv); the cells then
# install uv, Python 3.11 and mediapipe themselves, exactly as on Kaggle, with the variables the
# Kaggle kernel exports (kaggle_local_test.KAGGLE_ENV). Needs curl and internet access to GitHub
# (uv, Python builds) and PyPI; with --real-archive also the Microsoft download URL.
#
#     bash scripts/kaggle_linux_test.sh WORK_DIR [kaggle_local_test.py arguments]
#
# Example, from WSL:
#     bash scripts/kaggle_linux_test.sh ~/ev-kaggle-test \
#         --archive artifacts/v3/upload/edge-vision-src.tar.gz --reference data/live_clips_v2 \
#         test_clips_v2/bed_1.mp4 test_clips_v2/bird_1.mp4 test_clips_v2/book_1.mp4
#
# Everything goes under WORK_DIR (keep it on a Linux filesystem, not /mnt/c); delete it after,
# since it holds landmarks of the clips.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$1"
shift
mkdir -p "$WORK/bin"
WORK="$(cd "$WORK" && pwd)"
export UV_CACHE_DIR="$WORK/uv-cache" UV_PYTHON_INSTALL_DIR="$WORK/uv-python"
if [ ! -x "$WORK/bin/uv" ]; then
  curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
    | tar -xz -C "$WORK/bin" --strip-components=1
fi
"$WORK/bin/uv" --version
"$WORK/bin/uv" python install 3.13
KAGGLE_PYTHON="$("$WORK/bin/uv" python find --managed-python 3.13)"
# The harness itself: Python 3.13 with the repository's pinned requirements (locate imports them).
if [ ! -x "$WORK/harness/bin/python" ]; then
  "$WORK/bin/uv" venv --python "$KAGGLE_PYTHON" "$WORK/harness"
  "$WORK/bin/uv" pip install --python "$WORK/harness/bin/python" -r "$REPO/requirements.txt"
fi
cd "$REPO"
PYTHONPATH="$REPO/src" "$WORK/harness/bin/python" scripts/kaggle_local_test.py \
  --kaggle-python "$KAGGLE_PYTHON" --work "$WORK/run" "$@"
