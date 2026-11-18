"""Run the Kaggle extraction notebooks locally, end to end, on a few clips.

    python scripts/kaggle_local_test.py --archive artifacts/v3/upload/edge-vision-src.tar.gz \\
        --kaggle-python "C:/Program Files/Python313/python.exe" --work DIR \\
        test_clips_v2/bed_1.mp4 test_clips_v2/bird_1.mp4 test_clips_v2/book_1.mp4

It stands in for Kaggle. A venv of --kaggle-python plays Kaggle's own Python (pip, no uv, a
PYTHONPATH sitecustomize that breaks in other Pythons, as Kaggle's imports wrapt); a stand-in
ASL_Citizen.zip holding the clips (one per official split, gloss = the clip's sign in capitals;
the split lists also name every gloss the label mapping names, without a video) is served over
HTTP with range requests; each notebook's cell runs in bash, as written in
kaggle/*.ipynb, with EV_URL, EV_WORK, EV_INPUT and EV_TMP pointing into --work. The steps, in
the order of a Kaggle run: no code source (must stop), a code source that does not match the
notebooks (must stop), the smoke test, parts 1-3 (part 2 gets the archive unpacked, as Kaggle
may show it), the merge (the three part outputs and the smoke output as inputs), then
v3_kaggle.py locate on the merge output as a notebook output and as a zipped dataset, and the
landmarks read back and, with --reference, compared with a local Holistic run of the same
clips. Whether pose and hands are in most frames depends on the clips, so it is reported here
("info"), not checked. Writes report.json under --work and exits 1 when a check fails.

The clips stay on this machine; --work ends up holding landmarks of them, so delete it after.
"""

import argparse
import http.server
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
POSE, LEFT, RIGHT, N_POINTS = slice(489, 522), slice(468, 489), slice(522, 543), 543
# Kaggle's sitecustomize imports wrapt; this one fails the same way in any Python but the
# "Kaggle" one, so a venv run without clean() would print "Error in sitecustomize".
SITECUSTOMIZE = """import sys
if sys.version_info[:2] != ({major}, {minor}):
    raise ModuleNotFoundError("No module named 'wrapt' (stand-in for Kaggle's sitecustomize)")
"""
# Variables the Kaggle kernel exports into %%bash cells, as its logs show: the notebook's
# matplotlib backend (mediapipe imports matplotlib.pyplot; the venv has no matplotlib_inline)
# and uv's system-Python switch ("The `--system` flag has no effect" in uv venv's output).
KAGGLE_ENV = {
    "MPLBACKEND": "module://matplotlib_inline.backend_inline",
    "UV_SYSTEM_PYTHON": "true",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/nvidia/lib64",
    "KAGGLE_KERNEL_RUN_TYPE": "Batch",
}


class RangeServer(http.server.ThreadingHTTPServer):
    archive: Path


class RangeHandler(http.server.BaseHTTPRequestHandler):
    server: RangeServer

    def do_HEAD(self) -> None:
        self.reply(body=False)

    def do_GET(self) -> None:
        self.reply(body=True)

    def reply(self, body: bool) -> None:
        size = self.server.archive.stat().st_size
        start, end = 0, size - 1
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        if match:
            start = int(match.group(1))
            end = min(int(match.group(2)) if match.group(2) else size - 1, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        if body:
            with self.server.archive.open("rb") as handle:
                handle.seek(start)
                self.wfile.write(handle.read(end - start + 1))

    def log_message(self, format: str, *args: Any) -> None:
        pass


def named_glosses() -> list[str]:
    """Every gloss the label mapping names (strict mode requires them in the dataset)."""
    from edgevision.data import citizen_labels as cl

    named = {gloss for glosses, _ in cl.SYNONYMS.values() for gloss in glosses}
    return sorted(named | set(cl.VARIANT_EXCLUDED))


def standin_archive(clips: list[Path], out: Path) -> dict[str, str]:
    """ASL_Citizen.zip with the clips as videos, one split each in turn; {video file: label}."""
    rows: dict[str, list[str]] = {split: [] for split in SPLITS}
    labels = {}
    with zipfile.ZipFile(out, "w") as archive:
        for number, clip in enumerate(clips):
            sign = clip.stem.split("_")[0]
            name = f"{1000 + number}-{sign.upper()}.mp4"
            method = zipfile.ZIP_DEFLATED if number == 0 else zipfile.ZIP_STORED
            archive.write(clip, f"ASL_Citizen/videos/{name}", compress_type=method)
            rows[SPLITS[number % 3]].append(f"P{number + 1},{name},{sign.upper()},X_00_000")
            labels[name] = sign
        # The mapping checks that every gloss it names is in the dataset (a guard against another
        # release); list each with a video the archive lacks, as the real plan skips missing ones.
        for number, gloss in enumerate(named_glosses()):
            rows[SPLITS[number % 3]].append(f"P9,absent-{gloss}.mp4,{gloss},X_00_000")
        for split in SPLITS:
            text = "Participant ID,Video file,Gloss,ASL-LEX Code\n" + "".join(
                row + "\n" for row in rows[split]
            )
            archive.writestr(f"ASL_Citizen/splits/{split}.csv", text)
        archive.writestr("ASL_Citizen/use.txt", "Stand-in archive for a local test.\n")
        # The program reads each video with 1 KB of slack; keep data after the last video.
        archive.writestr("ASL_Citizen/zz_padding.txt", "-" * 8192)
    return labels


def cell_scripts(name: str, folder: Path) -> list[Path]:
    """Every cell of kaggle/<name>.ipynb as a bash script, in notebook order."""
    notebook = json.loads((ROOT / "kaggle" / f"{name}.ipynb").read_text("utf-8"))
    scripts = []
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        assert source.startswith("%%bash\n"), (name, cell["id"])
        script = folder / f"{name}.{cell['id']}.sh"
        script.write_text(source.split("\n", 1)[1], encoding="utf-8", newline="\n")
        scripts.append(script)
    return scripts


def tampered(archive: Path, out: Path) -> None:
    """The archive with one notebook source changed, so its fingerprint differs."""
    with tarfile.open(archive) as source, tarfile.open(out, "w:gz") as target:
        for member in source.getmembers():
            handle = source.extractfile(member) if member.isfile() else None
            data = handle.read() if handle else None
            if member.name == "scripts/v3_kaggle.py" and data is not None:
                data += b"\n# changed\n"
                member.size = len(data)
            if data is None:
                target.addfile(member)
            else:
                target.addfile(member, io.BytesIO(data))


def presence(landmarks: np.ndarray) -> tuple[float, float]:
    if len(landmarks) == 0:
        return 0.0, 0.0
    pose = np.isfinite(landmarks[:, POSE, 0]).any(axis=1)
    hands = np.isfinite(landmarks[:, LEFT, 0]).any(axis=1) | np.isfinite(
        landmarks[:, RIGHT, 0]
    ).any(axis=1)
    return float(pose.mean()), float(hands.mean())


def shard_landmarks(root: Path, video: str) -> np.ndarray:
    import csv

    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        row = next(r for r in csv.DictReader(handle) if r["video_file"] == video)
    with np.load(root / "shards" / f"shard_{int(row['shard']):05d}.npz") as data:
        start, count = int(row["start"]), int(row["n_frames"])
        return np.asarray(data["landmarks"][start : start + count])


def read_back(root: Path) -> list[dict[str, Any]]:
    """Per tracked video of a landmark folder: frames and the pose / hand presence."""
    import csv

    videos = []
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok":
                videos.append({"video": row["video_file"], "status": row["status"]})
                continue
            with np.load(root / "shards" / f"shard_{int(row['shard']):05d}.npz") as data:
                landmarks = data["landmarks"]
            start, count = int(row["start"]), int(row["n_frames"])
            pose, hands = presence(landmarks[start : start + count])
            videos.append(
                {
                    "video": row["video_file"],
                    "label": row["label"],
                    "rule": row["rule"],
                    "split": row["split"],
                    "frames": count,
                    "pose": round(pose, 3),
                    "hands": round(hands, 3),
                    "shape_ok": landmarks.shape[1:] == (N_POINTS, 3),
                }
            )
    return videos


class Run:
    def __init__(self, args: argparse.Namespace) -> None:
        self.work: Path = args.work.resolve()
        self.bash: str = args.bash or shutil.which("bash") or "bash"
        self.archive: Path = args.archive.resolve()
        self.reference: Path | None = args.reference
        self.checks: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.env: dict[str, str] = {}
        self.real_archive: bool = args.real_archive
        self.diagnostics = ""

    def check(self, step: str, what: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"step": step, "check": what, "ok": bool(ok), "detail": detail})
        print(f"  [{'ok' if ok else 'FAIL'}] {step}: {what}" + (f" ({detail})" if detail else ""))

    def info(self, step: str, text: str) -> None:
        """A result that depends on the clips' content: reported, not a pass/fail check."""
        self.notes.append(f"{step}: {text}")
        print(f"  [info] {step}: {text}")

    def run_cell(self, step: str, notebook: str, inputs: dict[str, Path]) -> tuple[str, Path]:
        """Run a notebook cell with these inputs ({mount name: file or folder}); (log, working)."""
        base = self.work / "runs" / step
        if base.exists():
            shutil.rmtree(base)
        working, input_root = base / "working", base / "input"
        working.mkdir(parents=True)
        input_root.mkdir()
        for mount, source in inputs.items():
            target = input_root / mount
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target / source.name)
        env = dict(self.env, EV_WORK=working.as_posix(), EV_INPUT=input_root.as_posix())
        # Like Save & Run All: every cell in order, each its own bash process.
        for script in cell_scripts(notebook, base):
            started = time.time()
            done = subprocess.run(
                [self.bash, str(script)], env=env, capture_output=True, text=True, timeout=7200
            )
            seconds = time.time() - started
            output = base / f"{script.stem}.output.txt"
            output.write_text(done.stdout + done.stderr, encoding="utf-8")
            print(f"{step}: {script.stem} exited {done.returncode} after {seconds:.0f} s")
            self.check(step, f"cell {script.stem} exits 0", done.returncode == 0)
        log_path = working / f"{notebook}.log"
        log = log_path.read_text("utf-8", errors="replace") if log_path.is_file() else ""
        diagnostics = working / "diagnostics.log"
        self.diagnostics = (
            diagnostics.read_text("utf-8", errors="replace") if diagnostics.is_file() else ""
        )
        print(f"{step}: log {log_path}")
        self.check(
            step, "no 'Error in sitecustomize' (clean() keeps Kaggle's hooks out of the venv)",
            "Error in sitecustomize" not in log,
        )  # fmt: skip
        return log, working

    def check_diagnostics(self, step: str, fresh: bool) -> None:
        """The smoke notebook's first cell: what it must print. A fresh run (no venv yet, as on
        Kaggle) must show the venv creation and install; a later one may reuse the venv."""
        text = self.diagnostics
        self.check(
            step, "diagnostics: Python version", bool(re.search(r"^3\.\d+\.\d+", text, re.M))
        )
        uv = re.search(r"^uv \d+\.\d+\.\d+.*$", text, re.M)
        self.check(step, "diagnostics: uv --version", bool(uv), uv[0] if uv else "")
        installed = "--- uv venv --python" in text and "--- uv pip install" in text
        reused = "reusing " in text
        self.check(step, "diagnostics: venv creation and install output"
                   + ("" if fresh else " (or the venv of an earlier step reused)"),
                   installed or (reused and not fresh))  # fmt: skip
        self.check(step, "diagnostics: Kaggle's MPLBACKEND breaks the import without clean()",
                   "matplotlib_inline.backend_inline' is not a valid value" in text)  # fmt: skip
        self.check(step, "diagnostics: mediapipe imports and runs Holistic with clean()",
                   "DIAGNOSTICS: mediapipe OK" in text)  # fmt: skip
        self.check(step, "diagnostics: free disk space", "--- free disk space" in text
                   and "Avail" in text)  # fmt: skip
        head = re.search(r"^\S+: (HTTP \d+.*|HEAD FAILED.*)$", text, re.M)
        answer = head[1] if head else "no HEAD line"
        self.check(step, "diagnostics: HEAD request answered with ranges",
                   "Accept-Ranges bytes" in answer, answer)  # fmt: skip

    def real_smoke(self) -> int:
        """The smoke notebook as on Kaggle, against the Microsoft download URL (20 videos)."""
        log, smoke = self.run_cell(
            "3-smoke-real", "asl_citizen_smoke", {"edge-vision-src": self.archive}
        )
        step = "3-smoke-real"
        self.check_diagnostics(step, fresh=True)
        venv = re.search(r"venv python (\S+): mediapipe (\S+), cv2 (\S+), numpy (\S+)", log)
        self.check(step, "venv versions printed", bool(venv), venv[0] if venv else "")
        archive = re.search(r"ASL_Citizen\.zip: ([\d.]+ GB)[^\n]*", log)
        self.check(
            step, "reads the real archive index", bool(archive), archive[0] if archive else ""
        )
        per_video = re.search(r"time: [^\n]*", log)
        self.check(
            step, "prints the seconds per video", bool(per_video), per_video[0] if per_video else ""
        )
        landmarks = re.search(r"landmarks: [^\n]*", log)
        self.info(step, landmarks[0] if landmarks else "no landmark summary")
        verdict = re.search(r"SMOKE (PASS|FAIL)[^\n]*", log)
        said = verdict[0] if verdict else "no verdict"
        self.check(
            step, "SMOKE PASS on real ASL Citizen videos", said.startswith("SMOKE PASS"), said
        )
        self.check(step, "writes only the -smoke folder",
                   (smoke / "asl-citizen-landmarks-smoke" / "SMOKE").is_file()
                   and not (smoke / "asl-citizen-landmarks").exists())  # fmt: skip
        return self.finish([])

    def finish(self, clips: list[Path], videos: list[dict[str, Any]] | None = None) -> int:
        failed = [check for check in self.checks if not check["ok"]]
        report = {"archive": str(self.archive), "clips": [str(c) for c in clips],
                  "videos": videos or [], "checks": self.checks, "notes": self.notes,
                  "failed": len(failed)}  # fmt: skip
        (self.work / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\n{len(self.checks) - len(failed)}/{len(self.checks)} checks passed; "
              f"report {self.work / 'report.json'}")  # fmt: skip
        return 1 if failed else 0

    def main(self, clips: list[Path], kaggle_python: str) -> int:
        self.work.mkdir(parents=True, exist_ok=True)
        # Kaggle's Python: a venv with pip and no uv, plus a sitecustomize on PYTHONPATH.
        kaggle_env = self.work / "kaggle-python"
        if not kaggle_env.exists():
            subprocess.run([kaggle_python, "-m", "venv", str(kaggle_env)], check=True)
        bin_dir = kaggle_env / ("Scripts" if os.name == "nt" else "bin")
        version = subprocess.run(
            [str(bin_dir / "python"), "-c", "import sys; print(*sys.version_info[:2])"],
            capture_output=True, text=True, check=True,
        ).stdout.split()  # fmt: skip
        hooks = self.work / "kaggle-pythonpath"
        hooks.mkdir(exist_ok=True)
        (hooks / "sitecustomize.py").write_text(
            SITECUSTOMIZE.format(major=version[0], minor=version[1]), encoding="utf-8"
        )
        labels: dict[str, str] = {}
        server = None
        url = ""
        if not self.real_archive:
            server_dir = self.work / "server"
            server_dir.mkdir(exist_ok=True)
            standin = server_dir / "ASL_Citizen.zip"
            labels = standin_archive(clips, standin)
            server = RangeServer(("127.0.0.1", 0), RangeHandler)
            server.archive = standin
            threading.Thread(target=server.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{server.server_address[1]}/ASL_Citizen.zip"
        print(
            f"Kaggle's Python: {'.'.join(version)} ({kaggle_env}); archive {url or 'Microsoft URL'}"
        )
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("EV_", "PYTHON", "VIRTUAL_ENV", "UV_", "CONDA"))
        }
        self.env.update(
            PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
            PYTHONPATH=str(hooks),
            EV_TMP=(self.work / "tmp").as_posix(),
            UV_CACHE_DIR=str(self.work / "uv-cache"),
            UV_PYTHON_INSTALL_DIR=str(self.work / "uv-python"),
            UV_PYTHON_PREFERENCE="only-managed",  # Kaggle has no system Python 3.11 either
        )
        self.env.update(KAGGLE_ENV)
        if self.real_archive:
            return self.real_smoke()
        # The stand-in archive: its URL, and one clip per shard so the three parts share them.
        self.env.update(EV_URL=url, EV_SHARD_SIZE="1", EV_WORKERS="2")
        source = {"edge-vision-src": self.archive}

        log, _ = self.run_cell("1-no-source", "asl_citizen_smoke", {})
        self.check("1-no-source", "stops with CODE SOURCE MISSING", "CODE SOURCE MISSING" in log)
        self.check("1-no-source", "never clones", "git clone" not in log)
        self.check(
            "1-no-source", "the main cell builds no environment", "pip install uv" not in log
        )
        self.check_diagnostics("1-no-source", fresh=True)  # it needs no input

        bad = self.work / "tampered" / "edge-vision-src.tar.gz"
        bad.parent.mkdir(exist_ok=True)
        tampered(self.archive, bad)
        log, _ = self.run_cell("2-mismatch", "asl_citizen_smoke", {"edge-vision-src": bad})
        self.check("2-mismatch", "stops with CODE SOURCE MISMATCH", "CODE SOURCE MISMATCH" in log)

        log, smoke = self.run_cell("3-smoke", "asl_citizen_smoke", source)
        self.check_diagnostics("3-smoke", fresh=False)
        commit = re.search(r"code commit: ([0-9a-f]{40})", log)
        self.check("3-smoke", "prints the code commit", bool(commit), commit[1] if commit else "")
        self.check("3-smoke", "the fingerprint matches", "(matches this notebook)" in log)
        venv = re.search(r"venv python (\S+): mediapipe (\S+), cv2 (\S+), numpy (\S+)", log)
        self.check("3-smoke", "venv versions printed", bool(venv), venv[0] if venv else "")
        if venv:
            self.check("3-smoke", "venv is Python 3.11", venv[1].startswith("3.11."), venv[1])
            self.check("3-smoke", "mediapipe 0.10.21", venv[2] == "0.10.21", venv[2])
            self.check("3-smoke", "numpy < 2", venv[4].startswith("1."), venv[4])
        per_video = re.search(r"time: ([\d.]+) s per video per worker", log)
        self.check(
            "3-smoke", "prints the seconds per video", bool(per_video),
            per_video[0] if per_video else "",
        )  # fmt: skip
        # The verdict needs pose and a hand in most frames, which depends on the clips (ASL
        # Citizen videos on Kaggle); here it is reported, and step 7 checks the tracking itself.
        verdict = re.search(r"SMOKE (PASS|FAIL)[^\n]*", log)
        self.check("3-smoke", "reaches a SMOKE PASS / FAIL verdict", verdict is not None)
        self.info("3-smoke", verdict[0] if verdict else "no verdict")
        smoke_out = smoke / "asl-citizen-landmarks-smoke"
        self.check(
            "3-smoke", "writes to the separate -smoke folder", (smoke_out / "SMOKE").is_file()
        )
        self.check("3-smoke", "never writes COMPLETE", not (smoke_out / "COMPLETE").exists())
        self.check("3-smoke", "never writes the real folder",
                   not (smoke / "asl-citizen-landmarks").exists())  # fmt: skip

        unpacked = self.work / "unpacked-source"
        if unpacked.exists():
            shutil.rmtree(unpacked)
        with tarfile.open(self.archive) as archive:
            archive.extractall(unpacked, filter="data")
        parts = {}
        for number in (1, 2, 3):
            step = f"4-part{number}"
            inputs = {"edge-vision-src": unpacked if number == 2 else self.archive}
            log, parts[number] = self.run_cell(step, f"asl_citizen_part{number}", inputs)
            if number == 2:
                self.check(
                    step, "takes the archive unpacked by Kaggle", "unpacked by Kaggle" in log
                )
            self.check(
                step, f"part {number - 1}/3 finished", f"part {number - 1}/3 finished" in log
            )
            complete = parts[number] / "asl-citizen-landmarks" / "COMPLETE"
            self.check(step, "no COMPLETE in a part", not complete.exists())

        merge_inputs = dict(source)
        for number, working in parts.items():
            merge_inputs[f"asl_citizen_part{number}"] = working
        merge_inputs["asl_citizen_smoke"] = smoke
        log, merge = self.run_cell("5-merge", "asl_citizen_merge", merge_inputs)
        carried = re.search(r"carried over (\d+) finished shards", log)
        shards_carried = carried[1] if carried else "none"
        self.check(
            "5-merge", "carries over the 3 part shards", shards_carried == "3", shards_carried
        )
        self.check("5-merge", "COMPLETE", "COMPLETE: all 3 shards" in log)
        self.check("5-merge", "tracks nothing again", "0 to go" in log)
        self.check(
            "5-merge", "dataset step runs and, without Kaggle secrets, prints the manual steps",
            "DATASET NOT CREATED AUTOMATICALLY" in log and "Private" in log,
        )  # fmt: skip
        out = merge / "asl-citizen-landmarks"

        # The training side: locate on the merge output (a notebook output) and on a zipped
        # dataset (kaggle datasets create --dir-mode zip uploads shards/ as shards.zip).
        locate_input = self.work / "locate" / "input"
        if locate_input.parent.exists():
            shutil.rmtree(locate_input.parent)
        shutil.copytree(merge, locate_input / "asl_citizen_merge")
        shutil.copytree(smoke, locate_input / "asl_citizen_smoke")
        shutil.copytree(parts[1], locate_input / "asl_citizen_part1")
        found = self.locate(locate_input)
        self.check("6-locate", "picks the merge output",
                   found.endswith("asl_citizen_merge/asl-citizen-landmarks"), found)  # fmt: skip
        zipped_input = self.work / "locate" / "zipped"
        dataset = zipped_input / "asl-citizen-landmarks-private"
        dataset.mkdir(parents=True)
        for path in out.iterdir():
            if path.is_file():
                shutil.copy2(path, dataset / path.name)
        shutil.make_archive(str(dataset / "shards"), "zip", out / "shards")
        found = self.locate(zipped_input)
        self.check("6-locate", "gathers a shards.zip dataset",
                   found.endswith("asl-citizen-landmarks-input"), found)  # fmt: skip

        videos = read_back(out) if (out / "manifest.csv").is_file() else []
        tracked = [video for video in videos if video.get("frames")]
        self.check("7-landmarks", "every clip tracked", len(tracked) == len(labels),
                   f"{len(tracked)} of {len(labels)}")  # fmt: skip
        stems = {
            f"{1000 + number}-{clip.stem.split('_')[0].upper()}.mp4": clip.stem
            for number, clip in enumerate(clips)
        }
        for video in tracked:
            name = video["video"]
            self.check("7-landmarks", f"{name}: label {video['label']}, (T, 543, 3) layout",
                       video["label"] == labels[name] and video["shape_ok"])  # fmt: skip
            most = video["pose"] > 0.5 and video["hands"] > 0.5
            self.info(
                "7-landmarks",
                f"{name}: {video['frames']} frames, pose in {video['pose']:.0%}, a hand in "
                f"{video['hands']:.0%} (both in most frames: {'yes' if most else 'no'})",
            )
            reference = self.reference / f"{stems[name]}.holistic.npz" if self.reference else None
            if reference and reference.is_file():
                self.compare(name, video, reference, out)
        if server:
            server.shutdown()
        return self.finish(clips, videos)

    def compare(self, name: str, video: dict[str, Any], reference: Path, out: Path) -> None:
        """The cell's landmarks against an earlier local Holistic run of the same clip
        (scripts/prepare_live_clips.py: <clip>.holistic.npz with pose, left_hand, right_hand)."""
        with np.load(reference) as data:
            pose, left, right = data["pose"], data["left_hand"], data["right_hand"]

        def found(block: np.ndarray) -> np.ndarray:
            return np.asarray(np.isfinite(block[:, :, 0]).any(axis=1))

        ref_pose = float(found(pose).mean())
        ref_hands = float((found(left) | found(right)).mean())
        landmarks = shard_landmarks(out, name)
        cell_right = landmarks[:, RIGHT]
        diff = float("nan")
        if len(right) == len(landmarks):
            both = found(right) & found(cell_right)
            if both.any():
                diff = float(np.median(np.abs(right[both, :, :2] - cell_right[both, :, :2])))
        self.check(
            "7-landmarks",
            f"{name}: matches the local Holistic run of the clip (frames, presence within 5 "
            "points, right-hand x/y within 0.02)",
            len(pose) == video["frames"]
            and abs(ref_pose - video["pose"]) <= 0.05
            and abs(ref_hands - video["hands"]) <= 0.05
            and diff <= 0.02,
            f"local: {len(pose)} frames, pose {ref_pose:.0%}, a hand {ref_hands:.0%}; median "
            f"right-hand x/y difference {diff:.4f}",
        )

    def locate(self, input_root: Path) -> str:
        unpack = input_root.parent / f"unpack-{input_root.name}"
        unpack.mkdir(exist_ok=True)
        done = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "v3_kaggle.py"), "locate",
             "--input", str(input_root), "--unpack-dir", str(unpack)],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        (input_root.parent / f"locate-{input_root.name}.txt").write_text(
            done.stdout + done.stderr, encoding="utf-8"
        )
        print(done.stderr.strip())
        match = re.search(r"^LANDMARKS=(.*)$", done.stdout, re.MULTILINE)
        value = match.group(1).strip("'") if match else ""
        return value.replace("\\", "/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("clips", nargs="*", type=Path, help="video clips named <sign>_<n>.mp4")
    parser.add_argument("--archive", type=Path, required=True, help="edge-vision-src.tar.gz")
    parser.add_argument("--kaggle-python", default=sys.executable, help="plays Kaggle's Python")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--bash", help="bash to run the cells with (default: the one on PATH)")
    parser.add_argument(
        "--reference", type=Path, help="<clip>.holistic.npz of the clips from a local run, e.g. "
        "data/live_clips_v2; the cell's landmarks must match them",
    )  # fmt: skip
    parser.add_argument(
        "--real-archive", action="store_true",
        help="run only the smoke notebook, unchanged, against the Microsoft download URL",
    )  # fmt: skip
    args = parser.parse_args(argv)
    if not args.real_archive and not args.clips:
        parser.error("give the clips, or --real-archive")
    return Run(args).main([clip.resolve() for clip in args.clips], args.kaggle_python)


if __name__ == "__main__":
    raise SystemExit(main())
