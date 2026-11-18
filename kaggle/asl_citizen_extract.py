# ASL Citizen -> MediaPipe Holistic landmarks for the edge-vision vocabulary (body of the Kaggle
# notebooks kaggle/asl_citizen_*.ipynb; scripts/build_kaggle_notebooks.py --script-out puts
# src/edgevision/data/citizen_labels.py and the 250 labels above this file).
#
# The archive (about 46 GB) is never downloaded whole: its central directory is read with HTTP
# range requests, and each video of a mapped sign is fetched on its own, decompressed in memory,
# tracked and deleted. Results go to OUT in shards of SHARD_SIZE videos; a shard is written
# through a temporary name, so a shard file on disk is complete. A rerun skips finished shards,
# also ones found under /kaggle/input (a previous version's output added as an input).
#
# Licence (Microsoft Research License Terms, use.txt in the archive): non-commercial research
# only; the data and anything derived from it (these landmarks included) must not be
# redistributed; keep the notebook and the dataset private; delete everything when the research
# is complete. See docs/DATA_LICENSES.md.

import csv
import hashlib
import io
import json
import multiprocessing
import os
import platform
import shutil
import struct
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

DEFAULT_URL = (
    "https://download.microsoft.com/download/b/8/8/"
    "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
)
# EV_URL points the program at a stand-in archive for local tests; leave it unset on Kaggle.
URL = os.environ.get("EV_URL") or DEFAULT_URL
# > 0: a smoke test on the first N planned videos (kaggle/asl_citizen_smoke.ipynb uses 20).
LIMIT_VIDEOS = int(os.environ.get("EV_LIMIT_VIDEOS", "0"))
SMOKE = LIMIT_VIDEOS > 0
_OUT = Path(os.environ.get("EV_OUT", "/kaggle/working/asl-citizen-landmarks"))
# A smoke test always writes to its own "-smoke" folder, so it never mixes with the real shards,
# never writes COMPLETE and is never picked up as training data (it writes SMOKE instead).
OUT = _OUT.with_name(_OUT.name + "-smoke") if SMOKE and not _OUT.name.endswith("-smoke") else _OUT
INPUT_ROOT = Path(os.environ.get("EV_INPUT", "/kaggle/input"))
SHARD_SIZE = int(os.environ.get("EV_SHARD_SIZE", "200"))
WORKERS = int(os.environ.get("EV_WORKERS", str(os.cpu_count() or 4)))
# Stop starting shards after this many hours, so a 12-hour commit still saves its output.
MAX_HOURS = float(os.environ.get("EV_MAX_HOURS", "11"))
# The smoke check. A video passes when pose is found in more than SMOKE_MOST_FRAMES of its
# frames and at least one hand in more than SMOKE_MOST_FRAMES of its signing span (its first to
# last frame with a hand), in at least SMOKE_MIN_HAND_FRAMES frames. The span, not the whole
# video: ASL Citizen signers start and end with their hands at rest out of view (on 20 real
# videos, 2026-11-18: a hand in 41% of all frames but 98% of the span, medians), and training
# keeps only frames with a hand (min_hand_frames 3). The smoke test passes when
# SMOKE_PASS_FRACTION of the videos were tracked and that many of the tracked ones pass.
SMOKE_MOST_FRAMES = 0.5
SMOKE_MIN_HAND_FRAMES = 3
SMOKE_PASS_FRACTION = 0.8
# "k/N": process only the shards whose index is k modulo N, so N copies of the notebook can run
# side by side; a run with every part's output as an input then merges them.
PART = os.environ.get("EV_PART", "0/1")
# Face landmarks are not used by any edge-vision model. They stay NaN in the stored layout
# unless EV_KEEP_FACE=1, which keeps the private dataset free of face meshes.
KEEP_FACE = os.environ.get("EV_KEEP_FACE", "0") == "1"
# Frames wider than this are scaled down (aspect kept) before tracking, as the live tools feed
# 640-pixel frames; landmarks are normalized to the frame, so the scale does not shift them.
MAX_WIDTH = 640
# scripts/extract_holistic.py: legacy Holistic as the asl-signs data was made (mediapipe 0.10.21,
# default detection and tracking confidence 0.5).
HOLISTIC = {
    "static_image_mode": False,
    "model_complexity": 1,
    "smooth_landmarks": True,
    "refine_face_landmarks": False,
}
# The asl-signs parquet layout: face 0-467, left_hand 468-488, pose 489-521, right_hand 522-542,
# x, y, z each, NaN where Holistic found nothing. left_hand is the signer's own left hand.
FACE, LEFT, POSE, RIGHT, N_POINTS = 0, 468, 489, 522, 543
SPLITS = ("train", "val", "test")
LAYOUT_VERSION = 1
HTTP_TRIES = 6


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def http_range(start: int, length: int) -> bytes:
    """bytes [start, start + length) of the archive, retried with backoff."""
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start + length - 1}"})
    for attempt in range(HTTP_TRIES):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise OSError(f"HTTP {response.status} for a range request")
                data = response.read()
            if len(data) != length:
                raise OSError(f"short read: {len(data)} of {length} bytes")
            return data
        except (OSError, urllib.error.URLError) as error:
            if attempt == HTTP_TRIES - 1:
                raise
            time.sleep(2**attempt)
            log(f"retrying range {start}+{length} after {type(error).__name__}: {error}")
    raise AssertionError("unreachable")


class RemoteFile(io.RawIOBase):
    """A seekable read-only view of the archive over HTTP range requests (for zipfile)."""

    def __init__(self, size: int) -> None:
        self.size, self.position = size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer) -> int:  # type: ignore[no-untyped-def]
        if self.position >= self.size:
            return 0
        length = min(len(buffer), self.size - self.position)
        data = http_range(self.position, length)
        buffer[:length] = data
        self.position += length
        return length


def archive_size() -> int:
    request = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise OSError("the server does not accept range requests; this notebook needs them")
        return int(response.headers["Content-Length"])


def member_bytes(entry: dict) -> bytes:
    """One archive member, fetched in a single range request and checked against its CRC."""
    slack = 30 + len(entry["name"].encode()) + 1024
    data = http_range(entry["offset"], slack + entry["compressed"])
    if data[:4] != b"PK\x03\x04":
        raise OSError(f"no local file header at {entry['offset']}")
    name_length, extra_length = struct.unpack("<HH", data[26:30])
    start = 30 + name_length + extra_length
    end = start + entry["compressed"]
    if end > len(data):
        data += http_range(entry["offset"] + len(data), end - len(data))
    payload = data[start:end]
    if entry["method"] == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(payload, -15)
    elif entry["method"] != zipfile.ZIP_STORED:
        raise OSError(f"unsupported compression method {entry['method']}")
    if len(payload) != entry["size"] or zlib.crc32(payload) != entry["crc"]:
        raise OSError("size or CRC mismatch after download")
    return payload


def points(landmarks, count: int):  # type: ignore[no-untyped-def]
    import numpy as np

    out = np.full((count, 3), np.nan, dtype=np.float32)
    if landmarks is not None:
        for index, point in enumerate(landmarks.landmark):
            out[index] = (point.x, point.y, point.z)
    return out


def track_video(path: str) -> dict:
    """Run Holistic over every frame; a new Holistic per video, so tracking never carries over."""
    import cv2
    import mediapipe as mp
    import numpy as np

    capture = cv2.VideoCapture(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    width = height = 0
    holistic = mp.solutions.holistic.Holistic(**HOLISTIC)
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            height, width = image.shape[:2]
            if width > MAX_WIDTH:
                image = cv2.resize(
                    image,
                    (MAX_WIDTH, round(height * MAX_WIDTH / width)),
                    interpolation=cv2.INTER_AREA,
                )
            result = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frame = np.full((N_POINTS, 3), np.nan, dtype=np.float32)
            if KEEP_FACE:
                frame[FACE:LEFT] = points(result.face_landmarks, LEFT - FACE)
            frame[LEFT:POSE] = points(result.left_hand_landmarks, POSE - LEFT)
            frame[POSE:RIGHT] = points(result.pose_landmarks, RIGHT - POSE)
            frame[RIGHT:N_POINTS] = points(result.right_hand_landmarks, N_POINTS - RIGHT)
            frames.append(frame)
    finally:
        holistic.close()
        capture.release()
    landmarks = np.stack(frames) if frames else np.zeros((0, N_POINTS, 3), dtype=np.float32)
    hands = np.isfinite(landmarks[:, LEFT, 0]) | np.isfinite(landmarks[:, RIGHT, 0])
    return {
        "landmarks": landmarks,
        "n_frames": len(frames),
        "fps": fps,
        "width": int(width),
        "height": int(height),
        "hand_frames": int(hands.sum()),
    }


def process(entry: dict) -> dict:
    """Download, track and delete one video. Errors become a status, never an exception."""
    started = time.perf_counter()
    record = {"file": entry["file"], "status": "ok", "error": ""}
    workdir = Path(tempfile.gettempdir()) / f"asl_citizen_{os.getpid()}"
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / Path(entry["file"]).name
    try:
        try:
            payload = member_bytes(entry)
        except Exception as error:
            record.update(status="download_error", error=f"{type(error).__name__}: {error}")
            return record
        path.write_bytes(payload)
        tracked = track_video(str(path))
        record.update(tracked)
        if tracked["n_frames"] == 0:
            record.update(status="decode_error", error="no frames decoded")
    except Exception as error:
        record.update(status="track_error", error=f"{type(error).__name__}: {error}")
    finally:
        path.unlink(missing_ok=True)
        record["seconds"] = time.perf_counter() - started
    return record


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def shard_done(directory: Path, index: int) -> bool:
    """Finished when both files exist and no video failed to download (those are retried)."""
    meta_path = directory / f"shard_{index:05d}.json"
    if not (meta_path.is_file() and (directory / f"shard_{index:05d}.npz").is_file()):
        return False
    try:
        videos = json.loads(meta_path.read_text())["videos"]
    except (OSError, ValueError, KeyError):
        return False
    return all(video["status"] != "download_error" for video in videos)


def write_shard(directory: Path, index: int, records: list[dict], plan_id: str) -> None:
    import numpy as np

    records = sorted(records, key=lambda item: item["file"])
    arrays = [r["landmarks"] for r in records if r["status"] == "ok"]
    landmarks = np.concatenate(arrays) if arrays else np.zeros((0, N_POINTS, 3), dtype=np.float32)
    starts, offset = [], 0
    for record in records:
        starts.append(offset if record["status"] == "ok" else -1)
        offset += record["n_frames"] if record["status"] == "ok" else 0
    buffer = io.BytesIO()
    np.savez_compressed(buffer, landmarks=landmarks.astype(np.float32))
    write_atomic(directory / f"shard_{index:05d}.npz", buffer.getvalue())
    videos = [
        {
            key: record.get(key)
            for key in (
                "file",
                "status",
                "error",
                "n_frames",
                "fps",
                "width",
                "height",
                "hand_frames",
                "seconds",
            )
        }
        | {"start": start}
        for record, start in zip(records, starts, strict=True)
    ]
    meta = {"plan_id": plan_id, "shard": index, "layout_version": LAYOUT_VERSION, "videos": videos}
    write_atomic(directory / f"shard_{index:05d}.json", json.dumps(meta, indent=1).encode())


def carry_over(shards: Path, plan_id: str) -> int:
    """Copy finished shards of the same plan from /kaggle/input (an earlier version's output)."""
    copied = 0
    if not INPUT_ROOT.is_dir():
        return 0
    for meta_path in sorted(INPUT_ROOT.rglob("shard_*.json")):
        target = shards / meta_path.name
        npz = meta_path.with_suffix(".npz")
        if target.exists() or not npz.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("plan_id") != plan_id:
            continue
        shutil.copyfile(npz, shards / npz.name)
        shutil.copyfile(meta_path, target)
        copied += 1
    return copied


def read_split_csv(archive: zipfile.ZipFile, split: str) -> list[dict]:
    text = archive.read(f"ASL_Citizen/splits/{split}.csv").decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    for row in rows:
        row["split"] = split
    return rows


def write_manifest(rows: list[dict], shards: Path, mapping: "MappingResult") -> dict:  # noqa: F821
    """manifest.csv over every planned video, from the shard files."""
    status: dict[str, dict] = {}
    for meta_path in sorted(shards.glob("shard_*.json")):
        meta = json.loads(meta_path.read_text())
        for video in meta["videos"]:
            status[video["file"]] = video | {"shard": meta["shard"]}
    columns = [
        "video_file", "participant", "split", "gloss", "asl_lex_code", "label", "rule", "shard",
        "start", "n_frames", "fps", "width", "height", "hand_frames", "status", "error",
    ]  # fmt: skip
    counts: dict[str, int] = {}
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            video = status.get(row["file"], {})
            mapped = mapping.by_gloss.get(row["gloss"])
            counts[video.get("status", "missing")] = (
                counts.get(video.get("status", "missing"), 0) + 1
            )
            writer.writerow(
                {
                    "video_file": row["file"],
                    "participant": row["participant"],
                    "split": row["split"],
                    "gloss": row["gloss"],
                    "asl_lex_code": row["code"],
                    "label": mapped.label if mapped else "",
                    "rule": mapped.rule if mapped else "review",
                    "shard": video.get("shard", ""),
                    "start": video.get("start", ""),
                    "n_frames": video.get("n_frames", ""),
                    "fps": video.get("fps", ""),
                    "width": video.get("width", ""),
                    "height": video.get("height", ""),
                    "hand_frames": video.get("hand_frames", ""),
                    "status": video.get("status", "missing"),
                    "error": video.get("error", ""),
                }
            )
    return counts


def presence(landmarks) -> dict:  # type: ignore[no-untyped-def]
    """Frames with a pose and with at least one hand (any finite x in the block): fractions of
    all frames, the fraction of the signing span (first to last hand frame) with a hand, and the
    number of hand frames."""
    import numpy as np

    if len(landmarks) == 0:
        return {"pose": 0.0, "hands": 0.0, "hands_in_span": 0.0, "hand_frames": 0}
    pose = np.isfinite(landmarks[:, POSE:RIGHT, 0]).any(axis=1)
    hands = np.isfinite(landmarks[:, LEFT:POSE, 0]).any(axis=1) | np.isfinite(
        landmarks[:, RIGHT:N_POINTS, 0]
    ).any(axis=1)
    found = np.flatnonzero(hands)
    in_span = float(hands[found[0] : found[-1] + 1].mean()) if len(found) else 0.0
    return {
        "pose": float(pose.mean()),
        "hands": float(hands.mean()),
        "hands_in_span": in_span,
        "hand_frames": len(found),
    }


def smoke_passes(row: dict) -> bool:
    return (
        row["pose"] > SMOKE_MOST_FRAMES
        and row["hands_in_span"] > SMOKE_MOST_FRAMES
        and row["hand_frames"] >= SMOKE_MIN_HAND_FRAMES
    )


def smoke_report(shards: Path, wall_seconds: float, total_planned: int) -> bool:
    """Per-video time and landmark presence from the written shards, and a pass/fail line."""
    import numpy as np

    rows = []
    for meta_path in sorted(shards.glob("shard_*.json")):
        meta = json.loads(meta_path.read_text())
        with np.load(meta_path.with_suffix(".npz"), allow_pickle=False) as data:
            landmarks = data["landmarks"]
        for video in meta["videos"]:
            found = presence(landmarks[:0])
            if video["status"] == "ok":
                start, count = int(video["start"]), int(video["n_frames"])
                found = presence(landmarks[start : start + count])
            rows.append(video | found)
    if not rows:
        log("SMOKE FAIL: no videos were processed")
        return False
    log("smoke test, per video: seconds (download + tracking, one worker), frames, fps, frames")
    log("with a pose, frames with a hand (of all frames, and of the signing span from the first")
    log("to the last hand frame); ok = passes the check")
    for number, row in enumerate(rows, start=1):
        log(
            f"  {number:2d}. {row['status']:<14} {row.get('seconds') or 0:6.1f} s "
            f"{row.get('n_frames') or 0:4d} frames {row.get('fps') or 0:5.1f} fps  "
            f"pose {100 * row['pose']:3.0f}%  hands {100 * row['hands']:3.0f}% of all, "
            f"{100 * row['hands_in_span']:3.0f}% of the span  "
            f"{'ok' if row['status'] == 'ok' and smoke_passes(row) else '--'}"
        )
    ok = [row for row in rows if row["status"] == "ok"]
    seconds = [row["seconds"] for row in rows if row.get("seconds")]
    wall_per_video = wall_seconds / len(rows)
    log(
        f"time: {np.mean(seconds):.1f} s per video per worker (median {np.median(seconds):.1f}); "
        f"wall {wall_per_video:.1f} s per video with {WORKERS} workers, so the full run of "
        f"{total_planned} videos needs about {total_planned * wall_per_video / 3600:.1f} h "
        f"({total_planned * wall_per_video / 3600 / 3:.1f} h per part with 3 parts)"
    )
    good = [row for row in ok if smoke_passes(row)]
    if ok:
        log(
            f"landmarks: {len(ok)}/{len(rows)} videos tracked; median pose "
            f"{100 * np.median([r['pose'] for r in ok]):.0f}% of frames, a hand in "
            f"{100 * np.median([r['hands'] for r in ok]):.0f}% of all frames and "
            f"{100 * np.median([r['hands_in_span'] for r in ok]):.0f}% of the signing span; "
            f"{len(good)}/{len(ok)} have a pose in more than {100 * SMOKE_MOST_FRAMES:.0f}% of "
            f"frames and a hand in more than {100 * SMOKE_MOST_FRAMES:.0f}% of the span"
        )
    tracked_enough = len(ok) >= SMOKE_PASS_FRACTION * len(rows)
    present_enough = bool(ok) and len(good) >= SMOKE_PASS_FRACTION * len(ok)
    if tracked_enough and present_enough:
        log("SMOKE PASS: downloads, tracking and shards work; run the full extraction")
        return True
    reasons = []
    if not tracked_enough:
        reasons.append(f"only {len(ok)} of {len(rows)} videos were tracked")
    if not present_enough:
        reasons.append(
            f"only {len(good)} of {len(ok)} tracked videos have a pose in most frames and a hand "
            "in most of the signing span"
        )
    log(f"SMOKE FAIL: {'; '.join(reasons)}. Read the errors above before the full run")
    return False


README = """# asl-citizen-landmarks-private

MediaPipe Holistic landmarks for the ASL Citizen videos whose glosses map onto the 250-sign
edge-vision vocabulary, plus the glosses kept for review. Made by kaggle/asl_citizen_merge.ipynb.

PRIVATE. ASL Citizen is licensed by Microsoft for non-commercial research only. These landmarks
are derived from it and must not be shared, published or redistributed; keep this dataset and
every notebook that uses it private, and delete them when the research is complete. The licence
text is in LICENSE_ASL_Citizen.txt.

- shards/shard_NNNNN.npz: "landmarks", float32 (frames, 543, 3) for the shard's videos one after
  another, in the asl-signs parquet layout (face 0-467, left_hand 468-488, pose 489-521,
  right_hand 522-542; x, y, z; NaN where nothing was found). Face points are NaN unless the run
  set EV_KEEP_FACE=1.
- shards/shard_NNNNN.json: per video, its file, status, n_frames, fps, frame size, hand frames
  and "start", the row of its first frame in "landmarks" (-1 when it has no frames).
- manifest.csv: every planned video with participant, official split, gloss, ASL-LEX code,
  mapped label and rule (exact, variant, synonym or review), shard and status.
- label_mapping.csv, unmapped_labels.txt, coverage.json: the label mapping (non-exact rules are
  AI-assisted and unconfirmed).
- plan.json: settings and the video list per shard.
"""


def dataset_metadata() -> dict:
    owner = os.environ.get("KAGGLE_USERNAME", "<your-kaggle-username>")
    return {
        "title": "asl-citizen-landmarks-private",
        "id": f"{owner}/asl-citizen-landmarks-private",
        "licenses": [{"name": "other"}],
    }


def main() -> int:
    started = time.time()
    log(f"python {platform.python_version()} on {platform.platform()}, {os.cpu_count()} CPUs")
    import cv2
    import mediapipe as mp
    import numpy as np

    log(f"mediapipe {mp.__version__}, opencv {cv2.__version__}, numpy {np.__version__}")
    if mp.__version__ != "0.10.21":
        log("WARNING: the training data was tracked with mediapipe 0.10.21")
    if SMOKE and OUT.name.endswith("-smoke") and OUT.exists():
        shutil.rmtree(OUT)  # a smoke folder is disposable: every smoke run starts empty
    OUT.mkdir(parents=True, exist_ok=True)
    shards = OUT / "shards"
    shards.mkdir(exist_ok=True)
    free = shutil.disk_usage(OUT).free / 1e9
    log(f"output {OUT} ({free:.1f} GB free), {WORKERS} workers, shards of {SHARD_SIZE} videos")

    if SMOKE:
        log(f"SMOKE TEST: the first {LIMIT_VIDEOS} planned videos into {OUT} (not the real shards)")
    if URL != DEFAULT_URL:
        log(f"EV_URL is set: reading {URL} instead of the Microsoft Download Center (local test)")
    size = archive_size()
    log(f"ASL_Citizen.zip: {size / 1e9:.2f} GB ({size:,} bytes); range requests accepted")
    archive = zipfile.ZipFile(io.BufferedReader(RemoteFile(size), buffer_size=1 << 20))
    entries = {info.filename: info for info in archive.infolist()}
    videos = [name for name in entries if name.startswith("ASL_Citizen/videos/")]
    log(f"archive: {len(entries)} entries, {len(videos)} videos")
    (OUT / "LICENSE_ASL_Citizen.txt").write_bytes(archive.read("ASL_Citizen/use.txt"))

    rows = []
    for split in SPLITS:
        for row in read_split_csv(archive, split):
            rows.append(
                {
                    "participant": row["Participant ID"].strip(),
                    "file": row["Video file"].strip(),
                    "gloss": row["Gloss"].strip(),
                    "code": row["ASL-LEX Code"].strip(),
                    "split": split,
                }
            )
    log(f"splits: {len(rows)} videos; " + ", ".join(
        f"{s} {sum(r['split'] == s for r in rows)}" for s in SPLITS))  # fmt: skip
    mapping = build_mapping(LABELS, [row["gloss"] for row in rows])  # noqa: F821
    (OUT / "label_mapping.csv").write_text(mapping_csv(mapping))  # noqa: F821
    (OUT / "unmapped_labels.txt").write_text(unmapped_text(mapping))  # noqa: F821
    coverage = mapping.coverage()
    (OUT / "coverage.json").write_text(json.dumps(coverage, indent=1))
    log(f"label mapping: {coverage}; unmapped: {', '.join(sorted(mapping.unmapped))}")

    wanted = set(mapping.candidate_glosses())
    planned = sorted((row for row in rows if row["gloss"] in wanted), key=lambda r: r["file"])
    missing = [row["file"] for row in planned if f"ASL_Citizen/videos/{row['file']}" not in entries]
    if missing:
        log(f"WARNING: {len(missing)} planned videos are not in the archive, e.g. {missing[:3]}")
    planned = [row for row in planned if f"ASL_Citizen/videos/{row['file']}" in entries]
    total_planned = len(planned)
    if SMOKE:
        planned = planned[:LIMIT_VIDEOS]
        log(f"smoke test: {len(planned)} of the {total_planned} planned videos")
    batches = [planned[i : i + SHARD_SIZE] for i in range(0, len(planned), SHARD_SIZE)]
    settings = {
        "holistic": HOLISTIC,
        "max_width": MAX_WIDTH,
        "keep_face": KEEP_FACE,
        "layout_version": LAYOUT_VERSION,
        "shard_size": SHARD_SIZE,
        "mediapipe": mp.__version__,
    }
    plan_id = hashlib.sha256(
        json.dumps([settings, [[r["file"] for r in b] for b in batches]]).encode()
    ).hexdigest()[:16]
    plan = {
        "plan_id": plan_id,
        "settings": settings,
        "shards": [[r["file"] for r in b] for b in batches],
    }
    (OUT / "plan.json").write_text(json.dumps(plan))
    stale = [
        p
        for p in shards.glob("shard_*.json")
        if json.loads(p.read_text()).get("plan_id") != plan_id
    ]
    if stale:
        log(f"ERROR: {len(stale)} shards in {shards} come from another plan; move them away first")
        return 1
    log(f"plan {plan_id}: {len(planned)} videos in {len(batches)} shards")
    copied = 0 if SMOKE else carry_over(shards, plan_id)
    if copied:
        log(f"carried over {copied} finished shards from {INPUT_ROOT}")
    part, parts = (int(value) for value in PART.split("/"))
    if not 0 <= part < parts:
        log(f"ERROR: EV_PART={PART} must be k/N with 0 <= k < N")
        return 1
    pending = [i for i in range(len(batches)) if not shard_done(shards, i)]
    log(f"{len(batches) - len(pending)} shards done, {len(pending)} to go")
    if parts > 1:
        pending = [i for i in pending if i % parts == part]
        log(f"part {part}/{parts}: {len(pending)} of those shards are this part's")

    done_videos, video_seconds = 0, 0.0
    session_start = time.time()
    with multiprocessing.get_context("spawn" if os.name == "nt" else "fork").Pool(WORKERS) as pool:
        for number, index in enumerate(pending, start=1):
            if (time.time() - started) / 3600 > MAX_HOURS:
                log(f"stopping after {MAX_HOURS} h; rerun with this output as an input to resume")
                break
            items = []
            for row in batches[index]:
                info = entries[f"ASL_Citizen/videos/{row['file']}"]
                items.append(
                    {
                        "file": row["file"],
                        "name": info.filename,
                        "offset": info.header_offset,
                        "compressed": info.compress_size,
                        "size": info.file_size,
                        "crc": info.CRC,
                        "method": info.compress_type,
                    }
                )
            shard_started = time.time()
            records = list(pool.imap_unordered(process, items, chunksize=1))
            write_shard(shards, index, records, plan_id)
            elapsed = time.time() - shard_started
            done_videos += len(records)
            video_seconds += sum(r.get("seconds", 0.0) for r in records)
            per_1000 = (time.time() - session_start) / done_videos * 1000
            remaining = sum(len(batches[i]) for i in pending[number:])
            bad = [r for r in records if r["status"] != "ok"]
            log(
                f"shard {index + 1}/{len(batches)}: {len(records)} videos in {elapsed:.0f} s "
                f"({len(bad)} not ok), {per_1000 / 60:.1f} min per 1,000 videos, "
                f"ETA {remaining * per_1000 / 1000 / 3600:.1f} h for {remaining} videos"
            )
            for record in bad[:3]:
                log(f"  {record['file']}: {record['status']} {record['error']}")

    finished = all(shard_done(shards, i) for i in range(len(batches)))
    counts = write_manifest(planned, shards, mapping)
    (OUT / "README.md").write_text(README)
    log(f"manifest: {counts}")
    if SMOKE:
        (OUT / "COMPLETE").unlink(missing_ok=True)
        (OUT / "SMOKE").write_text(
            f"smoke test of {len(planned)} videos; not training data, never upload as a dataset\n"
        )
        smoke_report(shards, time.time() - session_start, total_planned)
        return 0
    (OUT / "dataset-metadata.json").write_text(json.dumps(dataset_metadata(), indent=1))
    if finished:
        (OUT / "COMPLETE").write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
        log(f"COMPLETE: all {len(batches)} shards written to {OUT}")
    else:
        (OUT / "COMPLETE").unlink(missing_ok=True)
        own = [i for i in range(len(batches)) if i % parts == part]
        if parts > 1 and all(shard_done(shards, i) for i in own):
            log(
                f"part {part}/{parts} finished; to merge, run once more with EV_PART=0/1 and every "
                "part's output added as an input"
            )
        else:
            log("INCOMPLETE: save this version, add its output as an input and run again to resume")
    if done_videos:
        log(f"tracking: {video_seconds / done_videos:.2f} s per video per worker")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
