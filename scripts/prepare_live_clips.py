"""Turn the screen-recorded test clips into fake-camera input for both live pipelines.

The clips in test_clips/ are screen recordings of the web demo, not raw camera files: the camera
preview is a CSS-mirrored canvas inside the page, with the landmark overlay drawn on it. This
script finds the preview rectangle in each clip (the largest bright 4:3 region of the page,
measured on frames sampled across the clip), then has ffmpeg crop it, flip it back to the
unmirrored camera orientation, scale it to 640 x 480 and write it as YUV4MPEG2 (.y4m), the format
Chromium's --use-file-for-fake-video-capture plays. The Python Holistic pipeline reads the same
.y4m files (edgevision.live.y4m), so both pipelines see identical pixels.

The recorded preview carries the demo's landmark overlay (hand skeletons, pose lines, wrist
dots) wherever the recording app detected something. A palm detector cannot find a hand that
has a skeleton painted over it, so the overlay is removed first: pixels within a narrow HSV
band of the three overlay colours of web/src/overlay.ts are masked (dilated by 2 px for
anti-aliasing and compression halos) and inpainted (OpenCV Telea, radius 4) on the source-
resolution crop. The masked fraction is recorded per clip in clips.json; --keep-overlay skips
the step. Byte-identical input clips are reported, since a duplicated file cannot carry two
different sign labels.

Needs opencv-python and imageio-ffmpeg (requirements-live.txt); run with the live tools venv:

    .venv-live/Scripts/python scripts/prepare_live_clips.py --clips test_clips --out data/live_clips
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

OUT_WIDTH, OUT_HEIGHT = 640, 480
SAMPLES = 24
BRIGHT = 40
TARGET_ASPECT = 4 / 3
# web/src/overlay.ts COLORS (RGB): left hand cyan, right hand orange, pose yellow.
OVERLAY_RGB = ((0, 229, 255), (255, 158, 27), (255, 221, 0))
HUE_TOLERANCE = 8  # OpenCV hue units (0-179).
MIN_SATURATION = 150
MIN_VALUE = 150
INPAINT_RADIUS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sampled_mean(path: Path) -> tuple[np.ndarray, float, int]:
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    for index in np.linspace(0, max(count - 1, 0), SAMPLES).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    capture.release()
    if not frames:
        raise RuntimeError(f"{path}: no frames decoded")
    return np.mean(frames, axis=0), fps, count


def preview_rect(mean: np.ndarray) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the largest bright connected region, the camera preview."""
    mask = (mean > BRIGHT).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise RuntimeError("no preview region found")
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[best, :4])
    # Inset by 2 px to drop the canvas border, then trim to exactly 4:3 about the centre.
    x, y, w, h = x + 2, y + 2, w - 4, h - 4
    if w / h > TARGET_ASPECT:
        trimmed = round(h * TARGET_ASPECT)
        x, w = x + (w - trimmed) // 2, trimmed
    else:
        trimmed = round(w / TARGET_ASPECT)
        y, h = y + (h - trimmed) // 2, trimmed
    return x, y, w, h


def overlay_mask(bgr: np.ndarray) -> np.ndarray:
    """uint8 mask (1 = overlay pixel) of the three overlay colours, dilated by 2 px."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = (saturation >= MIN_SATURATION) & (value >= MIN_VALUE)
    mask = np.zeros(bgr.shape[:2], dtype=bool)
    for rgb in OVERLAY_RGB:
        reference = int(cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
        distance = np.abs(hue - reference)
        mask |= vivid & (np.minimum(distance, 180 - distance) <= HUE_TOLERANCE)
    return cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))


def convert(source: Path, target: Path, rect: tuple[int, int, int, int], inpaint: bool) -> float:
    """Write the unmirrored 640 x 480 .y4m; returns the mean masked fraction of the frames."""
    x, y, w, h = rect
    fps = cv2.VideoCapture(str(source)).get(cv2.CAP_PROP_FPS)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{OUT_WIDTH}x{OUT_HEIGHT}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "yuv4mpegpipe",
        str(target),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    capture = cv2.VideoCapture(str(source))
    masked: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        crop = frame[y : y + h, x : x + w]
        if inpaint:
            mask = overlay_mask(crop)
            masked.append(float(mask.mean()))
            crop = cv2.inpaint(crop, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        crop = cv2.flip(crop, 1)  # Undo the preview's CSS mirror.
        resized = cv2.resize(crop, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
        encoder.stdin.write(np.ascontiguousarray(resized).tobytes())
    capture.release()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {source}")
    return float(np.mean(masked)) if masked else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--clips", type=Path, default=Path("test_clips"))
    parser.add_argument("--out", type=Path, default=Path("data/live_clips"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-overlay", action="store_true", help="do not inpaint the recorded landmark overlay"
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    clips = sorted(args.clips.glob("*.mp4"))
    if not clips:
        print(f"no .mp4 files in {args.clips}", file=sys.stderr)
        return 1
    hashes: dict[str, list[str]] = {}
    entries = []
    for clip in clips:
        digest = sha256(clip)
        hashes.setdefault(digest, []).append(clip.stem)
        target = args.out / f"{clip.stem}.y4m"
        mean, fps, count = sampled_mean(clip)
        rect = preview_rect(mean)
        masked = None
        if args.overwrite or not target.exists():
            masked = convert(clip, target, rect, inpaint=not args.keep_overlay)
        entries.append(
            {
                "clip": clip.name,
                "label": clip.stem,
                "sha256": digest,
                "source_fps": fps,
                "source_frames": count,
                "source_size": [int(mean.shape[1]), int(mean.shape[0])],
                "preview_rect_xywh": list(rect),
                "y4m": target.name,
                "y4m_size": [OUT_WIDTH, OUT_HEIGHT],
                "overlay_inpainted": not args.keep_overlay,
                "overlay_masked_fraction": masked,
            }
        )
        print(f"{clip.name}: preview {rect} at {fps:.0f} fps, {count} frames -> {target}")
    duplicates = [names for names in hashes.values() if len(names) > 1]
    for entry in entries:
        entry["duplicate_of"] = next(
            (
                sorted(set(names) - {entry["label"]})
                for names in duplicates
                if entry["label"] in names
            ),
            [],
        )
    manifest = {
        "description": (
            "test_clips converted for the live-gap evaluation (scripts/prepare_live_clips.py)"
        ),
        "clips": entries,
        "duplicate_groups": duplicates,
    }
    (args.out / "clips.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for names in duplicates:
        print(f"WARNING: byte-identical clips {names}: one file cannot hold two different signs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
