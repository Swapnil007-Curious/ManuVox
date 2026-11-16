"""Run legacy MediaPipe Holistic (the tracker asl-signs was built with) over .y4m clips.

For each clip in --clips (scripts/prepare_live_clips.py output, unmirrored 640 x 480 frames) this
writes <clip>.holistic.npz next to it: frame times t_ms (frame index / fps), the frame size, the
33 pose landmarks (x, y, z, visibility) and both 21-point hands (x, y, z), NaN where Holistic
found nothing. Holistic's left_hand is the signer's own left hand (the hand at pose landmark 15),
the layout training used, so no assignment step follows. Byte-identical clips are processed
once and the result is copied.

Needs mediapipe==0.10.21 (the last release with mediapipe.solutions); run it with the live tools
venv (requirements-live.txt):

    .venv-live/Scripts/python scripts/extract_holistic.py --clips data/live_clips
"""

import argparse
import hashlib
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def y4m_frames(path: Path) -> Iterator[tuple[float, int, int, np.ndarray]]:
    """(fps, width, height, RGB frame) for every frame of a 4:2:0 .y4m file."""
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").split()
        if not header or header[0] != "YUV4MPEG2":
            raise ValueError(f"{path} is not a YUV4MPEG2 file")
        fields = {token[0]: token[1:] for token in header[1:]}
        width, height = int(fields["W"]), int(fields["H"])
        numerator, denominator = (int(part) for part in fields.get("F", "30:1").split(":"))
        fps = numerator / denominator
        size = width * height * 3 // 2
        while handle.readline():
            buffer = handle.read(size)
            if len(buffer) < size:
                return
            yuv = np.frombuffer(buffer, np.uint8).reshape(height * 3 // 2, width)
            yield fps, width, height, cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)


def points(landmarks: object, count: int, with_visibility: bool) -> np.ndarray:
    columns = 4 if with_visibility else 3
    out = np.full((count, columns), np.nan)
    if landmarks is None:
        return out
    for index, point in enumerate(landmarks.landmark):  # type: ignore[attr-defined]
        out[index, :3] = (point.x, point.y, point.z)
        if with_visibility:
            out[index, 3] = point.visibility
    return out


def extract(path: Path, model_complexity: int) -> dict[str, np.ndarray]:
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        refine_face_landmarks=False,
    )
    times, poses, lefts, rights = [], [], [], []
    width = height = 0
    fps = 30.0
    for index, (fps, frame_width, frame_height, rgb) in enumerate(y4m_frames(path)):
        width, height = frame_width, frame_height
        result = holistic.process(rgb)
        times.append(index * 1000.0 / fps)
        poses.append(points(result.pose_landmarks, 33, True))
        lefts.append(points(result.left_hand_landmarks, 21, False))
        rights.append(points(result.right_hand_landmarks, 21, False))
    holistic.close()
    return {
        "t_ms": np.asarray(times),
        "pose": np.asarray(poses),
        "left_hand": np.asarray(lefts),
        "right_hand": np.asarray(rights),
        "width": np.asarray(width),
        "height": np.asarray(height),
        "fps": np.asarray(fps),
        "model_complexity": np.asarray(model_complexity),
        "mediapipe_version": np.asarray(mp.__version__),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--clips", type=Path, default=Path("data/live_clips"))
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    clips = sorted(args.clips.glob("*.y4m"))
    if not clips:
        print(f"no .y4m files in {args.clips}", file=sys.stderr)
        return 1
    done: dict[str, Path] = {}
    for clip in clips:
        target = clip.with_suffix(".holistic.npz")
        digest = hashlib.sha256(clip.read_bytes()).hexdigest()
        if target.exists() and not args.overwrite:
            done.setdefault(digest, target)
            print(f"{clip.name}: kept existing {target.name}", flush=True)
            continue
        if digest in done:
            shutil.copyfile(done[digest], target)
            print(f"{clip.name}: identical to {done[digest].stem}; copied", flush=True)
            continue
        result = extract(clip, args.model_complexity)
        np.savez_compressed(target, **result)
        done[digest] = target
        hands = np.isfinite(result["left_hand"][:, 0, 0]) | np.isfinite(
            result["right_hand"][:, 0, 0]
        )
        print(
            f"{clip.name}: {len(result['t_ms'])} frames, hand in {hands.mean():.2f}, "
            f"pose in {np.isfinite(result['pose'][:, 0, 0]).mean():.2f} -> {target.name}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
