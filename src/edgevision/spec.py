"""features_spec.json, the single source of truth for the landmark features.

The repository-root features_spec.json is read once, here, and by web/src/spec.ts in the
browser demo; modules take their landmark constants from this module rather than defining them.
EDGEVISION_FEATURES_SPEC may point at another copy (tests use it to check that a change in the
file reaches the code).
"""

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

SPEC_FILE = "features_spec.json"
SPEC_ENV = "EDGEVISION_FEATURES_SPEC"
SUPPORTED_VERSION = 1


def spec_path() -> Path:
    """The spec file: $EDGEVISION_FEATURES_SPEC, else the repository root next to src/."""
    override = os.environ.get(SPEC_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / SPEC_FILE


@cache
def _load(path: str) -> dict[str, Any]:
    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(
            f"{location} not found: edgevision reads its landmark constants from the repository's "
            f"{SPEC_FILE} (install with pip install -e . from a checkout, or set {SPEC_ENV})"
        )
    spec: dict[str, Any] = json.loads(location.read_text(encoding="utf-8"))
    if spec.get("spec_version") != SUPPORTED_VERSION:
        raise ValueError(f"{location}: spec_version {spec.get('spec_version')} is not supported")
    return spec


def load_spec() -> dict[str, Any]:
    """The parsed spec (cached per path)."""
    return _load(str(spec_path()))


SPEC = load_spec()

LANDMARK_TYPE_ORDER: tuple[str, ...] = tuple(SPEC["landmark_types"]["order"])
LANDMARK_COUNTS: dict[str, int] = {
    name: int(SPEC["landmark_types"]["counts"][name]) for name in LANDMARK_TYPE_ORDER
}
HAND_POINTS = int(SPEC["selection"]["hand_points"])
POSE_SETS: dict[str, tuple[int, ...]] = {
    name: tuple(int(index) for index in indices)
    for name, indices in SPEC["selection"]["pose_sets"].items()
}
DEFAULT_POSE_SET: str = SPEC["selection"]["default_pose_set"]
POSE_POINTS: dict[str, int] = {name: int(index) for name, index in SPEC["pose_points"].items()}
POSE_SWAP_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (int(first), int(second)) for first, second in SPEC["pose_swap_pairs"]
)
HAND_POINT_NAMES: dict[str, Any] = SPEC["hand_points"]
WRIST = int(HAND_POINT_NAMES["wrist"])
MIDDLE_MCP = int(HAND_POINT_NAMES["middle_mcp"])
FINGERTIPS: tuple[int, ...] = tuple(int(index) for index in HAND_POINT_NAMES["fingertips"])
USE_Z: bool = bool(SPEC["coordinates"]["use_z"])
X_SNAP_GRID = float(SPEC["coordinates"]["x_snap_grid"])
MIN_HAND_FRAMES = int(SPEC["min_hand_frames"])
MIN_SCALE = float(SPEC["normalization"]["min_scale"])
NUM_FRAMES = int(SPEC["resampling"]["num_frames"])
STORAGE_DTYPE: str = SPEC["storage"]["dtype"]
INPUT_NAME: str = SPEC["model_input"]["input_name"]
OUTPUT_NAME: str = SPEC["model_input"]["output_name"]
ASPECT: dict[str, Any] = SPEC["aspect"]
LIVE: dict[str, Any] = SPEC["live"]


def pose_set_name(pose_indices: Any) -> str | None:
    """The name of the spec pose set equal to ``pose_indices``, or None."""
    wanted = tuple(int(index) for index in pose_indices)
    return next((name for name, indices in POSE_SETS.items() if indices == wanted), None)
