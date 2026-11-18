"""The v3 webcam augmentations: aspect jitter, frame-rate jitter and dropped hand frames."""

import math

import torch

from edgevision.train.augment import aspect, augment_batch, frame_rate, hand_drop
from edgevision.train.config import AugmentConfig, TrainConfig, dump_train_config
from test_augment_v2 import NONE, V2_RECIPE, make_batch

WEBCAM = {
    "aspect": NONE.model_copy(update={"aspect_p": 1.0}),
    "fps": NONE.model_copy(update={"fps_p": 1.0}),
    "hand_drop": NONE.model_copy(update={"hand_drop_p": 1.0}),
}


def split(x, landmarks=59):
    batch = x.shape[0]
    points = x[..., : 2 * landmarks].reshape(batch, 32, landmarks, 2)
    return points, x[..., 2 * landmarks :], (points == 0.0).all(dim=-1)


def test_zero_probabilities_leave_v2_batches_unchanged():
    x = make_batch()
    with_fields = V2_RECIPE.model_copy(update={"aspect_max": 0.5, "fps_min_ratio": 0.2})
    first = augment_batch(x, V2_RECIPE, torch.Generator().manual_seed(5))
    second = augment_batch(x, with_fields, torch.Generator().manual_seed(5))
    assert torch.equal(first, second)


def test_aspect_scales_x_and_y_inversely_and_keeps_missing_points():
    points, _, missing = split(make_batch())
    out = aspect(points, missing, WEBCAM["aspect"], torch.Generator().manual_seed(0))
    assert torch.equal(out[missing], torch.zeros_like(out[missing]))
    present = ~missing
    ratio_x = out[..., 0][present] / points[..., 0][present]
    ratio_y = out[..., 1][present] / points[..., 1][present]
    finite = torch.isfinite(ratio_x) & torch.isfinite(ratio_y)
    product = (ratio_x * ratio_y)[finite]
    assert torch.allclose(product, torch.ones_like(product), atol=1e-4)
    limit = math.sqrt(1.15) + 1e-4
    assert ratio_x[finite].abs().max() <= limit and ratio_x[finite].abs().min() >= 1 / limit


def test_frame_rate_holds_kept_frames_and_interpolates_between_them():
    x = make_batch(batch=64)
    points, presence, missing = split(x)
    cfg = WEBCAM["fps"].model_copy(update={"fps_min_ratio": 0.4})
    out, out_presence, out_missing = frame_rate(
        points, presence, missing, cfg, torch.Generator().manual_seed(1)
    )
    assert out.shape == points.shape and out_presence.shape == presence.shape
    assert torch.equal(out[out_missing], torch.zeros_like(out[out_missing]))
    # The first and last frames are always kept.
    assert torch.equal(out[:, 0], points[:, 0]) and torch.equal(out[:, -1], points[:, -1])
    assert torch.equal(out_presence[:, 0], presence[:, 0])
    assert set(torch.unique(out_presence).tolist()) <= {0.0, 1.0}
    assert not torch.equal(out, points)


def test_frame_rate_at_full_ratio_is_the_identity():
    points, presence, missing = split(make_batch())
    cfg = WEBCAM["fps"].model_copy(update={"fps_min_ratio": 1.0})
    out, out_presence, out_missing = frame_rate(
        points, presence, missing, cfg, torch.Generator().manual_seed(2)
    )
    assert torch.equal(out, points) and torch.equal(out_presence, presence)
    assert torch.equal(out_missing, missing)


def test_hand_drop_blanks_whole_hands_and_their_presence_only():
    x = make_batch(batch=32)
    points, presence, missing = split(x)
    presence = torch.ones_like(presence)
    cfg = WEBCAM["hand_drop"].model_copy(update={"hand_drop_max": 0.6})
    out, out_presence, out_missing = hand_drop(
        points, presence, missing, cfg, torch.Generator().manual_seed(3)
    )
    dropped = out_presence == 0.0  # (B, T, 2)
    assert dropped.any()
    for hand, block in enumerate((slice(0, 21), slice(21, 42))):
        rows = dropped[..., hand]
        assert torch.equal(out[:, :, block][rows], torch.zeros_like(out[:, :, block][rows]))
        assert out_missing[:, :, block][rows].all()
        assert torch.equal(out[:, :, block][~rows], points[:, :, block][~rows])
    assert torch.equal(out[:, :, 42:], points[:, :, 42:])  # Pose untouched.


def test_webcam_transforms_are_deterministic_and_bounded():
    x = make_batch()
    cfg = V2_RECIPE.model_copy(update={"aspect_p": 0.5, "fps_p": 0.3, "hand_drop_p": 0.3})
    first = augment_batch(x, cfg, torch.Generator().manual_seed(9))
    second = augment_batch(x, cfg, torch.Generator().manual_seed(9))
    assert torch.equal(first, second) and torch.isfinite(first).all()
    for name, only in WEBCAM.items():
        out = augment_batch(x, only, torch.Generator().manual_seed(4))
        assert not torch.equal(out, x), name


def test_recipe_v1_rejects_webcam_augmentation_and_defaults_stay_out_of_dumps():
    base = {"model": {"name": "cnn"}, "processed_dir": "x", "lr": 1e-3}
    try:
        TrainConfig.model_validate({**base, "augment": {"hand_drop_p": 0.2}})
    except ValueError as error:
        assert "webcam" in str(error)
    else:
        raise AssertionError("recipe v1 accepted a webcam augmentation")
    dumped = dump_train_config(TrainConfig.model_validate(base))
    for name in ("aspect_p", "fps_p", "hand_drop_p", "val_selection"):
        assert name not in dumped
    assert AugmentConfig(aspect_p=0.1).enabled
