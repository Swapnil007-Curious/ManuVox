# Per-sample probabilities. RECIPE_V2 is the repository's v2 recipe; RECIPE_V1 only the first three.
RECIPE_V1 = {"affine": 0.8, "temporal_crop": 0.5, "jitter": 0.5}
RECIPE_V2 = {**RECIPE_V1, "hand_rotation": 0.5, "frame_dropout": 0.5, "time_mask": 0.5, "landmark_mask": 0.3}


def _chosen(p, batch, gen, like):
    """Bernoulli(p) per sample, shape (B,)."""
    return torch.rand(batch, generator=gen, device=like.device) < p


def _uniform(shape, low, high, gen, like):
    return low + (high - low) * torch.rand(shape, generator=gen, device=like.device, dtype=like.dtype)


def _take_frames(values, index):
    """values[b, index[b, t]] for (B, T, ...) values and a (B, T) frame index."""
    shaped = index.reshape(*index.shape, *([1] * (values.dim() - 2)))
    return values.gather(1, shaped.expand(-1, -1, *values.shape[2:]))


def _blend(points, presence, missing, low, high, weight, nearest):
    """Linear blend of frames ``low`` and ``high``; a blend touching a missing point is missing."""
    start, end = _take_frames(points, low), _take_frames(points, high)
    gone = _take_frames(missing, low) | _take_frames(missing, high)
    blended = (start + (end - start) * weight[..., None, None]).masked_fill(gone[..., None], 0.0)
    return blended, _take_frames(presence, nearest), gone


def affine(points, missing, gen, rotate=15.0, scale=(0.85, 1.15), shift=0.1):
    """Rotate, scale and translate the whole skeleton."""
    b = points.shape[0]
    angle = _uniform((b,), -rotate, rotate, gen, points) * math.pi / 180
    size = _uniform((b,), *scale, gen, points)
    move = _uniform((b, 2), -shift, shift, gen, points)
    cos, sin = torch.cos(angle) * size, torch.sin(angle) * size
    matrix = torch.stack([torch.stack([cos, sin], -1), torch.stack([-sin, cos], -1)], -2)
    out = torch.einsum("btlc,bcd->btld", points, matrix) + move[:, None, None]
    return out.masked_fill(missing[..., None], 0.0)


def hand_rotation(points, missing, gen, degrees=10.0):
    """Rotate each hand about its own wrist by its own angle."""
    out = points.clone()
    angle = _uniform((points.shape[0], 2), -degrees, degrees, gen, points) * math.pi / 180
    for h, block in enumerate((LEFT, RIGHT)):
        cos, sin = torch.cos(angle[:, h])[:, None, None], torch.sin(angle[:, h])[:, None, None]
        hand = points[:, :, block]
        offset = hand - hand[:, :, :1]
        turned = hand[:, :, :1] + torch.stack(
            [offset[..., 0] * cos - offset[..., 1] * sin, offset[..., 0] * sin + offset[..., 1] * cos], -1
        )
        ok = ~missing[:, :, block] & ~missing[:, :, block][:, :, :1]
        out[:, :, block] = torch.where(ok[..., None], turned, hand)
    return out


def temporal_crop(points, presence, missing, gen, window_min=0.8):
    """Keep a random window of 80-100% of the sequence and stretch it back to 32 frames."""
    b, t = points.shape[:2]
    last = t - 1
    length = _uniform((b,), window_min, 1.0, gen, points) * last
    offset = _uniform((b,), 0.0, 1.0, gen, points) * (last - length)
    steps = torch.arange(t, device=points.device, dtype=points.dtype)
    position = (offset[:, None] + steps * (length / last)[:, None]).clamp(0, last)
    low = position.floor().long()
    weight = position - low
    high = torch.where(weight > 0, low + 1, low).clamp(max=last)
    return _blend(points, presence, missing, low, high, weight, (position + 0.5).floor().long().clamp(max=last))


def frame_dropout(points, presence, missing, gen, max_fraction=0.2):
    """Drop up to 20% of the frames at random and stretch the rest back to 32 frames."""
    b, t = points.shape[:2]
    n_drop = (_uniform((b,), 0.0, max_fraction, gen, points) * t).floor().long().clamp(max=t - 2)
    rank = torch.rand((b, t), generator=gen, device=points.device).argsort(1).argsort(1)
    steps = torch.arange(t, device=points.device)
    kept_order = torch.where(rank < n_drop[:, None], steps + t, steps).argsort(1)  # Kept frames first.
    last_kept = (t - 1 - n_drop)[:, None]
    position = steps[None].to(points.dtype) * (last_kept.to(points.dtype) / (t - 1))
    low = position.floor().long()
    weight = position - low
    high = torch.minimum(torch.where(weight > 0, low + 1, low), last_kept)
    nearest = torch.minimum((position + 0.5).floor().long(), last_kept)
    source = lambda index: kept_order.gather(1, index)  # noqa: E731
    return _blend(points, presence, missing, source(low), source(high), weight, source(nearest))


def time_mask(points, presence, missing, gen, windows=2, max_frames=4):
    """Blank one or two short windows of 1-4 frames."""
    b, t = points.shape[:2]
    dev = points.device
    count = torch.randint(1, windows + 1, (b,), generator=gen, device=dev)
    length = torch.randint(1, max_frames + 1, (b, windows), generator=gen, device=dev)
    start = (torch.rand((b, windows), generator=gen, device=dev) * (t - length + 1)).floor().long()
    steps = torch.arange(t, device=dev)[None, None]
    used = torch.arange(windows, device=dev)[None, :, None] < count[:, None, None]
    blank = ((steps >= start[..., None]) & (steps < (start + length)[..., None]) & used).any(1)
    return points.masked_fill(blank[..., None, None], 0.0), presence.masked_fill(blank[..., None], 0.0), missing | blank[..., None]


def landmark_mask(points, presence, missing, gen, max_points=4):
    """Hide 1-4 random landmarks (never a wrist) for the whole sequence."""
    b, _, n = points.shape[:3]
    dev = points.device
    count = torch.randint(1, max_points + 1, (b,), generator=gen, device=dev)
    scores = torch.rand((b, n), generator=gen, device=dev)
    scores[:, [LEFT.start, RIGHT.start]] = 2.0  # Wrists sort last, so they are never picked.
    hidden = scores.argsort(1).argsort(1) < count[:, None]
    return points.masked_fill(hidden[:, None, :, None], 0.0), presence, missing | hidden[:, None]


def jitter(points, missing, gen, std=0.01):
    """Add Gaussian noise to every visible point."""
    noise = torch.randn(points.shape, generator=gen, device=points.device, dtype=points.dtype)
    return (points + std * noise).masked_fill(missing[..., None], 0.0)


# Every transform maps (points, presence, missing, gen) to new (points, presence, missing).
TRANSFORMS = {
    "affine": lambda p, pr, m, g: (affine(p, m, g), pr, m),
    "hand_rotation": lambda p, pr, m, g: (hand_rotation(p, m, g), pr, m),
    "temporal_crop": temporal_crop,
    "frame_dropout": frame_dropout,
    "jitter": lambda p, pr, m, g: (jitter(p, m, g), pr, m),
    "time_mask": time_mask,
    "landmark_mask": landmark_mask,
}


def _where(chosen, new, old):
    """Per-sample choice between two tensors whose first axis is the batch."""
    return torch.where(chosen.reshape(-1, *([1] * (old.dim() - 1))), new, old)


def augment_batch(x, recipe, gen):
    """Apply the recipe to a (B, 32, 120) batch, in the order of TRANSFORMS; missing points stay 0."""
    b, t = x.shape[:2]
    points = x[..., :COORDS].reshape(b, t, N_LANDMARKS, 2)
    presence, missing = x[..., COORDS:], (points == 0).all(-1)
    for name, transform in TRANSFORMS.items():
        if name in recipe:
            chosen = _chosen(recipe[name], b, gen, x)
            new = transform(points, presence, missing, gen)
            points, presence, missing = (_where(chosen, n, o) for n, o in zip(new, (points, presence, missing)))
    return torch.cat([points.reshape(b, t, COORDS), presence], dim=-1)
