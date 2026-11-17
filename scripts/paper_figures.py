"""Draw the paper's figures (vector PDF) from the same result files as paper_tables.py.

    python scripts/paper_figures.py     # paper/figures/*.pdf

Colours: categorical slots 1-3 of the validated default palette (blue, orange, aqua; checked with
the dataviz skill's validate_palette.js for a light surface), with marker shape or hatching as a
second encoding so the figures also read in greyscale. The skeleton figure draws landmark
coordinates only (no pixels): one frame of a live clip's MediaPipe Holistic landmarks, snapshotted
to paper/data/skeleton_example.json so it rebuilds without the clips.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
DATA = ROOT / "paper" / "data"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COLUMN = 3.15  # inches, one ACL column

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: Any, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def accuracy_latency() -> None:
    rows = read_csv(ROOT / "results" / "model_selection.csv")
    labels = {
        "gru_v1": "GRU",
        "transformer_v1": "Transformer",
        "cnn_v1": "CNN",
        "cnn_v2feat_v1recipe": "CNN, v2 feat.",
        "cnn_v2feat_v2recipe": "CNN, v2 feat.+recipe",
        "hybrid_v2_fp16": "Hybrid (shipped, fp16)",
        "hybrid_v2_ensemble_fp16": "Hybrid ensemble (fp16, over size cap)",
    }
    # Label offsets (points) chosen so the clustered small models do not collide.
    offsets = {
        "gru_v1": (5, -3, "left"),
        "transformer_v1": (5, -5, "left"),
        "cnn_v1": (-4, -7, "left"),
        "cnn_v2feat_v1recipe": (5, 4, "left"),
        "cnn_v2feat_v2recipe": (5, 3, "left"),
        "hybrid_v2_fp16": (5, 0, "left"),
        "hybrid_v2_ensemble_fp16": (-5, 0, "right"),
    }
    fig, ax = plt.subplots(figsize=(COLUMN, 2.0))
    ax.axvspan(60, 70, color=GRID, alpha=0.6, lw=0)
    ax.text(65, 64.5, "over the\nbudget", fontsize=5.8, color=MUTED, ha="center", va="center")
    for r in rows:
        if r["candidate"] not in labels:
            continue
        x, y = float(r["latency_p95_ms"]), float(r["test_top1"]) * 100
        sd = float(r["test_top1_sd"]) * 100 if r["test_top1_sd"] else 0.0
        v2 = r["features"] == "v2"
        over_cap = float(r["onnx_mb"]) >= 10  # the shipping rule's size cap; hollow marker
        color = ORANGE if v2 else BLUE
        ax.errorbar(
            x,
            y,
            yerr=sd,
            fmt="s" if v2 else "o",
            color=color,
            mfc="white" if over_cap else color,
            ms=4.5,
            mec=color if over_cap else "white",
            mew=0.9 if over_cap else 0.6,
            elinewidth=0.8,
            capsize=1.5,
        )
        dx, dy, ha = offsets[r["candidate"]]
        ax.annotate(
            labels[r["candidate"]],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.8,
            color=INK,
            ha=ha,
            va="center",
        )
    ax.set_xlim(0, 70)
    ax.set_ylim(60, 72)
    ax.set_xlabel("Browser latency per segment, p95 (ms; wasm, 1 thread)")
    ax.set_ylabel("Test top-1 (%)")
    ax.plot([], [], "o", color=BLUE, ms=4, label="v1 features")
    ax.plot([], [], "s", color=ORANGE, ms=4, label="v2 features")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.2)
    save(fig, "acc_latency")


def quantization() -> None:
    quant = read_csv(ROOT / "results" / "tables" / "test_quant_summary.csv")
    bench = read_csv(ROOT / "results" / "tables" / "bench_summary.csv")
    variants = ["fp32", "fp16", "int8_dynamic", "int8_static", "int8_static_full"]
    short = {
        "fp32": "fp32",
        "fp16": "fp16",
        "int8_dynamic": "int8 dyn.",
        "int8_static": "int8 st.",
        "int8_static_full": "int8 st. (input q.)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN * 2, 1.75))
    for model, color, marker in (
        ("cnn", BLUE, "o"),
        ("gru", ORANGE, "s"),
        ("transformer", AQUA, "^"),
    ):
        xs, ys, lat = [], [], []
        for v in variants:
            q = next(r for r in quant if r["model"] == model and r["variant"] == v)
            b = next(
                r
                for r in bench
                if r["experiment"] == f"{model}_signer_independent"
                and r["variant"] == ("onnx_fp32" if v == "fp32" else v)
                and r["threads"] == "1"
            )
            xs.append(float(q["size_mb"]))
            ys.append(float(q["top1_mean"]) * 100)
            lat.append(float(b["p50_ms"]))
        label = {"cnn": "CNN", "gru": "GRU", "transformer": "Transformer"}[model]
        axes[0].plot(
            xs, ys, marker=marker, color=color, ms=4, lw=0.9, mec="white", mew=0.5, label=label
        )
        axes[1].plot(
            range(len(variants)),
            lat,
            marker=marker,
            color=color,
            ms=4,
            lw=0.9,
            mec="white",
            mew=0.5,
            label=label,
        )
    axes[0].set_xscale("log")
    ticks = [0.4, 0.7, 1, 2, 3, 5]
    axes[0].set_xticks(ticks, [f"{t:g}" for t in ticks])
    axes[0].minorticks_off()
    axes[0].set_xlabel("ONNX file size (MB, log scale)")
    axes[0].set_ylabel("Test top-1 (%)")
    axes[0].legend(frameon=False, loc="lower left")
    axes[1].set_xticks(range(len(variants)), [short[v] for v in variants])
    axes[1].set_ylabel("Python ORT p50 (ms, 1 thread)")
    axes[1].set_ylim(bottom=0)
    save(fig, "quant")


def live_gap() -> None:
    s = json.loads((ROOT / "results" / "live_gap_v2" / "summary.json").read_text(encoding="utf-8"))
    rows = {(r["pipeline"], r["config"]): r for r in s["rows"]}
    steps = [
        "c0_as_shipped",
        "c1_model",
        "c2_hands",
        "c3_aspect",
        "c4_time",
        "c5_segmenter",
        "c6_final",
        "x_presence",
    ]
    names = [
        "as shipped",
        "+model",
        "+hands (a)",
        "+aspect (c)",
        "+time (d)",
        "+segm. (e)",
        "+calib. (f)",
        "final, presence segm.",
    ]
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    width = 0.38
    for offset, (pipeline, color, hatch, label) in enumerate(
        (("python", BLUE, None, "Python + Holistic"), ("web", ORANGE, "////", "Browser + Tasks"))
    ):
        xs, ys, lo, hi = [], [], [], []
        for i, step in enumerate(steps):
            r = rows.get((pipeline, step))
            if r is None:
                continue
            xs.append(i + (offset - 0.5) * width)
            ys.append(r["clip_top1"] * 100)
            lo.append((r["clip_top1"] - r["clip_top1_ci_low"]) * 100)
            hi.append((r["clip_top1_ci_high"] - r["clip_top1"]) * 100)
        ax.bar(
            xs,
            ys,
            width=width - 0.04,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            hatch=hatch,
            label=label,
        )
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2)
    # The hybrid's test-split top-1, for scale (a different metric: pre-segmented sequences).
    test = read_csv(ROOT / "results" / "tables" / "selection" / "test_fp32_summary.csv")
    hybrid = next(r for r in test if r["experiment"] == "d_hybrid_v2feat_v2recipe")
    ax.axhline(float(hybrid["test_top1_mean"]) * 100, color=MUTED, lw=0.8, ls="--")
    ax.text(
        len(steps) - 0.5,
        float(hybrid["test_top1_mean"]) * 100 + 2,
        "hybrid, test split",
        fontsize=5.8,
        color=MUTED,
        ha="right",
        va="bottom",
    )
    ax.axvline(len(steps) - 1.5, color=GRID, lw=0.8)
    ax.set_xticks(range(len(steps)), names, rotation=30, ha="right")
    ax.set_ylabel("Clip top-1 (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper left", ncols=2)
    ax.grid(axis="x", visible=False)
    save(fig, "livegap")


def calibration() -> None:
    bins = read_csv(ROOT / "results" / "test_analysis" / "reliability.csv")
    curve = read_csv(ROOT / "results" / "test_analysis" / "coverage_curve.csv")
    cal = json.loads(
        (ROOT / "results" / "test_analysis" / "calibration.json").read_text(encoding="utf-8")
    )
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.6))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls="--")
    for kind, color, marker, label in (
        ("raw", BLUE, "o", "raw"),
        ("temperature_scaled", ORANGE, "s", "scaled"),
    ):
        pts = [
            (float(b["mean_confidence"]), float(b["accuracy"]))
            for b in bins
            if b["probabilities"] == kind and int(b["count"]) >= 50
        ]
        ax.plot(
            *zip(*pts, strict=True),
            marker=marker,
            color=color,
            ms=2.8,
            lw=0.9,
            mec="white",
            mew=0.4,
            label=label,
        )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    ax = axes[1]
    ax.plot(
        [float(c["coverage"]) for c in curve],
        [float(c["selective_accuracy"]) for c in curve],
        color=BLUE,
        lw=1.1,
    )
    cov, sel = cal["test_coverage_at_threshold"], cal["test_selective_accuracy_at_threshold"]
    ax.plot([cov], [sel], "o", color=ORANGE, ms=4, mec="white", mew=0.5)
    ax.annotate(
        "val threshold",
        (cov, sel),
        xytext=(-4, -9),
        textcoords="offset points",
        fontsize=5.8,
        ha="right",
        color=INK,
    )
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.6, 1.0)
    fig.tight_layout(w_pad=0.8)
    save(fig, "calibration")


HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
POSE_EDGES = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (9, 10),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6),
]


def skeleton_snapshot() -> dict[str, Any]:
    target = DATA / "skeleton_example.json"
    source = ROOT / "data" / "live_clips_v2" / "hello_1.holistic.npz"
    if source.exists():
        data = np.load(source)
        hands = [np.isfinite(data[k][:, 0, 0]) for k in ("right_hand", "left_hand")]
        present = np.flatnonzero(hands[0] | hands[1])
        frame = int(present[len(present) // 2])

        def pts(array: np.ndarray) -> list[list[float]] | None:
            xy = array[frame, :, :2]
            return (
                None
                if not np.isfinite(xy).all()
                else [[round(float(x), 4), round(float(y), 4)] for x, y in xy]
            )

        snap = {
            "clip": "test_clips_v2/hello_1 (MediaPipe Holistic landmarks; coordinates only, no pixels)",
            "frame": frame,
            "width": int(data["width"]),
            "height": int(data["height"]),
            "pose": pts(data["pose"]),
            "left_hand": pts(data["left_hand"]),
            "right_hand": pts(data["right_hand"]),
        }
        target.write_text(json.dumps(snap) + "\n", encoding="utf-8")
    return dict(json.loads(target.read_text(encoding="utf-8")))


def skeleton() -> None:
    snap = skeleton_snapshot()
    aspect = snap["height"] / snap["width"]
    fig, ax = plt.subplots(figsize=(1.45, 1.45 * aspect * 1.15))

    def draw(
        points: list[list[float]] | None, edges: list[tuple[int, int]], color: str, size: float
    ) -> None:
        if points is None:
            return
        p = np.asarray(points)
        for a, b in edges:
            ax.plot(p[[a, b], 0], p[[a, b], 1], color=color, lw=0.9, solid_capstyle="round")
        ax.plot(p[:, 0], p[:, 1], "o", color=color, ms=size, mec="white", mew=0.3)

    pose = snap["pose"]
    if pose is not None:
        p = np.asarray(pose)
        draw([list(x) for x in p], POSE_EDGES, MUTED, 1.6)
    draw(snap["left_hand"], HAND_EDGES, BLUE, 1.4)
    draw(snap["right_hand"], HAND_EDGES, ORANGE, 1.4)
    # The camera frame's outline, so the landmarks have their image context (no pixels).
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=GRID, lw=0.8)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(1.01, -0.01)
    ax.set_aspect(aspect)
    ax.axis("off")
    save(fig, "skeleton")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    accuracy_latency()
    quantization()
    live_gap()
    calibration()
    skeleton()
    print(f"figures -> {FIG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
