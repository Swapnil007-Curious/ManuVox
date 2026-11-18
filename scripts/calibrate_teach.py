"""Fit Teach Dex's blend on the validation split and write it into meta.json.

    python scripts/calibrate_teach.py --model-dir web/public/model --processed-dir data/processed_v2

The app scores a taught sign c as logits_c / T + weight * (s_c - offset), s_c the highest cosine
similarity between the segment's embedding and the user's examples of c
(web/src/personalization.ts).
weight and offset are fitted here, never on test data or the webcam clips, by simulating the
feature on val, whose signers the model never trained on:

- a draw picks one val signer and n ~ U{1..25} signs that signer made at least
  EXAMPLES_PER_SIGN + 1 times, and EXAMPLES_PER_SIGN of the signer's sequences of each as examples;
- every other sequence of that signer is a query, taught sign or not, so a blend that boosts taught
  signs wrongly is penalised;
- (weight, offset) minimise the mean negative log-likelihood over draws (a grid, then a finer grid
  around the best point).

The influence threshold (0.05) is a display choice for the "Personalized for you" badge, not fitted.
Writes meta.json's "personalization" block and results/teach_dex/calibration.json.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from edgevision.runtime import OrtModel
from edgevision.train.data import load_split_arrays
from edgevision.web_export import PersonalizationBlock, WebMeta

EXAMPLES_PER_SIGN = 3
MAX_TAUGHT = 25
DRAWS = 120
SEED = 0
INFLUENCE_THRESHOLD = 0.05
COARSE_WEIGHTS = np.arange(0.0, 20.01, 1.0)
COARSE_OFFSETS = np.arange(0.0, 1.001, 0.05)

FloatArray = npt.NDArray[np.float64]


def unit(rows: npt.NDArray[np.floating[Any]]) -> FloatArray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.asarray(rows, dtype=np.float64) / np.where(norms > 0, norms, 1.0)


def make_draws(
    y: npt.NDArray[np.int64], signer: npt.NDArray[np.int64], rng: np.random.Generator
) -> list[dict[str, Any]]:
    draws = []
    signers = np.unique(signer)
    for _ in range(DRAWS):
        person = rng.choice(signers)
        mine = np.flatnonzero(signer == person)
        signs, counts = np.unique(y[mine], return_counts=True)
        eligible = signs[counts > EXAMPLES_PER_SIGN]
        taught = rng.choice(
            eligible, size=int(rng.integers(1, min(MAX_TAUGHT, eligible.size) + 1)), replace=False
        )
        examples = {}
        for sign in taught:
            pool = mine[y[mine] == sign]
            examples[int(sign)] = rng.choice(pool, size=EXAMPLES_PER_SIGN, replace=False)
        used = np.concatenate(list(examples.values()))
        queries = np.setdiff1d(mine, used)
        draws.append(
            {
                "signer": int(person),
                "taught": [int(s) for s in taught],
                "examples": examples,
                "queries": queries,
                "y": y,
            }
        )
    return draws


def draw_terms(
    draw: dict[str, Any], scaled: FloatArray, embeddings: FloatArray
) -> dict[str, FloatArray]:
    """Per query: log of the untaught softmax mass (shifted), taught logits and similarities."""
    queries = draw["queries"]
    taught = np.array(draw["taught"])
    draw["labels"] = draw["y"][queries]
    z = scaled[queries]
    shift = z.max(axis=1, keepdims=True)
    mask = np.ones(z.shape[1], dtype=bool)
    mask[taught] = False
    untaught = np.exp(z[:, mask] - shift).sum(axis=1)
    similarity = np.stack(
        [
            (embeddings[queries] @ embeddings[draw["examples"][s]].T).max(axis=1)
            for s in draw["taught"]
        ],
        axis=1,
    )
    labels = draw["labels"]
    position = {sign: column for column, sign in enumerate(draw["taught"])}
    column = np.array([position.get(int(label), -1) for label in labels])
    target = z[np.arange(len(queries)), labels] - shift[:, 0]
    return {
        "untaught": untaught,
        "taught_z": z[:, taught] - shift,
        "similarity": similarity,
        "column": column,
        "target": target,
    }


def mean_nll(terms: list[dict[str, FloatArray]], weight: float, offset: float) -> float:
    """Mean over draws of the queries' mean negative log-likelihood under the blend."""
    total = []
    for term in terms:
        boosted = term["taught_z"] + weight * (term["similarity"] - offset)
        log_norm = np.log(term["untaught"] + np.exp(boosted).sum(axis=1))
        column = term["column"]
        rows = np.arange(column.size)
        target = np.where(column >= 0, boosted[rows, np.maximum(column, 0)], term["target"])
        total.append(float(np.mean(log_norm - target)))
    return float(np.mean(total))


def accuracy(
    draws: list[dict[str, Any]],
    terms: list[dict[str, FloatArray]],
    y: npt.NDArray[np.int64],
    scaled: FloatArray,
    weight: float,
    offset: float,
) -> dict[str, float]:
    hits = {"taught_base": [], "taught": [], "untaught_base": [], "untaught": []}
    for draw, term in zip(draws, terms, strict=True):
        z = scaled[draw["queries"]].copy()
        base = z.argmax(axis=1)
        z[:, draw["taught"]] += weight * (term["similarity"] - offset)
        mixed = z.argmax(axis=1)
        labels = y[draw["queries"]]
        taught = np.isin(labels, draw["taught"])
        hits["taught_base"] += list((base == labels)[taught])
        hits["taught"] += list((mixed == labels)[taught])
        hits["untaught_base"] += list((base == labels)[~taught])
        hits["untaught"] += list((mixed == labels)[~taught])
    return {key: float(np.mean(values)) for key, values in hits.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", type=Path, default=Path("web/public/model"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_v2"))
    parser.add_argument(
        "--splits", type=Path, help="default: <processed-dir>/splits_signer_independent.json"
    )
    parser.add_argument("--out", type=Path, default=Path("results/teach_dex/calibration.json"))
    args = parser.parse_args(argv)

    meta_path = args.model_dir / "meta.json"
    meta = json.loads(meta_path.read_text("utf-8"))
    if not meta.get("embedding"):
        raise SystemExit(
            f"{meta_path} has no embedding output; re-export with scripts/export_web.py"
        )
    splits = args.splits or args.processed_dir / "splits_signer_independent.json"
    val = load_split_arrays(args.processed_dir, splits, ("val",))["val"]
    model = OrtModel(args.model_dir / meta["model_file"])
    parts = [model.run_with_embedding(val.X[i : i + 512]) for i in range(0, len(val.X), 512)]
    logits = np.concatenate([p[0] for p in parts]).astype(np.float64)
    embeddings = unit(np.concatenate([p[1] for p in parts]))
    temperature = float(meta["calibration"]["temperature"])
    scaled = logits / temperature

    rng = np.random.default_rng(SEED)
    draws = make_draws(val.y, val.participant_id, rng)
    terms = [draw_terms(draw, scaled, embeddings) for draw in draws]

    def search(weights: FloatArray, offsets: FloatArray) -> tuple[float, float, float]:
        best = (np.inf, 0.0, 0.0)
        for weight in weights:
            for offset in offsets:
                value = mean_nll(terms, float(weight), float(offset))
                if value < best[0]:
                    best = (value, float(weight), float(offset))
        return best

    coarse = search(COARSE_WEIGHTS, COARSE_OFFSETS)
    fine = search(
        np.arange(max(0.0, coarse[1] - 1.0), coarse[1] + 1.001, 0.25),
        np.arange(coarse[2] - 0.05, coarse[2] + 0.0501, 0.01),
    )
    nll, weight, offset = fine
    before = mean_nll(terms, 0.0, 0.0)
    acc = accuracy(draws, terms, val.y, scaled, weight, offset)
    simulation = {
        "split": "val",
        "val_sequences": len(val.y),
        "signers": int(np.unique(val.participant_id).size),
        "draws": DRAWS,
        "examples_per_sign": EXAMPLES_PER_SIGN,
        "taught_signs_per_draw": f"uniform 1-{MAX_TAUGHT}",
        "queries": int(sum(len(d["queries"]) for d in draws)),
        "seed": SEED,
        "nll_before": before,
        "nll_after": nll,
        "top1_taught_before": acc["taught_base"],
        "top1_taught_after": acc["taught"],
        "top1_untaught_before": acc["untaught_base"],
        "top1_untaught_after": acc["untaught"],
        "grid": {
            "coarse": [coarse[1], coarse[2]],
            "weights": "0-20 step 1, then step 0.25",
            "offsets": "0-1 step 0.05, then step 0.01",
        },
    }
    block = PersonalizationBlock(
        computed_on="val",
        method=(
            "logits / T + weight * (max cosine similarity to the sign's examples - offset), "
            "taught signs only"
        ),
        weight=weight,
        offset=round(offset, 4),
        examples_per_sign=EXAMPLES_PER_SIGN,
        influence_threshold=INFLUENCE_THRESHOLD,
        simulation=simulation,
    )
    meta["personalization"] = block.model_dump(mode="json")
    WebMeta.model_validate(meta)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"model_sha256": meta["model_sha256"], **meta["personalization"]}, indent=2)
        + "\n",
        "utf-8",
    )
    print(
        f"weight {weight:.2f}, offset {offset:.3f}: val NLL {before:.4f} -> {nll:.4f}; "
        f"top-1 on taught signs {acc['taught_base']:.3f} -> {acc['taught']:.3f}, "
        f"on untaught signs {acc['untaught_base']:.3f} -> {acc['untaught']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
