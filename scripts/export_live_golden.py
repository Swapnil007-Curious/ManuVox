"""Write web/tests/golden/live_parity.json, the fixtures of the Python / browser parity test.

Twenty val sequences -> raw landmarks (dataset rows and MediaPipe Tasks-style live frames) ->
the Python reference pipeline -> model inputs -> logits of the exported model; plus a stream of
four of them for motion energy and segmentation. See edgevision.live.parity.

    python scripts/export_live_golden.py --processed-dir data/processed_v2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from edgevision.live.parity import build_live_parity, write_live_parity
from edgevision.runtime import OrtModel

SEED = 2026


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("web/public/model"))
    parser.add_argument("--splits", type=Path, help="split file (default: the exported run's)")
    parser.add_argument("--out", type=Path, default=Path("web/tests/golden/live_parity.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    meta = json.loads((args.model_dir / "meta.json").read_text(encoding="utf-8"))
    splits = args.splits or Path("results/runs") / meta["run"] / "splits_signer_independent.json"
    model = OrtModel(args.model_dir / meta["model_file"], intra_op_threads=1)

    def classify(model_input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        logits: npt.NDArray[np.float32] = model.run(model_input[None].astype(np.float32))[0]
        return logits

    golden = build_live_parity(args.processed_dir, splits, meta, classify, args.seed)
    write_live_parity(args.out, golden)
    agree = [int(np.argmax(case["logits"])) == case["label"] for case in golden["live_cases"]]
    segments = golden["stream"]["segments"]
    print(
        f"wrote {args.out}: {len(golden['dataset_cases'])} dataset and {len(golden['live_cases'])} "
        f"live cases (live top-1 = label for {sum(agree)}), stream of "
        f"{len(golden['stream']['frames'])} frames with {len(segments)} segments "
        f"({sum(s['kind'] == 'segment' for s in segments)} classified)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
