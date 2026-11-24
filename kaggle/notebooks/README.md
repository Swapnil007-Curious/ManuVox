# Kaggle notebook series

Public-facing Kaggle notebooks built from this repository's work. Each series lives in its own
folder; every notebook folder holds `notebook.ipynb`, `kernel-metadata.json` (private by default)
and `cover.png` (1280x720).

## ManuVox (`manuvox/`)

Five notebooks on the Google Isolated Sign Language Recognition data (Kaggle competition
`asl-signs`, CC BY 4.0). They use that competition's files and nothing else: no other dataset,
no model trained on other data, no API keys.

| # | Folder | Kaggle title | Accelerator |
|---|---|---|---|
| 1 | `01-asl-signs-eda` | ASL Signs EDA: animating 94k landmark sequences | CPU |
| 2 | `02-hand-swap-bug` | The hand-swap bug: train-serve skew in ASL models | GPU T4 |
| 3 | `03-random-vs-grouped-splits` | Why random splits overstate sign accuracy | GPU T4 |
| 4 | `04-cnn-transformer-hybrid` | CNN + Transformer hybrid for isolated ASL signs | GPU T4 |
| 5 | `05-onnx-quantization-calibration` | PyTorch to browser: ONNX, FP16, INT8, calibration | GPU T4, internet |

The notebooks are self-contained: the preprocessing, features, models and training loop are
ports of `src/edgevision/` written as small notebook functions, checked against the package
(identical feature block and logits for the hybrid, preprocessing equal to float32 rounding).

### Layout

- `manuvox/_src/*.nbsrc`: the notebook sources, cells separated by `%% md`, `%% code` and
  `%% include NAME` lines.
- `manuvox/_src/common/*.py`: code shared by several notebooks (setup, data reading,
  preprocessing, features, models, augmentation, training).
- `manuvox/series.json`: titles, folders, accelerator and internet flags, links, Kaggle username.
- `tools/build_notebooks.py`: writes `notebook.ipynb` and `kernel-metadata.json` from the
  sources, keeping outputs of code cells that did not change.
- `tools/run_local.py`: executes notebooks locally with Kaggle-style paths.
- `tools/check_notebooks.py`: leak, licence and style checks.
- `tools/download_asl_signs_sample.py`: downloads `train.csv`, the sign map and a seeded
  per-sign sample of the sequence files, for local runs. Kaggle rate-limits its per-file
  endpoint (HTTP 429 after a few hundred files), so the script makes one API call for the
  competition archive's signed URL and reads only the needed zip members with range requests,
  checking each against its CRC-32.

### Build, run and check locally

The notebooks target Kaggle's image (Python 3.13, numpy 2.1.3, pandas 2.3.3, pyarrow 23.0.1,
torch 2.11, onnx 1.23.2; notebook 5 installs onnxruntime 1.31.1). A matching local environment:

```bash
uv venv --python 3.13 .nbenv
uv pip install --python .nbenv torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .nbenv numpy==2.1.3 pandas==2.3.3 pyarrow==23.0.1 matplotlib==3.10.7 pillow \
    onnx==1.23.2 onnxruntime==1.31.1 nbclient nbformat ipykernel
```

Then, with a Kaggle API token and the competition rules accepted:

```bash
python kaggle/notebooks/tools/download_asl_signs_sample.py --per-sign 24      # into data/kaggle_input/
python kaggle/notebooks/tools/build_notebooks.py manuvox --username <kaggle-username>
.nbenv/Scripts/python kaggle/notebooks/tools/run_local.py manuvox --input data/kaggle_input --fast
python kaggle/notebooks/tools/check_notebooks.py manuvox
```

`--fast` sets `MANUVOX_FAST=1`: each notebook keeps up to 24 sequences per sign (about 6,000)
and trains for a few epochs, so a CPU laptop finishes in minutes. Without it the notebooks expect
the full competition data, as on Kaggle. The runner records the mode and wall time in each
notebook's metadata (`manuvox_run`). Downloaded data and run folders stay under `data/`, which
git ignores.

The notebooks committed here carry the outputs of such a FAST run on a 4-core laptop CPU
(6,000-sequence sample, 2 to 4 epochs). They show that every cell runs on the real competition
files, but their accuracies are far below a full run: 2 to 4 CPU epochs on a few thousand
sequences give 2% to 12% test top-1. Short CPU runs also differ by a point or two between reruns
with the same seed, because multi-threaded CPU kernels are not bit-for-bit deterministic.

### Publish

`kaggle kernels push` uploads the notebook and Kaggle runs it from the top, so the published
version shows the outputs of Kaggle's full run in place of the local FAST outputs. From the
repository root:

```bash
kaggle kernels push -p kaggle/notebooks/manuvox/01-asl-signs-eda
kaggle kernels status <kaggle-username>/asl-signs-eda-animating-94k-landmark-sequences
```

Each push creates a private notebook (`is_private: true`). Set the cover image and make it public
in the notebook's settings on kaggle.com once the run has finished and the outputs look right.
