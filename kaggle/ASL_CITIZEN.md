# ASL Citizen: extraction, cross-dataset evaluation and v3 training

Three steps, each run once, after the code archive (step 0). Read the licence rules first (docs/DATA_LICENSES.md): non-commercial
research only, never redistribute the data or the landmarks, keep every notebook and dataset
**Private**, delete everything when the research is complete.

## 0. The code: the private dataset edge-vision-src

Every notebook takes the repository code only from a private Kaggle dataset, never from a git
clone. Each cell finds the archive under its inputs, prints its commit and checks that it holds
the sources the notebook was generated from. If the archive is missing it stops with
`CODE SOURCE MISSING`; if it is from another commit it stops with `CODE SOURCE MISMATCH`.

1. Locally, with everything committed:
   `python scripts/build_kaggle_notebooks.py --source-archive artifacts/v3/upload/edge-vision-src.tar.gz`.
   It refuses a dirty tree and any data, video, landmark, local model or credential path, and
   writes `SOURCE_COMMIT` into the archive.
2. kaggle.com > Datasets > New Dataset: upload that file, title it `edge-vision-src`, leave
   Visibility on **Private**, Create. After a new commit, upload the new archive with New
   Version and import the notebooks again from the same commit.
3. Add it to every notebook below: Add Input > Your Work > Datasets > edge-vision-src.

Import each notebook with kaggle.com > Create > Notebook > File > Import Notebook. A new notebook
is Private; keep it that way.

## 1. Extract landmarks (kaggle/asl_citizen_*.ipynb)

The notebooks download only the videos they need. They read the archive's file index over HTTP
range requests, fetch each video of a mapped sign on its own, run MediaPipe Holistic 0.10.21
with the training settings (`scripts/extract_holistic.py`), and delete the video. The 46 GB
archive is never stored, so Kaggle's disk limit is not an issue.

- **What they process:** 9,571 videos. That is the 8,866 videos of the 290 glosses mapped onto
  236 of the 250 signs, plus 705 videos of 23 glosses kept for review, so a later mapping
  decision needs no second extraction.
- **What they write:** shards of 200 videos in the asl-signs parquet layout, a manifest with
  video, signer, official split, gloss, label and mapping rule, `label_mapping.csv`,
  `unmapped_labels.txt` and `coverage.json`. Face points are not kept.
- **Environment:** Kaggle runs Python 3.13, and mediapipe 0.10.21 has no wheel for it. Each cell
  installs uv, runs `uv python install 3.11` and `uv venv --python 3.11 /tmp/mpenv`, installs
  `mediapipe==0.10.21 opencv-python-headless "numpy<2" pandas pyarrow` there, and runs the
  extraction with Kaggle's PYTHONPATH and user site kept out. The log prints the venv's
  python, mediapipe, cv2 and numpy versions. If any step fails, the log shows the full error
  and `ENVIRONMENT FAILED`, and nothing is extracted.

Settings for all of them: **Accelerator: None** (MediaPipe's Python solutions run on the CPU) and
**Internet: On**. Run each with **Save Version > Save & Run All (Commit)**. A commit runs in the
background (12 h limit), tees everything to `<notebook>.log` and always finishes with exit 0, so
read the log in the version's Output.

| Order | Notebook file | Inputs | Takes |
|---|---|---|---|
| 1 | `kaggle/asl_citizen_smoke.ipynb` | edge-vision-src | a few minutes |
| 2 | `kaggle/asl_citizen_part1.ipynb`, `_part2`, `_part3` (side by side) | edge-vision-src | 3.5 to 5 h each |
| 3 | `kaggle/asl_citizen_merge.ipynb` | edge-vision-src + the three part notebooks' outputs | about 10 min plus the upload |

- **Smoke test:** tracks the first 20 videos into `asl-citizen-landmarks-smoke` (never the real
  shards, never a dataset). It prints the seconds per video and the share of frames with pose and
  hand landmarks, and ends with `SMOKE PASS` or `SMOKE FAIL`. Do not start the parts before it
  passes.
- **Parts:** each tracks every third shard (`EV_PART` 0/3, 1/3, 2/3). If Kaggle won't start
  three sessions at once, run them one after another. A finished part's log says
  `part k/3 finished`.
- **Merge:** add the three parts' finished versions as inputs (Add Input > Your Work >
  Notebooks). It copies their shards, tracks any shard still missing, writes the manifest and
  `COMPLETE`, then creates the dataset.
- **Resume:** if a run stops early, its log says INCOMPLETE. Add that version's own output as an
  input and commit again; finished shards are skipped.

### The private dataset (made by the merge run)

With the secrets, the merge run creates `asl-citizen-landmarks-private` itself (`kaggle datasets
create --dir-mode zip`, never `--public`) and prints its URL:

- Create an API token at kaggle.com > Settings > API > Create New Token.
- In the merge notebook, open Add-ons > Secrets and add `KAGGLE_USERNAME` and `KAGGLE_KEY` from
  `kaggle.json`.
- Tick both secrets for this notebook.

Without them, or if the upload fails, the log prints the manual steps. From the completed merge
version: Output > New Dataset (or Datasets > New Dataset > Notebook Output Files), choose the
`asl-citizen-landmarks` folder, title it `asl-citizen-landmarks-private` and leave Visibility on
**Private**. Or skip the dataset and add the merge notebook's output to `train_v3` directly.

The output is roughly 1 GB.

### Local test

`python scripts/kaggle_local_test.py --archive artifacts/v3/upload/edge-vision-src.tar.gz
--kaggle-python <a Python 3.13> --work <scratch dir> --reference data/live_clips_v2
test_clips_v2/<sign>_1.mp4 ...` runs the
cells as written, in bash, on a few clips. A venv plays Kaggle's Python and a stand-in archive
is served with range requests. It runs the missing and mismatched code source cases, the smoke
test, the three parts, the merge and the training input detection, then reads the landmarks
back and compares them with the local Holistic run of the same clips. Whether pose and hands are
in most frames depends on the clips, so the harness reports it rather than checking it (the demo
clips show a hand in only 21 to 38 percent of frames, in both runs alike). Delete the work folder
afterwards.

## 2. Evaluate the shipped v2 model on the ASL Citizen test split (local)

Download the dataset into `data/asl_citizen/landmarks/` (gitignored), from its page or with the
Kaggle CLI:

    kaggle datasets download <your-username>/asl-citizen-landmarks-private -p data/asl_citizen/landmarks --unzip

Then, once, before any v3 result exists:

    python scripts/citizen_eval.py --name v2

It logs the access in `results/asl_citizen/test_access_log.json` before reading test data,
refuses a second run on the same model, and writes `results/asl_citizen/v2_test.{json,md}` and
`v2_test_per_sign.csv`.

## 3. Train v3 (kaggle/train_v3.ipynb)

1. Import `kaggle/train_v3.ipynb` the same way.
2. Settings: **Accelerator: GPU T4** and **Internet: On**.
3. Inputs:
   - Your Work > Datasets > edge-vision-src (the code).
   - Competitions > Google - Isolated Sign Language Recognition.
   - The landmarks: Your Work > Datasets > asl-citizen-landmarks-private, or Your Work >
     Notebooks > asl_citizen_merge (the version whose log says COMPLETE). The cell finds the
     manifest and the shards at any depth, in a folder or zipped, and lists every candidate
     with the reason it was used or skipped.
4. **Save Version > Save & Run All (Commit).**

The cell checks that Kaggle's Python imports the training dependencies (after installing the
pinned requirements). If it cannot, the cell builds a Python 3.11 venv with uv, using the torch
build Kaggle ships and the same pins, and trains there.

The notebook does the following:

- Preprocesses asl-signs (about 17 min) and checks its split holds the same signers as v1 and
  v2.
- Preprocesses the ASL Citizen **train and val** videos only; their test split never enters
  Kaggle.
- Trains three seeds of `experiments/train_e_hybrid_v3citizen.yaml`. That is the v2 hybrid and
  recipe plus aspect, frame-rate and dropped-hand augmentation. Each epoch is selected by the
  mean of asl-signs and ASL Citizen val top-1.
- Picks the seed on validation.
- Exports fp16 ONNX with temperature and threshold fitted on the pooled val split.
- Packs `v3_bundle.zip`, which holds the run summaries, checkpoints, selection and export, and
  no data.

Expect about 3 hours. The v2 hybrid seeds took 0.3 to 0.7 h each on a T4. Finished seeds are
skipped when an earlier version's output is attached.

## 4. After the notebooks (local, run once)

Download `v3_bundle.zip` from the version's Output into `artifacts/v3/` (gitignored), then:

1. `mkdir -p web/models/v3-citizen && cp artifacts/v3/v3_export/model/* web/models/v3-citizen/`
2. Check the budget: p95 under 60 ms and file under 10 MB (`scripts/benchmark.py`, the browser
   bench).
3. Run the evaluations, each once:
   - `python scripts/citizen_eval.py --name v3 --model-dir web/models/v3-citizen` for the ASL
     Citizen test split.
   - `python scripts/final_eval.py --skip-runs --onnx-model web/models/v3-citizen/model.onnx --onnx-run d_hybrid_v2feat_v2recipe/seed_0 --processed-dir artifacts/v2/processed_v2 --final-dir results/final_selection --predictions-out results/v3_eval`
     for the asl-signs test split.
   - test_clips_v2 with `experiments/live_gap_v3.json`, Python and three browser runs
     (`scripts/live_gap.py`, `web/scripts/live-gap.mjs`).
4. `python scripts/compare_v3.py` writes `results/v2_vs_v3.{json,md,csv}`. That covers the three
   test sets side by side, with paired tests and a Holm correction across the top-1 comparisons.

## 5. When the research is complete

Delete the Kaggle dataset and the notebooks' versions and outputs. Locally, delete
`data/asl_citizen/`, `results/asl_citizen/predictions/`, `results/v3_eval/`, `artifacts/v3/` and
`web/models/v3-citizen/`.
