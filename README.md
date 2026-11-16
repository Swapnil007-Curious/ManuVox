# edge-vision

Landmark preprocessing, signer-independent splits and compact sequence classifiers for the
Kaggle [Google Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs)
dataset (`asl-signs`). Three models (bidirectional GRU, temporal CNN, small transformer) are
trained on preprocessed hand and pose landmarks and export to ONNX.

## Setup

Python 3.11, from the repository root (Windows PowerShell shown). torch is the CPU build from the
PyTorch index, so it is installed first:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
pip install -e .
```

`requirements.txt` holds the package dependencies (preprocessing, plus py-cpuinfo for benchmark
records), `requirements-train.txt` adds torch, onnx and onnxruntime, and `requirements-dev.txt`
adds the test and lint tools.

## Preprocessing

```powershell
python scripts/preprocess.py --help
python scripts/preprocess.py --data-root <asl-signs dir> --output-dir <output dir>
```

Settings live in `experiments/preprocess_default.yaml`, whose paths point at Kaggle.
`--data-root`, `--output-dir` and `--limit` override the YAML. `--no-cache` skips the per-sequence
cache in `<output dir>/cache/`, which otherwise makes reruns resumable.

The run writes `features.npz`, `manifest.csv`, `preprocess_report.json` and
`splits_<mode>.json` to the output directory.

## Training

Each model has a config in `experiments/train_<model>.yaml` (`gru`, `cnn`, `transformer`). A short
local smoke run on CPU:

```powershell
python scripts/train.py --config experiments/train_gru.yaml --seed 0 --processed-dir <processed dir> --epochs 2 --device cpu
```

A run writes `config.yaml`, `metrics.csv`, `best.pt`, `label_map.json`, `summary.json` and the
split file it used to `results/runs/<experiment>/seed_<seed>/`. A v1 config's experiment is
`<model>_<split_mode>` (e.g. `gru_signer_independent`); v2 configs name theirs. It refuses to
replace a completed run unless `--overwrite` is passed. The processed directory is only read.

Full training runs on a Kaggle GPU; see [kaggle/TRAINING.md](kaggle/TRAINING.md) for the notebook
cells.

## Evaluation protocol

- **Signer-independent split.** Participants are assigned whole to train, validation or test (4
  test and 3 validation signers), so test signers are never seen in training. A signer-dependent
  stratified split is trained only as a comparison.
- **Model selection on validation only.** Checkpoints, early stopping and the choice of best model
  use validation top-1. `scripts/aggregate_val.py` summarises validation metrics from the runs'
  summaries and never reads the test split.
- **One test evaluation script.** `scripts/final_eval.py` is the only code that loads the test
  split, for the checkpoints and (with `--quant-root`) their quantised variants, and it logs
  every invocation to `results/final/test_access_log.json`.
- **Seeds.** Each signer-independent configuration is trained with seeds 0, 1 and 2, and results
  are reported as the mean and standard deviation over seeds. Runs are seeded and use
  deterministic algorithms where available; GPU runs may still differ bitwise between repeats.
- **Seed ensembles.** For every experiment with at least two completed seeds, `aggregate_val.py`
  also scores the mean of the seeds' logits on validation (`val_ensemble.csv`) and
  `final_eval.py` on test (`test_ensemble.csv`).
- **Experiments.** Every table starts with an `experiment` column. Runs are found from their
  `summary.json` fields (model, split mode, feature set, recipe); summaries written before these
  fields existed are v1 runs.

## v2 ablation: features, recipe and architecture

The v2 grid changes one thing at a time against the v1 CNN (`cnn_signer_independent`), with the
same signer-independent split:

| Experiment | Model | Features | Recipe |
| --- | --- | --- | --- |
| `b_cnn_v2feat_v1recipe` | cnn | v2 | v1 |
| `c_cnn_v2feat_v2recipe` | cnn | v2 | v2 |
| `d_hybrid_v2feat_v2recipe` | hybrid | v2 | v2 |

- **v2 landmarks** (`experiments/preprocess_v2.yaml`): pose 0-16 (nose, eyes, ears, mouth
  corners, shoulders, elbows, wrists), so L = 59. Mirroring swaps every selected left/right pose
  pair. `preprocess_report.json` records the selection (`landmarks`) and, for v1 and v2 alike, the
  face/shoulder `aspect_stats`.
- **v2 features** (`models/features_v2.py`, inside the ONNX graph): coordinates, velocity,
  acceleration, the 210 pairwise distances of the dominant hand and 24 distances from its wrist
  and fingertips to the nose, mouth centre and shoulders; 588 features, standardised with train
  statistics. Any value derived from a missing point is 0.0.
- **v2 recipe**: 150 epochs, patience 30, an exponential moving average of the weights (decay
  0.999; validation and `best.pt` use it), and extra augmentations: frame dropout, time masking,
  landmark masking and hand-shape rotation.
- **Hybrid** (`models/hybrid.py`): three stages of two depthwise-conv blocks and one transformer
  block, with late head dropout (off for the first half of the epochs).

```powershell
python scripts/preprocess.py --config experiments/preprocess_v2.yaml --data-root <asl-signs dir> --output-dir <processed_v2 dir>
python scripts/run_grid.py --grid v2 --processed-dir <processed_v2 dir> --max-hours 10
```

`--max-hours` stops starting new runs once that much wall time has passed; a rerun resumes.
Quantisation handles v1-feature runs only; `scripts/export_web.py` also exports v2 runs (as fp32,
since they have no quantised variants) and, with `--ensemble`, the seed ensemble of a run's
experiment.
`src/edgevision/export/ensemble.py` exports the seeds of one experiment as one ONNX file that
averages their logits.

## Quantisation

Post-training quantisation with ONNX Runtime, for every completed signer-independent run:

```powershell
python scripts/quantize.py --runs-root results/runs --processed-dir data/processed
python scripts/final_eval.py --processed-dir data/processed --quant-root results/quant
```

`quantize.py` writes five variants per run to `results/quant/<model>_signer_independent/seed_<s>/`:

- `fp32.onnx`: the checkpoint exported at opset 17, velocity and standardisation included.
- `fp16.onnx`: float16 weights and compute with float32 input and output. If any val output is
  non-finite or val top-1 agreement with fp32 is below 99%, it is rebuilt with the feature block
  (velocity and standardisation) kept in float32, and the report says so.
- `int8_dynamic.onnx`: dynamic quantisation, int8 weights per channel.
- `int8_static.onnx`: static QDQ quantisation after `quant_pre_process`, uint8 activations and
  int8 per-channel weights, MinMax calibration on 1024 train sequences sampled with seed 0 and
  fed in batches of 64. Calibration reads only the train split. The feature block's nodes are
  excluded from quantisation, so the `landmarks` input, velocity and standardisation stay
  float32 and quantisation starts at the input of the first projection layer. Quantising the
  raw input would round coordinates to an 8-bit grid before velocity is computed, and could turn
  small coordinates (common near the torso centre after shoulder normalisation) into exact
  zeros, which the feature block treats as missing points. The build fails if no feature-block
  node is found, or if any QuantizeLinear node still reads the input or a tensor inside the block.
- `int8_static_full.onnx`: an ablation only, the same as `int8_static` (same calibration data
  and settings) but with nothing excluded, so the input and feature block are quantised too.
  It is evaluated and benchmarked like the other variants.

Each run's `quant_report.json` records, per variant, the file size, op counts, which MatMul, Gemm,
Conv and GRU nodes stayed float, the checker and onnxruntime load results, val top-1, val top-1
agreement with fp32 and the largest logit difference from fp32 on 512 val sequences. It also
records `excluded_node_count` and `excluded_node_names` (the nodes passed to the quantiser's
`nodes_to_exclude`, empty for every variant except `int8_static`) and `input_quantised`, whether a
QuantizeLinear or DynamicQuantizeLinear node reads `landmarks`. A variant that fails is recorded
there and the other variants still run. `results/tables/quant_ops_summary.csv` summarises the
reports. Existing variant files are reused unless `--overwrite` is passed; a file built from a
different `best.pt` is reported as stale. Each file records the schema version of its variant's
build recipe, and a file from an older version is rebuilt without `--overwrite`: `int8_static`
files built before the feature-block exclusion are replaced this way, while the other variants'
files are kept.

`final_eval.py --quant-root` evaluates every variant on test, first checking that each fp32
variant's test top-1 matches its checkpoint's within 0.1 percentage points. It writes
`results/final/quant/<model>/seed_<s>/<variant>_test_metrics.json` and
`results/tables/test_quant_summary.csv`, where deltas against fp32 are paired by seed. Without
`--quant-root` the script behaves as before. The `.onnx` files are not tracked by git; the reports
and tables are.

## Benchmarking

```powershell
python scripts/benchmark.py
```

The benchmark measures CPU latency for one sequence per call. Defaults: models `gru`, `cnn` and
`transformer` at seed 0, variants `torch_fp32`, `onnx_fp32`, `fp16`, `int8_dynamic`,
`int8_static` and `int8_static_full` (the ablation), and 1, 2 and 4 intra-op threads. `--batch N`
switches to a throughput mode whose output files carry a `_batch<N>` suffix.

Measurement protocol:

- **Inputs.** Val sequences in split order, cycled. The test split is never read.
- **Sessions.** Each configuration gets a fresh session: onnxruntime with `ORT_SEQUENTIAL`, one
  inter-op thread and `ORT_ENABLE_ALL`, or eager torch under `torch.inference_mode` with
  `torch.set_num_threads`. Session creation time and the approximate growth in process RSS are
  recorded.
- **Timing.** 100 warm-up calls, then 2000 calls each timed on its own with `time.perf_counter_ns`,
  with Python's garbage collector paused. The run reports mean, p50, p95, p99, min and max in ms,
  and throughput in sequences per second.
- **Rounds.** The full configuration list runs 3 times, in a new seeded random order each round,
  with a 3 s pause between configurations. The summary reports medians across rounds, plus
  `p50_spread_ms`, the largest minus the smallest p50. With the defaults, the pauses alone take
  about 8 minutes.
- **Machine state.** Run on AC power, with the Windows power mode set to Best performance and
  other applications closed. `results/bench/bench_env.json` records the CPU flags detected at
  runtime (AVX2, AVX-512, VNNI), the clock at the start and end, the power source, the active
  power scheme and power mode, and the library versions. A run on battery prints a warning and
  records it there.

Outputs: `results/bench/bench_raw.csv` (one row per configuration and round),
`results/bench/bench_env.json` and `results/tables/bench_summary.csv`. The summary's
`p95_share_of_33ms_frame` column is p95 latency as a share of one 30 fps frame.

## Web demo

Live: <https://swapnil007-curious.github.io/edge-vision/>

`web/` is an installable, offline-capable browser app (Vite and strict TypeScript, no UI
framework) that recognises one isolated sign at a time from the webcam. MediaPipe Tasks
HandLandmarker (two hands) and PoseLandmarker lite produce landmarks for each frame in a Web
Worker, a TypeScript port of the Python live pipeline (`web/src/livePipeline.ts`, `live.ts`,
`preprocess.ts`) maps them to the Holistic layout the model was trained on and segments the
stream into signs, and onnxruntime-web runs the exported classifier in a second worker. Every
step runs in the browser. The shipped model is the v2 hybrid (fp16), chosen in
`results/model_selection.md`; the live-gap evaluation on recorded webcam clips is in
`results/live_gap.md`.

- **Scope.** Recognises single isolated signs from a 250-sign ASL vocabulary (the asl-signs
  labels). It does not translate sentences.
- **Privacy.** Video never leaves the device. There is no analytics or telemetry, and after the
  first load the app makes no network requests: a Content-Security-Policy restricts every request
  to the app's own origin, and a service worker serves the app, the model and the MediaPipe files
  from the device. Spoken results, which are off by default, use on-device voices only.

### Run locally

Node 20.19 or newer, from `web/`:

```powershell
npm ci
npm run fetch-assets   # MediaPipe wasm and .task models into public/mediapipe/ (not in git)
npm run dev            # development server
npm run build          # production build into dist/
npm run preview        # serves dist/ at http://localhost:4173/
```

Camera access needs a secure context; `localhost` qualifies. `npm run fetch-assets` checks every
file against a pinned or npm-provided sha256, writes `public/mediapipe/ASSETS.json` (URL, sha256,
size, licence) and skips files that are already present.

Tests: `npx tsc --noEmit`, `npx vitest run` (the live-pipeline parity test on 20 val sequences,
preprocessing for the v1 and v2 landmark selections and the aspect correction against the Python
golden vectors, calibration, the recordings schema, pose interpolation, segmentation, scoring,
settings) and, after `npm run build`,
`npx playwright install chromium` then `npx playwright test` (model and preprocessing parity in
Chromium, offline reload through the service worker, a camera smoke test and a Record test signs
run with Chromium's fake camera). For a build under another base path, set `BASE_PATH` (for
example `/edge-vision/`) for both the build and the tests.

### Features spec: one source of truth

`features_spec.json` at the repository root holds every landmark constant: the selected
landmarks and their order (both hands, then the pose set; no face or lip points), the pose
points by name and the left/right pairs mirroring swaps, missing-point handling (NaN raw, a hand
present only when all 21 points are finite, 0.0 after resampling), the dominant-hand flip, the
shoulder normalisation and its fallback, the 32-frame resampling (by kept-frame index in
training, on the kept-frame clock live), z dropped, the model input layout, the aspect constants
and the live segmenter's thresholds. `src/edgevision/spec.py` reads it for training, export and
the offline pipeline and `web/src/spec.ts` imports it into the app bundle; neither defines those
constants itself. `scripts/calibrate_live.py` derives the data-dependent values (the training
aspect and the segmenter thresholds) from the train and val splits only.

Parity is tested end to end. `scripts/export_live_golden.py` writes
`web/tests/golden/live_parity.json`: 20 val sequences as raw landmarks (dataset rows and
MediaPipe Tasks-style frames with handedness labels), the Python pipeline's model inputs and the
exported model's logits, plus a stream of four sequences for motion energy and segmentation.
`web/tests/unit/liveParity.test.ts` runs the web pipeline and onnxruntime on the same raw
landmarks and fails unless every model input matches within 1e-4 and the top-1 is the same;
`tests/test_live_parity.py` checks that the fixture still matches Python, the committed model and
the spec.

### Live pipeline

Each frame, in order (`src/edgevision/live/pipeline.py` is the reference, `web/src/livePipeline.ts`
the port):

1. **Tracker mapping.** MediaPipe Tasks landmarks are mapped to Holistic's layout. Holistic's
   `left_hand` is the hand it crops at pose landmark 15 (the signer's left wrist), so a detected
   hand goes to the block of the nearer pose wrist, and two hands take the pairing with the
   smaller total distance; without a pose, Tasks' handedness label decides. z is dropped, as in
   training, so the different z scales of the two trackers never reach the model.
2. **Mirroring.** The landmarkers get the unmirrored camera frame; only the preview is drawn
   mirrored. Left-handed signing is flipped to the trained side by preprocessing's own rule (the
   left hand present in more kept frames than the right).
3. **Aspect.** Coordinates are fractions of the frame's width and height, and the training videos
   are portrait phone recordings. Every y is scaled about the frame's shoulder midpoint by
   k = train_width_over_height / (width / height), from the real size of the frame given to the
   landmarkers; train_width_over_height (0.723) comes from the face proportions of the training
   data. The body-ratio calibration below is kept as an alternative.
4. **Motion energy.** The speed of the wrist and fingertips over 100 ms, in shoulder widths per
   second, on the aspect-corrected points.
5. **Segmentation.** IDLE -> SIGNING when a hand is present and its energy reaches
   `start_energy`; SIGNING ends when the hands are gone for 200 ms, stay below `still_energy` for
   400 ms, or after 3 s. The thresholds are percentiles of the val split's energies
   (`calibrate_live.py`). While signing, an early guess is shown every 500 ms after the first
   600 ms; the model runs once on the finished segment.
6. **Resampling by time.** Every frame carries its capture time (`requestVideoFrameCallback`'s
   `captureTime`), and the 32 model frames are placed on the kept-frame clock (each kept frame
   advances it by its own capture interval, capped at 200 ms) instead of the kept-frame index.
   At a constant frame rate this is training's rule exactly; at 8-30 fps, or with dropped frames,
   the sign keeps its timing (`tests/test_live_resampling.py`).
7. **Decision.** softmax(logits / T) with the temperature T fitted on val (NLL), and a
   confidence threshold chosen on val to maximise coverage x (2 x selective accuracy - 1), i.e.
   accuracy traded against coverage, where an accepted wrong sign costs as much as an accepted
   right one gains. Both are stored in `meta.json` (`calibration`). Below the threshold the
   result reads "Not sure".

### Capture

The default automatic mode uses the motion segmenter above. Settings also offer the first demo's
presence rule (a segment starts after 3 consecutive frames with a hand and ends after 8 frames
without one, or after 3.0 s) and "hold to record": hold Space or press and hold the large button
while signing. Segments with fewer hand frames than the model's `min_hand_frames` are discarded.
The three most likely signs are always listed with their probabilities; if the top probability
reaches the threshold the top sign is shown large, otherwise the result reads "Not sure". "Segment details" under the result shows, for the last
segment, its frame count, normalisation method, aspect factor k, and the share of frames with
both shoulders visible and with a hand. All thresholds can be changed under Settings; settings
are kept in `localStorage` when the browser allows it. Framing can be the full camera frame or a
centred 3:4 portrait crop, applied before MediaPipe. Guidance messages report missing hands, hands
partly out of frame, low light, and a signer too close or too far (shoulder distance in
normalised units).

The pose selection and the left/right pose pairs that mirroring swaps come from `meta.json`, so
the same app runs v1 (7 pose points) and v2 (pose 0-16) models. MediaPipe runs on the GPU
delegate by default and falls back to the CPU if the GPU fails (at creation or while detecting) or is too slow;
the Performance panel shows which is in use and why. The hand landmarker runs on every frame and
the pose landmarker on every Nth frame (Settings, 1 to 3, default 2); pose points for the frames
in between are interpolated linearly in time between two runs, and frames after the last run of a
segment keep that run's pose.

### Aspect calibration (alternative mode)

The default aspect mode ("From the camera's frame size") uses the frame's width and height as described
above. The calibrated mode, the first demo's default, measures the body instead. The ratio
r = |y_nose - y_shoulder_mid| / |x_shoulder_left - x_shoulder_right| measures it;
`meta.json` carries the training median r_train from `preprocess_report.json`'s `aspect_stats`.
For 2 seconds after the camera starts (and on "Recalibrate"), the app takes the median r_live
over frames where the nose and both shoulders are inside the frame with visibility >= 0.5, and
sets k = r_train / r_live, clamped to [0.25, 4]. Before preprocessing, every landmark's y becomes
y_mid + (y - y_mid) * k, with y_mid the frame's shoulder-midpoint y (0.5 without both shoulders);
x is unchanged. Without visible shoulders the app asks the signer to move back and keeps k = 1.
Settings offer From the camera's frame size (default), Calibrated, Manual (a slider) and Off; the
Performance panel shows k and r_live. `src/edgevision/live/aspect.py` implements the same functions, and the web tests compare
them value for value.

### Record test signs and offline diagnosis

The "Record test signs" panel keeps, for every completed segment (classified, excluded or
discarded), the raw landmarks of each frame (hands after assignment and the 33 pose points, in
unmirrored normalised coordinates, before the aspect correction), frame times, pose visibility,
the camera resolution, framing, calibration and the app's features, model input and top-5, with
the sign chosen from a searchable list of the model's labels. No video or images are stored, and
recordings stay in memory until "Download recordings" saves them as one JSON file. Keep such
files under `data/live/`, which git ignores, then run:

```powershell
python scripts/live_diagnose.py --recordings data/live/<file>.json --meta web/public/model/meta.json --model web/public/model/model.onnx --processed-dir data/processed
```

It replays each recording through Python preprocessing and checks the app's features against it
(max |diff| <= 1e-4, exit code 1 otherwise), then evaluates top-1 and top-5 with no aspect
correction, with the app's recorded k, and with k = r_train / r_live computed offline per
recording session. It reports the predicted-label distribution, normalisation methods, shoulder
visibility, hand presence, segment lengths against the training lengths, r_live against
`aspect_stats`, and the standardised mean shift of each feature group against the model's own
FeatureBlock statistics, in `results/live/diagnosis_<timestamp>.json` and `.md`.

### Live-gap evaluation on recorded clips

The gap between dataset accuracy and the live demo is measured on webcam clips (file name = sign)
kept in `test_clips/`, which git ignores. Both pipelines see the same pixels: the offline Python
pipeline with legacy MediaPipe Holistic (the tracker asl-signs was built with), and the web app
in Chromium with the clip as its fake camera. Each config of `experiments/live_gap.json` changes
one setting, cumulatively from the shipped demo to the final pipeline, plus leave-one-out rows
that undo one fix at a time. Nothing is tuned on the clips: the constants come from
`features_spec.json` and `meta.json`, which are derived from the train and val splits.

```powershell
.venv-live\Scripts\python scripts/prepare_live_clips.py --clips test_clips --out data/live_clips  # requirements-live.txt
.venv-live\Scripts\python scripts/extract_holistic.py --clips data/live_clips
python scripts/live_gap.py python                                   # results/live_gap/python/
cd web; npm run build; npm run preview                              # another shell
node scripts/live-gap.mjs --channel chrome --delegate GPU           # results/live_gap/web/
cd ..; python scripts/live_gap.py report                            # results/live_gap.csv, .md
```

`prepare_live_clips.py` handles clips that are screen recordings of the demo page: it finds the
camera preview, removes the drawn landmark overlay by inpainting, flips the preview back to the
unmirrored camera view and writes 640 x 480, 30 fps `.y4m` files. The `c0_head_build` config runs
the demo as committed in 3fbafa3 (built with `experiments/head_build_hook.patch`, which only
records each segment's softmax) from a second preview server on port 4190.

### Exporting a model

`scripts/export_web.py` copies one ONNX variant to `web/public/model/model.onnx` and writes
`web/public/model/meta.json` (labels, feature set, input layout, landmark order, preprocessing
settings with the pose swap pairs, the training aspect statistics, sha256 of the ONNX file and of
each `best.pt`, opset, every variant candidate and the confidence threshold), plus the golden
vectors in `web/tests/golden/`:

```powershell
python scripts/export_web.py --run results/runs/cnn_signer_independent/seed_0 --processed-dir data/processed --verify-wasm
```

- **Variant rule.** Among fp32, fp16, int8_dynamic and int8_static, the smallest file whose val
  top-1 is within 0.5 percentage points of fp32 val top-1. For a run with a `quant_report.json`
  the candidates are its variants; otherwise (v2 runs, ensembles) they are built into
  `--work-dir`. Every candidate, with its size, val top-1 and why it was or was not eligible, is
  printed and stored in `meta.json`; `--variant <name>` overrides the rule and is recorded as
  forced.
- **onnxruntime-web check.** With `--verify-wasm` each exported file is built into the app and
  run through the Playwright model-parity test; if onnxruntime-web cannot run it, the next
  eligible variant is exported instead and the failure is recorded under `fallbacks`.
- **Seed ensemble.** `--ensemble` exports every seed of the run's experiment that has a `best.pt`
  as one ONNX file averaging their logits (`src/edgevision/export/ensemble.py`), and applies the
  same rule to its fp32, fp16 and int8_dynamic builds (int8_static finds the FeatureBlock by a
  node scope that ensemble members do not have, so it is not built).
- **Aspect statistics.** Taken from the processed dir's `preprocess_report.json`. A report written
  before `aspect_stats` existed can borrow them with `--aspect-report <report>` from another
  preprocessing run of the same raw data; the export checks that counts, kept sequences, split
  settings and the kept-frame rule match and that both selections hold the nose and shoulders.
- **Confidence threshold.** Computed on val only, from the exported variant's softmax: the
  smallest threshold t at which top-1 accuracy among val sequences with max probability >= t
  reaches `--target-precision` (default 0.80). If no t reaches it, the t with the highest
  accuracy among those accepting at least 20% of val is used and `meta.json` records a warning.
  Coverage, accuracy above t and overall val top-1 are stored with it. The test split is never
  read. The app uses this threshold only with the uncalibrated ("raw") decision setting.
- **Calibration.** Also on val only: a temperature T fitted by minimising the NLL of
  softmax(logits / T), then the threshold on the scaled probabilities that maximises
  coverage x (2 x selective accuracy - 1). `meta.json`'s `calibration` block stores T, the
  threshold, NLL and ECE before and after, coverage, selective accuracy and val top-1 / top-3;
  the app's default decision uses it.
- **Golden vectors.** `preprocess_cases.json` (v1 selection) and `preprocess_cases_v2.json`
  (pose 0-16) hold synthetic sequences built with the test suite's fixture builders (mirroring,
  ties, hand fallback, gaps, short and long sequences and both exclusion reasons) with the Python
  outputs; `aspect_cases.json` holds calibration, visibility and per-frame correction cases with
  the preprocessing that follows; `recording_example.json` is a recordings file in the download
  schema; `model_cases.json` holds 16 processed val inputs and the logits Python onnxruntime gives
  for them. The Python tests check that the committed files still match Python and the committed
  model.
- `--data-source synthetic` marks fixture data in `meta.json`; the app then shows a banner that
  the model does not recognise real signs. `meta.json` always states which data the committed
  model came from.
- Commit `model.onnx` only while it is under 10 MB; `.gitignore` admits only that one ONNX file.

### Performance

- **Workers.** `web/src/inference.worker.ts` runs in two instances: one holds the landmarkers and
  gets every frame as a transferred `ImageBitmap`, the other holds the onnxruntime-web session,
  so classifying a segment never stalls tracking. Frames are captured with
  `requestVideoFrameCallback`; while a detection runs, only the newest frame is kept and it goes
  to the worker the moment the detection finishes.
- **MediaPipe** runs on the GPU delegate (WebGL in the worker) and falls back to the CPU when the
  GPU fails, or when two frames after warm-up take over 250 ms each: with software-rendered
  WebGL (a blocklisted GPU, headless Chromium) the GPU delegate took ~1.3 s per frame against
  ~90 ms on the CPU.
- **onnxruntime-web** uses the wasm backend with SIMD. Wasm threads need cross-origin isolation:
  the dev and preview servers send COOP/COEP, and on GitHub Pages the service worker adds them to
  every response it serves (coi-serviceworker style), reloading the first visit once when it
  takes over. On the target laptop threads only slowed this model down, so the default is one
  thread (Settings can raise it). "Auto" also tries WebGPU and keeps it only if it is faster than
  wasm and its logits agree; otherwise, or on any WebGPU error, it falls back to wasm.

Measured on the target laptop (i3-1115G4, 4 logical cores, Intel UHD, 20 GB, Chrome 157), with a
recorded clip as the fake camera and the app's 60-second benchmark:

```powershell
npm run build; npm run preview            # in web/, another shell
node scripts/benchmark-tracking.mjs       # results/perf/tracking_gpu.json (--delegate CPU, --threads N)
node scripts/benchmark.mjs                # classifier alone: wasm x1/2/4 and WebGPU
```

| Configuration | Tracking fps | Hand landmarker p50 / p95 | Classifier per segment p50 / p95 |
| --- | --- | --- | --- |
| GPU delegate, classifier wasm x1 (default) | 24.1 | 30.6 / 43.0 ms | 18.6 / 27.2 ms |
| GPU delegate, classifier wasm x2 | 24.3 | 30.2 / 43.1 ms | 19.9 / 45.1 ms |
| GPU delegate, classifier wasm x3 | 22.5 | 33.3 / 46.4 ms | 40.4 / 81.5 ms |
| CPU delegate (fallback), wasm x1 | 9.8 | 85.2 / 127.6 ms | 20.6 / 22.0 ms |

The default meets the targets of at least 15 fps tracking and under 60 ms of model time per
segment; the CPU fallback does not reach 15 fps. The first demo (main thread, before these
changes) ran at 5-13 fps on the same clips.

### In-browser benchmark

Start the camera, open Performance and press "Run 60-second benchmark", then sign a few times.
The panel shows live FPS and rolling p50/p95 over the last 300 samples for the hand landmarker,
pose landmarker, preprocessing, classifier and the total per frame. At the end a JSON file is
downloaded with the user agent, `hardwareConcurrency`, `deviceMemory` where available, the actual
camera resolution and frame rate, the settings, the model variant and sha256, raw per-frame and
per-segment timings, p50/p95 summaries, end-to-end FPS, and 50 classifier runs on a fixed input.
It also records the requested and active MediaPipe delegate with any fallback reason, the pose
interval N (pose timings cover the frames where the pose landmarker ran), and the aspect mode, k,
r_live and r_train. The camera is requested at 640 x 480 and 30 fps. The file is only saved
locally.

### Deployment

`.github/workflows/pages.yml` builds `web/` with base `/edge-vision/` on every push to `main` and
deploys it with `actions/upload-pages-artifact` and `actions/deploy-pages`. In the repository,
Settings -> Pages -> Source must be set to "GitHub Actions". The `web` job in
`.github/workflows/ci.yml` runs the type check, unit tests, build and browser tests.

### Known limitations

- **Landmarks differ from the training data.** asl-signs was built with MediaPipe Holistic; the
  demo uses MediaPipe Tasks HandLandmarker and PoseLandmarker lite, whose landmarks, detection
  thresholds and failure modes are not identical, so accuracy in the browser can be lower than on
  the dataset.
- **Hand assignment.** Hands are assigned by the nearer pose wrist, Holistic's own rule. On
  frames without a pose the handedness label decides, and a hand far from both wrists (or two
  hands crossing) can still land in the wrong block. The Performance panel shows how often each
  hand sits at the matching pose wrist.
- **Camera aspect ratio.** Coordinates are normalised by the image width and height, so a camera
  whose aspect ratio differs from the recordings changes the geometry the model sees. The aspect
  correction (above) rescales y from the frame size (or, calibrated, from the nose and
  shoulders); it cannot correct lens distortion or a different viewing angle.
- **Isolated signs only.** Each segment is classified as one sign from the vocabulary; there is no
  sentence-level recognition, and signs outside the vocabulary are still mapped to one of its
  labels unless the confidence threshold rejects them.
- **CPU fallback is slow.** Without a working GPU delegate the hand landmarker takes about 85 ms
  per frame on the target laptop (it searches for a second hand on every frame), so tracking
  drops to about 10 fps. WebAssembly SIMD is required.
- **Small live evaluation.** The live-gap numbers come from 10 clips of one signer, two pairs of
  which are byte-identical files (see `results/live_gap.md`); they show the direction of each
  fix, not a precise accuracy.

## Tests

```powershell
ruff check .
ruff format --check .
mypy --strict src
pytest -q
```

Tests use synthetic fixtures only; the real dataset is not needed.

## Results

Result tables are generated into `results/tables/` by `scripts/aggregate_val.py`,
`scripts/final_eval.py`, `scripts/quantize.py` and `scripts/benchmark.py`; the v1 and v2
experiments together are in `results/tables/selection/`. For the web demo:

- `results/model_selection.md` (`scripts/model_selection.py`): test accuracy against classifier
  latency in Chrome on the target laptop, and the shipped model.
- `results/live_gap.md` and `.csv` (`scripts/live_gap.py report`): the live-gap ablation on the
  recorded clips, for the Python + Holistic and the browser + Tasks pipelines.
- `results/perf/`: the tracking and classifier benchmarks (`web/scripts/benchmark-tracking.mjs`,
  `web/scripts/benchmark.mjs`).
