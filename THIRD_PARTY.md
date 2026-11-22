# Third-party software and assets

ManuVox's own code is MIT-licensed (`LICENSE`). It builds on the third-party software, models and
fonts below, each under its own licence. Licences of the **datasets** (Google ISLR / asl-signs,
ASL Citizen, ASL-LEX 2.0) and the rules this repository follows for them are in
[`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md).

Licences were read from the installed packages' metadata (`npm`, `importlib.metadata`) and from
`web/public/mediapipe/ASSETS.json`, which `npm run fetch-assets` writes. Check the upstream
project for the authoritative text.

## Served to the browser (web/)

| Component | Version | Licence | Where |
|---|---|---|---|
| MediaPipe Tasks Vision (`@mediapipe/tasks-vision`, JS and wasm) | 0.10.35 | Apache-2.0 | `web/public/mediapipe/` (fetched, not committed) |
| MediaPipe HandLandmarker model (`hand_landmarker.task`, float16) | 1 | Apache-2.0 | `web/public/mediapipe/` (fetched, not committed) |
| MediaPipe PoseLandmarker lite model (`pose_landmarker_lite.task`, float16) | 1 | Apache-2.0 | `web/public/mediapipe/` (fetched, not committed) |
| ONNX Runtime Web (`onnxruntime-web`) | 1.31.1 | MIT | bundled by Vite |
| Manrope (`@fontsource-variable/manrope`) | 5.3.0 | OFL-1.1 | bundled by Vite |
| Bricolage Grotesque (`@fontsource-variable/bricolage-grotesque`) | 5.3.0 | OFL-1.1 | bundled by Vite |
| JetBrains Mono (`@fontsource-variable/jetbrains-mono`) | 5.3.0 | OFL-1.1 | bundled by Vite |

The model cards of the MediaPipe models are linked in `ASSETS.json`. The app's classifier
(`web/public/model/model.onnx`) is this project's own model, trained on the Google ISLR data
(CC BY 4.0, see `docs/DATA_LICENSES.md`).

Development-only JavaScript tools (Vite, TypeScript, Vitest, Playwright, Wrangler) are not
shipped; their versions are pinned in `web/package.json` and `deploy/worker/package.json`.

## Python (training, evaluation, live-gap tools)

| Package | Version | Licence |
|---|---|---|
| NumPy | 2.4.6 (`.venv`), 1.26.4 (`.venv-live`) | BSD-3-Clause (with bundled 0BSD, MIT, Zlib parts) |
| pandas | 3.1.0 | BSD-3-Clause |
| PyArrow | 26.0.0 | Apache-2.0 |
| pydantic | 2.14.2 | MIT |
| PyYAML | 6.0.3 | MIT |
| psutil | 7.2.2 | BSD-3-Clause |
| py-cpuinfo | 9.0.0 | MIT |
| PyTorch | 2.15.0 (CPU build locally, CUDA build on Kaggle) | BSD-3-Clause, with bundled Apache-2.0, BSD-2-Clause, BSL-1.0, ISC, MIT and Zlib parts |
| ONNX | 1.23.2 | Apache-2.0 |
| ONNX Runtime | 1.31.1 | MIT |
| MediaPipe (legacy Holistic) | 0.10.21 | Apache-2.0 |
| OpenCV (`opencv-contrib-python`) | 4.11.0.86 | Apache-2.0 |
| imageio-ffmpeg | 0.8.1 | BSD-2-Clause (the FFmpeg binary it downloads has its own LGPL/GPL licence) |
| Matplotlib (paper figures only) | 3.10.7 | Matplotlib licence (PSF-based) |

## Paper

`paper/acl.sty` and `paper/acl_natbib.bst` come from the ACL style files
(https://github.com/acl-org/acl-style-files) and keep their own licence terms.

## Hosted services (optional)

Sentence generation sends only the recognised glosses, through this project's Cloudflare Worker
(`deploy/worker/`), to a hosted language model (Groq). It is optional and subject to those
providers' terms; recognition itself runs entirely in the browser.
