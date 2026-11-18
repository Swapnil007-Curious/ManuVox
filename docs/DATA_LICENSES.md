# Data and model licences

What each external dataset and asset allows, and the rules this repository follows for it.

## ASL Citizen (Microsoft Research License Terms)

[ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (Desai et al., 2023):
about 83,400 videos of 2,731 ASL signs from 52 signers, with official train, val and test splits.
It is downloaded from the Microsoft Download Center
(`https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip`,
45.9 GB); the licence text ships in the archive as `use.txt`.

The licence grants use "solely for non-commercial, non-revenue generating, research purposes" and
forbids distributing the data "or your modifications to the data". It also covers personal data
(the videos show people): keep it confidential, never use it to contact anyone, and destroy it
and every copy when the research is complete. Results may be published as long as "no material
or substantial portion of the Materials is included".

Rules for this repository:

1. **Non-commercial research only.** ASL Citizen data, and anything trained on it, is used for
   research only.
2. **Never redistribute.** No ASL Citizen video, frame, landmark, processed feature, manifest or
   per-video prediction is committed, uploaded or shared. `.gitignore` covers `/data/`, `*.npz`
   and `web/models/`; the Kaggle notebooks write only to their private outputs.
3. **Keep Kaggle private.** The notebooks (`kaggle/asl_citizen_extract.ipynb`,
   `kaggle/train_v3.ipynb`) and the dataset `asl-citizen-landmarks-private` stay Private. Never
   pass `--public` to the Kaggle CLI or switch them to Public.
4. **Minimise personal data.** The extraction keeps hands and pose and leaves the face points as
   NaN unless `EV_KEEP_FACE=1`; no edge-vision model uses the face.
5. **Delete when the research is complete.** Delete the Kaggle dataset, the notebooks' outputs and
   versions, and the local copies: `data/asl_citizen/` (landmarks, processed features, split
   files), `results/asl_citizen/predictions/`, `results/v3_eval/`, `artifacts/v3/` and
   `web/models/v3-citizen/`.
6. **Papers report results, not data.** Aggregate numbers (accuracy, intervals, per-sign accuracy)
   may be published. No dataset frame, video still, landmark plot of an ASL Citizen signer or
   per-video listing goes into a paper or figure.
7. **The v3 model stays off the public site.** It is trained on ASL Citizen, so it is not shipped
   publicly: the deployed build uses the asl-signs model (`modelVariant` `v2-kaggle`), and the v3
   ONNX lives in the gitignored `web/models/v3-citizen/` (README.md, "Model variant").

The label mapping (`src/edgevision/data/citizen_labels.py`) names ASL Citizen glosses, which are
ASL-LEX 2.0 entry names, as code. The generated mapping table stays in the private outputs.

## asl-signs (Google - Isolated Sign Language Recognition)

The Kaggle competition data (PopSign ASL, Deaf Professional Arts Network and Georgia Tech) is
licensed CC BY 4.0. The processed features are not committed (they are large); the trained v1 and
v2 models and the shipped `web/public/model/model.onnx` are.

## ASL-LEX 2.0

ASL-LEX 2.0 (Sehyr et al., 2021; CC BY-NC 4.0) was used to read the English translations of the
sign variants when the label mapping was chosen. Its data is not committed; download `signdata.csv`
from https://osf.io/zpha4/ into `data/asl_lex/` to repeat the lookup.

## MediaPipe

MediaPipe Holistic (Python, training and evaluation) and the Tasks HandLandmarker and
PoseLandmarker models (browser) are Apache-2.0; see `web/public/mediapipe/ASSETS.json` after
`npm run fetch-assets`.
