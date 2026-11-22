# ManuVox paper

LaTeX preprint (ACL two-column style, `acl.sty` and `acl_natbib.bst` from
https://github.com/acl-org/acl-style-files) on the live gap of the ManuVox sign recognizer (research codename Edge-Vision).
Every number in the text is a macro generated from the result files; nothing is typed by hand.

## Build

From the repository root (Windows PowerShell; the same commands work in a POSIX shell):

```powershell
.venv\Scripts\Activate.ps1
pip install -r paper/requirements.txt          # matplotlib, only needed for the figures
python scripts/verify_references.py            # paper/data/reference_verification.json (network)
python scripts/paper_tables.py                 # tables/*.tex, tables/numbers.tex, data/*.json
python scripts/paper_figures.py                # figures/*.pdf
python scripts/paper_claims.py                 # CLAIMS.md (exit code 1 if any check fails)
cd paper
C:\Users\Admin\.local\tectonic\tectonic.exe main.tex   # main.pdf (any tectonic or TeX Live works)
```

Tectonic downloads the LaTeX packages it needs on first use. `latexmk -pdf main.tex` (or
pdflatex, bibtex, pdflatex twice) works as well. For Overleaf, upload `paper/` with `tables/` and
`figures/` already generated.

## Where the numbers come from

| Result | Produced by | Files read by the paper |
|---|---|---|
| Test accuracy, seeds, ensembles | `scripts/final_eval.py` | `results/tables/selection/*.csv` |
| Paired model comparisons, test calibration | `scripts/final_eval.py --predictions-out ...`, then `scripts/test_analysis.py` | `results/test_analysis/*` |
| Quantization accuracy | `scripts/final_eval.py --quant-root results/quant` | `results/tables/test_quant_summary.csv` |
| Python CPU latency | `scripts/benchmark.py` (v1 models; hybrid with `--quant-root results/web_export`) | `results/tables/bench_summary.csv`, `results/bench_v2/tables/bench_summary.csv`, `results/bench/bench_env.json` |
| Browser latency, model selection | `web/scripts/benchmark.mjs`, `scripts/model_selection.py` | `results/perf/classifier_bench.json`, `results/model_selection.csv` |
| Tracking fps (HUD on/off, CPU fallback) | `web/scripts/benchmark-tracking.mjs` | `results/perf/tracking_*.json` |
| Live gap v2 (50 clips) | `scripts/live_gap.py python`, `web/scripts/live-gap.mjs --per-clip`, `scripts/live_gap.py report` (with `--configs experiments/live_gap_v2.json`) | `results/live_gap_v2/summary.json`, `results/live_gap_v2.csv`, `.md` |
| Live gap pilot (10 clips) | the same, with `experiments/live_gap.json` | `results/live_gap.csv` |
| ASL Citizen test (v2 cross-dataset, v3), label mapping | `scripts/citizen_eval.py` (once per model), `src/edgevision/data/citizen_labels.py` | `results/asl_citizen/{v2,v3}_test.json`, `data/citizen_mapping.json` (counts snapshot) |
| v3 training and selection | `kaggle/train_v3.ipynb` (bundle in the git-ignored `artifacts/v3/`) | `data/v3_training.json` (counts snapshot) |
| v2 vs v3 on the three test sets | `scripts/compare_v3.py` | `results/v2_vs_v3.json` |
| v2 vs v3 browser latency | `web/scripts/benchmark.mjs`, `scripts/compare_latency.py` | `results/perf/v2_vs_v3_classifier_latency.json` |
| Teach Dex (personalization, separate) | `scripts/calibrate_teach.py`, `web/tests/e2e/teach-eval.spec.ts`, `scripts/teach_report.py` | `results/teach_dex.json`, `results/teach_dex_v3.json`, `results/teach_dex*/calibration.json` |

The live evaluation, end to end (clips are personal recordings and are not in git):

```powershell
.venv-live\Scripts\python scripts\prepare_live_clips.py --camera --clips test_clips_v2 --out data\live_clips_v2
.venv-live\Scripts\python scripts\extract_holistic.py --clips data\live_clips_v2
python scripts\live_gap.py python --configs experiments\live_gap_v2.json --results results\live_gap_v2
cd web; npm run build; npm run preview            # another shell; the head build on port 4190 (see live-gap.mjs)
node scripts\live-gap.mjs --configs experiments\live_gap_v2.json --per-clip --repeat 3 --channel chrome --delegate GPU
cd ..; python scripts\live_gap.py report --configs experiments\live_gap_v2.json --results results\live_gap_v2 `
  --csv results\live_gap_v2.csv --markdown results\live_gap_v2.md
```

## Layout

| path | content |
|---|---|
| `main.tex` | preamble, title, section order |
| `sections/*.tex` | the text |
| `tables/` | generated: one file per table and `numbers.tex` (the `\V{key}` macros) |
| `figures/` | generated vector figures (the live-pipeline figure is TikZ in `sections/livegap.tex`) |
| `data/` | `macro_sources.json` (key -> printed text, raw value, source file, field, scale), `reference_verification.json`, and snapshots of git-ignored inputs (dataset statistics, live-clip metadata, the skeleton frame's landmarks) |
| `references.bib` | bibliography; every entry checked against the ACL Anthology, arXiv, Crossref or the official proceedings page |
| `CLAIMS.md` | the claims audit |
| `REVIEW.md` | an independent review of the draft and how each point was handled |

## Open items before submission

- The Kaggle competition's citation block and data licence could not be read automatically (the
  pages render client-side; the API needs credentials). `kaggle2023islr` keeps the form verified
  against the page title and URL; confirm the author list and the licence sentence on
  https://www.kaggle.com/competitions/asl-signs (Overview > Citation, Data) in a logged-in
  browser.
- Limitations states: "The author is not a fluent signer; the signs in the live recordings were
  learned from online ASL video dictionaries." Confirm this describes the author's ASL background.
- How the 25 live-test signs were chosen from the 250 is not recorded; add one sentence to the
  Clips paragraph (Section 6.2) if it should be stated.
- Repeat the v3 Teach Dex run (Section 6.4 reports it as compromised; see
  `results/teach_dex_v3/VALIDITY.md`) on an idle laptop, with the frame rate logged, the browser
  restarted between runs and v2 and v3 interleaved.
- Have a fluent, ideally Deaf, signer check the 65 non-exact ASL Citizen label mappings and rate the
  50 live clips (second review, C5 and F1 in `REVIEW.md`).
