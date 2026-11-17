# Edge-Vision paper

LaTeX preprint (ACL two-column style, `acl.sty` and `acl_natbib.bst` from
https://github.com/acl-org/acl-style-files) on the live gap of the Edge-Vision sign recognizer.
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
| `figures/` | generated vector figures (Figure 1 is TikZ in `sections/system.tex`) |
| `data/` | `macro_sources.json` (key -> printed text, raw value, source file, field, scale), `reference_verification.json`, and snapshots of git-ignored inputs (dataset statistics, live-clip metadata, the skeleton frame's landmarks) |
| `references.bib` | bibliography; every entry checked against the ACL Anthology, arXiv, Crossref or the official proceedings page |
| `CLAIMS.md` | the claims audit |
| `REVIEW.md` | an independent review of the draft and how each point was handled |

## Author TODOs

- Name, surname and e-mail in `main.tex` (`\surname{TODO}`, `\email{TODO}`).
- The AI assistance disclosure in the acknowledgments (`TODO-AUTHOR-WRITES-THIS`).
- State your ASL experience in the Limitations (`\authortodo{...}`, printed in red).
- State how the 25 live signs were chosen and where the sign forms came from, and the recording
  setup (distance, lighting, rest pose), in Section 6.2 (`\authortodo{...}`).
- Paste Kaggle's official citation block for the competition into `kaggle2023islr` (the author list
  could not be verified automatically; the page renders client-side) and confirm the data licence
  sentence on the competition's Data tab.
- Pick a title (options below).

Title options (the review judged "Closing the Live Gap" an overclaim, since most of the gap remains;
option 1 is in `main.tex`):

1. Where Live Sign Recognition Breaks: Auditing the Train--Serve Gap of a Landmark-Based ASL Recognizer on a Low-Power Laptop
2. From Benchmark to Webcam: Measuring the Train--Serve Gap in Landmark-Based Sign Recognition
3. One Silent Fault: Hand-Slot Assignment and the Live Gap of an In-Browser Sign Recognizer
