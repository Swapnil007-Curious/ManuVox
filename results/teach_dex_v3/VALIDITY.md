# The v3 Teach Dex run is compromised

`results/teach_dex_v3.md` (written by `scripts/teach_report.py`) reports the three record-then-test
runs with the v3 model as measured. Do not read its effect as evidence about teaching v3:

- Its model-alone clip top-1 on take 2 (23%) is far below v3's own clip top-1 on the same take-2
  clips through the same final browser pipeline in the live-gap runs (39%,
  `results/live_gap_v3/web/v3_final*.json`). For v2 the two harnesses agree (31% and 32%).
- Classified segments per run fell from 28 to 23 to 15 (v2's runs: 26, 29, 30), and the number of
  segments badged "Personalized for you" from 13 to 11 to 4. The harness logs no frame rate, so the
  cause cannot be shown from the record (other programs may have been running on the laptop).
- A clip without a segment cannot change in either condition, so the paired gain (+5 points) is
  biased toward zero.

To repeat it: keep the laptop idle, log the tracker's frame rate per clip, restart the browser
between runs, and interleave v2 and v3 runs (`web/tests/e2e/teach-eval.spec.ts` with
`TEACH_EVAL=1 TEACH_MODEL_DIR=web/models/v3-citizen TEACH_OUT=results/teach_dex_v3/web`, then
`scripts/teach_report.py --runs results/teach_dex_v3/web --out results/teach_dex_v3`). The paper
(Section 6.4) and the README report the run as compromised.
