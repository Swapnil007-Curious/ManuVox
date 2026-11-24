# Review: "Closing the Live Gap: Signer-Independent Isolated Sign Recognition from Landmarks on a Low-Power Laptop"

Independent review of the draft in `paper/` (main.tex, sections, generated tables, `tables/numbers.tex`,
`data/macro_sources.json`, `CLAIMS.md`), checked against `results/`, `experiments/live_gap_v2.json`,
`features_spec.json` and the test-access logs. Browser live-gap numbers currently come from 2 of 3
runs; I do not comment on small numeric differences that will change. Where I re-analysed released
result files, I say so and label the numbers as **reviewer computations**. They are not numbers from
the paper.

How I re-analysed the live gap: clip scores come from `results/live_gap_v2/{python,web}/<config>*.json`
with the paper's rule (a clip's prediction is the mean probability vector of its classified segments;
a clip without one counts as wrong). Browser scores use complete runs only (currently 2). The test is
a sign-flip test with exact enumeration up to 20 non-zero differences, Holm-corrected within a
pipeline and metric. At the clip level this reproduces the paper's raw p-values (for example Python
c5 0.227, Python x_aspect_off 0.041, browser c2_hands 0.0029).

## Summary

The paper studies why a landmark-based isolated ASL recognizer trained on Google ISLR (250 signs; a
14/3/4 signer-disjoint split) nearly fails on a live webcam. It has three parts:

1. **Model comparison.** GRU, CNN and transformer baselines are compared with a CNN–transformer
   hybrid, with 3 seeds and sequence-level paired tests. The hybrid reaches 68.5% test top-1,
   +6.1 points over the v1 CNN.
2. **Efficiency on a low-power laptop.** Quantization and browser latency on an i3-1115G4.
   int8 does not pay off; the fp16 hybrid takes 18.6 ms per call in Chrome.
3. **Live-gap audit.** Six train–serve discrepancies are evaluated on 50 webcam clips (25 signs,
   2 takes) recorded by the author. Two pipelines are used: deterministic Python + Holistic, and a
   real-time browser + MediaPipe Tasks pipeline. The design combines a cumulative ablation with a
   leave-one-out ablation, a pre-specified clip-level analysis, bootstrap intervals and
   Holm-corrected sign-flip tests.

Live clip top-1 rises from 2% for the shipped build to 34% (browser) and 40% (Python). Hand-slot
assignment is the dominant fix. The motion segmenter did not help. Temperature scaling lowers test
ECE from 0.083 to 0.033. The paper concludes that the remaining gap is mostly distribution shift.

## Strengths

1. **Number provenance is exemplary.** Every number is a macro traced to a result file, and the
   automated audit passes (216/216). I found no transcription errors.
2. **The paper reports failures.** It reports a failure (2% live) and a negative result (the motion
   segmenter), and it explicitly refuses to switch to the better presence segmenter after seeing the
   live clips. This is the right call, and it is rare.
3. **The two-pipeline design is well chosen.** A deterministic Holistic reference next to the
   real-time browser pipeline is a good way to separate tracker effects from code effects. Running
   both cumulative and leave-one-out ablations, and reporting better/worse clip counts next to
   p-values, is good practice.
4. **The live analysis was planned in advance.** The unit, test, multiplicity correction and
   bootstrap were fixed beforehand. The unit's limitation (two takes per sign) is acknowledged in the
   plan itself.
5. **The model evaluation is careful.** It uses a signer-disjoint split, 3 seeds, logged test
   access, and an honest note that the shipping rule used test top-1, with a validation cross-check.
6. **The findings are useful to practitioners.** Hand-slot assignment is a silent failure mode.
   Golden-fixture parity tests in CI cover hand assignment with shuffled and mislabelled hands. int8
   is not worth it on this CPU, and one wasm thread is best.
7. **Limitations and ethics are candid.** They cover the single signer, the absence of Deaf users,
   ASL versus ISL, and the "not a translator" point.

## Issues

Severity: **major**, **minor** or **nit**. "(needs new experiments)" marks fixes that need new
runs, training or annotation; everything else is re-analysis or rewriting.

### A. Claims vs evidence

**1. [major] The "six faults" narrative is not what the ablation shows.**

*Location:* abstract ("We trace six faults"); introduction, bullet 3; §6 opening ("We traced the gap
to six faults"); §6.1; conclusion.

*Problem:* Only (a) has a large, consistent measured effect. Per `experiments/live_gap_v2.json`
(the `c0_*` configs and the `settings` block), the six items break down as follows:

- **(b) Mirroring.** The first demo did *not* detect on a mirrored frame (`mirrored: false`), so
  (b) was never present in the shipped build. `x_mirrored` tests a hypothetical.
- **(c) Aspect ratio.** The first demo *already* had an aspect correction: `aspect: "calibrated"`,
  a body-ratio estimate over the first 2 s. `c3_aspect` replaces that correction with the
  frame-based k; it does not add a correction where none existed. `x_aspect_calibrated` puts the
  first demo's method into the final pipeline and scores the same as the final pipeline (Python 40
  vs 40, 0 clips changed; browser +2, not significant). The data therefore show that *some*
  correction is needed (`x_aspect_off`: −20 and −16 points). They do not show that the first demo's
  aspect handling was a fault. The paper never says the first demo corrected aspect, and the table
  row "final, body-ratio aspect" is never explained.
- **(d) Frame rate.** No measurable effect: 0 in Python by construction, +1 in the browser, and the
  leave-one-out is also ~0.
- **(e) Segmentation.** The "fix" lowered the point estimates.
- **(f) Calibration.** Cannot change a segment's top-1, so it cannot be a cause of the accuracy gap.

*Fix:*
- Reframe the six items as "six candidate train–serve discrepancies, audited, of which one
  (hand-slot assignment) accounts for most of the recovery."
- In §6.1, state for each item what the first demo did and whether the shipped build actually had
  the problem. The `settings` block of the config already contains this.
- Explain `x_aspect_calibrated` in the text.
- Rewrite the abstract and conclusion to match.

**2. [major] "Most of that collapse was train–serve skew" and "what remains … is largely shift" are
not supported.**

*Location:* conclusion; abstract, last sentence; §6 "Python and browser agree".

*Problem:*

- **"Most" overstates the recovery.** Depending on the accounting, the code fixes recover about a
  third to a half of the collapse:
  - v1 CNN: 62.5% on test, 1–2% live as shipped, 28–30% in the final pipeline (`x_v1_model`), so
    about 45% recovered.
  - Hybrid: 68.5% on test, 10–16% live (`c1_model`), 34–40% in the final pipeline, so 34% recovered
    in the browser and 51% in Python.
- **The residual is not shown to be shift.** The Python–browser comparison isolates only the
  *tracker*. Both pipelines share the live code: segmenter, aspect correction and time resampling.
  A discrepancy that both live pipelines share relative to training preprocessing is invisible to
  this comparison.
- **The residual demonstrably includes pipeline effects.**
  - In the final pipeline, 7/50 clips (Python) and 10–12/50 clips per browser run have **no
    classified segment at all**. With presence segmentation the counts are 1 and 0 (reviewer count
    from `c6_final*.json` and `x_presence*.json`).
  - `x_presence` is 8–14 points better than the final pipeline.
  - 4 of the 5 most frequent browser "confusions" (`lg.web.confusions`) are "no segment".
- **The comparison metric is not like-for-like.** Live clip-level scoring (mean over segments; no
  segment counts as wrong) is not the same metric as top-1 on pre-segmented test sequences. Also,
  the 25 live signs average 72.9% test top-1 (Table persign), not 68.5%.
- **Other explanations are not separated.** These include the author's sign productions (the ASL
  experience is still a TODO, and dad→donkey happens in every take although "dad" is 96% on test),
  a 15 fps webcam against the assumed 30 fps of the dataset, and viewpoint and camera differences.
- **The prescriptive claim is untested.** "Closing it needs data from more signers" has no
  experiment behind it.

*Fix:*
- Rewrite to: "fixing pipeline code recovered between a third and a half of the collapse; the
  remaining gap includes segmentation failures, signer and camera shift, and possibly production
  differences, which this design cannot separate."
- Add controls:
  - (i) Run the Python live pipeline (segmenter, aspect, time resampling) on test-split landmark
    sequences padded with rest frames. This measures the pipeline's own cost on in-distribution
    data.
  - (ii) Add an oracle-segmentation condition on the live clips, using hand-marked sign boundaries.
  - (iii) Draw a learning curve over the number of training signers, to support "more signers".
- (i) and (ii) are cheap and need no new recordings. (needs new experiments)

**3. [major] The abstract and introduction apply different standards to (c) and (e).**

*Location:* abstract ("a motion-based segmenter lowered accuracy"); introduction, bullet 3; §6
"Which fixes mattered" and "The motion segmenter did not help".

*Problem:* The segmenter's harm is stated as a finding, although it is not significant:

- Python: Holm p = 1.000, raw p = 0.23, interval [−22, 2].
- Browser: Holm p = 0.176.

Removing the aspect correction costs *more*, yet it is described as "not surviving correction":

- Effect: −20 and −16 points, intervals excluding 0, 15–16 clips worse against 5 better.

Both effects are non-significant after Holm; one is reported as a result and the other as a null.

*Fix:* Use one standard throughout. For example: "Hand-slot assignment had a large effect in both
pipelines (+26 to +36 points). Removing aspect correction and switching to the motion segmenter
both lowered accuracy by 10–20 points, but with 50 clips neither survived Holm correction." The
conclusion's "did not help" is fine.

**4. [major] The headline significance claim for hand assignment depends on treating clips as the
unit and on the current browser runs.**

*Location:* abstract ("the fix whose effect is significant after correction"); introduction,
bullet 3 ("in both pipelines"); Table 4 asterisks.

*Problem:*

- **The browser result sits near the threshold.** The browser `c2_hands` comparison has
  p_Holm = 0.038 (raw 0.0029) from 2 of 3 runs. Its per-run top-1 currently differs by 8 points
  (46 vs 38; reviewer computation), so the third run can move it across 0.05.
- **The unit ignores the dependence between takes.** The two takes of a sign are not independent
  (acknowledged in Limitations), yet the test flips clips individually.
- **Re-running with the sign as the unit changes the browser result.** I re-ran the pre-specified
  test with 25 sign clusters (summing the two takes' differences), same scoring and same Holm
  families (reviewer computation):

  | Comparison | p_Holm, sign as unit |
  |---|---|
  | Python `c2_hands` | ≈ 0.003 |
  | Python `x_hands` | ≈ 0.003 |
  | Browser `x_hands` | ≈ 0.04 |
  | Browser `c2_hands` | ≈ 0.19 |
  | Browser `c5_segmenter` | ≈ 0.34 |
  | Browser `x_presence` | ≈ 0.23 |
  | Python `x_aspect_off` | ≈ 0.75 |
  | Browser `x_aspect_off` | ≈ 1.0 |

The *effect* of hand assignment is not in doubt. It is large and has the same direction in both
pipelines and in both ablation directions, with 16–19 clips better against 0–4 worse. What is
fragile is the specific claim "significant after correction in both pipelines".

*Fix:*
- Lead with effect sizes and consistency, not with a pattern of significance.
- Report cluster (sign-level) permutation tests and bootstrap intervals that resample signs with
  both takes, at least as a sensitivity analysis. State that the clip-level analysis was the
  pre-specified one.
- Drop "the fix whose effect is significant" from the abstract.

**5. [major] Test-split model claims are made at the sequence level but phrased as general.**

*Location:* introduction, bullet 1; abstract ("better on every test signer"); §4 Results;
Table comparisons.

*Problem:* With 4 test signers, the bootstrap and permutation tests treat 16,242 sequences from
4 people as independent. Limitations acknowledges this, but the introduction and §4 still say
"significant and consistent across all test signers" and "the training recipe and the architecture
account for the gain".

The per-signer differences in `results/test_analysis/model_comparisons.csv` are:

| Comparison | Signer 1 | Signer 2 | Signer 3 | Signer 4 |
|---|---|---|---|---|
| Recipe | +0.2 | +1.2 | +1.6 | +0.7 |
| Architecture | +6.5 | +5.5 | +2.3 | +6.1 |
| Hybrid vs CNN | +5.9 | +6.2 | +3.9 | +8.4 |

A paired t-test over the 4 signers (reviewer computation; fragile at n = 4) gives:

- recipe p ≈ 0.06;
- architecture p ≈ 0.013;
- hybrid vs CNN p ≈ 0.007;
- GRU vs CNN p ≈ 0.045.

Holm over the six rows leaves only hybrid vs CNN (≈ 0.044). A 4/4 sign test can never fall below
p = 0.125.

Hybrid accuracy per signer ranges from 51.9% to 77.7% (`test_per_participant.csv`). A signer-level
95% interval on the mean is therefore about ±20 points. This also matters for the live gap: the
live signer's 40–48% is close to the weakest test signer.

*Fix:*
- Add a per-signer accuracy table and the per-signer deltas to the main text.
- Describe the recipe effect as "small (+1.0) and positive for all four signers, but not
  distinguishable from signer variation."
- Use a signer-level analysis (a mixed model with a signer random effect).
- Ideally, rotate the test signers (k-fold over signers) so that more than 4 signers are tested.
  (needs new experiments for the k-fold part)

**6. [major] The attribution of the gain to "recipe and architecture" is confounded.**

*Location:* introduction, bullet 1; §4 Training and Results.

*Problem:*

- **The grid is cumulative.** It goes features → recipe → architecture, not "one factor at a time"
  as §4 says. Each step's effect is conditional on the earlier steps and on their order.
- **The "architecture" step changes more than architecture.** It also raises the parameter count
  8.5× (0.40M → 3.40M), the width from 192 to 256, and the training time from 10 to 31 minutes.
- **The "recipe" step is a bundle.** It combines 2.5× more epochs, EMA, a halved learning rate and
  four new augmentations.
- **The features result is narrow.** "Richer features alone do not help" was tested only with the
  CNN and the v1 recipe.

*Fix:*
- Call the grid a cumulative ablation.
- Say "the hybrid, with 8.5× the parameters" in the abstract and introduction, and soften
  "account for".
- For attribution, train a hybrid with the v1 features and/or the v1 recipe, and a CNN of
  comparable size with the v2 recipe. (needs new experiments)

**7. [major] The title overclaims.**

*Location:* `main.tex` title.

*Problem:* "Closing the Live Gap" contradicts the paper's own result: a 30–35 point gap remains
(34–40% live against 68.5% on test), measured on one signer.

*Fix:* Use a measurement title, such as option 3 in `paper/README.md` ("Where Live Sign Recognition
Breaks …"), or "Auditing the Train–Serve Gap …".

**8. [minor] The abstract misattributes the first deployment.**

*Location:* abstract.

*Problem:* "…a compact … hybrid reaches 68.5% … *Its* first live deployment recognized 2%". The
shipped build (`c0_head_build`) used the v1 CNN, not the hybrid. Also, "the fixes raise clip top-1
to 34%" counts from 2%, so the gain includes the model swap (`c1`), and the cumulative chain
includes a step that lowered accuracy.

*Fix:* "The first deployment, with a v1 CNN, recognized 2% …; the model change and the fixes
together raise …".

**9. [minor] "Calibration that transfers" holds only within the dataset.**

*Location:* introduction, bullet 4; §6.5 ("the rule transfers to unseen signers").

*Problem:*

- Live, the same threshold gives 44% coverage at 67% selective accuracy. The net score is about
  0.15, against 0.47 on test.
- "Holds back 76% of the wrong guesses" is one-sided: it also rejects about 28% of the correct
  segments.
- After temperature scaling, the upper bins become *overconfident* (`reliability.csv`: top bin
  0.969 confidence vs 0.946 accuracy; next bin 0.904 vs 0.866). That is the region the 48.8%
  threshold accepts.

*Fix:*
- Write "transfers to unseen dataset signers".
- Report the live net score and the rejection rate for correct segments.
- Mention the overconfidence after scaling.
- Add bootstrap intervals for ECE.
- Report segment-level ECE on the live clips (no new data needed).

**10. [minor] The claim that the two pipelines agree is not tested.**

*Location:* §6.3 "Python and browser agree".

*Problem:*

- **The differences are not "a few points".** They reach 6–7 points (c1, c5, c6).
- **The inference is untested.** No paired Python–browser test was run or pre-specified, yet the
  text concludes that "the browser trackers are therefore not the main remaining source of error".
- **"The same way" overstates the per-sign picture.** 9 of 25 signs fail in every take in both
  pipelines. Only dad→donkey, go→green, sad→mitten, hungry→giraffe and no→read are shared
  confusions. Bird, finish, happy, thankyou and water fail differently or produce "no segment", and
  book, dog, mom, no and please differ between the pipelines.

*Fix:*
- Report paired Python–browser differences with intervals for c6 and x_presence.
- Rephrase as "tracker differences account for at most about 6 points on these clips".
- Rewrite the per-sign sentence.

**11. [minor] Two statements in §6.3 misdescribe the data.**

*Location:* §6.3.

*Problem:*
- **Mirroring.** "(b) … fails only together with the first demo's label mapping (7%)" is not what
  the numbers show. `x_hands` *without* mirroring is already 8%, and `x_mirrored_label` against
  `x_hands` is −1. The failure is entirely the label mapping.
- **Calibration.** "Calibration (f) does not change a segment's top-1", yet clip top-1 moved by +2
  and +3. Temperature changes the mean of the probability vectors over segments; in the browser,
  run-to-run noise adds to this.

*Fix:* Correct both sentences.

**12. [minor] Several efficiency claims are untested or underspecified.**

*Location:* §5.

*Problem:*
- **"9.8 fps … too slow for fluent signing" is untested.** The live clips are themselves recorded
  at 15 fps.
- **"77 ms after the sign ended" is not user-perceived latency.** It is measured from the
  segmenter's end decision, which comes 200–400 ms after motion stops (`hand_lost_ms`,
  `still_ms`).
- **fp16 is slower on CPU.** It is slightly slower than fp32 in Python for the CNN, transformer
  and hybrid, and equal for the GRU (Table quant), which the text does not say.
- **The 60 ms budget is not justified.**

*Fix:*
- Test the 10 fps claim by subsampling the clips to 10 fps in the Python pipeline. (needs a cheap
  new run)
- State the user-perceived latency.
- Mention the fp16 slowdown.
- Justify the 60 ms budget.

### B. Statistics

**13. [minor] The live-gap intervals and the run-to-run variation are not handled.**

*Location:* §6.2 "Scoring and statistics"; Limitations.

*Problem:*
- **The bootstrap resamples the wrong unit.** The percentile bootstrap resamples clips; it should
  resample signs with both takes. Limitations correctly says the intervals are "if anything, too
  narrow", but the paper can fix this rather than disclaim it.
- **Browser run-to-run variation is averaged away and not reported.** For `c2_hands`, 6 clips
  change their top-1 between the two runs. A useful noise floor is `c6_final` vs `c5_segmenter`:
  they differ only in temperature, yet show 6 clips better and 3 worse in the browser.

*Fix:*
- Cluster the bootstrap by sign.
- Report per-run accuracies and a run-to-run SD per configuration.
- As a sensitivity analysis, consider a mixed-effects logistic model (clip nested in sign, runs as
  repeated measures).

**14. [minor] The intervals and the tests are mixed without saying so.**

*Location:* §6.3; Table livegap_ci.

*Problem:* The difference intervals are unadjusted, while the p-values are Holm-adjusted. The text
therefore repeatedly places intervals that exclude zero next to "not significant" (`x_aspect_off`,
browser `c5_segmenter`, browser `x_presence`). Table livegap_ci shows only p_Holm.

*Fix:*
- State once that the intervals are unadjusted, or report simultaneous intervals consistent with
  the correction.
- Add raw p-values next to the adjusted ones.

**15. [minor] Some multiplicity choices are not explained.**

*Location:* §6.2.

*Problem:*
- **The families include comparisons that are null by construction.** In Python, `c4_time` and
  `x_index` "cannot matter", and for top-1 calibration cannot change a segment's prediction. This
  was pre-specified, so keep it, but it costs power and should be noted.
- **The family-wise error rate is per family only.** Top-1 and top-3, and the two pipelines, are
  separate families (52 tests in all), and the family-wise rate across families is not controlled.

*Fix:* Note both points in the text.

**16. [minor] Some intervals are misdescribed or poorly suited.**

*Location:* §6.2; Table comparisons.

*Problem:*
- At n = 50, percentile bootstrap intervals for accuracies near 0 (for example 2% [0, 6])
  undercover.
- The test-split intervals use correctness averaged over seeds, so they exclude seed variance; the
  paper should say so.

*Fix:* Use Wilson or Clopper–Pearson intervals for the deterministic Python pipeline, and note that
the test-split intervals exclude seed variance.

**17. [minor] Low power is not discussed.**

*Location:* §6.3; Limitations.

*Problem:* With 50 clips (25 clusters) and Holm over 12–14 comparisons, effects much below
15 points are unlikely to be detected. The non-significant results for (c), (e) and the model swap
are therefore *inconclusive*, not evidence of no effect.

*Fix:* Give a minimum detectable effect and cite Card et al. (EMNLP 2020, "With Little Power Comes
Great Responsibility").

### C. Protocol

**18. [major] The test split was accessed repeatedly during development, and the disclosure is
incomplete and partly contradictory.**

*Location:* §3 "Signer-independent split"; §4 Results.

*Problem:*

- **§3 contradicts §4.** §3 says "all model selection … use[s] validation only". §4 says the
  shipping rule took the highest *test* top-1 (disclosed there).
- **There were at least six logged test evaluations, all with `git_dirty: true`:**

  | Log file | Date | What was evaluated |
  |---|---|---|
  | `results/final/test_access_log.json` | 2026-11-04 | v1 models |
  | `results/final/test_access_log.json` | 2026-11-04 | quantized v1 models |
  | `results/final_v2/test_access_log.json` | 2026-11-06 | v2 models |
  | `results/final_selection/test_access_log.json` | 2026-11-15, 11-16, 11-17 | all models; ONNX model |

- **The v2 design followed the v1 test results.** The v2 features, hybrid and recipe (commit
  3fbafa3, 2026-11-05) came after v1 test results were available, including test-set confusions
  (`results/tables/top_confusions.csv`: hear→ear, wake→awake, committed in d34f874 before 3fbafa3).
  Hand-to-face distance features are the kind of change such confusions suggest. I am not claiming
  leakage, but a reader cannot rule it out.
- **The selection thresholds came after the test accuracies.** The rule's thresholds
  (p95 < 60 ms, < 10 MB) were fixed after test accuracies were known. The 10 MB cap is what
  excludes the ensemble (and the fp32 hybrid). The cap is a repository convenience, not a
  deployment constraint: a 20.7 MB fp16 ensemble loads in a browser.

*Fix:*
- Correct the §3 sentence.
- Add an appendix table of every test access: date, commit, what was evaluated, and what changed
  afterwards.
- State that the v2 design followed the v1 test evaluation, and whether test confusions informed
  it.
- Justify the 10 MB cap or drop it.

**19. [minor] The pre-specification cannot be verified from the repository.**

*Location:* §6.2 ("the analysis below was fixed before the first run"); Appendix E.

*Problem:*
- **The plan is not committed.** `experiments/live_gap_v2.json` says it was "fixed on 2026-11-17
  before any v2 run", but the file is untracked in git.
- **The code is not committed either.** The harness and reporting scripts (`scripts/live_gap.py`,
  `web/scripts/live-gap.mjs`, `scripts/prepare_live_clips.py`) are uncommitted modifications. The
  appendix's "commit 36219ea" therefore does not contain the code that produced Table 4.
- **No primary comparison was designated.** The plan tests every comparison equally, which costs
  power.

*Fix:*
- Commit the plan and the code, and cite that commit.
- In future, timestamp or hash the plan before running.
- Consider designating one primary contrast per pipeline.

**20. [minor] Several choices around the live clips are not disclosed.**

*Location:* §6.2 "Clips"; Appendix D.

*Problem:*
- **Sign selection.** The paper does not say how the 25 signs were chosen, or whether this
  happened before or after per-class test accuracies were seen.
- **Fix order.** The order of fixes was "motivated by the pilot", which was recorded by the same
  person.
- **Experimenter effects.** The author is both the developer and the only test signer, and
  experimenter effects can run in either direction.

*Fix:* State all three points in §6.2.

### D. Method details and reproducibility

**21. [major] Details needed to interpret the main negative result (segmentation) are missing.**

*Location:* §6.1 (e); compare `features_spec.json` → `live.segmenter`.

*Problem:* None of the following is in the paper:

- (i) **The frame-rate assumption.** The thresholds were derived on validation sequences assuming
  frames are 30 fps apart. The dataset has no timestamps, so energies in shoulder widths per second
  depend on that assumption. If the true rate is lower, speeds are overestimated and the thresholds
  are too high. That would produce late starts, early ends and missed signs, which is what was
  observed.
- (ii) **How the thresholds were derived.** `start_energy` is the 10th percentile of the
  per-sequence peak energy; `still_energy` uses a 5% criterion.
- (iii) **Minimum length and gaps.** Segments shorter than 250 ms are discarded (13 discarded in
  Python c6, 18 in browser run 1), and the maximum gap is 200 ms.
- (iv) **Energy uses aspect-corrected coordinates.** So (c) and (e) interact. This likely explains
  why removing (c) costs 16–20 points in the final pipeline, while swapping the aspect method costs
  about 0 under presence segmentation (c3).
- (v) **The presence segmenter is not defined in the paper.** It is the best configuration
  measured, but it appears only in the config ("3 frames with a hand start, 8 without end, 3 s
  cap").

*Fix:*
- Add (i)–(v) to §6.1 or the appendix.
- Discuss the (c) × (e) interaction.
- Break down the segmenter's failures by cause: no segment, truncated segment, or
  misclassification. On the counts above, most of the Python loss comes from clips with no
  classified segment rather than from truncation, which differs from the paper's explanation that
  it "ends signs before their last frames".

**22. [minor] Other details needed for reproduction are missing.**

*Location:* §3, §6, §7 and the appendix.

*Problem:*
- How signers were assigned to splits (randomly? with which seed?), and the participant IDs.
- The dataset frame rate, and what "the recording rate" in (d) refers to.
- The first demo's decision rule ("target-precision threshold") and its body-ratio aspect method.
- The browser harness: Chrome fake camera from .y4m, real-time playback, GPU delegate, and a seeded
  random config order per clip. This is in `results/live_gap_v2.md` but not in the paper.
- The recording protocol: distance, lighting, rest pose, dominant hand, and where the sign forms
  came from (a dictionary? the dataset's reference videos?).
- Hybrid pooling and head (global average pooling, per the code).
- The hosted LLM used for sentences: provider, model and prompt.

*Fix:* Add these to §3, §6.2 and the appendix, and release the participant IDs.

**23. [minor] The 43.2% mirroring rate is unexplained.**

*Location:* §3.

*Problem:* 43.2% of sequences are mirrored by the dominance rule. That is far above the usual
share of left-handed signers. It suggests that some recordings are mirrored (front camera) or that
the left/right block convention differs between signers, which bears directly on (a) and (b).

*Fix:* Report the mirrored fraction per signer and discuss it.

### E. Related work

**24. [minor] The design lineage of the hybrid is not credited.**

*Location:* §2 "The Google ISLR data"; §4 Architectures.

*Problem:* The hybrid closely follows the widely circulated top solutions of the Google ISLR Kaggle
competition: depthwise 1D convolution blocks with kernel 17, interleaved with transformer blocks,
late dropout and EMA. "We found no peer-reviewed description" is a fine reason not to compare
scores, but it does not remove the need to credit the design. Readers will also wonder why the
accuracies reported on Kaggle's hidden test set are much higher.

*Fix:*
- Credit the write-ups as non-archival sources.
- Explain the difference in accuracy: a different test set (is it signer-disjoint?), 32 frames,
  and no lip or face points.

**25. [minor] Work directly relevant to the findings is missing.**

*Location:* §2.

*Problem:* The paper omits work in several areas:

| Area | Suggested citation(s) | Why it matters here |
|---|---|---|
| Sign segmentation | Renz et al., ICASSP 2021; Moryossef et al., Findings of EMNLP 2023 | Segmentation is the main remaining pipeline failure |
| Uncertainty under dataset shift | Ovadia et al., NeurIPS 2019 | Explains the live degradation of abstention |
| Statistical power | Card et al., EMNLP 2020 | Small-n live evaluation |
| Train–serve skew detection | Breck et al., MLSys 2019, "Data Validation for Machine Learning" | Directly relevant to the skew framing |
| Deaf-led critique | Desai et al., 2024, "Systemic Biases in Sign Language AI Research" | Relevant to the ethics framing |
| Holm's procedure | Holm (1979) | Currently cited to Dror et al. 2017 |

Signer adaptation and personalization are also an obvious alternative to "more signers" and should
be mentioned.

*Fix:* Add these citations where they apply.

### F. Tables and figures

**26. [minor] The Figure 1 (accuracy–latency) caption is factually wrong.**

*Location:* Figure 1 caption, §4.

*Problem:* The caption says "The shipped hybrid (fp16) is the most accurate model within the
latency budget." The plotted fp16 ensemble (71.0%, p95 41.5 ms) is inside the 60 ms budget; it is
excluded by the 10 MB size cap, not by latency. Also, the axis label says "per segment" while the
caption says "per call".

*Fix:* Correct the caption, mark the points that fail the size cap, and use one label.

**27. [minor] Latency numbers for the same model differ without explanation.**

*Location:* Table 1; abstract; §5; `results/model_selection.md`.

*Problem:*

| Source | p50 / p95 (ms) | Variant |
|---|---|---|
| Table 1 | 18.1 / 19.8 | fp32 |
| Model selection (`m.ship`) | 18.1 / 20.1 | fp16 |
| Abstract and §5 | 18.6 / 26.1 | fp16 |
| In-app | 16.5 / 17.0 | p50 values from two runs |
| `results/model_selection.md`, next to tracking | 18.6 / 27.2 | fp16 |

*Fix:* Choose one benchmark for headline numbers, report run-to-run variation, and label fp32 and
fp16 consistently.

**28. [minor] The live-gap tables and figure leave key context out.**

*Location:* Tables 4 and 7; Figure 4.

*Problem:*
- **The two baselines are never distinguished.** The text never explains "Shipped build" versus
  "As shipped (v1 CNN)". The shipped build (`c0_head_build`) is commit 3fbafa3 with MediaPipe and
  the classifier on the main thread, running at **7.3 fps**. "As shipped" runs the current app at
  14 fps. The halved frame rate is a confound worth one sentence.
- **Naming differs between tables.** Table 7 uses code names; Table 4 uses prose.
- **Run counts are wrong in captions.** Captions say "three runs", but the number of runs differs
  by configuration.
- **Figure 4 is incomplete.** It shows only the cumulative chain, so neither the best configuration
  (`x_presence`) nor the test-split reference is visible.

*Fix:*
- Explain the two baselines and the frame-rate difference.
- Use one naming scheme.
- Print the number of runs per row.
- Add `x_presence` and a test-accuracy line to Figure 4.

**29. [nit] Table and figure presentation.**

*Location:* Table persign; Table comparisons; Figure 5.

*Problem:*
- **Table persign.** "(none)" is not defined (it means "no classified segment"), and the values are
  fractions while every other table uses %.
- **Table comparisons.** The first two rows are standalone comparisons and the "+" rows are
  cumulative; nothing separates them.
- **Figure 5.** The pipeline diagram is the paper's overview, yet it is the *last* figure (page 8).
  The README calls it Figure 1.

*Fix:* Define "(none)" and use %, add a rule between the two kinds of rows, and move Figure 5 to §1
or §6.1.

### G. Writing, structure and format

**30. [minor] Structure.**

*Location:* §6.

*Problem:*
- §6.5 "Calibration and abstention" is a test-split analysis placed inside the live-gap section.
- The "Live abstention" paragraph uses the calibration numbers before they are introduced.
- A leftover `% LIVE-COVERAGE` comment remains in `livegap.tex`.

*Fix:* Move calibration to §4 or its own section, put live abstention after it, and remove the
comment.

**31. [minor] Terminology is inconsistent.**

*Location:* throughout.

*Problem:*

| Terms used | Suggested fix |
|---|---|
| "fault" / "fix" / "change" | "discrepancy" for the six items; "fix" for the code change |
| "shipped build" / "as shipped" / "first demo" / "first deployment" / "first version" / "v1" | Pick one term per concept |
| "Holistic" / "training tracker" / "Python + Holistic" | Pick one term |
| "browser trackers" / "Tasks" | Pick one term |
| "sign" / "segment" / "sequence" / "clip" | Use each term consistently; "72.8% of test signs" should read "sequences" |
| "the signer-independent Google ISLR split" | "our signer-independent split": it is the authors' own, not an official split |
| "one factor at a time" | "cumulative" |
| "compact" for a 3.4M-parameter model, about 10× the baseline | Drop "compact" or qualify it |

*Fix:* Apply the suggested replacements throughout.

**32. [minor] The text contradicts itself in two places.**

*Location:* §7 and Limitations; §3 and §4.

*Problem:*
- §7 says "Translate turns continuous signing into a row of glosses", while Limitations says the
  model "does not handle continuous signing".
- §3 says "All model selection … validation only", while §4 describes a test-based shipping rule.

*Fix:* Correct both.

**33. [minor] Format and venue.**

*Location:* `main.tex`; Limitations; Acknowledgments.

*Problem:*
- **Length is fine.** The main text ends on page 8 (within the long-paper limit), and Limitations
  and Ethics follow the conclusion, as ACL requires.
- **The draft is not anonymized.** For ARR, the author block, "Independent Researcher, India",
  "The author is based in India", commit hashes and repository references must be anonymized.
- **Author TODOs remain:** surname, e-mail, the AI-assistance disclosure, the author's ASL
  experience, and the dataset citation and licence.
- **Demo track limit.** If the target is the ACL System Demonstrations track, I believe the limit
  is 6 pages, and the paper would need to centre on the system.
- **Fit for the ACL main track.** The language content is thin: gloss classification plus an
  unevaluated LLM rewrite.

*Fix:*
- Anonymize and complete the TODOs.
- To fit the main track, engage more with sign linguistics, for example by analysing which sign
  parameters (handshape, location, movement, non-manual features) the live failures involve.
  Otherwise, a sign-language workshop or ASSETS may be a better audience.

### H. Ethics and limitations

**34. [major] The ASL experience of the only live signer is an unresolved TODO, but it determines
how the live results can be read.**

*Location:* Limitations ("\authortodo{state your ASL experience}"); §6.

*Problem:* If the author is not a fluent signer, some live errors may be production errors. For
example, "dad"→"donkey" happens in every take although "dad" is 96% on test, and "thankyou" is
never segmented. In that case, "distribution shift" includes "the signer signs differently from the
dataset", which is a different claim from camera or room shift.

*Fix:*
- Fill in the TODO.
- Have a fluent (ideally Deaf) signer rate each clip's production as acceptable or not, and report
  accuracy on the acceptable clips. (needs new human annotation)
- Relate the per-sign failures to sign parameters.

**35. [major] The possible harm of Practice mode is not discussed.**

*Location:* Ethics, "Motivation"; §7.

*Problem:* The ethics statement says Practice mode is "one place where an imperfect recognizer may
still help". With 34–48% live top-1 for the author, Practice would tell a learner that most
*correct* signs are "wrong". For some signs this happens every time (dad, sad, water and hungry are
near 0%). That could discourage learners, or teach them to distort signs to please the model. Given
the stated motivation (hearing parents of Deaf children), this risk matters.

*Fix:*
- Discuss the risk explicitly.
- In Practice mode, show "not sure" instead of "wrong" for low-confidence results.
- Show per-sign reliability in the app.
- Soften "may still help" until the mode has been evaluated with learners.

**36. [minor] "Not a translator" is stated, but the system contradicts it.**

*Location:* Ethics "Overstating translation"; §7; `web/index.html`.

*Problem:*
- The primary mode is named "Translate", both in §7 and in the UI tab and button, and the paper
  refers to "sentence translation".
- The sentence feature sends glosses to an unnamed third-party hosted LLM.
- A fluent English sentence can lend false authority to a wrong gloss sequence.

*Fix:*
- Rename the mode (for example "Recognize" or "Glosses", and "Sentence suggestion") in both the
  paper and the UI.
- Name the LLM provider, model and data-retention policy.
- Discuss the false-authority risk.

**37. [nit] The vocabulary's limits are not stated.**

*Location:* introduction; Limitations.

*Problem:* ASL versus ISL is handled adequately in Limitations. However, the introduction's "anyone
with a webcam" framing overreaches given what the vocabulary is.

*Fix:* Say that the vocabulary is a 250-sign list of early childhood words (the PopSign source;
for example "giraffe", "zipper", "kitty"). This limits the "sign-to-text aid" motivation.

## Overall assessment

This is an honest, unusually well-instrumented engineering case study, with one clear and
practically useful finding: hand-slot assignment between the training and serving trackers is a
silent failure that can erase a 68%-accurate model. The provenance machinery, the pre-specified
analysis, the two-pipeline design and the refusal to tune the segmenter on the live clips are all
to the authors' credit.

The headline narrative, however, says more than 50 clips from one signer and 4 test signers can
support:

- "Six faults" is really one demonstrated fault (issue 1).
- The gap is roughly a third to half closed, not closed (issues 2 and 7).
- "The residual is shift" is not tested, and the residual visibly contains segmentation failures
  (issue 2).
- Several significance claims rest on clips and sequences as the unit. With signs or signers as
  the unit, some of them no longer hold (issues 4 and 5).

Much of this can be fixed without new data: reframe the claims, run sign-level and signer-level
analyses, add a per-signer table, break down the segmenter's failures, report live calibration, and
document every test access. The strongest version would add the pipeline-on-dataset and
oracle-segmentation controls, and a fluent-signer validity check of the clips.

On the ARR scale I would currently rate it **Soundness 2.5/5, Excitement 2.5/5, Overall 2.5
(borderline)**. With the claims recalibrated and the no-new-data analyses added, it would be a solid
workshop paper or Findings-level short paper. With the controls in issue 2 and a second signer, it
would be a stronger main-track candidate.

---

## Response: how each issue was handled

Written by the drafting side after the review; the review above is unchanged. "Fixed" means the
text, tables, figures or analysis scripts now address the issue; "partly" means the part that needs
no new data was done; "not done" items need new experiments, data or annotation and are stated as
limitations in the paper. Every number added is a generated macro checked by `paper_claims.py`.

| # | Status | What changed |
|---|---|---|
| 1 | Fixed | "Six faults" became six *candidate* discrepancies; §6.1 states what the shipped build did for each (labels inverted; already unmirrored; body-ratio aspect over 2 s; index resampling; presence segmentation with its parameters; raw softmax with an 80%-precision threshold), says (b) was not a fault and that `x_aspect_calibrated` shows some correction is needed, not that the shipped one was wrong. Abstract, introduction and conclusion rewritten to match. |
| 2 | Partly | "Most of the collapse was skew" and "the remainder is largely shift" removed. The paper reports the recovered share of the hybrid's drop (generated, per pipeline), says the residual is shared by both live pipelines and includes segmentation (clips with no classified segment are reported) and signer, camera and production effects that the design cannot separate, and no longer claims "needs more signers" as a finding. The two controls (pipeline on test-split landmarks; oracle segmentation) and a signer learning curve are **not done** (new experiments); listed under Limitations ("Controls we did not run"). |
| 3 | Fixed | One standard: aspect removal and the motion segmenter are both reported as large, inconclusive drops (not significant at either unit); the abstract no longer states the segmenter result as a finding. |
| 4 | Fixed | Effect sizes lead. A sign-level sensitivity analysis (two takes = one unit; cluster sign-flip test with Holm; cluster bootstrap intervals) is computed by `scripts/live_gap.py report` and reported in §6.3 and Table livegap_ci, labelled as post hoc; the clip-level analysis is labelled pre-specified. The text states which hand-fix comparisons stay significant at sign level and which do not. "The fix whose effect is significant" removed from the abstract. |
| 5 | Fixed (analysis); k-fold not done | Per-signer accuracy table added (appendix), per-signer range in §4; the recipe effect is described as small and within signer variation; the tests are said to treat sequences as independent and to exclude seed variance. A signer-level mixed model and k-fold signer rotation are **not done** (k-fold needs retraining). |
| 6 | Partly | The grid is called a cumulative ablation; the hybrid step's parameter and training-time ratios (generated) and the bundled recipe are stated; "the architecture accounts for" removed; the features result is restricted to the CNN with the v1 recipe. A size-matched CNN or a hybrid with the v1 recipe is **not done** (new training). |
| 7 | Fixed | Title changed to "Where Live Sign Recognition Breaks: Auditing the Train-Serve Gap of a Landmark-Based ASL Recognizer on a Low-Power Laptop". |
| 8 | Fixed | Abstract: "The first deployment, with the CNN, recognized ..."; "with the hybrid and all fixes". |
| 9 | Fixed | "Within the dataset, the rule transfers to unseen signers"; live ECE, live net score and the share of correct segments the threshold rejects are reported (§6.3), plus test ECE bootstrap intervals and the upper bins' overconfidence after scaling (§4). |
| 10 | Fixed | Paired browser-minus-Python differences with sign-level intervals (final pipeline and presence segmentation) replace "within a few points"; the inference is narrowed to "the training tracker is not what separates these clips"; the per-sign sentence gives generated counts of signs failing in both pipelines and of shared most-frequent confusions. |
| 11 | Fixed | Mirroring: "the old label mapping fails with or without mirroring"; calibration: "does not change a segment's top-1, though averaging over segments can move a clip's". |
| 12 | Partly | fp16's slight CPU slowdown stated; user-perceived latency given as segmenter delay plus measured time (generated); the 60 ms budget justified in §4; "too slow for fluent signing" replaced by "below our 15 fps target and below the frame rate of our recordings". The 10 fps subsampling run is **not done**. |
| 13 | Partly | Sign-level cluster bootstrap intervals and run-to-run SD are computed (`summary.json`: `clip_top1_ci_*_sign`, `run_top1`, `run_top1_sd`); runs per configuration are printed in Table livegap_ci. A mixed-effects model is not fitted. |
| 14 | Fixed | The text states the difference intervals are not adjusted; raw and Holm p-values both appear in Table livegap_ci. |
| 15 | Fixed | §6.2 notes that the families include comparisons null by construction and that error rates are controlled within, not across, families. |
| 16 | Partly | Test-split intervals are said to exclude seed and signer variance. The pre-specified percentile bootstrap is kept; Wilson intervals are not added. |
| 17 | Fixed | §6.2 gives the minimum number of clips that must all move one way to reach significance at the first Holm step (generated) and cites Card et al. (2020); inconclusive is not absent. |
| 18 | Fixed | §3's "all selection on validation" corrected; an appendix table lists every test access from the three logs with commit and dirty flag; §3 states that the v2 design followed the v1 test evaluation and that the model to ship was chosen on test; §4 calls the 10 MB cap a repository convenience that alone excludes the ensemble. Whether test confusions informed the v2 design cannot be established from the record; the paper says the test split was not untouched. |
| 19 | Partly | The analysis plan, harness and report code are committed together with the paper and results. The paper says the plan was written into the configuration before the first run; this cannot be verified from git history. No primary contrast was designated after the fact. |
| 20 | Partly | §6.2 states that the author is both developer and only signer and that the fix order came from the pilot. How the 25 signs were chosen, and the source of the sign forms, is an author TODO in the text. |
| 21 | Fixed | §6.1(e) gives the 30 fps assumption, the threshold derivations, the minimum length and gap bridging, the energy-on-aspect-corrected-coordinates interaction and the presence segmenter's definition. The failure breakdown (clips with no classified segment, against presence segmentation) replaces the "ends signs early" explanation. |
| 22 | Partly | Added: seeded random signer assignment, no dataset timestamps, the shipped decision rule and aspect method, the browser harness (fake camera, real time, one browser per clip, seeded configuration order), hybrid pooling and head, the hosted LLM (provider and model). Participant IDs are in the committed split file. The recording protocol is an author TODO. |
| 23 | Fixed | §3 reports that mirroring is a per-signer property (generated counts above 80% and below 5%; every test signer at most 0.5%) and that the data cannot tell left-handed signers from mirrored recordings. |
| 24 | Partly | §4 says the hybrid resembles convolution-transformer models that circulated in the competition, none with an archival description; no specific solution is credited because none could be verified under the paper's citation rule. |
| 25 | Partly | Added and verified: Renz et al. 2021 (segmentation), Ovadia et al. 2019 (uncertainty under shift), Card et al. 2020 (power), Desai et al. 2024 (Deaf-led critique). Holm (1979) has no DOI in Crossref and JSTOR blocks automated checks, so the procedure stays cited through Dror et al. (2017), which describes it. Breck et al. 2019 and Moryossef et al. 2023 were not added, to stay within 35 references. |
| 26 | Fixed | Figure 1 caption corrected (the ensemble meets the latency budget and fails the size cap; it is drawn hollow); axis and caption both say "per call". |
| 27 | Fixed | §5 explains the three latency measurements (selection benchmark, shipped-file benchmark, in-app per-segment times). |
| 28 | Fixed | The two baselines are explained with their frame rates (generated); Table livegap_ci uses the same prose names as Table livegap and prints runs per row; the live-gap figure adds the presence-segmentation configuration and the hybrid's test top-1 line. |
| 29 | Fixed | Per-sign table in %, with "no segment" defined; a rule separates the standalone and cumulative rows of the comparisons table; the pipeline figure moved to the start of §6. |
| 30 | Fixed | Calibration moved to §4 (test split); live abstention follows the live results; the leftover comment is gone. |
| 31 | Partly | "Discrepancy" and "fix", "shipped build" and "as shipped", "our signer-independent split", "cumulative" and "sequences" are applied in the rewritten sections; "compact" is gone from the abstract and contributions. Not every occurrence was renamed. |
| 32 | Fixed | §7: "turns a sequence of isolated signs into a row of glosses"; the §3/§4 contradiction is removed. |
| 33 | Not changed | The draft is a preprint, so it is not anonymized; author TODOs remain (paper/README.md). Venue choice is left to the author. |
| 34 | Partly | Limitations now say production differences may explain part of the residual and that a fluent, ideally Deaf, signer's rating would separate them. The rating is **not done** (new annotation); the ASL-experience TODO remains. |
| 35 | Fixed (paper) | Ethics discusses that Practice would mark many correct signs wrong, could discourage learners or teach distorted signs, and should show "not sure" and per-sign reliability and be evaluated with learners and Deaf teachers. The app itself was not changed in this task. |
| 36 | Fixed (paper) | The "Translate" naming problem, the hosted model (Groq, llama-3.1-8b-instant), that only glosses are sent, and the false-authority risk are stated in §7 and the ethics statement. Renaming the UI mode is out of scope for this task. |
| 37 | Fixed | The introduction says the vocabulary is everyday and child-oriented and that the data was released with the PopSign partners. |


---

# Second review (2026-11-22): "ManuVox: Auditing the Train–Serve Gap of a Landmark-Based Sign Recognizer on a Laptop"

A fresh reviewer agent that had not seen the drafting reviewed the rebuilt PDF (`paper/main.pdf`),
the LaTeX sources, the macro sources and the result files. It worked read-only and was told not to
read the first review above. Its report follows, lightly condensed (wording kept; locations refer
to the draft it reviewed); the response table is at the end.

## Summary

A measurement study of how a landmark-based recognizer of 250 isolated ASL signs loses accuracy
between its offline test split and a live webcam on a low-power laptop. It trains GRU, CNN,
Transformer and conv–transformer "hybrid" models on a signer-independent split of the Google ISLR
landmarks with paired, Holm-corrected tests (hybrid 68.5% top-1, 18.6 ms in Chrome), then audits
six candidate train–serve discrepancies through a cumulative add-and-undo ablation on 50 webcam
clips from one signer (the author) through two pipelines. A hand-slot assignment bug is the one
large and robust fault; with all changes, clip top-1 is 34% in the browser and 40% in Python.
Further: a cross-dataset test on ASL Citizen through an AI-assisted label mapping (v2: 43.1%), a
second model v3 trained on Google ISLR plus ASL Citizen (92.1% in-domain, +1.0 on Google ISLR,
non-significant live gains), and a few-shot personalization feature, Teach Dex (+16 points with v2
on 25 clips; +5 with v3, not significant). The paper is unusually transparent (every number a
macro traced to a result file, every test access logged, power and null results discussed); its
contribution is narrow: one bug class, one signer, and mostly inconclusive secondary effects.

## Strengths

- Exemplary traceability; every spot-checked number matched its source.
- Honest test-split hygiene: every access logged; selection on test admitted and checked against
  validation.
- A well-designed live ablation: changes added and undone, two pipelines, repeated browser runs in
  seeded random order, failed records kept, sign-level sensitivity analysis next to the
  pre-specified clip-level one.
- An explicit power statement; inconclusive effects mostly labelled as such.
- v3's ASL Citizen score correctly framed as in-domain and as no validation of the mapping.
- Thoughtful ethics and limitations for a non-signer author.

## Issues

### A. Claims vs evidence

- **A1 (major).** The abstract and conclusion set the hybrid's 68.5% against the first deployment's
  3%, but that deployment ran the v1 CNN (62.5%); with no fixes the hybrid gets 16% (browser) and
  10% (Python). Pair 62.5% with 3%. *No new experiment.*
- **A2 (major).** The v3 Teach Dex result comes from a degraded harness yet is presented in the
  abstract, introduction and conclusion as "with v3 the gain is smaller". Classified segments per
  run fell 28 → 23 → 15 (v2: 26–30) and badged segments 13 → 11 → 4; v3's model-alone baseline on
  take 2 is 22.7%, while the reviewer's recomputation of the v3 live-gap browser records gives
  32/40/44% (mean 38.7%) on the same take-2 clips through the same pipeline; for v2 the harnesses
  agree (32% vs 30.7%). A clip with no segment cannot change in either condition, biasing v3's gain
  toward zero, so the v2-vs-v3 contrast cannot be attributed to the model. Re-run with a healthy
  harness, or drop the comparison and report the run as invalid. *New experiment, or no if dropped.*
- **A3 (minor, must fix).** The setup is misdescribed: the "three recorded examples" are replays of
  one recording; per run 2–4 signs got no example, so "every tested sign taught" is false; the
  3.5 s still lead-in is not mentioned. *No.*
- **A4 (major).** "The training tracker is therefore not what separates these clips" and "the rest
  of the gap is shared by both pipelines" read a non-significant difference (browser − Python −6
  points, sign-level interval [−15, 3]) as no effect; the pipelines also differ in real-time frame
  dropping. *No.*
- **A5 (minor).** "Mirroring … changes little (+3 points)" ignores the interval [−7, 13]; "some
  correction is needed" contradicts calling the no-aspect-correction effect inconclusive. *No.*
- **A6 (minor).** "With every fix" includes the motion segmenter, which lowered clip top-1; the
  recovery percentages do not say they start from the hybrid's no-fix baseline. *No.*
- **A7 (minor).** "Its abstention threshold accepts few predictions" (live coverage is 45%); on ASL
  Citizen the threshold kept its precision (79.8% vs 82.0%) and coverage fell because accuracy
  fell; what fails to transfer is ECE and coverage. *No.*
- **A8 (minor).** v3's gains are attributed to ASL Citizen data alone, but v3 also adds webcam
  augmentation and is a validation-selected seed compared with v2's seed-0 file. *No for wording.*
- **A9 (minor).** "Keep the Python and browser pipelines identical" overstates the parity test,
  which checks model inputs from identical landmarks. *No.*
- **A10 (minor).** The 25.3-point ASL Citizen drop mixes signer/recording shift with mapping errors
  and involves no live segmentation; use the exact-name subset (48.6%). *No.*
- **A11 (minor).** Missed analysis: v3 reaches 92.1% on ASL Citizen's unseen signers but 44.7–46%
  on the author's clips, pointing at segmentation or the author's productions; v3's segment-level
  live gains (53.6% vs 40.5% browser, 53.2% vs 42.9% Python) are unreported. *No.*
- **A12 (minor).** "The same CNN reaches 80.7% … (Table 1)": Table 1 has no signer-dependent row,
  and the two numbers come from different test sets. *No.*
- **A13 (minor).** The test split excludes sequences with too few hand frames while a live clip
  without a segment counts as wrong, so "drop from test accuracy" compares unequal denominators.
  *No for the note; one more test access to recompute.*
- **A14 (nit).** Latency reported selectively: v3's lower median p50 is quoted while median p95
  (21.25 vs 31.31 ms) is folded into a range. Report both medians.
- **A15 (nit).** Some per-sign errors may be same-sign label pairs (cat→kitty, sleep→sleepy); in the
  shared-confusion list, car and cat are ties in the browser.

### B. Statistics

- **B1 (minor, close to major).** The four-signer cluster bootstrap interval for v3 − v2 on Google
  ISLR ([0.8, 1.3]) is narrower than the sequence-level one and anti-conservative; report the four
  per-signer differences instead. *No.*
- **B2 (minor).** The v3-vs-v2 test compares two files and ignores training randomness. *No for
  wording; seed means need a new test access.*
- **B3 (minor).** "The recipe's small gain is within signer-to-signer variation" compares a paired
  effect with signer levels; the recipe gain is positive for all four signers. *No.*
- **B4 (nit).** The Holm family includes the foregone in-domain vs cross-domain ASL Citizen
  comparison.
- **B5 (minor).** Clip-level scoring is not what users see; report segment-level accuracy for the
  final pipelines. *No.*
- **B6.** Positive: power statement, Holm step-down, sign-level sensitivity analysis.

### C. Protocol

- **C1 (major, same root as A2).** The Teach Dex harness is unstable and logs too little (segment
  yield and examples recorded decay across runs; no fps). Log fps, restart the browser between
  runs, interleave models, pre-specify an exclusion rule. *New experiment.*
- **C2 (minor).** The live clips are not an independent test for v3 or Teach Dex: they were analysed
  before v3's webcam augmentation was designed, which targets the discrepancies they revealed.
- **C3 (minor).** "The analysis was written into the configuration before the first run" cannot be
  checked: configuration and results first appear in the same commit. Soften. *No.*
- **C4 (nit).** v3's 8 validation signers do not add up to 3 Google ISLR + 6 ASL Citizen; explain
  (and rule out an ID collision). *No.*
- **C5 (minor).** Have a fluent signer check the 65 non-exact mappings; use the exact-name subset as
  the primary cross-dataset number.
- **C6.** Positive: test reuse disclosed; ASL Citizen test read once per model after selection.

### D. Clarity and structure

- **D1 (minor).** "Shipped" is overloaded (first deployment, "as shipped", the v2 file).
- **D2 (minor).** "v2" is overloaded, including "Live-gap v2 in full" in a caption.
- **D3 (minor).** Appendices D (Calibration) and E (Quantization) have no text.
- **D4 (minor).** The introduction cites Section 7 for the golden-fixture test, which is in §6.1.
- **D5 (minor).** Six latency figures for one model; the abstract's 18.6 ms does not name its
  statistic; which clip the 25.2 fps throughput run used is not said.
- **D6–D11 (nits).** The config name "x_presence" leaks; "averaging its three seeds" means an
  ensemble; "none of these changes is significant" vs a starred top-3 cell; quantization ranges
  ordered inconsistently and missing the "full" static variant; "Teach Dex" used before it is
  introduced; "an expected calibration error of 0.066 and 0.066".
- **D12 (minor).** Much evidence sits in the appendix, acceptable as the text quotes key values.

### E. Formatting

- **E1.** Main text within 8 pages; no "??" references.
- **E2.** Empty appendix sections (D3).
- **E3 (nits).** "Muller" should be "Müller"; Starner et al. author list truncated; Fox et al. lacks
  volume and pages; the Kaggle entry prints no URL; Holm (1979) not cited; MediaPipe papers cited as
  preprints.
- **E4 (venue-dependent).** The PDF is not anonymized.
- **E5 (nit).** Figure 1 relies on colour.

### F. Ethics and limitations

- **F1 (minor).** Have a fluent, ideally Deaf, signer rate the 50 clips. *New (human rating).*
- **F2 (minor).** The public site serves Practice mode although the Ethics section says it should
  be evaluated first; state the safeguards or gate it.
- **F3 (minor).** No Deaf involvement at any stage, including the label mapping.
- **F4 (nit).** Say what the language-model provider retains; gloss sequences can reveal content.
- **F5 (nit).** How the 25 test signs were chosen is not stated.

## Overall assessment

Careful and transparent, with the main finding (a hand-slot bug dominates; most of the gap remains,
shared by both pipelines) supported for this one signer. Weakened by the headline pairing (A1), the
degraded v3 Teach Dex comparison (A2/C1) and misdescribed setup (A3), and null-as-no-effect wording
(A4) plus the four-cluster interval (B1). **Score 3/5, confidence 4/5** (a solid case study for
Findings, a workshop or a demo track).

## Response: how each issue was handled (second review)

| # | Status | What changed |
|---|---|---|
| A1 | Fixed | Abstract and conclusion pair the 3% with the first deployment's CNN (62.5%); the hybrid's 68.5% and latency are stated separately. |
| A2 | Fixed (paper); run not repeated | §6.4 reports the v3 run as compromised: its take-2 baseline (generated, 23%) against v3's own 39% on the same clips in the live runs (new generated macro from `results/live_gap_v3/web`), v2's agreeing pair (31%/32%), and the falling segment counts; the gain is said to be biased toward zero and to say nothing about teaching v3. The v2/v3 Teach comparison is gone from the abstract, introduction and conclusion; the README cell is marked compromised and `results/teach_dex_v3/VALIDITY.md` explains why. Repeating the run (idle laptop, fps logged, browser restarted per run, models interleaved) is an open item. |
| A3 | Fixed | §6.4: up to three examples replayed from the one take-1 clip (one production), 2–4 signs per run got none (generated), the still lead-in (generated); "every tested sign taught" replaced by "the test vocabulary equal to the signs offered for teaching"; abstract and introduction say "examples replayed from one recording per sign". |
| A4 | Fixed | "We see no evidence that the training tracker separates these clips … but the intervals allow a sizeable advantage for Holistic, and the pipelines also differ in real-time frame dropping"; conclusion: "with no evidence that the rest of the gap depends on the pipeline". |
| A5 | Fixed | Mirroring gives its interval and "compatible with no effect"; aspect: "Removing aspect correction tended to hurt … so the data do not show that the shipped correction was wrong". |
| A6 | Fixed | Abstract: "all pre-specified fixes", "one fix, a motion segmenter, if anything hurt", and the recovery as the share of "the gap between its unfixed live score and its test accuracy". |
| A7 | Fixed | Abstract/introduction: "its calibration degrades"; §4: the threshold keeps its precision but, as accuracy falls, accepts only 31% of videos. |
| A8 | Fixed (wording) | "v3's added data and augmentation"; §4 says the comparison is between two files, a validation-selected v3 seed and v2's seed 0, and that v3 changes data and augmentation together. No ablation. |
| A9 | Fixed | Conclusion: the code now computes "the same model inputs from the same landmarks". |
| A10 | Fixed | The conclusion no longer uses the ASL Citizen drop as evidence about the live gap; §4 keeps the exact-name (48.6%) vs variant breakdown. |
| A11 | Fixed | §6.3 adds v3's segment-level gains (generated, untested) and the 92.1%-vs-under-half contrast as pointing to segmentation or the signer's productions. |
| A12 | Fixed | "on a signer-dependent split (another test set, so not a paired comparison) … against the 62.5% on unseen signers in Table 1". |
| A13 | Fixed (note) | Limitations: the test split excludes sequences with too few hand frames while a live clip without a segment counts as wrong, so the drop somewhat overstates the live gap. Not recomputed (would need another test access). |
| A14 | Fixed | Both medians (p50 and p95) for v2 and v3, with "no difference beyond round-to-round variation". |
| A15 | Fixed | The shared-confusion count now requires a single most frequent wrong guess in each pipeline (generated; 9 → 6, car, cat and sad drop out); a note says a guess such as sleepy for sleep may be the same sign. |
| B1 | Fixed | `scripts/compare_v3.py` now also writes each signer's difference (`diff_by_cluster`; every other number unchanged); §4 reports v3 − v2 as positive for each test signer (+0.8 to +1.4, generated) instead of the four-cluster interval; the table caption says per-signer differences are in the text. |
| B2 | Fixed (wording) | See A8. Seed means not computed (new test access). |
| B3 | Fixed | "the recipe's small gain is positive for all four, which four signers cannot make significant". |
| B4 | Partly | The comparison stays in the Holm family; the caption says its direction was expected. |
| B5 | Partly | Segment top-1 of the final pipelines (40% browser, 43% Python) is in §6.3. What a clip contains beyond the clip metadata is not recorded. |
| C1 | Open | Needs the repeated run (see A2). |
| C2 | Fixed (Limitations) | The clips were analysed before v3's augmentation was designed, which targets what they revealed; Teach Dex reuses them. No fresh set. |
| C3 | Fixed | "fixed in the configuration before the first run (the repository cannot show this, as configuration and results were committed together)". |
| C4 | Explained | Not a collision (ASL Citizen IDs carry an offset): v3's validation signers are 3 Google ISLR and 5 ASL Citizen; one of ASL Citizen's six validation signers has no kept video of a mapped sign. Appendix C says so (generated from the split file). |
| C5 | Partly | The exact-name breakdown is reported; a fluent signer's check of the mapping is an open item. |
| D1 | Fixed | "Shipped build" now means only the first deployment; v2 is "the deployed v2 model/file" in text and generated captions. |
| D2 | Fixed | Caption "The live study in full". |
| D3 | Fixed | Appendices D and E have text. |
| D4 | Fixed | The introduction points to §6.1. |
| D5 | Partly | The abstract says "(median)"; the throughput run's camera is named (a pilot clip). No consolidated latency table (page limit). |
| D6–D11 | Fixed | "x_presence" removed; "an ensemble of its three seeds"; "none of these top-1 changes"; one generated int8 range over every variant; Teach Dex named in the introduction; the ECE sentence reworded. |
| D12 | Not changed | The text quotes the key values. |
| E1–E2 | OK / Fixed | Main text ends on page 8; see D3. |
| E3 | Partly | "Müller" fixed; the Kaggle URL is printed. Starner et al. keeps "and others" as in the source; Fox et al. keeps its DOI; Holm (1979) is not added (no verifiable record, as in the first review); MediaPipe entries stay as verified preprints. |
| E4 | Not changed | A preprint with the author block the author chose. |
| E5 | Fixed | The caption names the (a)–(f) labels. |
| F1 | Open | Needs a fluent, ideally Deaf, signer. |
| F2 | Fixed (paper) | Ethics: the public site's Practice mode does not yet report "not sure" or per-sign reliability and carries only the general note that ManuVox is not a full translator. The app was not changed in this freeze. |
| F3 | Fixed | Ethics: no Deaf signer was consulted on the label mapping either; consultation should precede further deployment. |
| F4 | Fixed | Ethics: the provider's retention policy has not been audited; a gloss sequence can reveal what a user said. |
| F5 | Fixed | Limitations: how the 25 signs were chosen was not recorded, and a selection of easy-to-learn signs would bias the live estimate. |

# Third review (2026-11-24, IEEE version): "ManuVox: Auditing the Train–Serve Gap of a Landmark-Based Sign Recognizer on a Laptop"

An independent reviewer, with no access to the sources or to the earlier reviews, read the 6-page
IEEE version (`paper/ieee/main_ieee.pdf`) as an IEEE conference reviewer and consulted the
extended version (`paper/main.pdf`) only where the short paper points to it. The review is
reproduced unchanged.

Reviewed file: the 6-page IEEE version (main_ieee.pdf). The extended version cited as [8] was checked only to see whether material missing from the 6-page paper is available there.

## 1. Summary of the paper

The paper studies why a landmark-based isolated-sign recognizer for 250 ASL signs, trained on MediaPipe Holistic landmarks from the Google ISLR Kaggle data, collapses when run live in a browser. On the author's signer-independent split of that data, the first deployed model (a depthwise-separable temporal CNN) scores 62.5% test top-1, but it got 3% of 50 webcam clips right. The author trains GRU, Transformer, CNN and a Conformer-like convolution–transformer "hybrid" (68.5% test top-1, 18.6 ms median per call in Chrome on an i3-1115G4). The paper lists six candidate train–serve discrepancies: hand-slot assignment, mirroring, aspect ratio, frame-rate resampling, segmentation, and calibration/abstention. It fixes them through a shared feature specification and a golden-fixture CI test, and ablates the fixes on 50 clips (25 signs × 2 takes) recorded by the author. The clips go through two pipelines: legacy Holistic in Python, and MediaPipe Tasks in Chrome. Only the hand-slot fix has a clearly significant effect (+36 / +25 points). With all fixes, clip top-1 reaches 40% (Python) and 34% (browser), which is 34–51% of the drop from test accuracy. The pre-specified motion segmenter appears to hurt. The paper also reports four further results. (i) A cross-dataset test on ASL Citizen gives 43.1% on 236 mapped signs, with degraded calibration. (ii) A model "v3", also trained on ASL Citizen with webcam-style augmentation, reaches 92.1% in-domain, and its live gains are not significant. (iii) "Teach Dex" adds a prototype-style personalization term that raises one signer's take-2 clip top-1 from 31% to 47%. (iv) The paper closes with limitations and ethics.

## 2. Strengths

- **Relevant problem.** The paper tackles a practically important problem that is rarely reported: the gap between benchmark accuracy and live accuracy in sign recognition. It also measures that gap quantitatively instead of treating it anecdotally.
- **Two-sided ablation design.** Table II has a cumulative block (add one fix per row) and a leave-one-out block (undo one fix of the final pipeline). It is run through two trackers (Holistic, the training tracker, and Tasks, the serving tracker). This is a sound way to attribute effects.
- **Statistical care above the norm for this kind of paper.** The analysis is described as pre-specified, uses paired sign-flip tests, applies Holm correction within each pipeline and metric, and adds a sign-level sensitivity analysis for the non-independence of the two takes. It states the minimum detectable effect explicitly ("significance at the first Holm step needs at least 10 clips ... in the browser and 9 in Python") and reports bootstrap intervals. Inconclusive and null results are labeled as such, not dressed up.
- **Candour about weaknesses.** The paper reports problems a less careful author would have left out:
  - the test split was reused during development ("the model to ship was chosen by test accuracy"), with a validation cross-check;
  - the hybrid's gain is confounded with an 8.6× parameter increase;
  - the pre-specification cannot be verified from the repository;
  - the v3 Teach Dex run was compromised;
  - the ASL Citizen label mapping is AI-assisted and unverified.
- **A reusable engineering remedy.** One feature specification read by both stacks, plus a golden-fixture parity test in CI with a 10^-4 tolerance, is a concrete practice other deployed-ML projects can adopt.
- **Realistic efficiency evaluation.** Latency is measured on genuinely low-end hardware and covers threads, WebGPU, fp16 and int8. It includes the useful negative finding that dynamic int8 slows depthwise-convolution models.
- **Good ethics section.** Section VII is substantive: Deaf community involvement, the risk of the Practice mode misleading learners, the burden-shifting nature of personalization, data release limits. The paper also includes the IEEE-required disclosure of AI-tool use.

## 3. Weaknesses and requested changes

**R1 [major] The live evaluation cannot support general conclusions about the train–serve gap.** (needs new experiments)
*Location:* Sections VI-B and VI-C, Table II, abstract.
- All live evidence comes from 50 clips of 25 signs by one person, who is the developer and "not a fluent signer". How the 25 signs were chosen "was not recorded", and 9 of the 25 signs "fail in every take in both pipelines".
- Section VII concedes that production differences cannot be separated from pipeline effects. This means the paper cannot say how much of the remaining 60–66% error is caused by the system rather than by the signing.
- The headline percentages ("closing 34–51% of the gap") are therefore properties of this signer, camera and sign selection, not of the recognizer.

*Suggestion:*
- Add clips from several signers, ideally fluent or Deaf signers, even a few signs each.
- At minimum, have a fluent signer rate each of the 50 clips as a correct or incorrect production, and report results on the clips rated correct.
- Without either, reframe the title, abstract and contributions explicitly as a single-signer case study.

**R2 [major] Cheap controls that would locate the remaining gap are named but not run.** (needs new experiments)
*Location:* Section VII ("Two controls would locate the remaining gap: running the live pipeline on test-split landmark sequences, and an oracle segmentation of the live clips") and Section VI-C ("What remains").
- Neither control needs new recordings. Oracle segmentation of 50 clips is a few minutes of annotation.
- A third control is available too. ASL Citizen's test split is 3,523 webcam videos from 11 unseen signers covering 236 of the signs. Running those untrimmed videos through both live pipelines (tracker → segmenter → model) would give a multi-signer, webcam, live-like evaluation. It would also measure the Holistic-versus-Tasks difference at a scale where it can be resolved, unlike the current "−6 points [−15, 3]".

*Suggestion:* run at least oracle segmentation and the live-pipeline-on-dataset control. These directly answer the paper's central open question ("The rest of the gap lies in what both pipelines share, above all segmentation, or in the clips themselves").

**R3 [major] The framing as a train–serve skew audit is looser than the evidence, and the one significant finding is an implementation bug.** (needs new experiments for the sensitivity analysis; the rest is text only)
*Location:* Section VI-A, Fig. 1, abstract, conclusion.
- Of the six "candidate discrepancies", only (a), (c) and (d) are parity questions, where serving should reproduce what training did.
- (b) was not present in the shipped build ("The shipped build already detected on the unmirrored frame ... is a counterfactual").
- (e), segmentation, has no training-side counterpart, because the dataset clips are pre-segmented. It is a new serving component, not a skew fix.
- (f) cannot change a segment's top-1. The extended version says so; the 6-page version does not.
- The one effect that survives correction is that the shipped build "mapped that label as if the frame were mirrored ... so every hand went into the wrong block". A reader will see this as a bug whose large effect is unsurprising. The generalizable lesson is thinner than the framing suggests.

*Suggestion:*
- Separate parity discrepancies from serving-only design choices in the text and in Fig. 1.
- Add an offline sensitivity analysis on the 16,242 test sequences. Inject each skew into the test landmarks:
  - swap the hand blocks;
  - apply the portrait-to-landscape aspect distortion;
  - subsample to 15 fps with index-based resampling;
  - crop or extend segment boundaries by realistic amounts.
- Measure the accuracy drop for each. This would give high-powered estimates of each discrepancy's cost, which the 50-clip study cannot provide (most of its effects are "large but inconclusive").

**R4 [major] Baselines are weak, and the model-comparison contribution is hard to place relative to prior work.** (needs new experiments)
*Location:* Section II ("we compare only against baselines trained under our own protocol"), Section IV, Table I, contribution bullet 1.
- No published pose-based isolated sign recognition model is trained under the author's split. The paper cites SPOTER [12] and the OpenHands pose models [13] but does not evaluate them.
- Public write-ups and code of the top Kaggle ISLR entries reportedly use 1D-convolution + transformer models similar to the hybrid, and report much higher accuracies on the competition's hidden set. They are not peer-reviewed and use a different split, but the released code could be retrained under this protocol.
- As it stands, 68.5% cannot be judged as strong or weak for signer-independent Google ISLR.
- The paper also admits that the hybrid's +4.9 points come with "8.6" times the parameters and "3.3" times the training time, "so its gain cannot be credited to the architecture alone".

*Suggestion:*
- Add at least one external pose-based baseline, for example SPOTER, or an ST-GCN/SL-GCN from OpenHands.
- Add one capacity-matched control: a wider CNN with about 3.4M parameters and the same training budget.
- Briefly acknowledge the reported Kaggle accuracies and why they are not comparable.

**R5 [major] The "gap closed" percentages are measured against the wrong ceiling.** (needs new experiments; re-analysis of saved predictions only)
*Location:* abstract ("closing 34–51% of the gap"), Section VI-C ("of the drop from its 68.5% test top-1").
The ceiling is the hybrid's mean test top-1 over all 250 signs, and it has three problems:
- (i) It is optimistically biased, because the test split guided the v2 design and the model choice (Section III).
- (ii) It averages over 250 signs, while the live clips cover a non-random subset of 25, and per-sign accuracy varies widely.
- (iii) Test sequences with fewer than 3 hand frames were removed (6,945 sequences), while a live clip "without one [segment] counts as wrong". The extended version itself notes that this asymmetry "somewhat overstates the live gap"; the 6-page version omits it.

*Suggestion:*
- Report the ceiling as the hybrid's validation and test accuracy on the same 25 signs.
- State the filtering asymmetry.
- Recompute the 34–51% figure on that basis.

**R6 [minor] Statistical details in Sections IV and VI are inconsistent or underspecified.** (text/presentation only)
- The power statement says significance in the browser "needs at least 10 clips (20 points)". But Table II marks the browser "+ v2 hybrid model" top-1 change (1 → 16, +15 points) as significant. Fractional per-clip scores (means over 3 runs) make the "20 points" statement wrong for the browser. Rephrase it in terms of discordant clips.
- "+25 in the browser (19 clips better, 4 worse)" will puzzle readers, because a net of 15 clips is 30 points. Explain that per-clip browser scores are fractional.
- State the Holm family sizes. They can be inferred as about 12 tests in Python and 15 in the browser, but the paper should give them.
- Give the family for Teach Dex ("Holm p = 0.031").
- Say whether intervals are 95% and whether they resample clips or signs. "Signer interval [39.3, 46.6]" is never defined.
- In Section IV, the paired tests over 16,242 sequences "treat the sequences as independent". With 4 test signers, signer-level evidence is n = 4, which the paper concedes. Even so, contribution bullet 1 presents "+6.1 points ... for each of the 4 test signers" as a finding. Add the per-signer CNN range, or a signer-level interval, next to it.
- The v3-versus-v2 comparison "+1.0 points [0.5, 1.5] (Holm p < 0.001)" compares two single files with a sequence-level test. The paper says so, but the abstract repeats "+1.0 points" without the caveat.

**R7 [minor] "Browser minus Python" confounds the tracker with real-time frame dropping and implementation.** (needs new experiments for the control; otherwise text only)
*Location:* Section VI-B ("Browser + Tasks ... in real time, so frames drop") and VI-C ("What remains").
- Python + Holistic processes every frame; the browser drops frames in real time. The −6 point difference therefore cannot be attributed to the tracker, which is how the sentence "We see no evidence that the training tracker separates these clips from the test split" reads.
- The paper also never establishes that the legacy Holistic used here has the same version and settings as the one that produced the Kaggle data (model complexity, smoothing).

*Suggestion:* run the browser pipeline offline on every frame to separate the tracker from frame dropping. Alternatively, state the confound explicitly and document the Holistic configuration.

**R8 [minor] Live calibration and abstention are a listed "fix" and an index term, but no live result is reported.** (text/presentation only)
*Location:* Section VI-A(f), Table II "+ calibration (f): final", Index Terms.
- Clip top-1 ignores the "not sure" threshold, so Table II says nothing about whether (f) works live.
- The extended version reports that live calibration fails (browser ECE 0.100, with the threshold accepting 45% of segments at 67% selective accuracy). That is a central, user-facing result for an "audit" paper.
- Clip-level scoring by "the mean probability vector of its classified segments" also hides spurious extra segments, which a user would see as extra words.

*Suggestion:*
- Add one or two sentences on live ECE, accepted fraction and selective accuracy.
- Report the mean number of segments per clip.
- Explain in the Table II caption how abstention is treated.

**R9 [minor] Latency numbers are inconsistent, and their precision exceeds their repeatability.** (needs new experiments: repeated timing runs only)
*Location:* Table I, Section V.
- The hybrid's browser p95 is given as 19.8 ms (Table I, fp32), 26.1 ms (Section V, fp16) and 21.2 ms (median p95 in interleaved rounds).
- v3, with "the same architecture and file size", has a median p95 of 31.3 ms, about 50% higher. Yet the text concludes "we see no difference beyond round-to-round variation". If rounds vary by ±50%, the 0.1-ms-precision values in Table I, and the rankings built on them, are not meaningful.
- The fp16 p95 (26.1) exceeds the fp32 p95 (19.8), which is not discussed.
- "Hybrid p50 21.96 vs. 4.10 ms" does not say which runtime (4.10 ms is clearly not onnxruntime-web, where the same model takes 18.6 ms), and uses a different decimal precision.
- "Post-training quantization of the v1 models ... halves every file" is followed by hybrid (v2) numbers.
- The 6-page version cannot explain how "tracking ran at 25.2 frames per second" when the recorded clips are "about 15 frames per second". The extended version says a pilot clip was used.

*Suggestion:*
- Report every latency as median and spread over N rounds, with the number of calls, warm-up, runtime and precision named.
- Fix the scope of the quantization sentence.

**R10 [minor] Teach Dex is under-specified and tangential, and the compromised v3 run adds nothing.** (text/presentation only, or new experiments to repeat the v3 run)
*Location:* Section VI-D.
- The method is one sentence. What is the "segment's embedding"? How are up to 3 examples combined (mean, max)? What are the fitted weight and offset, and how was the fit done?
- The evaluation is record-then-test from one sitting, and the examples "repeat one production". The paper candidly calls this the best case.
- Reporting a run that "says nothing about teaching v3" uses scarce space in a 6-page paper.

*Suggestion:* either move Teach Dex to the extended version, or give the method in two or three sentences and drop the v3 paragraph until the harness is fixed and the run repeated. The space saved could hold R2 or R3.

**R11 [minor] The ASL Citizen label mapping is unverified, and the cross-dataset number depends on it.** (needs new experiments; re-analysis only)
*Location:* Section III ("We chose the variants and synonyms with AI assistance ... no signer has confirmed them") and Section IV.
- 65 of 236 labels (35 variants, 30 synonyms) are mapped by judgement. Mapping errors count directly against v2's 43.1%.
- ASL-LEX is used as the gloss authority but is never cited.

*Suggestion:*
- Report v2 accuracy separately for the 171 name-matched and the 65 judgement-mapped signs.
- Have a signer verify the 65 mappings.
- Cite ASL-LEX.
- Define the "signer interval".

**R12 [minor] The v3 interpretation is speculative.** (text/presentation only)
*Location:* Section VI-C, last paragraph ("v3 recognizes 92.1% ... but under half of these clips, which points to segmentation or this signer's productions rather than to webcam conditions as such").
- The 92.1% is in-domain: ASL Citizen's test signers share its recording protocol, and v3 was trained on its training signers. It does not show that v3 handles webcam conditions in general.
- The trimmed-versus-untrimmed difference also confounds segmentation with the domain.

*Suggestion:* soften to "is consistent with", or test it directly with the R2 control (untrimmed ASL Citizen videos through the live pipeline).

**R13 [minor] The conclusion contradicts Section VI-C.** (text/presentation only)
*Location:* Section VIII.
- Section VIII says the hybrid reaches 34%/40% "with no evidence that the rest of the gap depends on the pipeline".
- Section VI-C says "The rest of the gap lies in what both pipelines share, above all segmentation". Table II shows the presence-segmenter variant at 48/49 against 40/34 for the final pipeline. Segmentation is part of the pipeline.
- The intended meaning is presumably "depends on which pipeline/tracker (browser vs. Python)". Rephrase.
- "One had a large and consistent effect" also overstates the browser result, since adding the fix is not significant at sign level (p = 0.152).

**R14 [minor] Pre-specification and blinding are weaker than the wording suggests.** (text/presentation only)
*Location:* Section VI-B ("Nothing was tuned on these clips ... the analysis was fixed in the configuration before the first run (the repository cannot show this ...)"; "a pilot with the same person motivated the order of the fixes").
- The fixes themselves were identified with live data from the same signer and setup.
- The pre-specification is asserted but unverifiable.

*Suggestion:*
- State which discrepancies were discovered from the pilot.
- Soften "nothing was tuned" to "no threshold or constant was fitted on these clips".
- For future work, recommend time-stamped preregistration.

**R15 [minor] Segmenter behaviour and the deployed configuration are unclear.** (text/presentation only)
*Location:* Section VI-A(e), Section VI-C.
- The motion segmenter's thresholds "come from the validation sequences under an assumed frame rate", and failure is attributed to frames possibly not being "30 fps apart". If speeds are computed on capture timestamps live, it is unclear why the dataset frame-rate assumption would mis-scale them. State the units (per frame or per second) on both sides.
- The paper keeps the motion segmenter "as specified" for analysis, which is right. It does not say which segmenter the public app now uses. Users would be better served by the one that scored higher, and the reader should know.

**R16 [presentation] Model, feature and recipe names are overloaded.** (text/presentation only)
*Location:* throughout.
- "v1/v2" denotes feature sets, recipes, and models. "v2" also means the deployed fp16 seed-0 hybrid file (68.4%), as distinct from the hybrid's seed mean (68.5%).
- The abstract uses "v2" without ever defining it ("from 31% to 47% with v2").
- "Shipped build" (3%) and "As shipped" (1% browser) differ, and the 6-page version never explains why. The extended version says the shipped build ran inference on the main thread at 7 fps.

*Suggestion:* introduce explicit model identifiers once (for example a short list or a column in Table I). Explain the two baselines in one sentence.

**R17 [presentation] Fig. 1 under-serves the paper's thesis.** (text/presentation only)
*Location:* Fig. 1, page 3.
- It shows only the browser path, so the reader cannot see where training and serving diverge.
- It omits the normalization steps (shoulder centering and scaling, dominant-hand mirroring), which interact with hand-slot errors.
- It lists (e) before (d).
- It relies on light-orange fill, which is barely distinguishable from grey when printed in grayscale.
- Its text is 7 pt.

*Suggestion:* draw the training path (Holistic → spec → model) and the serving path (Tasks → spec → segmenter → model → threshold) side by side, mark each discrepancy where the paths differ, and use 8 pt or larger text plus a non-colour cue.

**R18 [presentation] Table II needs more support in the 6-page version.** (text/presentation only)
*Location:* Table II, page 4.
- The "–" entries are not explained.
- Only significance stars are given; top-1 intervals, p-values and sign-level results are deferred to [8].
- Because the paper's argument rests on which effects are inconclusive, at least the top-1 intervals, or the counts of better/worse clips, should be in the table.
- The row "final, mirrored + old mapping" (7/11) is never discussed in the 6-page text. A reader would expect the old label mapping to be correct on a mirrored frame (see Q3).

**R19 [minor] Derived numbers do not match the tables.** (text/presentation only)
*Location:* contribution bullet 1 and Section IV.
- "improve test top-1 by +6.1 points over a CNN": Table I gives 68.5 − 62.5 = 6.0.
- "the hybrid +4.9": Table I gives 68.5 − 63.7 = 4.8.
- "multiplies the parameter count by 8.6": Table I gives 3.40 / 0.40 = 8.5.
- These probably come from unrounded values, but a reader checking the table finds mismatches. Compute from the displayed values or add a decimal.

**R20 [minor] Citations are missing.** (text/presentation only)
- Holm's procedure (Holm, 1979) is used throughout but never cited.
- ASL-LEX is used as the gloss authority but not cited.
- MediaPipe Holistic, the training tracker, and onnxruntime(-web) have no references.
- The related work omits prior studies of cross-dataset or in-the-wild degradation in sign recognition, and any mention of public Kaggle ISLR solutions (see R4).

**R21 [minor] Reproducibility and archival status.** (text/presentation only)
- The extended version [8] is a GitHub repository URL, which is not archival. Deposit it on arXiv or Zenodo with a DOI.
- The 6-page paper has no explicit code-availability statement.
- The live clips cannot be released, which is understandable, so the code and the exact configuration matter more. State what is released.

## 4. IEEE format compliance

- **Page count and size.** 6 pages, US Letter (612 × 792 pt), two columns, references included. This meets a 6-page limit including references.
  - The text block spans about x 48–565 pt and y 51–719 pt on every page. This is consistent with the IEEEtran conference layout, with no overflow into margins.
  - No page numbers, headers or footers, which is correct.
  - No copyright notice is present. That is acceptable for submission but needed for camera-ready.
- **Author block.** One author with affiliation and email. IEEE's template asks for organization, city and country; "Independent Researcher, India" gives no city. This is minor. Check whether the venue is double-blind: the named author block, the self-citation [8] and the GitHub URL containing the author's username would all violate anonymity.
- **Abstract.** Bold, about 246 words, which is at the top of IEEE's usual 150–250-word range. It contains undefined abbreviations ("ISLR", "CNN") and an undefined model name ("v2"). It is dense with numbers (13 distinct figures), which makes it hard to read. Cut it to the core claim and two or three numbers.
- **Index Terms.** Present and correctly formatted. "Train–serve skew" is not an IEEE Taxonomy term; consider adding taxonomy terms (for example "gesture recognition", "edge computing").
- **Section headings.** Roman numerals in small caps, lettered italic subsections, and italic run-in paragraph headings are all correct IEEE style. The run-in heading "Aspect correction and segmentation: large but inconclusive:" has a double colon. The acknowledgment is unnumbered and uses the American spelling "Acknowledgment", which is correct.
- **Tables and figure.** Table titles are above the tables ("TABLE I" / small-caps title), with explanatory notes below, which is acceptable. The figure caption is below the figure ("Fig. 1."), which is correct.
  - Fig. 1 is placed in Section V but first referenced in Section VI. Prefer placing it after its first reference.
  - The figure is vector, with 7-pt text; IEEE recommends 8–10 pt.
  - The figure relies on colour.
- **References.** IEEE numeric style, numbered in order of first citation, which is correct. Problems:
  - [4] gives the URL twice (in the note and after "[Online]. Available:"), and its title "Google - Isolated Sign Language Recognition" uses a hyphen instead of a dash.
  - [5] and [6] carry long proceedings hash URLs. In [5] this produces badly stretched inter-word spacing ("T. Starner et al., "PopSign ASL v1.0: An isolated" with large gaps). Drop the URLs for proceedings papers.
  - [8] is a non-archival repository link.
  - [34] lacks volume and pages.
  - [30] is usually dated 1993.
  - Page numbers are given for some proceedings papers and not others.
  - IEEE style normally abbreviates "Proceedings of the" to "in Proc."; this is optional but inconsistent with IEEE practice.
- **PDF checks (PDF eXpress-type).** PDF 1.5 produced by xdvipdfmx (XeLaTeX). All fonts are embedded subsets: Nimbus Roman (Times-compatible), CM math, Nimbus Mono. There are no raster images. Nothing obvious would fail PDF eXpress, but run the PDF through PDF eXpress anyway, because XeLaTeX output with Nimbus fonts sometimes triggers warnings.
- **Layout.**
  - The columns on the last page are not balanced: the left column is full, the right ends mid-page. IEEE asks for balanced final columns.
  - Page 5 left column has visibly stretched vertical space before "D. Personalization (Teach Dex)".
  - The dense single-paragraph style, with almost no displayed equations or lists after page 1, makes the paper read like an extended abstract of [8].
- **Language conventions.** IEEE uses American English, but the paper mixes spellings: "licence", "favour", "analysed" alongside "normalized", "recognizer", "center".
- **Voice.** A single author writes "we/our" throughout, refers to "the author" in Sections VI–VII, and writes "I" in the acknowledgment. Pick one convention.
- **AI disclosure.** The acknowledgment discloses AI-tool use, as current IEEE policy requires. This is compliant.

## 5. Writing quality

**Unclear sentences.**
- "We see no evidence that the training tracker separates these clips from the test split" (VI-C). It is unclear what "separates" means. Suggest: "The tracker difference does not account for a measurable part of the gap."
- "The baselines are the shipped build itself and as shipped, the current app with the shipped build's settings." This is hard to parse, and the difference between the two baselines is never explained.
- "Of six candidate discrepancies, assigning hands to slots as the training tracker did matters by far the most" (abstract). "Slots" is jargon at that point.
- "One fix, a motion segmenter, if anything hurt" (abstract). Colloquial, and it calls a component that was never a parity fix a "fix" (R3).
- "the first deployment's CNN reaches 62.5% top-1, yet it recognized 3% of 50 new webcam clips". The tense switches mid-sentence. Also, 3% of 50 clips is 1.5 clips; say that the value is a mean over three browser runs.
- "A second training set (v3)" is a run-in heading for a model.
- "9 of the 25 signs fail in every take ..." starts a sentence with a numeral.

**Inconsistent terms.**
- "golden parity test" (abstract, conclusion) vs. "golden-fixture test" (introduction, VI-A).
- "with the training tracker" vs. "Python + Holistic" vs. "in Python".
- "hand slots" vs. "hand assignment" vs. "blocks".
- "fix" is used for counterfactuals and new components alike.
- "Holistic layout" in Fig. 1 is never defined.

**Undefined abbreviations and symbols.**
- ISLR is never expanded. GRU, CNN, ONNX, fp16/fp32/int8 and p50/p95 are undefined. These are common in the field, but IEEE expects first-use expansion.
- "wasm SIMD" appears in the Table I caption before WebAssembly is introduced in Section V.
- r_train in "k = r_train/(w/h)" is not explicitly bound to "the training data's width-to-height ratio", and the direction of the rescaling (multiply or divide y) is not stated.
- "signer interval", "Gloss row" (Fig. 1), "Teach panel", "Practice mode" and the sentence-generation feature ("only the recognized glosses are sent to a third-party language model provider") first appear in Sections VI-D and VII without being introduced. Practice mode and sentence generation should be described briefly in the system description or removed from the ethics text.
- "Face and lip points ... are not used" sits uneasily with distances "to the nose, mouth and shoulders". Say that the mouth comes from pose landmarks.

**Numbers that look inconsistent (quoted).**
- "+6.1 points over a CNN" vs. Table I 62.5 → 68.5 (6.0).
- "the hybrid +4.9" vs. Table I 63.7 → 68.5 (4.8).
- "multiplies the parameter count by 8.6" vs. 3.40/0.40 (8.5).
- "A convolution–transformer hybrid reaches 68.5%" (abstract) vs. "68.4% for the deployed v2 file" (Section IV), which is the base for "25.3 points below its own test split". Explained, but two numbers for one model invite confusion.
- "needs at least 10 clips (20 points) ... in the browser" vs. Table II browser "+ v2 hybrid model" 1 → 16\* (+15 points, marked significant).
- "+25 in the browser (19 clips better, 4 worse)": a net of 15 clips would be 30 points.
- Hybrid browser latency: "19.8" (Table I, p95), "p95 26.1" (Section V), "21.2" (median p95 over rounds); and v3 "31.3" with identical architecture.
- "hybrid p50 21.96 vs. 4.10 ms" has an unstated runtime and two decimals where the rest of the paper uses one.
- "replacing presence segmentation by the motion segmenter cost −10 and −13" uses the upper block of Table II. The lower block gives −8 and −15 (48→40, 49→34). Say which comparison is meant.

**Claims stronger than the evidence.**
- Abstract, first sentence: recognizers "fail on a live webcam, because the serving pipeline does not reproduce the training features". The paper's own result is that most of the gap is not explained by the pipeline fixes.
- "one had a large and consistent effect" (conclusion). Adding the fix is not significant at sign level in the browser (p = 0.152).
- "now make Python and the browser compute the same model inputs from the same landmarks" (conclusion). The parity test covers 20 validation sequences; say "agree to 10^-4 on the 20 fixture sequences".
- "which points to segmentation or this signer's productions rather than to webcam conditions as such" (VI-C). This is speculative (R12).
- "MediaPipe keypoints [3] are small, appearance-free". Landmarks still encode body proportions and hand size; "largely appearance-free" would be accurate.
- "it beats the CNN for all four test signers" is presented in contribution bullet 1 without Section IV's own caveat ("weak evidence about new ones").

## 6. Questions for the author

1. What is the ceiling the 34–51% figure should be measured against: what are the hybrid's validation and test top-1 restricted to the 25 live signs?
2. Why do "Shipped build" (browser 3/9) and "As shipped (v1 CNN)" (browser 1/7) differ, if the latter is the current app with the shipped settings?
3. The shipped bug mapped handedness "as if the frame were mirrored". Why then does "final, mirrored + old mapping" also score only 7%? On a mirrored frame the old mapping should place hands in the correct blocks. Does the frame mirroring also flip x-coordinates, and how is that undone?
4. How does the dominant-hand mirroring rule ("a sequence whose left hand is present in more frames than its right is mirrored") interact with swapped hand blocks? For a one-handed sign, does the bug effectively present a mirror-image hand rather than a hand in the wrong slot?
5. In the Python + Holistic pipeline, Holistic already assigns hands by pose. How is "As shipped" / "+ hand assignment (a)" realized there? Is the old mapping simulated by swapping blocks?
6. Is the legacy Holistic used here the same version and configuration as the one that produced the Google ISLR landmarks? What evidence supports that?
7. Which segmenter does the publicly deployed app use now, and do you plan to change it given Table II?
8. In what units does the motion segmenter compute speed? Why would the dataset's frame-rate assumption mis-scale speeds if live speeds use capture timestamps?
9. How were the 25 signs chosen, and did any of them come from the pilot? What exactly did the pilot reveal, and which discrepancies were discovered from it?
10. How is a below-threshold ("not sure") segment treated in clip top-1? What are the live ECE, accepted fraction and selective accuracy?
11. Have you run untrimmed ASL Citizen test videos through either live pipeline? If not, is there a reason it is infeasible?
12. Why is the fp16 hybrid's p95 (26.1 ms) higher than the fp32 file's (19.8 ms)? Does onnxruntime-web's WebAssembly backend execute fp16 natively or upcast it? How many calls and rounds underlie each latency figure?
13. For Teach Dex: which layer gives the embedding, how are up to three examples combined, what are the fitted weight and offset, and what is the Holm family for p = 0.031?
14. What accuracy do published ASL Citizen baselines reach on the same 236-sign subset, to put v3's 92.1% in context?
15. Is the venue single- or double-blind? If double-blind, the self-citation [8] and its URL need to be anonymized.

## 7. Overall recommendation

**Recommendation:** Weak reject.
**Reviewer confidence:** 4/5.

**Justification:** The paper is unusually honest and statistically careful, and the shared-specification/parity-test practice is useful. However, its central empirical claims rest on 50 clips from one non-fluent signer who is also the developer. The only conclusive finding is a hand-slot bug whose large effect is unsurprising, and most other effects, including v3 and the segmenter, are explicitly inconclusive. Several cheap controls the paper itself names are not run: oracle segmentation, the live pipeline on dataset sequences or untrimmed ASL Citizen videos, and offline skew injection. Adding them, plus an external baseline and a tighter scope (moving Teach Dex and the compromised v3 run to the extended version), would make this a solid applied paper.

## Response: how each issue was handled (third review)

Fixes that needed no new experiment were made in both versions where the text is shared or
parallel; every number is still a generated macro (claims audit: 0 failures). Item numbers refer to
the review above, which read the IEEE PDF before these fixes.

| Item | Status | What changed |
|---|---|---|
| R1 | Open (needs new recordings) | The single-signer scope is stated in the abstract ("from one signer"), the contributions and Section VII. Recordings from other, ideally Deaf, signers and a fluent signer's rating of the 50 clips remain open items (`paper/README.md`). |
| R2 | Open (needs new runs) | Oracle segmentation and the live pipeline on test-split sequences are named as not run. Running the untrimmed ASL Citizen test videos through the live pipelines would be another access to that test split, which this project reads once per model. |
| R3 | Partly | Both versions now say that (a), (c) and (d) are parity questions, (b) is a counterfactual, (e) has no counterpart in training (its sequences each hold one sign) and (f) cannot change a segment's top-1. Skew injection into the test sequences is not run (a new experiment and another test-split access). |
| R4 | Open (needs new training) | External baselines and a capacity-matched CNN need new runs. Kaggle leaderboard accuracies are not quoted: no peer-reviewed source and a hidden test set, as the paper says. |
| R5 | Partly | The IEEE version now states the filtering asymmetry (the test split drops sequences with too few hand frames, a clip without a segment counts as wrong), as the full version already did. Recomputing the recovered share against a 25-sign ceiling would change a reported number and is not done. The bias from test reuse is stated in Section III. |
| R6 | Fixed (text) | The power statement counts clips, with "20 points if each moves by a whole clip"; browser clip scores move in thirds; the Holm family sizes (12 comparisons in Python, 14 in the browser) come from generated macros; intervals are named as 95% percentile bootstrap intervals over clips; the ASL Citizen "signer interval" is defined as a 95% bootstrap interval over signers; contribution 1 adds "(weak evidence about new signers)". The Teach Dex Holm family is not added (space). |
| R7 | Fixed (text) | "What remains" again says that the pipelines also differ in real-time frame dropping (the first IEEE draft had dropped it) and now reads "no evidence that the training tracker does better on these clips". An offline every-frame browser run is not done; the Holistic configuration is not documented in the paper. |
| R8 | Fixed (text) | The IEEE text reports live ECE, the accepted fraction and selective accuracy; the text and the live-gap table caption say that clip scores do not use the abstention threshold. Segments per clip are not added (a new number). |
| R9 | Partly | The quantization sentence names the runtime (Python onnxruntime) and the precisions compared (dynamic int8 against fp32); the IEEE text says the selection benchmark behind Table I differs from the latency run by a few milliseconds and that the throughput run used a pilot recording, not one of the 50 clips. Repeated timing runs are not done. |
| R10 | Fixed | The IEEE version states Teach Dex's method (the highest cosine similarity to the taught sign's examples in the model's penultimate-layer embedding, added with the fitted weight and offset, both given). The compromised v3 run is one sentence pointing to the extended version, where its details moved to Appendix C. |
| R11 | Partly | v2's accuracy on exact and on variant mappings is in the IEEE text. ASL-LEX is not cited: each version already cites the 35 references the claims audit allows (open item). A signer's check of the mapping remains open. |
| R12 | Fixed (text) | "which points to segmentation or this signer's productions" became "That is consistent with segmentation or this signer's productions, more than webcam conditions as such, limiting the live score, but an in-domain score cannot settle it" (both versions). |
| R13 | Fixed (text) | Conclusion: "one had a large effect in both pipelines"; "no evidence that the rest of the gap depends on which of the two pipelines runs"; the parity test now "checks" that Python and the browser compute the same inputs. |
| R14 | Partly | "Nothing was tuned on these clips" became "No threshold or constant was fitted on these clips". Which discrepancies the pilot revealed is not recorded beyond "motivated the fixes and their order". |
| R15 | Open | The segmenter's speed units and the segmenter of the public app are not stated; they need a check against the code and a decision on the deployed segmenter. |
| R16 | Partly | The abstract names the hybrid "v2" and expands ISLR; the IEEE text says that the shipped build ran inference on the page's main thread, with its frame rate. Model identifiers are otherwise unchanged. |
| R17 | Partly | The audited boxes have thick borders (a cue that survives grayscale) and the caption defines "Holistic layout". Redrawing the training and serving paths side by side is not done (space); the figure text stays at 7 pt. |
| R18 | Partly | The table caption explains the dashes (configurations that exist only in the browser); the text discusses the old mapping with and without mirroring. Intervals and better/worse counts stay in the extended version's table (space). |
| R19 | Fixed (text) | The IEEE text says the paired differences are computed before rounding. |
| R20 | Open | Holm (1979), ASL-LEX, MediaPipe Holistic and onnxruntime are not cited: both versions are at the 35-reference cap, and the earlier reviews found no verifiable record for Holm (1979) in the checked sources. |
| R21 | Partly | Section VII now says that the code, configurations and result files are in the repository of the extended version. The extended-version reference stays the repository address until a preprint identifier exists (open item). |
| Format | Fixed / kept | The run-in heading with a double colon was renamed; American spelling throughout both versions (license, favor, analyzed); `CTLuse_url = "no"` removes the duplicated Kaggle URL and the long proceedings URLs that stretched the reference lines; the last page's columns are balanced; the abstract is 247 words and expands ISLR. Kept: the author block (as the author specified), the voice ("we", "the author", and "I" in the Acknowledgment, whose text is fixed) and the index terms. |
| Writing | Fixed | "golden parity test" became "golden-fixture test" everywhere; no sentence starts with a numeral; "largely appearance-free"; "in part because the serving pipeline does not reproduce the training features"; the heading "A second training set (v3)" became "Training on a second dataset (v3)"; $r_{\mathrm{train}}$ is bound to the training ratio; the v2 mouth points are named as pose points. |
| Questions | Partly | Answered in the text: Q2 (the shipped build ran inference on the main thread), Q10 (abstention and clip scores), Q13 (Teach Dex's embedding, combination, weight and offset). The others need new analysis or experiments. |

Recommendation after the fixes: the remaining major points (R1, R2, R4 and the experimental parts
of R3 and R5) need new recordings, runs or test-split accesses, so the reviewer's weak reject
stands until they are addressed.

Earlier author TODOs: the AI-assistance disclosure that the first review listed under item 33 is
now the Acknowledgment section of both versions (`paper/sections/acknowledgment.tex`); the paper
sources contain no TODO or placeholder about it, and no separate AI declaration section.
