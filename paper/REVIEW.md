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
