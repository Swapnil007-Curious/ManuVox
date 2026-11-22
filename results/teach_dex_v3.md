# Teach Dex: personalised result (record-then-test)

Separate from every main result. One signer: take 1 of test_clips_v2 taught each sign through the app's Teach panel, up to three examples from repeated playback of the one take-1 clip. Each take-1 clip had a 3.5 s still lead-in (its first frame repeated) so the panel's countdown did not cover the recorded sign, and the listening window spanned the clip; each example is the first segment after the countdown. Take 2 was then recognised with the examples on, through the final live pipeline (installed Chrome, GPU tracker), 3 runs. Blend weight and offset come from the validation split (scripts/calibrate_teach.py), not from these clips. Model sha256 `a30b5b1b5bdb`. Written by scripts/teach_report.py.

| Clip top-1 on take 2 (%) | Value | 95% CI |
|---|---|---|
| Model alone | 23 | [9, 37] |
| Teach Dex, all 25 signs taught | 28 | [13, 44] |
| Control, 13 signs taught | 25 | [12, 40] |

| Comparison (paired over clips) | Difference (points) | 95% CI | Better / worse | p (Holm) |
|---|---|---|---|---|
| all 25 signs taught vs the model alone | +5 | [0, 12] | 3 / 0 | 0.500 (not significant) |
| 13 taught: the 12 untaught signs vs the model alone | +0 | [0, 0] | 0 / 0 | 1.000 (not significant) |
| 13 taught: the 13 taught signs vs the model alone (no test) | +5 | [0, 13] | 2 / 0 | |

Top-3: model 39%, Teach Dex 43%. Examples recorded per run: [64, 61, 59] of 75; signs left without examples: [['happy', 'mom', 'thankyou'], ['book', 'happy', 'mom', 'thankyou'], ['happy', 'thankyou']]. Segments badged "Personalized for you": [13, 11, 4] of [28, 23, 15] per run. The Node recomputation matched the app's decisions to 1.1e-16.

Limits: one signer, taught and tested on two takes recorded in one sitting, the test vocabulary equal to the taught signs (the all-taught condition is the best case), and 25 clips. The effect is +5 points; the untaught-sign change in the control is +0 points.
