# v2 vs v3

Top-1 and top-3 (%) with 95% percentile bootstrap intervals; the difference is v3 - v2 on the same items, with its paired interval, the items each model got right alone, the sign-flip p value and the Holm-adjusted p over the four top-1 comparisons. Written by scripts/compare_v3.py.

| Test set | Items | v2 top-1 | v3 top-1 | Difference | v3 better / worse | p (Holm) | v2 top-3 | v3 top-3 |
|---|---|---|---|---|---|---|---|---|
| asl-signs test | 16242 | 68.4 [67.7, 69.1] | 69.4 [68.7, 70.1] | 1.0 [0.5, 1.5] | 1010 / 846 | 0.000 (significant) | 82.3 | 83.4 |
| ASL Citizen test | 3523 | 43.1 [41.5, 44.8] | 92.1 [91.2, 93.0] | 49.0 [47.3, 50.7] | 1758 / 33 | 0.000 (significant) | 57.6 | 97.2 |
| test_clips_v2, browser | 50 | 34.0 [21.3, 46.7] | 44.7 [31.3, 57.3] | 10.7 [-1.3, 23.3] | 13 / 6 | 0.238 (not significant) | 51.3 | 56.0 |
| test_clips_v2, Python | 50 | 40.0 [26.0, 54.0] | 46.0 [32.0, 60.0] | 6.0 [-6.0, 18.0] | 6 / 3 | 0.508 (not significant) | 56.0 | 68.0 |
