# v2 on the ASL Citizen test split

Model `d_hybrid_v2feat_v2recipe/seed_0` (fp16, sha256 `f7d79f675549`), 3523 test videos of 236 mapped signs from 11 signers; 13 more were left out by preprocessing (too few hand frames) and 0 could not be tracked. Written by scripts/citizen_eval.py.

| Metric | Value | 95% CI (videos) | 95% CI (signers) |
|---|---|---|---|
| Top-1 (%) | 43.1 | [41.5, 44.8] | [39.3, 46.6] |
| Top-3 (%) | 57.6 | [56.0, 59.3] | [53.9, 61.1] |
| Top-1, excluded videos counted wrong (%) | 43.0 | | |
| ECE, raw softmax | 0.147 | [0.134, 0.162] | |
| ECE, temperature 0.868 (val) | 0.066 | [0.054, 0.080] | |
| Coverage at threshold 0.488 (%) | 31.1 | | |
| Selective accuracy (%) | 79.8 | | |

Per-signer top-1 ranges from 31.4% to 50.0%.

| Mapping rule | Videos | Top-1 (%) | Top-3 (%) |
|---|---|---|---|
| exact | 2083 | 48.6 | 64.6 |
| synonym | 481 | 43.9 | 54.9 |
| variant | 959 | 30.9 | 43.9 |

Variant and synonym mappings are AI-assisted and unconfirmed (src/edgevision/data/citizen_labels.py).
