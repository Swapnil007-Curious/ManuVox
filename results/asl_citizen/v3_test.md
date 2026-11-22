# v3 on the ASL Citizen test split

Model `e_hybrid_v3citizen/seed_2` (fp16, sha256 `a30b5b1b5bdb`), 3523 test videos of 236 mapped signs from 11 signers; 13 more were left out by preprocessing (too few hand frames) and 0 could not be tracked. Written by scripts/citizen_eval.py.

| Metric | Value | 95% CI (videos) | 95% CI (signers) |
|---|---|---|---|
| Top-1 (%) | 92.1 | [91.2, 93.0] | [90.3, 93.9] |
| Top-3 (%) | 97.2 | [96.7, 97.8] | [96.4, 98.0] |
| Top-1, excluded videos counted wrong (%) | 91.8 | | |
| ECE, raw softmax | 0.150 | [0.142, 0.158] | |
| ECE, temperature 0.863 (val) | 0.066 | [0.058, 0.073] | |
| Coverage at threshold 0.565 (%) | 88.1 | | |
| Selective accuracy (%) | 97.3 | | |

Per-signer top-1 ranges from 87.1% to 97.5%.

| Mapping rule | Videos | Top-1 (%) | Top-3 (%) |
|---|---|---|---|
| exact | 2083 | 91.1 | 96.4 |
| synonym | 481 | 93.6 | 98.5 |
| variant | 959 | 93.6 | 98.2 |

Variant and synonym mappings are AI-assisted and unconfirmed (src/edgevision/data/citizen_labels.py).
