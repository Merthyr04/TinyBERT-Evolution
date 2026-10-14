# Public-benchmark replication (AG News, 120 / 100 / 196)

Seeds: [7, 13, 37, 42, 97].  Four classes, 30 / 25 / 49 items per class in the train / dev / test split.
Complete: yes.

## (a) Accuracy

| System | PTC mean (%) | sd | per-seed PTC (%) | macro-F1 (%) |
|---|---|---|---|---|
| S0 | 78.7 | 2.1 | 76.0, 78.6, 77.0, 82.1, 79.6 | 78.7 |
| S1 | 77.8 | 1.5 | 76.0, 79.1, 78.1, 79.6, 76.0 | 77.9 |
| S3 | 80.8 | 1.4 | 81.1, 78.6, 82.7, 81.6, 80.1 | 80.9 |

Pooled predictor: S0 80.1%, S1 79.6%, S3 84.2%.

## (b) Paired contrast against S0

| Contrast | per-seed delta (pp) | mean delta (pp) | pooled delta (pp) | 95% CI | p bootstrap | p McNemar |
|---|---|---|---|---|---|---|
| S1 vs S0 | +0.0, +0.5, +1.0, -2.5, -3.6 | -0.92 | -0.51 | [-4.1, +3.1] | 0.8836 | 1.0000 |
| S3 vs S0 | +5.1, +0.0, +5.6, -0.5, +0.5 | +2.14 | 4.08 | [+0.0, +8.2] | 0.0506 | 0.0768 |

## (c) Dual head at theta = 0.5

Exit rate 71.9--98.0% (mean 89.0%); the two heads agree on 161--187 of 196 paragraphs; the exit path changes 4--17 decisions.
