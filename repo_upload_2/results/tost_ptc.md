# TOST at the pre-declared margin (LD-100K, 196 test paragraphs)

Estimator: per-seed-mean paired bootstrap, 10,000 resamples (the Table 7 estimator).  TOST p = max(P(d >= +m), P(d <= -m)) on that distribution.

| contrast | delta (pp) | 95% CI (pp) | TOST p at 4.4 | smallest equivalent margin (pp) | p at 2 / 3 / 4 / 5 |
|---|---|---|---|---|---|
| S7 vs A0 | +2.76 | [+0.92, +4.69] | 0.0433 | 4.4 | 0.7920 / 0.3894 / 0.0930 / 0.0112 |
| S7 vs F2 | +2.75 | [+1.43, +4.18] | 0.0137 | 4.000000000000002 | 0.8560 / 0.3524 / 0.0435 / 0.0022 |
| S7 vs S0 | +5.21 | [+3.27, +7.35] | 0.7676 | 6.999999999999991 | 0.9999 / 0.9875 / 0.8753 / 0.5765 |
| S7 vs BERT-base | -2.25 | [-4.90, +0.10] | 0.0482 | 4.4 | 0.5685 / 0.2622 / 0.0840 / 0.0228 |
