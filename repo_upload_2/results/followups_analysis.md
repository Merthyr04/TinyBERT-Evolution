# Follow-up runs: analysis

## A0  (n=5, seeds [7, 13, 37, 42, 97])
- mean +- sd: 91.22 +- 1.14
- pooled: 93.88
- MLM top-1: 0.00

## A0 = S0 - L_mlm, against the published baseline
- pooled Delta = +2.54 pp, 95% CI [+0.00, +5.10], bootstrap p = 0.0708 n.s., McNemar p = 0.1250 n.s. (only A0 6, only S0 1)
- per-seed mean difference = +2.45 pp
    - seed   7: A0 90.82 vs S0 89.29  (+1.53 pp)
    - seed  13: A0 92.35 vs S0 87.76  (+4.59 pp)
    - seed  37: A0 91.33 vs S0 88.78  (+2.55 pp)
    - seed  42: A0 89.29 vs S0 87.76  (+1.53 pp)
    - seed  97: A0 92.35 vs S0 90.31  (+2.04 pp)
- against A4 (S6 - L_mlm): Delta = -2.05 pp, p_boot 0.1324, p_mc 0.2188

## Pooled accuracy against each single-seed S0 baseline
| system | S0 seed 7 (89.29) | S0 seed 13 (87.76) | S0 seed 37 (88.78) | S0 seed 42 (87.76) | S0 seed 97 (90.31) |
|---|---|---|---|---|---|
| S0 | +2.04 | +3.57 | +2.55 | +3.57 | +1.02 |
| S4 | +5.10 | +6.63 | +5.61 | +6.63 | +4.08 |
| S6 | +5.10 | +6.63 | +5.61 | +6.63 | +4.08 |
| S7 | +5.10 | +6.63 | +5.61 | +6.63 | +4.08 |
| S8 | +5.61 | +7.14 | +6.12 | +7.14 | +4.59 |
| A0 | +4.59 | +6.12 | +5.10 | +6.12 | +3.57 |
