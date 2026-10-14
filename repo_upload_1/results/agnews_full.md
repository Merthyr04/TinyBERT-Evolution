# AG News full official test set (n = 7380)

Official test items 7600; excluded 220 (train/dev of the round-75 subsample); truncated at 96 pieces: 200 (longest item 276 pieces).

## Accuracy (always-deep path)

| system | mean acc (%) | sd | per-seed (%) |
|---|---|---|---|
| S0 | 79.41 | 1.09 | 79.24, 78.40, 78.10, 80.88, 80.41 |
| S1 | 79.09 | 0.55 | 79.24, 79.16, 79.12, 79.82, 78.12 |
| S3 | 80.70 | 1.90 | 82.64, 77.28, 81.82, 81.67, 80.11 |

## Paired contrasts vs S0 (pooled predictor)

| contrast | pooled delta (pp) | 95% CI (pp) | p boot | p McNemar | n01/n10 |
|---|---|---|---|---|---|
| S1 vs S0 | +0.91 | [+0.39, +1.41] | 0.0008 | 0.0006 | 153/220 |
| S3 vs S0 | +2.41 | [+1.87, +2.97] | 0.0000 | 0.0000 | 131/309 |
| S3 exit vs S0 | +2.57 | [+1.99, +3.16] | 0.0000 | 0.0000 | 148/338 |

## S3 exit behaviour at theta = 0.5
Exit rate 77.7%-97.2% (mean 92.0%); exit changes 194-479 of 7380 decisions per seed; aux = deep on 6210-7096.
