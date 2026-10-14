# Organic-corpus replication (OD-100K, Nuttall Encyclopaedia)

Seeds: [7, 13, 37, 42, 97].  Split 120 / 100 / 196, identical protocol.

## (a) System ordering

| System | PTC mean (%) | sd | per-seed PTC (%) | macro-F1 (%) | exit rate (%) | shallow/deep agreement (%) |
|---|---|---|---|---|---|---|
| 1. S6 | 58.0 | 2.8 | 60.2, 61.7, 57.6, 56.1, 54.1 | 41.0 | - | - |
| 2. A0 | 57.9 | 1.5 | 59.2, 58.7, 56.6, 55.6, 59.2 | 45.9 | - | - |
| 3. S4 | 56.7 | 2.9 | 58.2, 60.2, 58.2, 55.1, 52.0 | 37.3 | - | - |
| 4. S7 | 56.6 | 2.3 | 58.2, 58.2, 52.5, 58.7, 55.6 | 39.5 | 86.7 | 79.8 |
| 5. S0 | 55.1 | 1.4 | 54.1, 55.6, 53.1, 57.1, 55.6 | 40.6 | - | - |
| 6. F2 | 53.2 | 3.5 | 49.5, 53.6, 55.6, 49.0, 58.2 | 35.1 | - | - |

Exit rate and shallow/deep agreement are at theta = 0.5 on the 196 test paragraphs (the registered operating point); exit rate 80.6-91.8%, heads agree on 138-169 of 196, the exit changes 16-35 decisions.

| - BERT-base | 64.4 | 1.1 | 66.3, 63.3, 63.3, 64.3, 64.8 | - | - | - |
| - mBERT | 64.4 | 2.3 | 61.7, 63.8, 68.4, 65.3, 62.8 | - | - | - |

## (b) Per-seed delta against S0 and A0

| System | vs | per-seed ΔPTC (pp) | mean ΔPTC (pp) |
|---|---|---|---|
| S0 | S0 | +0.0, +0.0, +0.0, +0.0, +0.0 | +0.0 |
| S0 | A0 | -5.1, -3.1, -3.6, +1.5, -3.6 | -2.8 |
| A0 | S0 | +5.1, +3.1, +3.6, -1.5, +3.6 | +2.8 |
| A0 | A0 | +0.0, +0.0, +0.0, +0.0, +0.0 | +0.0 |
| S4 | S0 | +4.1, +4.6, +5.1, -2.0, -3.6 | +1.6 |
| S4 | A0 | -1.0, +1.5, +1.5, -0.5, -7.1 | -1.1 |
| S6 | S0 | +6.1, +6.1, +4.6, -1.0, -1.5 | +2.9 |
| S6 | A0 | +1.0, +3.1, +1.0, +0.5, -5.1 | +0.1 |
| S7 | S0 | +4.1, +2.5, -0.5, +1.5, +0.0 | +1.5 |
| S7 | A0 | -1.0, -0.5, -4.1, +3.1, -3.6 | -1.2 |
| F2 | S0 | -4.6, -2.0, +2.5, -8.2, +2.5 | -1.9 |
| F2 | A0 | -9.7, -5.1, -1.0, -6.6, -1.0 | -4.7 |

Paired tests on the pooled predictor (196 test paragraphs):

| System | vs | ΔPTC (pp) | 95% CI | bootstrap p | McNemar p |
|---|---|---|---|---|---|
| S0 | A0 | -2.5 | [-6.6, +1.5] | 0.2664 n.s. | 0.3323 n.s. |
| A0 | S0 | +2.5 | [-1.5, +6.6] | 0.2664 n.s. | 0.3323 n.s. |
| S4 | S0 | +3.0 | [-1.0, +7.1] | 0.1656 n.s. | 0.2101 n.s. |
| S4 | A0 | +0.5 | [-4.1, +5.1] | 0.9324 n.s. | 1.0000 n.s. |
| S6 | S0 | +4.6 | [+1.0, +8.2] | 0.0100 * | 0.0225 * |
| S6 | A0 | +2.0 | [-2.0, +6.1] | 0.4004 n.s. | 0.4807 n.s. |
| S7 | S0 | +4.6 | [+0.0, +9.2] | 0.0592 n.s. | 0.0784 n.s. |
| S7 | A0 | +2.0 | [-2.6, +7.1] | 0.4870 n.s. | 0.5413 n.s. |
| F2 | S0 | -1.5 | [-6.1, +3.1] | 0.5742 n.s. | 0.6476 n.s. |
| F2 | A0 | -4.1 | [-9.2, +1.0] | 0.1498 n.s. | 0.1849 n.s. |

## Lexical ceiling on the organic split

| probe | PTC (%) | macro-F1 (%) |
|---|---|---|
| majority class (9) | 19.4 | 2.7 |
| TF-IDF (1-2) + logistic regression, C = 1 | 29.1 | 7.7 |
| TF-IDF (1-2) + logistic regression, C = 10 | 36.7 | 16.6 |
| TF-IDF (1-2) + logistic regression, C = 100 | 38.3 | 17.5 |
| TF-IDF (1-2) + linear SVM, C = 1 | 38.8 | 21.0 |
| TF-IDF (1-1) + logistic regression, C = 10 | 36.2 | 16.1 |
| TF-IDF nearest class centroid (Rocchio) | 38.8 | 24.8 |
