### Accuracy of every system against the baseline

| System | seeds | PTC acc. (mean $\pm$ s.d.) | pooled acc. | $\Delta$ vs S0 | 95% CI of $\Delta$ | bootstrap p | McNemar p |
|---|---|---|---|---|---|---|---|
| **S0** | 5 | 88.8 $\pm$ 1.0 | 91.3 | - | - | - | - |
| S1 | 5 | 91.4 $\pm$ 1.2 | 93.9 | +2.6 | [+0.0, +5.1] | 0.0696 n.s. | 0.1250 n.s. |
| S2 | 5 | 91.0 $\pm$ 0.7 | 93.4 | +2.0 | [+0.5, +4.1] | 0.0380 * | 0.1250 n.s. |
| S3 | 5 | 89.9 $\pm$ 1.5 | 91.3 | +0.0 | [-2.0, +2.0] | 1.0000 n.s. | 1.0000 n.s. |
| S4 | 5 | 93.9 $\pm$ 1.2 | 94.4 | +3.0 | [+0.5, +6.1] | 0.0356 * | 0.0703 n.s. |
| S5 | 5 | 91.2 $\pm$ 0.7 | 93.9 | +2.5 | [+0.5, +5.1] | 0.0166 * | 0.0625 n.s. |
| S6 | 5 | 93.1 $\pm$ 1.4 | 94.4 | +3.0 | [+0.5, +6.1] | 0.0356 * | 0.0703 n.s. |
| S7 | 5 | 94.0 $\pm$ 0.7 | 94.4 | +3.1 | [+1.0, +5.6] | 0.0050 ** | 0.0312 * |
| BERT-base | 5 | 96.2 $\pm$ 0.8 | 96.9 | +5.6 | [+2.0, +9.2] | 0.0006 *** | 0.0034 ** |
| mBERT | 5 | 93.3 $\pm$ 1.4 | 95.4 | +4.1 | [+0.0, +8.2] | 0.0502 n.s. | 0.0768 n.s. |

### Ablation: removing one ingredient from the full model

| Ablation | removed ingredient | seeds | ablation PTC acc. | full model (same seeds) | $\Delta$ vs full model | 95% CI | bootstrap p | McNemar p |
|---|---|---|---|---|---|---|---|---|
| A1 | attention distillation $L_{attn}$ | 5 | 94.4 $\pm$ 0.9 | 94.4 | +0.0 | [-1.5, +1.5] | 1.0000 n.s. | 1.0000 n.s. |
| A2 | hidden-state distillation $L_{hid}$ | 5 | 95.4 $\pm$ 1.4 | 94.4 | +1.0 | [+0.0, +2.6] | 0.2488 n.s. | 0.5000 n.s. |
| A3 | prediction distillation $L_{pred}$ | 5 | 94.4 $\pm$ 1.3 | 94.4 | +0.0 | [+0.0, +0.0] | 1.0000 n.s. | 1.0000 n.s. |
| A4 | auxiliary MLM objective $L_{mlm}$ | 5 | 95.9 $\pm$ 0.5 | 94.4 | +1.5 | [-0.5, +4.1] | 0.2440 n.s. | 0.3750 n.s. |
| A5 | data augmentation | 5 | 92.9 $\pm$ 1.2 | 94.4 | -1.5 | [-4.1, +0.5] | 0.2492 n.s. | 0.3750 n.s. |

### Accuracy / efficiency

| System | params (M) | PTC acc. | MLM top-1 | passages/s | speed index | exit rate | train time (s) | peak train VRAM (MB) |
|---|---|---|---|---|---|---|---|---|
| S0 | 14.39 | 88.8 $\pm$ 1.0 | 34.8 | 1121.7 | 100.0 | 0.0 | 57 | 1147 |
| S1 | 14.39 | 91.4 $\pm$ 1.2 | 50.1 | 1148.7 | 102.4 | 0.0 | 142 | 1147 |
| S2 | 14.39 | 91.0 $\pm$ 0.7 | 33.2 | 1166.0 | 104.0 | 0.0 | 87 | 2464 |
| S3 | 14.39 | 89.9 $\pm$ 1.5 | 31.4 | 2022.5 | 180.3 | 97.1 | 60 | 2262 |
| S4 | 14.39 | 93.9 $\pm$ 1.2 | 48.1 | 1237.7 | 110.3 | 0.0 | 316 | 2187 |
| S5 | 14.39 | 91.2 $\pm$ 0.7 | 34.4 | 1183.4 | 105.5 | 0.0 | 161 | 2463 |
| S6 | 14.39 | 93.1 $\pm$ 1.4 | 46.1 | 1177.4 | 105.0 | 0.0 | 532 | 2462 |
| S7 | 14.39 | 94.0 $\pm$ 0.7 | 46.4 | 2095.1 | 186.8 | 99.5 | 483 | 2462 |
| BERT-base | 109.5 | 96.2 $\pm$ 0.8 | - | 99.6 | 8.9 | - | 90 | - |
| mBERT | 177.9 | 93.3 $\pm$ 1.4 | - | 101.2 | 9.0 | - | 115 | - |
| A1 | 14.39 | 92.9 $\pm$ 0.9 | 54.4 | 1243.4 | 110.8 | 0.0 | 397 | 2351 |
| A2 | 14.39 | 93.2 $\pm$ 1.4 | 50.4 | 1183.1 | 105.5 | 0.0 | 455 | 2426 |
| A3 | 14.39 | 92.8 $\pm$ 1.3 | 48.7 | 1150.2 | 102.5 | 0.0 | 288 | 2461 |
| A4 | 14.39 | 95.9 $\pm$ 0.5 | 0.0 | 1155.6 | 103.0 | 0.0 | 336 | 1777 |
| A5 | 14.39 | 91.3 $\pm$ 1.2 | 27.1 | 2034.3 | 181.4 | 98.1 | 159 | 2463 |

Significance: `*` p<0.05, `**` p<0.01, `***` p<0.001, `n.s.` not significant.  95% CIs from 10,000 paired bootstrap resamples of the LD-100K test split.


In the ablation table the full model is evaluated on exactly the seeds the ablation cell was run with (13, 37 and 97), so the difference is attributable to the removed ingredient and not to a different number of seeds.
