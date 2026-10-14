**Augmentation factor $A$, fixed 50 epochs (S1 configuration: DA only, no teacher).** Larger $A$ means more data *and* more gradient steps; the reference cell $A=2$ is the paper's S1.

| value | train size | epochs | steps | PTC mean$\pm$s.d. (%) | pooled PTC (%) | $\Delta$ vs ref | 95% CI | bootstrap p | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| $A=0$ |  120 | 50 | 350 | 88.8 $\pm$ 1.1 | 91.3 | -2.5 | [-5.1, +0.0] | 0.0696 n.s. | 0.1250 n.s. |
| $A=1$ |  240 | 50 | 750 | 89.9 $\pm$ 2.3 | 93.4 | -0.5 | [-2.5, +1.5] | 0.8444 n.s. | 1.0000 n.s. |
| $A=2$ |  360 | 50 | 1100 | 91.4 $\pm$ 1.4 | 93.9 | +0.0 | [+0.0, +0.0] | 1.0000 n.s. | 1.0000 n.s. |
| $A=4$ |  600 | 50 | 1850 | 93.0 $\pm$ 2.3 | 94.9 | +1.0 | [-1.5, +3.6] | 0.5362 n.s. | 0.6875 n.s. |
| $A=8$ |  1080 | 50 | 3350 | 93.7 $\pm$ 1.5 | 95.9 | +2.0 | [-0.5, +5.1] | 0.1954 n.s. | 0.2891 n.s. |

**Augmentation factor $A$, budget-matched.** Total gradient steps are pinned to the $A{=}2$ budget (1100 steps): epochs are $1100/\lfloor 120(1{+}A)/16\rfloor$, so a larger $A$ buys more diverse data at the *same* number of updates. The reference is the same $A{=}2$ cell as above.

| value | train size | epochs | steps | PTC mean$\pm$s.d. (%) | pooled PTC (%) | $\Delta$ vs ref | 95% CI | bootstrap p | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| $A=0$ |  120 | 157 | 1099 | 90.3 $\pm$ 1.5 | 93.4 | -0.5 | [-3.1, +2.0] | 0.8456 n.s. | 1.0000 n.s. |
| $A=1$ |  240 | 73 | 1095 | 91.7 $\pm$ 0.2 | 94.9 | +1.0 | [-1.5, +3.6] | 0.5310 n.s. | 0.6875 n.s. |
| $A=4$ |  600 | 30 | 1110 | 92.0 $\pm$ 2.5 | 94.9 | +1.0 | [-1.5, +4.1] | 0.5934 n.s. | 0.7266 n.s. |
| $A=8$ |  1080 | 16 | 1072 | 94.3 $\pm$ 0.6 | 95.4 | +1.5 | [-0.5, +4.1] | 0.2376 n.s. | 0.3750 n.s. |

**More updates at fixed data: fixed arm minus budget arm, same $A$.** A positive difference means the extra gradient steps of the fixed schedule (beyond the 1100-step budget) help at that $A$.

| $A$ | fixed PTC (%) | budget PTC (%) | $\Delta$ (fixed $-$ budget) | 95% CI | bootstrap p | McNemar p |
|---|---|---|---|---|---|---|
| $A=0$ | 91.3 | 93.4 | -2.0 | [-4.6, +0.0] | 0.1332 n.s. | 0.2188 n.s. |
| $A=1$ | 93.4 | 94.9 | -1.5 | [-4.1, +0.5] | 0.2508 n.s. | 0.3750 n.s. |
| $A=4$ | 94.9 | 94.9 | -0.0 | [-1.5, +1.5] | 1.0000 n.s. | 1.0000 n.s. |
| $A=8$ | 95.9 | 95.4 | +0.5 | [-1.0, +2.0] | 0.7832 n.s. | 1.0000 n.s. |

**Distillation temperature $T$ (S2 configuration: single-teacher distillation, no augmentation).** $T$ changes neither the data nor the number of updates, so the sweep is budget-neutral by construction; the reference cell $T=2$ is the paper's S2.

| value | train size | epochs | steps | PTC mean$\pm$s.d. (%) | pooled PTC (%) | $\Delta$ vs ref | 95% CI | bootstrap p | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| $T=1$ |  120 | 50 | 350 | 90.3 $\pm$ 1.8 | 92.3 | -1.0 | [-2.5, +0.0] | 0.2798 n.s. | 0.5000 n.s. |
| $T=2$ |  120 | 50 | 350 | 91.0 $\pm$ 0.8 | 93.4 | +0.0 | [+0.0, +0.0] | 1.0000 n.s. | 1.0000 n.s. |
| $T=4$ |  120 | 50 | 350 | 89.9 $\pm$ 2.0 | 93.4 | -0.0 | [-1.5, +1.5] | 1.0000 n.s. | 1.0000 n.s. |
| $T=8$ |  120 | 50 | 350 | 90.6 $\pm$ 0.3 | 93.9 | +0.5 | [-1.5, +2.5] | 0.8446 n.s. | 1.0000 n.s. |

Significance: `*` p<0.05, `**` p<0.01, `***` p<0.001, `n.s.` not significant.  95% CIs from 10,000 paired bootstrap resamples of the LD-100K test split; McNemar's exact test on the pooled 5-seed predictors.  All cells share seeds {7, 13, 37, 42, 97}.

