# Fixed-depth control F2

Seeds: [7, 13, 37, 42, 97].  Same recipe as S7 (DA + dual teacher + every distillation term) but the deployed classifier reads the layer-2 hidden state; no auxiliary head, no consistency term.

F2: per-seed mean **91.22 $\pm$ 1.72%**, pooled **93.37%**, MLM top-1 50.4%.

## Contrasts (pooled predictor, 196 test paragraphs)

| System | ref | PTC sys (%) | PTC ref (%) | Δ (pp) | 95% CI | bootstrap p | McNemar p |
|---|---|---|---|---|---|---|---|
| F2 | S0 | 93.4 | 91.3 | +2.04 | [-0.5, +5.1] | 0.2102 n.s. | 0.2891 n.s. |
| F2 | S4 | 93.4 | 94.4 | -1.01 | [-3.1, +1.0] | 0.4686 n.s. | 0.6250 n.s. |
| F2 | S6 | 93.4 | 94.4 | -1.01 | [-3.1, +1.0] | 0.4686 n.s. | 0.6250 n.s. |
| F2 | S7 | 93.4 | 94.4 | -1.02 | [-3.1, +1.0] | 0.4510 n.s. | 0.6250 n.s. |

## Per-seed

| seed | F2 PTC (%) | S0 PTC (%) | Δ vs S0 (pp) | S7 PTC (%) | Δ vs S7 (pp) |
|---|---|---|---|---|---|
| 7 | 92.9 | 89.3 | +3.6 | 94.4 | -1.5 |
| 13 | 89.3 | 87.8 | +1.5 | 94.4 | -5.1 |
| 37 | 93.4 | 88.8 | +4.6 | 93.4 | +0.0 |
| 42 | 89.3 | 87.8 | +1.5 | 92.9 | -3.6 |
| 97 | 91.3 | 90.3 | +1.0 | 94.9 | -3.6 |
