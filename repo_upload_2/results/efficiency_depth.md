# Mean encoder depth and analytic compute (deployed exit, theta = 0.5)

Student: 4 encoder layers, auxiliary head after layer 2, so a paragraph runs 2 layers if it exits and 4 otherwise. Computed over the 196 test paragraphs, five seeds.

| corpus | exit rate (%) | mean depth (layers) | layer-compute (%) | layer-only bound | FLOPs/paragraph (full) | (deployed) |
|---|---|---|---|---|---|---|
| LD-100K | 98.3 | 2.03 | 50.9 | 1.97x | 0.920 GFLOP | 0.468 GFLOP |
| OD-100K | 86.7 | 2.27 | 56.6 | 1.77x | 1.082 GFLOP | 0.613 GFLOP |
