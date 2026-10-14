# Shallow/deep head agreement on LD-100K

Exit threshold theta = 0.5.  Test split: 196 paragraphs.

| system | seed | aux acc | deep acc | exit acc | aux = deep | exit != deep |
|---|---|---|---|---|---|---|
| S9 | 7 | 94.4 | 95.4 | 94.9 | 190/196 | 5/196 |
| S9 | 13 | 96.4 | 96.4 | 95.4 | 192/196 | 2/196 |
| S9 | 37 | 93.9 | 95.4 | 94.9 | 188/196 | 6/196 |
| S9 | 42 | 94.9 | 95.9 | 95.4 | 192/196 | 1/196 |
| S9 | 97 | 94.9 | 95.4 | 94.9 | 191/196 | 5/196 |
