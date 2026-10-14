# Shallow/deep head agreement on OD-100K (Nuttall)

Exit threshold theta = 0.5.  Test split: 196 paragraphs.

| system | seed | aux acc | deep acc | exit acc | aux = deep | exit != deep |
|---|---|---|---|---|---|---|
| S7 | 7 | 57.1 | 58.2 | 57.6 | 169/196 | 19/196 |
| S7 | 13 | 58.7 | 58.2 | 57.6 | 163/196 | 16/196 |
| S7 | 37 | 55.1 | 52.5 | 55.6 | 157/196 | 29/196 |
| S7 | 42 | 51.0 | 58.7 | 52.0 | 138/196 | 35/196 |
| S7 | 97 | 56.1 | 55.6 | 56.6 | 155/196 | 23/196 |
