# Shallow/deep head agreement on AG News

Exit threshold theta = 0.5.  Test split: 196 paragraphs.

| system | seed | aux acc | deep acc | exit acc | aux = deep | exit != deep |
|---|---|---|---|---|---|---|
| S3 | 7 | 81.1 | 81.1 | 80.6 | 187/196 | 7/196 |
| S3 | 13 | 74.0 | 78.6 | 79.1 | 161/196 | 10/196 |
| S3 | 37 | 83.2 | 82.7 | 82.7 | 187/196 | 5/196 |
| S3 | 42 | 79.6 | 81.6 | 79.1 | 172/196 | 17/196 |
| S3 | 97 | 76.0 | 80.1 | 79.1 | 182/196 | 4/196 |
