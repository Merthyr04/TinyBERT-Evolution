# Lexical baselines on AG News

Split as published: 120 / 100 / 196.

## Real split (train on the 120 published training paragraphs)

| probe | vocab | train acc (%) | dev acc (%) | test acc (%) | test macro-F1 (%) |
|---|---|---|---|---|---|
| majority class (0) | 0 | 25.0 | 25.0 | 25.0 | 10.0 |
| TF-IDF (1-2) + logistic regression, C = 1 | 5902 | 100.0 | 71.0 | 65.3 | 65.0 |
| TF-IDF (1-2) + logistic regression, C = 10 | 5902 | 100.0 | 73.0 | 65.3 | 65.1 |
| TF-IDF (1-2) + logistic regression, C = 100 | 5902 | 100.0 | 73.0 | 64.8 | 64.5 |
| TF-IDF (1-2) + linear SVM, C = 1 | 5902 | 100.0 | 73.0 | 64.8 | 64.5 |
| TF-IDF (1-1) + logistic regression, C = 10 | 1964 | 100.0 | 72.0 | 65.3 | 65.1 |
| TF-IDF nearest class centroid (Rocchio) | 5902 | 100.0 | 71.0 | 65.3 | 64.9 |

## Positional split (early paragraphs per chapter for training)

| definition | n train | n test | train acc (%) | test acc (%) |
|---|---|---|---|---|
| first 50% of each chapter | 208 | 208 | 100.0 | 68.3 |
| first 60% of each chapter | 248 | 168 | 100.0 | 72.0 |
| first 70% of each chapter | 292 | 124 | 100.0 | 73.4 |
| first 80% of each chapter | 332 | 84 | 100.0 | 70.2 |
