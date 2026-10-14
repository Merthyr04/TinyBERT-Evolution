# Lexical baselines on OD-100K (Nuttall)

Split as published: 120 / 100 / 196.

## Real split (train on the 120 published training paragraphs)

| probe | vocab | train acc (%) | dev acc (%) | test acc (%) | test macro-F1 (%) |
|---|---|---|---|---|---|
| majority class (9) | 0 | 19.2 | 19.0 | 19.4 | 2.7 |
| TF-IDF (1-2) + logistic regression, C = 1 | 5704 | 74.2 | 29.0 | 29.1 | 7.7 |
| TF-IDF (1-2) + logistic regression, C = 10 | 5704 | 100.0 | 37.0 | 36.7 | 16.6 |
| TF-IDF (1-2) + logistic regression, C = 100 | 5704 | 100.0 | 40.0 | 38.3 | 17.5 |
| TF-IDF (1-2) + linear SVM, C = 1 | 5704 | 100.0 | 43.0 | 38.8 | 21.0 |
| TF-IDF (1-1) + logistic regression, C = 10 | 1889 | 100.0 | 41.0 | 36.2 | 16.1 |
| TF-IDF nearest class centroid (Rocchio) | 5704 | 100.0 | 45.0 | 38.8 | 24.8 |

## Positional split (early paragraphs per chapter for training)

| definition | n train | n test | train acc (%) | test acc (%) |
|---|---|---|---|---|
| first 50% of each chapter | 206 | 210 | 100.0 | 38.1 |
| first 60% of each chapter | 248 | 168 | 100.0 | 40.5 |
| first 70% of each chapter | 291 | 125 | 100.0 | 41.6 |
| first 80% of each chapter | 333 | 83 | 100.0 | 47.0 |
