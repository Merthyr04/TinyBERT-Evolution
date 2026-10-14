# Lexical baselines on LD-100K

Split as published: 120 / 100 / 196.

## Real split (train on the 120 published training paragraphs)

| probe | vocab | train acc (%) | dev acc (%) | test acc (%) | test macro-F1 (%) |
|---|---|---|---|---|---|
| majority class (11) | 0 | 9.2 | 10.0 | 9.2 | 1.4 |
| TF-IDF (1-2) + logistic regression, C = 1 | 2466 | 100.0 | 93.0 | 89.8 | 90.3 |
| TF-IDF (1-2) + logistic regression, C = 10 | 2466 | 100.0 | 95.0 | 91.3 | 91.6 |
| TF-IDF (1-2) + logistic regression, C = 100 | 2466 | 100.0 | 95.0 | 92.9 | 93.0 |
| TF-IDF (1-2) + linear SVM, C = 1 | 2466 | 100.0 | 96.0 | 94.9 | 95.0 |
| TF-IDF (1-1) + logistic regression, C = 10 | 733 | 100.0 | 94.0 | 91.3 | 91.6 |
| TF-IDF nearest class centroid (Rocchio) | 2466 | 100.0 | 95.0 | 90.8 | 91.5 |

## Positional split (early paragraphs per chapter for training)

| definition | n train | n test | train acc (%) | test acc (%) |
|---|---|---|---|---|
| first 50% of each chapter | 210 | 206 | 100.0 | 93.7 |
| first 60% of each chapter | 250 | 166 | 100.0 | 95.2 |
| first 70% of each chapter | 291 | 125 | 100.0 | 96.0 |
| first 80% of each chapter | 332 | 84 | 100.0 | 96.4 |
