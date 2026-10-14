# TinyBERT-Evolution

Code and corpora for the paper *"TinyBERT Under a Fixed Inference Budget: A
Dual-Head Training Signal for Safe Truncation"* (Boxin Chen, Huaqiao University).

A dual-head early exit (a shallow classifier after layer 2 of a 4-layer
TinyBERT student, trained with its own cross-entropy plus a consistency term
against the deep head) is evaluated under a two-baseline protocol with a
conjunctive bootstrap-and-McNemar decision rule, pre-declared TOST margins and
a measured power curve. The paper's central finding is a negative one: once
the pre-declared rule and its Holm correction are applied, no accuracy gain is
established against either baseline, and a gateless fixed-depth truncation
matches the deployed system once fused — so the dual head earns its cost at
training time, as the evidence that truncation is safe, not as a serving
mechanism.

## Repository layout

```
code/            training, benchmarking, statistics and corpus-construction scripts
figures_tables/  figure and table generation scripts
results/         all result JSONs; students*.json embed per-seed test
                 probabilities (test_probs) and gold labels (test_gold),
                 i.e. the per-seed prediction arrays behind every McNemar,
                 bootstrap and TOST number in the paper
data/            corpora and splits (CC BY 4.0, see below)
sysinfo.json     hardware/software snapshot of the measurement machine
```

## Corpora (CC BY 4.0)

| File | Content |
|---|---|
| `data/LD-100K.txt`, `data/LD-100K_labels.tsv` | LD-100K: a 100,116-character English article, 416 body paragraphs with 12 topic labels from chapter structure |
| `data/ptc.jsonl`, `data/label_map.json` | the 120 / 100 / 196 train / dev / test split used by every main-corpus number |
| `data/organic_entries.jsonl`, `data/organic_label_map.json`, `data/organic_ptc.jsonl` | OD-100K: 416 entries from *The Nuttall Encyclopaedia* (public domain), headwords removed, 12 classes, single-author labels (κ = 0.45 against a keyword rule) |
| `data/agnews_ptc.jsonl`, `data/agnews_label_map.json` | the AG News paragraph-classification subsample (196-item test split and full 7,380-item test evaluation); derived from the AG News corpus, redistributed as a subsample only |

## Environment

Python 3.13, PyTorch 2.9.1 (CUDA 12.6). The student is the publicly released
generally-distilled TinyBERT4 checkpoint (`huawei-noah/TinyBERT_General_4L_312D`
on HuggingFace); `code/fetch_models.py` downloads it. Model weights produced by
training are not redistributed — every script reruns from the public checkpoint.
Exact package versions: see `sysinfo.json`.

## Reproducing

1. `python code/fetch_models.py` — fetch the public TinyBERT4 / BERT-base / mBERT checkpoints
2. `python code/stage_teachers.py` then `python code/stage_students.py` — train teachers and students (5 seeds: 7, 13, 37, 42, 97)
3. `python code/benchmark.py` — round-robin throughput registration
4. `python code/review_stats.py` — bootstrap, McNemar, Holm, TOST and power simulation behind Table 4/5
5. `python ../figures_tables/make_figures.py`, `build_tables.py` — regenerate figures and tables from `results/`

Measurement caveat carried over from the paper: throughput ratios are only
interpreted inside a single round-robin run (within-run noise floor ~15%,
between-run ~30% on this laptop GPU).

## License

- Code: MIT (see `LICENSE`)
- Corpora and prediction arrays in `data/`: CC BY 4.0
