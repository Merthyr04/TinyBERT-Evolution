"""Leave-one-chapter-out (LOCO) evaluation of the lexical probe on LD-100K.

Quantifies how much of the TF-IDF probe's accuracy under the stratified
random split relies on within-chapter lexical leakage (paragraphs from the
same chapter appearing on both sides of the split).  12 folds: each fold
holds out one chapter entirely for testing and trains on the other 11.

Two probes, mirroring the paper's probe family:
  word:  TF-IDF word unigrams+bigrams + linear SVM
  char:  TF-IDF char_wb 2-5 grams   + linear SVM

Outputs tinybert_run/results/leakage_loco.json
"""
import json
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

ROOT = "E:/Second Paper"
TEXT = ROOT + "/LD-100K/LD-100K.txt"
LABELS = ROOT + "/LD-100K/LD-100K_labels.tsv"
OUT = ROOT + "/tinybert_run/results/leakage_loco.json"

# ---- load paragraphs (non-empty, non-heading lines, in order) ----
lines = open(TEXT, encoding="utf-8").read().split("\n")
first_ch = next(i for i, ln in enumerate(lines)
                if ln.strip().upper().startswith("CHAPTER "))
paras = []
for ln in lines[first_ch:]:
    s = ln.strip()
    if not s:
        continue
    if s.upper().startswith("CHAPTER "):
        continue
    if re.match(r"^\d+\.\d+\s", s) or re.match(r"^\d+\.\d+$", s):
        continue
    if s.startswith("LD-100K:"):
        continue
    paras.append(s)
print("body paragraphs extracted:", len(paras), file=sys.stderr)
assert len(paras) == 416, "expected 416 body paragraphs"

# ---- load labels ----
rows = []
with open(LABELS, encoding="utf-8") as f:
    header = f.readline().rstrip("\n").split("\t")
    assert header == ["para_id", "chapter", "label", "chars", "words", "split"]
    for line in f:
        pid, chapter, label, chars, words, split = line.rstrip("\n").split("\t")
        rows.append((int(pid), int(chapter), label, split))
assert len(rows) == 416
assert [r[0] for r in rows] == list(range(1, 417))
texts = [paras[r[0] - 1] for r in rows]
chapters = np.array([r[1] for r in rows])
labels = np.array([r[2] for r in rows])
splits = np.array([r[3] for r in rows])

# sanity: chapter i <-> label of chapter i is one-to-one
chap2lab = {}
for c, y in zip(chapters, labels):
    chap2lab.setdefault(c, set()).add(y)
assert all(len(v) == 1 for v in chap2lab.values())
print("chapters:", len(chap2lab), file=sys.stderr)


def make_probe(kind):
    if kind == "word":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    else:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                              min_df=2, sublinear_tf=True)
    return vec, LinearSVC(C=1.0, random_state=0)


def run_loco(kind):
    correct = 0
    n = 0
    per_fold = {}
    for c in sorted(set(chapters.tolist())):
        tr = chapters != c
        te = chapters == c
        vec, clf = make_probe(kind)
        Xtr = vec.fit_transform([texts[i] for i in np.where(tr)[0]])
        Xte = vec.transform([texts[i] for i in np.where(te)[0]])
        clf.fit(Xtr, labels[tr])
        pred = clf.predict(Xte)
        acc = float((pred == labels[te]).mean())
        vals, counts = np.unique(labels[tr], return_counts=True)
        maj = float(counts.max()) / int(tr.sum())
        per_fold[str(int(c))] = {
            "n_test": int(te.sum()),
            "acc": round(acc, 4),
            "majority_baseline": round(maj, 4),
        }
        correct += int((pred == labels[te]).sum())
        n += int(te.sum())
    return {"pooled_acc": round(correct / n, 4), "per_fold": per_fold}


def run_random_split(kind, seed=20261003):
    """Reference: train on train+dev, test on the 196-paragraph test side."""
    tr = splits != "test"
    te = splits == "test"
    vec, clf = make_probe(kind)
    Xtr = vec.fit_transform([texts[i] for i in np.where(tr)[0]])
    Xte = vec.transform([texts[i] for i in np.where(te)[0]])
    clf.fit(Xtr, labels[tr])
    pred = clf.predict(Xte)
    return {"test_acc": round(float((pred == labels[te]).mean()), 4),
            "n_test": int(te.sum())}


results = {}
for kind in ["word", "char"]:
    results[kind] = {
        "random_split": run_random_split(kind),
        "loco": run_loco(kind),
    }
    print(kind, "random:", results[kind]["random_split"]["test_acc"],
          "loco:", results[kind]["loco"]["pooled_acc"], file=sys.stderr)

# ---- position-only probe: chapter boundaries are contiguous blocks, so
# paragraph order alone may determine the label ----
from sklearn.tree import DecisionTreeClassifier

pid = np.array([r[0] for r in rows], dtype=float).reshape(-1, 1)
tr = splits != "test"
te = splits == "test"
tree = DecisionTreeClassifier(max_depth=6, random_state=0)
tree.fit(pid[tr], labels[tr])
pos_acc = float((tree.predict(pid[te]) == labels[te]).mean())
results["position_only"] = {
    "probe": "decision tree (depth 6) on paragraph index alone",
    "random_split_test_acc": round(pos_acc, 4),
}
print("position-only probe, random split:", round(pos_acc, 4), file=sys.stderr)

results["protocol"] = {
    "corpus": "LD-100K, 416 body paragraphs, 12 chapters/labels",
    "loco": "each fold holds out one whole chapter for testing, trains on "
            "the other 11; pooled accuracy over all 416 paragraphs",
    "random_split_reference": "train on train+dev (220), test on the "
                              "stratified test side (196), seed 20261003",
    "probes": "TF-IDF + linear SVM, word 1-2 grams and char_wb 2-5 grams",
    "leakage_diagnosis": "within-chapter adjacent-pair sharing across the "
                         "random split: 67.8% of same-chapter adjacent "
                         "paragraph pairs land on opposite sides",
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1)
print("wrote", OUT, file=sys.stderr)
