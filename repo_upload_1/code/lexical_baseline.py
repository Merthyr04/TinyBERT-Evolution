"""Lexical baselines on the real 120/100/196 split.

The paper reports a fine-tuned TinyBERT at 88.8% and a fine-tuned BERT-base at
96.2% on LD-100K, on a training set of 120 paragraphs.  Before those numbers
can be read as evidence about distillation, a reader needs to know how much of
the task is lexical: the topic classes are derived from chapter structure, so a
bag-of-words model with no context may already do most of the work.

This script measures, on exactly the split of the paper (data/ptc.jsonl):

  * the majority-class baseline;
  * a TF-IDF + logistic-regression probe;
  * a TF-IDF + linear-SVM probe;
  * a nearest-class-centroid (Rocchio) probe over TF-IDF vectors, which is the
    most explicitly lexical of the four: it predicts the class whose mean
    training vector is closest in cosine.

It also re-runs the probe under a positional split (paragraphs within a chapter
in reading order, early ones for training, late ones for testing), which is the
comparison Section 4.1 of the paper uses to argue against a positional split,
so that the argument can be checked against a stated definition.

Outputs results/lexical_baseline.json and results/lexical_baseline.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

DATA = os.path.join(C.DATA, "ptc.jsonl")
JSON_OUT = os.path.join(C.RESULTS, "lexical_baseline.json")
MD_OUT = os.path.join(C.RESULTS, "lexical_baseline.md")
CORPUS_NAME = "LD-100K"


def load():
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def macro_f1(pred, gold):
    f1 = []
    for c in np.unique(gold):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1))


def _dense(X):
    return np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)


def centroid_pred(Xtr, ytr, Xte):
    """Nearest class centroid in cosine: the most lexical possible probe."""
    Xtr = _dense(Xtr)
    Xte = _dense(Xte)
    classes = np.unique(ytr)
    cent = np.vstack([Xtr[ytr == c].mean(axis=0) for c in classes])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    Xn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ cent.T
    return classes[sim.argmax(axis=1)]


def evaluate(name, fit_predict, recs_tr, recs_ev, extra_recs=()):
    texts_tr = [r["text"] for r in recs_tr]
    y_tr = np.asarray([r["label_id"] for r in recs_tr])
    texts_ev = [r["text"] for r in recs_ev]
    y_ev = np.asarray([r["label_id"] for r in recs_ev])
    pred_ev, pred_tr, vec = fit_predict(texts_tr, y_tr, texts_ev, texts_tr)
    out = {
        "name": name,
        "n_train": len(texts_tr), "n_eval": len(texts_ev),
        "acc_eval": float((pred_ev == y_ev).mean()),
        "f1_eval": macro_f1(pred_ev, y_ev),
        "acc_train": float((pred_tr == y_tr).mean()),
        "f1_train": macro_f1(pred_tr, y_tr),
        "vocab": int(len(vec.vocabulary_)),
    }
    for tag, rs in extra_recs:
        t = [r["text"] for r in rs]
        y = np.asarray([r["label_id"] for r in rs])
        p = fit_predict(texts_tr, y_tr, t, t)[0]
        out["acc_" + tag] = float((p == y).mean())
        out["f1_" + tag] = macro_f1(p, y)
    return out


def tfidf_lr(Cp, ngram=(1, 2)):
    def f(texts_tr, y_tr, texts_ev, texts_tr_eval):
        vec = TfidfVectorizer(ngram_range=ngram, sublinear_tf=True,
                              min_df=1, lowercase=True)
        Xtr = vec.fit_transform(texts_tr)
        clf = LogisticRegression(C=Cp, max_iter=5000)
        clf.fit(Xtr, y_tr)
        return (clf.predict(vec.transform(texts_ev)),
                clf.predict(Xtr), vec)
    return f


def tfidf_svc(Cp=1.0, ngram=(1, 2)):
    def f(texts_tr, y_tr, texts_ev, texts_tr_eval):
        vec = TfidfVectorizer(ngram_range=ngram, sublinear_tf=True,
                              min_df=1, lowercase=True)
        Xtr = vec.fit_transform(texts_tr)
        clf = LinearSVC(C=Cp)
        clf.fit(Xtr, y_tr)
        return (clf.predict(vec.transform(texts_ev)),
                clf.predict(Xtr), vec)
    return f


def tfidf_centroid(ngram=(1, 2)):
    def f(texts_tr, y_tr, texts_ev, texts_tr_eval):
        vec = TfidfVectorizer(ngram_range=ngram, sublinear_tf=True,
                              min_df=1, lowercase=True)
        Xtr = vec.fit_transform(texts_tr)
        return (centroid_pred(Xtr, y_tr, vec.transform(texts_ev)),
                centroid_pred(Xtr, y_tr, Xtr), vec)
    return f


def positional_split(recs, head_frac):
    """Within each chapter, keep the first `head_frac` for training."""
    tr, te = [], []
    by_lab = {}
    for r in recs:
        by_lab.setdefault(r["label_id"], []).append(r)
    for lab, rs in by_lab.items():
        rs = sorted(rs, key=lambda r: r["para_id"])
        k = int(round(len(rs) * head_frac))
        tr.extend(rs[:k])
        te.extend(rs[k:])
    return tr, te


def main():
    global DATA, JSON_OUT, MD_OUT, CORPUS_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="path to the ptc-style jsonl corpus (default: data/ptc.jsonl)")
    ap.add_argument("--tag", default=None,
                    help="output suffix; results/lexical_baseline_<tag>.json/md")
    ap.add_argument("--name", default=None, help="corpus display name for the report")
    args = ap.parse_args()
    if args.data:
        DATA = args.data
    if args.tag:
        JSON_OUT = os.path.join(C.RESULTS, f"lexical_baseline_{args.tag}.json")
        MD_OUT = os.path.join(C.RESULTS, f"lexical_baseline_{args.tag}.md")
    if args.name:
        CORPUS_NAME = args.name

    recs = load()
    tr = [r for r in recs if r["split"] == "train"]
    dev = [r for r in recs if r["split"] == "dev"]
    te = [r for r in recs if r["split"] == "test"]
    print(f"train {len(tr)}  dev {len(dev)}  test {len(te)}")

    rep = {"split": {"train": len(tr), "dev": len(dev), "test": len(te)},
           "real_split": [], "positional": [], "ceiling": {}}

    # ---- majority class ---------------------------------------------------
    counts = {}
    for r in tr:
        counts[r["label_id"]] = counts.get(r["label_id"], 0) + 1
    maj = max(counts, key=counts.get)
    y_te = np.asarray([r["label_id"] for r in te])
    y_dev = np.asarray([r["label_id"] for r in dev])
    y_tr = np.asarray([r["label_id"] for r in tr])
    rep["real_split"].append({
        "name": f"majority class ({maj})",
        "n_train": len(tr), "n_eval": len(te),
        "acc_eval": float((y_te == maj).mean()), "f1_eval": macro_f1(
            np.full_like(y_te, maj), y_te),
        "acc_train": float((y_tr == maj).mean()),
        "f1_train": macro_f1(np.full_like(y_tr, maj), y_tr),
        "acc_dev": float((y_dev == maj).mean()), "f1_dev": macro_f1(
            np.full_like(y_dev, maj), y_dev),
        "vocab": 0,
    })

    # ---- lexical probes on the real split ---------------------------------
    for name, fn in (("TF-IDF (1-2) + logistic regression, C = 1",
                      tfidf_lr(1.0)),
                     ("TF-IDF (1-2) + logistic regression, C = 10",
                      tfidf_lr(10.0)),
                     ("TF-IDF (1-2) + logistic regression, C = 100",
                      tfidf_lr(100.0)),
                     ("TF-IDF (1-2) + linear SVM, C = 1", tfidf_svc(1.0)),
                     ("TF-IDF (1-1) + logistic regression, C = 10",
                      tfidf_lr(10.0, ngram=(1, 1))),
                     ("TF-IDF nearest class centroid (Rocchio)",
                      tfidf_centroid())):
        cell = evaluate(name, fn, tr, te, extra_recs=[("dev", dev)])
        rep["real_split"].append(cell)

    # ---- positional variants ---------------------------------------------
    for frac in (0.5, 0.6, 0.7, 0.8):
        p_tr, p_te = positional_split(recs, frac)
        cell = evaluate(f"positional head {frac:.0%} (TF-IDF + LR, C = 10)",
                        tfidf_lr(10.0), p_tr, p_te)
        cell["head_frac"] = frac
        rep["positional"].append(cell)

    C.json_dump(rep, JSON_OUT)

    L = [f"# Lexical baselines on {CORPUS_NAME}\n",
         f"Split as published: {len(tr)} / {len(dev)} / {len(te)}.\n",
         "## Real split (train on the 120 published training paragraphs)\n",
         "| probe | vocab | train acc (%) | dev acc (%) | test acc (%) | test macro-F1 (%) |",
         "|---|---|---|---|---|---|"]
    for c in rep["real_split"]:
        dev_s = "-" if "acc_dev" not in c else f"{c['acc_dev'] * 100:.1f}"
        L.append(f"| {c['name']} | {c['vocab']} | {c['acc_train'] * 100:.1f} | "
                 f"{dev_s} | {c['acc_eval'] * 100:.1f} | "
                 f"{c['f1_eval'] * 100:.1f} |")
    L.append("\n## Positional split (early paragraphs per chapter for training)\n")
    L.append("| definition | n train | n test | train acc (%) | test acc (%) |")
    L.append("|---|---|---|---|---|")
    for c in rep["positional"]:
        L.append(f"| first {c['head_frac']:.0%} of each chapter | {c['n_train']} | "
                 f"{c['n_eval']} | {c['acc_train'] * 100:.1f} | "
                 f"{c['acc_eval'] * 100:.1f} |")
    text = "\n".join(L) + "\n"
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
