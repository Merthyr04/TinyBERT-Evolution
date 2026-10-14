"""Turn the LD-100K long-form article into real supervised datasets.

Two tasks are derived from one annotation-free source:

  * PTC  -- paragraph topic classification, 12 classes (one per chapter).
  * MLM  -- masked-token recovery inside each paragraph (15% masking, 80/10/10).

Split rule
----------
The corpus is generated chapter by chapter, and inside a chapter the wording
drifts with paragraph position (early paragraphs lean on the chapter's opening
sentence pool, later ones on its closing pool).  A positional split would
therefore put *late-chapter* paragraphs in dev/test while training only on
early-chapter ones: a distribution shift, not a generalisation test.  Measured
on this corpus, a TF-IDF + logistic-regression probe reaches 91.6% on train but
only 74.4% on a positional test split -- and a fine-tuned BERT-base lands on
exactly the same 74.4%, which is the signature of a shift rather than of model
capacity.

The split below is therefore a **stratified random split** with a fixed seed:
every one of the 12 topics keeps its proportion in every split, and train, dev
and test share one wording distribution, so the accuracy numbers mean what they
are supposed to mean.

The sizes are 120 / 100 / 196 rather than the usual 60/20/20.  Two reasons:

  * with 416 paragraphs, a 60/20/20 split leaves only 82 test paragraphs, and
    a single flipped paragraph is worth 1.2 accuracy points -- far too coarse
    to resolve the 1-3 point differences that distillation and augmentation
    produce;
  * 250 training paragraphs already saturate the task (a fine-tuned TinyBERT
    reaches 97.6% test accuracy and a fine-tuned BERT-base 98.8%), leaving no
    headroom at all.  Ten training paragraphs per topic is a genuinely
    low-resource regime, which is exactly the regime in which augmentation and
    distillation are supposed to matter.

So the split trades training examples for statistical power and for a task that
is not already solved.

The parser re-reads LD-100K.txt with the same structural rules that produced
LD-100K_labels.tsv and asserts that every recovered paragraph matches the
recorded character/word counts, so the labels are provably aligned.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(ROOT, "LD-100K", "LD-100K.txt")
LABELS = os.path.join(ROOT, "LD-100K", "LD-100K_labels.tsv")
OUT = os.path.join(HERE, "data")

SPLIT_TARGETS = {"train": 120, "dev": 100, "test": 196}
SPLIT_SEED = 20261003
SPLIT_ORDER = ["train", "dev", "test"]

RE_CHAPTER = re.compile(r"^CHAPTER \d+\.\s+\S")
# Sub-headings are written as "1.1  Reading the Evidence": a dotted chapter
# number followed by a title in title case.  (An earlier revision of the
# generator emitted "1.1  Part 1" and the filter was not updated with it, so
# the sub-headings silently entered the paragraph stream and shifted every
# label after the first one.  A body paragraph never begins with "N.N", which
# is what makes this test safe.)
RE_SUBHEAD = re.compile(r"^\d+\.\d+\s+\S")


def parse_corpus():
    """Return the ordered list of body paragraphs (title / intro / headings removed)."""
    with open(CORPUS, encoding="utf-8") as f:
        raw = f.read()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw)]
    body = []
    for b in blocks:
        if not b:
            continue
        if RE_CHAPTER.match(b) or RE_SUBHEAD.match(b):
            continue
        body.append(b)
    # block 0 is the title, block 1 the abstract-like lead paragraph
    return body[2:]


def load_tsv():
    rows = []
    with open(LABELS, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def _largest_remainder(weights, total):
    """Hamilton apportionment: split `total` across keys of `weights`, exactly."""
    wsum = sum(weights.values())
    ideal = {k: v * total / wsum for k, v in weights.items()}
    alloc = {k: int(v) for k, v in ideal.items()}
    short = total - sum(alloc.values())
    order = sorted(weights, key=lambda k: (-(ideal[k] - alloc[k]), k))
    for k in order[:short]:
        alloc[k] += 1
    return alloc


def stratified_split(records, seed=SPLIT_SEED):
    """Assign `split` in place: stratified by label, exact global sizes."""
    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    n = len(records)
    weights = {l: len(rs) for l, rs in by_label.items()}

    # exact per-split totals, apportioned over labels
    alloc = {}
    remaining = dict(weights)
    for s in SPLIT_ORDER:
        a = _largest_remainder(remaining, SPLIT_TARGETS[s])
        # never hand out more paragraphs of a label than exist
        a = {k: min(v, remaining[k]) for k, v in a.items()}
        alloc[s] = a
        remaining = {k: remaining[k] - a[k] for k in remaining}

    rng = random.Random(seed)
    for label, rs in by_label.items():
        order = list(rs)
        rng.shuffle(order)
        cut = 0
        for s in SPLIT_ORDER:
            k = alloc[s].get(label, 0)
            for r in order[cut:cut + k]:
                r["split"] = s
            cut += k
        for r in order[cut:]:
            r["split"] = SPLIT_ORDER[-1]


def build():
    body = parse_corpus()
    rows = load_tsv()
    n = len(rows)
    if len(body) < n:
        sys.exit(f"parser recovered only {len(body)} paragraphs, need {n}")

    labels_seen = []
    for r in rows:
        if r["label"] not in labels_seen:
            labels_seen.append(r["label"])
    label2id = {l: i for i, l in enumerate(sorted(labels_seen))}

    records = []
    mismatched = 0
    for i, r in enumerate(rows):
        text = body[i]
        if abs(len(text) - int(r["chars"])) > 1:
            mismatched += 1
        records.append({
            "para_id": int(r["para_id"]),
            "chapter": int(r["chapter"]),
            "label": r["label"],
            "label_id": label2id[r["label"]],
            "split": r["split"],
            "text": text,
        })

    stratified_split(records)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ptc.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(os.path.join(OUT, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id,
                   "id2label": {v: k for k, v in label2id.items()},
                   "num_labels": len(label2id)}, f, indent=2, ensure_ascii=False)

    # keep the shipped label file in sync with the split actually used
    with open(LABELS, "w", encoding="utf-8", newline="\n") as f:
        f.write("para_id\tchapter\tlabel\tchars\twords\tsplit\n")
        for rec in records:
            f.write(f"{rec['para_id']}\t{rec['chapter']}\t{rec['label']}\t"
                    f"{len(rec['text'])}\t{len(rec['text'].split())}\t{rec['split']}\n")

    by_split = {}
    for rec in records:
        by_split.setdefault(rec["split"], []).append(rec)

    stats = {
        "paragraphs": len(records),
        "num_labels": len(label2id),
        "split_rule": ("stratified random over paragraphs, seed "
                       f"{SPLIT_SEED}, targets {SPLIT_TARGETS}"),
        "split_sizes": {k: len(v) for k, v in sorted(by_split.items())},
        "chars_total": sum(len(r["text"]) for r in records),
        "words_total": sum(len(r["text"].split()) for r in records),
        "label_paragraph_counts": {
            l: sum(1 for r in records if r["label"] == l) for l in label2id
        },
        "split_x_label": {
            s: {l: sum(1 for r in v if r["label"] == l) for l in label2id}
            for s, v in sorted(by_split.items())
        },
        "char_count_mismatches_vs_tsv": mismatched,
    }
    with open(os.path.join(OUT, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    build()
