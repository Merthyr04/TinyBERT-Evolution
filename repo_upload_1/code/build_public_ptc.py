"""Build the public-benchmark corpus for the Applied Intelligence revision.

R1 and R4 both asked for at least one public data set on which the paper's
claims can be located against published numbers.  The reviewer's suggestion
was "a low-resource public task, e.g. a GLUE subtask at ten examples per
class"; the organic replication already answers the *second corpus* question,
so what is needed here is a corpus that is (a) public, (b) outside the
authors' control, and (c) shaped like LD-100K.

AG News (Zhang, Zhao and LeCun, 2015; distributed by Hugging Face as
`SetFit/ag_news`) is used because its inputs are short news texts of the same
order as a 96-word-piece paragraph, it is a four-way topic task, and it has a
large reserve split, so the exact split cardinalities of LD-100K can be
reproduced from held-out material only.

Two properties of the construction matter for the paper:

  * every item comes from a *distinct* document, so the 120/100/196 split is
    a between-document split -- the objection that LD-100K trains and tests on
    the same article cannot be raised here;
  * only items that tokenise to at most 96 word-pieces are eligible, so the
    "no truncation" property of the protocol is preserved.

Writes data/agnews_ptc.jsonl (416 records) and data/agnews_label_map.json.
The source file is expected at ../pub/agnews_test.jsonl.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "pub", "agnews_test.jsonl")
OUT = os.path.join(HERE, "data", "agnews_ptc.jsonl")
MAP = os.path.join(HERE, "data", "agnews_label_map.json")

SEED = 20261011
N_TRAIN, N_DEV, N_TEST = 120, 100, 196
CLASSES = {0: "world", 1: "sports", 2: "business", 3: "sci_tech"}


def main():
    rng = random.Random(SEED)
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"source items: {len(rows)}")

    tok = AutoTokenizer.from_pretrained(C.resolve(C.TOKENIZER_EN))
    by_class = {c: [] for c in CLASSES}
    for r in rows:
        t = r["text"].replace("\\", " ").strip()
        if not t:
            continue
        ids = tok(t, truncation=False, add_special_tokens=True)["input_ids"]
        if len(ids) > C.MAX_LEN:            # keep the no-truncation property
            continue
        by_class[r["label"]].append(t)
    for c, v in by_class.items():
        print(f"  class {c} ({CLASSES[c]}): {len(v)} eligible")
        rng.shuffle(v)

    # 416 items, drawn per class so that the split cardinalities match LD-100K.
    per = {"train": N_TRAIN // len(CLASSES), "dev": N_DEV // len(CLASSES),
           "test": 196 // len(CLASSES)}
    recs = []
    pid = 0
    chapter = 0
    for c in CLASSES:
        pool = by_class[c]
        need = N_TRAIN // len(CLASSES) + N_DEV // len(CLASSES) + 196 // len(CLASSES)
        need += (N_TEST - (196 // len(CLASSES)) * len(CLASSES))
        taken = pool[:need]
        i = 0
        for split, n in (("train", per["train"]), ("dev", per["dev"]),
                         ("test", per["test"])):
            for _ in range(n):
                if i >= len(taken):
                    break
                pid += 1
                chapter += 1        # one document per item: between-document
                recs.append({"para_id": pid, "chapter": chapter,
                             "label": CLASSES[c], "label_id": c,
                             "split": split, "text": taken[i]})
                i += 1
    # top up the test split so that it is exactly 196 (4 classes, 49 each)
    short = N_TEST - sum(1 for r in recs if r["split"] == "test")
    k = 0
    while short > 0:
        c = k % len(CLASSES)
        pool = by_class[c]
        idx = N_TRAIN // len(CLASSES) + N_DEV // len(CLASSES) + (196 // len(CLASSES)) + k
        if idx < len(pool):
            pid += 1
            chapter += 1
            recs.append({"para_id": pid, "chapter": chapter,
                         "label": CLASSES[c], "label_id": c,
                         "split": "test", "text": pool[idx]})
            short -= 1
        k += 1

    order = {"train": 0, "dev": 1, "test": 2}
    recs.sort(key=lambda r: (order[r["split"]], r["label_id"], r["para_id"]))
    for i, r in enumerate(recs, 1):
        r["para_id"] = i

    with open(OUT, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(MAP, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in CLASSES.items()}, f, indent=1)

    counts = {}
    for r in recs:
        counts.setdefault(r["split"], []).append(r["label_id"])
    for s in ("train", "dev", "test"):
        v = counts.get(s, [])
        print(f"  {s}: {len(v)} items, per class "
              f"{[v.count(c) for c in CLASSES]}")
    print(f"wrote {OUT} ({len(recs)} records)")


if __name__ == "__main__":
    main()
