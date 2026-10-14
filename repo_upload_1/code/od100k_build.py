# -*- coding: utf-8 -*-
"""Build the OD-100K organic corpus in the LD-100K format.

* paragraphs  : the 416 entries selected by od100k_select.py, in the volume's
                alphabetical order
* `text`      : the entry's definition with the headword removed -- the exact
                analogue of LD-100K, where chapter and sub-headings are removed
                at inference.  `text_full` keeps the headword for reference.
* labels      : 12 subject classes, assigned by reading each entry
                (od100k_labels_part1/2.py), on the taxonomy the volume itself
                states on its title page and in its preface
* split       : stratified 120 / 100 / 196, seed 20261003, as in LD-100K

Writes data/organic_ptc.jsonl and data/organic_label_map.json.
"""
from __future__ import annotations

import collections
import io
import json
import os
import random
import re

from od100k_labels_part1 import LABELS, KEYS
from od100k_labels_part2 import LABELS2

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRIES = os.path.join(HERE, "data", "organic_entries.jsonl")
OUT = os.path.join(HERE, "data", "organic_ptc.jsonl")
OUT_MAP = os.path.join(HERE, "data", "organic_label_map.json")
OUT_STATS = os.path.join(HERE, "data", "organic_stats.json")

TARGET = {"train": 120, "dev": 100, "test": 196}
SEED = 20261003

# residual leading run of capitals left in the definition after the first
# headword was removed ("ABAUZIT, FIRMIN, a French ..." -> "FIRMIN, a French ...")
CAPS_HEAD = re.compile(r"^(?:[A-Z][A-Z0-9'’\.\-]*)(?:[ ,]+|$)")


def strip_headword(defn: str) -> str:
    s = defn.strip()
    while True:
        m = CAPS_HEAD.match(s)
        if not m:
            break
        rest = s[m.end():].strip()
        # only strip if what remains still looks like a definition
        if len(rest) < 25:
            break
        s = rest
    s = s.lstrip(",;: ").strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


def main():
    entries = [json.loads(l) for l in
               io.open(ENTRIES, encoding="utf-8")]
    labels = LABELS + LABELS2
    assert len(labels) == len(entries), (len(labels), len(entries))

    classes = sorted(set(labels))
    label2id = {c: i for i, c in enumerate(classes)}
    dist = collections.Counter(labels)
    print(f"entries {len(entries)}   classes {len(classes)}")
    print("class distribution:")
    for c in classes:
        print(f"  {c:32s} {dist[c]:4d}  ({dist[c] / len(entries) * 100:.1f}%)")

    recs = []
    for i, (e, lab) in enumerate(zip(entries, labels)):
        body = strip_headword(e["def"])
        recs.append({
            "para_id": i,
            "chapter": 0,
            "label": lab,
            "label_id": label2id[lab],
            "text": body,
            "text_full": e["text"],
            "head": e["head"],
            "chars": len(body),
        })

    # ---- stratified split, seed 20261003 --------------------------------
    rng = random.Random(SEED)
    by_class = collections.defaultdict(list)
    for r in recs:
        by_class[r["label"]].append(r["para_id"])
    for c in by_class:
        rng.shuffle(by_class[c])

    n = len(recs)
    remain = {k: int(round(TARGET[k] / sum(TARGET.values()) * n))
              for k in TARGET}
    # fix rounding: total must be n
    diff = n - sum(remain.values())
    remain["test"] += diff

    alloc = collections.defaultdict(dict)
    for c in classes:
        m = len(by_class[c])
        assign = {}
        for k in ("train", "dev", "test"):
            assign[k] = int(round(remain[k] / n * m))
        # correct the residual on the largest split
        resid = m - sum(assign.values())
        assign["test"] += resid
        alloc[c] = assign

    for c in classes:
        ids = by_class[c]
        pos = 0
        for k in ("train", "dev", "test"):
            take = alloc[c][k]
            for pid in ids[pos:pos + take]:
                recs[pid]["split"] = k
            pos += take
        for pid in ids[pos:]:
            recs[pid]["split"] = "test"      # safety net

    split_counts = collections.Counter(r["split"] for r in recs)
    print("split sizes:", dict(split_counts))
    xlab = collections.Counter((r["split"], r["label"]) for r in recs)
    for c in classes:
        print(f"  {c:32s} train {xlab[('train', c)]:3d}  dev {xlab[('dev', c)]:3d}"
              f"  test {xlab[('test', c)]:3d}")

    total_chars = sum(r["chars"] for r in recs)
    print(f"OD-100K: {len(recs)} paragraphs, {total_chars} characters "
          f"(mean {total_chars / len(recs):.1f})")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({k: r[k] for k in
                                ("para_id", "chapter", "label", "label_id",
                                 "text", "split")}, ensure_ascii=False) + "\n")
    with open(OUT_MAP, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id,
                   "id2label": {str(v): k for k, v in label2id.items()},
                   "num_labels": len(classes)}, f, indent=2)
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump({
            "source": "The Nuttall Encyclopaedia (Project Gutenberg #12342)",
            "paragraphs": len(recs),
            "characters": total_chars,
            "mean_chars_per_paragraph": round(total_chars / len(recs), 1),
            "num_labels": len(classes),
            "split_sizes": dict(split_counts),
            "label_paragraph_counts": {c: dist[c] for c in classes},
            "split_x_label": {f"{s}|{c}": xlab[(s, c)]
                              for s in ("train", "dev", "test") for c in classes},
            "split_rule": f"stratified random over paragraphs, seed {SEED}, "
                          f"targets {TARGET}",
            "headword_removed": True,
            "labelling": "assigned by reading each entry "
                         "(see od100k_labels_part1.py, od100k_labels_part2.py)",
        }, f, indent=2, ensure_ascii=False)
    print("wrote", OUT, OUT_MAP, OUT_STATS)


if __name__ == "__main__":
    main()
