"""Inter-annotator agreement machinery for the organic (OD-100K) corpus.

R4's fourth major point is that the twelve OD-100K labels were assigned by a
single reader, so the truth quality of the replication corpus is unquantified.
Two things are produced here:

  1. `data/od100k_secondpass_sample.json` -- a stratified subsample of 120
     entries (ten per class, fixed seed) with an empty `label_2` field.  This
     is the sheet a *human* second annotator fills in; the selection is
     deterministic so the sample can be described in the paper.

  2. An **automatic** second pass over the same sample, using a transparent
     keyword rule over the headword and definition, and its Cohen's kappa
     against the recorded labels.  This measures how much of the label is
     recoverable from surface cues and is reported as such -- it is *not* a
     substitute for human agreement, and the paper must say so.

Writes results/organic_agreement.json and results/organic_agreement.md.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import od100k_labels_part1 as P1  # noqa: E402
import od100k_labels_part2 as P2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
SEED = 20261011
PER_CLASS = 10

CLASSES = ["geography_places", "history_politics", "religion_theology",
           "literature_writing", "science_nature", "art_music",
           "war_military", "philosophy_scholarship", "myth_legend",
           "law_education_institutions", "commerce_industry",
           "peoples_languages"]

# Transparent keyword rule.  Order matters: the first class whose pattern
# matches wins, so the more specific vocabularies come first.
RULES = [
    ("war_military", r"\b(battle|war|siege|army|navy|regiment|general|"
                     r"admiral|troops|invasion|campaign|surrender|fortress|"
                     r"cavalry|artillery|fleet|soldier|colonel|captain)\b"),
    ("myth_legend", r"\b(myth|legend|deity|god|goddess|hero|nymph|"
                    r"fabulous|epic|titan|oracle|temple of)\b"),
    ("religion_theology", r"\b(theolog|church|bishop|pope|saint|monastery|"
                          r"priest|christian|protestant|catholic|clergy|"
                          r"sect|abbot|reformation|monk|puritan|gospel|"
                          r"biblical|jewish|muslim|hindu|buddh)\w*"),
    ("law_education_institutions", r"\b(law|legal|statute|act of|parliament|"
                                   r"court|jurist|university|college|school|"
                                   r"academy|institution|constitution|"
                                   r"treaty|charter)\w*"),
    ("commerce_industry", r"\b(commerc|trade|merchant|manufactur|industry|"
                          r"railway|bank|company|shipping|export|import|"
                          r"market|factory|steamship)\w*"),
    ("peoples_languages", r"\b(language|dialect|tribe|race|people|nation of|"
                           r"grammar|alphabet|tongue|ethn)\w*"),
    ("science_nature", r"\b(science|scientist|physics|chemistry|biology|"
                       r"astronom|geolog|botan|zoolog|mathematic|naturalist|"
                       r"species|mineral|plant|animal|discover)\w*"),
    ("art_music", r"\b(paint|sculpt|artist|composer|musician|music|opera|"
                  r"architect|portrait|symphony|gallery|engraver)\w*"),
    ("literature_writing", r"\b(poet|poem|novel|author|writer|literary|"
                           r"literature|essay|dramatist|playwright|editor|"
                           r"journalist|translat)\w*"),
    ("philosophy_scholarship", r"\b(philosoph|scholar|metaphysic|logic|"
                               r"ethic|academic|professor|historian|"
                               r"antiquary|lexicograph)\w*"),
    ("geography_places", r"\b(river|mountain|town|city|island|province|"
                         r"district|county|village|port|valley|peninsula|"
                         r"capital of|lake|coast|cape|strait)\b"),
    ("history_politics", r"\b(king|queen|emperor|reign|dynasty|minister|"
                         r"statesman|politician|duke|earl|baron|republic|"
                         r"empire|revolution|throne|governor)\b"),
]


def human_labels():
    if len(P1.LABELS) + len(P2.LABELS2) != 416:
        raise SystemExit("label parts do not cover the 416 entries")
    return list(P1.LABELS) + list(P2.LABELS2)


def auto_label(text):
    t = text.lower()
    for name, pat in RULES:
        if re.search(pat, t):
            return name
    return "history_politics"


def cohen_kappa(a, b, classes):
    idx = {c: i for i, c in enumerate(classes)}
    n = len(a)
    m = np.zeros((len(classes), len(classes)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = float((m.sum(0) * m.sum(1)).sum()) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po, pe


def main():
    rows = []
    with open(os.path.join(DATA, "organic_entries.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    labels = human_labels()
    for r, lab in zip(rows, labels):
        r["label"] = lab

    rng = random.Random(SEED)
    by_class = {c: [] for c in CLASSES}
    for r in rows:
        by_class[r["label"]].append(r)
    sample = []
    for c in CLASSES:
        pool = sorted(by_class[c], key=lambda r: r["idx"])
        rng.shuffle(pool)
        sample.extend(pool[:PER_CLASS])
    sample.sort(key=lambda r: r["idx"])
    for r in sample:
        r["label_2"] = ""       # to be filled by a second annotator

    with open(os.path.join(DATA, "od100k_secondpass_sample.json"), "w",
              encoding="utf-8") as f:
        json.dump([{"idx": r["idx"], "text": r["text"], "label_1": r["label"],
                    "label_2": ""} for r in sample], f, ensure_ascii=False,
                  indent=1)

    auto = [auto_label(r["text"]) for r in sample]
    gold = [r["label"] for r in sample]
    kappa, po, pe = cohen_kappa(gold, auto, CLASSES)
    per_class = {}
    for c in CLASSES:
        idxs = [i for i, g in enumerate(gold) if g == c]
        if idxs:
            per_class[c] = round(100.0 * np.mean([gold[i] == auto[i]
                                                  for i in idxs]), 1)

    out = dict(sample_size=len(sample), per_class=PER_CLASS,
               rule_kappa=round(kappa, 3), observed_agreement=round(100 * po, 2),
               expected_agreement=round(100 * pe, 2),
               per_class_recovery=per_class,
               sample_file="data/od100k_secondpass_sample.json",
               note=("The automatic pass is a keyword rule, not a person; it "
                     "bounds how much of the label is recoverable from surface "
                     "cues and does not replace human agreement."))
    with open(os.path.join(RES, "organic_agreement.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    L = ["# OD-100K annotation agreement (round 75)\n",
         f"Stratified sample: {len(sample)} entries, {PER_CLASS} per class,"
         f" seed {SEED}.\n",
         f"**Automatic second pass (keyword rule): Cohen's kappa = "
         f"{out['rule_kappa']}**, observed agreement {out['observed_agreement']}%,"
         f" expected {out['expected_agreement']}%.\n",
         "\n> A human second annotator is still required; the sheet to fill in"
         " is `data/od100k_secondpass_sample.json`.\n",
         "\n| class | how often the rule recovers the recorded label |",
         "|---|---|"]
    for c, v in per_class.items():
        L.append(f"| {c} | {v}% |")
    with open(os.path.join(RES, "organic_agreement.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
