"""A second, independent annotation pass over the OD-100K stratified sample.

`organic_agreement.py` produces the sheet a human second annotator fills in and
measures what a *keyword rule* recovers from surface cues (kappa = 0.45).  This
script adds a third measurement that sits between the two: a **model-based**
second annotation, produced by reading each entry and assigning one of the same
twelve classes under the same codebook, with the first label hidden.

Why it is worth having.  The keyword rule answers "how much of the label is
recoverable from a handful of words".  It cannot answer "would a careful reader,
given the codebook and nothing else, arrive at the same class", which is what a
second annotator is for.  A model pass does answer that, and its disagreements
are the interesting part: they mark the entries where the codebook itself, and
not the surface vocabulary, decides the class.

What it is not.  It is **not** human agreement, and the paper must not describe
it as such.  The released sheet keeps an empty `second_label` column precisely
so that a person can still do the check; this pass is reported alongside it, with
its provenance stated, or not at all.

Provenance of `MODEL_LABELS` below.  The 119 entries were read in sample order
with the first label hidden, and one label was assigned per entry from the
twelve-class codebook, using a single stated policy: label by the entity's
principal domain as the entry presents it -- for a person, the occupation or
achievement the entry leads with and spends most of its words on; for a place,
geography, unless the entry is entirely about an event that happened there; for
an abstract term, the domain of the term.

Writes results/od100k_model_pass.json and results/od100k_model_pass.md.
"""
from __future__ import annotations

import json
import os

import numpy as np

import organic_agreement as OA

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")

# Sample order, i.e. the order of data/od100k_secondpass_sample.json, which
# organic_agreement.py fixes with a seeded shuffle.  Do not reorder.
MODEL_LABELS = [
    "history_politics",            # 000 [5]   Aberdeen, Earl of
    "myth_legend",                 # 001 [7]   Absyrtus
    "science_nature",              # 002 [23]  Adipose tissue
    "war_military",                # 003 [25]  Admiral
    "war_military",                # 004 [33]  Agoust, Capt. de
    "philosophy_scholarship",      # 005 [34]  Agricola, Rudolphus
    "history_politics",            # 006 [36]  Agrippina
    "literature_writing",          # 007 [43]  Ainsworth, W. H.
    "literature_writing",          # 008 [44]  Aird, Thomas
    "science_nature",              # 009 [45]  Airy, Sir G. B.
    "religion_theology",           # 010 [46]  Ajalon, Valley of
    "war_military",                # 011 [51]  Alabama, The
    "science_nature",              # 012 [58]  Albategni
    "literature_writing",          # 013 [68]  Aldine editions
    "war_military",                # 014 [77]  Alexander Nevsky
    "history_politics",            # 015 [80]  Alexis Michaelovitch
    "peoples_languages",           # 016 [87]  Algonquins
    "history_politics",            # 017 [105] Almanach de Gotha
    "war_military",                # 018 [107] Almansur
    "science_nature",              # 019 [113] Alpaca
    "war_military",                # 020 [115] Alva, Duke of
    "myth_legend",                 # 021 [119] Amazons
    "philosophy_scholarship",      # 022 [122] Amberley, Lord
    "law_education_institutions",  # 023 [124] Amende honorable
    "commerce_industry",           # 024 [125] Amerbach, Johann
    "geography_places",            # 025 [126] America
    "peoples_languages",           # 026 [130] American Indians
    "philosophy_scholarship",      # 027 [138] Anacharsis
    "literature_writing",          # 028 [140] Ancient Mariner
    "history_politics",            # 029 [146] Andreossy, Count
    "religion_theology",           # 030 [147] Andrew, St.
    "art_music",                   # 031 [153] Anichini
    "geography_places",            # 032 [158] Anstruther, East and West
    "religion_theology",           # 033 [168] Antony of Padua
    "myth_legend",                 # 034 [170] Apes, Dead Sea
    "myth_legend",                 # 035 [172] Apollo
    "philosophy_scholarship",      # 036 [173] Apollonius of Tyana
    "religion_theology",           # 037 [176] Apostolic succession
    "myth_legend",                 # 038 [177] Apple of Discord
    "geography_places",            # 039 [186] Arctic Ocean
    "literature_writing",          # 040 [187] Arden, Enoch
    "science_nature",              # 041 [192] Aries
    "philosophy_scholarship",      # 042 [193] Aristarchus of Samothrace
    "peoples_languages",           # 043 [196] Armenians
    "literature_writing",          # 044 [198] Armstrong, John
    "commerce_industry",           # 045 [199] Armstrong, William George
    "art_music",                   # 046 [203] Arne, Thomas Augustine
    "literature_writing",          # 047 [207] Arnold, Sir Edwin
    "religion_theology",           # 048 [209] Arnold of Brescia
    "war_military",                # 049 [210] Arnold of Winkelried
    "history_politics",            # 050 [211] Artevelde, Philip van
    "philosophy_scholarship",      # 051 [216] Arundel Marbles
    "religion_theology",           # 052 [220] Askew, Anne
    "law_education_institutions",  # 053 [223] Assarotti
    "philosophy_scholarship",      # 054 [227] Assemani, Giuseppe
    "religion_theology",           # 055 [228] Asser, John
    "myth_legend",                 # 056 [230] Astolfo
    "philosophy_scholarship",      # 057 [235] Astruc, Jean
    "religion_theology",           # 058 [237] Athalia
    "religion_theology",           # 059 [239] Atheism
    "religion_theology",           # 060 [240] Atheism, Modern
    "religion_theology",           # 061 [243] Atterbury, Francis
    "war_military",                # 062 [245] Aubusson, Pierre d'
    "art_music",                   # 063 [249] Audran, Gerard
    "history_politics",            # 064 [252] August
    "history_politics",            # 065 [256] Aulic Council
    "geography_places",            # 066 [257] Aulis
    "history_politics",            # 067 [258] Aumale, Duc d'
    "history_politics",            # 068 [259] Aumale, Duc d'
    "literature_writing",          # 069 [261] Ausonius
    "law_education_institutions",  # 070 [263] Austin, John
    "peoples_languages",           # 071 [266] Avars
    "geography_places",            # 072 [267] Avebury
    "religion_theology",           # 073 [272] Aylmer, John
    "history_politics",            # 074 [276] Azeglio, Marchese d'
    "religion_theology",           # 075 [284] Babylonish Captivity
    "religion_theology",           # 076 [291] Badrinath
    "geography_places",            # 077 [294] Baffin Bay
    "philosophy_scholarship",      # 078 [300] Bailey, Samuel
    "art_music",                   # 079 [302] Baily, E. H.
    "war_military",                # 080 [304] Baird, Sir David
    "geography_places",            # 081 [307] Baker, Sir Samuel White
    "commerce_industry",           # 082 [309] Balance of trade
    "history_politics",            # 083 [312] Baldwin I.
    "history_politics",            # 084 [315] Balfour, A. J.
    "law_education_institutions",  # 085 [316] Balfour, Sir James
    "law_education_institutions",  # 086 [318] Baliol, Sir John de
    "science_nature",              # 087 [320] Ball, Sir R. S.
    "myth_legend",                 # 088 [322] Balmung
    "art_music",                   # 089 [334] Banks, Thomas
    "science_nature",              # 090 [336] Banting system
    "peoples_languages",           # 091 [337] Bantu
    "literature_writing",          # 092 [342] Barataria
    "peoples_languages",           # 093 [344] Barbarians
    "literature_writing",          # 094 [347] Barbauld, Anna Laetitia
    "commerce_industry",           # 095 [352] Baring, Sir Francis
    "myth_legend",                 # 096 [355] Barlaam and Josaphat
    "religion_theology",           # 097 [358] Barnabites
    "commerce_industry",           # 098 [361] Barnum
    "literature_writing",          # 099 [364] Barrie, James Matthew
    "art_music",                   # 100 [370] Barry, James
    "art_music",                   # 101 [371] Bartholdi
    "commerce_industry",           # 102 [373] Bartholomew Fair
    "law_education_institutions",  # 103 [374] Bartholomew Hospital
    "peoples_languages",           # 104 [375] Bartlett, John H.
    "religion_theology",           # 105 [377] Baruch
    "law_education_institutions",  # 106 [379] Basedow, Johann Bernard
    "peoples_languages",           # 107 [385] Basques
    "art_music",                   # 108 [387] Bassano, Jacopo da Ponte
    "peoples_languages",           # 109 [388] Bastian, Adolf
    "geography_places",            # 110 [393] Battersea
    "art_music",                   # 111 [394] Baudry, Paul
    "philosophy_scholarship",      # 112 [395] Baumgarten
    "geography_places",            # 113 [397] Bautzen
    "history_politics",            # 114 [398] Bazard, Saint-Amand
    "geography_places",            # 115 [400] Beachy Head
    "commerce_industry",           # 116 [402] Bear
    "history_politics",            # 117 [407] Beaufort, Duke of
    "history_politics",            # 118 [414] Beauvillier
]


def main():
    with open(os.path.join(DATA, "od100k_secondpass_sample.json"),
              encoding="utf-8") as f:
        sample = json.load(f)
    if len(sample) != len(MODEL_LABELS):
        raise SystemExit("od100k_model_pass.py: %d entries in the sample but "
                         "%d model labels" % (len(sample), len(MODEL_LABELS)))
    unknown = sorted({l for l in MODEL_LABELS if l not in OA.CLASSES})
    if unknown:
        raise SystemExit("od100k_model_pass.py: unknown classes %r" % unknown)

    gold = [r["label_1"] for r in sample]
    rule = [OA.auto_label(r["text"]) for r in sample]
    model = list(MODEL_LABELS)

    k_model, po_m, pe_m = OA.cohen_kappa(gold, model, OA.CLASSES)
    k_rule, po_r, pe_r = OA.cohen_kappa(gold, rule, OA.CLASSES)
    k_mr, po_mr, pe_mr = OA.cohen_kappa(rule, model, OA.CLASSES)

    per_class = {}
    for c in OA.CLASSES:
        idxs = [i for i, g in enumerate(gold) if g == c]
        if idxs:
            per_class[c] = round(100.0 * np.mean([gold[i] == model[i]
                                                  for i in idxs]), 1)

    disagree = [{"idx": sample[i]["idx"], "label_1": gold[i],
                 "label_2_model": model[i], "label_2_rule": rule[i],
                 "headword": sample[i]["text"].split(",")[0][:48]}
                for i in range(len(sample)) if gold[i] != model[i]]

    both_wrong = [d for d in disagree
                  if d["label_1"] != d["label_2_rule"]]
    out = {
        "sample_size": len(sample),
        "per_class": OA.PER_CLASS,
        "seed": OA.SEED,
        "model": {
            "kappa": round(k_model, 3),
            "observed_agreement": round(100 * po_m, 2),
            "expected_agreement": round(100 * pe_m, 2),
            "n_agree": sum(1 for a, b in zip(gold, model) if a == b),
            "per_class_agreement": per_class,
            "provenance": ("an independent model-based annotation of the same "
                           "119 entries under the same twelve-class codebook, "
                           "read with the first label hidden; NOT human "
                           "agreement"),
        },
        "rule": {"kappa": round(k_rule, 3),
                 "observed_agreement": round(100 * po_r, 2),
                 "expected_agreement": round(100 * pe_r, 2)},
        "rule_vs_model": {"kappa": round(k_mr, 3),
                          "observed_agreement": round(100 * po_mr, 2),
                          "expected_agreement": round(100 * pe_mr, 2)},
        "disagreements": disagree,
        "cases_both_second_passes_miss": len(both_wrong),
        "human_second_annotation": ("still outstanding; the released sheet "
                                    "data/od100k_secondpass_sample.json carries "
                                    "an empty second_label column"),
    }
    with open(os.path.join(RES, "od100k_model_pass.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    L = ["# OD-100K second-annotation passes (round 75)\n",
         "Stratified sample: %d entries, %d per class, seed %d.  The first "
         "label is the author's, made while reading the whole encyclopaedia."
         % (len(sample), OA.PER_CLASS, OA.SEED), "",
         "| second pass | observed agreement | expected agreement | Cohen's "
         "kappa |", "|---|---|---|---|",
         "| keyword rule | %.1f%% | %.1f%% | %.2f |" % (100 * po_r, 100 * pe_r, k_rule),
         "| model annotation | %.1f%% | %.1f%% | %.2f |" % (100 * po_m, 100 * pe_m, k_model),
         "| rule vs model | %.1f%% | %.1f%% | %.2f |" % (100 * po_mr, 100 * pe_mr, k_mr),
         "",
         "Per-class agreement of the model pass with the first annotation:", ""]
    for c in OA.CLASSES:
        if c in per_class:
            L.append("* %s: %.1f%%" % (c, per_class[c]))
    L += ["",
          "%d of %d entries disagree with the first annotation; in %d of them "
          "the keyword rule disagrees as well, so those two passes do not "
          "corroborate each other." % (len(disagree), len(sample), len(both_wrong)),
          "",
          "## Entries where the model pass disagrees", "",
          "| idx | headword | first label | model | keyword rule |",
          "|---|---|---|---|---|"]
    for d in disagree:
        L.append("| %d | %s | %s | %s | %s |"
                 % (d["idx"], d["headword"], d["label_1"], d["label_2_model"],
                    d["label_2_rule"]))
    L.append("")
    md = "\n".join(L)
    with open(os.path.join(RES, "od100k_model_pass.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    print(md.split("## Entries where")[0])


if __name__ == "__main__":
    main()
