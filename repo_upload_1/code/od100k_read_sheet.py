"""Turn a filled second-annotator sheet into Cohen's kappa.

Reads data/od100k_secondpass_filled.csv -- the file the click-through sheet
downloads, or the same columns typed into a spreadsheet -- and reports the
agreement of the human second annotation with the author's first annotation,
with the keyword rule of `organic_agreement.py`, and with the model pass of
`od100k_model_pass.py`.

Usage:
    python od100k_read_sheet.py [path/to/filled.csv]

With no argument it looks for data/od100k_secondpass_filled.csv.  A partially
filled sheet is allowed: rows with an empty second_label are dropped and the
count is reported, so the check can be run on however many entries have been
done rather than only on all 119.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

import organic_agreement as OA

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
DEFAULT = os.path.join(DATA, "od100k_secondpass_filled.csv")

ALIASES = {
    "art_music": "art_music", "art": "art_music", "music": "art_music",
    "commerce_industry": "commerce_industry", "commerce": "commerce_industry",
    "industry": "commerce_industry",
    "geography_places": "geography_places", "geography": "geography_places",
    "places": "geography_places",
    "history_politics": "history_politics", "history": "history_politics",
    "politics": "history_politics",
    "law_education_institutions": "law_education_institutions",
    "law": "law_education_institutions", "education": "law_education_institutions",
    "literature_writing": "literature_writing", "literature": "literature_writing",
    "writing": "literature_writing",
    "myth_legend": "myth_legend", "myth": "myth_legend", "legend": "myth_legend",
    "peoples_languages": "peoples_languages", "peoples": "peoples_languages",
    "languages": "peoples_languages",
    "philosophy_scholarship": "philosophy_scholarship",
    "philosophy": "philosophy_scholarship", "scholarship": "philosophy_scholarship",
    "religion_theology": "religion_theology", "religion": "religion_theology",
    "theology": "religion_theology",
    "science_nature": "science_nature", "science": "science_nature",
    "nature": "science_nature",
    "war_military": "war_military", "war": "war_military",
    "military": "war_military",
}


def normalise(raw):
    """Accept 'science_nature', 'Science and Nature', 'science' and 1-12."""
    t = raw.strip().lower().replace("-", "_").replace(" ", "_")
    t = t.replace("_and_", "_").replace("&", "and")
    if t in OA.CLASSES:
        return t
    if t in ALIASES:
        return ALIASES[t]
    if t.isdigit() and 1 <= int(t) <= len(OA.CLASSES):
        return OA.CLASSES[int(t) - 1]
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        raise SystemExit(
            "od100k_read_sheet.py: %s not found.\n"
            "Fill data/od100k_secondpass_sheet.html (it downloads this file) or "
            "the matching .csv." % path)

    with open(os.path.join(DATA, "od100k_secondpass_sample.json"),
              encoding="utf-8") as f:
        sample = {r["idx"]: r for r in json.load(f)}

    gold, second, used, skipped, bad = [], [], [], [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                idx = int(row["idx"])
            except (KeyError, TypeError, ValueError):
                bad.append(row)
                continue
            raw = (row.get("second_label") or "").strip()
            if not raw:
                skipped.append(idx)
                continue
            lab = normalise(raw)
            if lab is None or idx not in sample:
                bad.append(row)
                continue
            gold.append(sample[idx]["label_1"])
            second.append(lab)
            used.append(idx)

    if bad:
        raise SystemExit("od100k_read_sheet.py: %d row(s) could not be read, "
                         "first is %r" % (len(bad), bad[0]))
    if len(used) < 10:
        raise SystemExit("od100k_read_sheet.py: only %d filled row(s); at "
                         "least 10 are needed for a meaningful figure"
                         % len(used))

    k_h, po_h, pe_h = OA.cohen_kappa(gold, second, OA.CLASSES)

    model = {}
    mp = os.path.join(RES, "od100k_model_pass.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        for d in m["disagreements"]:
            model[d["idx"]] = d["label_2_model"]

    rule = [OA.auto_label(sample[i]["text"]) for i in used]
    k_r, po_r, pe_r = OA.cohen_kappa(second, rule, OA.CLASSES)

    per_class = {}
    for c in OA.CLASSES:
        sel = [j for j, g in enumerate(gold) if g == c]
        if sel:
            per_class[c] = round(100.0 * np.mean([gold[j] == second[j]
                                                  for j in sel]), 1)

    out = {
        "sheet": os.path.relpath(path, HERE).replace("\\", "/"),
        "entries_filled": len(used),
        "entries_skipped": len(skipped),
        "classes_used": len({*second}),
        "human_vs_first": {
            "kappa": round(k_h, 3),
            "observed_agreement": round(100 * po_h, 2),
            "expected_agreement": round(100 * pe_h, 2),
            "n_agree": sum(1 for a, b in zip(gold, second) if a == b),
            "per_class_agreement": per_class,
        },
        "human_vs_rule": {"kappa": round(k_r, 3),
                          "observed_agreement": round(100 * po_r, 2),
                          "expected_agreement": round(100 * pe_r, 2)},
        "human_vs_model": None,
        "disagreements": [
            {"idx": i, "first_label": g, "human_label_2": s}
            for i, g, s in zip(used, gold, second) if g != s],
    }
    ok = [i for i in used if i in model]
    if ok:
        kh2, po2, pe2 = OA.cohen_kappa(
            [sample[i]["label_1"] for i in ok], [model[i] for i in ok],
            OA.CLASSES)
        out["human_vs_model"] = {"entries": len(ok), "kappa": round(kh2, 3),
                                 "observed_agreement": round(100 * po2, 2),
                                 "expected_agreement": round(100 * pe2, 2)}

    with open(os.path.join(RES, "od100k_human_pass.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("sheet            : %s" % out["sheet"])
    print("entries compared : %d (skipped %d)" % (len(used), len(skipped)))
    print("human vs first   : observed %.1f%%  expected %.1f%%  kappa %.2f"
          % (100 * po_h, 100 * pe_h, k_h))
    print("wrote results/od100k_human_pass.json")


if __name__ == "__main__":
    main()
