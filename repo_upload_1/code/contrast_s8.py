"""S8 = DA + SD + DH (i.e. S4 with the dual head attached).

Purpose of the configuration: the deployed system S7 combines the dual head
with dual-teacher distillation (DD).  S8 removes DD and keeps the dual head, so
that the contrast S8 - S4 measures the dual head on its own (at fixed
augmentation + single-teacher distillation) and S8 - S7 measures what DD adds
once the dual head is present.

Both accuracy estimators used in the paper are reported:
  * per-seed mean  -- the average of the five individual runs' accuracies
                      (the estimator of Tables 4 and 7);
  * pooled         -- accuracy of the argmax of the seed-averaged posterior
                      (the estimator of Tables 5, 8 and 9).
McNemar's exact test always counts disagreements between the pooled predictors,
exactly as in Tables 5, 7 and 8.

Usage:  python contrast_s8.py
Writes results/contrast_s8.json and results/contrast_s8.md.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common as C
from ablations import (group, load, mcnemar, paired_bootstrap, pooled_pred,
                       stack)


def packed(rows):
    P, Y = stack(rows)
    return pooled_pred(P), (P.argmax(-1) == Y[None, :]), Y


def contrast(rows_a, rows_b, gold_ref=None):
    preda, corra, ga = packed(rows_a)
    predb, corrb, gb = packed(rows_b)
    assert (ga == gb).all(), "systems must share the test split"
    gold = ga

    # pooled estimator
    pa = (preda == gold)
    pb = (predb == gold)
    d_p, lo_p, hi_p, p_p = paired_bootstrap(pa.astype(float), pb.astype(float))
    pm, only_a, only_b = mcnemar(predb, preda, gold)   # n01 = only a correct

    # per-seed mean estimator, same paragraphs
    ma = corra.mean(axis=0)
    mb = corrb.mean(axis=0)
    d_m, lo_m, hi_m, p_m = paired_bootstrap(ma, mb)

    return {
        "acc_a_mean": float(corra.mean()), "acc_b_mean": float(corrb.mean()),
        "acc_a_pooled": float(pa.mean()), "acc_b_pooled": float(pb.mean()),
        "pooled": {"delta_pp": d_p * 100, "ci95_pp": [lo_p * 100, hi_p * 100],
                   "p_bootstrap": p_p, "p_mcnemar": pm,
                   "only_a_correct": int(only_a), "only_b_correct": int(only_b)},
        "mean": {"delta_pp": d_m * 100, "ci95_pp": [lo_m * 100, hi_m * 100],
                 "p_bootstrap": p_m},
        "n_test": int(gold.size),
    }


def main():
    sys_rows = group(load("students.json"))
    extra = group(load("students_S8.json"))
    for t, r in extra.items():
        sys_rows.setdefault(t, []).extend(r)

    assert len(sys_rows["S8"]) == 5, "expected five seeds for S8"
    refs = group(load("teachers.json"))

    pairs = [("S8", "S4"), ("S8", "S7"), ("S8", "S6"), ("S8", "S0"),
             ("S8", "S5"), ("S8", "BERT-base"), ("S8", "mBERT")]
    out = {"pairs": {}}
    for a, b in pairs:
        rows_a = sys_rows.get(a) or refs.get(a)
        rows_b = sys_rows.get(b) or refs.get(b)
        if not rows_a or not rows_b:
            print("skip", a, b, "missing rows")
            continue
        out["pairs"][f"{a}_vs_{b}"] = contrast(rows_a, rows_b)

    # S8 descriptives: MLM, exit rate if the run recorded one
    s8 = sys_rows["S8"]
    out["S8_mlm_top1"] = float(np.mean([r["mlm_top1"] for r in s8]))
    out["S8_mlm_top5"] = float(np.mean([r["mlm_top5"] for r in s8]))
    out["S8_acc_mean"] = float(np.mean([r["test_acc"] for r in s8]))

    with open(os.path.join(C.RESULTS, "contrast_s8.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    L = ["| pair | mean Δ (pp) | mean 95% CI | mean p | pooled Δ (pp) "
         "| pooled 95% CI | pooled p | McNemar p | only A | only B | n |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in out["pairs"].items():
        a, b = k.split("_vs_")
        L.append(
            f"| {a} vs {b} | {v['mean']['delta_pp']:+.2f} | "
            f"[{v['mean']['ci95_pp'][0]:+.1f}, {v['mean']['ci95_pp'][1]:+.1f}] | "
            f"{v['mean']['p_bootstrap']:.4f} | {v['pooled']['delta_pp']:+.2f} | "
            f"[{v['pooled']['ci95_pp'][0]:+.1f}, {v['pooled']['ci95_pp'][1]:+.1f}] | "
            f"{v['pooled']['p_bootstrap']:.4f} | {v['pooled']['p_mcnemar']:.4f} | "
            f"{v['pooled']['only_a_correct']} | {v['pooled']['only_b_correct']} | "
            f"{v['n_test']} |")
    L.append("")
    L.append(f"S8 MLM top-1 {out['S8_mlm_top1'] * 100:.1f}%, top-5 "
             f"{out['S8_mlm_top5'] * 100:.1f}%, per-seed mean PTC "
             f"{out['S8_acc_mean'] * 100:.2f}%.")
    md = "\n".join(L) + "\n"
    with open(os.path.join(C.RESULTS, "contrast_s8.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
