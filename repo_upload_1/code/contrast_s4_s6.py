"""Paired significance of S4 (DA + single teacher) against S6 (DD + DA).

Section 4.5/4.9(viii) of the paper reports the S4/S6 gap (94.73% vs 93.54%)
as *descriptive only*, because every paired test in the paper is taken
against S0 or a reference encoder.  This script closes that gap: both
systems have three seeds with per-paragraph test predictions already stored
in students.json, so the contrast costs no GPU time.

Predictors are built exactly as in ablations.py / contrast_vs_refs.py --
argmax of the class posterior averaged over the seeds -- so the numbers are
comparable with Tables 4 and 5.

Outputs results/contrast_s4_s6.json and .md.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common as C
from ablations import (group, load, mcnemar, paired_bootstrap, pooled_pred,
                       stack, stars)


def packed_pred(rows):
    """-> pooled argmax predictions, per-paragraph correctness, gold labels."""
    P, Y = stack(rows)          # Y is already the 1-D gold vector
    return pooled_pred(P), (P.argmax(-1) == Y[None, :]), Y


def main():
    sys_rows = group(load("students.json"))
    assert len(sys_rows["S4"]) == 5 and len(sys_rows["S6"]) == 5, \
        "expected five seeds per system"

    pred4, corr4, gold = packed_pred(sys_rows["S4"])
    pred6, corr6, gold6 = packed_pred(sys_rows["S6"])
    assert (gold == gold6).all(), "the two systems must share the test split"

    d, lo, hi, p = paired_bootstrap(corr4.mean(axis=0), corr6.mean(axis=0))
    # mcnemar(a, b, gold): n01 = only b correct, n10 = only a correct
    pm, only_s4, only_s6 = mcnemar(pred6, pred4, gold)

    out = {
        "pair": ["S4", "S6"],
        "acc_S4": float(corr4.mean()), "acc_S6": float(corr6.mean()),
        "delta_pp": d * 100, "ci95_pp": [lo * 100, hi * 100],
        "p_bootstrap": p, "p_mcnemar": pm,
        "discordant_only_S4_correct": only_s4,
        "discordant_only_S6_correct": only_s6,
        "n_test": int(gold.size),
    }
    lines = [
        "| pair | S4 PTC (%) | S6 PTC (%) | $\\Delta$ S4-S6 (pp) | 95% CI "
        "| bootstrap p | McNemar p | only S4 correct | only S6 correct |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| S4 vs S6 | {corr4.mean() * 100:.2f} | {corr6.mean() * 100:.2f} | "
        f"{d * 100:+.2f} | [{lo * 100:+.1f}, {hi * 100:+.1f}] | "
        f"{p:.4f} {stars(p)} | {pm:.4f} {stars(pm)} | {only_s4} | {only_s6} |",
    ]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(C.RESULTS, "contrast_s4_s6.md"), "w",
              encoding="utf-8") as f:
        f.write(text)
    C.json_dump(out, os.path.join(C.RESULTS, "contrast_s4_s6.json"))
    print(text)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
