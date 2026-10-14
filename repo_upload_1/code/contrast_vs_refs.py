"""Paired significance of every system against the two reference encoders.

ablations.py contrasts every system against the *baseline* S0 only.  Section
4.7 compares the enhanced students against BERT-base and mBERT, and that
comparison needs the same paired machinery rather than a difference of two
means: with 196 test paragraphs and a standard deviation of about 1 point, a
descriptive gap of 1 point is not evidence that the systems differ.

The predictors are built exactly as in ablations.py -- the argmax of the class
posterior averaged over the seeds of a system -- so the numbers are comparable
with Tables 4 and 7.

Outputs results/contrasts_vs_refs.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common as C
from ablations import (group, load, mcnemar, paired_bootstrap, pooled_pred,
                       stack, stars)

ORDER = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
REFS = ["BERT-base", "mBERT"]


def packed_pred(rows):
    """-> pooled argmax predictions and gold labels for a group of runs."""
    P, Y = stack(rows)
    return pooled_pred(P), (P.argmax(-1) == Y[None, :])


def main():
    sys_rows = group(load("students.json"))
    ref_rows = group(load("teachers.json"))

    ref_pred, ref_corr, ref_gold = {}, {}, {}
    for tag in REFS:
        pred, corr = packed_pred(ref_rows[tag])
        ref_pred[tag], ref_corr[tag] = pred, corr.mean(axis=0)
        ref_gold[tag] = np.asarray(ref_rows[tag][0]["test_gold"], dtype=np.int64)

    out = {"contrasts": []}
    lines = ["| System | reference | PTC of system (%) | PTC of reference (%) "
             "| $\\Delta$ (pp) | 95% CI | bootstrap p | McNemar p |",
             "|---|---|---|---|---|---|---|---|"]
    for tag in ORDER:
        pred, corr = packed_pred(sys_rows[tag])
        acc = corr.mean(axis=0)
        for ref in REFS:
            d, lo, hi, p = paired_bootstrap(acc, ref_corr[ref])
            pm, n01, n10 = mcnemar(pred, ref_pred[ref], ref_gold[ref])
            out["contrasts"].append({
                "system": tag, "vs": ref,
                "acc_system": float(acc.mean()), "acc_ref": float(ref_corr[ref].mean()),
                "delta": d, "ci95": [lo, hi],
                "p_bootstrap": p, "p_mcnemar": pm,
                "mcnemar_only_ref_correct": n01,
                "mcnemar_only_system_correct": n10,
            })
            lines.append(
                f"| {tag} | {ref} | {acc.mean() * 100:.1f} | "
                f"{ref_corr[ref].mean() * 100:.1f} | "
                f"{'+' if d >= 0 else ''}{d * 100:.1f} | "
                f"[{lo * 100:+.1f}, {hi * 100:+.1f}] | "
                f"{p:.4f} {stars(p)} | {pm:.4f} {stars(pm)} |")

    text = "\n".join(lines) + "\n"
    with open(os.path.join(C.RESULTS, "contrasts_vs_refs.md"), "w",
              encoding="utf-8") as f:
        f.write(text)
    C.json_dump(out, os.path.join(C.RESULTS, "contrasts_vs_refs.json"))
    print(text)


if __name__ == "__main__":
    main()
