"""Contrast S7nc (S7 with lambda_c = 0) against S7, pooled and per-seed.

Writes results/contrast_s7nc.json and results/contrast_s7nc.md
"""
from __future__ import annotations

import json

import numpy as np

from ablations import group, load, mcnemar, paired_bootstrap, pooled_pred, stack


def packed(rows):
    P, Y = stack(rows)
    return pooled_pred(P), (P.argmax(-1) == Y[None, :]), Y


def main():
    a_rows = group(load("students_S7nc.json"))["S7nc"]   # a = S7nc
    b_rows = group(load("students.json"))["S7"]          # b = S7
    a_rows = sorted(a_rows, key=lambda r: r["seed"])
    b_rows = sorted(b_rows, key=lambda r: r["seed"])
    preda, corra, ga = packed(a_rows)
    predb, corrb, gb = packed(b_rows)
    assert (ga == gb).all()
    gold = ga
    pa = (preda == gold)
    pb = (predb == gold)
    d_p, lo_p, hi_p, p_p = paired_bootstrap(pb.astype(float), pa.astype(float))
    pm, only_b, only_a = mcnemar(preda, predb, gold)  # n01 = only b (S7) correct
    ma = corra.mean(axis=0)
    mb = corrb.mean(axis=0)
    d_m, lo_m, hi_m, p_m = paired_bootstrap(mb, ma)
    out = {
        "acc_s7nc_mean": float(corra.mean()), "acc_s7_mean": float(corrb.mean()),
        "acc_s7nc_pooled": float(pa.mean()), "acc_s7_pooled": float(pb.mean()),
        "pooled": {"delta_pp": (pb.mean() - pa.mean()) * 100,
                   "ci95_pp": [lo_p * 100, hi_p * 100],
                   "p_bootstrap": p_p, "p_mcnemar": pm,
                   "only_s7_correct": int(only_b), "only_s7nc_correct": int(only_a)},
        "mean": {"delta_pp": (mb.mean() - ma.mean()) * 100,
                 "ci95_pp": [lo_m * 100, hi_m * 100], "p_bootstrap": p_m},
        "per_seed": {"s7": [float(x) for x in mb], "s7nc": [float(x) for x in ma]},
    }
    with open("results/contrast_s7nc.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
