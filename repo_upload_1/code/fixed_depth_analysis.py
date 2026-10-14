"""Fixed-depth control analysis (review M1a / Section 4.9(xiv)).

The deployed artefact of a dual-head system is the first half of the encoder
plus a classifier, because the exit is not adaptive on this corpus.  The
natural control is therefore the *same recipe* with the deployed classifier
moved onto the layer-2 hidden state, no auxiliary head and no consistency term
-- a genuine 2-layer deployment that keeps the 4-layer backbone only for
training.  That is configuration F2 (results/fixed_depth.json).

If F2 matches S7, the dual head is a convenience (it buys the exit through a
training-time consistency term) rather than a mechanism that confers accuracy.

Outputs results/fixed_depth_analysis.json and .md.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from ablations import group, load, stars  # noqa: E402
from followup_analysis import contrast, pp, stats_of  # noqa: E402

REFS = ["S0", "S4", "S6", "S7"]


def main():
    fd = group(load("fixed_depth.json"))
    if "F2" not in fd:
        raise SystemExit("fixed_depth.json has no F2 rows yet")
    all_rows = (load("students.json") + load("ablations.json") + load("fixed_depth.json"))
    S = {t: stats_of(r) for t, r in group(all_rows).items()}
    if "F2" not in S:
        raise SystemExit("F2 rows missing after merge")

    f2 = S["F2"]
    rep = {"f2": {"seeds": f2["seeds"], "acc_mean": f2["acc_mean"],
                  "acc_std": f2["acc_std"], "acc_pooled": f2["acc_pooled"],
                  "per_seed_acc_by_seed": f2["per_seed_acc_by_seed"],
                  "mlm_top1": f2["mlm_top1"]},
           "contrasts": {}}
    for ref in REFS:
        if ref in S:
            rep["contrasts"][ref] = contrast(f2, S[ref])

    # per-seed deltas against S0 and S7
    per_seed = []
    for seed in sorted(f2["per_seed_acc_by_seed"]):
        row = {"seed": seed, "F2": f2["per_seed_acc_by_seed"][seed]}
        for ref in ("S0", "S7"):
            if ref in S and seed in S[ref]["per_seed_acc_by_seed"]:
                row[ref] = S[ref]["per_seed_acc_by_seed"][seed]
                row[f"d_vs_{ref}"] = row["F2"] - row[ref]
        per_seed.append(row)
    rep["per_seed"] = per_seed

    C.json_dump(rep, os.path.join(C.RESULTS, "fixed_depth_analysis.json"))

    L = ["# Fixed-depth control F2\n",
         f"Seeds: {f2['seeds']}.  Same recipe as S7 (DA + dual teacher + every "
         "distillation term) but the deployed classifier reads the layer-2 "
         "hidden state; no auxiliary head, no consistency term.\n",
         f"F2: per-seed mean **{f2['acc_mean'] * 100:.2f} $\\pm$ "
         f"{f2['acc_std'] * 100:.2f}%**, pooled **{f2['acc_pooled'] * 100:.2f}%**, "
         f"MLM top-1 {f2['mlm_top1'] * 100:.1f}%.\n",
         "## Contrasts (pooled predictor, 196 test paragraphs)\n",
         "| System | ref | PTC sys (%) | PTC ref (%) | Δ (pp) | 95% CI | bootstrap p | McNemar p |",
         "|---|---|---|---|---|---|---|---|"]
    for ref, c in rep["contrasts"].items():
        L.append(f"| F2 | {ref} | {f2['acc_pooled'] * 100:.1f} | "
                 f"{S[ref]['acc_pooled'] * 100:.1f} | {pp(c['delta'])} | "
                 f"[{c['ci95'][0] * 100:+.1f}, {c['ci95'][1] * 100:+.1f}] | "
                 f"{c['p_bootstrap']:.4f} {stars(c['p_bootstrap'])} | "
                 f"{c['p_mcnemar']:.4f} {stars(c['p_mcnemar'])} |")

    L.append("\n## Per-seed\n")
    L.append("| seed | F2 PTC (%) | S0 PTC (%) | Δ vs S0 (pp) | S7 PTC (%) | Δ vs S7 (pp) |")
    L.append("|---|---|---|---|---|---|")
    for r in per_seed:
        L.append(f"| {r['seed']} | {r['F2'] * 100:.1f} | "
                 f"{r.get('S0', float('nan')) * 100:.1f} | "
                 f"{r.get('d_vs_S0', float('nan')) * 100:+.1f} | "
                 f"{r.get('S7', float('nan')) * 100:.1f} | "
                 f"{r.get('d_vs_S7', float('nan')) * 100:+.1f} |")

    text = "\n".join(L) + "\n"
    with open(os.path.join(C.RESULTS, "fixed_depth_analysis.md"), "w",
              encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
