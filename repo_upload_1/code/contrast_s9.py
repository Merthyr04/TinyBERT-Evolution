"""S9 = S8 - L_mlm.

The configuration the paper describes as the one it did *not* run
(Section 4.8, end of the A4 paragraph; Section 7, first open direction):
augmentation + single-teacher distillation + dual head, with the auxiliary
masked-LM objective removed.  S8 showed that the dual-teacher ingredient can be
dropped once the dual head is present, and A4 showed that dropping L_mlm from S6
raises accuracy; S9 applies both at once and closes the omission.

Every contrast below is seed-matched (all cells carry the same five seeds,
7 13 37 42 97) and is reported on both accuracy estimators used in the paper:

  * per-seed mean -- the average of the five individual runs' accuracies
                     (the estimator of Tables 4 and 7);
  * pooled         -- accuracy of the argmax of the seed-averaged posterior
                     (the estimator of Tables 5, 8 and 9).

McNemar's exact test always counts disagreements between the pooled
predictors, exactly as in Tables 5, 7 and 8.  TOST and the smallest equivalent
margin are computed on the per-seed-mean paired bootstrap, as in
baseline_matrix.py (c).

Usage:  python contrast_s9.py
Writes results/contrast_s9.json and results/contrast_s9.md.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from ablations import group, load, mcnemar, pooled_pred, stars, stack_seeds  # noqa: E402
from baseline_matrix import (bootstrap_dist, smallest_equivalent_margin,  # noqa: E402
                             stats_of)

# ---------------------------------------------------------------------------
# Pairs.  The first three are the ones the configuration exists to answer:
#   S9 vs S8  -> what L_mlm is worth inside the dual-head family;
#   S9 vs A4  -> what the dual head costs at fixed (no L_mlm);
#   S9 vs S7  -> the dual-teacher ingredient on top of S9.
# The rest place S9 in the existing tables.
PAIRS = [
    ("S9", "S8"), ("S9", "A4"), ("S9", "S7"), ("S9", "S6"), ("S9", "S4"),
    ("S9", "S3"), ("S9", "S0"), ("S9", "A0"),
    ("S9", "BERT-base"), ("S9", "mBERT"),
]

# Corpora that live in their own result files.
FILES = ["students.json", "students_S8.json", "students_S9.json",
         "ablations.json", "followups.json", "teachers.json"]


def load_all():
    rows = []
    for f in FILES:
        r = load(f)
        if not r:
            continue
        rows.extend(r)
    return group(rows)


def contrast(rows_a, rows_b, seeds):
    """Seed-matched, both estimators, McNemar on the pooled predictors."""
    Pa, Ya = stack_seeds(rows_a, seeds)
    Pb, Yb = stack_seeds(rows_b, seeds)
    assert (Ya == Yb).all(), "systems must share the test split"
    gold = Ya

    pred_a, pred_b = pooled_pred(Pa), pooled_pred(Pb)
    pa = (pred_a == gold).astype(np.float64)
    pb = (pred_b == gold).astype(np.float64)
    d_p, lo_p, hi_p, p_p = _boot(pa, pb)
    pm, only_a, only_b = mcnemar(pred_b, pred_a, gold)   # n01 = only a correct

    # per-paragraph correctness averaged over seeds -> its mean is the
    # per-seed-mean accuracy, the estimator of Tables 4 and 7.
    ma = (Pa.argmax(-1) == gold[None, :]).mean(axis=0).astype(np.float64)
    mb = (Pb.argmax(-1) == gold[None, :]).mean(axis=0).astype(np.float64)
    d_m, lo_m, hi_m, p_m = _boot(ma, mb)
    dist_m = bootstrap_dist(ma, mb)

    return {
        "acc_a_mean": float(ma.mean()), "acc_b_mean": float(mb.mean()),
        "acc_a_pooled": float(pa.mean()), "acc_b_pooled": float(pb.mean()),
        "pooled": {"delta_pp": d_p * 100, "ci95_pp": [lo_p * 100, hi_p * 100],
                   "p_bootstrap": p_p, "p_mcnemar": pm,
                   "only_a_correct": int(only_a), "only_b_correct": int(only_b)},
        "mean": {"delta_pp": d_m * 100, "ci95_pp": [lo_m * 100, hi_m * 100],
                 "p_bootstrap": p_m,
                 "tost_2pp": float(max((dist_m >= 0.02).mean(),
                                       (dist_m <= -0.02).mean())),
                 "equivalent_within_pp": smallest_equivalent_margin(dist_m)},
        "n_test": int(gold.size),
        "seeds": sorted(set(seeds)),
    }


def _boot(a, b):
    """Reuse the shared paired bootstrap so the seeds match the other tables."""
    from ablations import paired_bootstrap
    return paired_bootstrap(a, b)


def seedy(rows, tag, seeds):
    keep = sorted([r for r in rows if r["seed"] in set(seeds)],
                  key=lambda r: r["seed"])
    return keep


def main():
    G = load_all()

    assert len(G.get("S9", [])) == 5, (
        f"expected five seeds for S9, found {len(G.get('S9', []))}")
    seeds = sorted(r["seed"] for r in G["S9"])

    out = {"seeds": seeds, "pairs": {}, "per_seed": {}}

    for a, b in PAIRS:
        rows_a = G.get(a) or []
        rows_b = G.get(b) or []
        if not rows_a or not rows_b:
            print("skip", a, b, "missing rows")
            continue
        shared = sorted(set(r["seed"] for r in rows_a) &
                        set(r["seed"] for r in rows_b))
        if len(shared) != len(seeds):
            print(f"note {a} vs {b}: comparing on seeds {shared}")
        out["pairs"][f"{a}_vs_{b}"] = contrast(rows_a, rows_b, shared)

    # ---- S9 descriptives ---------------------------------------------------
    s9 = G["S9"]
    st = stats_of(s9)
    out["S9"] = {
        "n_seeds": len(s9), "seeds": sorted(r["seed"] for r in s9),
        "acc_mean": st["acc_mean"], "acc_std": st["acc_std"],
        "acc_pooled": st["acc_pooled"], "f1_pooled": st["f1_pooled"],
        "mlm_top1": st["mlm_top1"],
        "per_seed_acc": {r["seed"]: float(r["test_acc"]) for r in s9},
        "train_time_s": float(np.mean([r["train_time_s"] for r in s9])),
        "peak_train_vram_mb": float(np.mean(
            [r["peak_train_vram_mb"] for r in s9
             if r.get("peak_train_vram_mb") is not None])),
        "exit_rate_theta08": float(np.mean(
            [r["exit_rate"] for r in s9 if r.get("exit_rate") is not None])),
    }

    # same descriptives for the neighbours, so the table can be read whole
    for tag in ["S8", "A4", "S7", "S6", "S4", "S3", "S0", "A0",
                "BERT-base", "mBERT"]:
        if tag in G:
            s = stats_of(G[tag])
            out.setdefault("descriptives", {})[tag] = {
                "n_seeds": len(G[tag]),
                "acc_mean": s["acc_mean"], "acc_std": s["acc_std"],
                "acc_pooled": s["acc_pooled"], "f1_pooled": s["f1_pooled"],
                "mlm_top1": s["mlm_top1"],
            }

    # ---- per-seed deltas S9 - S8 and S9 - A4 (sign consistency) -----------
    for other in ["S8", "A4", "S7"]:
        if other not in G:
            continue
        d = {}
        for seed in seeds:
            ra = [r for r in G["S9"] if r["seed"] == seed]
            rb = [r for r in G[other] if r["seed"] == seed]
            if ra and rb:
                d[seed] = float(ra[0]["test_acc"] - rb[0]["test_acc"])
        if d:
            out["per_seed"][f"S9_minus_{other}"] = d
            out["per_seed"][f"S9_minus_{other}_mean_pp"] = (
                float(np.mean(list(d.values()))) * 100)

    C.json_dump(out, os.path.join(C.RESULTS, "contrast_s9.json"))

    # ---- report -----------------------------------------------------------
    L = []
    L.append("# S9 = S8 - L_mlm (the configuration the paper left unrun)\n")
    L.append(f"Five seeds {seeds}, LD-100K, the 196-paragraph test split.  "
             "Every contrast is seed-matched.\n")

    L.append("## Per-system accuracy\n")
    L.append("| tag | n seeds | PTC mean +- sd (%) | PTC pooled (%) | "
             "macro-F1 pooled (%) | MLM top-1 (%) |")
    L.append("|---|---|---|---|---|---|")
    desc = {"S9": out["S9"]}
    desc.update(out.get("descriptives", {}))
    for tag in ["S9", "S8", "A4", "S7", "S6", "S4", "S3", "S0", "A0",
                "BERT-base", "mBERT"]:
        if tag not in desc:
            continue
        s = desc[tag]
        mlm = ("n/a" if s["mlm_top1"] is None
               else f"{s['mlm_top1'] * 100:.1f}")
        L.append(f"| {'**' + tag + '**' if tag == 'S9' else tag} | "
                 f"{s['n_seeds']} | {s['acc_mean'] * 100:.2f} +- "
                 f"{s['acc_std'] * 100:.2f} | {s['acc_pooled'] * 100:.2f} | "
                 f"{s['f1_pooled'] * 100:.2f} | {mlm} |")

    L.append("\n## Contrasts\n")
    L.append("| pair | mean d (pp) | mean 95% CI | mean p | pooled d (pp) | "
             "pooled 95% CI | pooled p | McNemar p | only A | only B | "
             "equiv. within (pp) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in out["pairs"].items():
        a, b = k.split("_vs_")
        eq = v["mean"]["equivalent_within_pp"]
        L.append(
            f"| {a} vs {b} | {v['mean']['delta_pp']:+.2f} | "
            f"[{v['mean']['ci95_pp'][0]:+.2f}, {v['mean']['ci95_pp'][1]:+.2f}] | "
            f"{v['mean']['p_bootstrap']:.4f} {stars(v['mean']['p_bootstrap'])} | "
            f"{v['pooled']['delta_pp']:+.2f} | "
            f"[{v['pooled']['ci95_pp'][0]:+.2f}, {v['pooled']['ci95_pp'][1]:+.2f}] | "
            f"{v['pooled']['p_bootstrap']:.4f} {stars(v['pooled']['p_bootstrap'])} | "
            f"{v['pooled']['p_mcnemar']:.4f} {stars(v['pooled']['p_mcnemar'])} | "
            f"{v['pooled']['only_a_correct']} | {v['pooled']['only_b_correct']} | "
            f"{'none' if eq is None else '%.1f' % eq} |")

    L.append("\n## Per-seed deltas (test accuracy)\n")
    L.append("| pair | " + " | ".join(str(s) for s in seeds) + " | mean (pp) |")
    L.append("|---" * (len(seeds) + 2) + "|")
    for other in ["S8", "A4", "S7"]:
        key = f"S9_minus_{other}"
        if key not in out["per_seed"]:
            continue
        d = out["per_seed"][key]
        L.append(f"| S9 - {other} | " + " | ".join(
            f"{d.get(seed, float('nan')) * 100:+.2f}" if seed in d else "-"
            for seed in seeds) + f" | {out['per_seed'][key + '_mean_pp']:+.2f} |")

    L.append("")
    L.append(f"S9 MLM top-1 {out['S9']['mlm_top1'] * 100:.1f}%, per-seed mean "
             f"PTC {out['S9']['acc_mean'] * 100:.2f}%, pooled "
             f"{out['S9']['acc_pooled'] * 100:.2f}%, macro-F1 pooled "
             f"{out['S9']['f1_pooled'] * 100:.2f}%.  mean train time "
             f"{out['S9']['train_time_s']:.0f} s, mean peak train VRAM "
             f"{out['S9']['peak_train_vram_mb']:.0f} MB.")
    L.append("")
    L.append("Significance: `*` p<0.05, `**` p<0.01, `***` p<0.001, `n.s.` "
             "not significant.  10,000 paired bootstrap resamples of the "
             "196 test paragraphs; McNemar is exact and always counts the "
             "pooled predictors.\n")

    text = "\n".join(L) + "\n"
    with open(os.path.join(C.RESULTS, "contrast_s9.md"), "w",
              encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
