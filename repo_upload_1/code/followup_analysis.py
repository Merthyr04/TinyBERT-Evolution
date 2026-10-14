"""Analysis of the follow-up runs staged after the main comparison.

Reads results/followups.json (A0, F000..F100, S6eq) together with the
published student and ablation results, and reports:

  * A0 = S0 - L_mlm against S0: the pooled contrast, both paired tests, and the
    per-seed breakdown -- does removing the masked-LM term help the *baseline*
    the way it helps S6 (A4)?
  * the fixed-alpha sweep F000..F100, against the learned-gate S6: is
    alpha ~ 0.5 an optimum, or merely the point at which the gate stops?
  * S6eq, the equal-strength teacher pair (two fine-tunings of BERT-base),
    against S6 and against the deployed S7: does the dual teacher still fail
    to pay when both teachers are equally strong?

It also emits, for every system, its pooled accuracy against each of the five
S0 single-seed accuracies, which is what a worst-seed-baseline reading of the
accuracy table needs.

Outputs results/followups_analysis.json and results/followups_analysis.md.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from ablations import (load, group, stack, pooled_pred, per_seed_correct,  # noqa: E402
                       paired_bootstrap, mcnemar, stars)

JSON_OUT = os.path.join(C.RESULTS, "followups_analysis.json")
MD_OUT = os.path.join(C.RESULTS, "followups_analysis.md")


def stats_of(rows):
    P, Y = stack(rows)
    pred = pooled_pred(P)
    per_seed = per_seed_correct(P, Y)
    first = rows[0]
    return {
        "seeds": [r["seed"] for r in rows],
        "acc_mean": float(per_seed.mean()),
        "acc_std": float(per_seed.std(ddof=0)),
        "acc_pooled": float((pred == Y).mean()),
        "per_seed_acc": [float(x) for x in per_seed],
        "per_seed_acc_by_seed": {int(r["seed"]): float(x)
                                 for r, x in zip(rows, per_seed)},
        "mlm_top1": (float(np.mean([r["mlm_top1"] for r in rows]))
                     if first.get("mlm_top1") is not None else None),
        "alpha": (float(first["alpha"]) if first.get("alpha") is not None
                  else None),
        "alpha_fixed": bool(first.get("alpha_fixed", False)),
        "dual_teacher_source": first.get("dual_teacher_source"),
        "corr": (pred == Y).astype(np.float64),
        "pred": pred.tolist(),
        "gold": Y.tolist(),
    }


def contrast(a, b):
    """`a` against `b`, both on the pooled predictor; positive = a better."""
    d, lo, hi, p = paired_bootstrap(a["corr"], b["corr"])
    pm, n01, n10 = mcnemar(a["pred"], b["pred"], a["gold"])
    return {"delta": d, "ci95": [lo, hi], "p_bootstrap": p, "p_mcnemar": pm,
            "only_a_correct": n10, "only_b_correct": n01}


def pp(x):
    return f"{x * 100:+.2f}"


def main():
    follow = group(load("followups.json"))
    if not follow:
        raise SystemExit("followups.json is empty or missing")

    all_rows = (load("students.json") + load("ablations.json")
                + load("students_S8.json") + load("followups.json"))
    S = {t: stats_of(r) for t, r in group(all_rows).items()}

    rep = {"systems": {}, "a0": {}, "alpha_sweep": [],
           "equal_pair": {}, "vs_single_seed_s0": {}}
    for t, s in S.items():
        rep["systems"][t] = {
            "seeds": s["seeds"], "acc_mean": s["acc_mean"],
            "acc_std": s["acc_std"], "acc_pooled": s["acc_pooled"],
            "per_seed_acc_by_seed": s["per_seed_acc_by_seed"],
            "mlm_top1": s["mlm_top1"], "alpha": s["alpha"],
            "alpha_fixed": s["alpha_fixed"],
            "dual_teacher_source": s["dual_teacher_source"]}

    # ---- A0 = S0 - L_mlm ---------------------------------------------------
    if "A0" in S:
        a0, s0 = S["A0"], S["S0"]
        per_seed = []
        for seed in sorted(set(a0["seeds"]) & set(s0["seeds"])):
            per_seed.append({"seed": seed,
                             "a0": a0["per_seed_acc_by_seed"][seed],
                             "s0": s0["per_seed_acc_by_seed"][seed]})
        rep["a0"] = {
            "seeds": a0["seeds"],
            "acc_mean": a0["acc_mean"], "acc_std": a0["acc_std"],
            "acc_pooled": a0["acc_pooled"], "mlm_top1": a0["mlm_top1"],
            "vs_S0_pooled": contrast(a0, s0),
            "per_seed": per_seed,
            "per_seed_mean_delta": float(np.mean([p["a0"] - p["s0"]
                                                  for p in per_seed])),
        }
        if "A4" in S:
            rep["a0"]["vs_A4_pooled"] = contrast(a0, S["A4"])

    # ---- fixed-alpha sweep -------------------------------------------------
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        tag = f"F{int(round(a * 100)):03d}"
        if tag not in S:
            continue
        cell = {
            "tag": tag, "alpha": a,
            "acc_mean": S[tag]["acc_mean"], "acc_std": S[tag]["acc_std"],
            "acc_pooled": S[tag]["acc_pooled"], "mlm_top1": S[tag]["mlm_top1"],
            "seeds": S[tag]["seeds"],
        }
        for ref in ("S6", "S4", "A4"):
            if ref in S:
                cell[f"vs_{ref}"] = contrast(S[tag], S[ref])
        rep["alpha_sweep"].append(cell)

    # ---- equal-strength teacher pair --------------------------------------
    if "S6eq" in S:
        eq = S["S6eq"]
        rep["equal_pair"] = {
            "seeds": eq["seeds"],
            "acc_mean": eq["acc_mean"], "acc_std": eq["acc_std"],
            "acc_pooled": eq["acc_pooled"], "mlm_top1": eq["mlm_top1"],
            "alpha": eq["alpha"],
            "dual_teacher_source": eq["dual_teacher_source"],
            "vs_S6_pooled": contrast(eq, S["S6"]),
            "vs_S7_pooled": contrast(eq, S["S7"]),
            "vs_S4_pooled": contrast(eq, S["S4"]),
        }

    # ---- pooled accuracy against each single-seed S0 baseline -------------
    s0 = S["S0"]
    for tag in sorted(S):
        s = S[tag]
        if s["acc_pooled"] is None:
            continue
        rep["vs_single_seed_s0"][tag] = {
            f"seed{seed}": float(s["acc_pooled"] - acc)
            for seed, acc in sorted(s0["per_seed_acc_by_seed"].items())}

    C.json_dump(rep, JSON_OUT)

    # ---- human-readable report --------------------------------------------
    L = []
    L.append("# Follow-up runs: analysis\n")
    for tag in ("A0", "F000", "F025", "F050", "F075", "F100", "S6eq"):
        if tag not in S:
            continue
        s = S[tag]
        L.append(f"## {tag}  (n={len(s['seeds'])}, seeds {s['seeds']})")
        L.append(f"- mean +- sd: {s['acc_mean'] * 100:.2f} +- {s['acc_std'] * 100:.2f}")
        L.append(f"- pooled: {s['acc_pooled'] * 100:.2f}")
        mlm = "n/a" if s["mlm_top1"] is None else f"{s['mlm_top1'] * 100:.2f}"
        L.append(f"- MLM top-1: {mlm}")
        if s["alpha"] is not None:
            L.append(f"- alpha: {s['alpha']:.4f}"
                     f"{' (fixed)' if s['alpha_fixed'] else ' (learned)'}")
        if s["dual_teacher_source"]:
            L.append(f"- second teacher: {s['dual_teacher_source']}")
        L.append("")

    if rep["a0"]:
        a0 = rep["a0"]
        c = a0["vs_S0_pooled"]
        L.append("## A0 = S0 - L_mlm, against the published baseline")
        L.append(f"- pooled Delta = {pp(c['delta'])} pp, "
                 f"95% CI [{c['ci95'][0] * 100:+.2f}, {c['ci95'][1] * 100:+.2f}], "
                 f"bootstrap p = {c['p_bootstrap']:.4f} {stars(c['p_bootstrap'])}, "
                 f"McNemar p = {c['p_mcnemar']:.4f} {stars(c['p_mcnemar'])} "
                 f"(only A0 {c['only_a_correct']}, only S0 {c['only_b_correct']})")
        L.append(f"- per-seed mean difference = "
                 f"{a0['per_seed_mean_delta'] * 100:+.2f} pp")
        for p in a0["per_seed"]:
            L.append(f"    - seed {p['seed']:>3}: A0 {p['a0'] * 100:.2f} "
                     f"vs S0 {p['s0'] * 100:.2f}  ({pp(p['a0'] - p['s0'])} pp)")
        if "vs_A4_pooled" in a0:
            c4 = a0["vs_A4_pooled"]
            L.append(f"- against A4 (S6 - L_mlm): Delta = {pp(c4['delta'])} pp, "
                     f"p_boot {c4['p_bootstrap']:.4f}, p_mc {c4['p_mcnemar']:.4f}")
        L.append("")

    if rep["alpha_sweep"]:
        L.append("## Fixed-alpha teacher-mix sweep (alpha = weight on BERT-base)")
        L.append("| alpha | mean +- sd | pooled | MLM top-1 | vs S6 | p_boot | p_mc |")
        L.append("|---|---|---|---|---|---|---|")
        for cell in rep["alpha_sweep"]:
            c = cell.get("vs_S6", {})
            mlm = ("n/a" if cell["mlm_top1"] is None
                   else f"{cell['mlm_top1'] * 100:.2f}")
            L.append(
                f"| {cell['alpha']:.2f} | {cell['acc_mean'] * 100:.2f} +- "
                f"{cell['acc_std'] * 100:.2f} | {cell['acc_pooled'] * 100:.2f} | "
                f"{mlm} | "
                f"{pp(c.get('delta', 0)) if c else '-'} | "
                f"{c.get('p_bootstrap', float('nan')):.4f} | "
                f"{c.get('p_mcnemar', float('nan')):.4f} |")
        L.append("")

    if rep["equal_pair"]:
        eq = rep["equal_pair"]
        L.append("## Equal-strength teacher pair (BERT-base x2), S6eq")
        for ref in ("S6", "S7", "S4"):
            key = f"vs_{ref}_pooled"
            if key not in eq:
                continue
            c = eq[key]
            L.append(f"- vs {ref}: Delta = {pp(c['delta'])} pp, "
                     f"95% CI [{c['ci95'][0] * 100:+.2f}, {c['ci95'][1] * 100:+.2f}], "
                     f"p_boot {c['p_bootstrap']:.4f} {stars(c['p_bootstrap'])}, "
                     f"p_mc {c['p_mcnemar']:.4f} {stars(c['p_mcnemar'])}")
        L.append("")

    L.append("## Pooled accuracy against each single-seed S0 baseline")
    s0 = S["S0"]
    order = sorted(s0["per_seed_acc_by_seed"])
    L.append("| system | " + " | ".join(
        "S0 seed %d (%.2f)" % (k, s0["per_seed_acc_by_seed"][k] * 100)
        for k in order) + " |")
    L.append("|" + "---|" * (len(order) + 1))
    for tag in ["S0", "S4", "S6", "S7", "S8", "A0", "S6eq"] + \
            [c["tag"] for c in rep["alpha_sweep"]]:
        if tag not in rep["vs_single_seed_s0"]:
            continue
        row = rep["vs_single_seed_s0"][tag]
        L.append("| %s | " % tag + " | ".join(
            pp(row["seed%d" % k]) for k in order) + " |")

    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
