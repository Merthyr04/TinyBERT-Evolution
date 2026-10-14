"""Second baseline, equivalence tests and multiplicity control.

The main comparison (Table 5 of the paper) measures every system against S0.
The paper's own ablation study shows that S0 is not the strongest reference a
4-layer student can be given: removing the auxiliary masked-LM term from S0
gives A0, which is the *same deployed graph* at the same inference cost but
more accurate.  This script re-runs the whole comparison against both
baselines, so a reader can see how much of each gain survives the change of
reference.

It also adds three things the paper's statistics were missing:

  * TOST equivalence tests against the reference encoders, with an explicit
    margin, in place of reading "not significant" as "equivalent";
  * Holm-adjusted p-values over the declared family of primary contrasts;
  * per-class F1 and the confusion matrix for the systems the paper discusses.

Outputs results/baseline_matrix.json and results/baseline_matrix.md.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from ablations import (B, SEED, group, load, mcnemar, pooled_pred, stack,  # noqa: E402
                       stars)

JSON_OUT = os.path.join(C.RESULTS, "baseline_matrix.json")
MD_OUT = os.path.join(C.RESULTS, "baseline_matrix.md")

MAIN = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
ABL = ["A1", "A2", "A3", "A4", "A5"]
REFS = ["BERT-base", "mBERT"]
EXTRA = ["A0", "S8"]
TOST_MARGINS = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0]


def stats_of(rows):
    P, Y = stack(rows)
    pred = pooled_pred(P)
    per_seed = (P.argmax(-1) == Y[None, :]).mean(axis=1)
    corr_pooled = (pred == Y).astype(np.float64)
    # per-paragraph correctness averaged over seeds: the float vector whose mean
    # is the per-seed mean accuracy, i.e. the estimator Table 7 reports.
    corr_mean = (P.argmax(-1) == Y[None, :]).mean(axis=0).astype(np.float64)
    return {
        "seeds": [r["seed"] for r in rows],
        "n": int(len(Y)),
        "acc_pooled": float(corr_pooled.mean()),
        "acc_mean": float(per_seed.mean()),
        "acc_std": float(per_seed.std(ddof=0)),
        "pred": pred,
        "gold": Y,
        "corr_pooled": corr_pooled,
        "corr_mean": corr_mean,
        "mlm_top1": (float(np.mean([r["mlm_top1"] for r in rows]))
                     if rows[0].get("mlm_top1") is not None else None),
        "f1_pooled": float(_macro_f1(pred, Y)),
    }


def _macro_f1(pred, gold):
    f1 = []
    for c in np.unique(gold):
        tp = int(np.sum((pred == c) & (gold == c)))
        fp = int(np.sum((pred == c) & (gold != c)))
        fn = int(np.sum((pred != c) & (gold == c)))
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1))


def bootstrap_dist(a, b, n_boot=B, seed=SEED):
    """Paired bootstrap distribution of mean(a) - mean(b) over paragraphs."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    return a[idx].mean(axis=1) - b[idx].mean(axis=1)


def contrast(a, b, estimator="pooled"):
    """`a` against `b`; positive = a better.  estimator in {pooled, mean}."""
    if estimator == "pooled":
        d = bootstrap_dist(a["corr_pooled"], b["corr_pooled"])
    else:
        d = bootstrap_dist(a["corr_mean"], b["corr_mean"])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    out = {"delta": float(d.mean()), "ci95": [float(lo), float(hi)],
           "p_bootstrap": float(min(p, 1.0)), "estimator": estimator}
    if estimator == "pooled":
        pm, n01, n10 = mcnemar(a["pred"], b["pred"], a["gold"])
        out["p_mcnemar"] = pm
        out["only_b_correct"] = n01
        out["only_a_correct"] = n10
    return out


def tost(d, margin):
    """Equivalence test from the bootstrap distribution of the paired delta.

    TOST rejects the null of a difference of at least `margin` points when the
    whole (1-2a) CI of the difference lies inside (-margin, +margin); with the
    bootstrap distribution that is p = max(P(d >= +m), P(d <= -m)).
    """
    return float(max((d >= margin * 0.01).mean(), (d <= -margin * 0.01).mean()))


def smallest_equivalent_margin(d, p_target=0.05, hi=12.0, step=0.1):
    m = step
    while m <= hi:
        if tost(d, m) < p_target:
            return float(m)
        m += step
    return None


def holm(pairs, family_size):
    """Holm step-down over the p-values of the declared primary family."""
    order = sorted(pairs, key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (name, p) in enumerate(order):
        adj = min(1.0, (family_size - i) * p)
        running = max(running, adj)
        out[name] = (p, running)
    return out


def main():
    rows = (load("students.json") + load("teachers.json")
            + load("ablations.json") + load("students_S8.json")
            + load("followups.json"))
    G = group(rows)
    S = {t: stats_of(r) for t, r in G.items()}

    rep = {"systems": {}, "vs_S0": {}, "vs_A0": {}, "vs_refs": {},
           "holm": {}, "per_class": {}}

    for t, s in S.items():
        rep["systems"][t] = {
            "n_seeds": len(s["seeds"]), "seeds": s["seeds"],
            "acc_mean": s["acc_mean"], "acc_std": s["acc_std"],
            "acc_pooled": s["acc_pooled"], "f1_pooled": s["f1_pooled"],
            "mlm_top1": s["mlm_top1"],
        }

    # ---- every system against both baselines ------------------------------
    order = MAIN + ABL + REFS + EXTRA
    for tag in order:
        if tag == "S0":
            continue
        rep["vs_S0"][tag] = contrast(S[tag], S["S0"], "pooled")
        rep["vs_A0"][tag] = contrast(S[tag], S["A0"], "pooled")
    rep["vs_S0"]["A0"] = contrast(S["A0"], S["S0"], "pooled")

    # ---- the same systems against the reference encoders ------------------
    for tag in MAIN + ["A0", "A4", "S8"]:
        for ref in REFS:
            c = contrast(S[tag], S[ref], "mean")
            d = bootstrap_dist(S[tag]["corr_mean"], S[ref]["corr_mean"])
            c["tost"] = {f"{m:g}": tost(d, m) for m in TOST_MARGINS}
            c["equivalent_within_pp"] = smallest_equivalent_margin(d)
            pm, n01, n10 = mcnemar(S[tag]["pred"], S[ref]["pred"], S[tag]["gold"])
            c["p_mcnemar"] = pm
            c["only_ref_correct"] = n01
            c["only_system_correct"] = n10
            rep["vs_refs"][f"{tag}_vs_{ref}"] = c

    # ---- multiplicity -----------------------------------------------------
    for key, fam in (("vs_S0_primary_family", "vs_S0"),
                     ("vs_A0_primary_family", "vs_A0")):
        rows_b = [(t, rep[fam][t]["p_bootstrap"]) for t in MAIN if t != "S0"]
        rows_m = [(t, rep[fam][t]["p_mcnemar"]) for t in MAIN if t != "S0"]
        hb = holm(rows_b, len(rows_b))
        hm = holm(rows_m, len(rows_m))
        rep["holm"][key] = {
            "size": len(rows_b),
            "adjusted": {t: {"raw_bootstrap": hb[t][0], "holm_bootstrap": hb[t][1],
                             "raw_mcnemar": hm[t][0], "holm_mcnemar": hm[t][1]}
                         for t in hb}}

    # ---- per-class detail -------------------------------------------------
    for tag in ["S0", "A0", "S4", "S6", "S7", "A4", "S8", "BERT-base"]:
        if tag not in S:
            continue
        pred, gold = S[tag]["pred"], S[tag]["gold"]
        per_class = {}
        for c in np.unique(gold):
            tp = int(np.sum((pred == c) & (gold == c)))
            fp = int(np.sum((pred == c) & (gold != c)))
            fn = int(np.sum((pred != c) & (gold == c)))
            per_class[int(c)] = {
                "support": int(np.sum(gold == c)), "tp": tp, "fp": fp, "fn": fn,
                "f1": 0.0 if (2 * tp + fp + fn) == 0
                      else 2 * tp / (2 * tp + fp + fn)}
        rep["per_class"][tag] = per_class

    C.json_dump(rep, JSON_OUT)

    # ---- report -----------------------------------------------------------
    L = []
    L.append("# Second baseline, equivalence and multiplicity\n")
    L.append("## Per-system accuracy on the 196 test paragraphs\n")
    L.append("| tag | n seeds | mean +- sd (%) | pooled (%) | macro-F1 pooled (%) | MLM top-1 (%) |")
    L.append("|---|---|---|---|---|---|")
    for tag in order:
        if tag not in S:
            continue
        s = rep["systems"][tag]
        mlm = "n/a" if s["mlm_top1"] is None else f"{s['mlm_top1'] * 100:.1f}"
        L.append(f"| {tag} | {s['n_seeds']} | {s['acc_mean'] * 100:.2f} +- "
                 f"{s['acc_std'] * 100:.2f} | {s['acc_pooled'] * 100:.2f} | "
                 f"{s['f1_pooled'] * 100:.2f} | {mlm} |")

    def row(tag, ref, c):
        return (f"| {tag} | {ref} | {c['delta'] * 100:+.2f} | "
                f"[{c['ci95'][0] * 100:+.2f}, {c['ci95'][1] * 100:+.2f}] | "
                f"{c['p_bootstrap']:.4f} {stars(c['p_bootstrap'])} | "
                f"{c.get('p_mcnemar', float('nan')):.4f} "
                f"{stars(c.get('p_mcnemar', 1.0))} | "
                f"{c.get('only_b_correct', '-')} | {c.get('only_a_correct', '-')} |")

    L.append("\n## (a) every system against S0 (the published baseline)\n")
    L.append("| system | reference | delta (pp) | 95% CI | bootstrap p | McNemar p | only S0 | only system |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tag in order:
        if tag in rep["vs_S0"]:
            L.append(row(tag, "S0", rep["vs_S0"][tag]))

    L.append("\n## (b) every system against A0 (S0 minus L_mlm)\n")
    L.append("| system | reference | delta (pp) | 95% CI | bootstrap p | McNemar p | only A0 | only system |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tag in order:
        if tag in rep["vs_A0"]:
            L.append(row(tag, "A0", rep["vs_A0"][tag]))

    L.append("\n## (c) against the reference encoders, with TOST\n")
    L.append("| system | reference | delta (pp) | 95% CI | bootstrap p | McNemar p | "
             "TOST p (2 pp) | equivalent within (pp) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tag in MAIN + ["A0", "A4", "S8"]:
        for ref in REFS:
            c = rep["vs_refs"][f"{tag}_vs_{ref}"]
            eq = c["equivalent_within_pp"]
            L.append(f"| {tag} | {ref} | {c['delta'] * 100:+.2f} | "
                     f"[{c['ci95'][0] * 100:+.2f}, {c['ci95'][1] * 100:+.2f}] | "
                     f"{c['p_bootstrap']:.4f} {stars(c['p_bootstrap'])} | "
                     f"{c['p_mcnemar']:.4f} {stars(c['p_mcnemar'])} | "
                     f"{c['tost']['2']:.4f} | "
                     f"{'none <= %.1f' % 12.0 if eq is None else '%.1f' % eq} |")

    L.append("\n## (d) Holm adjustment over the declared primary family\n")
    L.append("The family is the seven student systems of the main comparison "
             "against one baseline.  Both tests are adjusted, because the "
             "paper's criterion requires both to reject.")
    for fam, key in (("vs S0", "vs_S0_primary_family"),
                     ("vs A0", "vs_A0_primary_family")):
        adj = rep["holm"][key]["adjusted"]
        L.append(f"\n{key.replace('_', ' ')} (family size {rep['holm'][key]['size']}):")
        L.append("| system | raw boot p | Holm boot p | raw McNemar p | Holm McNemar p |")
        L.append("|---|---|---|---|---|")
        for tag in MAIN:
            if tag in adj:
                a = adj[tag]
                L.append(f"| {tag} | {a['raw_bootstrap']:.4f} | "
                         f"{a['holm_bootstrap']:.4f} | {a['raw_mcnemar']:.4f} | "
                         f"{a['holm_mcnemar']:.4f} |")

    L.append("\n## (e) per-class F1 (pooled predictor)\n")
    tags = [t for t in ["S0", "A0", "S4", "S6", "S7", "A4", "S8", "BERT-base"]
            if t in rep["per_class"]]
    L.append("| class | support | " + " | ".join(tags) + " |")
    L.append("|---" * (len(tags) + 2) + "|")
    classes = sorted(rep["per_class"][tags[0]])
    for c in classes:
        sup = rep["per_class"][tags[0]][c]["support"]
        L.append(f"| {c} | {sup} | " + " | ".join(
            f"{rep['per_class'][t][c]['f1'] * 100:.0f}" for t in tags) + " |")

    text = "\n".join(L) + "\n"
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
