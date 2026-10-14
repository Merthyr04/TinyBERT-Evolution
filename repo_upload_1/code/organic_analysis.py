"""Organic-corpus replication analysis (review P1-1 / Section 6.6).

Reads results/organic_students.json (the key subset S0 / S4 / S6 / S7 / A0 / F2
on five seeds, trained on OD-100K), results/organic_teachers.json (the two
reference encoders on the same corpus) and results/lexical_baseline_organic.json,
and reports the three things the review asks for:

  (a) the system ordering on the organic document;
  (b) the per-seed delta against both S0 and A0, plus the paired tests on the
      pooled predictor;
  (c) the exit rate and the shallow/deep head agreement rate.

Outputs results/organic_analysis.json and results/organic_analysis.md.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common as C
from ablations import (group, load, mcnemar, paired_bootstrap, per_seed_correct,
                       pooled_pred, stack, stars)

SEEDS = [7, 13, 37, 42, 97]
ORDER = ["S0", "A0", "S4", "S6", "S7", "F2"]
REFS = ["S0", "A0"]


def mean_sd(v):
    """Mean and *population* standard deviation (ddof = 0), which is the
    convention every other table in the paper uses (`students.json` reports
    acc_std the same way); the two conventions differ by a factor of
    sqrt(4/5) on five seeds and would make Table 11(a) incomparable with
    Table 4."""
    v = np.asarray(v, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=0)) if len(v) > 1 else 0.0


def theta05_exit_rate(tag="S7"):
    """Exit rate at the registered operating point theta = 0.5, on the 196 test
    paragraphs.  The `exit_rate` field of organic_students.json is computed at
    the code default theta = 0.8 and is *not* the registered convention; the
    theta = 0.5 numbers come from head_agreement_organic.json, which measures
    the same split under the same window."""
    path = os.path.join(C.RESULTS, "head_agreement_organic.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    vals = [r["exit_rate"] for r in rep["rows"] if r["tag"] == tag]
    agree = [r["aux_eq_main"] for r in rep["rows"] if r["tag"] == tag]
    chg = [r["exit_ne_deep"] for r in rep["rows"] if r["tag"] == tag]
    if not vals:
        return None
    return {
        "theta": rep.get("theta"),
        "exit_rate_mean": round(float(np.mean(vals)), 4),
        "exit_rate_min": round(float(np.min(vals)), 4),
        "exit_rate_max": round(float(np.max(vals)), 4),
        "agreement_min": int(np.min(agree)), "agreement_max": int(np.max(agree)),
        "agreement_pct_mean": round(float(np.mean(agree)) / 196, 4),
        "changed_min": int(np.min(chg)), "changed_max": int(np.max(chg)),
    }


def main():
    sys_rows = group(load("organic_students.json"))
    ref_rows = group(load("organic_teachers.json"))
    missing = [t for t in ORDER if t not in sys_rows]
    if missing:
        raise SystemExit(f"missing systems in organic_students.json: {missing}")

    order_present = [t for t in ORDER if t in sys_rows]
    rep = {"seeds": SEEDS, "systems": {}, "ordering": [], "contrasts": [],
           "reference": {}, "lexical": {}, "exit_theta05": theta05_exit_rate("S7")}

    # ---- (a) per-system accuracy / macro-F1 and ordering -------------------
    for tag in order_present:
        rows = sorted(sys_rows[tag], key=lambda r: r["seed"])
        acc = [r["test_acc"] for r in rows]
        f1 = [r["test_macro_f1"] for r in rows]
        m, s = mean_sd(acc)
        exit_rates = [r.get("exit_rate") for r in rows]
        agree = [r.get("head_agreement") for r in rows]
        rep["systems"][tag] = {
            "per_seed_acc": [round(a, 4) for a in acc],
            "per_seed_macro_f1": [round(f, 4) for f in f1],
            "acc_mean": round(m, 4), "acc_sd": round(s, 4),
            "macro_f1_mean": round(float(np.mean(f1)), 4),
            "dual_head": bool(rows[0].get("dual_head")),
            "fixed_depth": bool(rows[0].get("fixed_depth")),
            "exit_rate_mean": (round(float(np.mean([e for e in exit_rates
                                                     if e is not None])), 4)
                               if any(e is not None for e in exit_rates) else None),
            "head_agreement_mean": (round(float(np.mean([a for a in agree
                                                         if a is not None])), 4)
                                    if any(a is not None for a in agree) else None),
        }

    ranking = sorted(order_present, key=lambda t: -rep["systems"][t]["acc_mean"])
    rep["ordering"] = ranking

    # ---- reference encoders ------------------------------------------------
    for tag in ("BERT-base", "mBERT"):
        if tag in ref_rows:
            acc = [r["test_acc"] for r in ref_rows[tag]]
            m, s = mean_sd(acc)
            rep["reference"][tag] = {"acc_mean": round(m, 4), "acc_sd": round(s, 4),
                                     "per_seed_acc": [round(a, 4) for a in acc]}

    # ---- (b) per-seed delta against S0 and against A0 ---------------------
    for tag in order_present:
        rows = {r["seed"]: r for r in sys_rows[tag]}
        entry = {"system": tag, "vs": {}}
        for ref in REFS:
            rr = {r["seed"]: r for r in sys_rows[ref]}
            shared = [s for s in SEEDS if s in rows and s in rr]
            d = [rows[s]["test_acc"] - rr[s]["test_acc"] for s in shared]
            entry["vs"][ref] = {
                "seeds": shared,
                "per_seed_delta_acc": [round(x, 4) for x in d],
                "mean_delta_acc": round(float(np.mean(d)), 4) if d else None,
            }
        rep["contrasts"].append(entry)

    # ---- paired tests on the pooled predictor -----------------------------
    pooled = {}
    for tag in order_present:
        P, Y = stack(sorted(sys_rows[tag], key=lambda r: r["seed"]))
        pred = pooled_pred(P)
        pooled[tag] = {"pred": pred,
                       "corr": (pred == Y).astype(np.float64), "gold": Y}
    tests = []
    for tag in order_present:
        for ref in REFS:
            if tag == ref:
                continue
            a, b = pooled[tag], pooled[ref]
            d, lo, hi, p = paired_bootstrap(a["corr"], b["corr"])
            pm, n01, n10 = mcnemar(a["pred"], b["pred"], a["gold"])
            tests.append({"system": tag, "vs": ref, "delta": d,
                          "ci95": [lo, hi], "p_bootstrap": p, "p_mcnemar": pm,
                          "only_ref_correct": n01, "only_system_correct": n10})
    rep["paired_tests"] = tests

    # ---- lexical ceiling on the organic split -----------------------------
    lex = load("lexical_baseline_organic.json")
    if lex:
        rep["lexical"] = {
            "split": lex.get("split"),
            "real_split": [{"name": c["name"], "acc": c["acc_eval"],
                            "f1": c["f1_eval"]} for c in lex.get("real_split", [])],
        }

    C.json_dump(rep, os.path.join(C.RESULTS, "organic_analysis.json"))

    # ---- markdown ---------------------------------------------------------
    L = ["# Organic-corpus replication (OD-100K, Nuttall Encyclopaedia)\n",
         f"Seeds: {SEEDS}.  Split 120 / 100 / 196, identical protocol.\n",
         "## (a) System ordering\n",
         "| System | PTC mean (%) | sd | per-seed PTC (%) | macro-F1 (%) | "
         "exit rate (%) | shallow/deep agreement (%) |",
         "|---|---|---|---|---|---|---|"]
    for i, tag in enumerate(ranking, 1):
        s = rep["systems"][tag]
        if tag == "S7" and rep["exit_theta05"]:
            ex = f"{rep['exit_theta05']['exit_rate_mean'] * 100:.1f}"
            ag = f"{rep['exit_theta05']['agreement_pct_mean'] * 100:.1f}"
        else:
            ex = "-"
            ag = "-"
        L.append(f"| {i}. {tag} | {s['acc_mean'] * 100:.1f} | "
                 f"{s['acc_sd'] * 100:.1f} | "
                 f"{', '.join(f'{a * 100:.1f}' for a in s['per_seed_acc'])} | "
                 f"{s['macro_f1_mean'] * 100:.1f} | {ex} | {ag} |")
    L.append("")
    if rep["exit_theta05"]:
        e = rep["exit_theta05"]
        L.append(f"Exit rate and shallow/deep agreement are at theta = {e['theta']} on "
                 f"the 196 test paragraphs (the registered operating point); "
                 f"exit rate {e['exit_rate_min'] * 100:.1f}-{e['exit_rate_max'] * 100:.1f}"
                 f"%, heads agree on {e['agreement_min']}-{e['agreement_max']} of 196, "
                 f"the exit changes {e['changed_min']}-{e['changed_max']} decisions.")
        L.append("")
    for tag, s in rep["reference"].items():
        L.append(f"| - {tag} | {s['acc_mean'] * 100:.1f} | {s['acc_sd'] * 100:.1f} | "
                 f"{', '.join(f'{a * 100:.1f}' for a in s['per_seed_acc'])} | - | - | - |")

    L.append("\n## (b) Per-seed delta against S0 and A0\n")
    L.append("| System | vs | per-seed ΔPTC (pp) | mean ΔPTC (pp) |")
    L.append("|---|---|---|---|")
    for e in rep["contrasts"]:
        for ref, v in e["vs"].items():
            ds = ", ".join(f"{x * 100:+.1f}" for x in v["per_seed_delta_acc"])
            md = "-" if v["mean_delta_acc"] is None else f"{v['mean_delta_acc'] * 100:+.1f}"
            L.append(f"| {e['system']} | {ref} | {ds} | {md} |")

    L.append("\nPaired tests on the pooled predictor (196 test paragraphs):\n")
    L.append("| System | vs | ΔPTC (pp) | 95% CI | bootstrap p | McNemar p |")
    L.append("|---|---|---|---|---|---|")
    for t in tests:
        L.append(f"| {t['system']} | {t['vs']} | "
                 f"{'+' if t['delta'] >= 0 else ''}{t['delta'] * 100:.1f} | "
                 f"[{t['ci95'][0] * 100:+.1f}, {t['ci95'][1] * 100:+.1f}] | "
                 f"{t['p_bootstrap']:.4f} {stars(t['p_bootstrap'])} | "
                 f"{t['p_mcnemar']:.4f} {stars(t['p_mcnemar'])} |")

    if rep["lexical"]:
        L.append("\n## Lexical ceiling on the organic split\n")
        L.append("| probe | PTC (%) | macro-F1 (%) |")
        L.append("|---|---|---|")
        for c in rep["lexical"]["real_split"]:
            L.append(f"| {c['name']} | {c['acc'] * 100:.1f} | {c['f1'] * 100:.1f} |")

    text = "\n".join(L) + "\n"
    with open(os.path.join(C.RESULTS, "organic_analysis.md"), "w",
              encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
