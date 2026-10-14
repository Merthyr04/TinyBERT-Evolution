"""Ablation study and significance testing.

Two tests are used, both paired on the same 196 test paragraphs of LD-100K:

  * paired bootstrap (Berg-Kirkpatrick et al., 2012).  For each system the
    per-paragraph prediction is the argmax of the class posterior averaged over
    the seeds, which makes the comparison deterministic.  10,000 bootstrap
    resamples of the 196 paragraphs give the 95% CI of the accuracy
    difference and a two-sided p-value.
  * McNemar's exact test on the discordant pairs of the same two predictors,
    which only uses the paragraphs where the two systems disagree and is
    therefore robust to the (very different) difficulty of individual
    paragraphs.

A difference is only reported as an improvement when both tests agree.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import binomtest

import common as C

B = 10000
SEED = 20261003

ORDER = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "BERT-base", "mBERT"]
ABL_ORDER = ["A1", "A2", "A3", "A4", "A5"]


def load(name):
    p = os.path.join(C.RESULTS, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def group(rows):
    out = {}
    for r in rows:
        out.setdefault(r["tag"], []).append(r)
    return out


def stack(rows):
    """-> (seeds, n, C) posterior stack and the gold labels."""
    P = np.asarray([r["test_probs"] for r in rows], dtype=np.float64)
    Y = np.asarray(rows[0]["test_gold"], dtype=np.int64)
    return P, Y


def stack_seeds(rows, seeds):
    """Same as stack(), restricted to the given seeds.

    The ablation cells are run with fewer seeds than the parent systems they
    are compared against.  Pooling a 3-seed parent against a 2-seed ablation
    would confound the removed ingredient with the extra seed, so the parent
    is restricted to exactly the seeds the ablation was run with.
    """
    keep = [r for r in rows if r["seed"] in set(seeds)]
    if not keep:
        raise ValueError(f"no rows for seeds {sorted(seeds)}")
    return stack(keep)


def pooled_pred(P):
    return P.mean(axis=0).argmax(axis=-1)


def per_seed_correct(P, Y):
    return (P.argmax(-1) == Y[None, :]).mean(axis=1)


def paired_bootstrap(corr_a, corr_b, n_boot=B, seed=SEED):
    """corr_* are per-paragraph 0/1 vectors (or float accuracies)."""
    rng = np.random.default_rng(seed)
    n = len(corr_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = corr_a[idx].mean(axis=1) - corr_b[idx].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def mcnemar(pred_a, pred_b, gold):
    """McNemar's exact test on the discordant pairs of two predictors.

    The three arguments arrive as plain Python lists (they are read back from
    the result JSONs).  They must be converted to arrays first: `list == list`
    is a single boolean in Python, so comparing two lists of predictions would
    silently produce zero discordant pairs and a p-value of 1.0 for every
    contrast.
    """
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    gold = np.asarray(gold)
    a_ok = pred_a == gold
    b_ok = pred_b == gold
    n01 = int(np.sum(~a_ok & b_ok))     # only B correct
    n10 = int(np.sum(a_ok & ~b_ok))     # only A correct
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return float(binomtest(n01, n01 + n10, 0.5).pvalue), n01, n10


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    students = load("students.json")
    teachers = load("teachers.json")
    ablations = load("ablations.json")

    sys_rows = group(students)
    ref_rows = group(teachers)
    abl_rows = group(ablations)

    stats = {}
    for tag, rows in list(sys_rows.items()) + list(ref_rows.items()) + \
            list(abl_rows.items()):
        P, Y = stack(rows)
        pred = pooled_pred(P)
        per_seed = per_seed_correct(P, Y)
        stats[tag] = {
            "seeds": [r["seed"] for r in rows],
            "acc_mean": float(per_seed.mean()),
            "acc_std": float(per_seed.std(ddof=0)),
            "acc_pooled": float((pred == Y).mean()),
            "macro_f1": float(np.mean([r["test_macro_f1"] for r in rows])),
            "mlm_top1": (float(np.mean([r["mlm_top1"] for r in rows]))
                         if "mlm_top1" in rows[0] and rows[0]["mlm_top1"] is not None
                         else None),
            "mlm_top5": (float(np.mean([r["mlm_top5"] for r in rows]))
                         if "mlm_top5" in rows[0] and rows[0]["mlm_top5"] is not None
                         else None),
            "pps": float(np.mean([r["passages_per_s"] for r in rows])),
            "exit_rate": (float(np.mean([r["exit_rate"] for r in rows]))
                          if rows[0].get("exit_rate") is not None else None),
            "train_time_s": float(np.mean([r["train_time_s"] for r in rows])),
            "peak_train_vram_mb": (float(np.mean([r["peak_train_vram_mb"] for r in rows]))
                                   if rows[0].get("peak_train_vram_mb") is not None else None),
            "params_m": rows[0].get("params_m"),
            "corr": (pred == Y).astype(np.float64),
            "pred": pred.tolist(),
            "gold": Y.tolist(),
        }

    base = stats.get("S0")
    out = {"per_system": {}, "contrasts": [], "ablation": []}

    def pct(x):
        return f"{x * 100:.1f}"

    # ---------------- main contrast table: everything vs S0 -------------
    lines = ["| System | seeds | PTC acc. (mean $\\pm$ s.d.) | pooled acc. | "
             "$\\Delta$ vs S0 | 95% CI of $\\Delta$ | bootstrap p | McNemar p |",
             "|---|---|---|---|---|---|---|---|"]
    for tag in ORDER:
        s = stats.get(tag)
        if s is None:
            continue
        if tag == "S0":
            lines.append(f"| **{tag}** | {len(s['seeds'])} | "
                         f"{pct(s['acc_mean'])} $\\pm$ {s['acc_std']*100:.1f} | "
                         f"{pct(s['acc_pooled'])} | - | - | - | - |")
            continue
        d, lo, hi, p = paired_bootstrap(s["corr"], base["corr"])
        pm, n01, n10 = mcnemar(s["pred"], base["pred"], base["gold"])
        out["contrasts"].append({
            "system": tag, "vs": "S0", "delta": d, "ci95": [lo, hi],
            "p_bootstrap": p, "p_mcnemar": pm,
            "mcnemar_only_target_correct": n01,
            "mcnemar_only_system_correct": n10,
        })
        lines.append(
            f"| {tag} | {len(s['seeds'])} | {pct(s['acc_mean'])} $\\pm$ "
            f"{s['acc_std']*100:.1f} | {pct(s['acc_pooled'])} | "
            f"{'+' if d >= 0 else ''}{d*100:.1f} | "
            f"[{lo*100:+.1f}, {hi*100:+.1f}] | {p:.4f} {stars(p)} | "
            f"{pm:.4f} {stars(pm)} |")

    # ---------------- ablation table ------------------------------------
    alines = ["| Ablation | removed ingredient | seeds | ablation PTC acc. | "
              "full model (same seeds) | $\\Delta$ vs full model | 95% CI | "
              "bootstrap p | McNemar p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for tag in ABL_ORDER:
        s = stats.get(tag)
        if s is None:
            continue
        parent = "S6" if tag != "A5" else "S7"
        par = stats.get(parent)
        if par is None or parent not in sys_rows:
            alines.append(f"| {tag} | (parent {parent} missing) | - | - | - | - | - | - |")
            continue
        # Seed-matched parent: same seeds the ablation cell was run with.
        Pab, Yab = stack_seeds(abl_rows[tag], s["seeds"])
        Ppa, Ypa = stack_seeds(sys_rows[parent], s["seeds"])
        corr_ab = (pooled_pred(Pab) == Yab).astype(np.float64)
        corr_pa = (pooled_pred(Ppa) == Ypa).astype(np.float64)
        pred_ab, pred_pa = pooled_pred(Pab), pooled_pred(Ppa)
        d, lo, hi, p = paired_bootstrap(corr_ab, corr_pa)
        pm, n01, n10 = mcnemar(pred_ab, pred_pa, Ypa)
        out["ablation"].append({
            "tag": tag, "parent": parent,
            "seeds": sorted(set(s["seeds"])),
            "acc": float(corr_ab.mean()),
            "acc_parent_seed_matched": float(corr_pa.mean()),
            "delta": d, "ci95": [lo, hi],
            "p_bootstrap": p, "p_mcnemar": pm,
        })
        alines.append(
            f"| {tag} | {ABL_REMOVES.get(tag, '')} | {len(s['seeds'])} | "
            f"{pct(corr_ab.mean())} $\\pm$ {s['acc_std']*100:.1f} | "
            f"{pct(corr_pa.mean())} | "
            f"{'+' if d >= 0 else ''}{d*100:.1f} | "
            f"[{lo*100:+.1f}, {hi*100:+.1f}] | {p:.4f} {stars(p)} | "
            f"{pm:.4f} {stars(pm)} |")

    # ---------------- accuracy / efficiency table -----------------------
    # Prefer the single-process, round-robin benchmark: in-training throughput
    # is contaminated by hours of GPU clock drift between runs.
    bench = load("benchmark.json") or {}
    bsys = bench.get("systems", {})
    bench_base = bench.get("baseline_pps")
    base_pps = bench_base or (stats["S0"]["pps"] if "S0" in stats else None)

    def pps_of(tag):
        if tag in bsys:
            return bsys[tag]["passages_per_s"]
        return stats.get(tag, {}).get("pps")

    elines = ["| System | params (M) | PTC acc. | MLM top-1 | passages/s | "
              "speed index | exit rate | train time (s) | peak train VRAM (MB) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for tag in ORDER + [t for t in sorted(stats) if t not in ORDER]:
        s = stats.get(tag)
        if s is None:
            continue
        pps_v = pps_of(tag)
        si = 100.0 * pps_v / base_pps if (base_pps and pps_v) else None
        ptv = ("-" if s["peak_train_vram_mb"] is None
               else f"{s['peak_train_vram_mb']:.0f}")
        exit_rate = s["exit_rate"]
        if tag in bsys and "exit_rate" in bsys[tag]:
            exit_rate = bsys[tag]["exit_rate"]
        ex = "-" if exit_rate is None else pct(exit_rate)
        m1 = "-" if s["mlm_top1"] is None else pct(s["mlm_top1"])
        out["per_system"][tag] = {k: v for k, v in s.items()
                                  if k not in ("corr", "pred", "gold")}
        out["per_system"][tag]["pps"] = pps_v
        out["per_system"][tag]["speed_index"] = si
        out["per_system"][tag]["drop_train_time_s"] = s["train_time_s"]
        elines.append(
            f"| {tag} | {s['params_m']} | {pct(s['acc_mean'])} $\\pm$ "
            f"{s['acc_std']*100:.1f} | {m1} | {pps_v:.1f} | {si:.1f} | "
            f"{ex} | {s['train_time_s']:.0f} | {ptv} |")

    md = []
    md.append("### Accuracy of every system against the baseline\n")
    md.extend(lines)
    md.append("\n### Ablation: removing one ingredient from the full model\n")
    md.extend(alines)
    md.append("\n### Accuracy / efficiency\n")
    md.extend(elines)
    md.append("\nSignificance: `*` p<0.05, `**` p<0.01, `***` p<0.001, "
              "`n.s.` not significant.  95% CIs from 10,000 paired bootstrap "
              "resamples of the LD-100K test split.\n")
    md.append("\nIn the ablation table the full model is evaluated on exactly "
              "the seeds the ablation cell was run with (13, 37 and 97), so the "
              "difference is attributable to the removed ingredient and not "
              "to a different number of seeds.\n")

    text = "\n".join(md)
    with open(os.path.join(C.RESULTS, "ablation.md"), "w", encoding="utf-8") as f:
        f.write(text)
    C.json_dump(out, os.path.join(C.RESULTS, "ablation.json"))
    print(text)


ABL_REMOVES = {
    "A1": "attention distillation $L_{attn}$",
    "A2": "hidden-state distillation $L_{hid}$",
    "A3": "prediction distillation $L_{pred}$",
    "A4": "auxiliary MLM objective $L_{mlm}$",
    "A5": "data augmentation",
}

if __name__ == "__main__":
    main()
