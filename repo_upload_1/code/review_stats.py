"""Round-75 revision support: statistics recomputed from stored predictions.

Nothing here trains or times anything -- every number is derived from the
`test_probs` / `test_gold` arrays already stored in results/*.json, plus the
timing tables.  It exists to answer four points raised by the Applied
Intelligence reviewer panel:

  A. per-class precision / recall / F1 and the confusion matrix for LD-100K
     (the panel asked for them; Table 4 reports accuracy and macro-F1 only);
  B. the operating characteristics of the paper's dual-test conjunction rule
     (Section 4.2): its real size at a nominal 0.05 and its power at the
     effect sizes the paper actually studies;
  C. the stability of the Figure 4 non-dominated set under the noise floor
     the paper itself declares (Section 4.9(v));
  D. the sampling properties of the Table 7 "smallest equivalent margin"
     column, which is a maximum and therefore upward biased.

Everything is vectorised: a paired data set is carried as its 2x2 table, and
a bootstrap replicate is a multinomial draw over that table.

Writes results/review_stats.json and results/review_stats.md.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

SEEDS = (7, 13, 37, 42, 97)
N_TEST = 196
BOOT = 2000          # inner bootstrap replicates per simulated data set
SIM = 1500           # simulated data sets per operating characteristic
SEED = 20261011


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
MAIN = ["students.json", "ablations.json", "followups.json",
        "students_S8.json", "students_S9.json", "fixed_depth.json",
        "teachers.json"]


def collect():
    out = {}
    for f in MAIN:
        with open(os.path.join(RES, f), encoding="utf-8") as fh:
            for r in json.load(fh):
                out.setdefault(r["tag"], {})[r["seed"]] = r
    return out


RUNS = collect()
GOLD = np.asarray(RUNS["S0"][SEEDS[0]]["test_gold"]).astype(int)


def tag_pred(tag, kind="pooled"):
    recs = [RUNS[tag][s] for s in SEEDS if s in RUNS[tag]]
    P = np.stack([np.asarray(r["test_probs"]) for r in recs])
    if kind == "pooled":
        return P.mean(axis=0).argmax(-1)
    return P.argmax(-1)[0]


def correctness(tag, kind="pooled"):
    """Per-paragraph correctness under one of the paper's two estimators.

    `pooled`  -- argmax of the seed-averaged posterior, a 0/1 vector;
    `mean`    -- average over seeds of per-seed correctness, a vector on the
                 1/|seeds| grid whose mean is the per-seed mean accuracy
                 (this is `corr_mean` in baseline_matrix.py, and it is the
                 vector the Table 7 equivalence margins are computed from).
    """
    recs = [RUNS[tag][s] for s in SEEDS if s in RUNS[tag]]
    P = np.stack([np.asarray(r["test_probs"]) for r in recs])
    if kind == "pooled":
        return (P.mean(axis=0).argmax(-1) == GOLD).astype(float)
    return (P.argmax(-1) == GOLD[None, :]).mean(axis=0).astype(float)


def acc(tag, kind="pooled"):
    return 100.0 * correctness(tag, kind).mean()


# ---------------------------------------------------------------------------
# exact McNemar probability table
# ---------------------------------------------------------------------------
_P_MCN = np.ones((N_TEST + 1, N_TEST + 1))
for _s in range(N_TEST + 1):
    for _k in range(_s + 1):
        _P_MCN[_s, _k] = binomtest(_k, _s, 0.5).pvalue


def boot_from_probs(probs, n, boot, rng):
    """(R,4) cell probabilities -> (R, boot, 4) multinomial counts."""
    R = probs.shape[0]
    big = np.repeat(probs, boot, axis=0)
    c = rng.multinomial(n, big)
    return c.reshape(R, boot, 4)


def diff_from_counts(c, n):
    """mean(A) - mean(B) from 2x2 counts ordered (11, 10, 01, 00)."""
    return (c[..., 1] - c[..., 2]) / n


# ---------------------------------------------------------------------------
# A. per-class metrics
# ---------------------------------------------------------------------------
def per_class(tag, kind="pooled"):
    yp = tag_pred(tag, kind)
    C = int(max(GOLD.max(), yp.max())) + 1
    cm = np.zeros((C, C), dtype=int)
    for g, p in zip(GOLD, yp):
        cm[g, p] += 1
    rows = []
    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(cls=c, n=int(cm[c, :].sum()), tp=int(tp),
                         precision=round(100 * prec, 2),
                         recall=round(100 * rec, 2), f1=round(100 * f1, 2)))
    return rows, cm.tolist()


# ---------------------------------------------------------------------------
# B. the conjunction rule
# ---------------------------------------------------------------------------
def pair_table(tag_a, tag_b):
    a = correctness(tag_a).astype(int)
    b = correctness(tag_b).astype(int)
    n = len(a)
    return dict(n=n,
                n11=int(np.sum((a == 1) & (b == 1))),
                n10=int(np.sum((a == 1) & (b == 0))),
                n01=int(np.sum((a == 0) & (b == 1))),
                n00=int(np.sum((a == 0) & (b == 0))))


def rule_characteristics(tab, delta_pts, reps=SIM, boot=BOOT, seed=SEED,
                         alpha=0.05):
    """Size / power of the conjunction rule at a target advantage.

    The 2x2 table `tab` is the empirical template.  A target advantage of
    `delta_pts` accuracy points is imposed by moving probability mass from
    (A wrong, B right) to (A right, B wrong); the concordant cells are held
    fixed.  Each replicate draws n items from the resulting multinomial, and
    each replicate is then bootstrapped `boot` times.  Both tests run at a
    nominal 0.05; the rule is their conjunction.
    """
    n = tab["n"]
    p = np.array([tab["n11"], tab["n10"], tab["n01"], tab["n00"]], float) / n
    # Hold the discordance rate fixed and split it evenly at delta = 0, so that
    # `delta_pts = 0` is a genuine size calculation rather than a rerun of the
    # pair's own observed advantage.
    q = p[1] + p[2]
    p10 = q / 2.0 + delta_pts / 200.0
    p01 = q / 2.0 - delta_pts / 200.0
    if p01 < 0:
        p10 += -p01
        p01 = 0.0
    if p10 < 0:
        p01 += -p10
        p10 = 0.0
    probs = np.array([p[0], p10, p01, p[3]])
    probs = probs / probs.sum()
    realised = 100.0 * (probs[1] - probs[2])

    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, probs, size=reps)          # (reps,4)
    emp = draws / n
    bc = boot_from_probs(emp, n, boot, rng)               # (reps,boot,4)
    d = diff_from_counts(bc, n)                           # (reps,boot)
    lo = np.percentile(d, 100 * alpha / 2, axis=1)
    hi = np.percentile(d, 100 * (1 - alpha / 2), axis=1)
    rej_boot = (lo > 0) | (hi < 0)

    # McNemar is applied to the data set itself (its own discordant pairs),
    # not to the bootstrap replicates.
    s = draws[:, 1] + draws[:, 2]
    k = draws[:, 1]
    rej_mcn = _P_MCN[np.clip(s, 0, N_TEST), np.clip(k, 0, N_TEST)] < alpha

    frac_sig_boot = float(rej_boot.mean())
    frac_sig_mcn = float(rej_mcn.mean())
    frac_both = float((rej_boot & rej_mcn).mean())
    frac_either = float((rej_boot | rej_mcn).mean())
    return dict(delta_pts=delta_pts, realised_delta=round(realised, 3),
                discordant_expected=round(float(n * (probs[1] + probs[2])), 1),
                bootstrap=round(100 * frac_sig_boot, 2),
                mcnemar=round(100 * frac_sig_mcn, 2),
                conjunction=round(100 * frac_both, 2),
                either=round(100 * frac_either, 2))


# ---------------------------------------------------------------------------
# C. Figure 4 frontier stability
# ---------------------------------------------------------------------------
def figure4_points():
    """The 16 configurations Figure 4 plots, exactly as make_figures.py builds
    them: per-seed-mean accuracy from baseline_matrix.json, speed index from
    the single round-robin in benchmark.json, and A0's index fixed at 100% by
    construction (it shares S0's deployed graph)."""
    with open(os.path.join(RES, "baseline_matrix.json"), encoding="utf-8") as f:
        stats = json.load(f)["systems"]
    with open(os.path.join(RES, "benchmark.json"), encoding="utf-8") as f:
        bench = json.load(f)["systems"]
    order = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7",
             "A0", "A1", "A2", "A3", "A4", "A5", "BERT-base", "mBERT"]
    pts = {}
    for t in order:
        if t not in stats:
            continue
        v = bench.get(t, {}).get("speed_index")
        if v is None:
            if t != "A0":
                continue
            v = 100.0
        pts[t] = (stats[t]["acc_mean"] * 100.0, float(v))
    return pts


def non_dominated(accs, sis):
    """Boolean mask of the non-dominated set (maximise both)."""
    A = accs[:, None] >= accs[None, :]
    S = sis[:, None] >= sis[None, :]
    strict = (accs[:, None] > accs[None, :]) | (sis[:, None] > sis[None, :])
    dom = A & S & strict
    np.fill_diagonal(dom, False)
    return ~dom.any(axis=0)


def non_dominated_tol(accs, sis, si_tol=0.15):
    """Frontier under the paper's own floor rule: A beats B only if A is at
    least `si_tol` faster (a difference larger than the declared floor) and at
    least as accurate."""
    keep = []
    for i in range(len(accs)):
        beaten = False
        for j in range(len(accs)):
            if i == j:
                continue
            if (sis[j] >= sis[i] * (1.0 + si_tol) and accs[j] >= accs[i]):
                beaten = True
                break
        if not beaten:
            keep.append(i)
    return keep


def frontier_stability(reps=4000, si_jitter=0.15, seed=SEED):
    pts = figure4_points()
    names = list(pts)
    base_a = np.array([pts[t][0] for t in names])
    base_s = np.array([pts[t][1] for t in names])
    base = [names[i] for i in np.where(non_dominated(base_a, base_s))[0]]
    floor_aware = [names[i] for i in non_dominated_tol(base_a, base_s)]

    sd = np.zeros(len(names))
    for i, t in enumerate(names):
        if t in RUNS:
            v = [100.0 * (np.asarray(RUNS[t][s]["test_probs"]).argmax(-1)
                          == GOLD).mean()
                 for s in SEEDS if s in RUNS[t]]
            sd[i] = float(np.std(v, ddof=0)) if v else 0.0

    rng = np.random.default_rng(seed)
    cnt_si = np.zeros(len(names))
    cnt_both = np.zeros(len(names))
    for _ in range(reps):
        j = 1.0 + rng.uniform(-si_jitter, si_jitter, size=len(names))
        cnt_si += non_dominated(base_a, base_s * j)
        a2 = base_a + (rng.normal(0, 1.0, size=len(names)) * sd)
        cnt_both += non_dominated(a2, base_s * j)
    return dict(
        baseline_frontier=base, floor_aware_frontier=floor_aware,
        reps=reps, si_jitter=si_jitter,
        points={t: dict(acc=round(pts[t][0], 2), si=round(pts[t][1], 2),
                        acc_sd=round(float(sd[i]), 2))
                for i, t in enumerate(names)},
        inclusion_si_only={t: round(100 * cnt_si[i] / reps, 2)
                           for i, t in enumerate(names)},
        inclusion_si_and_acc={t: round(100 * cnt_both[i] / reps, 2)
                              for i, t in enumerate(names)})


# ---------------------------------------------------------------------------
# D. the "smallest equivalent margin" column
# ---------------------------------------------------------------------------
def _joint36(a, b):
    """Joint distribution of two per-seed-mean correctness vectors.

    Each entry of a per-seed-mean vector is a multiple of 1/|seeds|, so the
    pair (a_i, b_i) lives on a 6x6 grid for five seeds.  Carrying the joint
    table lets the paired bootstrap run on a multinomial instead of on
    resampled float vectors.
    """
    S = len(SEEDS)
    ka = np.rint(a * S).astype(int)
    kb = np.rint(b * S).astype(int)
    cnt = np.bincount(ka * (S + 1) + kb, minlength=(S + 1) ** 2)
    vals = np.arange(S + 1) / S
    grid_ka = np.repeat(np.arange(S + 1), S + 1)
    grid_kb = np.tile(np.arange(S + 1), S + 1)
    dvec = vals[grid_ka] - vals[grid_kb]
    return cnt.astype(float), dvec


def margin_properties(tag_a, tag_b, reps=300, boot=1000, seed=SEED):
    """Sampling distribution of the Table 7 "smallest equivalent margin" column.

    The paper computes that column from the paired bootstrap of the
    **per-seed-mean** estimator (baseline_matrix.py), and `tost` here is the
    identical rule: the first grid point whose TOST p-value drops below 0.05.
    Because that p-value is estimated on a finite bootstrap sample, the column
    is the maximum of a random variable and carries an upward bias, which this
    function quantifies.
    """
    a = correctness(tag_a, "mean")
    b = correctness(tag_b, "mean")
    n = len(a)
    cnt, dvec = _joint36(a, b)
    p = cnt / n
    rng = np.random.default_rng(seed)

    outer = rng.multinomial(n, p, size=reps) / n        # (reps,36)
    big = np.repeat(outer, boot, axis=0)
    bc = rng.multinomial(n, big)                        # (reps*boot,36)
    d = (bc @ dvec).reshape(reps, boot) / n             # (reps,boot)

    grid = np.arange(0.1, 12.01, 0.1)
    ok = np.zeros((reps, len(grid)), dtype=bool)
    for gi, m in enumerate(grid):
        mm = m * 0.01
        ok[:, gi] = (np.maximum((d >= mm).mean(axis=1),
                                (d <= -mm).mean(axis=1)) < 0.05)
    margins = np.full(reps, 12.1)
    any_ok = ok.any(axis=1)
    first = np.argmax(ok, axis=1)
    margins[any_ok] = grid[first[any_ok]]

    d0 = (rng.multinomial(n, p, size=10000) @ dvec) / n
    point = 12.1
    for m in grid:
        mm = m * 0.01
        if max((d0 >= mm).mean(), (d0 <= -mm).mean()) < 0.05:
            point = float(m)
            break

    return dict(pair=f"{tag_a} vs {tag_b}",
                observed_delta=round(100 * float((a - b).mean()), 2),
                point_margin=round(point, 1),
                margin_mean=round(float(margins.mean()), 2),
                margin_sd=round(float(margins.std(ddof=0)), 2),
                margin_p05=round(float(np.percentile(margins, 5)), 1),
                margin_p50=round(float(np.percentile(margins, 50)), 1),
                margin_p95=round(float(np.percentile(margins, 95)), 1),
                frac_le_2pt=round(100 * float(np.mean(margins <= 2.0)), 1),
                frac_le_4pt=round(100 * float(np.mean(margins <= 4.0)), 1),
                reps=reps, boot=boot)


# ---------------------------------------------------------------------------
def main():
    out = {}

    figures = ["S0", "S4", "S6", "S7", "A4", "S8", "S9", "BERT-base", "mBERT"]
    out["per_class"] = {}
    for t in figures:
        if t not in RUNS:
            continue
        rows, cm = per_class(t)
        out["per_class"][t] = dict(acc_pooled=round(acc(t, "pooled"), 2),
                                   acc_mean=round(acc(t, "mean"), 2),
                                   rows=rows, confusion=cm)
    out["gold_dist"] = [int(x) for x in np.bincount(GOLD)]

    tmpl = pair_table("S7", "S0")
    out["template_s7_s0"] = tmpl
    out["conjunction"] = [rule_characteristics(tmpl, d)
                          for d in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    tmpl2 = pair_table("A4", "S0")
    out["template_a4_s0"] = tmpl2
    out["conjunction2"] = [rule_characteristics(tmpl2, d, seed=SEED + 1)
                           for d in (0.0, 2.0, 4.0)]

    out["frontier"] = frontier_stability()
    out["margins"] = [margin_properties("S7", "BERT-base"),
                      margin_properties("A4", "BERT-base"),
                      margin_properties("S6", "mBERT")]

    with open(os.path.join(RES, "review_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    L = ["# Revision statistics (round 75)\n",
         "Derived entirely from the stored predictions in `results/*.json`;"
         " nothing was retrained.\n"]

    L.append("\n## A. Per-class metrics, LD-100K (196 test paragraphs)\n")
    L.append("| system | acc (pooled) | acc (per-seed mean) | macro-F1 |"
             " min class recall | max class recall |")
    L.append("|---|---|---|---|---|---|")
    for t, d in out["per_class"].items():
        recs = [r["recall"] for r in d["rows"]]
        mf1 = float(np.mean([r["f1"] for r in d["rows"]]))
        L.append(f"| {t} | {d['acc_pooled']} | {d['acc_mean']} | {mf1:.2f} |"
                 f" {min(recs)} | {max(recs)} |")
    L.append("\n### Per-class detail\n")
    for t, d in out["per_class"].items():
        L.append(f"\n**{t}** (n per class from the test split)\n")
        L.append("| class | n | precision | recall | F1 |")
        L.append("|---|---|---|---|---|")
        for r in d["rows"]:
            L.append(f"| {r['cls']} | {r['n']} | {r['precision']} |"
                     f" {r['recall']} | {r['f1']} |")

    L.append("\n## B. The dual-test conjunction rule\n")
    L.append(f"Template (S7 vs S0, pooled): `{out['template_s7_s0']}`\n")
    L.append("| true advantage (pts) | realised | discordant (expected) |"
             " bootstrap rejects | McNemar rejects | **conjunction** | either |")
    L.append("|---|---|---|---|---|---|---|")
    for r in out["conjunction"]:
        L.append(f"| {r['delta_pts']} | {r['realised_delta']} |"
                 f" {r['discordant_expected']} | {r['bootstrap']}% |"
                 f" {r['mcnemar']}% | **{r['conjunction']}%** |"
                 f" {r['either']}% |")
    L.append(f"\nSecond template (A4 vs S0, pooled): `{out['template_a4_s0']}`\n")
    L.append("| true advantage (pts) | realised | bootstrap | McNemar |"
             " **conjunction** |")
    L.append("|---|---|---|---|---|")
    for r in out["conjunction2"]:
        L.append(f"| {r['delta_pts']} | {r['realised_delta']} |"
                 f" {r['bootstrap']}% | {r['mcnemar']}% |"
                 f" **{r['conjunction']}%** |")

    fr = out["frontier"]
    L.append("\n## C. Figure 4 non-dominated set under the declared noise floor\n")
    L.append(f"SI jitter +/-{100*fr['si_jitter']:.0f}% multiplicative,"
             f" {fr['reps']} draws.\n")
    L.append("| system | acc | SI | acc sd | non-dom. (SI only) |"
             " non-dom. (SI + acc) |")
    L.append("|---|---|---|---|---|---|")
    for t, p in fr["points"].items():
        L.append(f"| {t} | {p['acc']} | {p['si']} | {p['acc_sd']} |"
                 f" {fr['inclusion_si_only'][t]}% |"
                 f" {fr['inclusion_si_and_acc'][t]}% |")
    L.append(f"\nBaseline non-dominated set: {', '.join(fr['baseline_frontier'])}\n")
    L.append(f"Floor-aware frontier (a system is beaten only by one at least"
             f" {100*fr['si_jitter']:.0f}% faster *and* at least as accurate):"
             f" {', '.join(fr['floor_aware_frontier'])}\n")

    L.append("\n## D. The Table 7 'smallest equivalent margin' column\n")
    L.append("| pair | observed delta | point margin | mean | sd | p05 | p50 |"
             " p95 | share <= 2 pt | share <= 4 pt |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in out["margins"]:
        L.append(f"| {m['pair']} | {m['observed_delta']} | {m['point_margin']}"
                 f" | {m['margin_mean']} | {m['margin_sd']} | {m['margin_p05']}"
                 f" | {m['margin_p50']} | {m['margin_p95']} |"
                 f" {m['frac_le_2pt']}% | {m['frac_le_4pt']}% |")

    with open(os.path.join(RES, "review_stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote results/review_stats.json and results/review_stats.md")
    print(json.dumps(out["conjunction"], ensure_ascii=False))
    print(json.dumps(out["margins"], ensure_ascii=False))
    print(json.dumps(out["frontier"]["inclusion_si_only"], ensure_ascii=False))


if __name__ == "__main__":
    main()
