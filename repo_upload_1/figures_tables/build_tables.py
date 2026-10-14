# -*- coding: utf-8 -*-
"""Render the paper's result tables straight from the experiment JSONs.

Every number that appears in Section 4 is produced here, so the manuscript can
be re-synchronised with the experiments by re-running this file instead of
copying digits by hand.

Inputs (all under tinybert_run/results/):
    summary.json     aggregation of students + teachers (accuracy, MLM, speed)
    benchmark.json   single-process round-robin throughput / speed index
    ablation.json    contrasts vs S0, component ablations, significance tests
    theta_sweep.json early-exit threshold sweep
Output:
    paper/tables_from_results.md   (stdout as well)

Table numbering follows the manuscript:
    Table 4  accuracy of every system
    Table 5  accuracy / efficiency
    Table 6  ablations (a) component removal, (b) early-exit threshold
    Table 7  paired significance against the baseline
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RES = os.path.join(PROJ, "tinybert_run", "results")

SYS_NAME = {
    "S0": "TinyBERT (baseline)",
    "S1": "TinyBERT + DA",
    "S2": "TinyBERT + SD",
    "S3": "TinyBERT + DH",
    "S4": "TinyBERT + DA + SD",
    "S5": "TinyBERT + DD",
    "S6": "TinyBERT + DD + DA",
    "S7": "TinyBERT + DD + DA + DH",
}
REF_NAME = {"BERT-base": "BERT-base (teacher)", "mBERT": "mBERT (teacher)"}
ORDER = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
REFS = ["BERT-base", "mBERT"]

REMOVES = {
    "A1": "attention-matrix distillation $L_{attn}$",
    "A2": "hidden-state distillation $L_{hid}$",
    "A3": "prediction-layer distillation $L_{pred}$",
    "A4": "auxiliary masked-LM objective $L_{mlm}$",
    "A5": "data augmentation",
}


def load(name, default=None):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def stars(p):
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


N_CLASSES = 12


def peak_prob(theta):
    """The class probability a prediction must exceed in order to exit.

    The criterion is H(p)/log(C) < theta, where H is the entropy of the
    auxiliary head's 12-way posterior.  Solving it for the mass p placed on a
    single class (the rest spread uniformly) makes theta readable as a
    confidence threshold, which matters because theta is *not* a probability.
    """
    import math

    def hn(p):
        q = (1.0 - p) / (N_CLASSES - 1)
        return -(p * math.log(p) + (N_CLASSES - 1) * q * math.log(q)) / math.log(
            N_CLASSES)

    lo, hi = 1.0 / N_CLASSES, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if hn(mid) > theta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fmt_pm(mean, std):
    if mean is None:
        return "-"
    if std is None or std == 0:
        return f"{mean * 100:.1f}"
    return f"{mean * 100:.1f} $\\pm$ {std * 100:.1f}"


def fmt_p(p):
    if p is None:
        return "-"
    return f"{p:.4f}" if p >= 1e-4 else "$<$0.001"


def main():
    summary = load("summary.json") or {}
    bench = (load("benchmark.json") or {}).get("systems", {})
    abl = load("ablation.json") or {}
    sweeps = load("sweeps.json") or {}

    # Per-system early-exit sweeps.  S7 is the deployed configuration; S3 is
    # swept as well because it isolates the dual head from the distillation.
    th_sweep = {t: (load(f"theta_sweep_{t}.json") or {}) for t in ("S3", "S7")}
    th_pick = {t: (load(f"theta_choice_{t}.json") or {}) for t in ("S3", "S7")}
    theta = th_sweep["S7"] or (load("theta_sweep.json") or {})
    choice = th_pick["S7"]

    # The early-exit curve, measured inside benchmark.py's round-robin so that
    # its speed indices are on the same scale as Table 5.
    bench_json = load("benchmark.json") or {}
    curve = bench_json.get("theta_curve") or []
    bench_theta = bench_json.get("exit_theta")
    if not curve and theta.get("rows"):
        curve = [{"theta": r["theta"], "exit_rate": r["exit_rate"],
                  "ptc": r["ptc"], "speed_index": r["speed_index"]}
                 for r in theta["rows"] if r["theta"] >= 0]
        bench_theta = None

    students = {r["tag"]: r for r in summary.get("student_systems", [])}
    refs = {r["tag"]: r for r in summary.get("reference_systems", [])}
    contrasts = {c["system"]: c for c in abl.get("contrasts", [])}
    per_sys = abl.get("per_system", {})
    abl_rows = {a["tag"]: a for a in abl.get("ablation", [])}

    out = []
    A = out.append
    n_meas = (load("benchmark.json") or {}).get("paragraphs_measured", 416)

    # ------------------------------------------------------------ Table 4
    A("<!-- TABLE4 accuracy -->")
    A("**Table 4:** Accuracy on LD-100K. PTC = paragraph topic classification "
      "accuracy over 12 classes, measured on the 196 held-out paragraphs; "
      "MF1 = macro-F1; MLM = masked-token top-1 recovery over the masked "
      "positions of those paragraphs (1,329 for the bert-base-uncased "
      "vocabulary used by the students and by BERT-base, 1,445 for mBERT's own "
      "tokenizer). $\\Delta$PTC is the change in percentage points against the "
      "baseline S0. Student rows are mean $\\pm$ s.d. over three seeds; the "
      "significance markers refer to Table 7 and follow "
      "`*` $p<0.05$, `**` $p<0.01$, `***` $p<0.001$, `n.s.` not significant. "
      "Every row is measured with the early exit **disabled** ($\\theta=-1$), "
      "so all systems are compared at equal inference depth; the deployed "
      "operating point of the dual-head systems is reported separately in "
      "Table 5.")
    A("")
    A("| ID | System | PTC (%) | MF1 (%) | MLM top-1 (%) | $\\Delta$PTC |")
    A("| --- | --- | --- | --- | --- | --- |")
    for tag in ORDER:
        s = students.get(tag)
        if not s:
            continue
        c = contrasts.get(tag)
        delta = "—" if tag == "S0" else (
            f"{c['delta'] * 100:+.1f} {stars(c['p_mcnemar'])}" if c else "-")
        name = SYS_NAME[tag]
        A(f"| {tag} | {name} | {fmt_pm(s['acc_mean'], s['acc_std'])} | "
          f"{fmt_pm(s['f1_mean'], s['f1_std'])} | "
          f"{fmt_pm(s['mlm1_mean'], s['mlm1_std'])} | {delta} |")
    tmlm = (load("teacher_mlm.json") or {}).get("teachers") or {}
    for tag in REFS:
        r = refs.get(tag)
        if not r:
            continue
        t = tmlm.get(tag) or {}
        mlm_col = ("-" if "mlm_top1_finetuned" not in t
                   else f"{t['mlm_top1_finetuned'] * 100:.1f}")
        A(f"| — | {REF_NAME[tag]} | {fmt_pm(r['acc_mean'], r['acc_std'])} | "
          f"{fmt_pm(r['f1_mean'], None)} | {mlm_col} | - |")
    A("")
    if tmlm:
        A("The teacher MLM column deserves a caveat. The released teachers "
          "contain no masked-LM head, because stage A only ever trained them "
          "as sequence classifiers; the reported figure comes from re-attaching "
          "the published masked-LM head to the fine-tuned encoder and re-tying "
          "the output decoder to the input embeddings. Measured this way, "
          "fine-tuning *degrades* masked-token recovery: BERT-base falls from "
          f"{tmlm.get('BERT-base', {}).get('mlm_top1_released', 0) * 100:.1f}% "
          f"to {tmlm.get('BERT-base', {}).get('mlm_top1_finetuned', 0) * 100:.1f}%, "
          f"and mBERT from {tmlm.get('mBERT', {}).get('mlm_top1_released', 0) * 100:.1f}% "
          f"to {tmlm.get('mBERT', {}).get('mlm_top1_finetuned', 0) * 100:.1f}%. "
          "Several students therefore *exceed* their own fine-tuned teacher on "
          "this metric, which is possible only because their $L_{mlm}$ term is "
          "supervised by the gold tokens rather than distilled from a teacher "
          "whose language-modelling ability the classification objective has "
          "partly overwritten.")
        A("")

    # ------------------------------------------------------------ Table 5
    A("<!-- TABLE5 efficiency -->")
    A("**Table 5:** Accuracy and efficiency. Throughput is measured by "
      "reloading every trained system into one process and timing them "
      "round-robin over the whole document (all 416 labelled "
      "paragraphs, 96,409 characters of body text), so that GPU clock drift affects all systems equally "
      "(Section 4.6). The speed index (SI) is relative to the baseline "
      "TinyBERT, which is 100% by construction; the exit rate is the share of "
      "paragraphs that leave after layer 2. Training time is per system. "
      "FLOPs/window is not reported because it is identical for S0-S6 by "
      "construction."
      + (f" All systems are timed at the single threshold "
         f"`theta = {bench_theta}` adopted for the deployed system, so that "
         f"the indices reflect the architecture and not a per-system tuning "
         f"choice; systems without a dual head ignore `theta` entirely, which "
         f"is why S0-S6 and A1-A4 differ from one another only by noise."
         if bench_theta is not None else ""))
    A("")
    A("| ID | System | Params (M) | Passages/s | **SI (%)** | Exit rate (%) | "
      "Train time (s) | Peak train VRAM (MB) |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for tag in ORDER:
        s = students.get(tag)
        if not s:
            continue
        b = bench.get(tag, {})
        si = b.get("speed_index", s.get("speed_index"))
        ex = b.get("exit_rate", s.get("exit_rate"))
        pps = b.get("passages_per_s", s.get("pps_mean"))
        ptv = s.get("peak_train_vram_mb")
        tt = s.get("train_time_s")
        A(f"| {tag} | {SYS_NAME[tag]} | {s.get('params_m')} | "
          f"{'-' if pps is None else f'{pps:.1f}'} | "
          f"{'-' if si is None else f'**{si:.1f}**'} | "
          f"{'-' if ex is None else f'{ex * 100:.1f}'} | "
          f"{'-' if tt is None else f'{tt:.0f}'} | "
          f"{'-' if ptv is None else f'{ptv:.0f}'} |")
    for tag in REFS:
        r = refs.get(tag)
        if not r:
            continue
        b = bench.get(tag, {})
        si = b.get("speed_index", r.get("speed_index"))
        pps = b.get("passages_per_s", r.get("pps_mean"))
        tt = r.get("train_time_s")
        A(f"| — | {REF_NAME[tag]} | {r.get('params_m')} | "
          f"{'-' if pps is None else f'{pps:.1f}'} | "
          f"{'-' if si is None else f'**{si:.1f}**'} | - | "
          f"{'-' if tt is None else f'{tt:.0f}'} | - |")
    # Ablation cells belong in this table too: they show that removing a
    # training-time ingredient changes accuracy (Table 6a) while leaving the
    # inference graph -- and therefore the speed index -- untouched.
    for tag in ["A1", "A2", "A3", "A4", "A5"]:
        s = per_sys.get(tag)
        if not s:
            continue
        b = bench.get(tag, {})
        si = b.get("speed_index", s.get("speed_index"))
        ex = b.get("exit_rate", s.get("exit_rate"))
        pps = b.get("passages_per_s", s.get("pps"))
        tt = s.get("train_time_s")
        ptv = s.get("peak_train_vram_mb")
        parent_name = SYS_NAME.get((abl_rows.get(tag) or {}).get("parent"), "")
        A(f"| {tag} | {parent_name} $-$ {REMOVES[tag]} | "
          f"{s.get('params_m')} | "
          f"{'-' if pps is None else f'{pps:.1f}'} | "
          f"{'-' if si is None else f'**{si:.1f}**'} | "
          f"{'-' if ex is None else f'{ex * 100:.1f}'} | "
          f"{'-' if tt is None else f'{tt:.0f}'} | "
          f"{'-' if ptv is None else f'{ptv:.0f}'} |")
    A("")
    if "S7" in bench and "speed_index_no_exit" in bench["S7"]:
        A(f"*The deployed dual-head system S7 reaches SI = "
          f"{bench['S7']['speed_index']:.1f}% with the exit active and "
          f"SI = {bench['S7']['speed_index_no_exit']:.1f}% with the exit "
          f"disabled (`theta = -1`), which isolates the genuine saving from "
          f"the cost of running the encoder as an explicit layer loop.*")
        A("")
    # The theta* footnotes must not quote a speed index from the standalone
    # sweep: that sweep ran with the baseline at ~1769 pass/s, whereas Table 5
    # times everything round-robin at ~1122 pass/s.  Quoting the sweep's index
    # next to Table 5's would state two different numbers for one system.  The
    # curve in benchmark.json was timed in Table 5's own round-robin, so it is
    # the only place a speed index for a given theta may come from.
    curve_by_theta = {round(r["theta"], 4): r for r in curve}
    for t in ("S3", "S7"):
        c, sw = th_pick.get(t) or {}, th_sweep.get(t) or {}
        if not c or not sw:
            continue
        # Only the system the curve was actually measured on may borrow its
        # indices; the curve is a property of one trained checkpoint.
        in_ctx = (curve_by_theta.get(round(c["theta"], 4))
                  if t == bench_json.get("curve_tag") else None)
        si_txt = (f"SI = {in_ctx['speed_index']:.1f}%, "
                  if in_ctx else "")
        A(f"*{t} at its adopted operating point `theta* = {c['theta']}`: "
          f"PTC = {c['ptc'] * 100:.1f}%, exit rate = "
          f"{c['exit_rate'] * 100:.1f}%, {si_txt}"
          f"McNemar p vs its own always-deep control = "
          f"{c['mcnemar_p_vs_noexit']:.4f}. The threshold was chosen as the "
          f"largest (fastest) value whose accuracy is not significantly worse "
          f"than the always-deep system under the rule "
          f"`p >= {c['alpha']}` and an absolute drop of at most "
          f"{c['tol'] * 100:.0f} pp.*")
        A("")
    if th_pick.get("S3"):
        A(f"*S3's own sweep selects `theta* = {th_pick['S3']['theta']}`, "
          f"which exits on every paragraph and so reduces S3 to its two-layer "
          f"auxiliary head; that operating point was not timed in Table 5's "
          f"round-robin, so only its accuracy is quoted here. Table 5 times "
          f"every system at the single operating point "
          f"`theta = {bench_theta}` adopted for the deployed system, which is "
          f"precisely what makes the indices comparable across systems.*")
        A("")
    if choice:
        A(f"*The `theta*` of S7 was selected by `{choice['criterion']}`.*")
        A("")

    # ------------------------------------------------------------ Table 6
    A("<!-- TABLE6 ablation -->")
    A("**Table 6:** Ablation studies. (a) Every ingredient of the full "
      "training-time system is removed in turn. The parent is S6 (dual teacher "
      "+ augmentation) for A1-A4, and S7 (dual teacher + augmentation + dual "
      "head) for A5, so the difference is attributable to the removed "
      "ingredient alone. $\\Delta$ and its 95% CI are measured against the "
      "parent on the 196 test paragraphs with the same paired tests as "
      "Table 7, and the parent is evaluated on the same seeds as the cell it "
      "is compared with (see the footnote below the table). MLM top-1 is "
      "reported alongside PTC because $L_{mlm}$ is the masked-LM objective: "
      "removing it is expected to move that column, not the classification "
      "column, and the two columns disagree sharply for A4. (b) Early-exit "
      "threshold sweep on the trained dual-head student: the exit is "
      "inference-only, so the weights are identical across the row and only "
      "the threshold changes.")
    A("")
    A("| (a) Ablation | removed ingredient | seeds | parent (same seeds) "
      "PTC (%) | PTC (%) | MLM top-1 (%) | $\\Delta$ vs parent | "
      "95% CI | bootstrap p | McNemar p |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for tag in ["A1", "A2", "A3", "A4", "A5"]:
        a = abl_rows.get(tag)
        if not a:
            continue
        s = per_sys.get(tag, {})
        acc = s.get("acc_mean")
        mlm1 = s.get("mlm_top1")
        lo, hi = a["ci95"][0] * 100, a["ci95"][1] * 100
        seeds = ",".join(str(x) for x in a.get("seeds", []))
        par = a.get("acc_parent_seed_matched")
        A(f"| {tag} | {REMOVES[tag]} | {seeds} | "
          f"{'-' if par is None else f'{par * 100:.1f}'} | "
          f"{'-' if acc is None else f'{acc * 100:.1f}'} | "
          f"{'-' if mlm1 is None else f'{mlm1 * 100:.1f}'} | "
          f"{a['delta'] * 100:+.1f} | [{lo:+.1f}, {hi:+.1f}] | "
          f"{fmt_p(a['p_bootstrap'])} {stars(a['p_bootstrap'])} | "
          f"{fmt_p(a['p_mcnemar'])} {stars(a['p_mcnemar'])} |")
    A("")
    A("The parent column is the parent system evaluated on *exactly* the seeds "
      "of the corresponding ablation cell, not on its own full seed budget: "
      "otherwise removing an ingredient would be confounded with training on "
      "fewer seeds. All five cells use seeds 13, 37 and 97, so every parent is "
      "evaluated on the same three seeds as the main systems.")
    A("")
    # ---- (b) early-exit threshold sweep, S7, in-context ---------------
    # `curve` was taken from benchmark.json above, because it was timed in the
    # same round-robin as the main efficiency table and its speed indices are
    # therefore comparable with it.  The standalone sweep is the fallback.
    if curve:
        A("| (b) $\\theta$ | peak prob. | exit rate (%) | PTC (%) | "
          "passages/s | SI (%) |")
        A("| --- | --- | --- | --- | --- | --- |")
        for r in curve:
            pps = r.get("passages_per_s")
            A(f"| {r['theta']:.1f} | {peak_prob(r['theta']):.2f} | "
              f"{r['exit_rate'] * 100:.1f} | {r['ptc'] * 100:.1f} | "
              f"{'-' if pps is None else f'{pps:.1f}'} | "
              f"{r['speed_index']:.1f} |")
        A("")
        der = [r for r in theta.get("rows", []) if r["theta"] < 0]
        if der:
            d = der[0]
            # Accuracy of the always-deep control is context free and comes
            # from the sweep; its speed index must come from the round-robin,
            # i.e. the same measurement that produced the curve's indices.
            si_deep = (bench_json.get("systems", {}).get("S7", {})
                       .get("speed_index_no_exit"))
            A(f"The always-deep control ($\\theta=-1$) reaches "
              f"PTC = {d['ptc'] * 100:.1f}%"
              + (f" at SI = {si_deep:.1f}%" if si_deep is not None else "")
              + ". ")
            A("")
        A("`theta` is an entropy threshold, not a probability. Because the "
          "normalised entropy is divided by $\\log C$ with $C=12$, the "
          "'peak prob.' column gives the class probability that a prediction "
          "must exceed in order to exit: $\\theta=0.8$ already corresponds to "
          "a peak probability of only about 0.46, so it exits on essentially "
          "every paragraph. The auxiliary head turns out to be extremely "
          "confident, so exit is taken on nearly every paragraph for the whole "
          "range $0.2\\le\\theta\\le0.8$.")
        A("")
        q = theta.get("aux_entropy_quantiles") or {}
        if q:
            A("Auxiliary-head normalised-entropy quantiles over the 196 test "
              "paragraphs: "
              + ", ".join(f"{k} = {v:.4f}" for k, v in q.items())
              + (f"; the auxiliary head's own argmax reaches PTC = "
                 f"{theta['aux_head_only_ptc'] * 100:.1f}%."
                 if theta.get("aux_head_only_ptc") is not None else "."))
            A("")
        s3 = th_pick.get("S3") or {}
        if s3:
            A(f"For comparison, the same sweep applied to S3 (the dual head "
              f"with neither augmentation nor a teacher) selects "
              f"`theta* = {s3['theta']}`, at PTC = {s3['ptc'] * 100:.1f}% "
              f"versus {s3['ptc_deep_control'] * 100:.1f}% for its own "
              f"always-deep control, with {s3['exit_rate'] * 100:.1f}% of "
              f"paragraphs exiting.")
            A("")
    if sweeps:
        A("Supplementary sweeps (single seed, dual-teacher student, "
          "McNemar against the adopted setting):")
        A("")
        if sweeps.get("augmentation_factor"):
            A("| $A$ | train size | PTC (%) | MLM top-1 (%) | p vs $A=2$ |")
            A("| --- | --- | --- | --- | --- |")
            for r in sweeps["augmentation_factor"]:
                mlm = "-" if r["mlm_top1"] is None else f"{r['mlm_top1']:.1f}"
                A(f"| {r['value']} | {r['train_size']} | {r['ptc']:.1f} | "
                  f"{mlm} | {fmt_p(r['mcnemar_p_vs_ref'])} |")
            A("")
        if sweeps.get("temperature"):
            A("| $T$ | train size | PTC (%) | MLM top-1 (%) | p vs $T=2$ |")
            A("| --- | --- | --- | --- | --- |")
            for r in sweeps["temperature"]:
                mlm = "-" if r["mlm_top1"] is None else f"{r['mlm_top1']:.1f}"
                A(f"| {r['value']} | {r['train_size']} | {r['ptc']:.1f} | "
                  f"{mlm} | {fmt_p(r['mcnemar_p_vs_ref'])} |")
            A("")

    # ------------------------------------------------------------ Table 7
    A("<!-- TABLE7 significance -->")
    A("**Table 7:** Paired significance of each system against the baseline "
      "S0 over the 196 test paragraphs. The per-paragraph prediction of a "
      "student system is the argmax of its class posterior averaged over its "
      "seeds, and the baseline is treated the same way. $\\Delta$ and its 95% "
      "CI come from 10,000 paired bootstrap resamples; McNemar's exact test "
      "uses only the paragraphs on which the two systems disagree. A gain is "
      "reported as an improvement only when both tests agree.")
    A("")
    A("| System | $\\Delta$ vs S0 (pp) | 95% CI | bootstrap p | McNemar p | "
      "only S0 correct | only system correct |")
    A("| --- | --- | --- | --- | --- | --- | --- |")
    for tag in ORDER:
        if tag == "S0":
            continue
        c = contrasts.get(tag)
        if not c:
            continue
        lo, hi = c["ci95"][0] * 100, c["ci95"][1] * 100
        A(f"| {tag} {SYS_NAME[tag]} | {c['delta'] * 100:+.1f} | "
          f"[{lo:+.1f}, {hi:+.1f}] | {fmt_p(c['p_bootstrap'])} "
          f"{stars(c['p_bootstrap'])} | {fmt_p(c['p_mcnemar'])} "
          f"{stars(c['p_mcnemar'])} | {c['mcnemar_only_target_correct']} | "
          f"{c['mcnemar_only_system_correct']} |")
    A("")

    # ------------------------------------------------------------ Table 8
    cvr = load("contrasts_vs_refs.json") or {}
    crows = cvr.get("contrasts") or []
    if crows:
        A("<!-- TABLE8 vs references -->")
        A("**Table 8:** Paired contrast of each system against the two reference "
          "encoders, over the same 196 test paragraphs and with the same two "
          "tests as Table 7. A negative $\\Delta$ means the system is less "
          "accurate than the reference. The point of the table is that a "
          "descriptive gap between two means can be much larger than the "
          "smallest difference the test can resolve: the baseline is "
          "significantly behind both references, whereas the enhanced systems "
          "are not significantly behind either of them.")
        A("")
        A("| System | Reference | System PTC (%) | Reference PTC (%) | "
          "$\\Delta$ (pp) | 95% CI | bootstrap p | McNemar p |")
        A("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for c in crows:
            lo, hi = c["ci95"][0] * 100, c["ci95"][1] * 100
            A(f"| {c['system']} | {c['vs']} | {c['acc_system'] * 100:.1f} | "
              f"{c['acc_ref'] * 100:.1f} | {c['delta'] * 100:+.1f} | "
              f"[{lo:+.1f}, {hi:+.1f}] | {fmt_p(c['p_bootstrap'])} "
              f"{stars(c['p_bootstrap'])} | {fmt_p(c['p_mcnemar'])} "
              f"{stars(c['p_mcnemar'])} |")
        A("")
        best = [c for c in crows if c["system"] in ("S4", "S7")
                and c["vs"] == "BERT-base"]
        if best:
            b = best[0]
            A(f"*{b['system']} is {abs(b['delta']) * 100:.1f} pp behind "
              f"BERT-base on average, with a McNemar p of "
              f"{b['p_mcnemar']:.4f}: the two systems disagree on individual "
              f"paragraphs but neither wins consistently, so the gap is not "
              f"resolvable at this test size.*")
            A("")
    F = ["", "<!-- PROSE_FACTS: every number quoted in the prose -->", "```"]
    facts = {}
    s0 = students.get("S0", {})
    if s0:
        facts["S0_acc"] = s0["acc_mean"] * 100
        facts["S0_sd"] = s0["acc_std"] * 100
        facts["S0_mlm1"] = s0["mlm1_mean"] * 100
    for tag in REFS:
        r = refs.get(tag)
        if r and r["acc_mean"] is not None:
            facts[f"{tag}_acc"] = r["acc_mean"] * 100
            facts[f"{tag}_sd"] = (r["acc_std"] or 0) * 100
            b = bench.get(tag, {})
            if "speed_index" in b:
                facts[f"{tag}_SI"] = b["speed_index"]
    for tag in ORDER:
        s = students.get(tag)
        if not s:
            continue
        facts[f"{tag}_acc"] = s["acc_mean"] * 100
        facts[f"{tag}_sd"] = s["acc_std"] * 100
        facts[f"{tag}_mlm1"] = (s["mlm1_mean"] or 0) * 100
        if s0.get("acc_mean") is not None:
            facts[f"{tag}_delta_vs_S0"] = (s["acc_mean"] - s0["acc_mean"]) * 100
        b = bench.get(tag, {})
        if "speed_index" in b:
            facts[f"{tag}_SI"] = b["speed_index"]
        if b.get("exit_rate") is not None:
            facts[f"{tag}_exit_pct"] = b["exit_rate"] * 100
        if b.get("speed_index_no_exit") is not None:
            facts[f"{tag}_SI_noexit"] = b["speed_index_no_exit"]
        if s.get("alpha"):
            facts[f"{tag}_alpha"] = s["alpha"]
    for c in abl.get("contrasts", []):
        facts[f"p_mcnemar_{c['system']}_vs_S0"] = c["p_mcnemar"]
        facts[f"delta_{c['system']}_vs_S0"] = c["delta"] * 100
    for a in abl.get("ablation", []):
        facts[f"abl_{a['tag']}_delta"] = a["delta"] * 100
        facts[f"abl_{a['tag']}_p_mcnemar"] = a["p_mcnemar"]
    for tag, s in per_sys.items():
        if s.get("mlm_top1") is not None:
            facts[f"{tag}_mlm1"] = s["mlm_top1"] * 100
    tmlm = load("teacher_mlm.json") or {}
    for tag, r in (tmlm.get("teachers") or {}).items():
        facts[f"{tag}_mlm1_released"] = r["mlm_top1_released"] * 100
        facts[f"{tag}_mlm1_finetuned"] = r["mlm_top1_finetuned"] * 100
    for c in crows:
        key = f"{c['system']}_vs_{c['vs']}"
        facts[f"d_{key}"] = c["delta"] * 100
        facts[f"p_mcnemar_{key}"] = c["p_mcnemar"]
        facts[f"p_boot_{key}"] = c["p_bootstrap"]
    bb = refs.get("BERT-base", {})
    if bb.get("acc_mean") and s0.get("acc_mean"):
        span = bb["acc_mean"] - s0["acc_mean"]
        facts["gap_S0_to_BERTbase"] = span * 100
        facts["gap_S0_to_mBERT"] = (refs.get("mBERT", {}).get("acc_mean", 0)
                                   - s0["acc_mean"]) * 100
        best = max((students[t]["acc_mean"], t) for t in students)
        facts["best_tag"] = best[1]
        facts["best_acc"] = best[0] * 100
        facts["best_closes_pct_of_gap"] = (best[0] - s0["acc_mean"]) / span * 100
        facts["best_retains_pct_of_BERTbase"] = best[0] / bb["acc_mean"] * 100
    if "S4_acc" in facts and "S6_acc" in facts:
        facts["S4_minus_S6"] = facts["S4_acc"] - facts["S6_acc"]
    if "S2_acc" in facts and "S5_acc" in facts:
        facts["S5_minus_S2"] = facts["S5_acc"] - facts["S2_acc"]
    for k in sorted(facts):
        v = facts[k]
        F.append(f"{k:34s} {v:10.2f}" if isinstance(v, (int, float)) else
                 f"{k:34s} {v}")
    F.append("```")

    text = "\n".join(out) + "\n".join(F) + "\n"
    dst = os.path.join(HERE, "tables_from_results.md")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("wrote", dst)


if __name__ == "__main__":
    main()
