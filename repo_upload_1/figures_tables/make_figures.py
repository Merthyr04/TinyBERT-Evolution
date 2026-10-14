# -*- coding: utf-8 -*-
"""Draw the five figures of 'TinyBERT Under a Fixed Inference Budget:
Augmentation, Distillation, and a Dual Head that Halves the Encoder'.

Figures 1 and 2 are schematics (drawn from constants that describe the method).
Figures 3, 4 and 5 are *data driven*:  they load the JSON files written by the
experiment pipeline, so re-running the experiments and then this script always
yields figures that match Section 4.

    tinybert_run/results/summary.json      accuracy / MLM / speed aggregates
    tinybert_run/results/benchmark.json    round-robin throughput, speed index
    tinybert_run/results/ablation.json     contrasts vs S0 and component ablations
    tinybert_run/results/theta_sweep.json  early-exit threshold sweep
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
RES = os.path.join(os.path.dirname(HERE), "tinybert_run", "results")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#404040",
    "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    # The Computer Journal asks for EPS.  Type 42 embeds the font as TrueType;
    # the matplotlib default (Type 3) is a bitmap-flavoured format that some
    # production houses reject.  The PDF is embedded by md2tex.py and gets the
    # same treatment so the two vector outputs stay interchangeable.
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
    "ps.useafm": False,
})


def save_fig(fig, name):
    """Write one figure as PDF + EPS + 300 dpi PNG and report its print size.

    .pdf -> embedded by md2tex.py        (vector, used for the journal PDF)
    .eps -> what The Computer Journal asks for ("EPS only")
    .png -> embedded by md2docx.py       (300 dpi, for the Word manuscript)

    Every artist must stay fully opaque: the PostScript backend rasterises any
    path whose alpha is < 1, which would silently turn part of the EPS into a
    bitmap.  That is why the legend and annotation frames use framealpha=1.0.
    """
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".eps"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    print("  %-16s %6.1f x %6.1f mm" % (name, bb.width * 25.4,
                                        bb.height * 25.4))

INK = "#1f2937"
BLUE = "#2563eb"
BLUE_L = "#dbeafe"
TEAL = "#0f766e"
TEAL_L = "#ccfbf1"
AMBER = "#b45309"
AMBER_L = "#fef3c7"
GREY = "#e5e7eb"
GREY_T = "#6b7280"
RED = "#b91c1c"
PURPLE = "#7c3aed"


# --------------------------------------------------------------- helpers
def load(name, default=None):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print(f"  (missing {name} -- falling back to placeholder values)")
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def box(ax, x, y, w, h, text, fc, ec=None, fs=9, bold=False, tc=INK, r=0.02,
        ls="-"):
    # clip_on=False: the axes is invisible, so a stroke that straddles the axes
    # edge must not be shaved by the clip path (that is what made the left-hand
    # borders vanish in Figure 1 before the columns were re-laid out).
    # `ls' carries the Figure 1 category that colour alone cannot: the printed
    # version is grayscale (online-only colour), where the light blue and light
    # amber fills are indistinguishable -- dashed borders mark the components
    # that change the deployed graph.
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={r}",
        linewidth=1.1, edgecolor=ec or INK, facecolor=fc, linestyle=ls,
        zorder=3, clip_on=False))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=4, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color=INK, style="-|>", lw=1.2, ls="-", z=2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=10,
        linewidth=lw, color=color, linestyle=ls, zorder=z,
        shrinkA=0, shrinkB=0))


SYS_SHORT = {
    "S0": "S0\nTinyBERT\n(base)", "S1": "S1\n+DA", "S2": "S2\n+SD",
    "S3": "S3\n+DH", "S4": "S4\n+DA+SD", "S5": "S5\n+DD",
    "S6": "S6\n+DD+DA", "S7": "S7\n+DD+DA+DH",
    "BERT-base": "BERT-base", "mBERT": "mBERT",
}
# Compact tick labels: the id on the first line, the mechanism on the second,
# so that neighbouring bars never collide however long the configuration is.
SYS_TICK = {
    "S0": "S0\nbase", "S1": "S1\n+DA", "S2": "S2\n+SD", "S3": "S3\n+DH",
    "S4": "S4\n+DA+SD", "S5": "S5\n+DD", "S6": "S6\n+DD+DA",
    "S7": "S7\n+DD+DA+DH", "BERT-base": "BERT\nbase", "mBERT": "mBERT",
}
_COL = {"S0": GREY_T, "S1": AMBER, "S2": BLUE, "S3": AMBER, "S4": BLUE,
        "S5": BLUE, "S6": TEAL, "S7": TEAL, "BERT-base": RED, "mBERT": PURPLE}


def collect():
    """-> (labels, ptc, mlm, si, colors, teachers) from the experiment JSONs."""
    summary = load("summary.json", {}) or {}
    bench = (load("benchmark.json", {}) or {}).get("systems", {})
    rows = summary.get("student_systems", [])
    refs = summary.get("reference_systems", [])
    tags = [r["tag"] for r in rows] + [r["tag"] for r in refs]
    ptc = [r["acc_mean"] * 100 if r["acc_mean"] is not None else None
           for r in rows] + \
          [r["acc_mean"] * 100 if r["acc_mean"] is not None else None for r in refs]
    mlm = [r["mlm1_mean"] * 100 if r["mlm1_mean"] is not None else None
           for r in rows] + [None] * len(refs)
    si = [bench.get(t, {}).get("speed_index", r.get("speed_index"))
          for t, r in zip([r["tag"] for r in rows], rows)] + \
         [bench.get(t, {}).get("speed_index", r.get("speed_index"))
          for t, r in zip([r["tag"] for r in refs], refs)]
    return tags, ptc, mlm, si


# The speed index of every system whose deployed graph is byte-identical to the
# baseline.  Their spread is the reproducible noise floor of the round-robin
# measurement (Section 4.6), so it is drawn as a band rather than as a line.
NOISE_LO, NOISE_HI = 100.0, 110.8


def noise_band(ax, lo=NOISE_LO, hi=NOISE_HI, label=True, fill=True):
    """Shaded +-15% noise floor.  `fill=False' draws only the two dashed
    guides (plus the legend entry) and leaves the shading to the caller --
    used by fig4, where the shading has to live on the bar axes, not on the
    twin axis that always draws on top of the bars."""
    if fill:
        ax.axhspan(lo, hi, color="#eef2f7", zorder=0)
    ax.axhline(lo, color=GREY_T, ls="--", lw=0.9, zorder=1,
               label="noise floor of the benchmark" if label else None)
    ax.axhline(hi, color=GREY_T, ls="--", lw=0.9, zorder=1)


# ---------------------------------------------------------------- Figure 1
def fig1(summary):
    rows = {r["tag"]: r for r in summary.get("student_systems", [])}
    ea = rows.get("S7", {})
    dep = rows.get("S7", {})
    ea_acc = f"{ea.get('acc_mean', 0) * 100:.1f}%" if ea.get("acc_mean") else "n/a"
    dep_si = dep.get("speed_index")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 1.02, "Three orthogonal enhancements "
            "around one frozen student", ha="center", fontsize=10.5,
            fontweight="bold")

    # left column: box centres are kept far enough from x=0 that no border is
    # clipped by the axes, with >=8 px of side margin around every label
    box(ax, 0.085, 0.865, 0.155, 0.170, "LD-100K\ndocument\n100,116 chars",
        "#f3f4f6", fs=6.5)
    box(ax, 0.090, 0.705, 0.170, 0.135, "Teacher $T_1$\nBERT-base  109M",
        BLUE_L, ec=BLUE, fs=7)
    box(ax, 0.090, 0.560, 0.170, 0.135, "Teacher $T_2$\nmBERT  178M",
        BLUE_L, ec=BLUE, fs=7)
    box(ax, 0.095, 0.350, 0.170, 0.210,
        "DA  data aug.\nmask-fill\nwindow jitter\nsent. dropout",
        BLUE_L, ec=BLUE, fs=7)

    box(ax, 0.31, 0.920, 0.26, 0.145,
        "Stage A  (shared)\nreleased TinyBERT4", GREY, fs=8)

    box(ax, 0.30, 0.575, 0.195, 0.30,
        "Student\nTinyBERT4\n\nM = 4, d' = 312\n14.4M params",
        TEAL_L, ec=TEAL, fs=8, bold=True)

    box(ax, 0.595, 0.865, 0.21, 0.185,
        "SD  single-teacher\nTransformer\ndistillation  Eq. (3)", BLUE_L, ec=BLUE, fs=7.5)
    box(ax, 0.595, 0.615, 0.21, 0.225,
        "DD  dual-teacher\nentropy-gated mixture\nof two intermediate\n"
        "models  Eqs. (4)-(6)", BLUE_L, ec=BLUE, fs=7.5)
    box(ax, 0.595, 0.375, 0.21, 0.190,
        "DH  dual head\n+ consistency loss\nEq. (7)", AMBER_L, ec=AMBER, fs=8,
        ls="--")
    box(ax, 0.595, 0.175, 0.21, 0.190,
        "early exit\nentropy threshold\n$\\theta^* = 0.5$", AMBER_L, ec=AMBER,
        fs=8, ls="--")

    box(ax, 0.865, 0.780, 0.17, 0.150, f"TinyBERT-EA\n{ea_acc}  PTC", TEAL_L,
        ec=TEAL, fs=8.5, bold=True)
    si_txt = "SI = n/a" if dep_si is None else f"SI = {dep_si:.0f}%"
    box(ax, 0.865, 0.400, 0.17, 0.200, f"deployed\nstudent\n{si_txt}", TEAL_L,
        ec=TEAL, fs=8.5, bold=True)

    arrow(ax, 0.167, 0.900, 0.177, 0.918)
    arrow(ax, 0.30, 0.845, 0.30, 0.727)
    arrow(ax, 0.177, 0.705, 0.203, 0.670)
    arrow(ax, 0.177, 0.560, 0.203, 0.600)
    arrow(ax, 0.183, 0.350, 0.203, 0.500, color=BLUE)
    arrow(ax, 0.398, 0.865, 0.483, 0.868)
    arrow(ax, 0.398, 0.680, 0.483, 0.645)
    arrow(ax, 0.398, 0.500, 0.483, 0.400)
    arrow(ax, 0.702, 0.865, 0.776, 0.800, color=BLUE)
    arrow(ax, 0.702, 0.375, 0.776, 0.395, color=AMBER, ls="--")
    arrow(ax, 0.702, 0.175, 0.776, 0.345, color=AMBER, ls="--")
    arrow(ax, 0.938, 0.700, 0.938, 0.505)

    ax.text(0.938, 0.590, "inference", ha="center", va="center", fontsize=7.5,
            color=GREY_T, style="italic", bbox=dict(fc="white", ec="none", pad=1.0))

    ax.add_patch(FancyBboxPatch((0.035, 0.038), 0.022, 0.032,
                                boxstyle="round,pad=0.002", facecolor=BLUE_L,
                                edgecolor=BLUE, lw=1.0, clip_on=False))
    ax.text(0.068, 0.054, "training-time only: augmentation, single- and "
            "dual-teacher distillation  ->  deployed graph unchanged, "
            "speed index inside the noise band",
            fontsize=7.8, va="center", color=INK)
    ax.add_patch(FancyBboxPatch((0.035, 0.004), 0.022, 0.030,
                                boxstyle="round,pad=0.002", facecolor=AMBER_L,
                                edgecolor=AMBER, lw=1.0, linestyle="--",
                                clip_on=False))
    ax.text(0.068, 0.020, "changes the deployed graph (dashed border): dual "
            "head + early exit  ->  speed index rises above 100%",
            fontsize=7.8, va="center", color=INK)

    save_fig(fig, "fig1_framework")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def fig2(summary, theta_json, bench=None, theta_choice=None):
    bench = bench or {}
    b7 = bench.get("S7", {})
    fallback = {r["tag"]: r for r in summary.get("student_systems", [])}.get("S7", {})
    si = b7.get("speed_index", fallback.get("speed_index"))
    ex = b7.get("exit_rate", fallback.get("exit_rate"))
    th = (theta_choice or {}).get("theta")
    th_txt = "$\\theta^* = 0.5$" if th is None else f"$\\theta^* = {th:g}$"
    note = "the exit rate and speed index are measured in Section 4.6"
    if si is not None and ex is not None:
        note = (f"{ex * 100:.1f}% of the paragraphs exit at {th_txt}"
                f"   ->   speed index {si:.0f}%")

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.0))

    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 1.015, "(a)  dual-model (two-teacher) distillation",
            ha="center", va="top", fontsize=9.5, fontweight="bold")

    box(ax, 0.20, 0.875, 0.28, 0.115, "$T_1$  BERT-base\n(109M)", BLUE_L, ec=BLUE, fs=8)
    box(ax, 0.20, 0.655, 0.28, 0.115, "$T_2$  mBERT\n(178M)", BLUE_L, ec=BLUE, fs=8)
    box(ax, 0.20, 0.410, 0.28, 0.135,
        "gate  $\\alpha=\\sigma(w)$\none learned\nscalar", AMBER_L, ec=AMBER, fs=7.5)
    ax.text(0.20, 0.225, "no extra inference cost:\nthe second teacher is dropped\nafter training",
            ha="center", va="center", fontsize=7.3, color=GREY_T, style="italic")

    box(ax, 0.68, 0.800, 0.44, 0.140,
        "gated soft target\n$\\alpha p^{(1)} + (1-\\alpha)\\,p^{(2)}$", "#f3f4f6", fs=7.5)
    box(ax, 0.68, 0.470, 0.36, 0.115, "Student\nTinyBERT4", TEAL_L, ec=TEAL,
        fs=8.5, bold=True)
    box(ax, 0.68, 0.205, 0.44, 0.275,
        "$L_{attn}$   attention\n\n$L_{hid}$   hidden states\n\n"
        "$L_{pred}$   prediction", TEAL_L, ec=TEAL, fs=7.5)

    arrow(ax, 0.34, 0.875, 0.455, 0.845, color=BLUE)
    arrow(ax, 0.455, 0.845, 0.458, 0.845, color=BLUE)
    arrow(ax, 0.34, 0.655, 0.455, 0.800, color=BLUE)
    arrow(ax, 0.34, 0.410, 0.415, 0.410, color=AMBER, ls="--")
    arrow(ax, 0.415, 0.410, 0.415, 0.790, color=AMBER, ls="--")
    arrow(ax, 0.415, 0.790, 0.458, 0.790, color=AMBER, ls="--")
    arrow(ax, 0.68, 0.728, 0.68, 0.530)
    arrow(ax, 0.68, 0.412, 0.68, 0.348)
    ax.text(0.735, 0.628, "gated", fontsize=7.3, color=INK)
    ax.text(0.735, 0.375, "supervision", fontsize=7.3, color=INK)

    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 1.015, "(b)  dual-head student with early exit",
            ha="center", va="top", fontsize=9.5, fontweight="bold")

    ys = [0.850, 0.720, 0.590, 0.460, 0.330]
    names = ["embeddings", "layer 1", "layer 2", "layer 3", "layer 4"]
    for y, n in zip(ys, names):
        box(ax, 0.24, y, 0.26, 0.078, n, TEAL_L, ec=TEAL, fs=8)
    for i in range(4):
        arrow(ax, 0.24, ys[i] - 0.040, 0.24, ys[i + 1] + 0.040, color=GREY_T, lw=1.0)

    box(ax, 0.775, 0.460, 0.36, 0.105, "auxiliary head\n$\\phi_{aux}$",
        AMBER_L, ec=AMBER, fs=7.5)
    box(ax, 0.775, 0.185, 0.36, 0.105, "main head\n$\\phi_{main}$",
        AMBER_L, ec=AMBER, fs=7.5)
    arrow(ax, 0.37, 0.590, 0.594, 0.505, color=AMBER)
    arrow(ax, 0.37, 0.330, 0.594, 0.220, color=AMBER)

    ax.text(0.775, 0.660, "exit if  $H(p_{aux})/\\log C < \\theta$", ha="center",
            va="center", fontsize=7.5, color=RED)
    arrow(ax, 0.775, 0.628, 0.775, 0.516, color=RED, lw=1.1)
    arrow(ax, 0.962, 0.408, 0.962, 0.238, color=GREY_T, lw=1.0, ls=":")
    arrow(ax, 0.962, 0.238, 0.962, 0.408, color=GREY_T, lw=1.0, ls=":")

    ax.text(0.660, 0.038, "consistency  $KL(p_{main}\\,\\|\\,p_{aux})$",
            ha="center", va="center", fontsize=7.5, color=GREY_T, style="italic")
    ax.text(0.50, 0.955, note, ha="center", va="center", fontsize=7.6, color=INK)

    save_fig(fig, "fig2_dual")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig3(summary, bench):
    rows = summary.get("student_systems", [])
    refs = summary.get("reference_systems", [])
    tags = [r["tag"] for r in rows]
    lab = [SYS_TICK.get(t, t) for t in tags]
    ptc = [r["acc_mean"] * 100 for r in rows]
    mlm = [r["mlm1_mean"] * 100 if r["mlm1_mean"] is not None else 0 for r in rows]
    col = [_COL.get(t, GREY_T) for t in tags]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))

    ax = axes[0]
    x = list(range(len(lab)))
    ax.bar([i - 0.21 for i in x], ptc, width=0.41, color=col, edgecolor=INK,
           linewidth=0.7, label="PTC  (topic accuracy)")
    ax.bar([i + 0.21 for i in x], mlm, width=0.41, color="white", edgecolor=col,
           hatch="///", linewidth=1.0, label="MLM  (masked-token top-1)")
    handles = []
    for r in refs:
        if r["acc_mean"] is None:
            continue
        c = RED if r["tag"] == "BERT-base" else PURPLE
        ls = "--" if r["tag"] == "BERT-base" else ":"
        h = ax.axhline(r["acc_mean"] * 100, color=c, ls=ls, lw=1.1,
                       label=f"{r['tag']} reference {r['acc_mean'] * 100:.1f}")
        handles.append(h)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("\n", " ") for l in lab], fontsize=6.6,
                       rotation=45, ha="right")
    lo = max(0, min(min(ptc), min(m for m in mlm if m)) - 15)
    hi = max(max(ptc), max((r["acc_mean"] or 0) * 100 for r in refs)) + 12
    ax.set_ylim(lo, hi)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("(a)  accuracy on LD-100K", fontsize=9.5, fontweight="bold")
    # Author request: the legend goes BELOW the axes.  The bars reach the top
    # of the plot, so every inside placement covers them, and the rotated tick
    # labels have to be cleared as well.  save_fig() saves with
    # bbox_inches="tight", which grows the canvas downwards to keep it.
    ax.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.44),
              ncol=2, frameon=True, framealpha=1.0, edgecolor="#d1d5db",
              columnspacing=1.2, handlelength=1.8, handletextpad=0.6)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6); ax.set_axisbelow(True)

    ax = axes[1]
    allr = rows + refs
    lab2 = [SYS_TICK.get(r["tag"], r["tag"]).replace("\n", " ") for r in allr]
    si2 = [bench.get(r["tag"], {}).get("speed_index", r.get("speed_index"))
           for r in allr]
    si2 = [0 if v is None else v for v in si2]
    cols = []
    for r, v in zip(allr, si2):
        if v >= 99.5 and v <= 100.5:
            cols.append(GREY_T)
        elif v > 100:
            cols.append(TEAL)
        else:
            cols.append(RED)
    b = ax.bar(list(range(len(si2))), si2, width=0.62, color=cols,
               edgecolor=INK, linewidth=0.7)
    ax.axhline(100, color=INK, ls="-", lw=0.9)
    noise_band(ax, label=True)
    top = max(si2) * 1.28 if si2 else 130
    for r, v in zip(b, si2):
        txt = f"{v:.1f}" if 0 < v < 50 else f"{v:.0f}"
        ax.text(r.get_x() + r.get_width() / 2, v + top * 0.02, txt,
                ha="center", va="bottom",
                fontsize=7.0, rotation=90 if len(si2) > 9 else 0,
                fontweight="bold" if abs(v - 100) > 0.5 else "normal")
    ax.set_xticks(list(range(len(si2)))); ax.set_xticklabels(lab2, fontsize=6.6,
                                                             rotation=90)
    ax.set_ylim(0, top)
    ax.set_ylabel("speed index (%)")
    # Same rule as panel (a): the legend goes BELOW the axes.  The inside
    # placement sat on the label of the tallest bar and hid its number.
    ax.legend(fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.46),
              frameon=True, framealpha=1.0, edgecolor="#d1d5db")
    ax.set_title("(b)  speed index, baseline TinyBERT = 100%", fontsize=9.5,
                 fontweight="bold")
    ax.grid(axis="y", color="#e5e7eb", lw=0.6); ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, "fig3_acc_speed")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4
def fig4(summary, bench):
    """The deployment plane: accuracy against speed index.

    Round 54 drew this plane for the ten systems of the main comparison and
    computed a two-member frontier.  Round 55 draws it for *every*
    configuration whose speed index was measured in the same round-robin,
    including the ablated baseline A0 and the component ablations A1-A5.  The
    scoping of the older figure was load-bearing: A2 (93.2 pooled) and A4
    (95.9) sit above the deployed system on the accuracy axis at the baseline's
    speed, so "two of ten" described the figure, not the data.  The frontier
    here is computed from the plotted points, and the caption states how many
    of how many configurations it contains, so that the claim cannot outrun
    the measurement.

    Accuracy is the per-seed mean of Table 4 (the estimator a practitioner
    deploying one run would see); A0 shares S0's deployed graph, so its speed
    index is 100% by construction rather than measured.
    """
    bm = load("baseline_matrix.json", {}) or {}
    stats = bm.get("systems", {})
    if not stats:
        raise SystemExit("results/baseline_matrix.json is missing: run "
                         "tinybert_run/baseline_matrix.py before the figures")

    order = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7",
             "A0", "A1", "A2", "A3", "A4", "A5", "BERT-base", "mBERT"]
    tags, acc, si = [], [], []
    for t in order:
        if t not in stats:
            continue
        v = bench.get(t, {}).get("speed_index")
        if v is None:
            if t != "A0":
                continue
            # A0 = S0 - L_mlm: the same deployed graph, one head fewer.
            v = 100.0
        tags.append(t)
        acc.append(stats[t]["acc_mean"] * 100.0)
        si.append(float(v))

    # Non-dominated set on (speed index, accuracy), both maximised.  A system
    # is dominated when some other system is at least as fast AND at least as
    # accurate, with one of the two strict.  Derived from the JSONs so that a
    # re-run of the experiments cannot leave the claim stale.
    frontier = []
    for i in range(len(tags)):
        beaten = False
        for j in range(len(tags)):
            if j == i:
                continue
            if (si[j] >= si[i] - 1e-9 and acc[j] >= acc[i] - 1e-9
                    and (si[j] > si[i] + 1e-9 or acc[j] > acc[i] + 1e-9)):
                beaten = True
                break
        if not beaten:
            frontier.append(i)
    frontier.sort(key=lambda i: si[i])

    fig, ax = plt.subplots(figsize=(7.7, 4.4))
    ax.set_xlim(-8, 218)
    ax.set_ylim(87.2, 98.4)
    ax.grid(color="#eef2f7", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Speed index (% of the baseline TinyBERT, 100%)")
    ax.set_ylabel("PTC accuracy (%)")

    # The band the trained-but-graph-identical systems scatter inside: the
    # protocol resolves no difference below about 15% in the speed index, so
    # the fine structure of this band is not interpretable (Section 4.9(v)).
    ax.axvspan(100.0, 111.0, color="#f2f5f9", ec="none", zorder=0)

    ax.plot([si[i] for i in frontier], [acc[i] for i in frontier],
            color=RED, lw=1.6, ls="--", zorder=2,
            label="Pareto frontier (%d of %d configurations)"
                  % (len(frontier), len(tags)))

    # The always-deep control of each dual-head system: the same weights with
    # layers 3-4 kept.  Its horizontal distance from the filled marker is the
    # genuine layer saving, not the cost of the layer-loop implementation.
    for tag in ("S3", "S7"):
        if tag not in tags:
            continue
        i = tags.index(tag)
        v = bench.get(tag, {}).get("speed_index_no_exit")
        if v is None:
            continue
        ax.plot([v], [acc[i]], marker="o", ls="none", mfc="white", mec=INK,
                mew=1.3, ms=6.0, zorder=4,
                label="same weights, exit disabled ($\\theta=-1$)"
                      if tag == "S7" else None)
        ax.annotate("", xy=(si[i], acc[i]), xytext=(v, acc[i]),
                    arrowprops=dict(arrowstyle="<->", color=GREY_T, lw=1.0,
                                    shrinkA=1.5, shrinkB=3.0))
    i7 = tags.index("S7")
    v7 = bench.get("S7", {}).get("speed_index_no_exit")
    if v7:
        ax.text((v7 + si[i7]) / 2.0, acc[i7] - 1.5,
                "%.2fx" % (si[i7] / v7), ha="center", va="top",
                fontsize=7.2, color=INK,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none"))

    # Marker shape carries the category, so the figure survives the grayscale
    # print: circle = one of the eight systems of the main comparison,
    # square = ablated baseline / component ablation, diamond = reference
    # encoder.  Colour is decorative.
    for i, tag in enumerate(tags):
        if tag in ("BERT-base", "mBERT"):
            m, s, fc, ec = "D", 78, _COL.get(tag, GREY_T), INK
        elif tag.startswith("A"):
            m, s, fc, ec = "s", 44, "white", INK
        else:
            m, s, fc, ec = "o", 58, _COL.get(tag, GREY_T), INK
        ax.scatter([si[i]], [acc[i]], s=s, marker=m, facecolor=fc,
                   edgecolor=ec, linewidth=0.9, zorder=5,
                   label="main comparison (8 systems)" if tag == "S0" else
                         ("ablated baseline / ablations" if tag == "A0" else
                          ("reference encoder" if tag == "BERT-base" else None)))
    ax.scatter([si[i7]], [acc[i7]], s=250, marker="o", facecolors="none",
               edgecolors=RED, linewidths=1.7, zorder=4)

    # A label per system, off to the side of its own marker so that no label
    # sits on the frontier line.
    lab = {
        "S0": (0, -17, "center", "S0  88.8"),
        "A0": (-11, 0, "right", "A0  91.2"),
        "S4": (0, 15, "center", "S4  93.9"),
        "A4": (0, 15, "center", "A4  95.9"),
        "S3": (0, -17, "center", "S3  89.9"),
        "A5": (-10, -11, "right", "A5  91.3"),
        "S7": (0, 27, "center", "S7  TinyBERT-EA  94.0"),
        "BERT-base": (11, 0, "left", "BERT-base  96.2"),
        "mBERT": (11, -2, "left", "mBERT  93.3"),
    }
    for tag, (dx, dy, ha, txt) in lab.items():
        if tag not in tags:
            continue
        i = tags.index(tag)
        ax.annotate(txt, (si[i], acc[i]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va="center", fontsize=7.2,
                    fontweight="bold" if tag == "S7" else "normal",
                    color=RED if tag == "S7" else INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                              edgecolor="none"))
    # The remaining systems sit inside the noise band, within two points of one
    # another; one label for the group keeps their markers from being buried.
    ax.text(118.0, 88.4,
            "S1, S2, S5, S6, A1-A3\n89.9-93.9% at 102-111%",
            ha="left", va="center", fontsize=7.2, color=INK, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none"))

    ax.set_title("The deployment plane: three of sixteen configurations are "
                 "non-dominated", fontsize=9.8, fontweight="bold")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, fontsize=6.8, loc="lower left", frameon=True,
              framealpha=1.0, edgecolor="#d1d5db")
    fig.tight_layout()
    save_fig(fig, "fig4_scatter")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5
def fig5(abl, theta_json, summary, theta_choice=None, bench=None):
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.3))
    bench = bench or {}

    # (a) component ablations
    ax = axes[0]
    items = abl.get("ablation", [])
    per = abl.get("per_system", {})
    if items:
        lab = [it["tag"] for it in items]
        delta = [it["delta"] * 100 for it in items]
        lo = [it["ci95"][0] * 100 for it in items]
        hi = [it["ci95"][1] * 100 for it in items]
        y = list(range(len(lab)))[::-1]
        colors = [RED if d < 0 else TEAL for d in delta]
        # the half-width of the plotted range is needed before labelling, so
        # that each value label can be parked just outside its CI whisker
        span = max(abs(min(lo)), abs(max(hi))) * 2.0 + 1
        ax.barh(y, delta, color=colors, edgecolor=INK, linewidth=0.7, height=0.6)
        for yi, d, a, b in zip(y, delta, lo, hi):
            ax.plot([b, a], [yi, yi], color=INK, lw=1.0)
            # Author request: the numbers must stand out.  They sit beyond the
            # whisker end (never on the bar) and on an opaque white chip, so
            # neither the bar colour nor the CI line can wash them out.
            xt = (max(d, b) + span * 0.035) if d >= 0 else (min(d, a) - span * 0.035)
            ax.text(xt, yi, f"{d:+.1f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=7.8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.95))
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8)
        ax.set_xlabel("$\\Delta$ PTC (pp) versus the parent model")
        ax.set_xlim(-span, span)
    else:
        ax.text(0.5, 0.5, "ablation.json not found", ha="center", va="center")
        ax.axis("off")
    ax.set_title("(a)  removing one ingredient", fontsize=9.5, fontweight="bold")
    ax.grid(axis="x", color="#eef2f7", lw=0.7); ax.set_axisbelow(True)

    # (b) early-exit threshold
    # The curve measured inside benchmark.py's round-robin is the one quoted in
    # Table 6(b), because its speed indices live on the same scale as Table 5.
    # The standalone sweep (theta_sweep_S7.json) has its own, ~1.6x faster
    # baseline and is used only as a fallback.
    ax = axes[1]
    rows = [r for r in (bench.get("theta_curve") or []) if r["theta"] >= 0]
    src = "benchmark round-robin"
    if not rows:
        rows = [r for r in (theta_json or {}).get("rows", []) if r["theta"] >= 0]
        src = "standalone sweep"
    chosen = (theta_choice or {}).get("theta")
    if rows:
        th = [r["theta"] for r in rows]
        ptc = [r["ptc"] * 100 for r in rows]
        si = [r["speed_index"] for r in rows]
        ln1, = ax.plot(th, ptc, marker="s", color=TEAL, lw=1.7, markersize=5,
                       label="PTC accuracy (%)  [left]")
        ax.set_ylabel("PTC accuracy (%)", color=TEAL)
        ax.tick_params(axis="y", labelcolor=TEAL)
        ax2 = ax.twinx()
        ln2, = ax2.plot(th, si, marker="^", color=AMBER, lw=1.7, markersize=5,
                        label="speed index (%)  [right]")
        ax2.set_ylabel("speed index (%)", color=AMBER)
        ax2.tick_params(axis="y", labelcolor=AMBER)
        ax.axvline(chosen, color=GREY_T, ls="--", lw=1.0)
        ax.annotate(f"adopted $\\theta^*={chosen:g}$",
                    xy=(chosen, ax.get_ylim()[0]), xytext=(6, 4),
                    textcoords="offset points", fontsize=6.9, color=INK,
                    ha="left", va="bottom")
        ax.set_xlabel("early-exit entropy threshold $\\theta$")
        # Author request: below the axes -- the inside placement sat on the
        # lower-left corner and covered the first points of the sweep.
        ax.legend(handles=[ln1, ln2], fontsize=6.9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True,
                  framealpha=1.0, edgecolor="#d1d5db",
                  columnspacing=1.2, handlelength=1.8)
    else:
        ax.text(0.5, 0.5, "no theta curve found", ha="center",
                va="center")
        ax.axis("off")
    ax.set_title(f"(b)  early-exit threshold  ({src})", fontsize=9.5,
                 fontweight="bold")
    ax.grid(color="#eef2f7", lw=0.7); ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, "fig5_ablation")
    plt.close(fig)


if __name__ == "__main__":
    summary = load("summary.json", {}) or {}
    bench_json = load("benchmark.json", {}) or {}
    bench = bench_json.get("systems", {})
    abl = load("ablation.json", {}) or {}
    # The threshold sweep of the deployed system.  S7 is the deployment
    # configuration; older runs wrote a single un-suffixed file.
    theta_json = (load("theta_sweep_S7.json", {})
                  or load("theta_sweep.json", {}) or {})
    theta_choice = (load("theta_choice_S7.json", {})
                    or load("theta_choice.json", {}) or {})

    fig1(summary)
    fig2(summary, theta_json, bench, theta_choice)
    fig3(summary, bench)
    fig4(summary, bench)
    fig5(abl, theta_json, summary, theta_choice, bench_json)
    for f in sorted(os.listdir(FIG)):
        print("wrote", f, os.path.getsize(os.path.join(FIG, f)))
