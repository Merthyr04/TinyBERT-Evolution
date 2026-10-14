# -*- coding: utf-8 -*-
"""Geometric guard for the five manuscript figures.

Round 23 established that eyeballing a figure is not a check: three visual
passes failed to notice that (i) the framework boxes were shorter than the text
they had to hold and (ii) patches sitting at x ~ 0 had their strokes shaved off
by the axes clip path.  The fix was to stop looking and start measuring.  This
script keeps that measurement, so it can be re-run after any edit.

It calls the real figure functions with plt.close neutralised and save_fig
intercepted, so the figures are built exactly as published, then asserts:

  A. every FancyBboxPatch lies inside its axes
     -> catches a box drawn off the plotting area / a stroke shaved at x ~ 0;
  B. no two FancyBboxPatches on the same axes overlap
     -> catches a label that has grown into its neighbour;
  C. every Text whose centre falls inside a box lies wholly inside that box
     -> this is the round-23 defect: text taller than the box holding it;
  D. no visible Text is clipped away (clip_on with the bbox outside the axes)
     -> catches text the reader simply cannot see.

Text that is deliberately placed outside the axes (panel titles, side notes)
has clip_on=False and is reported as INFO, not as a fault.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_figures as mf                                     # noqa: E402
from matplotlib.patches import FancyBboxPatch                 # noqa: E402

CAPTURED = {}
INFO = []


def _capture(fig, name):
    CAPTURED[name] = fig


mf.save_fig = _capture
mf.plt.close = lambda *a, **k: None       # keep the real Agg canvas alive


def build_all():
    summary = mf.load("summary.json", {}) or {}
    bench_json = mf.load("benchmark.json", {}) or {}
    bench = bench_json.get("systems", {})
    abl = mf.load("ablation.json", {}) or {}
    theta_json = (mf.load("theta_sweep_S7.json", {})
                  or mf.load("theta_sweep.json", {}) or {})
    theta_choice = (mf.load("theta_choice_S7.json", {})
                    or mf.load("theta_choice.json", {}) or {})
    mf.fig1(summary)
    mf.fig2(summary, theta_json, bench, theta_choice)
    mf.fig3(summary, bench)
    mf.fig4(summary, bench)
    mf.fig5(abl, theta_json, summary, theta_choice, bench_json)


def _inside(inner, outer, tol=0.0):
    return (inner.x0 >= outer.x0 - tol and inner.x1 <= outer.x1 + tol
            and inner.y0 >= outer.y0 - tol and inner.y1 <= outer.y1 + tol)


def _overlap(a, b):
    return (min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0))


def main():
    build_all()
    faults = 0
    for name in sorted(CAPTURED):
        fig = CAPTURED[name]
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        nbox = ntext = 0
        for ai, ax in enumerate(fig.axes):
            axbb = ax.get_window_extent(renderer=r)
            boxes = [p for p in ax.patches
                     if isinstance(p, FancyBboxPatch) and p.get_visible()]
            bbs = [p.get_window_extent(renderer=r) for p in boxes]
            nbox += len(boxes)

            for bi, pb in enumerate(bbs):                      # A
                if not _inside(pb, axbb, tol=0.5):
                    print("  FAULT %s ax%d box%d outside axes: %s vs %s"
                          % (name, ai, bi, pb.extents, axbb.extents))
                    faults += 1
            for i in range(len(bbs)):                           # B
                for j in range(i + 1, len(bbs)):
                    ox, oy = _overlap(bbs[i], bbs[j])
                    if ox > 1.0 and oy > 1.0:
                        print("  FAULT %s ax%d boxes %d/%d overlap %.1fx%.1f px"
                              % (name, ai, i, j, ox, oy))
                        faults += 1

            texts = [t for t in ax.texts if t.get_visible()
                     and t.get_text().strip()]
            ntext += len(texts)
            for t in texts:
                tb = t.get_window_extent(renderer=r)
                if tb.width == 0 or tb.height == 0:
                    continue
                cx, cy = (tb.x0 + tb.x1) / 2.0, (tb.y0 + tb.y1) / 2.0
                host = None
                for pb in bbs:                                  # C
                    if (pb.x0 <= cx <= pb.x1 and pb.y0 <= cy <= pb.y1):
                        host = pb
                        break
                if host is not None:
                    if not _inside(tb, host):
                        print("  FAULT %s ax%d text %r escapes its box "
                              "(text %s vs box %s)"
                              % (name, ai,
                                 t.get_text()[:34].replace("\n", "|"),
                                 tb.extents, host.extents))
                        faults += 1
                elif not _inside(tb, axbb, tol=1.0):            # D
                    if t.get_clip_on():
                        print("  FAULT %s ax%d text %r clipped away (clip_on, "
                              "%s vs axes %s)"
                              % (name, ai,
                                 t.get_text()[:34].replace("\n", "|"),
                                 tb.extents, axbb.extents))
                        faults += 1
                    else:
                        INFO.append("%s ax%d %r sits outside the axes on "
                                    "purpose (clip_on=False)"
                                    % (name, ai,
                                       t.get_text()[:34].replace("\n", "|")))
        print("%-16s axes=%d boxes=%d texts=%d"
              % (name, len(fig.axes), nbox, ntext))
    if INFO:
        print("\nfor information (not faults):")
        for line in INFO:
            print("  info  " + line)
    print("\ngeometry faults:", faults)
    return 1 if faults else 0


if __name__ == "__main__":
    sys.exit(main())
