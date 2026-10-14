"""Pick the deployment threshold theta from the measured sweep.

Why this is a separate step
---------------------------
The early-exit threshold is a *deployment-time* knob, so it should be chosen
from data rather than fixed by hand.  The rule used here is the standard
accuracy-preserving one:

    theta* = the threshold whose measured speed index is the largest, among
             those whose accuracy is not significantly worse than the
             always-deep control.

"Not significantly worse" is enforced in two ways at once, because the McNemar
test alone is weak at n = 196 paragraphs:

    1. McNemar exact p vs the theta = -1 control must be >= ALPHA (0.05), i.e.
       the paired disagreement is not evidence of a real accuracy loss; and
    2. the absolute accuracy drop must be no more than TOL (1 percentage
       point), which catches a systematic-but-underpowered degradation.

The objective is the *speed index*, not theta itself.  Because theta is not a
monotone proxy for speed -- the accuracy/latency curve flattens out near the
top of the range, where several thresholds exit on almost every paragraph --
taking the largest qualifying theta would pick a point that is no faster than
its neighbours.  Maximising the measured speed index instead selects the point
that is actually fastest.

If no threshold satisfies both criteria, the selector falls back to the largest
speed index that satisfies (1) alone, and finally to the always-deep control.
The chosen criterion is written to disk so the paper can state exactly how
theta* was obtained.

Outputs results/theta_choice.json.
"""
from __future__ import annotations

import argparse
import json
import os

import common as C

ALPHA = 0.05        # McNemar significance level
TOL = 0.01          # tolerated absolute accuracy drop (1 percentage point)


def load(name):
    p = os.path.join(C.RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pick(rows, deep, alpha, tol, by_speed=True):
    cand = [r for r in rows if r["theta"] >= 0]
    if by_speed:
        cand = sorted(cand, key=lambda r: -r["speed_index"])
    else:
        cand = sorted(cand, key=lambda r: -r["theta"])
    for r in cand:
        if (r["mcnemar_p_vs_noexit"] >= alpha
                and r["ptc"] >= deep["ptc"] - tol):
            return r, "mcnemar and absolute-drop"
    for r in cand:
        if r["mcnemar_p_vs_noexit"] >= alpha:
            return r, "mcnemar only"
    return deep, "always deep (no threshold qualified)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="theta_sweep.json")
    ap.add_argument("--out", default="theta_choice.json")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    sweep = load(args.inp)
    if not sweep:
        raise SystemExit(f"results/{args.inp} not found; run theta_sweep.py first")

    rows = sweep["rows"]
    deep = next(r for r in rows if r["theta"] < 0)
    chosen, criterion = pick(rows, deep, args.alpha, args.tol)

    out = {"system": sweep.get("system"), "seed": sweep.get("seed"),
           "alpha": args.alpha, "tol": args.tol,
           "criterion": criterion,
           "rule": "maximise the measured speed index among thresholds whose "
                   "accuracy is not significantly worse than the always-deep "
                   "control",
           "theta": float(chosen["theta"]),
           "ptc": chosen["ptc"], "exit_rate": chosen["exit_rate"],
           "speed_index": chosen["speed_index"],
           "passages_per_s": chosen["passages_per_s"],
           "ptc_deep_control": deep["ptc"],
           "speed_index_deep_control": deep["speed_index"],
           "mcnemar_p_vs_noexit": chosen["mcnemar_p_vs_noexit"]}
    C.json_dump(out, os.path.join(C.RESULTS, args.out))

    print(f"theta* = {out['theta']}  ({criterion})")
    print(f"  PTC {out['ptc'] * 100:.1f}% vs {out['ptc_deep_control'] * 100:.1f}% "
          f"always-deep   exit {out['exit_rate'] * 100:.1f}%")
    print(f"  speed index {out['speed_index']:.1f}% "
          f"(always-deep control {out['speed_index_deep_control']:.1f}%)")
    print(f"  McNemar p = {out['mcnemar_p_vs_noexit']:.4f}")
    print(f"wrote results/{args.out}")


if __name__ == "__main__":
    main()
