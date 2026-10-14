"""Average encoder depth and analytic compute of the deployed exit (Section 4.6).

Motivation
----------
Every efficiency number in the paper is a *wall-clock* ratio measured on one
6 GB laptop GPU, and Section 4.3 says plainly that absolute latencies do not
transfer between devices.  The one quantity that does transfer is the fraction
of the encoder the early exit removes, and that fraction is fixed by the exit
rate alone: the student has four layers and the auxiliary head sits after layer
2 (Section 3.5), so a paragraph that exits runs two layers and a paragraph that
does not runs four.  This script turns the measured exit rate into

  * the mean number of encoder layers executed per paragraph;
  * the resulting fraction of the encoder's *layer-compute*;
  * the layer-only bound on the speed-up (L / mean_depth);
  * an analytic FLOP count for one paragraph of each corpus.

The exit rate used is the registered operating point theta = 0.5 on the 196
test paragraphs, from head_agreement.json (LD-100K) and
head_agreement_organic.json (OD-100K) -- the same measurement the paper reports
in Section 4.11.

Outputs results/efficiency_depth.json and results/efficiency_depth.md.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common as C

# Architecture of the deployed student (TinyBERT_General_4L_312D).
LAYERS = 4              # encoder layers (M in the paper)
EXIT_LAYER = 2          # auxiliary head after layer 2 (m* in the paper)
HIDDEN = 312
INTERMEDIATE = 1200

SEEDS = (7, 13, 37, 42, 97)

# corpus -> (head-agreement file, padded sequence length, paragraphs scored)
CORPORA = {
    "LD-100K": ("head_agreement.json", 96, 416),
    "OD-100K": ("head_agreement_organic.json", 112, 416),
}


def load(name):
    with open(os.path.join(C.RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def layer_flops(seq_len):
    """Forward FLOPs of one encoder layer for one token, standard BERT count
    (each multiply-add = 2 FLOPs):

        QKV projections   3 d^2
        attention QK^T / AV   2 S d
        output projection     d^2
        feed-forward      2 d d_ff
    """
    macs = 3 * HIDDEN ** 2 + 2 * seq_len * HIDDEN + HIDDEN ** 2 \
        + 2 * HIDDEN * INTERMEDIATE
    return 2 * macs


def analyse(label, fname, seq_len, n_par):
    rep = load(fname)
    rows = [r for r in rep["rows"] if r["tag"] == "S7"]
    if not rows:
        raise SystemExit(f"no deployed-system rows in {fname}")
    seeds = [r["seed"] for r in rows]

    # A paragraph exits after EXIT_LAYER layers or, failing the threshold,
    # runs the full LAYERS; depth is therefore a two-valued function of the
    # exit decision, whose mean follows exactly from the exit rate.
    depth = [EXIT_LAYER * r["exit_rate"] + LAYERS * (1.0 - r["exit_rate"])
             for r in rows]
    rate = [r["exit_rate"] for r in rows]

    dm = float(np.mean(depth))
    fl = layer_flops(seq_len)
    return {
        "corpus": label,
        "source": fname,
        "seq_len": seq_len,
        "paragraphs": n_par,
        "seeds": seeds,
        "exit_rate_mean": round(float(np.mean(rate)), 4),
        "exit_rate_min": round(float(np.min(rate)), 4),
        "exit_rate_max": round(float(np.max(rate)), 4),
        "mean_depth_mean": round(dm, 4),
        "mean_depth_min": round(float(np.min(depth)), 4),
        "mean_depth_max": round(float(np.max(depth)), 4),
        "layer_compute_fraction": round(dm / LAYERS, 4),
        "layer_only_speedup_bound": round(LAYERS / dm, 3),
        "flops": {
            "per_token_per_layer": int(fl),
            "per_paragraph_full_encoder": int(seq_len * LAYERS * fl),
            "per_paragraph_deployed": int(round(seq_len * dm * fl)),
        },
    }


def main():
    out = {
        "note": ("Mean encoder depth and analytic compute of the deployed exit, "
                 "derived from the registered theta = 0.5 exit rate on the 196 "
                 "test paragraphs (Section 4.6)."),
        "model": {"layers": LAYERS, "exit_layer_index": EXIT_LAYER,
                  "hidden": HIDDEN, "intermediate": INTERMEDIATE},
        "corpora": {},
    }
    for label, (fname, seq_len, n_par) in CORPORA.items():
        out["corpora"][label] = analyse(label, fname, seq_len, n_par)

    C.json_dump(out, os.path.join(C.RESULTS, "efficiency_depth.json"))

    L = ["# Mean encoder depth and analytic compute (deployed exit, theta = 0.5)\n",
         f"Student: {LAYERS} encoder layers, auxiliary head after layer "
         f"{EXIT_LAYER}, so a paragraph runs {EXIT_LAYER} layers if it exits and "
         f"{LAYERS} otherwise. Computed over the 196 test paragraphs, five seeds.\n",
         "| corpus | exit rate (%) | mean depth (layers) | layer-compute (%) | layer-only bound | FLOPs/paragraph (full) | (deployed) |",
         "|---|---|---|---|---|---|---|"]
    for label, d in out["corpora"].items():
        f = d["flops"]
        L.append(
            f"| {label} | {d['exit_rate_mean'] * 100:.1f} | {d['mean_depth_mean']:.2f} | "
            f"{d['layer_compute_fraction'] * 100:.1f} | {d['layer_only_speedup_bound']:.2f}x | "
            f"{f['per_paragraph_full_encoder'] / 1e9:.3f} GFLOP | "
            f"{f['per_paragraph_deployed'] / 1e9:.3f} GFLOP |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(C.RESULTS, "efficiency_depth.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
