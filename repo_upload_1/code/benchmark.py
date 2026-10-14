"""A single-process, drift-corrected inference benchmark.

Why this exists
---------------
Measuring each system's throughput inside its own training process is not
sound: the sweep is spread over hours, the laptop GPU thermally throttles, and
the SM clock therefore drifts by tens of percent.  Two *architecturally
identical* models (S0 and S6 share the same deployed graph) were measured at
1366 and 1843 passages/s in separate processes -- a 35% gap that is pure clock
drift.

This script therefore reloads every trained system into **one** process and
measures them **round-robin**:

    for round in rounds:
        for unit in units:
            time(unit over the whole test split)

Every unit is exposed to the same clock trajectory, so the systematic component
of the drift cancels and only the residual per-round noise remains.  The
measurement is also much longer (20 rounds x 26 batches instead of 3 x 26),
which shrinks that residual as well: in a validation run the two identical
graphs came out within 3% of each other.

A dual-head system is entered twice, once with the exit active (theta = the
value passed with --theta, 0.8 by default) and once with it disabled
(theta = -1).  Both entries take part in the same round-robin, so the
difference between them is a clean measurement of the genuine early-exit
saving, separated from the cost of running the encoder as a manual layer loop.

The speed index is defined exactly as in the paper:

    speed_index(system) = 100 * throughput(system) / throughput(S0)

so the TinyBERT baseline is 100% by construction.

The two teachers are fed the *same* token-id tensors as the students.  This is
deliberate: BERT-base and mBERT have byte-identical architectures, and the
compute performed by the encoder is a function of the input shape, not of which
vocabulary produced the ids.  Holding the shapes fixed is what makes the
comparison apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer, BertForSequenceClassification

import common as C

WARMUP_ROUNDS = 3
ROUNDS = 20
NOEXIT_ROUNDS = 10          # separate, shorter window for the no-exit control


def load_json(name):
    p = os.path.join(C.RESULTS, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def tag_flags(*row_lists):
    """tag -> {dual_head, head_on_layer, seed} using the first row per tag."""
    flags = {}
    for rows in row_lists:
        for r in rows:
            t = r["tag"]
            if t in flags:
                continue
            flags[t] = {"dual_head": bool(r.get("dual_head")),
                        "head_on_layer": r.get("head_on_layer"),
                        "seed": r["seed"], "kind": "student"}
    return flags


def build_test_batches(tokenizer, texts, labels, batch_size, max_len=None):
    max_len = C.MAX_LEN if max_len is None else max_len
    enc = tokenizer(list(texts), padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt")
    y = torch.tensor(labels)
    out = []
    for i in range(0, len(y), batch_size):
        out.append({k: v[i:i + batch_size] for k, v in enc.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--out", default="benchmark.json")
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--results", nargs="+",
                    default=["students.json", "ablations.json"],
                    help="result json files (inside results/) defining the systems")
    ap.add_argument("--scope", dest="scope", default="all",
                    choices=["all", "test"],
                    help="paragraphs used for the timing pass: 'all' = every "
                         "labelled paragraph of the document (416), 'test' = "
                         "the 196 held-out paragraphs")
    ap.add_argument("--no-teachers", action="store_true")
    ap.add_argument("--theta", type=float, default=C.EXIT_THETA,
                    help="early-exit entropy threshold used for the dual-head "
                         "systems (default: common.EXIT_THETA)")
    ap.add_argument("--curve-tag", default=None,
                    help="also measure a theta curve for this dual-head system, "
                         "using the same round-robin as the main table so the "
                         "speed indices are directly comparable")
    ap.add_argument("--thetas", nargs="+", type=float,
                    default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    help="thresholds of the curve (only used with --curve-tag)")
    ap.add_argument("--data", default=None,
                    help="ptc-style jsonl corpus (default: data/ptc.jsonl = LD-100K)")
    ap.add_argument("--models-subdir", default="",
                    help="read checkpoints/teachers from models/<subdir>/ "
                         "(organic replication); default = models/")
    ap.add_argument("--max-len", type=int, default=None,
                    help="override C.MAX_LEN (organic corpus: 112)")
    ap.add_argument("--num-labels", type=int, default=None,
                    help="override C.NUM_LABELS (public benchmark: 4)")
    args = ap.parse_args()
    if args.max_len:
        C.MAX_LEN = args.max_len
    if args.num_labels:
        C.NUM_LABELS = args.num_labels

    device = C.get_device()
    print("device:", device, flush=True)
    records = C.load_ptc(args.data)
    models_root = (os.path.join(C.MODELS, args.models_subdir)
                   if args.models_subdir else C.MODELS)
    # Accuracy is always measured on the held-out 196 paragraphs.  Throughput
    # is measured on the whole document by default, which is what the paper
    # reports: all 416 labelled paragraphs, i.e. all 96,409 characters of
    # body text (the remaining characters of the 100,116-character file are
    # chapter headings and sub-headings, which carry no topic label).
    scope = records if args.scope == "all" else C.split_of(records, "test")
    texts = [r["text"] for r in scope]
    labels = [r["label_id"] for r in scope]
    tokenizer = AutoTokenizer.from_pretrained(C.resolve(C.TOKENIZER_EN))
    batches = [dict(b) for b in build_test_batches(tokenizer, texts, labels,
                                                   args.batch_size)]
    for b in batches:                     # keep them on the GPU, as in serving
        for k in b:
            b[k] = b[k].to(device)
    n_items = len(labels)
    print(f"timing pass: {n_items} paragraphs in {len(batches)} batches "
          f"(scope={args.scope})", flush=True)

    flags = tag_flags(*[load_json(n) for n in args.results])
    if args.tags:
        flags = {k: v for k, v in flags.items() if k in args.tags}
    if not flags:
        raise SystemExit("no systems found in " + ", ".join(args.results))

    ckpt_dir = os.path.join(models_root, "students")

    # ---- build every model ------------------------------------------------
    models, meta = {}, {}
    for tag in sorted(flags):
        path = os.path.join(ckpt_dir, f"{tag}_{flags[tag]['seed']}.pt")
        if not os.path.exists(path):
            print(f"  !! missing checkpoint for {tag}: {path}", flush=True)
            continue
        m = C.TinyStudent(head_layer=flags[tag].get("head_on_layer"))
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.to(device).eval()
        models[tag] = m
        meta[tag] = {"params_m": round(C.count_params(m) / 1e6, 2),
                     "dual_head": flags[tag]["dual_head"],
                     "head_on_layer": flags[tag].get("head_on_layer"),
                     "seed": flags[tag]["seed"], "kind": "student"}
        print(f"  loaded {tag} (params {meta[tag]['params_m']}M, "
              f"dual_head={meta[tag]['dual_head']}, "
              f"head_on_layer={meta[tag]['head_on_layer']})", flush=True)

    if not args.no_teachers:
        for tag in ("BERT-base", "mBERT"):
            path = os.path.join(models_root, tag)
            if not os.path.isdir(path):
                continue
            m = BertForSequenceClassification.from_pretrained(
                path, attn_implementation="eager").to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            models[tag] = m
            meta[tag] = {"params_m": round(C.count_params(m) / 1e6, 1),
                         "dual_head": False, "seed": None, "kind": "reference"}
            print(f"  loaded {tag} (params {meta[tag]['params_m']}M)", flush=True)

    # ---- measurement units ------------------------------------------------
    def full_fn(m):
        return lambda b: m(b["input_ids"], b["attention_mask"])

    def exit_fn(m, theta):
        return lambda b: m.forward_with_exit(b["input_ids"], b["attention_mask"],
                                             theta)

    def ref_fn(m):
        return lambda b: m(input_ids=b["input_ids"], attention_mask=b["attention_mask"])

    def fixed_depth_fn(m):
        return lambda b: m.forward_fixed_depth(b["input_ids"], b["attention_mask"])

    units = []                       # (label, tag, fn, is_noexit)
    for t in sorted(models):
        m, kind = models[t], meta[t]["kind"]
        if meta[t]["dual_head"]:
            units.append((t, t, exit_fn(m, args.theta), False))
            units.append((t + "@noexit", t, exit_fn(m, -1.0), True))
        elif meta[t].get("head_on_layer") is not None:
            units.append((t, t, fixed_depth_fn(m), False))
        elif kind == "student":
            units.append((t, t, full_fn(m), False))
        else:
            units.append((t, t, ref_fn(m), False))

    # Threshold curve for one dual-head system, entered into the *same*
    # round-robin as everything else.  The speed index of a threshold is
    # therefore measured against a baseline that saw the identical clock
    # trajectory, which is not true of a separately launched sweep.
    curve_labels = []
    if args.curve_tag:
        if args.curve_tag not in models:
            raise SystemExit(f"--curve-tag {args.curve_tag} is not among the "
                             f"loaded systems: {sorted(models)}")
        if not meta[args.curve_tag]["dual_head"]:
            raise SystemExit(f"--curve-tag {args.curve_tag} is not dual-head")
        for th in args.thetas:
            lbl = f"{args.curve_tag}@theta={th}"
            units.append((lbl, args.curve_tag, exit_fn(models[args.curve_tag], th),
                          False))
            curve_labels.append((lbl, th))

    # ---- clock warm-up ----------------------------------------------------
    print("warming up the GPU clock (3 untimed rounds) ...", flush=True)
    with torch.inference_mode():
        for _ in range(WARMUP_ROUNDS):
            for _, _, fn, _ in units:
                for b in batches:
                    fn(b)
    torch.cuda.synchronize()

    # ---- round-robin measurement -----------------------------------------
    elapsed = {lbl: 0.0 for lbl, _, _, _ in units}
    extra = {lbl: 0.0 for lbl, _, _, ne in units if ne}

    def timed_pass(lbl, fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for b in batches:
                fn(b)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    with torch.inference_mode():
        for rd in range(args.rounds):
            for lbl, _, fn, is_noexit in units:
                if is_noexit and rd >= NOEXIT_ROUNDS:
                    continue            # the control is a shorter window
                dt = timed_pass(lbl, fn)
                if is_noexit:
                    extra[lbl] += dt
                else:
                    elapsed[lbl] += dt
            if (rd + 1) % 5 == 0:
                print(f"  round {rd + 1}/{args.rounds}", flush=True)

    pps = {}
    for lbl, _, _, is_noexit in units:
        if is_noexit:
            pps[lbl] = n_items * NOEXIT_ROUNDS / extra[lbl]
        else:
            pps[lbl] = n_items * args.rounds / elapsed[lbl]

    base_tag = "S0" if "S0" in models else sorted(models)[0]
    base = pps[base_tag]

    # ---- early-exit rate (deterministic, no timing involved) -------------
    exit_rate = {}
    for t in sorted(models):
        if not meta[t]["dual_head"]:
            continue
        total = exited = 0
        with torch.inference_mode():
            for b in batches:
                _, conf = models[t].forward_with_exit(
                    b["input_ids"], b["attention_mask"], args.theta)
                total += conf.numel()
                exited += int(conf.sum().item())
        exit_rate[t] = exited / max(1, total)

    # ---- exit rate and accuracy as a function of theta -------------------
    # The accuracy is measured on the held-out paragraphs, the exit rate over
    # the same paragraphs the timing pass used.  Both are deterministic.
    curve = []
    if curve_labels:
        te = C.split_of(records, "test")
        te_y = torch.tensor([r["label_id"] for r in te], device=device)
        te_batches = []
        for i, b in enumerate(build_test_batches(
                tokenizer, [r["text"] for r in te],
                [r["label_id"] for r in te], args.batch_size)):
            b = dict(b)
            for k in b:
                b[k] = b[k].to(device)
            b["labels"] = te_y[i * args.batch_size:(i + 1) * args.batch_size]
            te_batches.append(b)
        model = models[args.curve_tag]
        for lbl, th in curve_labels:
            correct = exited = total = 0
            with torch.inference_mode():
                for b in te_batches:
                    logits, conf = model.forward_with_exit(
                        b["input_ids"], b["attention_mask"], th)
                    correct += int((logits.argmax(-1) == b["labels"]).sum().item())
                    exited += int(conf.sum().item())
                    total += conf.numel()
            curve.append({
                "theta": th,
                "exit_rate": round(exited / max(1, total), 4),
                "ptc": round(correct / len(te), 4),
                "passages_per_s": round(pps[lbl], 2),
                "speed_index": round(100.0 * pps[lbl] / base, 2),
            })
            print(f"  curve theta={th}: exit={exited / max(1, total) * 100:.1f}% "
                  f"PTC={correct / len(te) * 100:.1f}% "
                  f"SI={100.0 * pps[lbl] / base:.1f}%", flush=True)

    out = {"batch_size": args.batch_size, "scope": args.scope,
           "paragraphs_measured": n_items,
           "rounds": args.rounds, "noexit_rounds": NOEXIT_ROUNDS,
           "exit_theta": args.theta,
           "baseline_tag": base_tag, "baseline_pps": base,
           "curve_tag": args.curve_tag,
           "theta_curve": curve,
           "systems": {}}
    for lbl, tag, _, is_noexit in units:
        if is_noexit:
            continue
        rec = dict(meta[tag])
        rec["passages_per_s"] = round(pps[tag], 2)
        rec["speed_index"] = round(100.0 * pps[tag] / base, 2)
        rec["round_seconds_mean"] = round(elapsed[tag] / args.rounds, 4)
        if meta[tag]["dual_head"]:
            ne = pps[tag + "@noexit"]
            rec["passages_per_s_no_exit"] = round(ne, 2)
            rec["speed_index_no_exit"] = round(100.0 * ne / base, 2)
            rec["exit_rate"] = round(exit_rate.get(tag, 0.0), 4)
        out["systems"][tag] = rec

    C.json_dump(out, os.path.join(C.RESULTS, args.out))

    # ---- markdown --------------------------------------------------------
    lines = [f"Throughput over {n_items} paragraphs "
             f"(exit threshold theta = {args.theta}).", "",
             "| System | params (M) | passages/s | speed index | exit rate | "
             "speed index (no exit) |", "|---|---|---|---|---|---|"]
    for t in sorted(out["systems"]):
        r = out["systems"][t]
        ex = "-" if "exit_rate" not in r else f"{r['exit_rate'] * 100:.1f}"
        ne = ("-" if "speed_index_no_exit" not in r
              else f"{r['speed_index_no_exit']:.1f}")
        lines.append(f"| {t} | {r['params_m']} | {r['passages_per_s']:.1f} | "
                     f"{r['speed_index']:.1f} | {ex} | {ne} |")
    md = "\n".join(lines) + "\n"
    if curve:
        md += (f"\nThreshold curve of {args.curve_tag}, measured in the same "
               f"round-robin:\n\n"
               "| theta | exit rate (%) | PTC (%) | passages/s | speed index (%) |\n"
               "|---|---|---|---|---|\n")
        for r in curve:
            md += (f"| {r['theta']:.1f} | {r['exit_rate'] * 100:.1f} | "
                   f"{r['ptc'] * 100:.1f} | {r['passages_per_s']:.1f} | "
                   f"{r['speed_index']:.2f} |\n")
    md_name = (args.out[:-5] if args.out.endswith(".json") else args.out) + ".md"
    with open(os.path.join(C.RESULTS, md_name), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"wrote results/{args.out} and results/{md_name}")


if __name__ == "__main__":
    main()
