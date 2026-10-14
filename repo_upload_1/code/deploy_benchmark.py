"""Deployment-side benchmark for the Applied Intelligence revision.

The reviewer panel asked for three things the paper did not have:

  (1) an external, deployment-level baseline -- an INT8 quantised student and
      a graph-optimised student -- so that "dual head + early exit" can be
      compared against the ordinary ways of serving a two-layer encoder;
  (2) single-item (batch-1) latency, because a service that sees one request
      at a time is where an early exit actually pays, while a batched
      workload can hide it;
  (3) a *measurement* of the thermal drift the paper calls its largest
      error source, instead of a hypothesis about it.

Two round-robins are run, each internally consistent (the paper's rule is
that a speed ratio is only defensible inside one round-robin):

  * GPU   -- S0, F2, S7(exit), S7(no exit), and graph-optimised (TorchScript)
             copies of the two non-branching graphs.  INT8 dynamic
             quantisation is CPU-only in PyTorch, so it cannot appear here.
  * CPU   -- the same systems plus INT8 copies of S0, F2 and S7(exit).  This
             is the deployment path the paper's own framing ("edge, latency
             constrained") implies, and it is where INT8 is available.

Throughout both, a sampler records SM clock, temperature and board power, and
batch-1 latency is timed inside the same round-robin as everything else.

Writes results/deploy_benchmark.json and results/deploy_benchmark.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEED = 20261011


# ---------------------------------------------------------------------------
def load_json(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def tag_flags(*files):
    """tag -> {dual_head, head_on_layer, seed} using the first row per tag."""
    out = {}
    for rows in files:
        for r in rows or []:
            if r["tag"] in out:
                continue
            out[r["tag"]] = {"dual_head": bool(r.get("dual_head")),
                             "head_on_layer": r.get("head_on_layer"),
                             "seed": r["seed"], "da": r.get("da")}
    return out


def build_batches(tokenizer, texts, labels, batch_size):
    feats = tokenizer(texts, padding="max_length", truncation=True,
                      max_length=C.MAX_LEN, return_tensors="pt")
    n = len(texts)
    for i in range(0, n, batch_size):
        sl = slice(i, min(i + batch_size, n))
        yield {"input_ids": feats["input_ids"][sl],
               "attention_mask": feats["attention_mask"][sl],
               "labels": torch.tensor(labels[sl])}


# ---------------------------------------------------------------------------
class GpuSampler(threading.Thread):
    """Poll nvidia-smi while the round-robin runs."""

    QUERY = ("clocks.sm,temperature.gpu,power.draw,memory.used,"
             "utilization.gpu")

    def __init__(self, period=0.25):
        super().__init__(daemon=True)
        self.period = period
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self.QUERY}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                if out:
                    parts = [p.strip() for p in out.splitlines()[0].split(",")]
                    self.samples.append(dict(
                        t=time.time(),
                        clock_mhz=float(parts[0]),
                        temp_c=float(parts[1]),
                        power_w=float(parts[2]),
                        mem_mib=float(parts[3]),
                        util=float(parts[4])))
            except Exception:
                pass
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        self.join(timeout=3)

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x["t"] <= t1]
        if not s:
            return None
        p = np.array([x["power_w"] for x in s])
        ts = np.array([x["t"] for x in s])
        energy_j = float(np.trapezoid(p, ts)) if len(ts) > 1 else 0.0
        return dict(n=len(s),
                    clock_mean=round(float(np.mean([x["clock_mhz"] for x in s])), 1),
                    clock_min=round(min(x["clock_mhz"] for x in s), 1),
                    clock_max=round(max(x["clock_mhz"] for x in s), 1),
                    temp_mean=round(float(np.mean([x["temp_c"] for x in s])), 1),
                    temp_max=round(max(x["temp_c"] for x in s), 1),
                    power_mean=round(float(p.mean()), 2),
                    power_max=round(float(p.max()), 2),
                    energy_j=round(energy_j, 1))


# ---------------------------------------------------------------------------
def make_units(models, meta, batches, batch1, theta, device):
    def full_fn(m):
        return lambda b: m(b["input_ids"], b["attention_mask"])

    def exit_fn(m, th):
        return lambda b: m.forward_with_exit(b["input_ids"], b["attention_mask"],
                                             th)

    def fixed_fn(m):
        return lambda b: m.forward_fixed_depth(b["input_ids"],
                                               b["attention_mask"])

    units = []
    for t in sorted(models):
        m, kind = models[t], meta[t]["kind"]
        if kind == "dual_head":
            units.append((f"{t}@exit", exit_fn(m, theta), False))
            units.append((f"{t}@noexit", exit_fn(m, -1.0), False))
        elif kind == "fixed_depth":
            units.append((t, fixed_fn(m), False))
        else:
            units.append((t, full_fn(m), False))
    return units


def round_robin(units, batches, batch1, rounds, warmup_rounds, sampler=None):
    """Interleaved timing; returns per-unit seconds and the sampling window."""
    with torch.inference_mode():
        for _ in range(warmup_rounds):
            for _, fn, _ in units:
                for b in batches:
                    fn(b)
    _sync()
    t0 = time.time()
    elapsed = {lbl: 0.0 for lbl, _, _ in units}
    b1 = {lbl: [] for lbl, _, _ in units}
    for _ in range(rounds):
        for lbl, fn, _ in units:
            t = time.perf_counter()
            with torch.inference_mode():
                for b in batches:
                    fn(b)
            _sync()
            elapsed[lbl] += time.perf_counter() - t
            for b in batch1:
                t = time.perf_counter()
                with torch.inference_mode():
                    fn(b)
                _sync()
                b1[lbl].append(time.perf_counter() - t)
    t1 = time.time()
    return elapsed, b1, sampler.window(t0, t1) if sampler else None, t1 - t0


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
class _LogitsOnly(nn.Module):
    """Adapter so that the traced graph returns a single tensor.

    `TinyStudent.forward` returns a dict that mixes a Tensor with an optional
    attention tuple, which the tracer refuses; the deployed path only needs the
    classifier logits.
    """

    def __init__(self, m, mode):
        super().__init__()
        self.m = m
        self.mode = mode

    def forward(self, input_ids, attention_mask):
        if self.mode == "fixed_depth":
            return self.m.forward_fixed_depth(input_ids, attention_mask)
        return self.m(input_ids, attention_mask)["main"]


def torchscript_copy(m, mode="plain"):
    """Graph-optimised copy.  Tracing fixes data-dependent branches, so this is
    only applied to the graphs whose control flow does not depend on the input;
    the early-exit path is therefore excluded by the caller.

    Tracing and `optimize_for_inference` both happen on the module's own
    device: optimize_for_inference freezes parameters into constants, and a
    frozen module no longer follows `.to(device)`.
    """
    dev = next(m.parameters()).device
    wrapped = _LogitsOnly(m.eval(), mode).eval()
    for p in wrapped.parameters():
        p.requires_grad_(False)
    ex = (torch.zeros(2, C.MAX_LEN, dtype=torch.long, device=dev),
          torch.ones(2, C.MAX_LEN, dtype=torch.long, device=dev))
    traced = torch.jit.trace(wrapped, ex, strict=False)
    traced = torch.jit.optimize_for_inference(traced)
    for p in traced.parameters():
        p.requires_grad_(False)
    return traced.eval()


def quantise(m):
    return torch.quantization.quantize_dynamic(
        m, {nn.Linear}, dtype=torch.qint8)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--warmup-rounds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--batch1-n", type=int, default=32)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--out", default="deploy_benchmark.json")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    records = C.load_ptc(None)
    scope = records                       # all 416 labelled paragraphs
    texts = [r["text"] for r in scope]
    labels = [r["label_id"] for r in scope]
    tokenizer = AutoTokenizer.from_pretrained(C.resolve(C.TOKENIZER_EN))

    out = {"theta": args.theta, "batch_size": args.batch_size,
           "rounds": args.rounds, "paragraphs": len(texts),
           "batch1_n": args.batch1_n}

    # ---------------- GPU round-robin ------------------------------------
    gpu_dev = torch.device("cuda")
    if not args.skip_gpu and torch.cuda.is_available():
        batches = [dict(b) for b in build_batches(tokenizer, texts, labels,
                                                  args.batch_size)]
        for b in batches:
            for k in list(b):
                b[k] = b[k].to(gpu_dev)
        b1_texts = texts[:args.batch1_n]
        b1_lab = labels[:args.batch1_n]
        batch1 = []
        for b in build_batches(tokenizer, b1_texts, b1_lab, 1):
            for k in list(b):
                b[k] = b[k].to(gpu_dev)
            batch1.append(b)

        flags = tag_flags(load_json("students.json"),
                          load_json("fixed_depth.json"))
        models, meta = {}, {}
        ck = os.path.join(C.MODELS, "students")
        for tag in ("S0", "F2", "S7"):
            if tag not in flags:
                print(f"  !! no flag row for {tag}", flush=True)
                continue
            path = os.path.join(ck, f"{tag}_{flags[tag]['seed']}.pt")
            m = C.TinyStudent(head_layer=flags[tag].get("head_on_layer"))
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.to(gpu_dev).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            if flags[tag]["dual_head"]:
                kind = "dual_head"
            elif flags[tag].get("head_on_layer") is not None:
                kind = "fixed_depth"
            else:
                kind = "plain"
            models[tag] = m
            meta[tag] = {"kind": kind,
                         "params_m": round(C.count_params(m) / 1e6, 2),
                         "seed": flags[tag]["seed"]}
            print(f"  [gpu] loaded {tag} ({kind}, "
                  f"{meta[tag]['params_m']}M)", flush=True)

        # graph-optimised copies of the two non-branching graphs
        try:
            for tag in ("S0", "F2"):
                if tag in models:
                    ts = torchscript_copy(models[tag], meta[tag]["kind"])
                    models[tag + "@jit"] = ts
                    meta[tag + "@jit"] = {"kind": "jit",
                                          "params_m": meta[tag]["params_m"],
                                          "seed": None}
                    print(f"  [gpu] TorchScript copy of {tag} ready", flush=True)
        except Exception as e:                          # pragma: no cover
            print(f"  !! TorchScript failed: {type(e).__name__}: {e}",
                  flush=True)

        units = make_units(models, meta, batches, batch1, args.theta, gpu_dev)
        sampler = GpuSampler()
        sampler.start()
        time.sleep(0.5)
        print(f"  [gpu] round-robin over {len(units)} units x "
              f"{args.rounds} rounds", flush=True)
        elapsed, b1, window, wall = round_robin(units, batches, batch1,
                                                args.rounds,
                                                args.warmup_rounds, sampler)
        sampler.stop()

        ref = elapsed["S0"]
        rows = {}
        for lbl, _, _ in units:
            arr = np.array(b1[lbl]) * 1000.0            # ms
            rows[lbl] = dict(
                seconds=round(elapsed[lbl], 3),
                passages_per_s=round(len(texts) * args.rounds / elapsed[lbl], 2),
                speed_index=round(100.0 * ref / elapsed[lbl], 2),
                batch1_p50_ms=round(float(np.percentile(arr, 50)), 3),
                batch1_p99_ms=round(float(np.percentile(arr, 99)), 3),
                batch1_mean_ms=round(float(arr.mean()), 3),
                batch1_n=len(arr))
        # pool the INT8/CPU numbers separately; here just the GPU ones
        out["gpu"] = dict(
            baseline="S0", wall_seconds=round(wall, 1),
            clock=dict(
                mean_clock=window["clock_mean"] if window else None,
                min_clock=window["clock_min"] if window else None,
                max_clock=window["clock_max"] if window else None,
                mean_temp=window["temp_mean"] if window else None,
                max_temp=window["temp_max"] if window else None,
                mean_power=window["power_mean"] if window else None,
                max_power=window["power_max"] if window else None,
                samples=window["n"] if window else 0),
            energy_j=window["energy_j"] if window else None,
            energy_j_per_1k=round(window["energy_j"]
                                  / (len(texts) * args.rounds) * 1000, 2)
            if window else None,
            systems=rows)
        print("  [gpu] done", flush=True)

    # ---------------- CPU round-robin ------------------------------------
    if not args.skip_cpu:
        cpu_dev = torch.device("cpu")
        torch.set_num_threads(8)
        batches = [dict(b) for b in build_batches(tokenizer, texts, labels,
                                                  args.batch_size)]
        b1_texts = texts[:args.batch1_n]
        b1_lab = labels[:args.batch1_n]
        batch1 = list(build_batches(tokenizer, b1_texts, b1_lab, 1))

        flags = tag_flags(load_json("students.json"),
                          load_json("fixed_depth.json"))
        models, meta = {}, {}
        ck = os.path.join(C.MODELS, "students")
        for tag in ("S0", "F2", "S7"):
            if tag not in flags:
                continue
            path = os.path.join(ck, f"{tag}_{flags[tag]['seed']}.pt")
            m = C.TinyStudent(head_layer=flags[tag].get("head_on_layer"))
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            kind = ("dual_head" if flags[tag]["dual_head"]
                    else "fixed_depth" if flags[tag].get("head_on_layer")
                    is not None else "plain")
            models[tag] = m
            meta[tag] = {"kind": kind, "params_m": round(C.count_params(m) / 1e6, 2),
                         "seed": flags[tag]["seed"]}
            q = quantise(m)
            models[tag + "@int8"] = q
            meta[tag + "@int8"] = {"kind": "int8", "params_m": None,
                                   "seed": None}
            print(f"  [cpu] loaded {tag} and its INT8 copy", flush=True)

        units = make_units(models, meta, batches, batch1, args.theta, cpu_dev)
        elapsed, b1, _, wall = round_robin(units, batches, batch1,
                                           max(2, args.rounds // 3),
                                           args.warmup_rounds, None)
        ref = elapsed["S0"]
        rows = {}
        for lbl, _, _ in units:
            arr = np.array(b1[lbl]) * 1000.0
            rows[lbl] = dict(
                seconds=round(elapsed[lbl], 3),
                passages_per_s=round(len(texts) * max(2, args.rounds // 3)
                                     / elapsed[lbl], 2),
                speed_index=round(100.0 * ref / elapsed[lbl], 2),
                batch1_p50_ms=round(float(np.percentile(arr, 50)), 3),
                batch1_p99_ms=round(float(np.percentile(arr, 99)), 3),
                batch1_mean_ms=round(float(arr.mean()), 3),
                batch1_n=len(arr))
        out["cpu"] = dict(baseline="S0", wall_seconds=round(wall, 1),
                          threads=torch.get_num_threads(),
                          rounds=max(2, args.rounds // 3), systems=rows)
        print("  [cpu] done", flush=True)

    with open(os.path.join(RES, args.out), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    L = ["# Deployment-side benchmark (round 75)\n",
         f"Threshold theta = {args.theta}; batch size {args.batch_size};"
         f" {args.rounds} rounds; {out['paragraphs']} paragraphs per pass.\n"]
    if "gpu" in out:
        g = out["gpu"]
        L.append("\n## GPU round-robin (baseline = S0)\n")
        L.append(f"Wall clock of the timed region: {g['wall_seconds']} s."
                 f"  SM clock {g['clock']['min_clock']}-{g['clock']['max_clock']} MHz"
                 f" (mean {g['clock']['mean_clock']});"
                 f" temperature mean {g['clock']['mean_temp']} C,"
                 f" max {g['clock']['max_temp']} C;"
                 f" board power mean {g['clock']['mean_power']} W,"
                 f" max {g['clock']['max_power']} W;"
                 f" {g['clock']['samples']} samples.")
        L.append(f"Energy over the timed region: {g['energy_j']} J"
                 f" ({g['energy_j_per_1k']} J per 1000 paragraphs"
                 f" per round of the whole table).\n")
        L.append("| unit | passages/s | speed index | batch-1 p50 (ms) |"
                 " batch-1 p99 (ms) |")
        L.append("|---|---|---|---|---|")
        for lbl, r in sorted(g["systems"].items(),
                             key=lambda kv: -kv[1]["speed_index"]):
            L.append(f"| {lbl} | {r['passages_per_s']} | {r['speed_index']} |"
                     f" {r['batch1_p50_ms']} | {r['batch1_p99_ms']} |")
    if "cpu" in out:
        c = out["cpu"]
        L.append(f"\n## CPU round-robin (baseline = S0, {c['threads']} threads,"
                 f" {c['rounds']} rounds)\n")
        L.append("| unit | passages/s | speed index | batch-1 p50 (ms) |"
                 " batch-1 p99 (ms) |")
        L.append("|---|---|---|---|---|")
        for lbl, r in sorted(c["systems"].items(),
                             key=lambda kv: -kv[1]["speed_index"]):
            L.append(f"| {lbl} | {r['passages_per_s']} | {r['speed_index']} |"
                     f" {r['batch1_p50_ms']} | {r['batch1_p99_ms']} |")
    with open(os.path.join(RES, "deploy_benchmark.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote results/" + args.out + " and results/deploy_benchmark.md")


if __name__ == "__main__":
    main()
