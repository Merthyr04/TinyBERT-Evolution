"""Serving-level measurements added in the Applied Intelligence revision.

Three measurements, all in ONE process so they share a clock trajectory:

1. Batch-size sweep: S0, S7 (exit at theta = 0.5) and F2 (fixed two-layer)
   at batch sizes 1/2/4/8/16/32 over all 416 paragraphs, round-robin.
   The speed index is reported within each batch size against S0 at that
   size, because the S0 denominator itself changes with the batch size.

2. Single-stream latency percentiles: per-paragraph wall-clock latency for
   the same three systems at batch size 1, p50/p95/p99 in milliseconds.

3. INT8 attribution (CPU): dynamic INT8 flattens the layer saving on the
   CPU.  Two probes without claiming a kernel-level mechanism:
   (a) full-model CPU throughput, FP32 vs INT8, batch 1 and 16;
   (b) an isolated nn.Linear micro-benchmark at the student's shapes,
       showing the per-call quantise-dequantise overhead that does not
       scale with depth.

Writes results/serving_bench.json
"""
from __future__ import annotations

import json
import time

import torch
from transformers import AutoTokenizer

import common as C

SEED = 13
THETA = 0.5
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
SWEEP_ROUNDS = 6
LAT_ROUNDS = 8
OUT = "results/serving_bench.json"


def load_student(tag, head_on_layer=None):
    m = C.TinyStudent(head_layer=head_on_layer)
    m.load_state_dict(torch.load(f"models/students/{tag}_{SEED}.pt",
                                 map_location="cpu"))
    return m.to(C.get_device()).eval()


def build_batches(tokenizer, texts, batch_size):
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=C.MAX_LEN, return_tensors="pt")
    return [{k: v[i:i + batch_size].to(C.get_device())
             for k, v in enc.items()}
            for i in range(0, len(texts), batch_size)]


def timed_pass(fn, batches):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for b in batches:
            fn(b)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    device = C.get_device()
    print("device:", device, flush=True)
    records = C.load_ptc(None)
    texts = [r["text"] for r in records]
    tokenizer = AutoTokenizer.from_pretrained(C.resolve(C.TOKENIZER_EN))

    s0 = load_student("S0")
    s7 = load_student("S7")
    f2 = load_student("F2", head_on_layer=2)

    units = [
        ("S0", lambda b: s0(b["input_ids"], b["attention_mask"])),
        ("S7@exit", lambda b: s7.forward_with_exit(b["input_ids"],
                                                   b["attention_mask"], THETA)),
        ("F2", lambda b: f2.forward_fixed_depth(b["input_ids"],
                                                b["attention_mask"])),
    ]

    # ---- warm-up (largest workload covers the clock) ----------------------
    print("warming up ...", flush=True)
    warm = build_batches(tokenizer, texts, 32)
    with torch.inference_mode():
        for _ in range(3):
            for _, fn in units:
                for b in warm:
                    fn(b)
    torch.cuda.synchronize()

    # ---- 1. batch-size sweep, round-robin ---------------------------------
    sweep = {}
    batches_by_bs = {bs: build_batches(tokenizer, texts, bs)
                     for bs in BATCH_SIZES}
    print("batch-size sweep ...", flush=True)
    acc = {(lbl, bs): 0.0 for lbl, _ in units for bs in BATCH_SIZES}
    with torch.inference_mode():
        for rd in range(SWEEP_ROUNDS):
            for bs in BATCH_SIZES:
                for lbl, fn in units:
                    acc[(lbl, bs)] += timed_pass(fn, batches_by_bs[bs])
    for (lbl, bs), dt in acc.items():
        pps = len(texts) * SWEEP_ROUNDS / dt
        sweep.setdefault(str(bs), {})[lbl] = round(pps, 1)
    for bs in BATCH_SIZES:
        base = sweep[str(bs)]["S0"]
        for lbl, _ in units:
            sweep[str(bs)][lbl + "_idx"] = round(100.0 * sweep[str(bs)][lbl]
                                                 / base, 1)

    # ---- 2. single-stream latency percentiles -----------------------------
    print("latency percentiles ...", flush=True)
    lat = {}
    single = build_batches(tokenizer, texts, 1)
    with torch.inference_mode():
        for lbl, fn in units:
            # one untimed warm pass
            for b in single:
                fn(b)
            torch.cuda.synchronize()
            times = []
            for _ in range(LAT_ROUNDS):
                for b in single:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    fn(b)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
            times.sort()
            n = len(times)

            def pct(q):
                return round(1000.0 * times[min(n - 1, int(q * n))], 3)

            lat[lbl] = {"p50_ms": pct(0.50), "p95_ms": pct(0.95),
                        "p99_ms": pct(0.99), "mean_ms":
                            round(1000.0 * sum(times) / n, 3),
                        "n": n * len(single) and n}
    torch.cuda.synchronize()

    # ---- 3. INT8 attribution on CPU ---------------------------------------
    print("INT8 attribution (CPU) ...", flush=True)
    cpu_batches = {bs: [{k: v.cpu() for k, v in b.items()}
                        for b in batches_by_bs[bs]] for bs in (1, 16)}
    s0_cpu = C.TinyStudent().eval()

    def cpu_time(model, bs, reps=5):
        bs_batches = cpu_batches[bs]
        with torch.inference_mode():
            for b in bs_batches:
                model(b["input_ids"], b["attention_mask"])
            t0 = time.perf_counter()
            for _ in range(reps):
                for b in bs_batches:
                    model(b["input_ids"], b["attention_mask"])
            return (time.perf_counter() - t0) / (reps * len(bs_batches))

    int8 = {}
    for bs in (1, 16):
        t_fp = cpu_time(s0_cpu, bs)
        q = torch.ao.quantization.quantize_dynamic(
            s0_cpu, {torch.nn.Linear}, dtype=torch.qint8)
        t_q = cpu_time(q, bs)
        int8[str(bs)] = {"fp32_ms_per_pass": round(1000 * t_fp, 3),
                         "int8_ms_per_pass": round(1000 * t_q, 3),
                         "speedup": round(t_fp / t_q, 2)}
    # isolated Linear probe at the student's shapes
    lin = torch.nn.Linear(312, 1200).eval()
    qlin = torch.ao.quantization.quantize_dynamic(
        lin, {torch.nn.Linear}, dtype=torch.qint8)
    probe = {}
    for bs in (1, 16, 64, 256):
        x = torch.randn(bs, 312)
        with torch.inference_mode():
            for m, name in ((lin, "fp32"), (qlin, "int8")):
                # pyright: ignore[reportArgumentType]
                for _ in range(20):
                    m(x)
                t0 = time.perf_counter()
                for _ in range(2000):
                    m(x)
                probe.setdefault(str(bs), {})[name + "_us_per_call"] = \
                    round(1e6 * (time.perf_counter() - t0) / 2000, 2)
    out = {"seed": SEED, "theta": THETA, "n_paragraphs": len(texts),
           "batch_sweep": sweep, "latency_percentiles": lat,
           "int8_cpu": {"full_model": int8, "linear_probe_us": probe},
           "protocol": "one process, round-robin over batch sizes; latency "
                       "percentiles over per-paragraph timings at batch 1; "
                       "INT8 = torch.ao.quantization.quantize_dynamic on CPU"}
    C.json_dump(out, OUT)
    print(json.dumps(out, indent=1)[:2200], flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
