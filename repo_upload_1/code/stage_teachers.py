"""Stage A: fine-tune the two teachers on paragraph topic classification.

Both teachers are trained with the identical protocol so that any difference
between them is attributable to the pretrained checkpoint and not to tuning.
The fine-tuned teachers serve two purposes in the paper:

  * they are the reference systems in the accuracy / efficiency comparison;
  * their class posteriors are the soft targets used by the distillation
    systems in Stage B.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertForSequenceClassification

import common as C


def encode(tokenizer, texts, max_len=None):
    max_len = C.MAX_LEN if max_len is None else max_len
    return tokenizer(list(texts), padding="max_length", truncation=True,
                     max_length=max_len, return_tensors="pt")


def make_batches(enc, labels, batch_size, shuffle=False, device=None):
    n = len(labels)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        batch = {k: v[idx].to(device) for k, v in enc.items()}
        yield batch, labels[idx].to(device)


def make_batches_dynamic(tokenizer, texts, labels, batch_size, shuffle=False,
                         device=None):
    """Tokenise per batch and pad to the longest member.

    LD-100K paragraphs are 6-86 word-pieces long with a mean of 40, so padding
    every batch to a flat 96 wastes roughly half the compute.  The test bundle
    used for the throughput measurement is deliberately *not* built this way:
    every system has to be timed on byte-identical inputs.
    """
    n = len(labels)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        sel = [texts[j] for j in idx.tolist()]
        enc = tokenizer(sel, padding="longest", truncation=True,
                        max_length=C.MAX_LEN, return_tensors="pt")
        yield {k: v.to(device) for k, v in enc.items()}, labels[idx].to(device)


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    logits, labels = [], []
    for batch, y in batches:
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"])
        logits.append(out.logits.float().cpu())
        labels.append(y.cpu())
    L = torch.cat(logits)
    Y = torch.cat(labels)
    return C.accuracy(L, Y), C.macro_f1(L, Y), L, Y


def train_teacher(tag, model_name, records, tokenizer_name=None, epochs=20,
                  batch_size=16, lr=3e-5, warmup_ratio=0.1, weight_decay=0.01,
                  seed=13, device=None, models_dir=None, result_prefix=""):
    device = device or C.get_device()
    tokenizer_name = tokenizer_name or model_name
    models_dir = models_dir or C.MODELS
    tok = AutoTokenizer.from_pretrained(C.resolve(tokenizer_name))

    tr = C.split_of(records, "train")
    dv = C.split_of(records, "dev")
    te = C.split_of(records, "test")

    tr_texts = [r["text"] for r in tr]
    dv_texts = [r["text"] for r in dv]
    y_tr = torch.tensor([r["label_id"] for r in tr])
    y_dv = torch.tensor([r["label_id"] for r in dv])
    enc_te = encode(tok, [r["text"] for r in te])
    y_te = torch.tensor([r["label_id"] for r in te])

    C.set_seed(seed)
    model = BertForSequenceClassification.from_pretrained(
        C.resolve(model_name), num_labels=C.NUM_LABELS,
        attn_implementation="eager").to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.999), eps=1e-8)
    steps_per_epoch = max(1, len(tr) // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        prog = (step - warmup) / float(max(1, total_steps - warmup))
        return max(0.0, 1.0 - prog)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    lossf = nn.CrossEntropyLoss()

    best = {"dev_acc": -1.0, "state": None, "epoch": -1}
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    epoch_log = []
    for ep in range(1, epochs + 1):
        model.train()
        run_loss, nseen = 0.0, 0
        for batch, y in make_batches_dynamic(tok, tr_texts, y_tr, batch_size,
                                             True, device):
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"], labels=y)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run_loss += float(loss.item()) * y.shape[0]
            nseen += y.shape[0]
        dev_acc, dev_f1, _, _ = evaluate(model, list(make_batches_dynamic(
            tok, dv_texts, y_dv, batch_size, False, device)))
        epoch_log.append({"epoch": ep, "train_loss": run_loss / max(1, nseen),
                          "dev_acc": dev_acc, "dev_macro_f1": dev_f1})
        if dev_acc > best["dev_acc"]:
            best = {"dev_acc": dev_acc, "dev_macro_f1": dev_f1, "epoch": ep,
                    "state": copy.deepcopy({k: v.detach().cpu()
                                            for k, v in model.state_dict().items()})}
        print(f"  [{tag}] epoch {ep:02d}/{epochs}  loss={run_loss / max(1, nseen):.4f}  "
              f"dev_acc={dev_acc:.4f}", flush=True)

    train_time = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device.type == "cuda" else 0.0
    model.load_state_dict(best["state"])
    model.to(device)
    te_acc, te_f1, te_logits, te_y = evaluate(
        model, list(make_batches(enc_te, y_te, batch_size, False, device)))

    # ---- inference throughput -------------------------------------------
    test_batches = list(make_batches(enc_te, y_te, batch_size, False, device))

    def fn(b):
        return model(input_ids=b[0]["input_ids"],
                     attention_mask=b[0]["attention_mask"])

    with torch.no_grad():
        pps, speed_peak = C.measure_throughput(fn, test_batches, warmup=2, repeats=3)

    outdir = os.path.join(models_dir, tag)
    os.makedirs(outdir, exist_ok=True)
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)

    result = {
        "tag": tag, "model_name": model_name, "role": "teacher/reference",
        "params_m": round(C.count_params(model) / 1e6, 1),
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "warmup_steps": warmup, "total_steps": total_steps,
        "seed": seed,
        "train_time_s": round(train_time, 1),
        "best_epoch": best["epoch"],
        "dev_acc": round(best["dev_acc"], 4),
        "dev_macro_f1": round(best.get("dev_macro_f1", 0.0), 4),
        "test_acc": round(te_acc, 4),
        "test_macro_f1": round(te_f1, 4),
        # per-paragraph posteriors: the reference systems take part in the
        # paired significance tests, so they need the same two fields as the
        # student systems.
        "test_probs": [[round(float(v), 5) for v in row]
                       for row in torch.softmax(te_logits.float(), dim=-1).tolist()],
        "test_gold": [int(v) for v in te_y.tolist()],
        "passages_per_s": round(pps, 2),
        "peak_vram_mb": round(peak, 1),
        "speed_peak_vram_mb": round(speed_peak, 1),
        "epoch_log": epoch_log,
    }
    C.json_dump(result, os.path.join(C.RESULTS, f"{result_prefix}{tag}.json"))
    return result, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[13])
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--tags", nargs="+", default=["BERT-base", "mBERT"])
    ap.add_argument("--data", default=None,
                    help="ptc-style jsonl corpus (default: data/ptc.jsonl = LD-100K)")
    ap.add_argument("--models-subdir", default="",
                    help="save fine-tuned teachers under models/<subdir>/ (organic run)")
    ap.add_argument("--result-prefix", default="",
                    help="prefix for the per-teacher result json (organic run)")
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
    models_dir = os.path.join(C.MODELS, args.models_subdir) if args.models_subdir else C.MODELS

    all_results = []
    for seed in args.seeds:
        for tag, name in (("BERT-base", C.TEACHER_EN), ("mBERT", C.TEACHER_MULTI)):
            if tag not in args.tags:
                continue
            run_tag = tag + args.tag_suffix
            print(f"=== training {run_tag} (seed {seed}) ===", flush=True)
            res, _ = train_teacher(run_tag, name, records, epochs=args.epochs,
                                   batch_size=args.batch_size, lr=args.lr,
                                   seed=seed, device=device,
                                   models_dir=models_dir,
                                   result_prefix=args.result_prefix)
            all_results.append(res)
            print(json.dumps(res, indent=2)[:600], flush=True)
            del _
            torch.cuda.empty_cache()

    out_name = (f"{args.result_prefix}teachers"
                f"{args.tag_suffix}.json" if args.tag_suffix
                else f"{args.result_prefix}teachers.json")
    C.json_dump(all_results, os.path.join(C.RESULTS, out_name))
    print("teachers done")


if __name__ == "__main__":
    main()
