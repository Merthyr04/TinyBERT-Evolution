"""Shallow/deep head agreement and exit-vs-deep decisions.

Section 4.6 of the paper describes the deployed dual-head system as a fixed
two-layer encoder, on the grounds that the auxiliary (shallow) head and the
deep head agree almost everywhere and the exit therefore rarely changes an
answer.  This script measures that claim exactly, for every dual-head system
and every seed:

  * the fraction of the 196 test paragraphs on which the auxiliary head's
    argmax equals the deep head's;
  * how many of the 196 decisions differ between the exit path at
    theta = 0.5 and the always-deep path (theta = -1);
  * the accuracy of each of the three paths.

By default it runs on LD-100K, the corpus of the main study, and writes
results/head_agreement.json.  `--data / --models-subdir / --max-len` reroute it
at the second, organically written document (OD-100K, Section 4.11), whose
checkpoints live under models/organic/.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import common as C
from transformers import AutoTokenizer

DUAL = ["S3", "S7", "A5", "S8"]

# The deployed operating point of Table 6 (Section 4.6): the exit is entered at
# theta = 0.5 for every system, which is the registration convention and not the
# output of a selection rule.
THETA = 0.5


def resolve_ckpt(tag, seed, subdir=""):
    root = os.path.join(C.MODELS, subdir) if subdir else C.MODELS
    return os.path.join(root, "students", f"{tag}_{seed}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="corpus to score (default: LD-100K)")
    ap.add_argument("--models-subdir", default="",
                    help="read checkpoints from models/<subdir>/ (organic run)")
    ap.add_argument("--max-len", type=int, default=None,
                    help="override C.MAX_LEN (organic corpus: 112)")
    ap.add_argument("--num-labels", type=int, default=None,
                    help="override C.NUM_LABELS (the public-benchmark "
                         "replication uses four classes instead of twelve)")
    ap.add_argument("--tags", nargs="*", default=DUAL,
                    help="systems to score (default: every dual-head system)")
    ap.add_argument("--out", default="head_agreement",
                    help="stem of the result files under results/")
    ap.add_argument("--label", default="LD-100K",
                    help="corpus name used in the markdown report")
    args = ap.parse_args()
    if args.max_len:
        C.MAX_LEN = args.max_len
    if args.num_labels:
        C.NUM_LABELS = args.num_labels

    dev = C.get_device()
    tok = AutoTokenizer.from_pretrained(C.resolve(C.TOKENIZER_EN))
    recs = C.load_ptc(args.data)
    te = C.split_of(recs, "test")
    enc = tok([r["text"] for r in te], padding="max_length", truncation=True,
              max_length=C.MAX_LEN, return_tensors="pt")
    y = torch.tensor([r["label_id"] for r in te])

    rows = []
    for tag in args.tags:
        for seed in (7, 13, 37, 42, 97):
            p = resolve_ckpt(tag, seed, args.models_subdir)
            if not os.path.exists(p):
                continue
            m = C.TinyStudent()
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.to(dev).eval()
            A, M, E, X = [], [], [], []
            with torch.no_grad():
                for i in range(0, len(y), 32):
                    ids = enc["input_ids"][i:i + 32].to(dev)
                    am = enc["attention_mask"][i:i + 32].to(dev)
                    o = m(ids, am)
                    A.append(o["aux"].argmax(-1).cpu())
                    M.append(o["main"].argmax(-1).cpu())
                    lg, cf = m.forward_with_exit(ids, am, THETA)
                    E.append(lg.argmax(-1).cpu())
                    X.append(cf.cpu())
            A, M, E = torch.cat(A), torch.cat(M), torch.cat(E)
            X = torch.cat(X)
            n = int(y.numel())
            rows.append({
                "tag": tag, "seed": seed, "n": n,
                "aux_acc": round(float((A == y).float().mean()), 4),
                "main_acc": round(float((M == y).float().mean()), 4),
                "exit_acc": round(float((E == y).float().mean()), 4),
                "aux_eq_main": int((A == M).sum()),
                "exit_ne_deep": int((E != M).sum()),
                "exit_ne_aux": int((E != A).sum()),
                "exit_rate": round(float(X.float().mean()), 4),
            })
            print(f"{tag} seed {seed}: aux={rows[-1]['aux_acc']:.3f} "
                  f"main={rows[-1]['main_acc']:.3f} exit={rows[-1]['exit_acc']:.3f} | "
                  f"aux==main {rows[-1]['aux_eq_main']}/{n} | "
                  f"exit!=deep {rows[-1]['exit_ne_deep']}/{n} | "
                  f"exit rate {rows[-1]['exit_rate'] * 100:.1f}%", flush=True)
            del m
            torch.cuda.empty_cache()

    rep = {"theta": THETA, "corpus": args.label, "rows": rows}
    C.json_dump(rep, os.path.join(C.RESULTS, args.out + ".json"))

    L = [f"# Shallow/deep head agreement on {args.label}\n",
         f"Exit threshold theta = {THETA}.  Test split: 196 paragraphs.\n",
         "| system | seed | aux acc | deep acc | exit acc | aux = deep | exit != deep |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['tag']} | {r['seed']} | {r['aux_acc'] * 100:.1f} | "
                 f"{r['main_acc'] * 100:.1f} | {r['exit_acc'] * 100:.1f} | "
                 f"{r['aux_eq_main']}/{r['n']} | {r['exit_ne_deep']}/{r['n']} |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(C.RESULTS, args.out + ".md"), "w",
              encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
