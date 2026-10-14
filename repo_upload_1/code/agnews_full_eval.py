"""AG News full official test set (revision item: R2/R3, round 81).

The public-benchmark replication of round 75 scored the deployed systems on a
196-item subsample of the official AG News test set, matched to the LD-100K
split cardinalities.  The NLPCC panel (round 81) objected that 196 items give
discordance counts too small for the conjunction rule to move.  This script
re-scores the same fifteen checkpoints on the full official test set.

Contamination note, stated in the paper with the numbers: the round-75
subsample drew all 416 of its items (train 120, dev 100, test 196) from the
official test split, so 220 of the 7,600 official test items have been used
for training or checkpoint selection.  The evaluation set is therefore the
official test set minus those 220 items: 7,380 held-out items.  The 196
subsample test items remain: they were never trained on.

Inputs longer than 96 word-pieces are truncated at MAX_LEN (the in-domain
protocol has nothing to truncate; a corpus-wide benchmark does).  The number
of truncated items is reported.

Writes results/agnews_full.json and results/agnews_full.md.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

import common as C
from ablations import mcnemar
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "pub", "agnews_test.jsonl")
SUBSAMPLE = os.path.join(HERE, "data", "agnews_ptc.jsonl")

SYSTEMS = ["S0", "S1", "S3"]
SEEDS = [7, 13, 37, 42, 97]
THETA = 0.5          # the registered operating point
B = 10000            # bootstrap resamples (chunked)
BOOT_SEED = 20261012


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def bootstrap_ci_p(corr_a, corr_b, n_boot=B, seed=BOOT_SEED, chunk=1000):
    """Chunked paired bootstrap of mean(a) - mean(b) on 0/1 correctness
    vectors (float32 to keep the index gather at ~30 MB per chunk)."""
    rng = np.random.default_rng(seed)
    n = len(corr_a)
    ds = []
    for _ in range(0, n_boot, chunk):
        k = min(chunk, n_boot)
        idx = rng.integers(0, n, size=(k, n))
        ds.append(corr_a[idx].mean(axis=1, dtype=np.float64)
                  - corr_b[idx].mean(axis=1, dtype=np.float64))
    d = np.concatenate(ds)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0)), d


def main():
    C.NUM_LABELS = 4     # AG News heads: the checkpoints carry 4-class classifiers
    dev = C.get_device()
    tok = AutoTokenizer.from_pretrained(C.resolve(C.TOKENIZER_EN))

    full = load_rows(SRC)
    sub = load_rows(SUBSAMPLE)
    used = {r["text"] for r in sub if r["split"] in ("train", "dev")}
    held = [{"text": r["text"].replace("\\", " ").strip(), "label_id": r["label"]}
            for r in full if r["text"].replace("\\", " ").strip() not in used]
    held = [r for r in held if r["text"]]
    n_trunc = 0
    lens = []
    for r in held:
        n = len(tok(r["text"], add_special_tokens=True)["input_ids"])
        lens.append(n)
        if n > C.MAX_LEN:
            n_trunc += 1
    texts = [r["text"] for r in held]
    y = torch.tensor([r["label_id"] for r in held])
    print(f"official test items: {len(full)}; excluded (train/dev of the "
          f"round-75 subsample): {len(used)}; evaluation set: {len(held)}; "
          f"longer than {C.MAX_LEN} pieces (truncated): {n_trunc}", flush=True)

    enc = tok(texts, padding="max_length", truncation=True,
              max_length=C.MAX_LEN, return_tensors="pt")

    out_rows = []
    preds = {}          # (tag, kind) -> dict seed -> (pred, corr)
    for tag in SYSTEMS:
        for seed in SEEDS:
            p = os.path.join(C.MODELS, "agnews", "students", f"{tag}_{seed}.pt")
            m = C.TinyStudent()
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.to(dev).eval()
            Pm, Pa, Pe, Cf = [], [], [], []
            with torch.no_grad():
                for i in range(0, len(y), 128):
                    ids = enc["input_ids"][i:i + 128].to(dev)
                    am = enc["attention_mask"][i:i + 128].to(dev)
                    o = m(ids, am)
                    Pm.append(torch.softmax(o["main"], -1).cpu())
                    Pa.append(o["aux"].argmax(-1).cpu())
                    lg, cf = m.forward_with_exit(ids, am, THETA)
                    Pe.append(torch.softmax(lg, -1).cpu())
                    Cf.append(cf.cpu())
            Pm, Pe = torch.cat(Pm), torch.cat(Pe)
            Cf, Pa = torch.cat(Cf), torch.cat(Pa)
            pm = Pm.argmax(-1)
            pe = Pe.argmax(-1)
            acc = float((pm == y).float().mean())
            row = {
                "tag": tag, "seed": seed, "n": int(len(y)),
                "deep_acc": round(acc, 4),
                "exit_acc": round(float((pe == y).float().mean()), 4)
                if tag == "S3" else None,
                "exit_rate": round(float(Cf.float().mean()), 4)
                if tag == "S3" else None,
                "aux_eq_deep": int((Pa == pm).sum()) if tag == "S3" else None,
                "exit_ne_deep": int((pe != pm).sum()) if tag == "S3" else None,
            }
            out_rows.append(row)
            preds[(tag, "deep", seed)] = pm.numpy()
            preds[(tag, "probs_deep", seed)] = Pm.numpy()
            if tag == "S3":
                preds[(tag, "exit", seed)] = pe.numpy()
                preds[(tag, "probs_exit", seed)] = Pe.numpy()
            print(f"{tag} seed {seed}: deep={acc:.4f}"
                  + (f" exit={row['exit_acc']:.4f} exit_rate={row['exit_rate']:.3f} "
                     f"exit!=deep {row['exit_ne_deep']}" if tag == "S3" else ""),
                  flush=True)
            del m
            torch.cuda.empty_cache()

    gold = y.numpy()
    rep = {
        "corpus": "AG News official test set",
        "official_n": len(full),
        "excluded_train_dev": len(used),
        "n": len(held),
        "n_truncated_at_maxlen": n_trunc,
        "max_pieces": int(max(lens)), "theta": THETA, "seeds": SEEDS,
        "rows": out_rows, "contrasts": [], "s3_exit": {},
    }

    # per-seed-mean accuracy per system
    for tag in SYSTEMS:
        accs = [r["deep_acc"] for r in out_rows if r["tag"] == tag]
        v = np.asarray(accs)
        rep.setdefault("systems", {})[tag] = {
            "acc_mean": round(float(v.mean()), 4),
            "acc_sd": round(float(v.std(ddof=0)), 4),
            "per_seed": accs,
        }

    # pooled predictor (mean softmax over seeds), paired bootstrap + McNemar
    for tag in ["S1", "S3"]:
        P = np.stack([preds[(tag, "probs_deep", s)] for s in SEEDS])
        Q = np.stack([preds[("S0", "probs_deep", s)] for s in SEEDS])
        pa = P.mean(0).argmax(-1)
        pb = Q.mean(0).argmax(-1)
        ca = (pa == gold).astype(np.float64)
        cb = (pb == gold).astype(np.float64)
        delta, lo, hi, p_boot, _ = bootstrap_ci_p(ca, cb)
        p_m, n01, n10 = mcnemar(pa.tolist(), pb.tolist(), gold.tolist())
        d_seed = [round(float((preds[(tag, "deep", s)] == gold).mean()
                              - (preds[("S0", "deep", s)] == gold).mean()), 4)
                  for s in SEEDS]
        rep["contrasts"].append({
            "system": tag, "vs": "S0", "path": "always-deep",
            "pooled_acc_system": round(float(ca.mean()), 4),
            "pooled_acc_ref": round(float(cb.mean()), 4),
            "pooled_delta_pp": round(100 * float(ca.mean() - cb.mean()), 2),
            "per_seed_delta_pp": d_seed,
            "mean_per_seed_delta_pp": round(100 * float(np.mean(d_seed)), 2),
            "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
            "p_bootstrap": round(p_boot, 4), "p_mcnemar": round(p_m, 4),
            "only_ref_correct": n01, "only_system_correct": n10,
        })

    # S3 exit path (deployed) vs S0
    P = np.stack([preds[("S3", "probs_exit", s)] for s in SEEDS]).mean(0)
    Q = np.stack([preds[("S0", "probs_deep", s)] for s in SEEDS]).mean(0)
    pe_mode = P.argmax(-1)
    pb = Q.argmax(-1)
    ca = (pe_mode == gold)
    cb = (pb == gold)
    delta, lo, hi, p_boot, _ = bootstrap_ci_p(ca.astype(np.float64),
                                              cb.astype(np.float64))
    p_m, n01, n10 = mcnemar(pe_mode.tolist(), pb.tolist(), gold.tolist())
    rep["s3_exit_vs_s0"] = {
        "path": "exit (theta=0.5, per-item seed majority)",
        "exit_pooled_acc": round(float(ca.mean()), 4),
        "s0_pooled_acc": round(float(cb.mean()), 4),
        "pooled_delta_pp": round(100 * float(ca.mean() - cb.mean()), 2),
        "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
        "p_bootstrap": round(p_boot, 4), "p_mcnemar": round(p_m, 4),
        "only_s0_correct": n01, "only_exit_correct": n10,
    }

    # S3 exit behaviour at full scale
    rows3 = [r for r in out_rows if r["tag"] == "S3"]
    rep["s3_exit"] = {
        "exit_rate_mean": round(float(np.mean([r["exit_rate"] for r in rows3])), 4),
        "exit_rate_min": round(float(np.min([r["exit_rate"] for r in rows3])), 4),
        "exit_rate_max": round(float(np.max([r["exit_rate"] for r in rows3])), 4),
        "exit_ne_deep_min": int(np.min([r["exit_ne_deep"] for r in rows3])),
        "exit_ne_deep_max": int(np.max([r["exit_ne_deep"] for r in rows3])),
        "aux_eq_deep_min": int(np.min([r["aux_eq_deep"] for r in rows3])),
        "aux_eq_deep_max": int(np.max([r["aux_eq_deep"] for r in rows3])),
    }

    C.json_dump(rep, os.path.join(C.RESULTS, "agnews_full.json"))

    L = [f"# AG News full official test set (n = {rep['n']})\n",
         f"Official test items {rep['official_n']}; excluded {rep['excluded_train_dev']} "
         f"(train/dev of the round-75 subsample); truncated at {C.MAX_LEN} pieces: "
         f"{rep['n_truncated_at_maxlen']} (longest item {rep['max_pieces']} pieces).\n",
         "## Accuracy (always-deep path)\n",
         "| system | mean acc (%) | sd | per-seed (%) |", "|---|---|---|---|"]
    for tag in SYSTEMS:
        s = rep["systems"][tag]
        L.append("| %s | %.2f | %.2f | %s |" % (tag, s["acc_mean"] * 100,
                 s["acc_sd"] * 100, ", ".join("%.2f" % (a * 100) for a in s["per_seed"])))
    L += ["", "## Paired contrasts vs S0 (pooled predictor)\n",
          "| contrast | pooled delta (pp) | 95% CI (pp) | p boot | p McNemar | n01/n10 |",
          "|---|---|---|---|---|---|"]
    for c in rep["contrasts"]:
        L.append("| %s vs %s | %+.2f | [%+.2f, %+.2f] | %.4f | %.4f | %d/%d |"
                 % (c["system"], c["vs"], c["pooled_delta_pp"], c["ci95_pp"][0],
                    c["ci95_pp"][1], c["p_bootstrap"], c["p_mcnemar"],
                    c["only_ref_correct"], c["only_system_correct"]))
    s = rep["s3_exit_vs_s0"]
    L.append("| S3 exit vs S0 | %+.2f | [%+.2f, %+.2f] | %.4f | %.4f | %d/%d |"
             % (s["pooled_delta_pp"], s["ci95_pp"][0], s["ci95_pp"][1],
                s["p_bootstrap"], s["p_mcnemar"], s["only_s0_correct"],
                s["only_exit_correct"]))
    e = rep["s3_exit"]
    L += ["", "## S3 exit behaviour at theta = 0.5",
          "Exit rate %.1f%%-%.1f%% (mean %.1f%%); exit changes %d-%d of %d "
          "decisions per seed; aux = deep on %d-%d."
          % (e["exit_rate_min"] * 100, e["exit_rate_max"] * 100,
             e["exit_rate_mean"] * 100, e["exit_ne_deep_min"],
             e["exit_ne_deep_max"], rep["n"], e["aux_eq_deep_min"],
             e["aux_eq_deep_max"])]
    md = "\n".join(L) + "\n"
    with open(os.path.join(C.RESULTS, "agnews_full.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    sys.exit(main())
