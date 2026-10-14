"""Summarise the existing per-system results: tag, seed, test_acc, alpha, mlm.

Usage: python peek_results.py [json-name ...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

names = sys.argv[1:] or ["students.json", "ablations.json", "students_S8.json"]

for name in names:
    path = os.path.join(C.RESULTS, name)
    if not os.path.exists(path):
        print(f"--- {name}: MISSING")
        continue
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    print(f"--- {name}: {len(rows)} runs")
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append(r)
    for tag in sorted(by_tag):
        rs = by_tag[tag]
        accs = [x["test_acc"] for x in rs]
        mlms = [x.get("mlm_top1") for x in rs]
        alphas = [x.get("alpha") for x in rs if x.get("alpha") is not None]
        cfg = rs[0]
        print(f"  {tag:>4}  n={len(rs)}  seeds={[x['seed'] for x in rs]}")
        print(f"        da={cfg.get('da')} single={cfg.get('single_teacher')} "
              f"dual={cfg.get('dual_teacher')} dh={cfg.get('dual_head')} "
              f"use_mlm={cfg.get('use_mlm')}")
        print(f"        test_acc={['%.4f' % a for a in accs]}  mean={sum(accs)/len(accs):.4f}")
        if any(m is not None for m in mlms):
            print(f"        mlm_top1={mlms}")
        if alphas:
            print(f"        alpha={alphas}")
