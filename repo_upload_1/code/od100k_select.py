# -*- coding: utf-8 -*-
"""Select the OD-100K paragraph set from the Nuttall volume.

The book is read in order and every entry whose definition is a body-like
sentence group of 150-400 characters is kept; the first `N` of them form the
document.  The window is contiguous in the volume, so OD-100K is a real,
unaltered stretch of a real reference work -- not a selection of topics.

Output: data/organic_entries.jsonl  (one entry per line: idx, head, def, text,
alphabetical rank within the volume).
"""
from __future__ import annotations

import io
import json
import os
import sys

import build_organic as B

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "organic_entries.jsonl")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 416
LO, HI = 150, 400


def main():
    entries = B.parse_entries(B.load_body())
    print(f"parsed {len(entries)} entries from the volume")
    keep = []
    for e in entries:
        if LO <= len(e["def"]) <= HI:
            keep.append(e)
    print(f"body-like entries (def {LO}-{HI} chars): {len(keep)}")
    sel = keep[:N]
    total = sum(len(e["text"]) for e in sel)
    print(f"selected {len(sel)} entries, {total} characters, "
          f"mean {total/len(sel):.0f} chars/entry")
    print(f"headwords: {sel[0]['head']!r} .. {sel[-1]['head']!r}")
    with open(OUT, "w", encoding="utf-8") as f:
        for i, e in enumerate(sel):
            f.write(json.dumps({"idx": i, "head": e["head"],
                                "def": e["def"], "text": e["text"]},
                               ensure_ascii=False) + "\n")
    print("wrote", OUT)
    print("\n--- 8 sampled entries ---")
    for e in sel[::max(1, len(sel) // 8)][:8]:
        print(f"  [{e['head']}] {e['text'][:160]}")


if __name__ == "__main__":
    main()
