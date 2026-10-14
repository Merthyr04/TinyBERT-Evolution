# -*- coding: utf-8 -*-
"""Parse the Nuttall Encyclopaedia volume into entries and report the
structure we need in order to build the organic long-document corpus.

Structure of the volume (verified by inspection):
  * body starts after the Project Gutenberg START banner; the preface ends at
    the line "THE NUTTALL ENCYCLOPAEDIA" (all caps);
  * entries are separated by blank lines;
  * each entry is one paragraph: a run of CAPS headword(s), then a comma or a
    full stop, then a terse definition -- usually one to three sentences;
  * single capital letters ("A", "B", ...) mark the alphabet sections.
"""
from __future__ import annotations

import io
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "organic_raw", "nuttall_encyclopaedia.txt")


def load_body():
    s = io.open(RAW, encoding="utf-8", errors="replace").read()
    i = s.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    i = s.find("\n", i) + 1
    j = s.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    body = s[i:j]
    k = body.find("THE NUTTALL ENCYCLOP")
    if k != -1:
        body = body[k:]
        body = body[body.find("\n"):]
    return body


ALPHA = re.compile(r"^[A-Z]$")
# headword: leading capitals, allowing apostrophes, hyphens, spaces, commas
HEAD = re.compile(r"^([A-Z][A-Z'\-\. ]{1,60}?)(?:,\s|\.\s|,\s*$|\.\s*$)")


def parse_entries(body):
    out = []
    for blk in re.split(r"\n\s*\n", body):
        p = " ".join(blk.split())
        if not p or ALPHA.match(p):
            continue
        if len(p) < 40:
            continue
        m = HEAD.match(p)
        if not m:
            continue
        head = m.group(1).strip(" ,.")
        rest = p[m.end():].strip()
        if len(rest) < 40:
            continue
        out.append({"head": head, "def": rest, "text": p})
    return out


def main():
    entries = parse_entries(load_body())
    lens = [len(e["def"]) for e in entries]
    print(f"parsed entries: {len(entries)}")
    print(f"body-definition length: min {min(lens)}  max {max(lens)}  "
          f"median {statistics.median(lens):.0f}  mean {statistics.mean(lens):.0f}")
    import collections
    buckets = collections.Counter()
    for L in lens:
        if L < 120:
            buckets["<120"] += 1
        elif L < 180:
            buckets["120-180"] += 1
        elif L < 260:
            buckets["180-260"] += 1
        elif L < 400:
            buckets["260-400"] += 1
        elif L < 700:
            buckets["400-700"] += 1
        else:
            buckets[">700"] += 1
    print("length buckets:", dict(buckets))
    for lo, hi in [(150, 400), (180, 500), (200, 600)]:
        sel = [e for e in entries if lo <= len(e["def"]) <= hi]
        print(f"  entries with def in [{lo},{hi}]: {len(sel)}  "
              f"total chars {sum(len(e['text']) for e in sel)}")
    print("\n--- first 6 parsed entries ---")
    for e in entries[:6]:
        print(f"  HEAD={e['head']!r}\n      DEF={e['def'][:150]!r}")
    print("\n--- 3 long entries ---")
    for e in sorted(entries, key=lambda x: -len(x["def"]))[:3]:
        print(f"  HEAD={e['head']!r}\n      DEF={e['def'][:220]!r}")


if __name__ == "__main__":
    main()
