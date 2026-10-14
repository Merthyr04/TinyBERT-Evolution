"""Three lexical augmentations for the LD-100K training split.

  * mask_fill       -- replace a fraction of the words with the top-1 token
                       predicted by an English BERT MLM head.  This is the
                       "mask-fill" operator: the fill comes from a pretrained
                       model, so the variant stays fluent.
  * window_jitter   -- keep a random contiguous run of 70-100% of the
                       sentences.
  * sentence_dropout-- delete one sentence at random.

All three preserve the paragraph topic label by construction, so the
augmented copies inherit the gold label of their source paragraph.
"""
from __future__ import annotations

import random
import re

import torch

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_SPLIT = re.compile(r"\S+")


def _word_spans(text):
    return [(m.start(), m.end()) for m in WORD_SPLIT.finditer(text)]


@torch.no_grad()
def mask_fill(text, tokenizer, mlm_model, device, ratio=0.12, rng=None):
    rng = rng or random
    spans = _word_spans(text)
    if len(spans) < 5:
        return text
    k = max(1, int(round(len(spans) * ratio)))
    chosen = sorted(rng.sample(range(len(spans)), min(k, len(spans))))

    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    max_length=512)
    offsets = enc["offset_mapping"]

    hit = []
    for wi in chosen:
        s, e = spans[wi]
        positions = [j for j, (a, b) in enumerate(offsets) if a < e and b > s]
        positions = [j for j in positions
                     if offsets[j][1] > offsets[j][0]]
        if positions:
            hit.append((wi, positions[0], s, e))

    if not hit:
        return text

    ids = list(enc["input_ids"])
    for _, pos, _, _ in hit:
        ids[pos] = tokenizer.mask_token_id
    inp = torch.tensor([ids], device=device)
    logits = mlm_model(inp).logits[0]
    preds = logits.argmax(-1).tolist()

    out = text
    for wi, pos, s, e in sorted(hit, key=lambda z: -z[2]):
        tok = tokenizer.convert_ids_to_tokens(preds[pos])
        if tok is None or tok in tokenizer.all_special_tokens:
            continue
        word = tok.replace("##", "")
        if not word:
            continue
        out = out[:s] + word + out[e:]
    return out


def window_jitter(text, rng=None, low=0.7, high=1.0):
    rng = rng or random
    sents = [s for s in SENT_SPLIT.split(text) if s.strip()]
    if len(sents) < 2:
        return text
    keep = rng.uniform(low, high)
    n_keep = max(1, int(round(len(sents) * keep)))
    start = rng.randint(0, len(sents) - n_keep)
    return " ".join(sents[start:start + n_keep])


def sentence_dropout(text, rng=None):
    rng = rng or random
    sents = [s for s in SENT_SPLIT.split(text) if s.strip()]
    if len(sents) < 2:
        return text
    drop = rng.randrange(len(sents))
    kept = [s for i, s in enumerate(sents) if i != drop]
    return " ".join(kept)


def build_variants(records, tokenizer, mlm_model, device, variants=3,
                   seed=7, verbose=True):
    """Return a new record list = originals + `variants` per source paragraph."""
    rng = random.Random(seed)
    out = []
    for i, rec in enumerate(records):
        out.append(dict(rec, aug="none"))
        for v in range(variants):
            if v == 0:
                text = mask_fill(rec["text"], tokenizer, mlm_model, device,
                                 ratio=0.12, rng=rng)
                tag = "mask_fill_12"
            elif v == 1:
                text = mask_fill(rec["text"], tokenizer, mlm_model, device,
                                 ratio=0.20, rng=rng)
                tag = "mask_fill_20"
            else:
                if rng.random() < 0.5:
                    text = window_jitter(rec["text"], rng=rng)
                    tag = "window_jitter"
                else:
                    text = sentence_dropout(rec["text"], rng=rng)
                    tag = "sentence_dropout"
            if not text.strip():
                continue
            out.append(dict(rec, text=text, aug=tag))
        if verbose and (i + 1) % 50 == 0:
            print(f"    augmented {i + 1}/{len(records)}", flush=True)
    return out
