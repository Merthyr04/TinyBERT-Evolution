"""Masked-token recovery of the two reference encoders.

Why this is separate from stage_teachers.py
-------------------------------------------
The teachers are trained as sequence classifiers, so their checkpoints contain
no masked-LM head.  To report the masked-token retention of Section 4.6 we
therefore rebuild a BertForMaskedLM around the *fine-tuned* encoder:

    fine-tuned encoder  (from models/<tag>, BertForSequenceClassification)
  + masked-LM head      (from the released checkpoint of the same model)
  + re-tied output decoder -> input word embeddings

Both parts come from the same published model, so nothing is introduced that
was not already in the checkpoint; the only new thing is the combination.

Two numbers are produced per encoder, so that the effect of task fine-tuning can
be separated from the architecture:

    finetuned   encoder after PTC fine-tuning + the released MLM head
    released    the published model, untuned

The masked positions are drawn with the same generator seed as the student
evaluation, so for BERT-base -- which shares the student's tokenizer -- the
teacher and the students are scored on *exactly* the same tokens.  mBERT uses a
different vocabulary and therefore its own tokenizer, which is stated in the
paper.

Outputs results/teacher_mlm.json.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import (AutoTokenizer, BertForMaskedLM,
                          BertForSequenceClassification)

import common as C

# Must match stage_students.build_test_bundles so the BERT-base mask is the
# same object the students were scored on.
MLM_SEED = 4242


def load_json(name):
    p = os.path.join(C.RESULTS, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def mlm_of(model, masked, labels, batch_size, device):
    """Masked-token top-1/top-5 of a BertForMaskedLM.

    The encoder is run on the *masked* ids, exactly as the student evaluation
    does in TinyStudent.forward, so the two are comparable.
    """
    model.eval()
    logits, labs = [], []
    for i in range(0, len(masked["input_ids"]), batch_size):
        am = masked["attention_mask"][i:i + batch_size].to(device)
        mid = masked["mlm_input_ids"][i:i + batch_size].to(device)
        lab = labels[i:i + batch_size].to(device)
        logits.append(model(input_ids=mid, attention_mask=am).logits.float().cpu())
        labs.append(lab.cpu())
    return C.mlm_scores(torch.cat(logits), torch.cat(labs))


def build_masked(texts, tokenizer, max_len=C.MAX_LEN):
    enc = tokenizer(list(texts), padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt")
    g = torch.Generator().manual_seed(MLM_SEED)
    mid, lab = C.make_mlm_inputs(enc["input_ids"], enc["attention_mask"],
                                 tokenizer, generator=g)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "mlm_input_ids": mid}, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="teacher_mlm.json")
    args = ap.parse_args()

    device = C.get_device()
    records = C.load_ptc()
    texts = [r["text"] for r in C.split_of(records, "test")]
    print(f"test paragraphs: {len(texts)}  device: {device}", flush=True)

    out = {"mlm_seed": MLM_SEED, "test_paragraphs": len(texts), "teachers": {}}

    for tag, model_name in (("BERT-base", C.TEACHER_EN),
                            ("mBERT", C.TEACHER_MULTI)):
        tok = AutoTokenizer.from_pretrained(C.resolve(model_name))
        masked, labels = build_masked(texts, tok)
        n_masked = int((labels != -100).sum().item())
        print(f"\n{tag}: {n_masked} masked positions "
              f"(tokenizer {model_name})", flush=True)

        # --- released, untuned model -------------------------------------
        rel = BertForMaskedLM.from_pretrained(C.resolve(model_name)).to(device)
        t1, t5 = mlm_of(rel, masked, labels, args.batch_size, device)
        print(f"  released : top-1 {t1 * 100:.1f}  top-5 {t5 * 100:.1f}",
              flush=True)
        del rel

        # --- fine-tuned encoder + released MLM head ----------------------
        seq = BertForSequenceClassification.from_pretrained(
            os.path.join(C.MODELS, tag)).to(device)
        mlm = BertForMaskedLM.from_pretrained(C.resolve(model_name)).to(device)
        mlm.bert = seq.bert
        mlm.tie_weights()
        dec = mlm.get_output_embeddings().weight
        emb = mlm.bert.embeddings.word_embeddings.weight
        tied = dec.data_ptr() == emb.data_ptr()
        print(f"  weights re-tied: {tied}", flush=True)
        if not tied:
            raise SystemExit(f"{tag}: output decoder is not tied to the "
                             f"encoder's word embeddings; refusing to report a "
                             f"number that would be measuring a random head")
        f1, f5 = mlm_of(mlm, masked, labels, args.batch_size, device)
        print(f"  finetuned: top-1 {f1 * 100:.1f}  top-5 {f5 * 100:.1f}",
              flush=True)
        del mlm, seq
        torch.cuda.empty_cache()

        out["teachers"][tag] = {
            "tokenizer": model_name, "n_masked": n_masked,
            "mlm_top1_released": round(t1, 4), "mlm_top5_released": round(t5, 4),
            "mlm_top1_finetuned": round(f1, 4), "mlm_top5_finetuned": round(f5, 4),
        }

    C.json_dump(out, os.path.join(C.RESULTS, args.out))
    print("\n| Encoder | tokenizer | masked pos. | released top-1 | "
          "fine-tuned top-1 |")
    print("|---|---|---|---|---|")
    for tag, r in out["teachers"].items():
        print(f"| {tag} | {r['tokenizer']} | {r['n_masked']} | "
              f"{r['mlm_top1_released'] * 100:.1f} | "
              f"{r['mlm_top1_finetuned'] * 100:.1f} |")
    print(f"wrote results/{args.out}")


if __name__ == "__main__":
    main()
