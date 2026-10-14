"""Stage B: the eight student systems S0-S7 on LD-100K.

                DA   single-teacher   dual-teacher   dual head   lr
    S0           -         -               -            -       5e-5   baseline TinyBERT
    S1           yes       -               -            -       5e-5   + data augmentation
    S2           -         yes             -            -       5e-5   + single-teacher distillation
    S3           -         -               -            yes     5e-5   + dual head
    S4           yes       yes             -            -       5e-5   + DA + single teacher
    S5           -         -               yes          -       4e-5   + dual-teacher distillation
    S6           yes       -               yes          -       4e-5   + dual teacher + DA
    S7           yes       -               yes          yes     4e-5   + dual teacher + DA + dual head

Only S3 and S7 change the inference graph; S0-S2 and S4-S6 are training-time
interventions that leave the deployed network byte-identical to the baseline.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification

import common as C
import augment as A

LAM = dict(pred=1.0, hid=1.0, attn=1.0, mlm=0.5, aux=1.0, cons=1.0)

# Learning rate.  Every system is trained with the same optimiser settings, so
# the comparison isolates the architectural and objective differences rather
# than the result of per-system tuning.  lr = 1e-4 is therefore the single
# value used in the paper; the per-entry "lr" keys below are kept only so that
# an individual run can be overridden, and the run scripts pass --lr 1e-4
# (which overwrites all of them, see the --lr handling in main()).
LR = 1e-4

SYSTEMS = [
    dict(tag="S0", da=False, single=False, dual=False, dual_head=False, lr=LR),
    dict(tag="S1", da=True, single=False, dual=False, dual_head=False, lr=LR),
    dict(tag="S2", da=False, single=True, dual=False, dual_head=False, lr=LR),
    dict(tag="S3", da=False, single=False, dual=False, dual_head=True, lr=LR),
    dict(tag="S4", da=True, single=True, dual=False, dual_head=False, lr=LR),
    dict(tag="S5", da=False, single=False, dual=True, dual_head=False, lr=LR),
    dict(tag="S6", da=True, single=False, dual=True, dual_head=False, lr=LR),
    dict(tag="S7", da=True, single=False, dual=True, dual_head=True, lr=LR),
    # S8 is not one of the eight main systems: it is the extra configuration
    # added after the main comparison to test whether the dual head still pays
    # off when the dual-teacher (DD) ingredient is removed -- i.e. S4 + DH.
    dict(tag="S8", da=True, single=True, dual=False, dual_head=True, lr=LR),
]

FINAL_TAG = "S7"

# --------------------------------------------------------------------------
# Ablations.  Each one removes exactly one ingredient from the full model and
# is otherwise identical to it, which is what makes the difference
# attributable.  A1-A4 are S6 minus one distillation term (or minus the
# auxiliary MLM objective); A5 is S7 minus the DA stage.
# --------------------------------------------------------------------------
ABLATIONS = [
    dict(tag="A1", da=True, single=False, dual=True, dual_head=False, lr=LR,
         use_attn=False),
    dict(tag="A2", da=True, single=False, dual=True, dual_head=False, lr=LR,
         use_hid=False),
    dict(tag="A3", da=True, single=False, dual=True, dual_head=False, lr=LR,
         use_pred=False),
    dict(tag="A4", da=True, single=False, dual=True, dual_head=False, lr=LR,
         use_mlm=False),
    dict(tag="A5", da=False, single=False, dual=True, dual_head=True, lr=LR),
]

ABLATION_OF = {a["tag"]: "S6" if a["tag"] != "A5" else "S7" for a in ABLATIONS}
ABLATION_REMOVES = {
    "A1": "attention-matrix distillation ($L_{attn}$)",
    "A2": "hidden-state distillation ($L_{hid}$)",
    "A3": "prediction-layer distillation ($L_{pred}$)",
    "A4": "auxiliary masked-LM objective ($L_{mlm}$)",
    "A5": "data augmentation",
}
ABLATION_LABELS = {
    "A1": "S6 $-$ $L_{attn}$",
    "A2": "S6 $-$ $L_{hid}$",
    "A3": "S6 $-$ $L_{pred}$",
    "A4": "S6 $-$ $L_{mlm}$",
    "A5": "S7 $-$ DA",
}


# --------------------------------------------------------------------------
# Follow-up configurations, staged after the main comparison in response to
# review.  They live in their own results file (followups.json) so that no
# already-published number has to be recomputed.
#
#   A0          the baseline without the auxiliary masked-LM objective: the
#               S0-level counterpart of A4, which removes the same term from
#               S6.  Separates the effect of L_mlm from augmentation and
#               distillation.
#   F000-F100   the dual-teacher configuration with the gate pinned to a
#               constant alpha instead of learned.  alpha is the weight on the
#               English teacher, so F000 distils from the mBERT target alone,
#               F100 from the English target alone, and F050 their equal mix --
#               the value every learned gate converges to.  This answers
#               whether alpha ~ 0.5 is an optimum or simply the fixed point at
#               which the gate stops moving.
#   S6eq        the dual-teacher configuration with two *equal-strength*
#               teachers -- two independent fine-tunings of BERT-base -- in
#               place of the unequal (BERT-base, mBERT) pair.  If the dual
#               teacher still fails to pay here, the failure is caused by
#               having two teachers rather than by the strength mismatch.
FOLLOWUPS = [
    dict(tag="A0", da=False, single=False, dual=False, dual_head=False, lr=LR,
         use_mlm=False),
]
FOLLOWUPS += [
    dict(tag=f"F{int(round(a * 100)):03d}", da=True, single=False, dual=True,
         dual_head=False, lr=LR, fixed_alpha=a)
    for a in (0.0, 0.25, 0.5, 0.75, 1.0)
]
FOLLOWUPS.append(
    dict(tag="S6eq", da=True, single=False, dual=True, dual_head=False, lr=LR,
         dual_b2=True))


# --------------------------------------------------------------------------
# Supplementary configuration staged after the main comparison.
#
#   S9 = S8 - L_mlm   (augmentation + single-teacher distillation + dual head,
#                      with the auxiliary masked-LM objective removed)
#
# This is the one configuration the paper's own results jointly point to and
# that had never been run.  Three independent findings converge on it: A4 shows
# that dropping L_mlm from S6 is the only ablation that raises student-side
# accuracy (Section 4.8); S8 shows that the dual-teacher ingredient can be
# dropped once the dual head is present, at no measurable accuracy cost; and the
# dual head itself is accuracy-neutral on this corpus.  S9 applies all three at
# once, so
#   * S9 - S8    isolates L_mlm inside the dual-head family,
#   * S9 - A4    isolates the single-teacher recipe + dual head at fixed (no
#                L_mlm), i.e. what the deployed head costs in accuracy when the
#                harmful regulariser is already gone,
#   * S9 - S7    is the dual-teacher (DD) ingredient on top of S9.
# It uses exactly the S8 recipe otherwise (same augmentation variants, same
# teacher, same seeds), so the single difference is use_mlm=False.
SUPPLEMENTARY = [
    dict(tag="S9", da=True, single=True, dual=False, dual_head=True, lr=LR,
         use_mlm=False),
    # Round 75 (Applied Intelligence revision, R1's second major point).
    # A1/A2/A3 repeat the intermediate-distillation ablation on top of S6,
    # i.e. *with* augmentation.  B1-B3 repeat it on top of S5, which is the
    # same dual-teacher configuration with augmentation switched off, so that
    # "the three intermediate signals are individually removable" can be
    # separated from "120 un-augmented paragraphs do not support the extra
    # supervision".
    dict(tag="B1", da=False, single=False, dual=True, dual_head=False, lr=LR,
         use_attn=False),
    dict(tag="B2", da=False, single=False, dual=True, dual_head=False, lr=LR,
         use_hid=False),
    dict(tag="B3", da=False, single=False, dual=True, dual_head=False, lr=LR,
         use_pred=False),
    # Round 76 (Applied Intelligence revision, lambda_c ablation).
    # S7nc = S7 with the consistency term switched off (lambda_c = 0): the
    # auxiliary head keeps its own cross-entropy but no longer receives the
    # KL agreement signal against the detached main head.  Isolates the one
    # ingredient the paper had never ablated (L_DH = CE(aux) + lambda_c*KL).
    dict(tag="S7nc", da=True, single=False, dual=True, dual_head=True, lr=LR,
         cons_lambda=0.0),
]


# --------------------------------------------------------------------------
# Fixed-depth control.  The deployed artefact of the dual-head systems is the
# first half of the encoder plus a classifier (Section 4.6: the exit is not
# adaptive).  The natural control is therefore the *same recipe* with the
# deployed classifier moved onto the layer-2 hidden state and no auxiliary head
# and no consistency term -- i.e. a genuine 2-layer deployment with the 4-layer
# backbone retained for training.  `head_on_layer = 2` makes `TinyStudent` read
# hidden_states[2] in the main head, which is the output of the first two
# layers; at inference `forward_fixed_depth` runs exactly those two layers.
# If this matches S7, the dual head is a convenience rather than a mechanism.
FIXED_DEPTH = [
    dict(tag="F2", da=True, single=False, dual=True, dual_head=False,
         head_on_layer=2, lr=LR),
]


# --------------------------------------------------------------------------
def encode(tokenizer, texts, max_len=None, pad="max_length"):
    """`pad="longest"` is used for training batches: the paragraphs in LD-100K
    are 6-86 word-pieces long, so padding every batch to a flat 96 wastes about
    a third of the compute.  `max_len` defaults to `C.MAX_LEN` read at *call*
    time so that a runtime `--max-len` override actually applies."""
    max_len = C.MAX_LEN if max_len is None else max_len
    return tokenizer(list(texts), padding=pad, truncation=True,
                     max_length=max_len, return_tensors="pt")


def batches_from_enc(enc, labels, batch_size, shuffle, generator=None):
    n = len(labels)
    if shuffle:
        order = torch.randperm(n, generator=generator)
    else:
        order = torch.arange(n)
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        yield {k: v[idx] for k, v in enc.items()}, labels[idx]


class TeacherPack:
    """Both fine-tuned teachers, frozen, kept on the GPU.

    `dual_source` selects the second teacher: "mBERT" (the default, the
    published configuration) or the tag of another fine-tuned checkpoint under
    models/ -- used for the equal-strength-pair control, where both teachers
    are independent fine-tunings of BERT-base.
    """

    def __init__(self, device, need_dual=True, dual_source="mBERT",
                 models_dir=None):
        self.device = device
        self.models_dir = models_dir or C.MODELS
        self.tok_multi = AutoTokenizer.from_pretrained(C.resolve(C.TEACHER_MULTI))
        self.en = BertForSequenceClassification.from_pretrained(
            os.path.join(self.models_dir, "BERT-base"),
            attn_implementation="eager").to(device).eval()
        for p in self.en.parameters():
            p.requires_grad_(False)
        self.multi = None
        self.tok_dual = None
        if need_dual:
            if dual_source == "mBERT":
                # the published configuration, kept byte-for-byte as before
                self.tok_dual = self.tok_multi
                dual_dir = os.path.join(self.models_dir, "mBERT")
            else:
                # an alternative second teacher (the equal-strength pair): a
                # fine-tuned checkpoint carries its own tokeniser next to the
                # weights, so the pair is always consistent.
                dual_dir = os.path.join(self.models_dir, dual_source)
                self.tok_dual = AutoTokenizer.from_pretrained(dual_dir)
            self.multi = BertForSequenceClassification.from_pretrained(
                dual_dir, attn_implementation="eager").to(device).eval()
            for p in self.multi.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def en_targets(self, input_ids, attention_mask, want_attn=True):
        out = self.en(input_ids=input_ids, attention_mask=attention_mask,
                      output_hidden_states=True, output_attentions=want_attn)
        return out.logits, out.hidden_states, out.attentions

    @torch.no_grad()
    def multi_logits(self, texts):
        enc = self.tok_dual(list(texts), padding="max_length", truncation=True,
                            max_length=C.MAX_LEN, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return self.multi(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"]).logits


# --------------------------------------------------------------------------
def train_one(cfg, seed, records_train, test_batches, test_mlm, tokenizer,
              device, epochs, batch_size, teachers, verbose=True):
    C.set_seed(seed)
    head_on_layer = cfg.get("head_on_layer")
    if head_on_layer is not None and cfg["dual_head"]:
        raise SystemExit("head_on_layer and dual_head are mutually exclusive")
    student = C.TinyStudent(seed=seed, head_layer=head_on_layer).to(device)
    n_layers = student.num_layers
    projs = nn.ModuleList([
        nn.Linear(student.hidden_size, teachers.en.config.hidden_size)
        for _ in range(n_layers)]).to(device) if (cfg["single"] or cfg["dual"]) else None
    # The gate of the dual-teacher systems is learned unless the configuration
    # pins it (cfg["fixed_alpha"]), in which case it is a constant and takes no
    # part in the optimisation -- this is what the fixed-alpha sweep varies.
    fixed_alpha = cfg.get("fixed_alpha")
    alpha_raw = (nn.Parameter(torch.zeros(1, device=device))
                 if (cfg["dual"] and fixed_alpha is None) else None)

    params = list(student.parameters())
    if projs is not None:
        params += list(projs.parameters())
    if alpha_raw is not None:
        params += [alpha_raw]

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01,
                            betas=(0.9, 0.999), eps=1e-8)
    steps_per_epoch = max(1, len(records_train) // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * 0.1))

    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        prog = (step - warmup) / float(max(1, total_steps - warmup))
        return max(0.0, 1.0 - prog)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ce = nn.CrossEntropyLoss()

    texts = [r["text"] for r in records_train]
    labels = torch.tensor([r["label_id"] for r in records_train])
    adv_generator = torch.Generator().manual_seed(seed)

    dist_scale = None
    dv = None
    best = {"dev_acc": -1.0, "state": None, "epoch": -1}
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    epoch_log = []
    for ep in range(1, epochs + 1):
        student.train()
        order = torch.randperm(len(texts), generator=adv_generator)
        run_loss, run_acc, nseen = 0.0, 0.0, 0
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            sel = [texts[j] for j in idx.tolist()]
            y = labels[idx].to(device)
            enc = encode(tokenizer, sel, pad="longest")
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)

            mlm_ids, mlm_lab = C.make_mlm_inputs(ids, am, tokenizer,
                                                 generator=adv_generator)
            mlm_ids = mlm_ids.to(device)
            mlm_lab = mlm_lab.to(device)

            use_mlm = bool(cfg.get("use_mlm", True))
            use_pred = bool(cfg.get("use_pred", True))
            use_hid = bool(cfg.get("use_hid", True))
            use_attn = bool(cfg.get("use_attn", True))
            distill = bool(cfg["single"] or cfg["dual"])
            need_attn = distill and use_attn

            out = student(ids, am,
                          mlm_input_ids=mlm_ids if use_mlm else None,
                          want_attn=need_attn)

            loss = ce(out["main"], y)
            if out["mlm"] is not None:
                loss = loss + LAM["mlm"] * ce(out["mlm"].transpose(1, 2), mlm_lab)

            if distill:
                t_logits, t_hidden, t_attn = teachers.en_targets(
                    ids, am, want_attn=use_attn)
                # ---- gated prediction target
                alpha = None
                if use_pred:
                    if cfg["dual"]:
                        if alpha_raw is not None:
                            alpha = torch.sigmoid(alpha_raw)
                        else:
                            alpha = torch.as_tensor(float(fixed_alpha),
                                                    device=device)
                        m_logits = teachers.multi_logits(sel)
                        t_probs = C.gated_teacher_probs(t_logits, m_logits, alpha)
                    else:
                        t_probs = C.soft_probs(t_logits)

                raw = {}
                if use_pred:
                    raw["pred"] = C.pred_kl(out["main"], t_probs)
                if use_hid:
                    hid_terms = [F.mse_loss(projs[si](out["hiddens"][si]),
                                            t_hidden[C.LAYER_MAP[si]].detach())
                                 for si in range(n_layers)]
                    raw["hid"] = torch.stack(hid_terms).mean()
                if need_attn:
                    attn_terms = [F.mse_loss(out["attns"][si],
                                             t_attn[C.ATTN_MAP[si]].detach())
                                  for si in range(n_layers)]
                    raw["attn"] = torch.stack(attn_terms).mean()
                # Balance the distillation terms by their magnitude on the
                # first batch: raw hidden-state MSE is O(10-100) while the
                # cross-entropies are O(1), so an unweighted sum would let the
                # fitnet term swamp the task gradient. The constants are fixed
                # after the first step and never updated again.
                if raw:
                    if dist_scale is None:
                        dist_scale = {k: max(float(v.detach()), 1e-6)
                                      for k, v in raw.items()}
                    for k, v in raw.items():
                        loss = loss + LAM[k] * v / dist_scale[k]

            if cfg["dual_head"]:
                loss = loss + LAM["aux"] * ce(out["aux"], y)
                with torch.no_grad():
                    target = F.softmax(out["main"], dim=-1)
                loss = loss + cfg.get("cons_lambda", 1.0) * LAM["cons"] \
                    * C.pred_kl(out["aux"], target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            run_loss += float(loss.item()) * y.shape[0]
            run_acc += float((out["main"].argmax(-1) == y).float().sum().item())
            nseen += y.shape[0]

        dv = evaluate(student, test_batches["dev"], batch_size)
        epoch_log.append({"epoch": ep, "train_loss": run_loss / max(1, nseen),
                          "train_acc": run_acc / max(1, nseen),
                          "dev_acc": dv["acc"], "dev_macro_f1": dv["macro_f1"]})
        if dv["acc"] > best["dev_acc"]:
            best = {"dev_acc": dv["acc"], "dev_macro_f1": dv["macro_f1"],
                    "epoch": ep,
                    "state": copy.deepcopy({k: v.detach().cpu()
                                            for k, v in student.state_dict().items()})}
        if verbose:
            print(f"    epoch {ep:02d}/{epochs} loss={run_loss / max(1, nseen):.4f} "
                  f"train={run_acc / max(1, nseen):.4f} dev={dv['acc']:.4f}", flush=True)

    train_time = time.perf_counter() - t0
    peak_train = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device.type == "cuda" else 0.0
    student.load_state_dict(best["state"])
    student.to(device).eval()

    # ---- test metrics ----------------------------------------------------
    res = evaluate(student, test_batches["test"], batch_size)
    probs, gold = test_predictions(student, test_batches["test"], batch_size)
    mlm_top1, mlm_top5 = mlm_eval(student, test_mlm, batch_size, tokenizer, device)

    # ---- inference throughput + early-exit statistics --------------------
    tb = test_batches["test"]

    def _ids(b):
        return b[0]["input_ids"].to(device), b[0]["attention_mask"].to(device)

    if cfg["dual_head"]:
        def fn(b):
            i, a = _ids(b)
            student.forward_with_exit(i, a, C.EXIT_THETA)
        with torch.no_grad():
            pps, peak_speed = C.measure_throughput(fn, tb, warmup=2, repeats=3)
        # theta = -1 disables the exit, isolating the loop overhead from the
        # genuine saving; both are reported so the two effects are separable.
        def fn_noexit(b):
            i, a = _ids(b)
            student.forward_with_exit(i, a, -1.0)
        with torch.no_grad():
            pps_noexit, _ = C.measure_throughput(fn_noexit, tb, warmup=2, repeats=3)
        with torch.no_grad():
            pps_hf, _ = C.measure_throughput(
                lambda b: student(*_ids(b)), tb, warmup=2, repeats=3)
        exit_rate = exit_statistics(student, tb, device)
    elif head_on_layer is not None:
        # fixed-depth deployment: the two-layer path is the whole model, so
        # the timer measures exactly what a serving stack would run.
        def fn(b):
            i, a = _ids(b)
            student.forward_fixed_depth(i, a)
        with torch.no_grad():
            pps, peak_speed = C.measure_throughput(fn, tb, warmup=2, repeats=3)
        with torch.no_grad():
            pps_hf, _ = C.measure_throughput(
                lambda b: student(*_ids(b)), tb, warmup=2, repeats=3)
        exit_rate = 1.0        # every window is scored by the shallow head
        pps_noexit = None
    else:
        def fn(b):
            return student(*_ids(b))
        with torch.no_grad():
            pps, peak_speed = C.measure_throughput(fn, tb, warmup=2, repeats=3)
        exit_rate = 0.0
        pps_noexit = None
        pps_hf = None

    out = {
        "tag": cfg["tag"], "seed": seed, "role": "student",
        "da": cfg["da"], "single_teacher": cfg["single"], "dual_teacher": cfg["dual"],
        "dual_head": cfg["dual_head"], "lr": cfg["lr"],
        "cons_lambda": float(cfg.get("cons_lambda", 1.0)),
        "head_on_layer": head_on_layer,
        "fixed_depth": bool(head_on_layer is not None),
        "use_attn": bool(cfg.get("use_attn", True)),
        "use_hid": bool(cfg.get("use_hid", True)),
        "use_pred": bool(cfg.get("use_pred", True)),
        "use_mlm": bool(cfg.get("use_mlm", True)),
        "test_probs": [[round(float(v), 5) for v in row] for row in probs.tolist()],
        "test_gold": [int(v) for v in gold.tolist()],
        "params_m": round(C.count_params(student) / 1e6, 2),
        "train_size": len(records_train),
        "epochs": epochs, "batch_size": batch_size,
        "warmup_steps": warmup, "total_steps": total_steps,
        "train_time_s": round(train_time, 1),
        "best_epoch": best["epoch"], "dev_acc": round(best["dev_acc"], 4),
        "dev_macro_f1": round(best["dev_macro_f1"], 4),
        "test_acc": round(res["acc"], 4), "test_macro_f1": round(res["macro_f1"], 4),
        "mlm_top1": round(mlm_top1, 4), "mlm_top5": round(mlm_top5, 4),
        "passages_per_s": round(pps, 2),
        "passages_per_s_no_exit": round(pps_noexit, 2) if pps_noexit else None,
        "passages_per_s_hf_path": round(pps_hf, 2) if pps_hf else None,
        "exit_rate": round(exit_rate, 4),
        # shallow/deep head agreement (review Section 6.6, item (c)); None for
        # systems without an auxiliary head.
        "head_agreement": (round(head_agreement(student, test_batches["test"]), 4)
                           if cfg["dual_head"] else None),
        "peak_train_vram_mb": round(peak_train, 1),
        "speed_peak_vram_mb": round(peak_speed, 1),
        "dist_scale": {k: round(v, 4) for k, v in (dist_scale or {}).items()},
        "mlm_head_pretrained": bool(student.mlm_pretrained),
        "encoder_max_abs_diff_vs_hf": round(student.verify_split_encoder(
            tb[0][0]["input_ids"].to(device), tb[0][0]["attention_mask"].to(device)), 9),
        "epoch_log": epoch_log,
    }
    if cfg["dual"] and bool(cfg.get("use_pred", True)):
        if alpha_raw is not None:
            out["alpha"] = round(float(torch.sigmoid(alpha_raw).item()), 4)
        else:
            out["alpha"] = round(float(fixed_alpha), 4)
            out["alpha_fixed"] = True
    out["dual_teacher_source"] = ("BERT-base-b2" if cfg.get("dual_b2")
                                  else ("mBERT" if cfg["dual"] else None))
    return out, student, (alpha_raw, projs)


def test_predictions(student, batches, batch_size=None):
    """Per-paragraph test probabilities -- needed for the paired significance
    tests in the ablation study."""
    dev = next(student.parameters()).device
    student.eval()
    P, Y = [], []
    with torch.no_grad():
        for b, y in batches:
            out = student(b["input_ids"].to(dev), b["attention_mask"].to(dev))
            P.append(F.softmax(out["main"].float(), dim=-1).cpu())
            Y.append(y.cpu())
    return torch.cat(P), torch.cat(Y)


def head_agreement(student, batches):
    """Fraction of test windows on which the shallow (auxiliary) head and the
    deep (main) head agree -- the diagnostic the review asks for alongside the
    exit rate (Section 6.6).  Only meaningful for dual-head systems."""
    dev = next(student.parameters()).device
    student.eval()
    agree = total = 0
    with torch.no_grad():
        for b, _ in batches:
            out = student(b["input_ids"].to(dev), b["attention_mask"].to(dev))
            a = out["aux"].argmax(-1)
            m = out["main"].argmax(-1)
            agree += int((a == m).sum().item())
            total += int(a.numel())
    return agree / max(1, total)


def evaluate(student, batches, batch_size=None):
    dev = next(student.parameters()).device
    student.eval()
    logits, labels = [], []
    with torch.no_grad():
        for b, y in batches:
            out = student(b["input_ids"].to(dev), b["attention_mask"].to(dev))
            logits.append(out["main"].float().cpu())
            labels.append(y.cpu())
    L = torch.cat(logits)
    Y = torch.cat(labels)
    return {"acc": C.accuracy(L, Y), "macro_f1": C.macro_f1(L, Y)}


def mlm_eval(student, test_mlm, batch_size, tokenizer, device):
    student.eval()
    logits, labels = [], []
    with torch.no_grad():
        for i in range(0, len(test_mlm["input_ids"]), batch_size):
            ids = test_mlm["input_ids"][i:i + batch_size].to(device)
            am = test_mlm["attention_mask"][i:i + batch_size].to(device)
            mid = test_mlm["mlm_input_ids"][i:i + batch_size].to(device)
            lab = test_mlm["mlm_labels"][i:i + batch_size].to(device)
            out = student(ids, am, mlm_input_ids=mid)
            logits.append(out["mlm"].float().cpu())
            labels.append(lab.cpu())
    L = torch.cat(logits)
    Y = torch.cat(labels)
    return C.mlm_scores(L, Y)


@torch.no_grad()
def exit_statistics(student, batches, device, theta=C.EXIT_THETA):
    total, exited = 0, 0
    for b, _ in batches:
        _, conf = student.forward_with_exit(b["input_ids"].to(device),
                                            b["attention_mask"].to(device), theta)
        total += conf.numel()
        exited += int(conf.sum().item())
    return exited / max(1, total)


def build_test_bundles(records, tokenizer, device, batch_size, dev_batch_size=16):
    te = C.split_of(records, "test")
    dv = C.split_of(records, "dev")
    enc_te = encode(tokenizer, [r["text"] for r in te])
    y_te = torch.tensor([r["label_id"] for r in te])
    enc_dv = encode(tokenizer, [r["text"] for r in dv])
    y_dv = torch.tensor([r["label_id"] for r in dv])
    g = torch.Generator().manual_seed(4242)
    mlm_ids, mlm_lab = C.make_mlm_inputs(enc_te["input_ids"], enc_te["attention_mask"],
                                         tokenizer, generator=g)
    test_mlm = {"input_ids": enc_te["input_ids"], "attention_mask": enc_te["attention_mask"],
                "mlm_input_ids": mlm_ids, "mlm_labels": mlm_lab}
    return {
        "test": list(batches_from_enc(enc_te, y_te, batch_size, False)),
        "dev": list(batches_from_enc(enc_dv, y_dv, dev_batch_size, False)),
    }, test_mlm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[13])
    ap.add_argument("--systems", nargs="+", default=[s["tag"] for s in SYSTEMS])
    ap.add_argument("--aug-variants", type=int, default=3)
    ap.add_argument("--lr", type=float, default=None,
                    help="override the learning rate of every system")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--out", default=None,
                    help="output json name inside results/ (default students.json)")
    ap.add_argument("--resume", action="store_true",
                    help="skip (system, seed) pairs already present in the output json")
    ap.add_argument("--data", default=None,
                    help="ptc-style jsonl corpus (default: data/ptc.jsonl = LD-100K)")
    ap.add_argument("--models-subdir", default="",
                    help="read teachers and save checkpoints under models/<subdir>/ "
                         "(organic replication); default = models/")
    ap.add_argument("--max-len", type=int, default=None,
                    help="override C.MAX_LEN (organic corpus: 112 keeps the "
                         "'no truncation' property of the protocol)")
    ap.add_argument("--num-labels", type=int, default=None,
                    help="override C.NUM_LABELS (the public-benchmark "
                         "replication uses four classes)")
    args = ap.parse_args()
    if args.max_len:
        C.MAX_LEN = args.max_len
    if args.num_labels:
        C.NUM_LABELS = args.num_labels

    pool = {c["tag"]: c for c in
            SYSTEMS + ABLATIONS + FOLLOWUPS + FIXED_DEPTH + SUPPLEMENTARY}
    unknown = [t for t in args.systems if t not in pool]
    if unknown:
        ap.error(f"unknown system(s): {unknown} "
                 f"(known: {sorted(pool)})")
    sel = [pool[t] for t in args.systems]
    if args.lr is not None:
        for c in sel:
            c["lr"] = args.lr
    if args.tag_suffix:
        for c in sel:
            c["tag"] = c["tag"] + args.tag_suffix
    device = C.get_device()
    print("device:", device, flush=True)
    records = C.load_ptc(args.data)
    models_dir = os.path.join(C.MODELS, args.models_subdir) if args.models_subdir else C.MODELS
    tokenizer = AutoTokenizer.from_pretrained(C.resolve(C.TOKENIZER_EN))

    test_batches, test_mlm = build_test_bundles(records, tokenizer, device,
                                                args.batch_size)
    print(f"test windows: {len(test_batches['test'])}  dev windows: {len(test_batches['dev'])}",
          flush=True)

    base_train = C.split_of(records, "train")
    needs_da = any(s["da"] for s in sel)
    aug_train = None
    if needs_da:
        print("building augmented training split (one-off) ...", flush=True)
        from transformers import BertForMaskedLM
        mlm_model = BertForMaskedLM.from_pretrained(
            C.resolve(C.TEACHER_EN), attn_implementation="eager").to(device).eval()
        t_aug = time.perf_counter()
        aug_train = A.build_variants(base_train, tokenizer, mlm_model, device,
                                     variants=args.aug_variants, seed=7)
        aug_time = time.perf_counter() - t_aug
        print(f"  augmented train size: {len(aug_train)}  ({aug_time:.1f}s)", flush=True)
        del mlm_model
        torch.cuda.empty_cache()
        aug_name = (f"augmentation_{args.models_subdir}.json"
                    if args.models_subdir else "augmentation.json")
        C.json_dump({"base_train": len(base_train), "augmented_train": len(aug_train),
                     "variants_per_paragraph": args.aug_variants,
                     "augmentation_seconds": round(aug_time, 1)},
                    os.path.join(C.RESULTS, aug_name))

    teachers = None
    all_results = []

    if args.out:
        out_name = args.out
    else:
        out_name = f"students{args.tag_suffix}.json" if args.tag_suffix else "students.json"
    out_path = os.path.join(C.RESULTS, out_name)

    # ---- resumable bookkeeping ------------------------------------------
    # A full sweep is ~2-4 h on one 6 GB laptop GPU, so every finished
    # (system, seed) pair is flushed to disk immediately.  With --resume the
    # run picks up exactly where it stopped instead of restarting.
    done = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            all_results = json.load(f)
        done = {(r["tag"], r["seed"]) for r in all_results}
        print(f"resume: {len(done)} run(s) already on disk", flush=True)

    final_state = None
    for cfg in sel:
        need_teacher = cfg["single"] or cfg["dual"]
        if need_teacher and teachers is None:
            sources = {bool(s.get("dual_b2", False)) for s in sel if s["dual"]}
            if len(sources) > 1:
                raise SystemExit(
                    "cannot mix the equal-strength pair and the "
                    "(BERT-base, mBERT) pair in one run -- split --systems")
            teachers = TeacherPack(
                device, need_dual=any(s["dual"] for s in sel),
                dual_source=("BERT-base-b2" if sources == {True} else "mBERT"),
                models_dir=models_dir)
        train_records = aug_train if cfg["da"] else base_train
        for seed in args.seeds:
            if (cfg["tag"], seed) in done:
                print(f"--- skip {cfg['tag']} seed {seed} (cached) ---", flush=True)
                continue
            print(f"=== {cfg['tag']} seed {seed} (train={len(train_records)}) ===", flush=True)
            res, student, _ = train_one(cfg, seed, train_records, test_batches,
                                        test_mlm, tokenizer, device, args.epochs,
                                        args.batch_size, teachers)
            all_results.append(res)
            C.json_dump(all_results, out_path)          # flush after every run
            # keep the weights: the efficiency benchmark reloads every system
            # in a single process so that all throughput numbers are taken
            # under identical thermal / clock conditions.
            ckpt_dir = os.path.join(models_dir, "students")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in student.state_dict().items()},
                       os.path.join(ckpt_dir, f"{cfg['tag']}_{seed}.pt"))
            print(json.dumps({k: v for k, v in res.items()
                              if k not in ("epoch_log", "test_probs")},
                             indent=2), flush=True)
            if cfg["tag"] == FINAL_TAG and seed == args.seeds[0]:
                final_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
                os.makedirs(os.path.join(models_dir, FINAL_TAG), exist_ok=True)
                torch.save(final_state, os.path.join(models_dir, FINAL_TAG, "student.pt"))
            del student
            torch.cuda.empty_cache()

    C.json_dump(all_results, out_path)
    print("students done")


if __name__ == "__main__":
    main()
