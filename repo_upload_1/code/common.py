"""Shared models, objectives, metrics and timing utilities.

Everything runs on a single 6 GB RTX 3060 Laptop GPU.  Design notes:

  * max sequence length 128 (the mean LD-100K paragraph is 35 words);
  * teachers are inference-only oracles under torch.no_grad();
  * layer-wise distillation matches student layer i to teacher layer 3i, the
    standard TinyBERT mapping for a 4-layer student;
  * attention / hidden-state distillation is taken from the English teacher
    only.  Attention matrices are index-aligned objects and a 119k-entry
    multilingual vocabulary does not tokenise English text identically to a
    30k English vocabulary, so cross-tokeniser attention alignment would be
    unsound.  The multilingual teacher therefore contributes through the
    gated prediction-layer soft target, where only the 12-way class posterior
    is compared and token alignment is irrelevant;
  * `attn_implementation="eager"` is requested explicitly: the fused SDPA path
    in transformers 5.x builds its own mask layout and does not return
    attention weights in the same way, and this work needs the weights.

The early exit is implemented with two *real* 2-layer BertModels that share the
student's own layer objects (head = embeddings + layers 0-1, tail = layers
2-3).  Because both are ordinary BertModels, mask construction is delegated to
the library, and because the layers are shared rather than copied, skipping the
tail is a genuine compute saving rather than a simulation.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)          # E:/Second Paper
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")
RESULTS = os.path.join(HERE, "results")
HF_ROOT = os.environ.get("HF_MODELS_DIR", os.path.join(PROJECT, "hf_models"))


def resolve(name):
    """Prefer the locally prefetched checkpoint, else fall back to the hub id."""
    local = os.path.join(HF_ROOT, name.split("/")[-1])
    return local if os.path.isdir(local) else name


STUDENT_NAME = "huawei-noah/TinyBERT_General_4L_312D"
TEACHER_EN = "bert-base-uncased"
TEACHER_MULTI = "bert-base-multilingual-cased"
TOKENIZER_EN = "bert-base-uncased"

MAX_LEN = 96      # longest paragraph in LD-100K is 86 word-pieces: no truncation,
                  # 25% cheaper than a flat 128
NUM_LABELS = 12
LAYER_MAP = [3, 6, 9, 12]      # student layer i (1..4)  <- teacher layer 3i
ATTN_MAP = [2, 5, 8, 11]       # student layer i (0..3)  <- teacher layer 3i+2
TEMPERATURE = 2.0
EXIT_THETA = 0.8
DEFAULT_SEEDS = (13, 37, 97)


# --------------------------------------------------------------------------
# reproducibility / device
# --------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_ptc(path=None):
    """Load a ptc-style corpus (jsonl).  Defaults to LD-100K (data/ptc.jsonl);
    an explicit path is used by the organic-corpus replication, which reuses the
    identical pipeline on data/organic_ptc.jsonl."""
    path = path or os.path.join(DATA, "ptc.jsonl")
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def split_of(recs, name):
    return [r for r in recs if r["split"] == name]


def make_mlm_inputs(input_ids, attention_mask, tokenizer, mask_prob=0.15,
                    generator=None):
    """Return (masked_input_ids, mlm_labels); labels are -100 where unmasked.

    Masking is done on the CPU because `generator` is a CPU generator; the
    result is moved back to the input's device.
    """
    dev = input_ids.device
    ids = input_ids.cpu()
    g = generator
    labels = ids.clone()
    special = ((ids == tokenizer.cls_token_id)
               | (ids == tokenizer.sep_token_id)
               | (ids == tokenizer.pad_token_id)
               | (ids == tokenizer.mask_token_id))
    prob = torch.full(ids.shape, float(mask_prob), dtype=torch.float)
    prob[special] = 0.0
    selected = torch.bernoulli(prob, generator=g).bool()
    if not bool(selected.any()):
        cand = (~special).nonzero(as_tuple=False)
        if len(cand):
            selected[cand[0, 0], cand[0, 1]] = True

    labels[~selected] = -100
    rand = torch.rand(ids.shape, generator=g)
    masked = ids.clone()
    masked[selected & (rand < 0.8)] = tokenizer.mask_token_id
    swap = selected & (rand >= 0.8) & (rand < 0.9)
    if bool(swap.any()):
        rnd = torch.randint(0, tokenizer.vocab_size, ids.shape, generator=g)
        masked[swap] = rnd[swap]
    return masked.to(dev), labels.to(dev)


# --------------------------------------------------------------------------
# student
# --------------------------------------------------------------------------
class _FreshMLMHead(nn.Module):
    """Fallback masked-LM head, used only if the checkpoint ships none."""

    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    def forward(self, hidden_states):
        h = self.transform_act_fn(self.dense(hidden_states))
        h = self.LayerNorm(h)
        return self.decoder(h) + self.bias


def _layer_hidden(layer, hidden, mask):
    """BertLayer returns a bare tensor in transformers 5.x and an
    (hidden, attentions) tuple in 4.x; normalise to the hidden state."""
    out = layer(hidden, attention_mask=mask)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class TinyStudent(nn.Module):
    """4-layer TinyBERT + main head + auxiliary (early-exit) head + MLM head.

    `head_layer` moves the *deployed* classifier onto an intermediate hidden
    state, which is what the fixed-depth control of Section 4.9(xiv) needs:
    with head_layer = 2 the classifier reads the output of the first two
    layers and the tail is never run at inference, so the model is a genuine
    2-layer encoder with a 4-layer backbone.  The default (None) keeps the
    published behaviour, classifier on the last layer.
    """

    def __init__(self, name=STUDENT_NAME, num_labels=None, seed=None,
                 head_layer=None):
        super().__init__()
        # Resolved here rather than in the signature so that a caller which
        # rebinds NUM_LABELS (the public-benchmark replication uses four
        # classes instead of twelve) is honoured.
        if num_labels is None:
            num_labels = NUM_LABELS
        from transformers import BertForMaskedLM
        self.lm = BertForMaskedLM.from_pretrained(
            resolve(name), attn_implementation="eager")
        cfg = self.lm.config
        self.config = cfg
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        if seed is not None:
            set_seed(seed)
        self.main_head = nn.Linear(cfg.hidden_size, num_labels)
        self.aux_head = nn.Linear(cfg.hidden_size, num_labels)
        self.num_labels = num_labels
        self.exit_layer_index = max(1, self.num_layers // 2)   # layer 2 of 4
        self.head_layer = head_layer        # None -> classifier on last layer

        self.mlm_head = getattr(self.lm, "cls", None)
        self.mlm_pretrained = self.mlm_head is not None
        if self.mlm_head is None:
            self.mlm_head = _FreshMLMHead(cfg)
            self.mlm_head.decoder.weight = (
                self.lm.bert.embeddings.word_embeddings.weight)


    # ---- full forward ----------------------------------------------------
    def forward(self, input_ids, attention_mask, mlm_input_ids=None,
                want_attn=False):
        out = self.lm.bert(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True,
                           output_attentions=want_attn)
        hs = out.hidden_states                      # len = num_layers + 1
        hl = self.num_layers if self.head_layer is None else self.head_layer
        last = hs[hl]
        main_logits = self.main_head(self.dropout(last[:, 0]))
        aux_logits = self.aux_head(
            self.dropout(hs[self.exit_layer_index][:, 0]))
        mlm_logits = None
        if mlm_input_ids is not None:
            m = self.lm.bert(input_ids=mlm_input_ids, attention_mask=attention_mask)
            mlm_logits = self.mlm_head(m.last_hidden_state)
        return {"main": main_logits, "aux": aux_logits, "mlm": mlm_logits,
                "hiddens": hs[1:], "attns": out.attentions if want_attn else None}

    # ---- early-exit machinery -------------------------------------------
    def _bidirectional_mask(self, emb, attention_mask):
        """Use the model's own mask construction so the layout always matches
        whatever attention backend is active."""
        if attention_mask is None:
            return None
        bb = self.lm.bert
        if hasattr(bb, "_create_attention_masks"):
            m, _ = bb._create_attention_masks(attention_mask, None, emb, None, None)
            return m
        ext = attention_mask[:, None, None, :].to(emb.dtype)
        return (1.0 - ext) * torch.finfo(emb.dtype).min

    @torch.no_grad()
    def forward_with_exit(self, input_ids, attention_mask, theta=EXIT_THETA):
        """Single-pass forward with a real early exit.

        Layers 1..k are always evaluated, because the auxiliary head needs
        them.  Layers k+1..n are evaluated only for the windows whose
        auxiliary head is still uncertain.  With theta = -1 nothing exits and
        the result is bit-for-bit the full forward.
        """
        bb = self.lm.bert
        layers = bb.encoder.layer
        k = self.exit_layer_index

        h = bb.embeddings(input_ids)
        am = self._bidirectional_mask(h, attention_mask)
        for i in range(k):
            h = _layer_hidden(layers[i], h, am)

        aux_logits = self.aux_head(self.dropout(h[:, 0]))
        conf = normalised_entropy(F.softmax(aux_logits, dim=-1)) < theta

        logits = aux_logits.clone()
        todo = ~conf
        if bool(todo.any()):
            if bool(todo.all()):
                sub, sub_am = h, am          # avoid a pointless gather
            else:
                sub = h[todo]
                sub_am = am[todo] if am is not None else None
            for i in range(k, len(layers)):
                sub = _layer_hidden(layers[i], sub, sub_am)
            if bool(todo.all()):
                logits = self.main_head(self.dropout(sub[:, 0]))
            else:
                logits[todo] = self.main_head(self.dropout(sub[:, 0]))
        return logits, conf

    @torch.no_grad()
    def forward_fixed_depth(self, input_ids, attention_mask, depth=None):
        """Deployed graph of the fixed-depth control: embeddings + the first
        `depth` layers (default: the exit point, i.e. half the encoder) + the
        classifier.  No exit decision is taken -- this is what a model trained
        with the same recipe but with the tail simply not run does at serving
        time, and it uses the same manual layer loop as forward_with_exit so
        that the two timings are implementation-matched.
        """
        bb = self.lm.bert
        layers = bb.encoder.layer
        k = self.exit_layer_index if depth is None else depth
        h = bb.embeddings(input_ids)
        am = self._bidirectional_mask(h, attention_mask)
        for i in range(k):
            h = _layer_hidden(layers[i], h, am)
        return self.main_head(self.dropout(h[:, 0]))

    @torch.no_grad()
    def verify_split_encoder(self, input_ids, attention_mask):
        """With no exit taken, the looped forward must equal BertModel.forward."""
        logits, conf = self.forward_with_exit(input_ids, attention_mask, theta=-1.0)
        ref_h = self.lm.bert(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state
        ref = self.main_head(self.dropout(ref_h[:, 0]))
        d = float((logits - ref).abs().max().item())
        assert float(conf.float().sum().item()) == 0.0, "theta=-1 must disable the exit"
        return d


# --------------------------------------------------------------------------
# distillation objectives
# --------------------------------------------------------------------------
# The temperature is read from the module global at *call* time rather than
# bound as a default argument, so a caller that overrides
# `common.TEMPERATURE = T` (e.g. for a temperature sweep) actually changes the
# behaviour of every objective below.
def pred_kl(student_logits, teacher_probs, T=None):
    T = TEMPERATURE if T is None else T
    log_p = F.log_softmax(student_logits / T, dim=-1)
    return F.kl_div(log_p, teacher_probs, reduction="batchmean") * (T * T)


def gated_teacher_probs(logits_a, logits_b, alpha, T=None):
    T = TEMPERATURE if T is None else T
    p1 = F.softmax(logits_a / T, dim=-1)
    p2 = F.softmax(logits_b / T, dim=-1)
    return alpha * p1 + (1.0 - alpha) * p2


def soft_probs(logits, T=None):
    T = TEMPERATURE if T is None else T
    return F.softmax(logits / T, dim=-1)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def accuracy(logits, labels):
    return (logits.argmax(-1) == labels).float().mean().item()


def macro_f1(logits, labels, num_labels=None):
    if num_labels is None:
        num_labels = NUM_LABELS
    pred = logits.argmax(-1).cpu().numpy()
    gold = labels.cpu().numpy()
    f1s = []
    for c in range(num_labels):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def mlm_scores(logits, labels):
    mask = labels != -100
    n = int(mask.sum().item())
    if n == 0:
        return 0.0, 0.0
    sel = logits[mask]
    gold = labels[mask]
    top1 = (sel.argmax(-1) == gold).float().mean().item()
    k = min(5, sel.shape[-1])
    top5 = (sel.topk(k, dim=-1).indices == gold.unsqueeze(-1)).any(-1).float().mean().item()
    return top1, top5


def normalised_entropy(probs):
    c = probs.shape[-1]
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1) / math.log(c)


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def _batch_size_of(batch):
    """Number of passages in a batch, for either (dict, labels) tuples or a
    plain dict of tensors."""
    first = batch[0] if isinstance(batch, (tuple, list)) else batch
    if isinstance(first, dict):
        if "input_ids" in first:
            return int(first["input_ids"].shape[0])
        for v in first.values():
            if hasattr(v, "shape") and v.dim() >= 1:
                return int(v.shape[0])
        return 0
    if hasattr(first, "shape"):
        return int(first.shape[0])
    return 0


def measure_throughput(fn, batches, warmup=2, repeats=3):
    """fn(batch) -> None.  Returns (passages_per_second, peak_vram_mb)."""
    dev = get_device()
    n_items = sum(_batch_size_of(b) for b in batches)
    for b in batches[:warmup]:
        fn(b)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(repeats):
        for b in batches:
            fn(b)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if dev.type == "cuda" else 0.0
    return (n_items * repeats) / dt, peak


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
