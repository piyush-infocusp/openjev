"""Prism-style attention for OpenJev on a Gemma 4 backbone.

Sarvam Prism (https://www.sarvam.ai/blogs/sarvam-prism) changes only the
attention mask of a decoder LLM:
  * the state and question block is read in both directions,
  * each option sees that block and itself, never the other options,
  * every option starts from the same position,
  * only the answer token sees everything, so options are compared there.

OpenJev already puts one <|flag|> token after each option. Here the flag plays
the part of Prism's answer token: a flag sees the shared block, its own option
and every option of its own question. The flag head and the Jev request and
answer format do not change. Only internal details (mask and position ids) do.

Segments of one encoded request (see schema.encode):
  shared  : <|ctx|> state <|/ctx|> and every <|q|> name <|instr|> text <|/q|>
  option  : <|opt|> label     (one segment per option, all options of a
            question start at the same position id)
  flag    : <|flag|>          (belongs to its option, and also sees the
            sibling options of its question)

Gemma 4 mixes sliding-window and full-attention layers. Gemma4TextModel accepts
a dict {"full_attention": mask, "sliding_attention": mask}, so both masks are
built here, the sliding one limited to |pos_q - pos_k| < window.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .collate import Batch
from .data import Record
from .schema import FLAG, OPT, encode

SHARED = 0  # segment id of the shared (bidirectional) block


@dataclass
class PrismBatch(Batch):
    position_ids: torch.Tensor = None  # (B, T) long
    segment: torch.Tensor = None       # (B, T) long: 0 shared, >0 option id, -1 pad
    question: torch.Tensor = None      # (B, T) long: question index of option/flag, -1 else
    is_flag: torch.Tensor = None       # (B, T) bool

    def to(self, device) -> "PrismBatch":
        super().to(device)
        for k in ("position_ids", "segment", "question", "is_flag"):
            setattr(self, k, getattr(self, k).to(device))
        return self


def layout(ids: list[int], readouts: list[dict], opt_id: int, flag_id: int):
    """Segment, question and position ids for one encoded request."""
    T = len(ids)
    seg = [SHARED] * T
    qix = [-1] * T
    flag = [False] * T
    spans = []  # (question index, start, end) with end = flag position
    for qi, q in enumerate(readouts):
        for f in q["flags"]:
            s = f
            while ids[s] != opt_id:
                s -= 1
            spans.append((qi, s, f))
    for n, (qi, s, f) in enumerate(spans, start=1):
        for t in range(s, f + 1):
            seg[t], qix[t] = n, qi
        flag[f] = True

    # Shared tokens count up. Options of one question all restart at the
    # position after the shared prefix; the next shared token continues after
    # the longest option.
    pos = [0] * T
    starts = {}
    p, t = 0, 0
    while t < T:
        if seg[t] == SHARED:
            pos[t] = p
            p += 1
            t += 1
            continue
        qi = qix[t]
        base = starts.setdefault(qi, p)
        longest = 0
        while t < T and seg[t] != SHARED and qix[t] == qi:
            s0, k = t, seg[t]
            while t < T and seg[t] == k:
                pos[t] = base + (t - s0)
                t += 1
            longest = max(longest, t - s0)
        p = base + longest
    return seg, qix, flag, pos


def prism_collate(tokenizer, records: list[Record], max_state_tokens: int = 2048,
                  pad_to_multiple_of: int = 8) -> PrismBatch:
    enc = [encode(tokenizer, r.state, r.schema, max_state_tokens) for r in records]
    opt_id = tokenizer.convert_tokens_to_ids(OPT)
    flag_id = tokenizer.convert_tokens_to_ids(FLAG)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    n = max(len(e.input_ids) for e in enc)
    if pad_to_multiple_of:
        n = -(-n // pad_to_multiple_of) * pad_to_multiple_of
    B = len(enc)
    input_ids = torch.full((B, n), pad_id, dtype=torch.long)
    attn = torch.zeros((B, n), dtype=torch.long)
    pos = torch.zeros((B, n), dtype=torch.long)
    seg = torch.full((B, n), -1, dtype=torch.long)
    qix = torch.full((B, n), -1, dtype=torch.long)
    isf = torch.zeros((B, n), dtype=torch.bool)

    rows, cols, groups, start = [], [], [], 0
    for i, e in enumerate(enc):
        L = len(e.input_ids)
        s, q, f, p = layout(e.input_ids, e.readouts, opt_id, flag_id)
        input_ids[i, :L] = torch.tensor(e.input_ids)
        attn[i, :L] = 1
        seg[i, :L], qix[i, :L] = torch.tensor(s), torch.tensor(q)
        isf[i, :L], pos[i, :L] = torch.tensor(f), torch.tensor(p)
        pos[i, L:] = torch.arange(max(p) + 1, max(p) + 1 + n - L)
        ex = []
        for rq in e.readouts:
            rows += [i] * len(rq["flags"])
            cols += rq["flags"]
            ex.append({"type": rq["type"], "name": rq["name"], "start": start,
                       "end": start + len(rq["flags"])})
            start += len(rq["flags"])
        groups.append(ex)
    flag_index = torch.tensor(list(zip(rows, cols)), dtype=torch.long).reshape(-1, 2)
    return PrismBatch(input_ids, attn, flag_index, groups,
                      [list(r.answers) for r in records], [r.schema for r in records],
                      position_ids=pos, segment=seg, question=qix, is_flag=isf)


def prism_mask(segment, question, is_flag, position_ids, window: int | None = None):
    """(B, 1, T, T) bool mask, True = may attend."""
    sq, sk = segment[:, :, None], segment[:, None, :]
    qq, qk = question[:, :, None], question[:, None, :]
    real_k = sk >= 0
    m = (sk == SHARED)                                   # everyone sees the shared block
    m = m & (sq != SHARED) | (sq == SHARED) & (sk == SHARED)  # shared sees only shared
    m = m | ((sq == sk) & (sq > 0))                      # an option sees itself (both ways)
    m = m | (is_flag[:, :, None] & (qq == qk) & (qk >= 0))  # a flag sees its question's options
    m = m & real_k
    if window is not None:
        d = (position_ids[:, :, None] - position_ids[:, None, :]).abs()
        m = m & (d < window)
    eye = torch.eye(segment.shape[1], dtype=torch.bool, device=segment.device)
    m = m | eye[None]                                    # pad rows: no empty row, no NaN
    return m[:, None]


def to_additive(mask: torch.Tensor, dtype) -> torch.Tensor:
    out = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return out.masked_fill(~mask, torch.finfo(dtype).min)


def gemma_text_backbone(name: str, dtype=torch.float32):
    """Load only the text decoder of a Gemma 4 checkpoint (drops vision and audio)."""
    from transformers import AutoModel

    # The audio tower has no SDPA support, so load everything with eager
    # attention, keep only the text decoder, then switch it to SDPA (the
    # attention layers read config._attn_implementation at forward time).
    m = AutoModel.from_pretrained(name, dtype=dtype, attn_implementation="eager")
    text = getattr(m, "language_model", m)
    for tower in ("audio_tower", "vision_tower", "embed_audio", "embed_vision"):
        if hasattr(m, tower):
            setattr(m, tower, None)
    text.config._attn_implementation = "sdpa"
    return text


class PrismOpenJev(nn.Module):
    """OpenJev flag head on a Gemma 4 text decoder with the Prism mask."""

    def __init__(self, backbone, causal: bool = False):
        super().__init__()
        self.backbone = backbone
        self.causal = causal  # True = plain causal mask (ablation baseline)
        self.flag_head = nn.Linear(backbone.config.hidden_size, 1)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def resize_token_embeddings(self, n):
        return self.backbone.resize_token_embeddings(n)

    def masks(self, b: PrismBatch):
        cfg = self.backbone.config
        dtype = self.get_input_embeddings().weight.dtype
        if self.causal:
            T = b.input_ids.shape[1]
            tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=b.input_ids.device))
            full = tri[None, None] & (b.attention_mask[:, None, None, :] > 0)
            full = full | torch.eye(T, dtype=torch.bool, device=full.device)
            ar = torch.arange(T, device=full.device)
            slide = full & ((ar[:, None] - ar[None, :]) < cfg.sliding_window)
        else:
            full = prism_mask(b.segment, b.question, b.is_flag, b.position_ids)
            slide = prism_mask(b.segment, b.question, b.is_flag, b.position_ids,
                               window=getattr(cfg, "sliding_window", None))
        return {"full_attention": to_additive(full, dtype),
                "sliding_attention": to_additive(slide, dtype)}

    def set_template_ids(self, ids: list[int]) -> None:
        """Make the template tokens learnable without training the whole
        embedding matrix (262144 x H: an fp32 AdamW state for it alone is
        2-5 GB). A small fp32 delta per template token is added after the
        frozen embedding lookup."""
        self.register_buffer("template_ids", torch.tensor(ids, dtype=torch.long), persistent=False)
        self.template_delta = nn.Parameter(torch.zeros(len(ids), self.backbone.config.hidden_size))

    def _text_backbone(self):
        m = self.backbone
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    def embed(self, ids):
        e = self.get_input_embeddings()(ids)
        if getattr(self, "template_delta", None) is not None:
            match = ids[..., None] == self.template_ids            # (B, T, n)
            delta = match.to(self.template_delta.dtype) @ self.template_delta
            e = e + delta.to(e.dtype)
        return e

    def flag_logits(self, b: PrismBatch) -> torch.Tensor:
        pos = b.position_ids if not self.causal else None
        ids = b.input_ids
        kw = {"inputs_embeds": self.embed(ids)}
        text = self._text_backbone()
        v_ple = getattr(text.config, "vocab_size_per_layer_input", None)
        if v_ple and hasattr(text, "get_per_layer_inputs"):
            # The template tokens are added after Gemma's vocabulary, but the
            # Per-Layer Embedding table keeps its size. Those tokens get the
            # pad token's PLE row; their main embedding is their own.
            pad = text.config.pad_token_id or 0
            ple_ids = torch.where(ids >= v_ple, torch.full_like(ids, pad), ids)
            ple_dev = text.embed_tokens_per_layer.weight.device
            ple = text.get_per_layer_inputs(ple_ids.to(ple_dev), None)  # table may live on CPU
            kw["per_layer_inputs"] = ple.to(ids.device)
        hs = self.backbone(**kw, attention_mask=self.masks(b), position_ids=pos,
                           use_cache=False).last_hidden_state.float()
        idx = b.flag_index
        return self.flag_head(hs[idx[:, 0], idx[:, 1]]).squeeze(-1)
