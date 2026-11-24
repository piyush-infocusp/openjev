"""Prism-style attention for OpenJev on a Gemma 4 backbone.

Sarvam Prism (https://www.sarvam.ai/blogs/sarvam-prism) changes only the
attention mask of a decoder LLM:
  * the state and question block is read in both directions,
  * each option sees that block and itself, never the other options,
  * every option starts from the same position,
  * only the answer token sees everything, so options are compared there.

OpenJev already puts one <|flag|> token after each option. Here the flag plays
the part of Prism's answer token: a flag sees the shared block, its own option
and the sibling FLAGS of its own question (not the sibling option tokens). The
flag head and the Jev request and answer format do not change. Only internal
details (mask and position ids) do.

Why not the sibling option tokens (OJ-080): all options of a question start at
the same position and every flag has the same embedding. If a flag could see
every option of its question, all flags would attend to the same key set and
differ only by their own position, so options of equal token length got exactly
the same logit and training could not separate them (v8: near-uniform output,
per-source accuracy frozen). A flag's own option tokens are visible to it alone,
which breaks that symmetry; the sibling flags still let options be compared.

Segments of one encoded request (see schema.encode):
  shared  : <|ctx|> state <|/ctx|> and every <|q|> name <|instr|> text <|/q|>
  option  : <|opt|> label     (one segment per option, all options of a
            question start at the same position id)
  flag    : <|flag|>          (belongs to its option, and also sees the
            sibling flags of its question)

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
    doc: torch.Tensor = None           # (B, T) long: request index inside its row, -1 pad
                                       # (all 0 when unpacked; packing: OJ-060)

    def to(self, device) -> "PrismBatch":
        super().to(device)
        for k in ("position_ids", "segment", "question", "is_flag", "doc"):
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
                  pad_to_multiple_of: int = 8, pack_len: int = 0, rows_of=None) -> PrismBatch:
    """Encode `records` into one PrismBatch.

    pack_len = 0: one request per row (right-padded). pack_len > 0 (OJ-060): requests are
    packed into rows of <= pack_len tokens (openjev.packing.pack_rows; a longer request
    gets its own row). Inside a row each request keeps its own layout and positions that
    restart at 0, and `doc` (its index in the row) makes the mask block-diagonal, so the
    flag logits equal the unpacked ones. Either way flag_index, groups, labels and schemas
    follow the order of `records`, so the logits come out in the same order.

    rows_of: a precomputed row assignment (tools/freeze_packed.py layout for this micro-batch);
    must partition range(len(records)); replaces pack_rows."""
    enc = [encode(tokenizer, r.state, r.schema, max_state_tokens) for r in records]
    opt_id = tokenizer.convert_tokens_to_ids(OPT)
    flag_id = tokenizer.convert_tokens_to_ids(FLAG)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    lens = [len(e.input_ids) for e in enc]
    if rows_of is not None:
        rows_of = [list(r) for r in rows_of]
        if sorted(i for r in rows_of for i in r) != list(range(len(enc))):
            raise ValueError("rows_of must partition the records")
    elif pack_len:
        from .packing import pack_rows
        rows_of = pack_rows(lens, pack_len)
    else:
        rows_of = [[i] for i in range(len(enc))]

    n = max(sum(lens[i] for i in r) for r in rows_of)
    if pad_to_multiple_of:
        n = -(-n // pad_to_multiple_of) * pad_to_multiple_of
    B = len(rows_of)
    input_ids = torch.full((B, n), pad_id, dtype=torch.long)
    attn = torch.zeros((B, n), dtype=torch.long)
    pos = torch.zeros((B, n), dtype=torch.long)
    seg = torch.full((B, n), -1, dtype=torch.long)
    qix = torch.full((B, n), -1, dtype=torch.long)
    isf = torch.zeros((B, n), dtype=torch.bool)
    doc = torch.full((B, n), -1, dtype=torch.long)

    where = {}  # record index -> (row, column offset)
    for ri, row in enumerate(rows_of):
        off, last = 0, -1
        for d, i in enumerate(row):
            e, L = enc[i], lens[i]
            s, q, f, p = layout(e.input_ids, e.readouts, opt_id, flag_id)
            sl = slice(off, off + L)
            input_ids[ri, sl] = torch.tensor(e.input_ids)
            attn[ri, sl] = 1
            seg[ri, sl], qix[ri, sl] = torch.tensor(s), torch.tensor(q)
            isf[ri, sl], pos[ri, sl] = torch.tensor(f), torch.tensor(p)
            doc[ri, sl] = d
            where[i] = (ri, off)
            off, last = off + L, max(p)
        pos[ri, off:] = torch.arange(last + 1, last + 1 + n - off)

    rows, cols, groups, start = [], [], [], 0
    for i, e in enumerate(enc):
        ri, off = where[i]
        ex = []
        for rq in e.readouts:
            rows += [ri] * len(rq["flags"])
            cols += [off + f for f in rq["flags"]]
            ex.append({"type": rq["type"], "name": rq["name"], "start": start,
                       "end": start + len(rq["flags"])})
            start += len(rq["flags"])
        groups.append(ex)
    flag_index = torch.tensor(list(zip(rows, cols)), dtype=torch.long).reshape(-1, 2)
    return PrismBatch(input_ids, attn, flag_index, groups,
                      [list(r.answers) for r in records], [r.schema for r in records],
                      position_ids=pos, segment=seg, question=qix, is_flag=isf, doc=doc)


def prism_mask(segment, question, is_flag, position_ids, window: int | None = None,
               doc: torch.Tensor | None = None):
    """(B, 1, T, T) bool mask, True = may attend. doc (B, T): with packing, a token only
    sees tokens of its own request (block-diagonal); segment, question and position ids
    restart in every request, so without it requests would see each other."""
    sq, sk = segment[:, :, None], segment[:, None, :]
    qq, qk = question[:, :, None], question[:, None, :]
    real_k = sk >= 0
    m = (sk == SHARED)                                   # everyone sees the shared block
    m = m & (sq != SHARED) | (sq == SHARED) & (sk == SHARED)  # shared sees only shared
    m = m | ((sq == sk) & (sq > 0))                      # an option sees itself (both ways)
    # A flag also sees the sibling FLAGS of its question, never the sibling
    # option tokens: seeing those makes every flag of a question attend to the
    # same keys, so equal-length options get identical logits (OJ-080).
    m = m | (is_flag[:, :, None] & is_flag[:, None, :] & (qq == qk) & (qk >= 0))
    m = m & real_k
    if doc is not None:
        m = m & (doc[:, :, None] == doc[:, None, :])
    if window is not None:
        d = (position_ids[:, :, None] - position_ids[:, None, :]).abs()
        m = m & (d < window)
    eye = torch.eye(segment.shape[1], dtype=torch.bool, device=segment.device)
    m = m | eye[None]                                    # pad rows: no empty row, no NaN
    return m[:, None]


def to_additive(mask: torch.Tensor, dtype) -> torch.Tensor:
    out = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return out.masked_fill(~mask, torch.finfo(dtype).min)


def host_to_device(t: torch.Tensor, device) -> torch.Tensor:
    """Copy t to device. CPU -> CUDA (the looked-up PLE rows with ple_cpu=1) goes through
    pinned memory with non_blocking=True, so the copy is a DMA that does not stall the
    host; the caching host allocator keeps the pinned block alive until the copy is done."""
    device = torch.device(device)
    if t.device == device:
        return t
    if t.device.type == "cpu" and device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def log_rss(stage: str) -> dict:
    """Print the host memory of this process: VmRSS now and the peak (ru_maxrss), in GB,
    so a Kaggle log shows where host RAM peaks (r2 SIGKILLs). Never raises."""
    out = {"host_rss": stage}
    try:
        import resource
        out["maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)  # KiB
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["vmrss_gb"] = round(int(line.split()[1]) / 2**20, 2)
    except Exception as e:  # not Linux, etc.
        out["error"] = type(e).__name__
    import json
    print(json.dumps(out), flush=True)
    return out


def gemma_text_backbone(name: str, dtype=torch.float32, convert: bool = True):
    """Load only the text decoder of a Gemma 4 checkpoint (drops vision and audio)."""
    from transformers import AutoModel

    # The audio tower has no SDPA support, so load everything with eager
    # attention, keep only the text decoder, then switch it to SDPA (the
    # attention layers read config._attn_implementation at forward time).
    # Host-RAM safety: Gemma 4 ships in bf16. Asking from_pretrained for fp16 or
    # fp32 converts the WHOLE multimodal model in host RAM, which held two copies
    # and was SIGKILLed on Kaggle (~30 GB RAM, optuna r2). So load in bf16, drop
    # the vision/audio towers first, then convert the text decoder one tensor at
    # a time (peak = one extra tensor, not one extra model).
    load_dtype = torch.bfloat16 if dtype in (torch.float16, torch.float32) else dtype
    m = AutoModel.from_pretrained(name, dtype=load_dtype, attn_implementation="eager",
                                  low_cpu_mem_usage=True)
    log_rss("after_load")
    text = getattr(m, "language_model", m)
    for tower in ("audio_tower", "vision_tower", "embed_audio", "embed_vision"):
        if hasattr(m, tower):
            setattr(m, tower, None)
    del m
    import gc
    gc.collect()
    log_rss("after_drop_towers")
    if convert and dtype != load_dtype:
        convert_inplace(text, dtype, skip_trainable=False)   # nothing is frozen yet
        log_rss("after_convert")
    text.config._attn_implementation = "sdpa"
    return text


def _cast(t: torch.Tensor, dtype, chunk: int = 1 << 24) -> torch.Tensor:
    """Cast t to dtype. When both dtypes have the same element size (bf16 <-> fp16),
    overwrite t's own storage chunk by chunk, so peak extra memory is one chunk, not a
    full copy. The PLE table is a single 2.35B-element tensor: a plain .to() held a
    second 4.7 GB copy in host RAM (optuna r2b SIGKILL)."""
    if (t.element_size() != torch.empty((), dtype=dtype).element_size()
            or not t.is_contiguous()):
        return t.to(dtype)
    flat = t.view(-1)
    out = flat.view(dtype)                      # same bytes, reinterpreted
    for i in range(0, flat.numel(), chunk):
        out[i:i + chunk] = flat[i:i + chunk].to(dtype)   # read chunk first, then overwrite
    return out.view(t.shape)


def convert_inplace(module: nn.Module, dtype, skip_trainable: bool = True) -> nn.Module:
    """Cast floating parameters and buffers one tensor at a time (in place when possible).
    skip_trainable: parameters with requires_grad (LoRA, flag head, template_delta) keep
    their dtype, so they stay fp32 for AdamW. gemma_text_backbone passes False: right after
    from_pretrained every parameter still has requires_grad=True. Runs on the device the
    tensors live on, so after place() the cast happens on the GPU (r3 host-RAM SIGKILL)."""
    done = {}                                  # a shared parameter is cast once, then reused
    with torch.no_grad():
        for sub in module.modules():
            for name, p in list(sub.named_parameters(recurse=False)):
                if id(p) in done:
                    sub._parameters[name] = done[id(p)]
                elif p.is_floating_point() and p.dtype != dtype and not (skip_trainable and p.requires_grad):
                    done[id(p)] = sub._parameters[name] = nn.Parameter(_cast(p.data, dtype),
                                                                      requires_grad=p.requires_grad)
            for name, b in list(sub.named_buffers(recurse=False)):
                if b is not None and b.is_floating_point() and b.dtype != dtype:
                    sub._buffers[name] = b.to(dtype)
    return module


# OJ-075: quantized frozen base weights (bitsandbytes, QLoRA-style). Only the projections that
# LoRA wraps, inside the decoder layers, are quantized: never the embedding, the PLE table and
# its projections, or the flag head.
QUANTS = ("none", "int8", "nf4")
QUANT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
INT8_THRESHOLD = 6.0   # LLM.int8() outlier threshold (Dettmers et al. 2022); 0.0 = pure int8
BNB_PIN = "bitsandbytes==0.50.2"


def quant_targets(text: nn.Module) -> list[str]:
    """Names (relative to `text`) of the modules quantize_linears replaces: exact nn.Linear
    under the decoder `layers` whose leaf name is a LoRA target."""
    return [n for n, m in text.named_modules()
            if type(m) is nn.Linear and n.startswith("layers.")
            and n.rsplit(".", 1)[-1] in QUANT_TARGETS]


def quantize_linears(text: nn.Module, quant: str, device, compute_dtype=torch.float16) -> list[str]:
    """Replace quant_targets(text) with bitsandbytes layers (int8: Linear8bitLt, LLM.int8();
    nf4: Linear4bit, no double quant). Each layer is quantized straight onto `device`, one
    tensor at a time: its weight is cast to compute_dtype on the host (nf4's quant_state keeps
    that dtype; on a T4 bnb's fast path needs fp16), then moved, so host RAM peaks at one
    tensor. The shells are built on 'meta', so no RNG is used and the LoRA / flag head init
    stays equal to quant=none. Sets text.is_loaded_in_8bit / _4bit so PEFT wraps them with its
    bnb LoRA layers. Call before the LoRA wrap. Returns the replaced names."""
    if quant == "none":
        return []
    if quant not in QUANTS:
        raise ValueError(f"quant {quant!r} not in {QUANTS}")
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(f"quant={quant} needs bitsandbytes (pip install {BNB_PIN})") from e
    names = quant_targets(text)
    with torch.no_grad():
        for name in names:
            old = text.get_submodule(name)
            src = old.weight.data.to("cpu", dtype=compute_dtype)   # Int8Params quantizes from CPU
            bias = old.bias is not None
            if quant == "nf4":
                new = bnb.nn.Linear4bit(old.in_features, old.out_features, bias=bias,
                                        compute_dtype=compute_dtype, compress_statistics=False,
                                        quant_type="nf4", device="meta")
                new.weight = bnb.nn.Params4bit(src, requires_grad=False, compress_statistics=False,
                                               quant_type="nf4", module=new)
            else:
                new = bnb.nn.Linear8bitLt(old.in_features, old.out_features, bias=bias,
                                          has_fp16_weights=False, threshold=INT8_THRESHOLD,
                                          device="meta")
                new.weight = bnb.nn.Int8Params(src, requires_grad=False, has_fp16_weights=False)
            if bias:
                new.bias = nn.Parameter(old.bias.data.to("cpu", dtype=compute_dtype),
                                        requires_grad=False)
            new = new.to(device)                  # quantizes the weight on `device`
            parent, attr = name.rsplit(".", 1)
            setattr(text.get_submodule(parent), attr, new)
            del old, src
    text.is_loaded_in_8bit = quant == "int8"   # PEFT dispatch (peft.tuners.lora.bnb)
    text.is_loaded_in_4bit = quant == "nf4"
    return names


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
        # mean_resizing=False: Gemma 4 also resizes the 2.35B-element PLE table, and HF's mean
        # resizing builds fp32 copies + an 8960-dim covariance of it on the host (r3small:
        # maxrss 28.9 GB at after_resize of ~30 GB). add_special_tokens sets the new rows.
        return self.backbone.resize_token_embeddings(n, mean_resizing=False)

    def get_per_layer_input_embeddings(self):
        f = getattr(self._text_backbone(), "get_per_layer_input_embeddings", None)
        return f() if f else None

    def masks(self, b: PrismBatch):
        cfg = self.backbone.config
        dtype = self.get_input_embeddings().weight.dtype
        if self.causal:
            # Plain causal ablation: ignores segments, positions and doc, so it cannot
            # take packed rows (train_prism rejects --causal with --packing on).
            T = b.input_ids.shape[1]
            tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=b.input_ids.device))
            full = tri[None, None] & (b.attention_mask[:, None, None, :] > 0)
            full = full | torch.eye(T, dtype=torch.bool, device=full.device)
            ar = torch.arange(T, device=full.device)
            slide = full & ((ar[:, None] - ar[None, :]) < cfg.sliding_window)
        else:
            full = prism_mask(b.segment, b.question, b.is_flag, b.position_ids, doc=b.doc)
            slide = prism_mask(b.segment, b.question, b.is_flag, b.position_ids,
                               window=getattr(cfg, "sliding_window", None), doc=b.doc)
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
            # Gemma 4 E2B/E4B have no spare rows (vocab == tokenizer size), so adding
            # the template tokens resizes the Per-Layer Embedding table too and grows
            # vocab_size_per_layer_input with it (transformers 5.x). The guard below
            # then never fires: each template token reads its OWN PLE row, frozen at
            # its fresh-build value (add_special_tokens: mean + seeded noise), saved in
            # ple_rows.pt and pasted back by train_prism.build on --init / reload.
            # Only a backbone whose PLE table did not grow maps ids past it to pad.
            pad = text.config.pad_token_id or 0
            ple_ids = torch.where(ids >= v_ple, torch.full_like(ids, pad), ids)
            ple_dev = text.embed_tokens_per_layer.weight.device
            ple = text.get_per_layer_inputs(ple_ids.to(ple_dev), None)  # table may live on CPU
            kw["per_layer_inputs"] = host_to_device(ple, ids.device)
        hs = self.backbone(**kw, attention_mask=self.masks(b), position_ids=pos,
                           use_cache=False).last_hidden_state.float()
        idx = b.flag_index
        return self.flag_head(hs[idx[:, 0], idx[:, 1]]).squeeze(-1)

    def forward(self, b: PrismBatch) -> torch.Tensor:
        """= flag_logits. DistributedDataParallel calls forward (train_prism under torchrun)."""
        return self.flag_logits(b)


# ---------------------------------------------------------------- template PLE rows

PLE_ROWS = "ple_rows.pt"   # {str(token id): PLE row} of the template tokens, in a checkpoint


def template_ple_rows(model: PrismOpenJev, ids: list[int]) -> dict | None:
    """{str(id): row} of the template tokens' Gemma 4 PLE rows, in the table's dtype (exact),
    or None when there is no PLE table or it does not reach the template ids (then
    flag_logits maps them to pad, nothing to save). The rows are frozen, so their value is
    whatever add_special_tokens drew on the fresh SFT build (mean + seeded noise)."""
    ple = model.get_per_layer_input_embeddings()
    if ple is None or max(ids) >= ple.weight.shape[0]:
        return None
    w = ple.weight
    return {str(i): w[i].detach().cpu().clone() for i in ids}


def paste_ple_rows(model: PrismOpenJev, rows: dict, where: str = "") -> None:
    """Write saved template PLE rows back into the (resized) table, in its dtype and on its
    device. Raises on a missing table or a shape mismatch (e.g. an E4B checkpoint on E2B)."""
    ple = model.get_per_layer_input_embeddings()
    if ple is None:
        raise ValueError(f"{where}: {PLE_ROWS} present but the backbone has no PLE table")
    w = ple.weight
    ids = [int(i) for i in rows]
    if max(ids) >= w.shape[0]:
        raise ValueError(f"{where}: {PLE_ROWS} id {max(ids)} >= PLE table rows {w.shape[0]}")
    v = torch.stack([rows[str(i)] for i in ids])
    if v.shape[1:] != w.shape[1:]:
        raise ValueError(f"{where}: {PLE_ROWS} row shape {tuple(v.shape[1:])} != "
                         f"PLE table {tuple(w.shape[1:])}")
    with torch.no_grad():
        w[torch.tensor(ids, device=w.device)] = v.to(w.device, w.dtype)
