"""Swappable attention backends for PrismOpenJev (speed experiments, round 2).

Gemma 4 reads `config._attn_implementation` on every forward and looks the name up
in transformers' AttentionInterface, so a backend can be changed at runtime without
reloading the model. The Prism mask itself is not touched: every backend starts
from the boolean masks that `PrismOpenJev.masks()` builds with `prism_mask` (the
float conversion `to_additive` is swapped out for the duration of the call), so a
change to the mask rules (OJ-080) reaches every backend at once.

  sdpa_default  the current path: additive float mask, transformers' "sdpa".
  sdpa_bool     boolean mask (True = attend) to the same "sdpa" function: half the
                mask memory, and a different SDPA kernel may become eligible.
  chunked_math  plain math attention over query chunks of `chunk` rows, each chunk
                checkpointed when grad is on, so the attention probabilities kept
                for backward are O(chunk * T) per layer, not O(T^2).
  flex          torch.nn.attention.flex_attention with a BlockMask whose mask_mod
                looks up the same boolean Prism mask (sliding window included).
                Compiled on CUDA, eager on CPU (forward only: flex has no CPU
                backward). Raises Unsupported when flex cannot run.

Gemma 4 semantics kept: `scaling` comes from the caller (1.0, q/k are RMS-normed),
softcap is applied only when passed (Gemma 4 text attention passes none), GQA maps
query head h to kv head h // (H / H_kv) like repeat_kv, and KV-shared layers just
hand in the shared key/value tensors. The `sliding_window` kwarg is ignored: the
sliding mask already holds |pos_q - pos_k| < window on Prism position ids.
"""

from __future__ import annotations

import contextlib
import types

import torch
import torch.nn.functional as F
from torch.profiler import record_function
from torch.utils.checkpoint import checkpoint

from . import prism

BACKENDS = ("sdpa_default", "sdpa_bool", "chunked_math", "flex")
DEFAULT_CHUNK = 256
_state = {"chunk": DEFAULT_CHUNK}


class Unsupported(RuntimeError):
    """The backend cannot run here (bench_trials classifies it as 'unsupported')."""


# ---------------------------------------------------------------- attention functions

def _chunk_attn(q, k, v, mask, scaling, softcap, dropout, training):
    """q (B, Hkv, G, c, D), k/v (B, Hkv, 1, S, D), mask (B, 1|Hkv, 1|G, c, S)."""
    s = torch.matmul(q, k.transpose(-1, -2)) * scaling
    if softcap is not None:
        s = torch.tanh(s / softcap) * softcap
    if mask is not None:
        s = s.masked_fill(~mask, torch.finfo(s.dtype).min) if mask.dtype == torch.bool else s + mask
    p = F.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    if dropout:
        p = F.dropout(p, p=dropout, training=training)
    return torch.matmul(p, v)


def _group_mask(mask, B, Hkv, G, T, S):
    if mask is None:
        return None
    mask = mask[..., :T, :S]
    if mask.shape[1] == 1:
        return mask[:, :, None]                       # (B, 1, 1, T, S)
    return mask.reshape(mask.shape[0], Hkv, G, T, S)  # per-head mask


def chunked_attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                              scaling=None, softcap=None, **kwargs):
    with record_function("oj::attention"):
        if scaling is None:
            scaling = module.head_dim ** -0.5
        B, H, T, D = query.shape
        Hkv, S = key.shape[1], key.shape[2]
        G = H // Hkv
        q = query.reshape(B, Hkv, G, T, D)            # head h -> kv head h // G (repeat_kv order)
        k, v = key[:, :, None], value[:, :, None]
        m = _group_mask(attention_mask, B, Hkv, G, T, S)
        grad = torch.is_grad_enabled() and (query.requires_grad or key.requires_grad
                                            or value.requires_grad)
        c = max(1, int(_state["chunk"]))
        outs = []
        for s0 in range(0, T, c):
            qc = q[:, :, :, s0:s0 + c]
            mc = m[..., s0:s0 + c, :] if m is not None else None
            args = (qc, k, v, mc, scaling, softcap, dropout, module.training)
            if grad:
                outs.append(checkpoint(_chunk_attn, *args, use_reentrant=False))
            else:
                outs.append(_chunk_attn(*args))
        out = torch.cat(outs, dim=3).reshape(B, H, T, -1)
        return out.transpose(1, 2).contiguous(), None


_flex = {"compiled": None}
FLEX_BLOCK = 128


def _flex_fn(device_type):
    from torch.nn.attention.flex_attention import flex_attention
    if device_type != "cuda":
        return flex_attention                         # eager reference path (CPU tests)
    if _flex["compiled"] is None:
        import torch._dynamo as dynamo
        for name in ("cache_size_limit", "recompile_limit"):   # one graph per (T, bs, head_dim)
            if hasattr(dynamo.config, name):
                setattr(dynamo.config, name, max(getattr(dynamo.config, name), 64))
        _flex["compiled"] = torch.compile(flex_attention, dynamic=False)
    return _flex["compiled"]


def flex_kernel_options(head_dim: int, device_type: str):
    """Small Triton tiles for Gemma 4's 256/512 head dims (T4 has 64 KB shared memory)."""
    if device_type != "cuda" or head_dim <= 128:
        return None
    b = 32 if head_dim <= 256 else 16
    return {"BLOCK_M": b, "BLOCK_N": b, "BLOCK_M1": b, "BLOCK_N1": b, "BLOCK_M2": b,
            "BLOCK_N2": b, "num_stages": 1}


def flex_attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                           scaling=None, softcap=None, **kwargs):
    from torch.nn.attention.flex_attention import BlockMask
    with record_function("oj::attention"):
        if not isinstance(attention_mask, BlockMask):
            raise Unsupported("flex backend needs BlockMask masks (set_attn_backend(model, 'flex'))")
        if dropout:
            raise Unsupported("flex backend does not support attention dropout")
        if query.device.type == "cpu" and torch.is_grad_enabled() and (
                query.requires_grad or key.requires_grad or value.requires_grad):
            raise Unsupported("FlexAttention does not support backward on CPU")
        score_mod = None
        if softcap is not None:
            def score_mod(score, b, h, q_idx, kv_idx):
                return torch.tanh(score / softcap) * softcap
        G = query.shape[1] // key.shape[1]
        out = _flex_fn(query.device.type)(
            query, key, value, score_mod=score_mod, block_mask=attention_mask,
            scale=module.head_dim ** -0.5 if scaling is None else scaling, enable_gqa=G > 1,
            kernel_options=flex_kernel_options(query.shape[-1], query.device.type))
        return out.transpose(1, 2).contiguous(), None


def labelled_sdpa_forward(*args, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    with record_function("oj::attention"):
        return sdpa_attention_forward(*args, **kwargs)


def _register():
    from transformers import AttentionInterface
    AttentionInterface.register("openjev_chunked", chunked_attention_forward)
    AttentionInterface.register("openjev_flex", flex_attention_forward)
    AttentionInterface.register("openjev_sdpa", labelled_sdpa_forward)


_register()


# ---------------------------------------------------------------- masks

def bool_masks(model, b) -> dict:
    """PrismOpenJev.masks() with the boolean masks (True = attend) instead of additive."""
    orig = prism.to_additive
    prism.to_additive = lambda mask, dtype: mask
    try:
        return type(model).masks(model, b)
    finally:
        prism.to_additive = orig


def block_mask(dense: torch.Tensor):
    """BlockMask for a (B, 1, T, T) boolean mask; mask_mod looks the mask up."""
    from torch.nn.attention.flex_attention import create_block_mask
    B, _, Tq, Tk = dense.shape
    d = dense[:, 0]

    def mask_mod(b, h, q_idx, kv_idx):
        return d[b, q_idx, kv_idx]
    return create_block_mask(mask_mod, B, None, Tq, Tk, device=dense.device, BLOCK_SIZE=FLEX_BLOCK)


def flex_masks(model, b) -> dict:
    return {k: block_mask(v) for k, v in bool_masks(model, b).items()}


# ---------------------------------------------------------------- switch

def flex_available() -> tuple[bool, str]:
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # noqa: F401
    except Exception as e:  # old torch
        return False, f"{type(e).__name__}: {e}"
    return True, ""


def current(model) -> str:
    return getattr(model, "_attn_backend", "sdpa_default")


def set_attn_backend(model, name: str, chunk: int = DEFAULT_CHUNK, label: bool = False) -> None:
    """Switch a PrismOpenJev to backend `name` in place (no reload).
    label=True wraps masks, attention and the PLE lookup in record_function ranges
    (oj::masks, oj::attention, oj::ple) for the profiler."""
    if name not in BACKENDS:
        raise ValueError(f"attn_backend {name!r} not in {BACKENDS}")
    if name == "flex":
        ok, why = flex_available()
        if not ok:
            raise Unsupported(f"flex_attention not available: {why}")
    text = model._text_backbone()
    impl = {"sdpa_default": "sdpa", "sdpa_bool": "sdpa", "chunked_math": "openjev_chunked",
            "flex": "openjev_flex"}[name]
    if label and impl == "sdpa":
        impl = "openjev_sdpa"
    _state["chunk"] = int(chunk)
    text.config._attn_implementation = impl
    for attr in ("masks",):
        model.__dict__.pop(attr, None)
    text.__dict__.pop("get_per_layer_inputs", None)
    masks = {"sdpa_bool": bool_masks, "chunked_math": bool_masks, "flex": flex_masks}.get(name)
    if label:
        inner = masks or (lambda m, b: type(m).masks(m, b))

        def masks(m, b, inner=inner):
            with record_function("oj::masks"):
                return inner(m, b)
        if hasattr(text, "get_per_layer_inputs"):
            ple = type(text).get_per_layer_inputs

            def get_per_layer_inputs(self, *a, **k):
                with record_function("oj::ple"):
                    return ple(self, *a, **k)
            text.get_per_layer_inputs = types.MethodType(get_per_layer_inputs, text)
    if masks is not None:
        model.masks = types.MethodType(masks, model)
    model._attn_backend = name


def reset_attn_backend(model) -> None:
    set_attn_backend(model, "sdpa_default")


@contextlib.contextmanager
def attn_backend(model, name: str, **kw):
    """Use backend `name` inside the block; always back to sdpa_default after."""
    try:
        set_attn_backend(model, name, **kw)
        yield model
    finally:
        reset_attn_backend(model)


# ---------------------------------------------------------------- diagnostics

def sdpa_eligibility(model, device: str, dtype, T: int = 128) -> dict:
    """Which SDPA kernels accept our case, per head dim and mask type (CUDA only).
    Answers 'does a boolean mask make another kernel eligible' directly."""
    if device != "cuda":
        return {}
    out = {}
    try:
        from torch.backends.cuda import (SDPAParams, can_use_efficient_attention,
                                         can_use_flash_attention)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    checks = {"flash": can_use_flash_attention, "efficient": can_use_efficient_attention}
    try:
        from torch.backends.cuda import can_use_cudnn_attention
        checks["cudnn"] = can_use_cudnn_attention
    except Exception:
        pass
    text = model._text_backbone()
    cfg = text.config
    H = cfg.num_attention_heads
    dims = sorted({layer.self_attn.head_dim for layer in text.layers})
    for D in dims:
        layer = next(lay.self_attn for lay in text.layers if lay.self_attn.head_dim == D)
        Hkv = H // layer.num_key_value_groups
        q = torch.randn(1, H, T, D, device=device, dtype=dtype)
        k = torch.randn(1, Hkv, T, D, device=device, dtype=dtype).repeat_interleave(H // Hkv, 1)
        masks = {"bool": torch.ones(1, 1, T, T, device=device, dtype=torch.bool),
                 "float": torch.zeros(1, 1, T, T, device=device, dtype=dtype)}
        row = {}
        for mk, m in masks.items():
            res = {}
            for kn, fn in checks.items():
                try:
                    try:
                        p = SDPAParams(q, k, k, m, 0.0, False, False)
                    except TypeError:
                        p = SDPAParams(q, k, k, m, 0.0, False)
                    res[kn] = bool(fn(p, False))
                except Exception as e:
                    res[kn] = f"{type(e).__name__}"[:40]
            row[mk] = res
        out[str(D)] = row
    return out
