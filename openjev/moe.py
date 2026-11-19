"""Sparse upcycling of a dense Qwen3 MLP into a mixture-of-experts (MoE).

Why: no public pretrained MoE has fewer than 1B *total* parameters (see
PLAN.md section 0, decision 2) -- small MoE checkpoints are only small in
*active* parameters. Sparse upcycling (Komatsuzaki et al. 2022,
arXiv:2212.05055) copies a dense model's MLP E times to build a real MoE
under 1B total while keeping active compute close to the dense model, since
each token only visits top_k of the E experts.

At init all experts are identical copies of the dense MLP, so with top_k=1
and renormalized routing weights the MoE forward pass is numerically
identical to the dense forward pass -- upcycling changes nothing until
training moves the experts apart. This ships only if the resulting MoE
beats the dense backbone at equal active compute.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpcycledMoE(nn.Module):
    """Replaces a dense MLP with `n_experts` copies of it plus a top-k router."""

    def __init__(self, mlp: nn.Module, n_experts: int = 4, top_k: int = 1,
                 router_noise: float = 1e-2):
        super().__init__()
        if top_k > n_experts:
            raise ValueError("top_k must be <= n_experts")
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(copy.deepcopy(mlp) for _ in range(n_experts))
        hidden = mlp.hidden_size
        self.router = nn.Linear(hidden, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=router_noise)
        self.last_aux = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        flat = x.reshape(-1, h)
        logits = self.router(flat)  # (N, E)
        probs = torch.softmax(logits, dim=-1)

        top_w, top_idx = probs.topk(self.top_k, dim=-1)  # (N, k)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)

        out = flat.new_zeros(flat.shape)
        for e in range(self.n_experts):
            mask = (top_idx == e)  # (N, k)
            if not mask.any():
                continue
            token_mask = mask.any(dim=-1)  # (N,)
            weight = (top_w * mask.float()).sum(dim=-1)  # (N,)
            tokens = flat[token_mask]
            if tokens.numel() == 0:
                continue
            expert_out = self.experts[e](tokens)
            out[token_mask] += expert_out * weight[token_mask].unsqueeze(-1)

        # Switch-style load-balancing loss over all tokens (no padding mask
        # available here): n_experts * sum_e(frac_tokens_e * mean_prob_e).
        n = flat.shape[0]
        with torch.no_grad():
            chosen = top_idx[:, 0] if self.top_k >= 1 else top_idx.new_zeros(n)
        frac = torch.zeros(self.n_experts, device=flat.device, dtype=probs.dtype)
        frac.scatter_add_(0, chosen, torch.ones(n, device=flat.device, dtype=probs.dtype))
        frac = frac / max(n, 1)
        mean_prob = probs.mean(dim=0)
        self.last_aux = self.n_experts * (frac * mean_prob).sum()

        return out.reshape(b, t, h)


def _find_layers(backbone):
    if hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
        return backbone.model.layers
    raise AttributeError("could not find decoder layers on backbone")


def upcycle(backbone, n_experts: int = 4, last_n_layers: int = 8, top_k: int = 1):
    """Replace `.mlp` of the last `last_n_layers` decoder layers with UpcycledMoE."""
    layers = _find_layers(backbone)
    n = len(layers)
    for i in range(max(0, n - last_n_layers), n):
        layer = layers[i]
        layer.mlp = UpcycledMoE(layer.mlp, n_experts=n_experts, top_k=top_k)
    return backbone


def aux_loss(backbone) -> torch.Tensor:
    """Sum of `last_aux` over all UpcycledMoE modules in `backbone`."""
    total = None
    for m in backbone.modules():
        if isinstance(m, UpcycledMoE):
            total = m.last_aux if total is None else total + m.last_aux
    if total is None:
        return torch.tensor(0.0)
    return total


def param_counts(model) -> dict:
    """{"total": ..., "active": ...} parameter counts.

    "active" counts only top_k of n_experts expert parameters per MoE layer
    (plus everything else, counted once).
    """
    total = 0
    active = 0
    skip_prefixes = []
    for name, m in model.named_modules():
        if isinstance(m, UpcycledMoE):
            router_params = sum(p.numel() for p in m.router.parameters())
            expert_params = sum(p.numel() for p in m.experts[0].parameters())
            total += router_params + expert_params * m.n_experts
            active += router_params + expert_params * m.top_k
            skip_prefixes.append(name + ".")

    for name, m in model.named_modules():
        if isinstance(m, UpcycledMoE) or any(name.startswith(p) for p in skip_prefixes):
            continue
        p_here = sum(p.numel() for p in m.parameters(recurse=False))
        total += p_here
        active += p_here
    return {"total": total, "active": active}


def projected_counts(config, n_experts: int = 4, last_n_layers: int = 8,
                      top_k: int = 1) -> dict:
    """Analytic dense + upcycled parameter counts from an HF config only."""
    h = config.hidden_size
    i = config.intermediate_size
    L = config.num_hidden_layers
    V = config.vocab_size

    n_heads = getattr(config, "num_attention_heads", None)
    n_kv = getattr(config, "num_key_value_heads", n_heads)
    head_dim = getattr(config, "head_dim", h // n_heads if n_heads else None)

    # OpenJev wraps the bare backbone (transformers AutoModel), which has no
    # lm_head -- only the input embedding table.
    embed_params = V * h

    # attention: q,k,v,o projections (no bias in Qwen3) + q_norm/k_norm (RMSNorm, head_dim each)
    q_params = h * (n_heads * head_dim)
    k_params = h * (n_kv * head_dim)
    v_params = h * (n_kv * head_dim)
    o_params = (n_heads * head_dim) * h
    qk_norm_params = 2 * head_dim  # q_norm + k_norm RMSNorm weights
    attn_params_per_layer = q_params + k_params + v_params + o_params + qk_norm_params

    # mlp: gate, up, down (no bias)
    mlp_params_per_layer = h * i + h * i + i * h

    # 2 RMSNorm per layer (input_layernorm, post_attention_layernorm) + final norm
    norm_params_per_layer = 2 * h
    final_norm_params = h

    per_layer = attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer
    dense_total = embed_params + L * per_layer + final_norm_params

    router_params_per_layer = h * n_experts
    extra_expert_params = (n_experts - 1) * mlp_params_per_layer * last_n_layers
    extra_router_params = router_params_per_layer * last_n_layers

    upcycled_total = dense_total + extra_expert_params + extra_router_params
    active_extra_expert_params = 0  # active experts count == top_k, same as dense's 1 "expert"
    upcycled_active = dense_total + extra_router_params + active_extra_expert_params

    return {
        "dense_total": dense_total,
        "upcycled_total": upcycled_total,
        "upcycled_active": upcycled_active,
        "mlp_params_per_layer": mlp_params_per_layer,
    }
