import copy

import pytest
import torch
from transformers import AutoConfig

from openjev.model import OpenJev
from openjev.moe import UpcycledMoE, aux_loss, param_counts, projected_counts, upcycle

MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _questions(n_tokens=6):
    """A minimal readouts structure: one question with a few flag positions."""
    return [[{"type": "noul", "name": "q", "flags": list(range(n_tokens))}]]


def test_upcycle_identity_at_init():
    torch.manual_seed(0)
    dense = OpenJev(MODEL)
    moe_model = OpenJev(MODEL)
    moe_model.backbone.load_state_dict(dense.backbone.state_dict())
    moe_model.flag_head.load_state_dict(dense.flag_head.state_dict())
    upcycle(moe_model.backbone, n_experts=4, last_n_layers=2, top_k=1)

    input_ids = torch.randint(0, dense.backbone.config.vocab_size, (1, 6))
    attention_mask = torch.ones_like(input_ids)
    readouts = _questions(6)

    dense.eval()
    moe_model.eval()
    with torch.no_grad():
        out_dense = dense(input_ids, attention_mask, readouts)
        out_moe = moe_model(input_ids, attention_mask, readouts)

    logits_dense = out_dense[0][0]["logits"]
    logits_moe = out_moe[0][0]["logits"]
    assert torch.allclose(logits_dense, logits_moe, atol=1e-5)


def test_param_counts():
    model = OpenJev(MODEL)
    before = param_counts(model)
    # before upcycling there are no UpcycledMoE modules, so total==active and
    # equals the plain sum of all params
    real_total = sum(p.numel() for p in model.parameters())
    assert before["total"] == real_total
    assert before["active"] == real_total

    n_experts, last_n, top_k = 4, 2, 1
    layer = model.backbone.layers[-1].mlp
    mlp_params = sum(p.numel() for p in layer.parameters())
    upcycle(model.backbone, n_experts=n_experts, last_n_layers=last_n, top_k=top_k)
    after = param_counts(model)

    router_params_total = 0
    for m in model.backbone.modules():
        if isinstance(m, UpcycledMoE):
            router_params_total += sum(p.numel() for p in m.router.parameters())

    expected_total = before["total"] + (n_experts - 1) * mlp_params * last_n + router_params_total
    expected_active = before["active"] + router_params_total
    assert after["total"] == expected_total
    assert after["active"] == expected_active


def test_backward_and_expert_gradients():
    torch.manual_seed(1)
    model = OpenJev(MODEL)
    upcycle(model.backbone, n_experts=4, last_n_layers=2, top_k=1, )
    # raise router noise so routing doesn't collapse to a single expert
    for m in model.backbone.modules():
        if isinstance(m, UpcycledMoE):
            torch.nn.init.normal_(m.router.weight, std=1.0)

    n_tokens = 64
    input_ids = torch.randint(0, model.backbone.config.vocab_size, (1, n_tokens))
    attention_mask = torch.ones_like(input_ids)
    readouts = _questions(n_tokens)

    out = model(input_ids, attention_mask, readouts)
    logits = out[0][0]["logits"]
    loss = logits.sum() + aux_loss(model.backbone)
    loss.backward()

    moe_modules = [m for m in model.backbone.modules() if isinstance(m, UpcycledMoE)]
    assert moe_modules
    m = moe_modules[-1]
    assert m.router.weight.grad is not None

    experts_with_grad = sum(
        1 for e in m.experts
        if any(p.grad is not None and p.grad.abs().sum() > 0 for p in e.parameters())
    )
    assert experts_with_grad >= 2


def test_aux_loss_finite_and_near_minimum_at_uniform_routing():
    moe = UpcycledMoE(_FakeMLP(hidden=8), n_experts=4, top_k=1, router_noise=0.0)
    x = torch.randn(2, 16, 8)
    out = moe(x)
    assert torch.isfinite(moe.last_aux)
    # router weight is exactly zero (noise=0), so routing is uniform:
    # aux loss should sit at its minimum of 1.0
    assert moe.last_aux.item() >= 1.0 - 1e-5


def test_projected_counts_matches_real_tiny_model():
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL)
    real_total = sum(p.numel() for p in model.parameters())
    proj = projected_counts(model.config, n_experts=4, last_n_layers=2, top_k=1)
    assert proj["dense_total"] == real_total


def test_projected_counts_qwen3_0_6b():
    try:
        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:  # network unavailable
        pytest.skip(f"could not fetch Qwen3-0.6B config: {e}")

    proj_4_8 = projected_counts(config, n_experts=4, last_n_layers=8, top_k=1)
    proj_8_4 = projected_counts(config, n_experts=8, last_n_layers=4, top_k=1)

    print("Qwen3-0.6B dense total:", proj_4_8["dense_total"])
    print("Qwen3-0.6B upcycled (E=4,N=8) total/active:",
          proj_4_8["upcycled_total"], proj_4_8["upcycled_active"])
    print("Qwen3-0.6B upcycled (E=8,N=4) total/active:",
          proj_8_4["upcycled_total"], proj_8_4["upcycled_active"])

    assert proj_4_8["upcycled_total"] < 1e9


class _FakeMLP(torch.nn.Module):
    """Minimal stand-in for Qwen3MLP: has hidden_size and a forward(x)."""

    def __init__(self, hidden):
        super().__init__()
        self.hidden_size = hidden
        self.lin = torch.nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.lin(x)
