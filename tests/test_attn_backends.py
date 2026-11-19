"""Round-2 attention backends (openjev/attn_backends.py) give the same flag logits as
the default SDPA path on a tiny Gemma 4 with the real Gemma 4 features: GQA, KV-shared
layers, mixed head dims (full layers wider), a sliding window shorter than T, padding,
two questions and options of unequal token length."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openjev import Schema, add_special_tokens
from openjev import attn_backends as ab
from openjev.data import Record
from openjev.prism import PrismOpenJev, prism_collate, to_additive

CHUNK = 7  # does not divide T, so the last chunk is short


def rec(order, state):
    opts = {"billing": "Payments and refunds", "bug": "Bugs", "sales": "Buy more seats now"}
    s = (Schema().choice("dept", {k: opts[k] for k in order}, "Which team?")
         .noul("loud", "Has '!'?", {"true": "yes", "false": "no"}))
    return Record("r", "src", "test", state, s, [order[0], True])


@pytest.fixture(scope="module")
def tiny(tok):
    from transformers import Gemma4TextConfig, Gemma4TextModel
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=len(tok) + 64, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=16, sliding_window=8, vocab_size_per_layer_input=100,
                           hidden_size_per_layer_input=8, num_kv_shared_layers=2,
                           per_layer_config={"0": {"head_dim": 32}, "3": {"head_dim": 32}},
                           layer_types=["full_attention", "sliding_attention",
                                        "sliding_attention", "full_attention"])
    cfg._attn_implementation = "sdpa"
    m = PrismOpenJev(Gemma4TextModel(cfg)).eval()
    add_special_tokens(tok, m)
    return m


@pytest.fixture(scope="module")
def batch(tok):
    b = prism_collate(tok, [rec(["billing", "bug", "sales"], "Hi, refund please! The app crashed."),
                            rec(["bug", "sales"], "No.")])
    assert (b.attention_mask[1] == 0).any()                 # padding present
    assert b.input_ids.shape[1] > 8 + CHUNK                  # window and chunks really cut
    return b


def test_model_has_the_gemma4_features(tiny):
    text = tiny.backbone
    assert {lay.self_attn.head_dim for lay in text.layers} == {16, 32}
    assert any(lay.self_attn.is_kv_shared_layer for lay in text.layers)
    assert all(lay.self_attn.num_key_value_groups == 2 for lay in text.layers)


def test_bool_masks_equal_additive(tiny, batch):
    add = PrismOpenJev.masks(tiny, batch)
    boo = ab.bool_masks(tiny, batch)
    for k in add:
        assert boo[k].dtype == torch.bool
        assert torch.equal(boo[k], add[k] == 0)
    again = PrismOpenJev.masks(tiny, batch)                  # the to_additive swap was undone
    assert again["full_attention"].dtype != torch.bool and torch.equal(again["full_attention"],
                                                                       add["full_attention"])
    import openjev.prism as prism
    assert prism.to_additive is to_additive


@pytest.mark.parametrize("name", ["sdpa_bool", "chunked_math"])
def test_backend_matches_sdpa_default(tiny, batch, name):
    with torch.no_grad():
        ref = tiny.flag_logits(batch)
        with ab.attn_backend(tiny, name, chunk=CHUNK):
            out = tiny.flag_logits(batch)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-4, rtol=0), (out - ref).abs().max()


def _grads(model, batch):
    model.zero_grad(set_to_none=True)
    model.flag_logits(batch).pow(2).sum().backward()
    g = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return g


@pytest.mark.parametrize("ckpt", [False, True])
def test_chunked_math_gradients_match(tiny, batch, ckpt):
    import train_prism as tp
    tiny.train()
    tp.set_grad_ckpt(tiny, ckpt)
    try:
        ref = _grads(tiny, batch)
        with ab.attn_backend(tiny, "chunked_math", chunk=CHUNK):
            got = _grads(tiny, batch)
    finally:
        tp.set_grad_ckpt(tiny, False)
        tiny.eval()
    assert ref.keys() == got.keys() and any("q_proj" in n for n in ref)
    for n in ref:   # 1e-4 relative to the largest entry (embedding grads reach ~1e3)
        err = (got[n] - ref[n]).abs().max() / max(1.0, ref[n].abs().max().item())
        assert err < 1e-4, (n, err)


def test_flex_matches_sdpa_default(tiny, batch):
    ok, why = ab.flex_available()
    if not ok:
        pytest.skip(f"flex_attention not available: {why}")
    with torch.no_grad():
        ref = tiny.flag_logits(batch)
        try:
            with ab.attn_backend(tiny, "flex"):
                out = tiny.flag_logits(batch)
        except ab.Unsupported as e:
            pytest.skip(f"flex unsupported here: {e}")
    assert torch.allclose(out, ref, atol=1e-4, rtol=0), (out - ref).abs().max()


def test_flex_training_on_cpu_is_unsupported(tiny, batch):
    import bench_trials
    if not ab.flex_available()[0]:
        pytest.skip("flex_attention not available")
    tiny.train()
    try:
        with ab.attn_backend(tiny, "flex"):
            with pytest.raises(ab.Unsupported) as e:
                tiny.flag_logits(batch).sum().backward()
    finally:
        tiny.zero_grad(set_to_none=True)
        tiny.eval()
    assert bench_trials.classify(e.value, "flex") == "unsupported"


def test_switch_back_restores_default(tiny):
    text = tiny._text_backbone()
    for name in ab.BACKENDS:
        with ab.attn_backend(tiny, name, label=True):
            assert ab.current(tiny) == name
        assert text.config._attn_implementation == "sdpa"
        assert "masks" not in tiny.__dict__ and "get_per_layer_inputs" not in text.__dict__
        assert ab.current(tiny) == "sdpa_default"
    with pytest.raises(ValueError):
        ab.set_attn_backend(tiny, "nope")


def test_labels_reach_the_profiler(tiny, batch):
    import bench_trials
    from torch.profiler import ProfilerActivity, profile
    with ab.attn_backend(tiny, "chunked_math", chunk=CHUNK, label=True), torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            tiny.flag_logits(batch)
    p = bench_trials.summarize_profile(prof.key_averages(), 1.0, 1)
    assert {"attention_fwd", "masks", "ple_lookup", "matmul_ops"} <= set(p["shares"])
    assert p["top_cpu"] and len(p["top_cpu"]) <= 25 and p["top_cuda"] == []
    assert not any(r[0].startswith("oj::") for r in p["top_cpu"])


def test_chunked_math_keeps_no_T2_probabilities():
    """With grad on, each chunk is checkpointed: what autograd saves for one attention
    call is O(T*D) (q, k, v, mask), not the O(T^2) probabilities SDPA math keeps."""
    import types
    from torch.nn.attention import SDPBackend, sdpa_kernel
    B, H, Hkv, T, D = 1, 4, 2, 256, 8
    mod = types.SimpleNamespace(head_dim=D, training=True, num_key_value_groups=H // Hkv)
    q = torch.randn(B, H, T, D, requires_grad=True)
    k = torch.randn(B, Hkv, T, D, requires_grad=True)
    v = torch.randn(B, Hkv, T, D, requires_grad=True)
    mask = torch.ones(B, 1, T, T, dtype=torch.bool).tril()

    def saved_bytes(fn):
        seen = {}

        def pack(t):
            seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()  # views count once
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            fn()
        return sum(seen.values())

    def ref():
        with sdpa_kernel(SDPBackend.MATH):
            kk, vv = (x.repeat_interleave(H // Hkv, 1) for x in (k, v))
            torch.nn.functional.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, scale=1.0)
    ab._state["chunk"] = 32
    try:
        chunked = saved_bytes(lambda: ab.chunked_attention_forward(mod, q, k, v, mask, scaling=1.0))
    finally:
        ab._state["chunk"] = ab.DEFAULT_CHUNK
    probs = B * H * T * T * 4
    assert saved_bytes(ref) >= probs > 2 * chunked, (saved_bytes(ref), chunked)
