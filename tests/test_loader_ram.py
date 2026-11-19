"""Regression (optuna r2 SIGKILL): fp16/fp32 must be loaded as bf16 and then
converted per tensor, never by from_pretrained on the whole multimodal model."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openjev import prism


def test_fp16_request_loads_bf16_then_converts(monkeypatch):
    from transformers import Gemma4TextConfig, Gemma4TextModel
    cfg = Gemma4TextConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                           vocab_size_per_layer_input=64, hidden_size_per_layer_input=4,
                           num_kv_shared_layers=0, layer_types=["sliding_attention", "full_attention"])
    seen = {}

    class Fake:
        @staticmethod
        def from_pretrained(name, dtype=None, **kw):
            seen["dtype"] = dtype
            return Gemma4TextModel(cfg).to(dtype)

    import transformers
    monkeypatch.setattr(transformers, "AutoModel", Fake)
    text = prism.gemma_text_backbone("x", dtype=torch.float16)
    assert seen["dtype"] == torch.bfloat16
    assert {p.dtype for p in text.parameters()} == {torch.float16}
    assert text.config._attn_implementation == "sdpa"


def test_inplace_cast_is_bit_exact_and_reuses_storage():
    t = (torch.randn(3, 1000) * 5).to(torch.bfloat16)
    ref = t.to(torch.float16)
    ptr = t.data_ptr()
    out = prism._cast(t, torch.float16, chunk=257)      # chunk not dividing numel
    assert out.dtype == torch.float16 and out.shape == t.shape
    assert out.data_ptr() == ptr                         # no second full copy
    assert torch.equal(out, ref)
