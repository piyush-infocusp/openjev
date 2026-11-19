"""Regression: with the PLE table on CPU, every tensor of that module (weight
AND buffers such as embed_scale) must be on CPU, and the rest on the device.
Kaggle v6 failed with 'found at least two devices, cuda:0 and cpu'. The 'meta'
device stands in for CUDA so this runs without a GPU."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openjev.prism import PrismOpenJev
from train_prism import place


def test_ple_module_fully_on_cpu():
    from transformers import Gemma4TextConfig, Gemma4TextModel
    cfg = Gemma4TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=16, vocab_size_per_layer_input=128,
                           hidden_size_per_layer_input=4, num_kv_shared_layers=0,
                           layer_types=["sliding_attention", "full_attention"])
    m = place(PrismOpenJev(Gemma4TextModel(cfg)), "meta", ple_cpu=True)
    ple = m.backbone.embed_tokens_per_layer
    tensors = list(ple.parameters()) + list(ple.buffers())
    assert tensors and all(t.device.type == "cpu" for t in tensors)
    assert m.flag_head.weight.device.type == "meta"
    assert m.backbone.embed_tokens.weight.device.type == "meta"


def test_ple_host_to_device_pinned_non_blocking(monkeypatch):
    """The looked-up PLE rows (ple_cpu=1) go CPU -> CUDA through pinned memory with
    non_blocking=True, and the values are unchanged. A spy stands in for CUDA on CPU."""
    from openjev import prism
    x = torch.randn(2, 5, 3, 4)
    assert prism.host_to_device(x, "cpu") is x                  # same device: no copy
    calls = {}
    real_to = torch.Tensor.to

    def fake_pin(self):
        calls["pinned"] = True
        return self.clone()

    def fake_to(self, *a, **k):
        if a and torch.device(a[0]).type == "cuda":
            calls["non_blocking"] = k.get("non_blocking")
            return self.clone()                                   # "on the GPU"
        return real_to(self, *a, **k)
    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin)
    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    y = prism.host_to_device(x, torch.device("cuda"))
    assert calls == {"pinned": True, "non_blocking": True}
    assert torch.equal(y, x)


def test_ple_host_to_device_real_cuda():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from openjev import prism
    x = torch.randn(2, 7, 3, 4)
    y = prism.host_to_device(x, "cuda")
    torch.cuda.synchronize()
    assert y.is_cuda and torch.equal(y.cpu(), x)
