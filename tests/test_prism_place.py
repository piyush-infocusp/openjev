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
