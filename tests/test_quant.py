"""OJ-075: quantized base weights (bitsandbytes int8 / nf4) on a tiny Gemma 4.

CPU-safe: quant=none never touches the quantizer and gives the same tensors and logits as a
build without the flag; the targets are only the decoder-layer LoRA projections; the bench
reload handler. CUDA only: int8 / nf4 builds (real bnb layers under PEFT's bnb LoRA layers,
same trainable init as quant=none, logits close to fp16, gradients reach LoRA)."""
import glob
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train_prism as tp  # noqa: E402
from openjev import prism  # noqa: E402
from openjev.data import read_jsonl  # noqa: E402
from openjev.prism import prism_collate  # noqa: E402

# Tolerances: max |a - b| / max |b| (relative). Measured with bitsandbytes 0.50.2 on its CPU
# backend (fp32 compute; no GPU on the dev box); CUDA fp16 uses the same bounds, which carry
# margin for fp16 rounding:
#   LAYER_TOL   each quantized layer vs its fp layer, same random input (max over layers):
#               measured int8 0.014, nf4 0.115
#   WIRING_TOL  whole model vs quant=none with the DEQUANTIZED weights copied in, so only the
#               kernels differ: nf4 0.0 (dequant + the same matmul); int8 0.148 (LLM.int8() also
#               quantizes the activations to int8, so this is not weight-only)
#   E2E_TOL     whole model vs quant=none (fp weights): measured int8 0.148, nf4 0.69. A random
#               tiny Gemma amplifies weight noise (attention scaling 1.0 with q/k norm is very
#               sharp, nf4 is ~10% noise per layer), so this is a loose sanity bound only.
LAYER_TOL = {"int8": 0.05, "nf4": 0.2}
WIRING_TOL = {"int8": 0.5, "nf4": 1e-2}
E2E_TOL = {"int8": 0.5, "nf4": 1.5}


def tiny_gemma(tmp_path, tok):
    """Hidden 64 and intermediate 128 (multiples of the int8 / 4-bit block sizes)."""
    from transformers import Gemma4TextConfig, Gemma4TextModel
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=16, sliding_window=16, vocab_size_per_layer_input=len(tok),
                           hidden_size_per_layer_input=8, num_kv_shared_layers=0,
                           pad_token_id=tok.pad_token_id or 0,
                           layer_types=["sliding_attention", "full_attention"])
    base = tmp_path / "base"
    Gemma4TextModel(cfg).save_pretrained(base)
    tok.save_pretrained(base)
    return base


def build(base, quant, device="cpu", dtype="fp32", drop_quant_attr=False):
    argv = ["--model", str(base), "--lora-r", "4", "--lora-alpha", "8", "--dtype", dtype,
            "--quant", quant] + (["--cpu"] if device == "cpu" else [])
    args = tp.parse(argv)
    if drop_quant_attr:                   # an args namespace from before OJ-075
        del args.quant
    random.seed(0)
    torch.manual_seed(0)
    model, tok, _ = tp.build(args, device)
    return model.eval(), tok


def batch(tok, device):
    recs = [r for p in sorted(glob.glob(str(ROOT / "tests/fixtures/*.jsonl")))
            for r in list(read_jsonl(p))[:2]]
    return prism_collate(tok, recs[:6], 48).to(device)


def logits_of(model, tok, device):
    b = batch(tok, device)
    with torch.no_grad(), tp.autocast(device, model):
        return model.flag_logits(b).float()


def trainables(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


# ---------------------------------------------------------------- CPU-safe

def test_quant_none_is_identical_and_never_quantizes(tmp_path, tok, monkeypatch):
    base = tiny_gemma(tmp_path, tok)
    old_model, old_tok = build(base, "none", drop_quant_attr=True)
    ref_logits, ref_sd = logits_of(old_model, old_tok, "cpu"), old_model.state_dict()

    def boom(*a, **k):
        raise AssertionError("quant=none must not call quantize_linears")
    monkeypatch.setattr(prism, "quantize_linears", boom)
    model, tok2 = build(base, "none")
    assert torch.equal(logits_of(model, tok2, "cpu"), ref_logits)
    sd = model.state_dict()
    assert sd.keys() == ref_sd.keys()
    assert all(torch.equal(sd[k], ref_sd[k]) for k in sd)
    text = model._text_backbone()
    assert not getattr(text, "is_loaded_in_4bit", False) and not getattr(text, "is_loaded_in_8bit", False)
    for name in prism.quant_targets(text):        # still the plain LoRA-wrapped nn.Linear
        assert type(text.get_submodule(name)) is nn.Linear


def test_quant_targets_are_decoder_lora_projections(tmp_path, tok):
    from transformers import Gemma4TextModel
    text = Gemma4TextModel.from_pretrained(tiny_gemma(tmp_path, tok))
    names = prism.quant_targets(text)
    assert names and all(n.startswith("layers.") for n in names)
    assert {n.rsplit(".", 1)[-1] for n in names} == set(prism.QUANT_TARGETS)
    assert not [n for n in names if "embed" in n or "per_layer" in n]
    # Linear layers that stay unquantized: PLE projections, per-layer gates.
    others = {n for n, m in text.named_modules() if type(m) is nn.Linear} - set(names)
    assert "per_layer_model_projection" in others
    assert any("per_layer_input_gate" in n for n in others)


def test_quant_args_and_bench_reload_handler():
    import bench_trials as bt
    assert "quant" in bt.RELOAD_PARAMS
    targs = tp.parse([])
    assert targs.quant == "none"
    assert bt.build_args(targs, {"quant": "nf4"}).quant == "nf4"
    assert bt.build_args(targs, {"quant": "int8", "ple_cpu": 1}).quant == "int8"
    with pytest.raises(ValueError):
        bt.build_args(targs, {"quant": "8bit"})
    with pytest.raises(SystemExit):
        tp.parse(["--quant", "nf4", "--lora-r", "0"])
    with pytest.raises(ValueError):
        prism.quantize_linears(nn.Module(), "fp4", "cpu")
    assert prism.quantize_linears(nn.Module(), "none", "cpu") == []


# ---------------------------------------------------------------- CUDA (bitsandbytes)

def dequantized(q):
    """The fp weight a bnb layer computes with."""
    import bitsandbytes.functional as bf
    w = q.weight
    if hasattr(w, "quant_state"):
        return bf.dequantize_4bit(w.data, w.quant_state)
    scb = w.SCB if w.SCB is not None else q.state.SCB   # a forward moves SCB to q.state
    return bf.int8_vectorwise_dequant(w.data, scb)


def rel_err(a, b) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def check_quant_build(base, quant, device):
    """The int8 / nf4 checks; returns {layer, wiring, e2e} relative errors."""
    import bitsandbytes as bnb
    from peft.tuners.lora import bnb as lora_bnb
    dtype = "fp16" if device == "cuda" else "fp32"
    ref, tok = build(base, "none", device, dtype)
    ref_logits, ref_train = logits_of(ref, tok, device), trainables(ref)
    model, tok = build(base, quant, device, dtype)
    text, ref_text = model._text_backbone(), ref._text_backbone()
    bnb_cls = bnb.nn.Linear4bit if quant == "nf4" else bnb.nn.Linear8bitLt
    lora_cls = lora_bnb.Linear4bit if quant == "nf4" else lora_bnb.Linear8bitLt
    names = prism.quant_targets(type(text)(text.config))   # same names, unquantized twin
    assert names
    err = {"layer": 0.0}
    for name in names:
        wrapped = text.get_submodule(name)
        assert isinstance(wrapped, lora_cls), (name, type(wrapped))   # PEFT's bnb LoRA layer
        q = wrapped.get_base_layer()
        assert isinstance(q, bnb_cls) and not q.weight.requires_grad
        assert not q.weight.is_floating_point()                      # uint8 / int8 storage
        if quant == "nf4":
            assert q.weight.quant_state.dtype == (torch.float16 if device == "cuda" else torch.float32)
        fp = ref_text.get_submodule(name).get_base_layer()
        x = torch.randn(16, fp.in_features, device=device, dtype=fp.weight.dtype)
        with torch.no_grad():
            err["layer"] = max(err["layer"], rel_err(q(x), fp(x)))
    assert err["layer"] < LAYER_TOL[quant], err
    # Unquantized: embedding, PLE table, flag head; every trainable is fp32.
    assert text.embed_tokens.weight.is_floating_point()
    assert text.embed_tokens_per_layer.weight.is_floating_point()
    assert model.flag_head.weight.dtype == torch.float32
    train = trainables(model)
    assert train.keys() == ref_train.keys()
    assert all(v.dtype == torch.float32 for v in train.values())
    # Same RNG use as quant=none: LoRA, flag head and template delta start equal.
    assert all(torch.equal(train[k], ref_train[k]) for k in train)
    q_logits = logits_of(model, tok, device)
    err["e2e"] = rel_err(q_logits, ref_logits)
    # Wiring: quant=none with the dequantized weights must give (nearly) the same logits.
    with torch.no_grad():
        for name in names:
            fp = ref_text.get_submodule(name).get_base_layer()
            fp.weight.copy_(dequantized(text.get_submodule(name).get_base_layer()).to(fp.weight.dtype))
    err["wiring"] = rel_err(q_logits, logits_of(ref, tok, device))
    del ref
    assert err["wiring"] < WIRING_TOL[quant], err
    assert err["e2e"] < E2E_TOL[quant], err
    # One backward: gradients reach LoRA B, nothing frozen gets a gradient.
    model.train()
    b = batch(tok, device)
    with tp.autocast(device, model):
        loss = tp.sft_loss(model.flag_logits(b), b)
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    return err


@pytest.mark.parametrize("quant", ["int8", "nf4"])
def test_quant_cuda(tmp_path, tok, quant):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    pytest.importorskip("bitsandbytes")
    err = check_quant_build(tiny_gemma(tmp_path, tok), quant, "cuda")
    print(f"{quant}: relative errors {err}")
