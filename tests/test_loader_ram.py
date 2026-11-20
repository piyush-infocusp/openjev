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


def test_convert_inplace_skips_trainable_params():
    m = torch.nn.Linear(4, 4).to(torch.bfloat16)
    extra = torch.nn.Linear(4, 4)            # fp32, trainable (like LoRA / flag head)
    m.add_module("lora", extra)
    m.weight.requires_grad_(False); m.bias.requires_grad_(False)
    prism.convert_inplace(m, torch.float16)
    assert m.weight.dtype == torch.float16
    assert extra.weight.dtype == torch.float32


def test_convert_inplace_casts_a_shared_parameter_once():
    a, b = torch.nn.Linear(4, 4).to(torch.bfloat16), torch.nn.Linear(4, 4).to(torch.bfloat16)
    b.weight = a.weight                                   # tied
    a.requires_grad_(False); b.requires_grad_(False)
    ref = a.weight.detach().to(torch.float16)
    m = torch.nn.Sequential(a, b)
    prism.convert_inplace(m, torch.float16)
    assert a.weight is b.weight and torch.equal(a.weight, ref)


def _tiny_base(tmp_path, tok):
    from transformers import Gemma4TextConfig, Gemma4TextModel
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=16, sliding_window=16, vocab_size_per_layer_input=len(tok),
                           hidden_size_per_layer_input=4, num_kv_shared_layers=0,
                           pad_token_id=tok.pad_token_id or 0,
                           layer_types=["sliding_attention", "full_attention"])
    base = tmp_path / "base"
    Gemma4TextModel(cfg).to(torch.bfloat16).save_pretrained(base)   # ships in bf16 like Gemma 4
    tok.save_pretrained(base)
    return base


def _check_dtypes(model, dtype):
    frozen = {p.dtype for p in model.parameters() if not p.requires_grad and p.is_floating_point()}
    assert frozen == {dtype}
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    assert model.template_delta.dtype == torch.float32
    assert model.get_input_embeddings().weight.dtype == dtype


def _forward(model, tok):
    from openjev.data import Record
    from openjev.prism import prism_collate
    from openjev.schema import Schema
    r = Record("r", "src", "test", "the answer is blue",
               Schema().choice("c", {"blue": "blue", "red": "red"}, "pick"), ["blue"])
    return model.flag_logits(prism_collate(tok, [r]))


def test_build_cpu_converts_on_host_to_requested_dtype(tmp_path, tok):
    import train_prism as tp
    base = _tiny_base(tmp_path, tok)
    model, _, dtype = tp.build(tp.parse(["--model", str(base), "--cpu", "--dtype", "fp16"]), "cpu")
    assert dtype == torch.float16
    _check_dtypes(model, torch.float16)


def test_build_cuda_path_converts_after_place(tmp_path, tok, monkeypatch):
    """r3 SIGKILL: on CUDA the bf16 weights must reach place() unconverted (no host
    conversion), resize / grad-ckpt / LoRA must work on bf16, and convert_inplace runs
    after place() on the frozen weights only. place() is stubbed to stay on CPU."""
    import train_prism as tp
    from openjev import prism as pr
    base = _tiny_base(tmp_path, tok)
    seen = {}

    def fake_place(model, device, ple_cpu):
        seen["frozen_at_place"] = {p.dtype for p in model.parameters() if not p.requires_grad}
        seen["device"] = device
        return model

    monkeypatch.setattr(tp, "place", fake_place)
    rss = []
    monkeypatch.setattr(tp, "log_rss", lambda s: rss.append(s))
    monkeypatch.setattr(pr, "log_rss", lambda s: rss.append(s))
    args = tp.parse(["--model", str(base), "--dtype", "fp16", "--grad-ckpt"])
    model, btok, dtype = tp.build(args, "cuda")
    assert seen == {"frozen_at_place": {torch.bfloat16}, "device": "cuda"}
    assert "after_convert" not in rss and rss[-3:] == ["after_resize", "after_place", "after_device_convert"]
    _check_dtypes(model, torch.float16)
    assert any("lora" in n for n, p in model.named_parameters() if p.requires_grad)
    model.train()
    out = _forward(model, btok)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert model.flag_head.weight.grad is not None


def test_template_resize_skips_hf_mean_resizing(tok, monkeypatch):
    """r3small: maxrss 28.9 GB at after_resize. Gemma 4 resizes the PLE table too, and HF
    mean_resizing builds fp32 copies + a covariance of it on the host. The resize must pass
    mean_resizing=False, and add_special_tokens sets the template rows of BOTH tables
    from chunked row stats (finite, near the old mean)."""
    from transformers import AutoTokenizer, Gemma4TextConfig, Gemma4TextModel
    from openjev.schema import SPECIAL_TOKENS, add_special_tokens
    # Fresh tokenizer: the session `tok` may already carry the special tokens (other tests
    # call add_special_tokens on it), which would make add_special_tokens a no-op here.
    tk = AutoTokenizer.from_pretrained(tok.name_or_path)
    cfg = Gemma4TextConfig(vocab_size=len(tk), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                           vocab_size_per_layer_input=len(tk), hidden_size_per_layer_input=4,
                           num_kv_shared_layers=0, layer_types=["sliding_attention", "full_attention"])
    torch.manual_seed(0)
    model = prism.PrismOpenJev(Gemma4TextModel(cfg).to(torch.bfloat16))
    seen = []
    real = model.backbone._get_resized_embeddings

    def spy(emb, n, pad, mean_resizing=True):
        seen.append(mean_resizing)
        return real(emb, n, pad, mean_resizing)

    monkeypatch.setattr(model.backbone, "_get_resized_embeddings", spy)
    ple_old = model.get_per_layer_input_embeddings().weight[:len(tk)].float().mean(0)
    add_special_tokens(tk, model)
    assert seen == [False, False]                           # input embeddings + PLE
    ids = tk.convert_tokens_to_ids(SPECIAL_TOKENS)
    ple = model.get_per_layer_input_embeddings().weight
    assert ple.shape[0] >= len(tk) and ple.dtype == torch.bfloat16
    rows = ple[ids].float()
    assert torch.isfinite(rows).all()
    assert (rows - ple_old).abs().mean() < ple[:len(tk) - len(ids)].float().std() * 0.5


def test_row_stats_match_torch():
    from openjev.schema import _row_stats
    w = torch.randn(1000, 7) * 3 + 1
    mean, std = _row_stats(w, 900, chunk=64)
    assert torch.allclose(mean, w[:900].mean(0), atol=1e-5)
    assert torch.allclose(std, w[:900].std(0).mean(), atol=1e-5)
