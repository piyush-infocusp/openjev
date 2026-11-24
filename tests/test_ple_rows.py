"""Gemma 4 template PLE rows are saved (ple_rows.pt) and restored on every --init build.

Gemma 4 has no spare embedding rows, so adding the 7 template tokens also resizes the
Per-Layer-Embedding table, and the template tokens read their own (frozen) PLE rows. A fresh
SFT build draws them in add_special_tokens (mean + seeded noise); an --init build (RL, reload)
gets the resize's random rows unless the checkpoint carries them. Tiny Gemma 4 with PLE, CPU.
"""
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import eval_prism as ep
import train_prism as tp
from openjev.data import read_jsonl
from openjev.prism import PLE_ROWS, prism_collate
from openjev.schema import SPECIAL_TOKENS

ROOT = Path(__file__).resolve().parent.parent
FIX = sorted(str(p) for p in (ROOT / "tests" / "fixtures").glob("*.jsonl"))
MST = 32


def _records():
    return [r for p in FIX for r in read_jsonl(p)][:6]


def _logits_of(model, t):
    model.eval()
    with torch.no_grad():
        return model.flag_logits(prism_collate(t, _records(), MST)).float()


def _ple_rows(model, t):
    w = model.get_per_layer_input_embeddings().weight
    return torch.stack([w[i].detach().clone() for i in t.convert_tokens_to_ids(SPECIAL_TOKENS)])


def _seeded_build(args):
    random.seed(args.seed)               # as train_prism.main's load()
    torch.manual_seed(args.seed)
    model, t, _ = tp.build(args, "cpu")
    return model, t


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Bit-level comparisons across processes: the CPU matmul reduction order depends on the
    thread count, and other tests (bench_trials) change it. The probe subprocess pins 1 too."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def sft(tmp_path_factory, one_thread):
    """An SFT checkpoint of a tiny Gemma 4 whose LoRA B, flag head and template delta are
    random. Returns (base, ckpt, training-time logits, training-time template PLE rows)."""
    from transformers import AutoTokenizer, Gemma4TextConfig, Gemma4TextModel
    # Its own tokenizer, without the template tokens, so the build takes the PLE resize path.
    tok = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
    root = tmp_path_factory.mktemp("ple")
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=16, sliding_window=16, vocab_size_per_layer_input=len(tok),
                           hidden_size_per_layer_input=4, num_kv_shared_layers=0,
                           pad_token_id=tok.pad_token_id or 0,
                           layer_types=["sliding_attention", "full_attention"])
    base = root / "base"
    Gemma4TextModel(cfg).save_pretrained(base)
    tok.save_pretrained(base)
    args = tp.parse(["--model", str(base), "--cpu", "--max-state-tokens", str(MST),
                     "--lora-r", "4", "--lora-alpha", "8"])
    model, t = _seeded_build(args)
    text = model._text_backbone()
    assert text.config.vocab_size_per_layer_input >= len(t)    # the PLE table was resized
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n or n.startswith("flag_head") or n == "template_delta":
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    logits, rows = _logits_of(model, t), _ple_rows(model, t)
    ckpt = root / "sft"
    tp.save(model, t, args, ckpt)
    return base, ckpt, logits, rows


def _init_args(base, ckpt, seed):
    return tp.parse(["--model", str(base), "--cpu", "--max-state-tokens", str(MST),
                     "--algo", "exact", "--init", str(ckpt), "--seed", str(seed)])


INIT_PROBE = r"""
import random, sys, torch
sys.path.insert(0, sys.argv[1])
import train_prism as tp
from openjev.data import read_jsonl
from openjev.prism import prism_collate
from openjev.schema import SPECIAL_TOKENS
base, ckpt, seed, out, mst = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
args = tp.parse(["--model", base, "--cpu", "--max-state-tokens", str(mst), "--algo", "exact",
                 "--init", ckpt, "--seed", seed])
torch.set_num_threads(1)
random.seed(args.seed); torch.manual_seed(args.seed)
model, tok, _ = tp.build(args, "cpu")
model.eval()
recs = [r for p in sys.argv[7:] for r in read_jsonl(p)][:6]
w = model.get_per_layer_input_embeddings().weight
with torch.no_grad():
    logits = model.flag_logits(prism_collate(tok, recs, mst)).float()
rows = torch.stack([w[i].clone() for i in tok.convert_tokens_to_ids(SPECIAL_TOKENS)])
torch.save({"logits": logits, "rows": rows}, out)
"""


def _probe(base, ckpt, seed, out):
    r = subprocess.run([sys.executable, "-c", INIT_PROBE, str(ROOT), str(base), str(ckpt),
                        str(seed), str(out), str(MST), *FIX], cwd=ROOT, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return torch.load(out), r.stderr


def test_save_writes_ple_rows_and_slim_adapter(sft):
    base, ckpt, _, rows = sft
    saved = torch.load(ckpt / PLE_ROWS)
    assert len(saved) == len(SPECIAL_TOKENS)
    assert torch.equal(torch.stack(list(saved.values())), rows)
    with safe_open(str(ckpt / "adapter_model.safetensors"), "pt") as f:
        keys = list(f.keys())
    assert keys and all("lora_" in k for k in keys), [k for k in keys if "lora_" not in k]
    assert not any("embed_tokens" in k for k in keys)
    # the row files hold rows, not a view of the whole table
    assert (ckpt / PLE_ROWS).stat().st_size < 64 * 1024
    assert (ckpt / "embeddings.pt").stat().st_size < 64 * 1024


@pytest.mark.parametrize("seed", [0, 123])
def test_fresh_process_init_build_is_exact_for_any_seed(sft, tmp_path, seed):
    base, ckpt, logits, rows = sft
    got, _ = _probe(base, ckpt, seed, tmp_path / "probe.pt")
    assert torch.equal(got["rows"], rows)
    assert (got["logits"] - logits).abs().max() <= 1e-6


def test_without_ple_rows_init_build_differs_and_warns(sft, tmp_path):
    """The check above is not vacuous: an old checkpoint (no ple_rows.pt) gets the resize's
    random rows, which depend on the seed and move the logits."""
    base, ckpt, logits, rows = sft
    old = tmp_path / "old"
    shutil.copytree(ckpt, old)
    (old / PLE_ROWS).unlink()
    a, err = _probe(base, old, 0, tmp_path / "a.pt")
    b, _ = _probe(base, old, 123, tmp_path / "b.pt")
    assert f"has no {PLE_ROWS}" in err
    assert not torch.equal(a["rows"], rows) and not torch.equal(a["rows"], b["rows"])
    assert (a["logits"] - b["logits"]).abs().max() >= 1e-3
    assert (a["logits"] - logits).abs().max() >= 1e-3


def test_frozen_ref_kl0_and_init_policy_equals_sft(sft):
    """After an --init build the step-0 policy IS the SFT policy (incl. its PLE rows), and
    the frozen KL reference gives KL = 0 at step 0."""
    base, ckpt, logits, _ = sft
    args = _init_args(base, ckpt, 7)
    model, t = _seeded_build(args)
    b = prism_collate(t, _records(), MST)
    ref = tp.FrozenRef(model)
    with torch.no_grad():
        l0 = model.flag_logits(b)
    r0 = ref.logits(b, "cpu")
    _, st = tp.exact_pg_loss(l0, r0, b, args)
    assert st["kl"] == pytest.approx(0.0, abs=1e-7)
    assert (l0.float() - logits).abs().max() <= 1e-6


def test_old_format_adapter_with_embedding_still_loads(sft, tmp_path):
    """A checkpoint saved before save_embedding_layers=False (adapter carries the whole
    resized embed_tokens) still loads to the same logits."""
    base, ckpt, logits, _ = sft
    model, t = _seeded_build(_init_args(base, ckpt, 0))
    old = tmp_path / "old_adapter"
    shutil.copytree(ckpt, old)
    model.backbone.save_pretrained(old, save_embedding_layers=True)
    with safe_open(str(old / "adapter_model.safetensors"), "pt") as f:
        assert any("embed_tokens" in k for k in f.keys())
    m2, t2 = _seeded_build(_init_args(base, old, 5))
    assert (_logits_of(m2, t2) - logits).abs().max() <= 1e-6


def test_eval_prism_loads_new_checkpoint_exactly(sft):
    _, ckpt, logits, rows = sft
    model, t, _ = ep.load_prism(ckpt, "cpu")
    assert torch.equal(_ple_rows(model, t), rows)
    assert (_logits_of(model, t) - logits).abs().max() <= 1e-6


def test_eval_prism_old_checkpoint_replays_and_warns(sft, tmp_path):
    """Fallback kept for checkpoints saved before the fix: seeded replay, with a warning."""
    _, ckpt, logits, rows = sft
    old = tmp_path / "old"
    shutil.copytree(ckpt, old)
    (old / PLE_ROWS).unlink()
    with pytest.warns(UserWarning, match=f"has no {PLE_ROWS}"):
        model, t, _ = ep.load_prism(old, "cpu")
    assert torch.equal(_ple_rows(model, t), rows)
    assert (_logits_of(model, t) - logits).abs().max() <= 1e-5


def test_rl_checkpoint_carries_the_sft_rows(sft, tmp_path):
    """RL (--init) saves the pasted SFT rows again, so RL best/ reloads with the same rows."""
    base, ckpt, _, rows = sft
    args = _init_args(base, ckpt, 3)
    model, t = _seeded_build(args)
    rl = tmp_path / "rl"
    tp.save(model, t, args, rl)
    assert torch.equal(torch.stack(list(torch.load(rl / PLE_ROWS).values())), rows)
    m2, t2, _ = ep.load_prism(rl, "cpu")
    assert torch.equal(_ple_rows(m2, t2), rows)
    assert json.loads((rl / "prism_config.json").read_text())["args"]["init"] == str(ckpt)
