"""OJ-079: torch.compile of the decoder layers (openjev/compile.py) and bench_trials'
torch_compile param; plus bench --validate batching like train_prism (length grouping).

off must be today's eager path bit for bit. Inductor on CPU (C++ backend) gives logits
within tolerance on a tiny Gemma 4; the training path (LoRA + grad checkpointing + KV
sharing) is checked with the aot_eager backend (same dynamo/AOT graphs, exact, fast)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import bench_trials as bt  # noqa: E402
import train_prism as tp  # noqa: E402
from openjev import add_special_tokens  # noqa: E402
from openjev import compile as tc  # noqa: E402
from openjev.prism import PrismOpenJev, prism_collate  # noqa: E402
from test_attn_backends import rec  # noqa: E402

FIX = str(ROOT / "tests/fixtures/*.jsonl")


def tiny(tok, n_layers=4, kv_shared=2, lora=True, ckpt=True):
    from transformers import Gemma4TextConfig, Gemma4TextModel
    torch.manual_seed(0)
    types = (["full_attention", "sliding_attention", "sliding_attention", "full_attention"]
             if n_layers == 4 else ["sliding_attention"] * (n_layers - 1) + ["full_attention"])
    cfg = Gemma4TextConfig(vocab_size=len(tok) + 64, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=16, sliding_window=8, vocab_size_per_layer_input=100,
                           hidden_size_per_layer_input=8, num_kv_shared_layers=kv_shared,
                           layer_types=types)
    cfg._attn_implementation = "sdpa"
    m = PrismOpenJev(Gemma4TextModel(cfg))
    add_special_tokens(tok, m)
    if ckpt:
        tp.set_grad_ckpt(m, True)
    if lora:
        from peft import LoraConfig, get_peft_model
        m.backbone = get_peft_model(m.backbone, LoraConfig(r=4, lora_alpha=8, target_modules=tp.LORA_TARGETS))
    from openjev.schema import SPECIAL_TOKENS
    m.set_template_ids(tok.convert_tokens_to_ids(SPECIAL_TOKENS))
    return m


def batches(tok):
    return [prism_collate(tok, [rec(["billing", "bug", "sales"], "Hi, refund please! The app crashed."),
                                rec(["bug", "sales"], "No.")]),
            prism_collate(tok, [rec(["bug", "sales"], "a much longer state " * (3 + j)) for j in range(3)])]


def fwd_bwd(m, b):
    m.zero_grad(set_to_none=True)
    out = m.flag_logits(b)
    out.sum().backward()
    return out.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def test_off_is_bit_identical(tok):
    m = tiny(tok).train()
    bs = batches(tok)
    ref = [fwd_bwd(m, b) for b in bs]
    assert tc.set_torch_compile(m, "off") == "off" and tc.current(m) == "off"
    assert all(getattr(l, "_compiled_call_impl", None) is None for l in tc.decoder_layers(m))
    for (lo, g), b in zip(ref, bs):
        lo2, g2 = fwd_bwd(m, b)
        assert torch.equal(lo, lo2) and g.keys() == g2.keys()
        assert all(torch.equal(g[k], g2[k]) for k in g)
    with pytest.raises(ValueError):
        tc.set_torch_compile(m, "max-autotune")


def test_compile_training_path_aot_eager(tok, monkeypatch):
    """LoRA + grad checkpointing + KV-shared layers + mixed head dims, two shapes: compiled
    logits and grads equal eager; graphs stop growing once the shapes repeat; unset = eager."""
    m = tiny(tok).train()
    bs = batches(tok)
    ref = [fwd_bwd(m, b) for b in bs]
    monkeypatch.setenv("OJ_COMPILE_BACKEND", "aot_eager")
    tc.set_torch_compile(m, "default")
    try:
        assert tc.current(m) == "default"
        assert all(l._compiled_call_impl is not None for l in tc.decoder_layers(m))
        for _ in range(2):
            for (lo, g), b in zip(ref, bs):
                lo2, g2 = fwd_bwd(m, b)
                assert torch.allclose(lo, lo2, atol=1e-6)
                assert all(torch.allclose(g[k], g2[k], atol=1e-6, rtol=1e-6) for k in g)
            s0 = tc.stats()
        lo2, _ = fwd_bwd(m, bs[0])
        assert tc.stats()["graphs"] == s0["graphs"]           # no recompile on a seen shape
    finally:
        tc.unset_torch_compile(m)
    assert tc.current(m) == "off"
    assert all(l._compiled_call_impl is None for l in tc.decoder_layers(m))
    lo3, g3 = fwd_bwd(m, bs[0])
    assert torch.equal(lo3, ref[0][0])


def test_inductor_cpu_logits_within_tolerance(tok):
    """Real Inductor (C++ on CPU), eval forward, a 2-layer tiny Gemma 4."""
    m = tiny(tok, n_layers=2, kv_shared=0, lora=False, ckpt=False).eval()
    b = batches(tok)[0]
    with torch.no_grad():
        ref = m.flag_logits(b)
        tc.set_torch_compile(m, "default")
        try:
            out = m.flag_logits(b)
            st = tc.stats()
        finally:
            tc.unset_torch_compile(m)
    assert st["graphs"] >= 1
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4), (ref - out).abs().max()


def test_modes_reach_torch_compile(tok, monkeypatch):
    m = tiny(tok, n_layers=2, kv_shared=0, lora=False, ckpt=False)
    seen = []
    for l in tc.decoder_layers(m):
        monkeypatch.setattr(l, "compile", lambda **kw: seen.append(kw), raising=False)
    monkeypatch.delenv("OJ_COMPILE_BACKEND", raising=False)
    tc.set_torch_compile(m, "reduce-overhead")
    assert seen == [{"dynamic": True, "mode": "reduce-overhead"}] * 2
    seen.clear()
    m._torch_compile = "off"                                   # nothing was really compiled
    tc.set_torch_compile(m, "default")
    assert seen == [{"dynamic": True}] * 2
    import torch._dynamo
    assert torch._dynamo.config.recompile_limit >= tc.RECOMPILE_LIMIT
    m._torch_compile = "off"


class BackendCompilerFailed(RuntimeError):
    pass


def test_compile_failures_are_unsupported():
    triton = RuntimeError("No valid triton configs. OutOfResources: out of resource: shared memory, "
                          "Required: 65536 Hardware limit: 65536")
    bcf = BackendCompilerFailed("backend='inductor' raised: CppCompileError")
    for e in (triton, bcf, RuntimeError("torch._dynamo hit config.recompile_limit (64)")):
        assert bt.classify(e, "sdpa_default", "default") == "unsupported"
        assert bt.trial_status(e, {"fixed_mb": True, "micro_batch": 8}, None, "reduce-overhead") == "unsupported"
        assert bt.classify(e, "sdpa_default", "off") == "error"       # compile off: a real error
    oom = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
    assert bt.trial_status(oom, {"fixed_mb": True, "micro_batch": 8}, None, "default") == "infeasible_at_mb8"
    assert not tc.is_compile_error(ValueError("bad shape"))


def test_train_prism_falls_back_to_eager(tok):
    """A compiler error on the check micro-batch: back to eager, logged, training goes on."""
    m = tiny(tok, n_layers=2, kv_shared=0, lora=False, ckpt=False).train()
    args = tp.parse(["--bs", "2", "--torch-compile", "default", "--max-state-tokens", "48"])
    dp = type("DP", (), {"world": 1, "sync": lambda self, last: __import__("contextlib").nullcontext()})()

    def net(b):
        raise BackendCompilerFailed("backend='inductor' raised: No valid triton configs")
    recs = [rec(["bug", "sales"], "hello"), rec(["billing", "bug"], "x")]
    os.environ["OJ_COMPILE_BACKEND"] = "aot_eager"
    try:
        info = tp.compile_or_fallback(m, net, tok, recs, args, "cpu", dp)
        assert info["mode"] == "off" and "triton" in info["fallback"] and tc.current(m) == "off"
        info = tp.compile_or_fallback(m, m, tok, recs, args, "cpu", dp)   # works: stays compiled
        assert info["mode"] == "default" and info["fallback"] is None and tc.current(m) == "default"
        assert all(p.grad is None for p in m.parameters())
    finally:
        os.environ.pop("OJ_COMPILE_BACKEND", None)
        tc.unset_torch_compile(m)
        import torch._dynamo
        torch._dynamo.config.suppress_errors = False


def run(cmd, env=None):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
    return r


def test_bench_torch_compile_trials(tmp_path, tok):
    """bench --bench-lengths: no param == torch_compile=off exactly; a compiled trial records
    compile_s / compile_graphs apart from timing, builds no graph in the timed buckets, gives
    the same losses within tolerance, and the next (off) trial is eager again."""
    from test_bench_lengths import tiny_bench_set
    from test_optuna_loop import tiny_gemma
    base = tiny_gemma(tmp_path, tok)
    bl = tiny_bench_set(tmp_path, base)
    frozen = json.loads(bl.read_text())
    bench = {"global_batch": 2, "micro_batch": 1, "warmup_steps": 1, "timed_steps": 1,
             "bench_set_sha": frozen["sha"]}
    ps = [{}, {"torch_compile": "off"}, {"torch_compile": "default"}, {"torch_compile": "off"}]
    trials = [{"trial_number": i, "params": {"dtype": "fp32", **p}, "bench": bench} for i, p in enumerate(ps)]
    tf, res = tmp_path / "trials.json", tmp_path / "res.json"
    tf.write_text(json.dumps(trials))
    env = {**os.environ, "OJ_COMPILE_BACKEND": "aot_eager"}
    run([sys.executable, str(ROOT / "bench_trials.py"), "--trials", str(tf), "--out", str(res),
         "--model", str(base), "--cpu", "--train", FIX, "--max-state-tokens", "48",
         "--bench-lengths", str(bl)], env)
    r = {x["trial_number"]: x for x in json.loads(res.read_text())["results"]}
    assert all(x["status"] == "ok" for x in r.values()), r
    keys = ("loss_start", "loss_guard", "loss_end", "train_loss_first", "train_loss_last", "data_ids_sha")
    for k in keys:
        assert r[0][k] == r[1][k] == r[3][k], k
    assert r[0]["torch_compile"] == r[1]["torch_compile"] == r[3]["torch_compile"] == "off"
    assert "compile_s" not in r[0] and "compile_s" not in r[3]
    c = r[2]
    assert c["torch_compile"] == "default" and c["compile_s"] > 0 and c["compile_graphs"] >= 1
    assert c["compile_shapes"] >= 1 and c["graphs_in_timed_loop"] == 0
    assert isinstance(c["compile_recompiles"], int) and 0 <= c["compile_recompiles"] <= c["compile_graphs"]
    for k in ("loss_start", "loss_guard", "loss_end"):
        assert c[k] == pytest.approx(r[1][k], abs=1e-5), k


# ---------------------------------------------------------------- B: --validate batching

def test_validate_stream_groups_like_train_prism(tok):
    from openjev.data import read_jsonl
    from openjev.packing import request_lengths
    import glob
    recs = [r for p in sorted(glob.glob(FIX)) for r in read_jsonl(p)]
    args = tp.parse(["--max-state-tokens", "64"])
    assert args.length_grouping == "on" and args.packing == "off" and args.pack_len == 1024
    mb, chunk = 2, 3
    args.group_chunk = chunk
    assert bt.validate_n_refs(10, mb, args) == 12 and bt.validate_n_refs(12, mb, args) == 12
    n = mb * chunk
    s = bt.validate_stream(tok, recs[:n], args, mb, 64)
    out = s.take(n)
    assert sorted(r.id for r in out) == sorted(r.id for r in recs[:n])     # every record once
    lens = request_lengths(tok, out, 64)
    srt = sorted(request_lengths(tok, recs[:n], 64))
    runs = [lens[i:i + mb] for i in range(0, n, mb)]
    assert all(rn == sorted(rn) for rn in runs)                             # sorted inside a mb
    assert sorted(tuple(rn) for rn in runs) == sorted(tuple(srt[i:i + mb]) for i in range(0, n, mb))
    s2 = bt.validate_stream(tok, recs[:n], args, mb, 64)
    assert [r.id for r in s2.take(n)] == [r.id for r in out]               # same per config
    off = tp.parse(["--length-grouping", "off"])
    s3 = bt.validate_stream(tok, recs[:n], off, mb, 64)
    assert [r.id for r in s3.take(n)] == [r.id for r in recs[:n]]          # frozen order
    assert bt.validate_n_refs(10, mb, off) == 10
    cur = tp.parse(["--length-curriculum", "on"])                           # curriculum: no grouping
    assert not bt.validate_grouped(cur)


def test_bench_torch_compile_under_ddp(tmp_path, tok):
    """ddp=on (2 gloo ranks via OJ_DDP_NPROC) with torch_compile: the pre-pass under no_sync,
    then real all-reduces; same records and losses as the eager DDP trial."""
    from test_bench_lengths import tiny_bench_set
    from test_ddp import tiny_gemma
    base = tiny_gemma(tmp_path / "base", tok, kv_shared=False)
    bl = tiny_bench_set(tmp_path, base)
    frozen = json.loads(bl.read_text())
    bench = {"global_batch": 2, "micro_batch": 1, "warmup_steps": 1, "timed_steps": 1,
             "bench_set_sha": frozen["sha"]}
    ps = [{"ddp": "on", "torch_compile": "off"}, {"ddp": "on", "torch_compile": "default"}]
    trials = [{"trial_number": i, "params": {"dtype": "fp32", **p}, "bench": bench,
               "reload_keys": ["ddp", "dtype"]} for i, p in enumerate(ps)]
    tf, res = tmp_path / "trials.json", tmp_path / "res.json"
    tf.write_text(json.dumps(trials))
    env = {**os.environ, "OJ_DDP_NPROC": "2", "OJ_COMPILE_BACKEND": "aot_eager"}
    run([sys.executable, str(ROOT / "bench_trials.py"), "--trials", str(tf), "--out", str(res),
         "--model", str(base), "--cpu", "--train", FIX, "--max-state-tokens", "48",
         "--bench-lengths", str(bl)], env)
    r = {x["trial_number"]: x for x in json.loads(res.read_text())["results"]}
    off, on = r[0], r[1]
    assert off["status"] == on["status"] == "ok", r
    assert off["world"] == on["world"] == 2 and on["torch_compile"] == "default"
    assert on["compile_s"] > 0 and on["compile_graphs"] >= 1 and on["graphs_in_timed_loop"] == 0
    assert on["data_ids_sha"] == off["data_ids_sha"]
    for k in ("loss_start", "loss_guard", "loss_end"):
        assert on[k] == pytest.approx(off[k], abs=1e-5), k
