"""Data parallel training (OJ-062) on CPU: 2 gloo processes under torchrun with a tiny random
Gemma 4 must end with the same weights as ONE process with the same global batch
(2 ranks x accum 2 == 1 process x accum 4), for SFT (packing + option sub-sampling), exact PG
and GRPO; local_sgd with K=1 equals it with a linear optimizer (SGD). The DDP model uses
KV-shared layers like E2B, so find_unused_parameters=False is checked on that layout."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIX = str(ROOT / "tests/fixtures/*.jsonl")

from openjev import dist as jdist  # noqa: E402


def tiny_gemma(path: Path, tok, kv_shared: bool = True) -> Path:
    """kv_shared: 3 layers, the last one KV-shared (no k/v projections, like E2B)."""
    from transformers import Gemma4TextConfig, Gemma4TextModel
    torch.manual_seed(0)
    layers = ["sliding_attention", "full_attention"] + (["full_attention"] if kv_shared else [])
    cfg = Gemma4TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=len(layers), num_attention_heads=2, num_key_value_heads=1,
                           head_dim=16, sliding_window=16, vocab_size_per_layer_input=len(tok),
                           hidden_size_per_layer_input=4, num_kv_shared_layers=int(kv_shared),
                           pad_token_id=tok.pad_token_id or 0, layer_types=layers)
    Gemma4TextModel(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """A fresh tokenizer: the session `tok` may carry tokens other tests added, which would
    change the tiny model (and these float-noise tolerances) with the test order."""
    from transformers import AutoTokenizer
    from conftest import MODEL_NAME
    return tiny_gemma(tmp_path_factory.mktemp("ddp") / "base", AutoTokenizer.from_pretrained(MODEL_NAME))


def run(cmd, env=None):
    e = {**os.environ, "OMP_NUM_THREADS": "1", **(env or {})}
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=e, timeout=900)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-4000:])
    return r.stdout


TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2"]


def train_args(base, out, *extra):
    return [str(ROOT / "train_prism.py"), "--model", str(base), "--cpu", "--train", FIX, "--val", FIX,
            "--bs", "2", "--grad-accum", "4", "--steps", "3", "--eval-every", "3", "--eval-max", "6",
            "--max-state-tokens", "48", "--lr", "1e-3", "--head-lr", "1e-2", "--grad-ckpt",
            "--packing", "on", "--pack-len", "128", "--option-subsample-k", "4",
            "--save-dir", str(out), *extra]


def weights(d: Path) -> dict:
    from safetensors.torch import load_file
    w = {f"lora.{k}": v for k, v in load_file(str(d / "adapter_model.safetensors")).items()}
    w.update({f"head.{k}": v for k, v in torch.load(d / "heads.pt").items()})
    w["template_delta"] = torch.load(d / "template_delta.pt")
    return w


def assert_close(one: dict, two: dict, atol=2e-5, worst=1e-4):
    """Equal within float noise (the all-reduce sums in another order): >= 99% of each tensor
    within atol, every element within `worst` (the runs move LoRA B by ~2e-3). Adam divides
    by sqrt(v), so an element whose true gradient is ~0 turns that noise into a step of up to
    lr: the flag-head bias is one (softmax is shift-invariant, its gradient is exactly 0), so
    it is not compared."""
    assert one.keys() == two.keys()
    for k in one:
        if k == "head.bias":
            continue
        d = (one[k] - two[k]).abs()
        bad = int((d > atol).sum())
        assert bad <= max(1, 0.01 * d.numel()) and d.max() <= worst, (k, bad, d.numel(), d.max().item())


def trained(w: dict):
    lora_b = [k for k in w if "lora_B" in k]
    assert lora_b and all(w[k].abs().max() > 1e-4 for k in lora_b)   # B starts at 0


# ---------------------------------------------------------------- sharding (no processes)

def test_assign_partition():
    costs = [100, 7, 50, 49, 3, 300, 12, 12]
    parts = [jdist.assign(costs, 2, r) for r in range(2)]
    assert sorted(parts[0] + parts[1]) == list(range(8)) and len(parts[0]) == len(parts[1]) == 4
    loads = sorted(sum(costs[i] for i in p) for p in parts)
    assert loads == [100 + 50 + 49 + 12, 300 + 12 + 7 + 3]          # LPT with equal counts
    assert parts == [jdist.assign(costs, 2, r) for r in range(2)]   # deterministic
    assert jdist.assign([5, 5, 1, 1], 2, 0) == [0, 2] and jdist.assign([5, 5, 1, 1], 2, 1) == [1, 3]
    assert jdist.assign(costs, 1, 0) == list(range(8))
    with pytest.raises(ValueError):
        jdist.assign(costs[:7], 2, 0)


def test_single_process_context_is_a_noop():
    dp = jdist.DataParallel("cpu", "local_sgd", 1)
    assert dp.world == 1 and not dp.ddp and not dp.local_sgd
    m = torch.nn.Linear(2, 1)
    assert dp.attach(m) is m and dp.mine([3, 1, 2]) == [0, 1, 2]
    assert dp.after_step() is False and dp.mean([1.5]) == [1.5] and dp.any(True)
    with pytest.raises(ValueError):
        jdist.DataParallel("cpu", "allreduce")


def test_kv_shared_layout_every_trainable_gets_a_grad(base, tok):
    """find_unused_parameters=False needs a gradient for EVERY trainable in every backward.
    Gemma 4's KV-shared layers build no k/v projections, so no LoRA sits idle."""
    import train_prism as tp
    from openjev.data import read_jsonl
    args = tp.parse(["--model", str(base), "--cpu", "--max-state-tokens", "48", "--grad-ckpt"])
    model, tk, _ = tp.build(args, "cpu")
    text = model._text_backbone()
    assert text.config.num_kv_shared_layers == 1
    assert not hasattr(text.layers[2].self_attn, "k_proj")
    recs = list(read_jsonl(sorted(Path(ROOT / "tests/fixtures").glob("*.jsonl"))[0]))[:2]
    b = tp.train_batch(tk, recs, args, "0:0")
    tp.sft_loss(model(b), b).backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_ignore_frozen_lets_ddp_wrap_a_mixed_device_module(tmp_path):
    """ple_cpu=1 on CUDA leaves the frozen PLE table on the CPU next to GPU trainables; DDP
    rejects parameters on two device types unless the frozen ones are ignored. Stand-in: a
    frozen 'meta' table next to CPU trainables, in a 1-process gloo group."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 1)
            self.embed_tokens_per_layer = torch.nn.Embedding(10, 4, device="meta")
            self.embed_tokens_per_layer.weight.requires_grad_(False)
            self.register_buffer("scale", torch.ones(1, device="meta"))

    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        with pytest.raises(ValueError, match="same type of devices"):
            DDP(M())
        m = M()
        assert set(jdist.ignore_frozen(m)) == {"embed_tokens_per_layer.weight", "scale"}
        net = DDP(m, broadcast_buffers=False, find_unused_parameters=False)
        assert net.module.embed_tokens_per_layer.weight.device.type == "meta"
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------- 2 ranks == 1 process

def test_ddp_two_ranks_equal_one_process(base, tmp_path):
    """SFT, packing + option sub-sampling (per-record RNG), grad-ckpt: 2 ranks x accum 2
    (no_sync on the first micro-step, token-balanced shares of whole micro-batches) vs one
    process x accum 4. Same weights, same rank-0 eval."""
    out1 = run([sys.executable] + train_args(base, tmp_path / "one"))
    out2 = run(TORCHRUN + train_args(base, tmp_path / "ddp"))
    assert '"dist": {"world": 2, "sync_mode": "ddp"' in out2 and '"accum_per_rank": 2' in out2
    one, two = weights(tmp_path / "one" / "last"), weights(tmp_path / "ddp" / "last")
    trained(one)
    assert_close(one, two)
    v1, v2 = (json.loads((tmp_path / d / "log.jsonl").read_text().splitlines()[-1]) for d in ("one", "ddp"))
    for k in ("acc", "nll", "ece"):
        assert abs(v1[k] - v2[k]) < 1e-3, (k, v1, v2)
    assert out2.count('"val"') == 1                      # rank 0 only
    assert "must divide" in subprocess.run(
        TORCHRUN + train_args(base, tmp_path / "bad", "--grad-accum", "3"), cwd=ROOT,
        capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"}).stderr


SGD_WRAPPER = """import sys
sys.path.insert(0, {root!r})
import torch
import train_prism as tp
# Linear optimizer, no clipping: averaging the weights after per-rank steps then equals one
# step on the averaged gradient (with AdamW it does not: avg of Adam steps != Adam of avg).
tp.make_optimizer = lambda model, args: torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad], lr=args.lr)
tp.MAX_GRAD_NORM = float("inf")
tp.main(sys.argv[1:])
"""


def test_local_sgd_k1_equals_ddp_with_sgd(base, tmp_path):
    """local_sgd --sync-every 1 == ddp == 1 process with SGD (linear, no clip). With AdamW the
    same local_sgd K=1 run is far off (avg of per-rank Adam steps != Adam on the avg grad),
    which also shows the tolerance can tell the two apart."""
    wrap = tmp_path / "sgd_main.py"
    wrap.write_text(SGD_WRAPPER.format(root=str(ROOT)))
    # lr 1e-3: this tiny model is sensitive to summation order (one process with only the
    # micro-batch order changed drifts ~2.5e-6 in 3 steps), so equal means within ~5e-5
    # while the weights move ~1e-2; per-rank AdamW + averaging is off by far more.
    # --length-grouping off: the Mixture order this test's lr / tolerances were tuned on (with
    # the grouped order SGD at lr 1e-3 moves some lora_B by < 1e-4 in 3 steps; grouping under
    # DDP is covered by test_ddp_two_ranks_equal_one_process).
    sgd = ["--lr", "1e-3", "--head-lr", "1e-3", "--length-grouping", "off"]
    lsgd = ["--sync-mode", "local_sgd", "--sync-every", "1"]
    run([sys.executable, str(wrap)] + train_args(base, tmp_path / "one")[1:] + sgd)
    out = run(TORCHRUN + [str(wrap)] + train_args(base, tmp_path / "lsgd")[1:] + sgd + lsgd)
    assert '"sync_mode": "local_sgd", "sync_every": 1' in out
    run(TORCHRUN + [str(wrap)] + train_args(base, tmp_path / "ddp")[1:] + sgd)
    one = weights(tmp_path / "one" / "last")
    trained(one)

    def gap(d):
        w = weights(tmp_path / d / "last")
        return max((one[k] - w[k]).abs().max().item() for k in one)
    assert gap("lsgd") < 5e-5 and gap("ddp") < 5e-5, (gap("lsgd"), gap("ddp"))
    run(TORCHRUN + train_args(base, tmp_path / "adam", *lsgd))          # AdamW, lr 1e-3
    run([sys.executable] + train_args(base, tmp_path / "adam1"))
    adam1, adam = weights(tmp_path / "adam1" / "last"), weights(tmp_path / "adam" / "last")
    adam_gap = max((adam1[k] - adam[k]).abs().max().item() for k in adam1)
    assert adam_gap > 5e-4, adam_gap


def test_local_sgd_adamw_runs_and_averages(base, tmp_path):
    """local_sgd with AdamW (the real optimizer) and K=2 over 3 steps: one periodic average
    plus the one before the final eval/save; it trains. Not equal to ddp (see dist.py)."""
    out = run(TORCHRUN + train_args(base, tmp_path / "l", "--sync-mode", "local_sgd", "--sync-every", "2"))
    assert '"val"' in out
    trained(weights(tmp_path / "l" / "last"))


@pytest.mark.parametrize("algo", ["exact", "grpo"])
def test_ddp_rl_equals_one_process(base, tmp_path, algo):
    """RL from an SFT checkpoint: exact PG (soft ECE per micro-batch) and GRPO (seeded
    sampling per micro-batch, ppo-epoch backwards under no_sync, KL reference with the
    adapter off on the raw module): 2 ranks == 1 process."""
    run([sys.executable] + train_args(base, tmp_path / "sft"))
    init = ["--algo", algo, "--init", str(tmp_path / "sft" / "best"), "--ppo-epochs", "2", "--group", "4"]
    run([sys.executable] + train_args(base, tmp_path / "one", *init))
    run(TORCHRUN + train_args(base, tmp_path / "ddp", *init))
    assert_close(weights(tmp_path / "one" / "last"), weights(tmp_path / "ddp" / "last"))


# ---------------------------------------------------------------- bench_trials ddp=on

def test_bench_ddp_trials_split_to_torchrun(tmp_path, tok):
    """A plain bench_trials launch runs ddp=off trials itself and hands ddp=on trials to one
    torchrun launch (OJ_DDP_NPROC=2 gloo ranks), merged into --out. Same global batch ->
    same records (data_ids_sha) and the same guard loss for ddp; local_sgd K=1 runs."""
    pytest.importorskip("optuna")
    import test_optuna_loop as tol
    base = tiny_gemma(tmp_path / "base", tok, kv_shared=False)
    tj, res = tmp_path / "t.json", tmp_path / "r.json"
    g = ["--db", str(tmp_path / "s.db")] + tol.space(tmp_path)
    tol.ol.main(g + ["propose", "--n", "0", "--out", str(tj), "--enqueue-json",
                     '[{"dtype": "fp32", "packing": "on", "pack_len": 1024},'
                     ' {"dtype": "fp32", "packing": "on", "pack_len": 1024, "ddp": "on"},'
                     ' {"dtype": "fp32", "packing": "on", "pack_len": 1024, "ddp": "on",'
                     '  "sync_mode": "local_sgd", "sync_every": 1}]'])
    t = json.loads(tj.read_text())[-3:]          # drop the round baseline (fp16: no CPU backward)
    tj.write_text(json.dumps(t))
    assert t[1]["params"]["ddp"] == "on" and "ddp" in t[1]["reload_keys"]
    # The frozen packed layout (tools/freeze_packed.py) next to the stream, as in r7: the
    # ranks look rows up by stream position (pos + k * mb).
    sys.path.insert(0, str(ROOT / "tools"))
    import freeze_packed as fp
    stream = Path(g[g.index("--stream") + 1])
    fp.main(["--stream", str(stream), "--out", str(stream.with_name("packed_layout.json")), "--train", FIX,
             "--tokenizer", str(base), "--micro-batch", "2", "--max-state-tokens", "48"])
    tol.bench(tj, res, base, g)
    doc = json.loads(res.read_text())
    assert doc["meta"].get("packed_layout_sha")
    r = {x["trial_number"]: x for x in doc["results"]}
    single, ddp, lsgd = (r[x["trial_number"]] for x in t)
    for x in (single, ddp, lsgd):
        assert x["status"] == "ok", x
    assert "world" not in single and ddp["world"] == 2 and ddp["accum_per_rank"] == 1
    assert ddp["global_batch"] == single["global_batch"] == 4 and lsgd["sync_mode"] == "local_sgd"
    for x in (ddp, lsgd):
        assert x["data_ids_sha"] == single["data_ids_sha"] and x["warmup_ids_sha"] == single["warmup_ids_sha"]
        assert x["records_used"] == single["records_used"]
        # same packed rows: global sums over the ranks equal the single process exactly
        assert x["pad_frac"] == single["pad_frac"] and x["rows_per_mb"] == single["rows_per_mb"]
    assert abs(ddp["loss_guard"] - single["loss_guard"]) < 1e-4, (ddp, single)
    assert abs(ddp["loss_end"] - single["loss_end"]) < 5e-4
    assert lsgd["syncs"] >= 3                       # every step (K=1) + before each probe
