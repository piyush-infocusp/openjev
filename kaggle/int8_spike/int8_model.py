"""int8 spike, Part 2: int8 GEMMs for the frozen MLP base linears of the real model.

The frozen gate/up/down base linears (105 in E2B; under PEFT, `...mlp.gate_proj.base_layer`)
change class in place to Int8ToggleLinear. That is an nn.Linear subclass that keeps its fp16
weight and adds two int8 copies as non-persistent buffers: rows (per out-channel scale) for
Y = X W^T and columns (per in-channel scale) for dX = dY W. Activations and gradients get
dynamic per-row scales. `Int8ToggleLinear.mode` switches every layer at once, so fp16 and
int8 run on the same weights after one model load:
  fp16      F.linear (the current path)
  eager     torch._int_mm, quantize/dequant in eager
  compiled  the same function under torch.compile(dynamic=True) (Part 1 i8_row_compiled)
  fused     + inductor force_fuse_int_mm_with_mul (Part 1 i8_row_fused, Triton int8 GEMM)
  measure   returns fp16, records the int8 vs fp16 error of Y on the real activations
q/k/v/o stay fp16: Part 1 measured them at <= 1x. LoRA is not touched.

  python int8_model.py --cpu-check          # tiny Gemma 4 + LoRA on CPU (correctness)
  python int8_model.py --real --data DIR    # Kaggle T4: E2B, <=512 bench bucket, A/B/A/B
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("gate_proj", "up_proj", "down_proj")


def i8mm(x, wq, sw):
    """x (M,K) float, wq (K,N) int8 mat2, sw (1,N) fp32 -> (M,N) in x.dtype."""
    s = x.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / 127
    q = (x / s).round().clamp(-127, 127).to(torch.int8)
    return (torch._int_mm(q, wq) * s * sw).to(x.dtype)


def fakeq(x, wq, sw):
    """Float reference of i8mm (same rounding, fp32 GEMM): equal to i8mm while K * 127^2 < 2^24."""
    s = x.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / 127
    q = (x / s).round().clamp(-127, 127)
    return ((q.float() @ wq.float()) * s * sw).to(x.dtype)


FNS: dict = {"eager": (i8mm, i8mm), "fakeq": (fakeq, fakeq)}
ERRORS: dict = {}


def setup_compiled():
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 512
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = 512
    FNS["compiled"] = (torch.compile(i8mm, dynamic=True), torch.compile(i8mm, dynamic=True))
    o = {"force_fuse_int_mm_with_mul": True}
    FNS["fused"] = (torch.compile(i8mm, dynamic=True, options=o), torch.compile(i8mm, dynamic=True, options=o))


def graph_count():
    try:
        from torch._dynamo.utils import counters
        return int(counters["stats"]["unique_graphs"])
    except Exception:
        return None


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mod, mode):
        ctx.mod, ctx.mode = mod, mode
        x2 = x.reshape(-1, x.shape[-1]).to(mod.weight.dtype)
        y = FNS[mode][0](x2, mod.i8_f, mod.i8_sf)
        return y.view(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx, dy):
        mod = ctx.mod
        d2 = dy.reshape(-1, dy.shape[-1]).to(mod.weight.dtype)
        dx = FNS[ctx.mode][1](d2, mod.i8_b, mod.i8_sb)
        return dx.view(*dy.shape[:-1], -1), None, None


class Int8ToggleLinear(nn.Linear):
    mode = "fp16"
    min_m = 17                      # torch._int_mm needs M > 16 on CUDA
    stats: dict = {}

    def forward(self, x):
        mode = Int8ToggleLinear.mode
        M = x.numel() // self.in_features
        if mode == "fp16" or M < Int8ToggleLinear.min_m:
            return F.linear(x, self.weight, self.bias)
        if mode == "measure":
            y = F.linear(x, self.weight, self.bias)
            with torch.no_grad():
                x2 = x.reshape(-1, x.shape[-1]).to(self.weight.dtype)
                yi = i8mm(x2, self.i8_f, self.i8_sf).float()
                yr = y.reshape(-1, y.shape[-1]).float()
                ch = x2.abs().amax(0).float()
                st = Int8ToggleLinear.stats.setdefault(self._oj_kind, [])
                st.append({"rel_fro": ((yi - yr).norm() / yr.norm().clamp_min(1e-12)).item(),
                           "max_rel": ((yi - yr).abs().max() / yr.abs().max().clamp_min(1e-12)).item(),
                           "chan_absmax_ratio": (ch.max() / ch.median().clamp_min(1e-12)).item()})
            return y
        y = _Fn.apply(x, self, mode)
        return y if self.bias is None else y + self.bias


def q_rows(w):
    s = w.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / 127
    return (w / s).round().clamp(-127, 127).to(torch.int8), s


@torch.no_grad()
def convert(model, targets=TARGETS) -> list[str]:
    """In place: the frozen base linears of `targets` -> Int8ToggleLinear (+ int8 buffers)."""
    done = []
    for name, m in list(model.named_modules()):
        if type(m) is not nn.Linear or ".layers." not in f".{name}":
            continue
        parts = name.split(".")
        leaf = parts[-2] if parts[-1] == "base_layer" else parts[-1]
        if leaf not in targets or any(t in name for t in ("lora_", "vision", "audio")):
            continue
        w = m.weight.detach()
        wr, sr = q_rows(w)                              # (N,K) int8, (N,1)
        wc, sc = q_rows(w.t().contiguous())             # (K,N) int8, (K,1)
        m.__class__ = Int8ToggleLinear
        m.register_buffer("i8_f", wr.t(), persistent=False)          # (K,N) column-major
        m.register_buffer("i8_b", wc.t(), persistent=False)          # (N,K) column-major
        m.register_buffer("i8_sf", sr.reshape(1, -1), persistent=False)
        m.register_buffer("i8_sb", sc.reshape(1, -1), persistent=False)
        m._oj_kind = f"{leaf}_{m.out_features}x{m.in_features}"
        done.append(name)
    return done


# ------------------------------------------------- CPU correctness on the tiny Gemma 4

def cpu_check():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, root)
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import Linear as LoraLinear
    from transformers import AutoTokenizer, Gemma4TextConfig, Gemma4TextModel
    import train_prism as tp
    from openjev import Schema, add_special_tokens
    from openjev.data import Record
    from openjev.prism import PrismOpenJev, prism_collate
    from openjev.schema import SPECIAL_TOKENS

    tok = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=len(tok) + 64, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=16, sliding_window=64, vocab_size_per_layer_input=100,
                           hidden_size_per_layer_input=8, num_kv_shared_layers=0,
                           layer_types=["sliding_attention", "sliding_attention", "full_attention"],
                           initializer_range=0.1)
    cfg._attn_implementation = "sdpa"
    model = PrismOpenJev(Gemma4TextModel(cfg))
    add_special_tokens(tok, model)
    model.set_template_ids(tok.convert_tokens_to_ids(SPECIAL_TOKENS))
    model.backbone = get_peft_model(model.backbone, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                                               target_modules=list(tp.LORA_TARGETS)))
    names = convert(model)
    assert len(names) == 3 * 3, names
    for n in names:
        m = model.get_submodule(n)
        assert isinstance(m, nn.Linear) and type(m) is Int8ToggleLinear
        assert isinstance(model.get_submodule(n.rsplit(".", 1)[0]), LoraLinear)
        assert not m.weight.requires_grad
    assert not getattr(model.backbone, "is_loaded_in_8bit", False)
    assert not any(k.endswith(("i8_f", "i8_b", "i8_sf", "i8_sb")) for k in model.state_dict())
    tp.set_grad_ckpt(model, True)
    rng = random.Random(0)
    words = ["red", "blue", "green", "pink", "black", "white", "gold", "gray"]
    recs = []
    for i in range(8):
        opts = rng.sample(words, 4)
        y = rng.choice(opts)
        recs.append(Record(f"r{i}", "toy", "train", f"the answer is {y} " * 3,
                           Schema().choice("c", {o: "" for o in opts}, "pick"), [y]))
    b = prism_collate(tok, recs)
    out = {"converted": len(names), "tokens_M": int(b.input_ids.numel())}
    res = {}
    for mode in ("fp16", "fakeq", "eager", "compiled"):
        if mode == "compiled" and "compiled" not in FNS:
            setup_compiled()
        Int8ToggleLinear.mode = mode
        model.zero_grad(set_to_none=True)
        model.train()
        loss = tp.sft_loss(model.flag_logits(b), b)
        loss.backward()
        lora = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                if "lora_" in n and p.grad is not None}
        assert all(model.get_submodule(n).weight.grad is None for n in names)
        assert lora and all(torch.isfinite(g).all() for g in lora.values())
        res[mode] = (loss.item(), lora)
        out[f"loss_{mode}"] = loss.item()

    def rel(a, b):
        num = math.sqrt(sum(float(((a[k] - b[k]) ** 2).sum()) for k in b))
        return num / math.sqrt(sum(float((b[k] ** 2).sum()) for k in b))
    # implementation check: int8 kernels == the float reference of the same quantization
    out["lora_grad_rel_eager_vs_fakeq"] = rel(res["eager"][1], res["fakeq"][1])
    out["lora_grad_rel_compiled_vs_eager"] = rel(res["compiled"][1], res["eager"][1])
    # numerical effect of int8 vs the float path (the tiny random model is very sensitive:
    # 1% Gaussian noise on each MLP Y and dX moves its LoRA grads by 50-75% too)
    out["lora_grad_rel_eager_vs_fp"] = rel(res["eager"][1], res["fp16"][1])
    out["loss_rel_eager_vs_fp"] = abs(res["eager"][0] - res["fp16"][0]) / abs(res["fp16"][0])
    nz = sum(int(v.abs().sum() > 0) for v in res["fp16"][1].values())
    out["lora_grads_nonzero"] = f"{nz}/{len(res['fp16'][1])} (lora_A grads are 0 while lora_B = 0 at init)"
    print("CPU_CHECK", json.dumps(out, indent=1))
    assert out["lora_grad_rel_eager_vs_fakeq"] < 1e-4, out
    assert out["lora_grad_rel_compiled_vs_eager"] < 1e-4, out
    assert abs(out["loss_eager"] - out["loss_fakeq"]) < 1e-5 and out["loss_rel_eager_vs_fp"] < 0.05, out
    return out


# ------------------------------------------------- real E2B on the T4

def real(data_dir, out_path, timed_steps, reps, modes):
    import bench_trials as bt
    import train_prism as tp
    from openjev.prism import prism_collate
    res = {"t0": time.time(), "modes": modes, "timed_steps": timed_steps, "reps": reps}

    def save():
        json.dump(res, open(out_path, "w"), indent=1)

    own, targs = bt.parse(["--trials", "none.json", "--bench-lengths", "bench_lengths.json",
                           "--model", "google/gemma-4-E2B-it", "--train", f"{data_dir}/*.train.jsonl",
                           "--bs", "4", "--max-state-tokens", "1024", "--grad-ckpt"])
    device = tp.pick_device(targs)
    assert device == "cuda" and "T4" in torch.cuda.get_device_name()
    data, probe, dmeta = bt.load_data(own, targs)
    res["bench"] = {k: v for k, v in dmeta.items() if k != "stems"}
    random.seed(targs.seed)
    torch.manual_seed(targs.seed)
    t = time.perf_counter()
    model, tok, dtype = tp.build(bt.build_args(targs, {"ple_cpu": 1, "dtype": "fp16", "quant": "none"}), device)
    res["load_s"] = time.perf_counter() - t
    model.train()
    base_threads = torch.get_num_threads()
    params = {"grad_ckpt": "on", "torch_threads": 4}
    _mst, backend = bt.apply_params(model, params, targs, base_threads)
    mem0 = torch.cuda.memory_allocated() / 2**30
    names = convert(model)
    res["converted"] = len(names)
    res["converted_example"] = names[:2]
    res["mem_gb_before_after_convert"] = [mem0, torch.cuda.memory_allocated() / 2**30]
    res["sdpa_backend"] = backend
    print("CONVERTED", len(names), res["mem_gb_before_after_convert"], flush=True)
    setup_compiled()
    snap = bt.snapshot(model)
    mst = data.max_state_tokens
    gb, mb = 32, 8
    recs = data.buckets[512]
    step_mbs = [recs[i:i + mb] for i in range(0, gb, mb)]

    # 1) int8 vs fp16 error of Y on the real activations (probe forward, fp16 output)
    Int8ToggleLinear.mode, Int8ToggleLinear.stats = "measure", {}
    bt.probe_loss(model, tok, probe, targs, device)
    agg = {}
    for k, v in Int8ToggleLinear.stats.items():
        agg[k] = {f: {"median": sorted(x[f] for x in v)[len(v) // 2], "max": max(x[f] for x in v)}
                  for f in ("rel_fro", "max_rel", "chan_absmax_ratio")} | {"n": len(v)}
    res["real_act_error"] = agg
    print("REAL_ACT_ERROR", json.dumps(agg), flush=True)
    save()

    # 2) probe loss on the same (start) weights, per mode
    res["loss_start"] = {}
    for mode in ["fp16"] + modes:
        Int8ToggleLinear.mode = mode
        t = time.perf_counter()
        try:
            res["loss_start"][mode] = bt.probe_loss(model, tok, probe, targs, device)
        except Exception as e:
            res["loss_start"][mode] = f"{type(e).__name__}: {str(e)[:500]}"
            ERRORS[mode] = traceback.format_exc()[-3000:]
            print(ERRORS[mode], flush=True)
        res.setdefault("probe_s", {})[mode] = time.perf_counter() - t
    print("LOSS_START", res["loss_start"], flush=True)
    save()

    # 3) A/B/A/B on the <=512 bucket: one warm-up step (compiles), then timed steps
    def arm(mode, rep):
        r = {"mode": mode, "rep": rep}
        Int8ToggleLinear.mode = mode
        bt.reset(model, data, snap, None, targs.seed)
        bt.free(device)
        opt = tp.make_optimizer(model, targs)
        scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16)
        counts = {"real": 0, "pad": 0}

        def batches():
            for m in step_mbs:
                b = prism_collate(tok, m, mst, pad_to_multiple_of=8)
                counts["real"] += int(b.attention_mask.sum())
                counts["pad"] += b.attention_mask.numel()
                yield b
        losses, skipped = [], 0
        t = time.perf_counter()
        loss, sk = bt.accum_step(model, opt, scaler, batches(), len(step_mbs), device, backend)
        torch.cuda.synchronize()
        r["warm_s"] = time.perf_counter() - t
        losses.append(float(loss))
        skipped += sk
        g0 = graph_count()
        torch.cuda.reset_peak_memory_stats()
        counts.update(real=0, pad=0)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(timed_steps):
            loss, sk = bt.accum_step(model, opt, scaler, batches(), len(step_mbs), device, backend)
            losses.append(loss)
            skipped += sk
        torch.cuda.synchronize()
        sec = time.perf_counter() - t
        r.update(seconds=sec, samples=timed_steps * gb, samples_per_sec=timed_steps * gb / sec,
                 tokens_per_sec=counts["real"] / sec, padded_tokens=counts["pad"],
                 pad_frac=1 - counts["real"] / max(counts["pad"], 1),
                 peak_gb=torch.cuda.max_memory_allocated() / 2**30, skipped_steps=skipped,
                 graphs_in_timed_loop=(graph_count() - g0) if g0 is not None else None,
                 train_losses=[float(x) for x in losses])
        r["loss_after"] = bt.probe_loss(model, tok, probe, targs, device)
        return r

    res["arms"] = []
    for rep in range(reps):
        for mode in ["fp16"] + modes:
            if isinstance(res["loss_start"].get(mode), str):
                continue                                   # mode failed at the probe
            try:
                r = arm(mode, rep)
            except Exception as e:
                r = {"mode": mode, "rep": rep, "error": f"{type(e).__name__}: {str(e)[:500]}"}
                print(traceback.format_exc()[-3000:], flush=True)
                model.zero_grad(set_to_none=True)
                bt.free(device)
            res["arms"].append(r)
            print("ARM", json.dumps({k: v for k, v in r.items() if k != "train_losses"}), flush=True)
            save()
    Int8ToggleLinear.mode = "fp16"
    res["errors_tb"] = ERRORS
    save()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", default="int8_model.json")
    ap.add_argument("--timed-steps", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--modes", default="compiled,fused")
    a = ap.parse_args()
    if a.cpu_check:
        cpu_check()
    if a.real:
        real(a.data, a.out, a.timed_steps, a.reps, a.modes.split(","))


if __name__ == "__main__":
    main()
