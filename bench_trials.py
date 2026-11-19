"""Run a batch of Optuna speed trials (from tools/optuna_loop.py propose) in one process.

Loading Gemma 4 E2B on Kaggle takes ~5 min, so the model is loaded once per group of
trials that share the same `requires_reload` params (for example ple_cpu). Each trial:

  1. restores the start weights (LoRA, flag head, template delta), the data stream
     position and the seed, and builds a fresh optimizer, so all trials start equal;
  2. applies its params (bs, grad_ckpt, max_state_tokens, torch_threads, sdpa_backend,
     attn_backend; the attention backend is swapped at runtime, see openjev/attn_backends.py);
  3. measures the loss on a fixed probe set (loss_start);
  4. runs --warmup-steps SFT steps (one optimizer step each) and measures the probe
     loss again (loss_guard). The quality guard uses this one: every trial has taken
     the same number of optimizer steps here, whatever its speed;
  5. runs a timed loop of real SFT steps on the real mixture for --trial-seconds
     (at least one step), then measures loss_end (informational) and peak GPU memory;
  6. profile=1 only: after all of the above (so samples/s, the guard and the data stream
     equal a non-profile twin), runs torch.profiler (CPU + CUDA) over --profile-steps more
     steps and stores a compact table (top ops by self CUDA / self CPU time, and the share
     of attention, matmul, mask building, PLE lookup, copies and collate) as "profile".

The fp16 GradScaler scale is carried from trial to trial inside a group, so a fresh
2^16 scale does not skip the first steps of every trial; skipped steps are counted.

OOM, a missing SDPA kernel and other errors are caught per trial. Results go to
--out (rewritten after every trial) and one JSON line per trial is printed.

Every flag train_prism.py knows (--model, --train, --lr, --dtype, --cpu, ...) is passed
through; --bs, --max-state-tokens and --grad-ckpt are the defaults for trials that do
not set them, and --max-state-tokens is also the probe truncation.

  python bench_trials.py --trials trials.json --trial-seconds 50 \
      --model google/gemma-4-E2B-it --train 'data/*.train.jsonl'
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import glob
import json
import math
import random
import time
import traceback
from pathlib import Path

import torch

import train_prism as tp
from openjev import attn_backends as ab
from openjev.mix import Mixture
from openjev.prism import prism_collate

# Params this script knows how to apply. A trial with any other param fails, so a
# dimension marked `implemented: true` without code here never gives fake numbers.
TRIAL_PARAMS = {"bs", "grad_ckpt", "max_state_tokens", "torch_threads", "sdpa_backend",
                "attn_backend", "profile"}
RELOAD_PARAMS = {"ple_cpu"}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", default="trials.json")
    ap.add_argument("--trial-seconds", type=float, default=50.0)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--probe-n", type=int, default=8, help="records in the fixed probe set")
    ap.add_argument("--profile-steps", type=int, default=2, help="profiled steps (profile=1 trials)")
    ap.add_argument("--attn-chunk", type=int, default=ab.DEFAULT_CHUNK,
                    help="query rows per chunk for attn_backend=chunked_math")
    ap.add_argument("--out", default="/kaggle/working/out/optuna_results.json")
    own, rest = ap.parse_known_args(argv)
    return own, tp.parse(rest)


# ---------------------------------------------------------------- params

def is_on(v) -> bool:
    return str(v).lower() in ("on", "true", "1", "yes")


def build_args(targs, reload_params: dict):
    """train_prism args for one reload group."""
    a = copy.copy(targs)
    for k, v in reload_params.items():
        if k == "ple_cpu":
            a.ple_cpu = int(v)
        else:
            raise ValueError(f"reload param {k!r} has no handler in bench_trials.py")
    return a


def sdpa_ctx(name):
    if name in (None, "default"):
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backend = {"efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH,
               "flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}[name]
    return sdpa_kernel(backend)


# Errors that mean "flex/Triton cannot build a kernel here" (e.g. head_dim 512 on sm75).
FLEX_UNSUPPORTED = ("out of resource", "shared memory", "triton", "inductor", "backendcompilerfailed",
                    "notimplementederror", "flex")


def classify(e: BaseException, backend: str | None = None) -> str:
    msg = str(e)
    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg.lower():
        return "oom"
    if isinstance(e, ab.Unsupported) or "No available kernel" in msg or "No viable backend" in msg \
            or "does not support backward on CPU" in msg:
        return "unsupported"
    if backend == "flex" and any(k in f"{type(e).__name__}: {msg}".lower() for k in FLEX_UNSUPPORTED):
        return "unsupported"
    return "error"


# ---------------------------------------------------------------- helpers

def snapshot(model):
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def restore(model, snap):
    for n, p in model.named_parameters():
        if n in snap:
            p.copy_(snap[n])
            p.grad = None


@torch.no_grad()
def probe_loss(model, tok, probe, targs, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(probe), 4):
        chunk = probe[i:i + 4]
        b = prism_collate(tok, chunk, targs.max_state_tokens).to(device)
        with tp.autocast(device, model):
            loss = tp.sft_loss(model.flag_logits(b), b)
        total += float(loss) * len(chunk)
        n += len(chunk)
    model.train()
    return total / max(n, 1)


def clean(x):
    if isinstance(x, float):
        return round(x, 5) if math.isfinite(x) else None
    return x


# ---------------------------------------------------------------- profiler

PROFILE_REGIONS = {"attention_fwd": "oj::attention", "masks": "oj::masks", "ple_lookup": "oj::ple",
                   "collate": "oj::collate"}
MATMUL_OPS = {"aten::mm", "aten::bmm", "aten::addmm", "aten::baddbmm", "aten::addbmm", "aten::_scaled_mm"}
COPY_OPS = {"aten::copy_", "aten::_to_copy", "aten::to", "aten::_copy_from"}


def op_category(name: str):
    """Op-name bucket (self time, forward + backward)."""
    n = name.lower()
    if name in MATMUL_OPS:
        return "matmul_ops"
    if "attention" in n or "softmax" in n or "flex" in n:
        return "attention_ops"
    if name in COPY_OPS or "memcpy" in n:
        return "copy_ops"
    if "embedding" in n:
        return "embedding_ops"
    return None


def _ev(e, *names):
    for n in names:
        v = getattr(e, n, None)
        if v is not None:
            return float(v)
    return 0.0


def summarize_profile(events, wall_s: float, steps: int, top: int = 25) -> dict:
    """Compact table from prof.key_averages() (times in ms).

    shares: oj::* rows are record_function ranges (inclusive; attention_fwd, masks and
    ple_lookup cover the forward only, backward is not attributable by range): cpu share
    is of the wall time, cuda share of all kernel time. *_ops rows are op-name buckets by
    self time over forward + backward: cpu share of all self CPU time, cuda share of all
    kernel time. Ranges and buckets overlap (math attention's bmm is in attention_fwd and
    in matmul_ops)."""
    from torch.autograd import DeviceType
    rows = []
    for e in events:
        if getattr(e, "device_type", DeviceType.CPU) != DeviceType.CPU:
            continue                       # kernel rows: their time is in the launching op
        rows.append((e.key, int(e.count), _ev(e, "self_cpu_time_total"),
                     _ev(e, "self_device_time_total", "self_cuda_time_total"),
                     _ev(e, "cpu_time_total"), _ev(e, "device_time_total", "cuda_time_total")))
    ann = lambda k: k.startswith("oj::") or k.startswith("ProfilerStep")   # noqa: E731
    cpu_total = sum(r[2] for r in rows) or 1.0
    cuda_total = sum(r[3] for r in rows)
    wall_us = wall_s * 1e6 or 1.0
    shares = {}
    for name, label in PROFILE_REGIONS.items():
        hit = [r for r in rows if r[0] == label]
        if hit:
            c, d = sum(r[4] for r in hit), sum(r[5] for r in hit)
            shares[name] = {"cpu_ms": round(c / 1e3, 2), "cpu": round(c / wall_us, 4),
                            "cuda_ms": round(d / 1e3, 2), "cuda": round(d / cuda_total, 4) if cuda_total else None}
    cats = {}
    for r in rows:
        k = None if ann(r[0]) else op_category(r[0])
        if k:
            c = cats.setdefault(k, [0.0, 0.0])
            c[0] += r[2]
            c[1] += r[3]
    for k, (c, d) in sorted(cats.items()):
        shares[k] = {"cpu_ms": round(c / 1e3, 2), "cpu": round(c / cpu_total, 4),
                     "cuda_ms": round(d / 1e3, 2), "cuda": round(d / cuda_total, 4) if cuda_total else None}
    ops = [r for r in rows if not ann(r[0])]

    def table(i, total):
        best = sorted((r for r in ops if r[i] > 0), key=lambda r: -r[i])[:top]
        return [[r[0][:70], round(r[i] / 1e3, 3), round(r[i] / total, 4), r[1]] for r in best]
    return {"steps": steps, "wall_ms": round(wall_s * 1e3, 1), "cpu_self_total_ms": round(cpu_total / 1e3, 1),
            "cuda_total_ms": round(cuda_total / 1e3, 1), "shares": shares,
            "top_cuda": table(3, cuda_total) if cuda_total else [], "top_cpu": table(2, cpu_total),
            "columns": ["op", "self_ms", "share", "calls"]}


def run_profile(step, device: str, steps: int) -> dict:
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    with profile(activities=acts, record_shapes=False) as prof:
        t0 = time.perf_counter()
        for _ in range(steps):
            step()
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    return summarize_profile(prof.key_averages(), wall, steps)


def free(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- one trial

def run_trial(model, tok, mixture, mix_state, snap, probe, targs, params, device, dtype, own,
              base_threads, scale):
    unknown = set(params) - TRIAL_PARAMS - RELOAD_PARAMS
    if unknown:
        raise ValueError(f"params {sorted(unknown)} have no handler in bench_trials.py")
    attn = params.get("attn_backend", "sdpa_default")
    try:
        ab.set_attn_backend(model, attn, chunk=own.attn_chunk)
        return _run_trial(model, tok, mixture, mix_state, snap, probe, targs, params, device, dtype,
                          own, base_threads, scale, attn)
    finally:
        ab.reset_attn_backend(model)   # a failed flex trial must not change the next trial


def _run_trial(model, tok, mixture, mix_state, snap, probe, targs, params, device, dtype, own,
               base_threads, scale, attn):
    restore(model, snap)
    mixture.set_state(mix_state)
    random.seed(targs.seed)
    torch.manual_seed(targs.seed)
    bs = int(params.get("bs", targs.bs))
    mst = int(params.get("max_state_tokens", targs.max_state_tokens))
    tp.set_grad_ckpt(model, is_on(params["grad_ckpt"]) if "grad_ckpt" in params else targs.grad_ckpt)
    torch.set_num_threads(int(params.get("torch_threads", base_threads)))
    backend = params.get("sdpa_backend", "default")
    profile = is_on(params.get("profile", 0))
    # flex compiles one graph per shape: pad T to the BlockMask block so shapes repeat.
    pad_mult = ab.FLEX_BLOCK if attn == "flex" else 8

    free(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Constant lr, no warmup: a scheduler would barely move the weights in a few steps.
    opt = tp.make_optimizer(model, targs)
    scaler = torch.amp.GradScaler("cuda", init_scale=scale["value"],
                                  enabled=(device == "cuda" and dtype == torch.float16))
    scale_start = scale["value"]
    loss_start = probe_loss(model, tok, probe, targs, device)

    losses, tok_real, tok_pad, skipped = [], 0, 0, 0

    def step():
        nonlocal skipped
        with torch.profiler.record_function("oj::collate"):
            b = prism_collate(tok, mixture.take(bs), mst, pad_to_multiple_of=pad_mult)
        real, pad = int(b.attention_mask.sum()), b.attention_mask.numel()   # CPU, no sync
        b = b.to(device)
        with sdpa_ctx(backend):
            with tp.autocast(device, model):
                loss = tp.sft_loss(model.flag_logits(b), b)
            scaler.scale(loss).backward()
        before = scaler.get_scale()        # scaler.step syncs anyway (found_inf)
        tp.optimizer_step(opt, scaler)
        skipped += int(scaler.is_enabled() and scaler.get_scale() < before)
        losses.append(loss.detach())
        return real, pad

    t_w = time.perf_counter()
    for _ in range(own.warmup_steps):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t_w      # includes flex compile time
    losses.clear()
    loss_guard = probe_loss(model, tok, probe, targs, device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 0
    while True:
        real, pad = step()
        tok_real, tok_pad, steps = tok_real + real, tok_pad + pad, steps + 1
        if time.perf_counter() - t0 >= own.trial_seconds:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    train = torch.stack(losses).float().cpu().tolist()
    loss_end = probe_loss(model, tok, probe, targs, device)
    if scaler.is_enabled():
        scale["value"] = scaler.get_scale()
    nan = not all(math.isfinite(x) for x in train + [loss_start, loss_guard, loss_end])
    k = max(1, min(3, len(train) // 2))
    extra = {"attn_backend": attn, "warmup_s": warmup_s}
    if profile:
        # After loss_end and timing: the profiled steps change nothing that is measured.
        ab.set_attn_backend(model, attn, chunk=own.attn_chunk, label=True)
        extra["profile"] = run_profile(step, device, own.profile_steps)
        extra["is_profile"] = True
    return {**extra, "status": "nan" if nan else "ok", "steps": steps, "samples": steps * bs,
            "seconds": sec, "samples_per_sec": steps * bs / sec, "s_per_sample": sec / (steps * bs),
            "s_per_step": sec / steps, "tokens_per_sec": tok_real / sec,
            "padded_tokens_per_sec": tok_pad / sec, "pad_frac": 1 - tok_real / max(tok_pad, 1),
            "peak_mem_gb": peak, "loss_start": loss_start, "loss_guard": loss_guard,
            "guard_steps": own.warmup_steps, "loss_end": loss_end, "scale_start": scale_start,
            "scale_end": scale["value"], "skipped_steps": skipped,
            "train_loss_first": sum(train[:k]) / k, "train_loss_last": sum(train[-k:]) / k}


# ---------------------------------------------------------------- main

def main(argv=None):
    own, targs = parse(argv)
    trials = json.loads(Path(own.trials).read_text())
    device = tp.pick_device(targs)
    n_gpu = torch.cuda.device_count() if device == "cuda" else 0
    meta = {"device": device, "n_gpu": n_gpu,
            "gpu_name": torch.cuda.get_device_name(0) if n_gpu else None,
            "torch": torch.__version__, "model": targs.model, "trial_seconds": own.trial_seconds,
            "warmup_steps": own.warmup_steps, "probe_n": own.probe_n, "trials_file": own.trials}
    out = Path(own.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []

    def emit(r):
        r = {k: clean(v) for k, v in r.items()}
        results.append(r)
        print(json.dumps({"optuna_trial": r}), flush=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
        tmp.replace(out)

    paths = sorted(glob.glob(targs.train))
    if not paths:
        raise SystemExit(f"no train files match {targs.train}")
    mixture = Mixture(paths, alpha=targs.alpha, seed=targs.seed)
    probe = mixture.take(own.probe_n)            # fixed probe set, same for every trial
    mix_state = mixture.get_state()
    base_threads = torch.get_num_threads()

    groups: dict[tuple, list] = {}
    for t in trials:
        keys = t.get("reload_keys") or sorted(RELOAD_PARAMS & set(t["params"]))
        key = tuple((k, t["params"][k]) for k in sorted(keys) if k in t["params"])
        groups.setdefault(key, []).append(t)
    print(json.dumps({"bench": meta, "groups": [dict(k) for k in groups],
                      "trials": len(trials)}), flush=True)

    model = None
    for key, group in groups.items():
        model = None
        free(device)
        t_load = time.perf_counter()
        try:
            # Same seed for every group: the random flag head, LoRA A and new embedding
            # rows are then equal, so loss_start matches across reload groups.
            random.seed(targs.seed)
            torch.manual_seed(targs.seed)
            model, tok, dtype = tp.build(build_args(targs, dict(key)), device)
            model.train()
            snap = snapshot(model)
            try:
                kernels = ab.sdpa_eligibility(model, device, dtype)
            except Exception as e:  # diagnostics only
                kernels = {"error": f"{type(e).__name__}: {e}"[:200]}
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            for t in group:
                emit({"trial_number": t["trial_number"], "params": t["params"], "status": classify(e),
                      "error": "model load failed: " + err[:400], "n_gpu": n_gpu,
                      "gpu_name": meta["gpu_name"]})
            continue
        load_s = time.perf_counter() - t_load
        scale = {"value": 2.0 ** 16}          # GradScaler scale, carried within the group
        print(json.dumps({"group": dict(key), "load_s": round(load_s, 1)}), flush=True)
        for t in group:
            r = {"trial_number": t["trial_number"], "params": t["params"], "n_gpu": n_gpu,
                 "gpu_name": meta["gpu_name"], "group_load_s": load_s}
            if kernels:
                r["sdpa_kernels"] = kernels
            failed = None
            try:
                r.update(run_trial(model, tok, mixture, mix_state, snap, probe, targs, t["params"],
                                   device, dtype, own, base_threads, scale))
            except Exception as e:
                failed = (classify(e, t["params"].get("attn_backend")), f"{type(e).__name__}: {e}"[:400],
                          traceback.format_exc()[-1500:])
            if failed:
                # Free outside the except block: the traceback holds the tensors.
                r.update(status=failed[0], error=failed[1])
                if failed[0] == "error":
                    print(failed[2], flush=True)
                model.zero_grad(set_to_none=True)
                free(device)
            if device == "cuda":
                r.setdefault("peak_mem_gb", torch.cuda.max_memory_allocated() / 2**30)
            emit(r)
        del model, snap
        model = None
    torch.set_num_threads(base_threads)
    print(json.dumps({"bench_done": len(results), "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
