"""Run a batch of Optuna speed trials (from tools/optuna_loop.py propose) in one process.

Loading Gemma 4 E2B on Kaggle takes ~5 min, so the model is loaded once per group of
trials that share the same `requires_reload` params (ple_cpu, dtype). Each trial:

  1. resets: the start weights (LoRA, flag head, template delta), no grads, the data
     stream at its start, the seed; then a fresh optimizer, so all trials start equal;
  2. applies its params (grad_ckpt, max_state_tokens, torch_threads, sdpa_backend,
     attn_backend; the attention backend is swapped at runtime, see openjev/attn_backends.py;
     option_subsample_k, packing and pack_len change only the TRAINING micro-batches: the
     probe and the diagnostic keep all options, unpacked, so the guard stays comparable);
  3. round baseline only (`diag_mb` in its entry): one forward+backward at micro-batch
     diag_mb (16) to record whether it WOULD fit (peak < 0.92 x GPU memory), then the
     reset of step 1 again. It never changes the micro-batch;
  4. measures the loss on a fixed probe set (loss_start);
  5. runs the warm-up optimizer steps and measures the probe loss again (loss_guard).
     The quality guard uses this one: every trial has taken the same optimizer steps on
     the same records here, whatever its speed;
  6. runs the timed loop, then measures loss_end (informational) and peak GPU memory;
  7. profile=1 only: after all of the above (so samples/s, the guard and the data stream
     equal a non-profile twin), runs torch.profiler (CPU + CUDA) over --profile-steps more
     steps and stores a compact table as "profile".

Step = ONE optimizer step. Round 3+ entries carry study settings ("bench": global_batch,
micro_batch, warmup_steps, timed_steps): a step is grad_accum = global_batch / micro_batch
micro-batches of micro_batch records (forward + backward each, loss / grad_accum), then one
optimizer step, so samples/s = global_batch * steps / seconds at a FIXED micro-batch and
global batch for every trial. The `bs` param is ignored then (recorded as bs_ignored). A
trial that runs out of memory at that micro-batch is "infeasible_at_mb<micro_batch>"
(pruned, never retried smaller). With timed_steps the timed loop is exactly that many
steps (every trial times the same records; --trial-seconds is not used). Entries without
settings (rounds 1-2) keep the old behaviour: one optimizer step per micro-batch of `bs`,
timed for --trial-seconds.

Data: with --trial-stream (round 3+), every trial consumes the frozen trial stream
(openjev/trial_data.py, record ids) from its start, and the probe set is the frozen one;
otherwise a seeded Mixture (rewound per trial) and its first --probe-n records.

The fp16 GradScaler scale is carried from trial to trial inside a group, so a fresh
2^16 scale does not skip the first steps of every trial. A skipped WARM-UP step is redone on
the same records (warmup_retries), so the guard always follows warmup_steps applied steps;
skipped steps in the timed loop are counted (skipped_steps).

OOM, a missing SDPA kernel and other errors are caught per trial. Results go to
--out (rewritten after every trial) and one JSON line per trial is printed.

--validate FILE (quality checkpoint, PROFILE=optuna_validate; from optuna_loop.py
validate-plan) runs instead: for each config (round best, round baseline), from the same
start weights and the same seeded Mixture, `steps` optimizer steps of real SFT at the
fixed micro-batch / global batch (lr: train_prism warm-up + cosine over the steps), then
train_prism.evaluate on the frozen val set (--val-set): acc, nll, ece, per-source acc.

Every flag train_prism.py knows (--model, --train, --lr, --dtype, --cpu, ...) is passed
through; --bs, --max-state-tokens and --grad-ckpt are the defaults for trials that do
not set them, and --max-state-tokens is also the probe truncation.

  python bench_trials.py --trials trials.json --trial-stream trial_stream.json \\
      --model google/gemma-4-E2B-it --train 'data/*.train.jsonl'
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import glob
import hashlib
import json
import math
import random
import time
import traceback
from pathlib import Path

import torch

import train_prism as tp
from openjev import attn_backends as ab
from openjev import trial_data as td
from openjev.data import subsample_options
from openjev.mix import Mixture
from openjev.prism import log_rss, prism_collate

# Params this script knows how to apply. A trial with any other param fails, so a
# dimension marked `implemented: true` without code here never gives fake numbers.
TRIAL_PARAMS = {"bs", "grad_ckpt", "max_state_tokens", "torch_threads", "sdpa_backend",
                "attn_backend", "profile", "option_subsample_k", "packing", "pack_len"}
RELOAD_PARAMS = {"ple_cpu", "dtype"}
# yaml dtype choice -> train_prism --dtype. "bf16_emulated" is what --dtype auto picked on
# T4 before OJ-081 (bf16 without native support on sm75).
DTYPES = {"fp16": "fp16", "bf16_emulated": "bf16", "fp32": "fp32"}
HEADROOM = 0.92        # a micro-batch "fits" if its peak memory < HEADROOM x GPU memory
WARMUP_RETRIES = 10    # redo a warm-up step the fp16 GradScaler skipped, at most this often


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", default="trials.json")
    ap.add_argument("--trial-seconds", type=float, default=50.0,
                    help="timed loop length for entries without timed_steps")
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="default: the entry's warmup_steps, else 2")
    ap.add_argument("--probe-n", type=int, default=8, help="records in the fixed probe set (no stream)")
    ap.add_argument("--profile-steps", type=int, default=2, help="profiled steps (profile=1 trials)")
    ap.add_argument("--attn-chunk", type=int, default=ab.DEFAULT_CHUNK,
                    help="query rows per chunk for attn_backend=chunked_math")
    ap.add_argument("--trial-stream", default="", help="frozen trial stream (optuna_loop.py freeze)")
    ap.add_argument("--validate", default="", help="quality checkpoint plan (optuna_loop.py validate-plan)")
    ap.add_argument("--val-set", default="", help="frozen val set for --validate")
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
        elif k == "dtype":
            if v not in DTYPES:
                raise ValueError(f"dtype {v!r} not in {sorted(DTYPES)}")
            a.dtype = DTYPES[v]
        else:
            raise ValueError(f"reload param {k!r} has no handler in bench_trials.py")
    return a


def settings_of(entry: dict, targs, own) -> dict:
    """Effective step settings of a trials.json entry (see the module docstring)."""
    st = dict(entry.get("bench") or {})
    gb, mb = st.get("global_batch"), st.get("micro_batch")
    if gb and mb:
        gb, mb = int(gb), int(mb)
        if mb < 1 or gb % mb:
            raise ValueError(f"global_batch {gb} is not a multiple of micro_batch {mb}")
        out = {"fixed_mb": True, "global_batch": gb, "micro_batch": mb, "grad_accum": gb // mb}
    else:
        bs = int(entry.get("params", {}).get("bs", targs.bs))
        out = {"fixed_mb": False, "global_batch": bs, "micro_batch": bs, "grad_accum": 1}
    w = own.warmup_steps if own.warmup_steps is not None else st.get("warmup_steps", 2)
    out["warmup_steps"] = int(w)
    out["timed_steps"] = int(st["timed_steps"]) if st.get("timed_steps") else None
    out["diag_mb"] = int(st["diag_mb"]) if st.get("diag_mb") else None
    out["trial_stream_sha"] = st.get("trial_stream_sha")
    return out


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


def trial_status(e: BaseException, settings: dict, backend: str | None = None) -> str:
    """classify(), but an OOM at the fixed micro-batch is infeasible_at_mb<mb> (pruned:
    the config needs more memory than the round's micro-batch allows)."""
    c = classify(e, backend)
    if c == "oom" and settings.get("fixed_mb"):
        return f"infeasible_at_mb{settings['micro_batch']}"
    return c


# ---------------------------------------------------------------- helpers

def snapshot(model):
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def restore(model, snap):
    for n, p in model.named_parameters():
        if n in snap:
            p.copy_(snap[n])
            p.grad = None


def reset(model, data, snap, data_state, seed):
    """The trial reset: start weights, no grads, data stream at its start, seed."""
    restore(model, snap)
    model.zero_grad(set_to_none=True)
    data.set_state(data_state)
    random.seed(seed)
    torch.manual_seed(seed)


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


def ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def peak_gb(device):
    """Peak reserved memory since the last reset (None on CPU). Tests monkeypatch it."""
    return torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else None


def mem_limit_gb(device):
    """Total GPU memory (None on CPU). Tests monkeypatch it."""
    return torch.cuda.mem_get_info()[1] / 2**30 if device == "cuda" else None


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


def dynamo_graphs():
    """Graphs torch.compile has built so far (None if dynamo is not loaded)."""
    import sys
    if "torch._dynamo" not in sys.modules:
        return None
    try:
        from torch._dynamo.utils import counters
        return int(counters["stats"]["unique_graphs"])
    except Exception:
        return None


def free(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- one optimizer step

def accum_step(model, opt, scaler, next_batch, accum: int, device, backend):
    """ONE optimizer step: `accum` micro-batches (forward + backward, loss / accum),
    then unscale, clip, step. Shared by the speed trials and the quality checkpoint.
    Returns (mean micro-batch loss as a tensor, 1 if the fp16 scaler skipped the step)."""
    total = None
    for _ in range(accum):
        b = next_batch().to(device)
        with sdpa_ctx(backend):
            with tp.autocast(device, model):
                loss = tp.sft_loss(model.flag_logits(b), b)
            scaler.scale(loss / accum).backward()
        total = loss.detach() if total is None else total + loss.detach()
    before = scaler.get_scale()            # scaler.step syncs anyway (found_inf)
    tp.optimizer_step(opt, scaler)
    return total / accum, int(scaler.is_enabled() and scaler.get_scale() < before)


def mb_diag(model, tok, data, mst: int, mb: int, device, backend, pad_mult: int) -> dict:
    """Would micro-batch `mb` fit? One forward + backward on the next `mb` stream records;
    fits = no OOM and peak < HEADROOM x GPU memory. Only recorded, never used to pick the
    micro-batch. The caller resets weights, grads and data afterwards."""
    t0 = time.perf_counter()
    free(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    oom, peak = False, None
    try:
        b = prism_collate(tok, data.take(mb), mst, pad_to_multiple_of=pad_mult).to(device)
        with sdpa_ctx(backend):
            with tp.autocast(device, model):
                loss = tp.sft_loss(model.flag_logits(b), b)
            loss.backward()
        del b, loss
        if device == "cuda":
            torch.cuda.synchronize()
        peak = peak_gb(device)
    except Exception as e:
        if classify(e) != "oom":
            raise
        oom = True
    model.zero_grad(set_to_none=True)       # outside the except block: frees the tensors
    free(device)
    limit = mem_limit_gb(device)
    fits = (not oom) and (peak is None or limit is None or peak < HEADROOM * limit)
    return {"mb": mb, "fits": fits, "oom": oom, "peak_gb": peak, "limit_gb": limit,
            "headroom": HEADROOM, "probe_s": time.perf_counter() - t0}


# ---------------------------------------------------------------- one trial

def run_trial(model, tok, data, data_state, snap, probe, targs, params, device, dtype, own,
              base_threads, scale, settings=None):
    unknown = set(params) - TRIAL_PARAMS - RELOAD_PARAMS
    if unknown:
        raise ValueError(f"params {sorted(unknown)} have no handler in bench_trials.py")
    settings = settings or settings_of({"params": params}, targs, own)
    attn = params.get("attn_backend", "sdpa_default")
    try:
        ab.set_attn_backend(model, attn, chunk=own.attn_chunk)
        return _run_trial(model, tok, data, data_state, snap, probe, targs, params, device, dtype,
                          own, base_threads, scale, attn, settings)
    finally:
        ab.reset_attn_backend(model)   # a failed flex trial must not change the next trial


def apply_params(model, params, targs, base_threads):
    """Runtime params -> (max_state_tokens, sdpa backend)."""
    tp.set_grad_ckpt(model, is_on(params["grad_ckpt"]) if "grad_ckpt" in params else targs.grad_ckpt)
    torch.set_num_threads(int(params.get("torch_threads", base_threads)))
    return int(params.get("max_state_tokens", targs.max_state_tokens)), params.get("sdpa_backend", "default")


def data_params(params, targs) -> tuple[int, int]:
    """(option_subsample_k, pack_len or 0) for the TRAINING batches of a trial; the probe,
    the diagnostic and eval stay on all options, unpacked. Defaults = train_prism flags."""
    k = int(params.get("option_subsample_k", targs.option_subsample_k))
    on = is_on(params["packing"]) if "packing" in params else targs.packing == "on"
    return k, (int(params.get("pack_len", targs.pack_len)) if on else 0)


def train_collate(tok, recs, mst, pad_mult, k, pack_len, seed):
    """A training micro-batch. Sub-sampling draws from a per-record seeded RNG (str seed:
    stable across processes), so every trial, a redone warm-up step and a second run see
    the same options for the same record."""
    if k:
        recs = [subsample_options(r, k, random.Random(f"{seed}:{r.id}")) for r in recs]
    return prism_collate(tok, recs, mst, pad_to_multiple_of=pad_mult, pack_len=pack_len)


def _run_trial(model, tok, data, data_state, snap, probe, targs, params, device, dtype, own,
               base_threads, scale, attn, st):
    reset(model, data, snap, data_state, targs.seed)
    mb, accum, gb = st["micro_batch"], st["grad_accum"], st["global_batch"]
    mst, backend = apply_params(model, params, targs, base_threads)
    sub_k, pack_len = data_params(params, targs)
    profile = is_on(params.get("profile", 0))
    # flex compiles one graph per shape: pad T to the BlockMask block so shapes repeat.
    pad_mult = ab.FLEX_BLOCK if attn == "flex" else 8
    extra = {"attn_backend": attn, "dtype": str(dtype).replace("torch.", ""), "mb_used": mb,
             "grad_accum": accum, "global_batch": gb, "fixed_mb": st["fixed_mb"]}
    if st["fixed_mb"] and "bs" in params:
        extra["bs_ignored"] = params["bs"]
    if st["diag_mb"]:
        extra["mb_diag"] = mb_diag(model, tok, data, mst, st["diag_mb"], device, backend, pad_mult)
        reset(model, data, snap, data_state, targs.seed)   # no learning or data leaks into the trial

    free(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Constant lr, no warmup: a scheduler would barely move the weights in a few steps.
    opt = tp.make_optimizer(model, targs)
    scaler = torch.amp.GradScaler("cuda", init_scale=scale["value"],
                                  enabled=(device == "cuda" and dtype == torch.float16))
    scale_start = scale["value"]
    loss_start = probe_loss(model, tok, probe, targs, device)

    losses, counts, skipped, ids = [], {"real": 0, "pad": 0, "rows": 0, "mbs": 0}, 0, []

    def next_batch():
        with torch.profiler.record_function("oj::collate"):
            recs = data.take(mb)
            b = train_collate(tok, recs, mst, pad_mult, sub_k, pack_len, targs.seed)
        ids.extend(r.id for r in recs)
        counts["real"] += int(b.attention_mask.sum())          # CPU, no sync
        counts["pad"] += b.attention_mask.numel()
        counts["rows"] += b.input_ids.shape[0]
        counts["mbs"] += 1
        return b

    def step():
        nonlocal skipped
        loss, sk = accum_step(model, opt, scaler, next_batch, accum, device, backend)
        skipped += sk
        losses.append(loss)

    # Warm-up: exactly warmup_steps APPLIED optimizer steps before the guard. A step the fp16
    # GradScaler skips (inf/NaN grads; a fresh 2^16 scale at the start of a reload group) is
    # redone on the same records: the skip changed no weights or Adam state, only the scale.
    # Otherwise the baseline's guard drop could be ~0 and the guard would pass everything.
    t_w = time.perf_counter()
    retries = 0
    for _ in range(st["warmup_steps"]):
        for _attempt in range(WARMUP_RETRIES + 1):
            pos, n_ids = data.get_state(), len(ids)
            _loss, sk = accum_step(model, opt, scaler, next_batch, accum, device, backend)
            if not sk:
                break
            data.set_state(pos)
            del ids[n_ids:]
            retries += 1
        else:
            raise RuntimeError(f"fp16 GradScaler skipped the warm-up step {WARMUP_RETRIES + 1} times")
    extra["warmup_retries"] = retries
    if device == "cuda":
        torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t_w      # includes flex compile time
    warm_ids = list(ids)
    losses.clear()
    loss_guard = probe_loss(model, tok, probe, targs, device)
    counts.update(real=0, pad=0, rows=0, mbs=0)
    if device == "cuda":
        torch.cuda.synchronize()
    graphs0 = dynamo_graphs()
    t0 = time.perf_counter()
    steps = 0
    while True:
        step()
        steps += 1
        if st["timed_steps"]:
            if steps >= st["timed_steps"]:
                break
        elif time.perf_counter() - t0 >= own.trial_seconds:
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
    extra.update(warmup_s=warmup_s, warmup_ids_sha=ids_sha(warm_ids), data_ids_sha=ids_sha(ids),
                 records_used=len(ids), timed_steps=st["timed_steps"],
                 rows_per_mb=counts["rows"] / max(counts["mbs"], 1))
    if getattr(data, "wrapped", False):
        extra["stream_wrapped"] = True
    if graphs0 is not None:
        # > 0: torch.compile (flex) built new graphs inside the timed loop, so this
        # trial's samples/s includes compile time.
        extra["graphs_in_timed_loop"] = dynamo_graphs() - graphs0
    if profile:
        # After loss_end and timing: the profiled steps change nothing that is measured.
        ab.set_attn_backend(model, attn, chunk=own.attn_chunk, label=True)
        extra["profile"] = run_profile(step, device, own.profile_steps)
        extra["is_profile"] = True
    n = steps * gb
    return {**extra, "status": "nan" if nan else "ok", "steps": steps, "samples": n,
            "seconds": sec, "samples_per_sec": n / sec, "s_per_sample": sec / n,
            "s_per_step": sec / steps, "tokens_per_sec": counts["real"] / sec,
            "padded_tokens_per_sec": counts["pad"] / sec,
            "pad_frac": 1 - counts["real"] / max(counts["pad"], 1),
            "peak_mem_gb": peak, "loss_start": loss_start, "loss_guard": loss_guard,
            "guard_steps": st["warmup_steps"], "loss_end": loss_end, "scale_start": scale_start,
            "scale_end": scale["value"], "skipped_steps": skipped,
            "train_loss_first": sum(train[:k]) / k, "train_loss_last": sum(train[-k:]) / k}


# ---------------------------------------------------------------- data

def load_data(own, targs):
    """(stream, probe, meta): the frozen trial stream (--trial-stream) or a seeded Mixture."""
    if own.trial_stream:
        frozen = json.loads(Path(own.trial_stream).read_text())
        if td.sha_of({"stream": frozen["stream"], "probe": frozen["probe"]}) != frozen["sha"]:
            raise SystemExit(f"{own.trial_stream}: content does not match its sha")
        recs = td.load_refs(targs.train, frozen["stream"])
        probe = td.load_refs(targs.train, frozen["probe"])
        stems = [r[0] for r in frozen["stream"]]
        return td.FixedStream(recs), probe, {"trial_stream_sha": frozen["sha"],
                                             "trial_stream_len": len(recs), "stems": stems}
    paths = sorted(glob.glob(targs.train))
    if not paths:
        raise SystemExit(f"no train files match {targs.train}")
    mixture = Mixture(paths, alpha=targs.alpha, seed=targs.seed)
    probe = mixture.take(own.probe_n)            # fixed probe set, same for every trial
    return mixture, probe, {}


def stream_tokens(tok, stream, stems) -> dict:
    """p50 / p95 state tokens per source file of the stream (real tokenizer, so it can be
    compared with data/STATS.md "State token length")."""
    per: dict[str, list] = {}
    for r, s in zip(stream.records, stems):
        per.setdefault(s, []).append(len(tok(r.state, add_special_tokens=False)["input_ids"]))
    return {s: {"n": len(v), "p50": td.quantile(v, 0.5), "p95": td.quantile(v, 0.95)}
            for s, v in sorted(per.items())}


# ---------------------------------------------------------------- quality checkpoint

def run_validate(own, targs, device, meta, emit):
    plan = json.loads(Path(own.validate).read_text())
    st = plan["settings"]
    gb, mb, n_steps = int(st["global_batch"]), int(st["micro_batch"]), int(st["steps"])
    if gb % mb:
        raise SystemExit(f"global_batch {gb} is not a multiple of micro_batch {mb}")
    vs = json.loads(Path(own.val_set).read_text())
    if st.get("val_set_sha") and vs["sha"] != st["val_set_sha"]:
        raise SystemExit(f"val set sha {vs['sha']} != planned {st['val_set_sha']}")
    val = td.load_refs(targs.val, vs["records"])
    paths = sorted(glob.glob(targs.train))
    # The training records: steps x global_batch refs sampled like the Mixture (per-source
    # counts by the alpha weights, seeded), loaded by line. The full Mixture (403k records)
    # costs ~1.9 GB host RAM before the model load; this holds only what is trained on.
    log_rss("start")
    refs = td.freeze_stream(paths, n_steps * gb, 0, alpha=targs.alpha, seed=targs.seed, check=False)
    mixture = td.FixedStream(td.load_refs(paths, refs["stream"]))
    state0 = mixture.get_state()
    log_rss("after_data")
    meta.update(validate=True, plan=st, train_refs_sha=refs["sha"], val_set_sha=vs["sha"], val_n=len(val), steps=n_steps,
                global_batch=gb, micro_batch=mb)
    base_threads = torch.get_num_threads()
    model = None
    for cfg in plan["configs"]:
        params = cfg["params"]
        r = {"name": cfg["name"], "trial_number": cfg.get("trial_number"), "params": params,
             "steps_planned": n_steps, "global_batch": gb, "micro_batch": mb, "grad_accum": gb // mb}
        failed = None
        try:
            model = None
            free(device)
            random.seed(targs.seed)
            torch.manual_seed(targs.seed)
            reload = {k: params[k] for k in sorted(RELOAD_PARAMS) if k in params}
            log_rss("before_load")
            model, tok, dtype = tp.build(build_args(targs, reload), device)
            model.train()
            r.update(_validate_one(model, tok, mixture, state0, val, targs, params, device, dtype,
                                   own, base_threads, gb, mb, n_steps))
        except Exception as e:
            st_ = {"fixed_mb": True, "micro_batch": mb}
            failed = (trial_status(e, st_, params.get("attn_backend")), f"{type(e).__name__}: {e}"[:400],
                      traceback.format_exc()[-1500:])
        if failed:
            r.update(status=failed[0], error=failed[1])
            print(failed[2], flush=True)
        model = None
        free(device)
        emit(r)
    torch.set_num_threads(base_threads)


def _validate_one(model, tok, mixture, state0, val, targs, params, device, dtype, own,
                  base_threads, gb, mb, n_steps):
    attn = params.get("attn_backend", "sdpa_default")
    ab.set_attn_backend(model, attn, chunk=own.attn_chunk)
    try:
        mixture.set_state(state0)
        mst, backend = apply_params(model, params, targs, base_threads)
        sub_k, pack_len = data_params(params, targs)
        pad_mult = ab.FLEX_BLOCK if attn == "flex" else 8
        opt = tp.make_optimizer(model, targs)
        warm = max(1, int(targs.warmup * n_steps))          # same schedule as train_prism
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / n_steps))))
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and dtype == torch.float16))
        losses, skipped = [], 0
        t0 = time.perf_counter()
        for _ in range(n_steps):
            loss, sk = accum_step(model, opt, scaler,
                                  lambda: train_collate(tok, mixture.take(mb), mst, pad_mult, sub_k,
                                                        pack_len, targs.seed),
                                  gb // mb, device, backend)
            sched.step()
            losses.append(float(loss))
            skipped += sk
        sec = time.perf_counter() - t0
        eargs = copy.copy(targs)
        eargs.bs, eargs.max_state_tokens = mb, mst
        with sdpa_ctx(backend):
            m = tp.evaluate(model, tok, val, eargs, device)
    finally:
        ab.reset_attn_backend(model)
    k = max(1, min(10, len(losses) // 2))
    nan = not (all(math.isfinite(x) for x in losses) and math.isfinite(m["nll"]))
    return {"status": "nan" if nan else "ok", "steps": n_steps, "seconds": sec,
            "skipped_steps": skipped, "train_loss_first": sum(losses[:k]) / k,
            "train_loss_last": sum(losses[-k:]) / k, "val": m}


# ---------------------------------------------------------------- main

def main(argv=None):
    own, targs = parse(argv)
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

    if own.validate:
        run_validate(own, targs, device, meta, emit)
        print(json.dumps({"validate_done": len(results), "out": str(out)}), flush=True)
        return

    trials = json.loads(Path(own.trials).read_text())
    log_rss("start")
    data, probe, dmeta = load_data(own, targs)
    log_rss("after_data")
    meta.update({k: v for k, v in dmeta.items() if k != "stems"})
    data_state = data.get_state()
    base_threads = torch.get_num_threads()
    want_sha = {(t.get("bench") or {}).get("trial_stream_sha") for t in trials} - {None}
    if want_sha and want_sha != {dmeta.get("trial_stream_sha")}:
        raise SystemExit(f"trials.json was proposed for trial stream {sorted(want_sha)}, "
                         f"but --trial-stream is {dmeta.get('trial_stream_sha')}")

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
        if device == "cuda" and dict(key).get("dtype") == "fp32":
            # fp32 E2B text weights are ~18.5 GB: more than a 16 GB T4, and the
            # load would hold ~22 GB of host RAM (r2 SIGKILL). Skip without loading.
            for t in group:
                emit({"trial_number": t["trial_number"], "params": t["params"],
                      "status": "unsupported", "n_gpu": n_gpu, "gpu_name": meta["gpu_name"],
                      "error": "fp32 skipped: weights exceed T4 memory and host RAM"})
            continue
        t_load = time.perf_counter()
        try:
            # Same seed for every group: the random flag head, LoRA A and new embedding
            # rows are then equal, so loss_start matches across reload groups.
            random.seed(targs.seed)
            torch.manual_seed(targs.seed)
            log_rss("before_load")
            model, tok, dtype = tp.build(build_args(targs, dict(key)), device)
            model.train()
            snap = snapshot(model)
            if dmeta.get("stems") and "stream_state_tokens" not in meta:
                meta["stream_state_tokens"] = stream_tokens(tok, data, dmeta["stems"])
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
            if dmeta.get("trial_stream_sha"):
                r["trial_stream_sha"] = dmeta["trial_stream_sha"]
            if kernels:
                r["sdpa_kernels"] = kernels
            failed, st = None, None
            try:
                st = settings_of(t, targs, own)
                r.update(mb_used=st["micro_batch"], grad_accum=st["grad_accum"],
                         global_batch=st["global_batch"])
                r.update(run_trial(model, tok, data, data_state, snap, probe, targs, t["params"],
                                   device, dtype, own, base_threads, scale, st))
            except Exception as e:
                failed = (trial_status(e, st or {}, t["params"].get("attn_backend")),
                          f"{type(e).__name__}: {e}"[:400], traceback.format_exc()[-1500:])
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
