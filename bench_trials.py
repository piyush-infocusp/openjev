"""Run a batch of Optuna speed trials (from tools/optuna_loop.py propose) in one process.

Loading Gemma 4 E2B on Kaggle takes ~5 min, so the model is loaded once per group of
trials that share the same `requires_reload` params (ple_cpu, dtype, quant). Each trial:

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
With --stage-streams (length curriculum, tools/length_curriculum.py freeze) each trial times
the frozen stream of its `length_stage` param (records whose encoded length is in that stage,
never cut; 0 = the mixed curriculum-off reference) from its start, packed at
stage_pack_mult x stage max (0 = unpacked, length-bucketed); results carry length_stage,
stage_stream_sha and pack_len_used. Compare stage trials only within a stage.

--bench-lengths (study openjev-speed-v3+, user decision 2026-11-23: no single pack_len) is the
FIXED multi-length bench set (tools/length_curriculum.py bench; runs/optuna/bench_lengths.json):
32 uncut records per length bucket (<=128 .. <=2048) at the file's max_state_tokens (2048,
overrides the param), unpacked (inside a bucket, packing at the bucket max = unpacked). A trial:
probe loss; warmup_steps shared warm-up optimizer step(s) of one micro-batch per bucket, longest
bucket first (allocator at its peak shape first; buckets 2048, 1024, 512, 256 = one global batch);
probe loss (the guard); then each bucket in turn (128 -> 2048), timed_steps optimizer steps of
that bucket's records only (micro-batches of micro_batch, global_batch records per step), with
samples/s, tokens/s, pad_frac and peak GB per bucket ("buckets"). The objective samples_per_sec
is the weighted mix (packing.mix_objective, weights stored in the file = share of training steps
per bucket): 1 / sum_b w_b x seconds per sample in b. An OOM in any bucket prunes the trial
(infeasible_at_mb<micro_batch>): the mix is undefined without every bucket. Trials carry
bench_set_sha; a trials.json proposed for another set (or for a trial stream) is refused.

The fp16 GradScaler scale is carried from trial to trial inside a group, so a fresh
2^16 scale does not skip the first steps of every trial. A skipped WARM-UP step is redone on
the same records (warmup_retries), so the guard always follows warmup_steps applied steps;
skipped steps in the timed loop are counted (skipped_steps).

OOM, a missing SDPA kernel and other errors are caught per trial. Results go to
--out (rewritten after every trial) and one JSON line per trial is printed.

DDP (OJ-062): a trial with ddp=on needs every GPU, so a plain launch hands the ddp=on trials
to its own torchrun launch (`run_ddp_split`: --nproc_per_node = the GPU count, or
OJ_DDP_NPROC on CPU) after the other trials (plain or --dual-gpu, unchanged) and merges both
result files. Under torchrun every rank loads the model (one rank at a time: host RAM),
runs the same trials and reads the same frozen stream; each optimizer step takes the whole
global batch, cuts the single-process micro-batches and runs a token-balanced share of them
(openjev/dist.py), so the records (data_ids_sha) equal a single-process trial's.
sync_mode=ddp: one gradient all-reduce per step; sync_mode=local_sgd: weights averaged every
sync_every steps and before each probe loss. torch_threads is split over the ranks. Only
rank 0 prints and writes --out; any trial error ends the launch (a rank stuck in an
all-reduce would otherwise hang).

--validate FILE (quality checkpoint, PROFILE=optuna_validate; from optuna_loop.py
validate-plan) runs instead: for each config (round best, round baseline), from the same
start weights and the same seeded Mixture, `steps` optimizer steps of real SFT at the
fixed micro-batch / global batch (lr: train_prism warm-up + cosine over the steps), then
train_prism.evaluate on the frozen val set (--val-set): acc, nll, ece, per-source acc.
Batching follows train_prism's defaults: length-grouped micro-batches (--length-grouping on,
--group-chunk mega-chunks of the plan's micro_batch, seeded) and packing per data_params
(--packing on / --pack-len 1024 unless the config sets packing); --length-grouping off = the
frozen order (the pre-2026-11-23 behaviour).

torch_compile (OJ-079, openjev/compile.py): the decoder layers only, dynamic shapes, compiled
cold per trial (Inductor FX graph cache off). Before loss_start a compile pre-pass (probe under
no_grad + forward/backward on every distinct micro-batch shape of the trial, no optimizer step)
builds the graphs: compile_s, compile_graphs, compile_recompiles, compile_shapes are recorded
apart from the timed numbers; graphs_in_timed_loop > 0 flags a compile inside the timing.
Inductor / Triton / CUDA-graph / recompile-limit failures are "unsupported"; the layers are
uncompiled (dynamo reset) after every trial, so the next trial is eager again.

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
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch

import train_prism as tp
from openjev import attn_backends as ab
from openjev import compile as tc
from openjev import dist as jdist
from openjev import trial_data as td
from openjev.data import subsample_options
from openjev.mix import Mixture
from openjev.packing import BENCH_KIND, mix_objective
from openjev.prism import QUANTS, log_rss, prism_collate

# Params this script knows how to apply. A trial with any other param fails, so a
# dimension marked `implemented: true` without code here never gives fake numbers.
TRIAL_PARAMS = {"bs", "grad_ckpt", "max_state_tokens", "torch_threads", "sdpa_backend",
                "attn_backend", "profile", "option_subsample_k", "packing", "pack_len",
                "sync_mode", "sync_every",
                "length_stage", "stage_pack_mult",   # length curriculum: --stage-streams
                "torch_compile"}                     # OJ-079: openjev/compile.py
# quant: OJ-075 bitsandbytes base weights. ddp: OJ-062, needs a torchrun launch
# (run_ddp_split), so it groups like a reload param.
RELOAD_PARAMS = {"ple_cpu", "dtype", "quant", "ddp"}
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
    ap.add_argument("--stage-streams", default="",
                    help="frozen per-stage streams (tools/length_curriculum.py freeze); each trial "
                         "times the stage of its length_stage param, from that stream's start")
    ap.add_argument("--bench-lengths", default="",
                    help="the fixed multi-length bench set (tools/length_curriculum.py bench); every "
                         "trial times every bucket, objective = weighted samples/s (study v3+)")
    ap.add_argument("--validate", default="", help="quality checkpoint plan (optuna_loop.py validate-plan)")
    ap.add_argument("--val-set", default="", help="frozen val set for --validate")
    ap.add_argument("--out", default="/kaggle/working/out/optuna_results.json")
    ap.add_argument("--dual-gpu", action="store_true",
                    help="split trials in 2 shards, one subprocess per GPU (CUDA_VISIBLE_DEVICES=0/1)")
    ap.add_argument("--gpu-index", type=int, default=None, help="(shard child) physical GPU index")
    ap.add_argument("--ready-file", default="", help="(shard child) touched once the model load is done")
    ap.add_argument("--wait-ready", type=float, default=1500.0,
                    help="dual-gpu: max seconds to wait for shard 0's ready file before starting shard 1")
    own, rest = ap.parse_known_args(argv)
    return own, tp.parse(rest)


# ---------------------------------------------------------------- dual-gpu shards

LOAD_COST, TRIAL_COST = 7.0, 1.0       # minutes: a model load, one trial (balancing only)


def _group_key(t):
    keys = t.get("reload_keys") or sorted(RELOAD_PARAMS & set(t["params"]))
    return tuple((k, t["params"][k]) for k in sorted(keys) if k in t["params"])


def split_shards(trials: list, n: int = 2) -> list[list]:
    """Split trials into n shards. Whole reload groups stay together (a group costs one model
    load) and are balanced greedily by cost; if that leaves a shard empty (e.g. one group), the
    largest group is dealt round-robin instead. Order inside a shard follows the input."""
    groups: dict[tuple, list] = {}
    for t in trials:
        groups.setdefault(_group_key(t), []).append(t)
    cost = lambda g: LOAD_COST + TRIAL_COST * len(g)
    shards: list[list[list]] = [[] for _ in range(n)]
    load = [0.0] * n
    for g in sorted(groups.values(), key=cost, reverse=True):
        i = load.index(min(load))
        shards[i].append(g)
        load[i] += cost(g)
    out = [[t for g in sh for t in g] for sh in shards]
    if len(trials) >= n and any(not s for s in out):
        out = [[] for _ in range(n)]
        for g in groups.values():
            for j, t in enumerate(g):
                out[j % n].append(t)
    order = {id(t): i for i, t in enumerate(trials)}
    return [sorted(s, key=lambda t: order[id(t)]) for s in out]


def merge_results(paths: list, out: Path, extra_meta: dict | None = None) -> dict:
    """Merge shard result files into one optuna_results.json ({"meta","results"}, the format
    tools/optuna_loop.py ingest reads). Missing/partial shards are skipped; results sorted by
    trial_number; meta comes from the first shard that has one."""
    meta, results = {}, []
    for p in paths:
        try:
            d = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            continue
        meta = meta or d.get("meta") or {}
        results += d.get("results") or []
    results.sort(key=lambda r: r.get("trial_number", 0))
    doc = {"meta": {**meta, **(extra_meta or {})}, "results": results}
    tmp = Path(out).with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(out)
    return doc


def strip_argv(argv, with_value=(), flags=()) -> list:
    """argv without the flags in `with_value` (and their value) and the bare `flags`."""
    base, k, argv = [], 0, list(argv)
    while k < len(argv):
        a = argv[k]
        if a in flags:
            k += 1
        elif a in with_value:
            k += 2
        elif a.split("=")[0] in with_value:
            k += 1
        else:
            base.append(a)
            k += 1
    return base


def is_ddp(t) -> bool:
    return is_on(t["params"].get("ddp", "off"))


def ddp_nproc(device: str) -> int:
    """Processes for a ddp=on launch: every GPU on CUDA; OJ_DDP_NPROC (default 2) on CPU."""
    if device == "cuda":
        return torch.cuda.device_count()
    return int(os.environ.get("OJ_DDP_NPROC", "2"))


def run_ddp_split(own, argv, trials, device) -> int:
    """Parent: the ddp=off trials as before (one process, or --dual-gpu shards), then the
    ddp=on trials in ONE torchrun launch over all GPUs; the part files are merged into
    --out. With < 2 GPUs the ddp=on trials are recorded as unsupported."""
    out = Path(own.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    base = strip_argv(argv, ("--trials", "--out"))
    script = str(Path(__file__).resolve())
    files, codes = [], []
    for name, ts in (("single", [t for t in trials if not is_ddp(t)]), ("ddp", [t for t in trials if is_ddp(t)])):
        if not ts:
            continue
        tf, rf = out.with_name(f"{out.stem}.{name}.trials.json"), out.with_name(f"{out.stem}.{name}.json")
        tf.write_text(json.dumps(ts))
        rf.unlink(missing_ok=True)
        files.append(rf)
        env = dict(os.environ)
        if name == "single":
            cmd = [sys.executable, script, *base]
        else:
            n = ddp_nproc(device)
            if n < 2:
                rf.write_text(json.dumps({"meta": {}, "results": [
                    {"trial_number": t["trial_number"], "params": t["params"], "status": "unsupported",
                     "error": f"ddp=on needs >= 2 processes, have {n} GPU(s)"} for t in ts]}))
                codes.append(0)
                continue
            # torchrun sets OMP_NUM_THREADS=1 when it is unset: give each rank its share.
            env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // n)))
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node",
                   str(n), script, *strip_argv(base, (), ("--dual-gpu",))]
        print(json.dumps({"bench_part": name, "trials": [t["trial_number"] for t in ts]}), flush=True)
        codes.append(subprocess.call([*cmd, "--trials", str(tf), "--out", str(rf)], env=env))
    merge_results(files, out, {"ddp_split": True, "part_exit": codes})
    print(json.dumps({"bench_done_split": str(out), "exit": codes}), flush=True)
    return max(codes) if codes else 0


def run_dual(own, argv) -> int:
    """Parent: one bench_trials.py subprocess per GPU, shard 1 staggered until shard 0 has
    loaded its model (two concurrent host-RAM load peaks would risk a SIGKILL)."""
    out = Path(own.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shards = [s for s in split_shards(json.loads(Path(own.trials).read_text()), 2) if s]
    base = strip_argv(argv, ("--trials", "--out", "--gpu-index", "--ready-file", "--wait-ready"),
                      ("--dual-gpu",))
    procs, files, prev_ready = [], [], None
    for i, sh in enumerate(shards):
        tf, rf, ready = (out.with_name(f"{out.stem}.shard{i}{x}") for x in (".trials.json", ".json", ".ready"))
        tf.write_text(json.dumps(sh))
        ready.unlink(missing_ok=True)
        files.append(rf)
        if i > 0:
            t0 = time.time()
            while not prev_ready.exists() and procs[-1].poll() is None and time.time() - t0 < own.wait_ready:
                time.sleep(2)
            print(json.dumps({"dual_gpu_stagger": i, "waited_s": round(time.time() - t0, 1),
                              "ready": prev_ready.exists()}), flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), *base, "--trials", str(tf), "--out", str(rf),
               "--gpu-index", str(i), "--ready-file", str(ready)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}
        print(json.dumps({"dual_gpu_launch": i, "trials": [t["trial_number"] for t in sh]}), flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
        prev_ready = ready
    codes = [p.wait() for p in procs]
    merge_results(files, out, {"dual_gpu": True, "shards": len(shards), "shard_exit": codes})
    print(json.dumps({"bench_done_dual": str(out), "exit": codes}), flush=True)
    return max(codes) if codes else 0


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
        elif k == "quant":
            if v not in QUANTS:
                raise ValueError(f"quant {v!r} not in {list(QUANTS)}")
            a.quant = v
        elif k == "ddp":
            pass                       # the launch (run_ddp_split / torchrun), not a build flag
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
    out["bench_set_sha"] = st.get("bench_set_sha")
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


def classify(e: BaseException, backend: str | None = None, compile_mode: str | None = None) -> str:
    msg = str(e)
    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg.lower():
        return "oom"
    if compile_mode not in (None, "off") and tc.is_compile_error(e):
        return "unsupported"      # Inductor / Triton / CUDA graphs / recompile limit (OJ-079)
    if isinstance(e, ab.Unsupported) or "No available kernel" in msg or "No viable backend" in msg \
            or "does not support backward on CPU" in msg:
        return "unsupported"
    if backend == "flex" and any(k in f"{type(e).__name__}: {msg}".lower() for k in FLEX_UNSUPPORTED):
        return "unsupported"
    return "error"


def trial_status(e: BaseException, settings: dict, backend: str | None = None,
                 compile_mode: str | None = None) -> str:
    """classify(), but an OOM at the fixed micro-batch is infeasible_at_mb<mb> (pruned:
    the config needs more memory than the round's micro-batch allows)."""
    c = classify(e, backend, compile_mode)
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

def accum_step(model, opt, scaler, batches, accum: int, device, backend, dp=None):
    """ONE optimizer step: the `accum` micro-batches of `batches` (an iterable; THIS rank's
    share under DDP; forward + backward, loss / accum), then unscale, clip, step. Shared by
    the speed trials and the quality checkpoint. dp (openjev.dist.DataParallel): no_sync on
    all but the last micro-step (ddp), the weight average every sync_every steps (local_sgd).
    Returns (mean micro-batch loss as a tensor, 1 if the fp16 scaler skipped the step; on
    any rank under DDP)."""
    net = dp.net if dp is not None else model
    total = None
    for i, b in enumerate(batches):
        b = b.to(device)
        with dp.sync(i == accum - 1) if dp is not None else contextlib.nullcontext():
            with sdpa_ctx(backend):
                with tp.autocast(device, model):
                    loss = tp.sft_loss(net(b), b)
                scaler.scale(loss / accum).backward()
        total = loss.detach() if total is None else total + loss.detach()
    before = scaler.get_scale()            # scaler.step syncs anyway (found_inf)
    tp.optimizer_step(opt, scaler)
    skipped = int(scaler.is_enabled() and scaler.get_scale() < before)
    if dp is not None:
        if dp.world > 1 and scaler.is_enabled():
            skipped = int(dp.any(skipped))
        dp.after_step()
    return total / accum, skipped


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
              base_threads, scale, settings=None, dp=None):
    unknown = set(params) - TRIAL_PARAMS - RELOAD_PARAMS
    if unknown:
        raise ValueError(f"params {sorted(unknown)} have no handler in bench_trials.py")
    settings = settings or settings_of({"params": params}, targs, own)
    attn = params.get("attn_backend", "sdpa_default")
    try:
        ab.set_attn_backend(model, attn, chunk=own.attn_chunk)
        apply_compile(model, params)
        if isinstance(data, BenchLengths):
            return _run_bench_lengths(model, tok, data, snap, probe, targs, params, device, dtype,
                                      own, base_threads, scale, attn, settings, dp)
        return _run_trial(model, tok, data, data_state, snap, probe, targs, params, device, dtype,
                          own, base_threads, scale, attn, settings, dp)
    finally:
        ab.reset_attn_backend(model)   # a failed flex trial must not change the next trial
        tc.unset_torch_compile(model)  # nor a compiled one (eager again, dynamo reset)


def apply_compile(model, params) -> str:
    """torch_compile param -> compiled decoder layers (off: nothing touched). Bench compiles
    cold: Inductor's on-disk FX graph cache is turned off, so every compile trial pays (and
    records) the full compile time, as a fresh Kaggle session would; a recompile-limit hit
    raises (unsupported) instead of silently timing eager layers."""
    mode = params.get("torch_compile", "off")
    if mode != "off":
        import torch._inductor.config as icfg
        icfg.fx_graph_cache = False
    return tc.set_torch_compile(model, mode, fail_on_limit=True)


def compile_prepass(model, tok, probe, targs, device, backend, batches, dp=None) -> dict:
    """torch_compile on only, before loss_start: build every graph the trial will need, so
    compile time stays out of loss/speed numbers. The probe once under no_grad (eval graphs),
    then forward + backward (no scaler, no optimizer step, no_sync under DDP) on one micro-batch
    of every distinct (rows, T) shape the trial will train on (reduce-overhead records one CUDA
    graph per shape). Grads are dropped; the caller resets weights, data and seed after."""
    before = tc.stats()
    net = dp.net if dp is not None else model
    shapes, first = 0, None
    with tc.Timer() as t:
        probe_loss(model, tok, probe, targs, device)
        seen = set()
        for b in batches:
            key = tuple(b.input_ids.shape)
            if key in seen:
                continue
            seen.add(key)
            shapes += 1
            if shapes == 2:
                first = tc.stats()          # graphs after the probe + the first train shape
            b = b.to(device)
            with dp.sync(False) if dp is not None else contextlib.nullcontext():
                with sdpa_ctx(backend):
                    with tp.autocast(device, model):
                        loss = tp.sft_loss(net(b), b)
                    loss.backward()
            del b, loss
        model.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
    # compile_graphs: every graph built (layer variants x train / no-grad x shapes);
    # compile_recompiles: those built for the 2nd+ train shape (guard failures on new shapes).
    return {"compile_s": t.s, "compile_graphs": tc.graphs_since(before),
            "compile_recompiles": tc.graphs_since(first) if first is not None else 0,
            "compile_shapes": shapes}


def apply_params(model, params, targs, base_threads, world: int = 1):
    """Runtime params -> (max_state_tokens, sdpa backend). torch_threads is per machine:
    under DDP each of the `world` ranks gets its share."""
    tp.set_grad_ckpt(model, is_on(params["grad_ckpt"]) if "grad_ckpt" in params else targs.grad_ckpt)
    torch.set_num_threads(max(1, int(params.get("torch_threads", base_threads)) // world))
    return int(params.get("max_state_tokens", targs.max_state_tokens)), params.get("sdpa_backend", "default")


def data_params(params, targs) -> tuple[int, int]:
    """(option_subsample_k, pack_len or 0) for the TRAINING batches of a trial; the probe,
    the diagnostic and eval stay on all options, unpacked. Defaults = train_prism flags."""
    k = int(params.get("option_subsample_k", targs.option_subsample_k))
    stage = int(params.get("length_stage", 0))
    if stage:      # length curriculum: pack_len = stage_pack_mult x stage max (0 = unpacked)
        return k, int(params.get("stage_pack_mult", 1)) * stage
    on = is_on(params["packing"]) if "packing" in params else targs.packing == "on"
    return k, (int(params.get("pack_len", targs.pack_len)) if on else 0)


def frozen_rows(data, pos, mb, mst, k, pack_len):
    """Frozen packed rows (tools/freeze_packed.py) for the micro-batch that starts at stream
    position `pos`, or None (no layout, other conditions: pack on the fly, same rule)."""
    L = getattr(data, "layout", None)
    if (L is None or pos is None or not pack_len or k or pos % mb or L["micro_batch"] != mb
            or L["pack_len"] != pack_len or L["max_state_tokens"] != mst):
        return None
    return L["mbs"][pos // mb % len(L["mbs"])]


def subsample_recs(recs, k, seed):
    """Sub-sampling from a per-record seeded RNG (str seed: stable across processes), so
    every trial, a redone warm-up step, a second run and any rank see the same options for
    the same record."""
    if k:
        recs = [subsample_options(r, k, random.Random(f"{seed}:{r.id}")) for r in recs]
    return recs


def train_collate(tok, recs, mst, pad_mult, k, pack_len, seed, rows_of=None):
    """A training micro-batch (sub-sampled, see subsample_recs)."""
    return prism_collate(tok, subsample_recs(recs, k, seed), mst, pad_to_multiple_of=pad_mult,
                         pack_len=pack_len, rows_of=rows_of)


def trial_dp(dps: dict, model, params, device):
    """The DataParallel context of a trial (sync_mode / sync_every params). One per sync
    mode per loaded model, so DDP wraps the model once (a collective, in the same trial on
    every rank). None outside torchrun."""
    if jdist.world() <= 1:
        return None
    mode = params.get("sync_mode", "ddp")
    if mode not in dps:
        dps[mode] = jdist.DataParallel(device, mode)
        dps[mode].attach(model)
    dp = dps[mode]
    dp.sync_every = max(1, int(params.get("sync_every", 1)))
    dp.reset()
    return dp


def _run_trial(model, tok, data, data_state, snap, probe, targs, params, device, dtype, own,
               base_threads, scale, attn, st, dp=None):
    reset(model, data, snap, data_state, targs.seed)
    mb, accum, gb = st["micro_batch"], st["grad_accum"], st["global_batch"]
    world = dp.world if dp is not None else 1
    if accum % world:
        raise ValueError(f"grad_accum {accum} does not split over {world} ranks")
    accum_local = accum // world
    mst, backend = apply_params(model, params, targs, base_threads, world)
    sub_k, pack_len = data_params(params, targs)
    profile = is_on(params.get("profile", 0))
    # flex compiles one graph per shape: pad T to the BlockMask block so shapes repeat.
    pad_mult = ab.FLEX_BLOCK if attn == "flex" else 8
    extra = {"attn_backend": attn, "dtype": str(dtype).replace("torch.", ""), "mb_used": mb,
             "grad_accum": accum, "global_batch": gb, "fixed_mb": st["fixed_mb"]}
    if st["fixed_mb"] and "bs" in params:
        extra["bs_ignored"] = params["bs"]
    if dp is not None:
        extra.update(world=world, accum_per_rank=accum_local, sync_mode=dp.mode,
                     sync_every=dp.sync_every if dp.local_sgd else None)
    if st["diag_mb"]:
        extra["mb_diag"] = mb_diag(model, tok, data, mst, st["diag_mb"], device, backend, pad_mult)
        reset(model, data, snap, data_state, targs.seed)   # no learning or data leaks into the trial
    extra["torch_compile"] = tc.current(model)
    if tc.current(model) != "off":
        # The records of the warm-up and timed steps (stream peeked, then rewound).
        n_steps = st["warmup_steps"] + (st["timed_steps"] or 1)
        recs = subsample_recs(data.take(gb * n_steps), sub_k, targs.seed)
        extra.update(compile_prepass(
            model, tok, probe, targs, device, backend,
            (prism_collate(tok, recs[i:i + mb], mst, pad_to_multiple_of=pad_mult, pack_len=pack_len)
             for i in range(0, len(recs), mb)), dp))
        reset(model, data, snap, data_state, targs.seed)

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

    def step_batches():
        """This rank's micro-batches of one optimizer step (lazy: collate interleaves with
        compute). The whole global batch is taken, so `ids` are the global records."""
        pos = data.get_state()[0] if hasattr(data, "layout") else None
        recs = data.take(gb)
        ids.extend(r.id for r in recs)
        recs = subsample_recs(recs, sub_k, targs.seed)
        mbs = [recs[i:i + mb] for i in range(0, len(recs), mb)]
        mine = range(len(mbs)) if dp is None else \
            dp.mine([jdist.mb_cost(m, mst, bool(pack_len)) for m in mbs] if world > 1 else [0] * len(mbs))
        for k in mine:
            with torch.profiler.record_function("oj::collate"):
                b = prism_collate(tok, mbs[k], mst, pad_to_multiple_of=pad_mult, pack_len=pack_len,
                                  rows_of=frozen_rows(data, None if pos is None else pos + k * mb,
                                                      mb, mst, sub_k, pack_len))
            counts["real"] += int(b.attention_mask.sum())          # CPU, no sync
            counts["pad"] += b.attention_mask.numel()
            counts["rows"] += b.input_ids.shape[0]
            counts["mbs"] += 1
            yield b

    def step():
        nonlocal skipped
        loss, sk = accum_step(model, opt, scaler, step_batches(), accum_local, device, backend, dp)
        skipped += sk
        losses.append(loss)

    def probe_now():
        if dp is not None:
            dp.average()                    # local_sgd: the probe sees the averaged weights
        return probe_loss(model, tok, probe, targs, device)

    # Warm-up: exactly warmup_steps APPLIED optimizer steps before the guard. A step the fp16
    # GradScaler skips (inf/NaN grads; a fresh 2^16 scale at the start of a reload group) is
    # redone on the same records: the skip changed no weights or Adam state, only the scale.
    # Otherwise the baseline's guard drop could be ~0 and the guard would pass everything.
    t_w = time.perf_counter()
    retries = 0
    for _ in range(st["warmup_steps"]):
        for _attempt in range(WARMUP_RETRIES + 1):
            pos, n_ids = data.get_state(), len(ids)
            _loss, sk = accum_step(model, opt, scaler, step_batches(), accum_local, device, backend, dp)
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
    loss_guard = probe_now()
    counts.update(real=0, pad=0, rows=0, mbs=0)
    if device == "cuda":
        torch.cuda.synchronize()
    jdist.barrier()
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
    jdist.barrier()                         # the slowest rank's time
    sec = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    train = torch.stack(losses).float().cpu().tolist()
    loss_end = probe_now()
    if dp is not None and world > 1:        # global numbers: all ranks' micro-batches
        train = dp.mean(train)
        counts.update(zip(counts, (int(round(v)) for v in dp.sum(list(counts.values())))))
        extra["syncs"] = dp.syncs
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


def _run_bench_lengths(model, tok, data, snap, probe, targs, params, device, dtype, own,
                       base_threads, scale, attn, st, dp=None):
    """One trial on the fixed multi-length bench set (see the module docstring)."""
    if not st["fixed_mb"]:
        raise ValueError("--bench-lengths needs fixed micro_batch / global_batch settings")
    reset(model, data, snap, None, targs.seed)
    mb, accum, gb = st["micro_batch"], st["grad_accum"], st["global_batch"]
    world = dp.world if dp is not None else 1
    if accum % world:
        raise ValueError(f"grad_accum {accum} does not split over {world} ranks")
    accum_local = accum // world
    _mst, backend = apply_params(model, params, targs, base_threads, world)
    mst = data.max_state_tokens                 # the set is bucketed there (never cut)
    sub_k, _pack = data_params(params, targs)   # packing / pack_len: training only, not here
    pad_mult = ab.FLEX_BLOCK if attn == "flex" else 8
    steps_per = st["timed_steps"] or 1
    extra = {"attn_backend": attn, "dtype": str(dtype).replace("torch.", ""), "mb_used": mb,
             "grad_accum": accum, "global_batch": gb, "fixed_mb": True, "bench_set_sha": data.sha,
             "max_state_tokens_used": mst, "pack_len_used": 0, "timed_steps": steps_per,
             "bench_weights": data.weights}
    if "bs" in params:
        extra["bs_ignored"] = params["bs"]
    if st["diag_mb"]:
        extra["mb_diag"] = "skipped (bench set)"
    if dp is not None:
        extra.update(world=world, accum_per_rank=accum_local, sync_mode=dp.mode,
                     sync_every=dp.sync_every if dp.local_sgd else None)
    order = sorted(data.buckets, reverse=True)
    warm = [data.buckets[order[i % len(order)]][:mb] for i in range(accum)]
    extra["torch_compile"] = tc.current(model)
    if tc.current(model) != "off":
        # Every micro-batch of the warm-up and of each bucket's timed steps (same cut as below).
        mbs = list(warm)
        for k in sorted(data.buckets):
            recs = data.buckets[k]
            for j in range(steps_per):
                step_recs = [recs[(j * gb + i) % len(recs)] for i in range(gb)]
                mbs += [step_recs[i:i + mb] for i in range(0, gb, mb)]
        extra.update(compile_prepass(
            model, tok, probe, targs, device, backend,
            (prism_collate(tok, subsample_recs(m, sub_k, targs.seed), mst, pad_to_multiple_of=pad_mult)
             for m in mbs), dp))
        reset(model, data, snap, None, targs.seed)
    free(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    opt = tp.make_optimizer(model, targs)
    scaler = torch.amp.GradScaler("cuda", init_scale=scale["value"],
                                  enabled=(device == "cuda" and dtype == torch.float16))
    scale_start = scale["value"]
    loss_start = probe_loss(model, tok, probe, targs, device)
    counts, skipped, ids, losses = {}, 0, [], []

    def batches(mbs):
        """This rank's share of the micro-batches of one optimizer step (lazy)."""
        mbs = [subsample_recs(m, sub_k, targs.seed) for m in mbs]
        mine = range(len(mbs)) if dp is None else \
            dp.mine([jdist.mb_cost(m, mst, False) for m in mbs] if world > 1 else [0] * len(mbs))
        for k in mine:
            with torch.profiler.record_function("oj::collate"):
                b = prism_collate(tok, mbs[k], mst, pad_to_multiple_of=pad_mult)
            counts["real"] += int(b.attention_mask.sum())
            counts["pad"] += b.attention_mask.numel()
            yield b

    def one_step(mbs):
        nonlocal skipped
        ids.extend(r.id for m in mbs for r in m)
        loss, sk = accum_step(model, opt, scaler, batches(mbs), accum_local, device, backend, dp)
        skipped += sk
        losses.append(loss)
        return sk

    def probe_now():
        if dp is not None:
            dp.average()
        return probe_loss(model, tok, probe, targs, device)

    # Warm-up: one micro-batch per bucket, longest first, as optimizer step(s) of accum micro-
    # batches; a step the fp16 scaler skips is redone (the guard follows warmup_steps applied steps).
    counts.update(real=0, pad=0)
    t_w, retries = time.perf_counter(), 0
    for _ in range(st["warmup_steps"]):
        for _attempt in range(WARMUP_RETRIES + 1):
            n_ids = len(ids)
            if not one_step(warm):
                break
            del ids[n_ids:]                 # the redone step's records count once
            retries += 1
        else:
            raise RuntimeError(f"fp16 GradScaler skipped the warm-up step {WARMUP_RETRIES + 1} times")
    if device == "cuda":
        torch.cuda.synchronize()
    extra.update(warmup_s=time.perf_counter() - t_w, warmup_retries=retries,
                 warmup_buckets=[order[i % len(order)] for i in range(accum)], warmup_ids_sha=ids_sha(ids))
    losses.clear()
    loss_guard = probe_now()
    graphs0 = tc.stats()
    per, peak_all = {}, None
    for k in sorted(data.buckets):
        recs = data.buckets[k]
        counts.update(real=0, pad=0)
        free(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        jdist.barrier()
        t0, sk0 = time.perf_counter(), skipped
        for j in range(steps_per):          # one global batch per step (the bucket wraps)
            step_recs = [recs[(j * gb + i) % len(recs)] for i in range(gb)]
            one_step([step_recs[i:i + mb] for i in range(0, gb, mb)])
        if device == "cuda":
            torch.cuda.synchronize()
        jdist.barrier()
        sec = time.perf_counter() - t0
        c = dict(counts)
        if dp is not None and world > 1:
            c = dict(zip(c, (int(round(v)) for v in dp.sum(list(c.values())))))
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
        peak_all = peak if peak_all is None else max(peak_all, peak)
        n = steps_per * gb
        per[k] = {"seconds": sec, "samples": n, "real_tokens": c["real"], "padded_tokens": c["pad"],
                  "samples_per_sec": n / sec, "tokens_per_sec": c["real"] / sec,
                  "padded_tokens_per_sec": c["pad"] / sec, "pad_frac": 1 - c["real"] / max(c["pad"], 1),
                  "peak_gb": peak, "skipped_steps": skipped - sk0}
    mix = mix_objective(per, data.weights)
    if graphs0 is not None:      # > 0: graphs were built inside the timed buckets
        extra["graphs_in_timed_loop"] = tc.graphs_since(graphs0)
    train = torch.stack(losses).float().cpu().tolist()
    loss_end = probe_now()
    if dp is not None and world > 1:
        train = dp.mean(train)
        extra["syncs"] = dp.syncs
    if scaler.is_enabled():
        scale["value"] = scaler.get_scale()
    if is_on(params.get("profile", 0)):      # after timing: profiles the <=512 bucket's step
        ab.set_attn_backend(model, attn, chunk=own.attn_chunk, label=True)
        k = 512 if 512 in data.buckets else sorted(data.buckets)[0]
        prof = [data.buckets[k][i:i + mb] for i in range(0, gb, mb)]
        extra["profile"] = run_profile(lambda: one_step(prof), device, own.profile_steps)
        extra["is_profile"] = True
    nan = not all(math.isfinite(x) for x in train + [loss_start, loss_guard, loss_end])
    steps = steps_per * len(per)
    sec = sum(r["seconds"] for r in per.values())
    return {**extra, "status": "nan" if nan else "ok", "buckets": {str(k): v for k, v in per.items()},
            **mix, "steps": steps, "samples": steps * gb, "seconds": sec, "s_per_step": sec / steps,
            "data_ids_sha": ids_sha(ids), "records_used": len(ids), "peak_mem_gb": peak_all,
            "loss_start": loss_start, "loss_guard": loss_guard, "guard_steps": st["warmup_steps"],
            "loss_end": loss_end, "scale_start": scale_start, "scale_end": scale["value"],
            "skipped_steps": skipped, "train_loss_first": train[0], "train_loss_last": train[-1]}


# ---------------------------------------------------------------- data

def load_data(own, targs):
    """(stream, probe, meta): the frozen trial stream (--trial-stream), the frozen stage
    streams (--stage-streams, a StageStreams), or a seeded Mixture."""
    if getattr(own, "bench_lengths", ""):
        return load_bench_lengths(own.bench_lengths, targs)
    if getattr(own, "stage_streams", ""):
        return load_stage_streams(own.stage_streams, targs)
    if own.trial_stream:
        frozen = json.loads(Path(own.trial_stream).read_text())
        if td.sha_of({"stream": frozen["stream"], "probe": frozen["probe"]}) != frozen["sha"]:
            raise SystemExit(f"{own.trial_stream}: content does not match its sha")
        recs = td.load_refs(targs.train, frozen["stream"])
        probe = td.load_refs(targs.train, frozen["probe"])
        stems = [r[0] for r in frozen["stream"]]
        stream = td.FixedStream(recs)
        # Frozen packed layout (tools/freeze_packed.py) next to the stream; absent = pack on the fly.
        lp = Path(own.trial_stream).with_name("packed_layout.json")
        stream.layout = td.load_layout(lp, frozen["sha"], len(recs))
        meta = {"trial_stream_sha": frozen["sha"], "trial_stream_len": len(recs), "stems": stems}
        if stream.layout:
            meta["packed_layout_sha"] = stream.layout["sha"]
        return stream, probe, meta
    paths = sorted(glob.glob(targs.train))
    if not paths:
        raise SystemExit(f"no train files match {targs.train}")
    mixture = Mixture(paths, alpha=targs.alpha, seed=targs.seed)
    probe = mixture.take(own.probe_n)            # fixed probe set, same for every trial
    return mixture, probe, {}


class StageStreams(dict):
    """--stage-streams: stage (int) -> FixedStream of that stage's frozen records."""
    max_state_tokens = None

    def get_state(self):
        return {k: v.get_state() for k, v in self.items()}


def load_stage_streams(path, targs):
    d = json.loads(Path(path).read_text())
    body = {k: d[k] for k in ("stages", "max_state_tokens", "streams", "probe")}
    if d.get("kind") != "stage_streams" or td.sha_of(body) != d["sha"]:
        raise SystemExit(f"{path}: not a stage_streams file, or content does not match its sha")
    data = StageStreams({int(k): td.FixedStream(td.load_refs(targs.train, v)) for k, v in d["streams"].items()})
    data.max_state_tokens, data.stage_sha = d["max_state_tokens"], d["stage_sha"]
    meta = {"trial_stream_sha": d["sha"], "stage_streams": {k: len(v.records) for k, v in data.items()},
            "stage_stream_sha": d["stage_sha"], "stage_streams_max_state_tokens": d["max_state_tokens"]}
    return data, td.load_refs(targs.train, d["probe"]), meta


class BenchLengths:
    """--bench-lengths: bucket (int) -> its frozen records, the bucket weights, the sha."""

    def __init__(self, buckets: dict, weights: dict, max_state_tokens: int, sha: str):
        self.buckets, self.weights, self.max_state_tokens, self.sha = buckets, weights, max_state_tokens, sha

    def get_state(self):              # fixed records, no stream position
        return None

    def set_state(self, state) -> None:
        pass


def load_bench_lengths(path, targs):
    d = json.loads(Path(path).read_text())
    body = {k: d[k] for k in ("stages", "max_state_tokens", "weights", "buckets", "probe")}
    if d.get("kind") != BENCH_KIND or td.sha_of(body) != d["sha"]:
        raise SystemExit(f"{path}: not a bench_lengths file, or content does not match its sha")
    data = BenchLengths({int(k): td.load_refs(targs.train, v) for k, v in d["buckets"].items()},
                        d["weights"], d["max_state_tokens"], d["sha"])
    meta = {"bench_set_sha": d["sha"], "bench_weights": d["weights"],
            "bench_buckets": {k: len(v) for k, v in d["buckets"].items()},
            "bench_max_state_tokens": d["max_state_tokens"]}
    return data, td.load_refs(targs.train, d["probe"]), meta


def trial_stream_for(data, data_state, params, targs):
    """(stream, start state, extra result fields) of one trial: the stage stream of its
    length_stage with --stage-streams (0 = the mixed curriculum-off reference stream, with
    the trial's packing / pack_len), else the one stream."""
    stage = int(params.get("length_stage", 0))
    if isinstance(data, BenchLengths):
        if stage:
            raise ValueError("length_stage needs --stage-streams, not --bench-lengths")
        return data, None, {}
    if not isinstance(data, StageStreams):
        if stage:
            raise ValueError("length_stage needs --stage-streams")
        return data, data_state, {}
    if stage not in data:
        raise ValueError(f"length_stage {stage} has no frozen stream (have {sorted(data)})")
    mst = int(params.get("max_state_tokens", targs.max_state_tokens))
    if mst != data.max_state_tokens:
        raise ValueError(f"stage streams were bucketed at max_state_tokens {data.max_state_tokens}, "
                         f"trial uses {mst}")
    return data[stage], data_state[stage], {
        "length_stage": stage, "stage_stream_sha": data.stage_sha[str(stage)],
        "pack_len_used": data_params(params, targs)[1]}


def stream_tokens(tok, stream, stems) -> dict:
    """p50 / p95 state tokens per source file of the stream (real tokenizer, so it can be
    compared with data/STATS.md "State token length")."""
    per: dict[str, list] = {}
    for r, s in zip(stream.records, stems):
        per.setdefault(s, []).append(len(tok(r.state, add_special_tokens=False)["input_ids"]))
    return {s: {"n": len(v), "p50": td.quantile(v, 0.5), "p95": td.quantile(v, 0.95)}
            for s, v in sorted(per.items())}


# ---------------------------------------------------------------- quality checkpoint

def validate_grouped(targs) -> bool:
    """--validate batches like train_prism's defaults: length-grouped micro-batches unless
    --length-grouping off (train_prism.parse resolves the default: on, off with the curriculum)."""
    return getattr(targs, "length_grouping", "off") == "on"


def validate_n_refs(n: int, mb: int, targs) -> int:
    """Training refs to freeze: n, rounded up to whole length-grouping mega-chunks (mb x
    group_chunk records) so no refill wraps the frozen stream and repeats records."""
    if not validate_grouped(targs):
        return n
    c = mb * targs.group_chunk
    return -(-n // c) * c


def validate_stream(tok, records, targs, mb: int, mst: int):
    """The --validate training stream over the frozen records: train_prism.train_stream's
    LengthGroupedStream (micro-batch = the plan's micro_batch, --group-chunk, --seed) when
    length grouping is on, else the records in their frozen order. Packing (pack_len) is
    applied per micro-batch by data_params, as before (train_prism default: on, 1024)."""
    stream = td.FixedStream(records)
    if not validate_grouped(targs):
        return stream
    from openjev.packing import LengthGroupedStream, request_lengths
    return LengthGroupedStream(stream, lambda recs: request_lengths(tok, recs, mst), mb,
                               targs.group_chunk, targs.seed)


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
    grouped = validate_grouped(targs)
    refs = td.freeze_stream(paths, validate_n_refs(n_steps * gb, mb, targs), 0, alpha=targs.alpha,
                            seed=targs.seed, check=False)
    records = td.load_refs(paths, refs["stream"])
    log_rss("after_data")
    meta.update(validate=True, plan=st, train_refs_sha=refs["sha"], val_set_sha=vs["sha"], val_n=len(val), steps=n_steps,
                global_batch=gb, micro_batch=mb, length_grouping="on" if grouped else "off",
                group_chunk=targs.group_chunk if grouped else None)
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
            # Rebuilt per config (lengths at its max_state_tokens): every config trains on the
            # same micro-batches in the same order.
            mst_c = int(params.get("max_state_tokens", targs.max_state_tokens))
            mixture = validate_stream(tok, records, targs, mb, mst_c)
            r.update(length_grouping=meta["length_grouping"], pack_len_used=data_params(params, targs)[1])
            r.update(_validate_one(model, tok, mixture, mixture.get_state(), val, targs, params, device,
                                   dtype, own, base_threads, gb, mb, n_steps))
        except Exception as e:
            st_ = {"fixed_mb": True, "micro_batch": mb}
            failed = (trial_status(e, st_, params.get("attn_backend"), params.get("torch_compile")),
                      f"{type(e).__name__}: {e}"[:400],
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
        apply_compile(model, params)
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
            batches = (train_collate(tok, mixture.take(mb), mst, pad_mult, sub_k, pack_len, targs.seed)
                       for _ in range(gb // mb))
            loss, sk = accum_step(model, opt, scaler, batches, gb // mb, device, backend)
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
        tc.unset_torch_compile(model)
    k = max(1, min(10, len(losses) // 2))
    nan = not (all(math.isfinite(x) for x in losses) and math.isfinite(m["nll"]))
    return {"status": "nan" if nan else "ok", "steps": n_steps, "seconds": sec,
            "skipped_steps": skipped, "train_loss_first": sum(losses[:k]) / k,
            "train_loss_last": sum(losses[-k:]) / k, "val": m}


# ---------------------------------------------------------------- main

def main(argv=None):
    own, targs = parse(argv)
    device = tp.pick_device(targs)
    if jdist.env_world() <= 1 and not own.validate and own.gpu_index is None:
        all_trials = json.loads(Path(own.trials).read_text())
        if any(is_ddp(t) for t in all_trials):
            raise SystemExit(run_ddp_split(own, sys.argv[1:] if argv is None else argv, all_trials, device))
    if own.dual_gpu and not own.validate and own.gpu_index is None:
        raise SystemExit(run_dual(own, sys.argv[1:] if argv is None else argv))
    _rank, world = jdist.init(device)
    if world > 1 and own.validate:
        raise SystemExit("--validate runs in one process (no torchrun)")
    n_gpu = torch.cuda.device_count() if device == "cuda" else 0
    meta = {"device": device, "n_gpu": n_gpu, "world": world,
            "gpu_name": torch.cuda.get_device_name(0) if n_gpu else None,
            "torch": torch.__version__, "model": targs.model, "trial_seconds": own.trial_seconds,
            "warmup_steps": own.warmup_steps, "probe_n": own.probe_n, "trials_file": own.trials}
    out = Path(own.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []

    def emit(r):
        if _rank != 0:                    # torchrun: rank 0 reports for all ranks
            return
        r = {k: clean(v) for k, v in r.items()}
        if own.gpu_index is not None:
            r["gpu_index"] = own.gpu_index
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
    want_set = {(t.get("bench") or {}).get("bench_set_sha") for t in trials} - {None}
    if want_set and want_set != {dmeta.get("bench_set_sha")}:      # fail closed: no fallback data
        raise SystemExit(f"trials.json was proposed for bench set {sorted(want_set)}, but "
                         f"--bench-lengths is {dmeta.get('bench_set_sha') or 'not given'}")

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
            bargs = build_args(targs, dict(key))
            # torchrun: one rank at a time (each load peaks at ~14.6 GB of host RAM).
            model, tok, dtype = jdist.staggered(lambda: tp.build(bargs, device))
            model.train()
            dps = {}                          # DataParallel per sync mode (one DDP wrap)
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
            if world > 1:
                raise                         # the other ranks would wait in a collective
            continue
        load_s = time.perf_counter() - t_load
        if own.ready_file:
            Path(own.ready_file).touch()
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
                tdata, tstate, sx = trial_stream_for(data, data_state, t["params"], targs)
                r.update(sx)
                dp = trial_dp(dps, model, t["params"], device)
                r.update(run_trial(model, tok, tdata, tstate, snap, probe, targs, t["params"],
                                   device, dtype, own, base_threads, scale, st, dp=dp))
            except Exception as e:
                failed = (trial_status(e, st or {}, t["params"].get("attn_backend"),
                                       t["params"].get("torch_compile")),
                          f"{type(e).__name__}: {e}"[:400], traceback.format_exc()[-1500:])
            if failed:
                # Free outside the except block: the traceback holds the tensors.
                r.update(status=failed[0], error=failed[1])
                if failed[0] == "error" or t["params"].get("torch_compile", "off") != "off":
                    print(failed[2], flush=True)   # compile: Inductor's real cause is past 400 chars
                model.zero_grad(set_to_none=True)
                free(device)
            if device == "cuda":
                r.setdefault("peak_mem_gb", torch.cuda.max_memory_allocated() / 2**30)
            emit(r)
            if failed and world > 1:
                # One rank's OOM leaves the others in an all-reduce: end the launch (torchrun
                # stops every rank); the parent merges what rank 0 wrote.
                raise SystemExit(f"rank {_rank}: trial {t['trial_number']} {failed[0]}: {failed[1]}")
        del model, snap
        model = None
    torch.set_num_threads(base_threads)
    if own.ready_file:
        Path(own.ready_file).touch()
    if _rank == 0:
        print(json.dumps({"bench_done": len(results), "out": str(out)}), flush=True)
    jdist.cleanup()


if __name__ == "__main__":
    main()
