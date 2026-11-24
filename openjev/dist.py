"""Data parallel training on several GPUs (OJ-062): one process per GPU under torchrun.

  python -m torch.distributed.run --standalone --nproc_per_node 2 train_prism.py ...

Without torchrun (WORLD_SIZE unset or 1) every helper is a no-op and the single-process
path is unchanged.

Sharding. An optimizer step is `grad_accum` micro-batches of `bs` records over ALL ranks
(--grad-accum is global, so the global batch and the lr schedule do not change with the
number of GPUs). Every rank reads the same seeded stream, takes the whole step
(bs x grad_accum records), cuts it into the same micro-batches a single process would
build, and keeps a disjoint share of WHOLE micro-batches (`assign`: token-balanced, equal
counts, deterministic, so no communication is needed). Each micro-batch keeps its exact
records, so every per-micro-batch statistic (the SFT mean over questions, the exact-PG
soft ECE, GRPO's per-question group baseline) is the one a single process computes, and

  DDP mean over ranks of (1 / accum_local) * sum of my micro-batch grads
    = (1 / grad_accum) * sum over all micro-batch grads

is the single-process gradient. No reward / advantage statistic crosses micro-batches,
so none needs an all-reduce.

Sync modes:
  ddp        DistributedDataParallel over the trainables only (LoRA, flag head, template
             delta: a few MB); no_sync on every backward but the last of the step, so one
             gradient all-reduce per optimizer step.
  local_sgd  no gradient all-reduce: every rank steps its own optimizer on its share of
             the step, and the trainable weights are averaged every `sync_every` steps
             (Lin et al. 2020, "Don't use large mini-batches, use local SGD", ICLR;
             Stich 2019, arXiv:1805.09767). Optimizer state (AdamW moments, the GradScaler
             scale) stays per rank. With a linear optimizer (SGD) and K=1 it equals ddp;
             with AdamW it does not (avg of per-rank Adam steps != Adam on the avg grad).

Frozen parameters and all buffers are hidden from DDP: it then neither broadcasts ~5B
frozen weights at wrap time nor rejects the module when the frozen PLE table stays on the
CPU (ple_cpu=1) while the rest is on the GPU. The ranks build the model under the same
seed, so frozen weights and buffers are equal anyway; the trainables are broadcast from
rank 0 when DDP wraps the model.
"""

from __future__ import annotations

import contextlib
import os
from datetime import timedelta

import torch
import torch.distributed as dist

SYNC_MODES = ("ddp", "local_sgd")


def env_world() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def init(device: str) -> tuple[int, int]:
    """(rank, world). Starts the process group when launched by torchrun with >1 process:
    NCCL on CUDA (after set_device(LOCAL_RANK), so "cuda" means this rank's GPU), gloo on
    CPU. OJ_DIST_TIMEOUT_MIN (default 60) bounds a collective: staggered model loads wait
    in a barrier, so it must exceed one Gemma load (~5-7 min on Kaggle)."""
    world = env_world()
    if world <= 1:
        return 0, 1
    if not dist.is_initialized():
        if device == "cuda":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        timeout = timedelta(minutes=float(os.environ.get("OJ_DIST_TIMEOUT_MIN", "60")))
        dist.init_process_group("nccl" if device == "cuda" else "gloo", timeout=timeout)
    return dist.get_rank(), dist.get_world_size()


def active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    return dist.get_rank() if active() else 0


def world() -> int:
    return dist.get_world_size() if active() else 1


def barrier() -> None:
    if active():
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def staggered(fn):
    """fn() one rank at a time (rank 0 first). A Gemma 4 load peaks at ~14.6 GB of host RAM
    per process; two at once would pass Kaggle's ~30 GB and get SIGKILLed."""
    out = None
    for r in range(world()):
        if r == rank():
            out = fn()
        barrier()
    return out


# ---------------------------------------------------------------- sharding

def approx_tokens(rec, max_state_tokens: int) -> int:
    """Cheap, deterministic token estimate of one record (~4 chars per token, the state
    capped at max_state_tokens): balancing must not tokenize the other ranks' records."""
    chars = min(len(rec.state), 4 * max_state_tokens)
    for q in rec.schema.questions:
        chars += len(q.name) + len(q.instructions) + sum(len(l) + 8 for l in q.labels)
    return chars // 4 + 4


def mb_cost(recs, max_state_tokens: int, packed: bool) -> int:
    """Estimated compute of one micro-batch: its tokens when packed, else rows x longest."""
    t = [approx_tokens(r, max_state_tokens) for r in recs]
    return sum(t) if packed else len(t) * max(t, default=0)


def assign(costs: list[int], n_ranks: int, r: int) -> list[int]:
    """Indices (ascending) of the micro-batches rank r runs: equal counts, costs balanced
    greedily (largest first onto the least-loaded rank that has room; ties -> lower index,
    lower rank). Deterministic: every rank computes the same partition."""
    n = len(costs)
    if n_ranks < 1 or n % n_ranks:
        raise ValueError(f"{n} micro-batches do not split evenly over {n_ranks} ranks")
    per = n // n_ranks
    bins: list[list[int]] = [[] for _ in range(n_ranks)]
    load = [0] * n_ranks
    for i in sorted(range(n), key=lambda i: (-costs[i], i)):
        k = min((k for k in range(n_ranks) if len(bins[k]) < per), key=lambda k: (load[k], k))
        bins[k].append(i)
        load[k] += costs[i]
    return sorted(bins[r])


# ---------------------------------------------------------------- the training context

def ignore_frozen(model: torch.nn.Module) -> list[str]:
    """Hide every frozen parameter and every buffer from DDP (see the module docstring)."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    names = [n for n, p in model.named_parameters() if not p.requires_grad]
    names += [n for n, _ in model.named_buffers()]
    DDP._set_params_and_buffers_to_ignore_for_model(model, names)
    return names


class DataParallel:
    """What a training loop needs from data parallelism. world 1: a no-op context
    (net = the model, every micro-batch is mine, no collectives)."""

    def __init__(self, device: str, mode: str = "ddp", sync_every: int = 1):
        if mode not in SYNC_MODES:
            raise ValueError(f"sync mode {mode!r} not in {SYNC_MODES}")
        self.device, self.mode, self.sync_every = device, mode, max(1, int(sync_every))
        self.rank, self.world = rank(), world()
        self.model = self.net = None
        self.steps = 0          # optimizer steps since the last reset (local_sgd cadence)
        self.syncs = 0          # local_sgd weight averages done

    @property
    def ddp(self) -> bool:
        return self.world > 1 and self.mode == "ddp"

    @property
    def local_sgd(self) -> bool:
        return self.world > 1 and self.mode == "local_sgd"

    def attach(self, model, bucket_cap_mb: float = 25):
        """Wrap `model` for training. ddp: DistributedDataParallel over the trainables
        (find_unused_parameters=False: every trainable gets a gradient in every backward;
        Gemma 4's KV-shared layers have no k/v projections, so no LoRA sits idle).
        Collective: every rank must call it at the same point."""
        self.model = self.net = model
        if self.ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            ignore_frozen(model)
            ids = [torch.cuda.current_device()] if self.device == "cuda" else None
            self.net = DDP(model, device_ids=ids, broadcast_buffers=False,
                           find_unused_parameters=False, bucket_cap_mb=bucket_cap_mb)
        return self.net

    def reset(self) -> None:
        self.steps = self.syncs = 0

    def mine(self, costs: list[int]) -> list[int]:
        return assign(costs, self.world, self.rank) if self.world > 1 else list(range(len(costs)))

    def sync(self, last: bool):
        """Around the forward AND backward of a micro-step: ddp skips the gradient
        all-reduce (no_sync) on every backward but the last of the optimizer step."""
        if self.ddp and not last:
            return self.net.no_sync()
        return contextlib.nullcontext()

    def after_step(self) -> bool:
        """Call after every optimizer step. local_sgd: average the weights every
        sync_every steps. Returns True if it averaged."""
        self.steps += 1
        if self.local_sgd and self.steps % self.sync_every == 0:
            self.average()
            return True
        return False

    @torch.no_grad()
    def average(self) -> None:
        """local_sgd: replace every trainable tensor by its mean over the ranks (one
        coalesced all-reduce). Call on every rank, e.g. before an eval or a save."""
        if not self.local_sgd:
            return
        ps = [p for p in self.model.parameters() if p.requires_grad]
        flat = torch.cat([p.detach().reshape(-1).float() for p in ps])
        dist.all_reduce(flat)
        flat /= self.world
        off = 0
        for p in ps:
            n = p.numel()
            p.copy_(flat[off:off + n].view_as(p))
            off += n
        self.syncs += 1

    def mean(self, values: list[float]) -> list[float]:
        """Mean of each value over the ranks (NaN/inf propagate, so a finite check on the
        result agrees on every rank). Collective."""
        if self.world <= 1 or not values:
            return [float(v) for v in values]
        t = torch.tensor([float(v) for v in values], dtype=torch.float64 if self.device == "cpu"
                         else torch.float32, device=self.device)
        dist.all_reduce(t)
        return (t / self.world).tolist()

    def sum(self, values: list[float]) -> list[float]:
        return [v * self.world for v in self.mean(values)]

    def any(self, flag: bool) -> bool:
        """True if flag is True on any rank. Collective."""
        return bool(self.mean([float(bool(flag))])[0] > 0)
