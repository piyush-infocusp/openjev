"""torch.compile of the decoder layers only (OJ-079; Optuna dim `torch_compile`).

  off              nothing is touched: no torch._dynamo import, no config write, the eager path.
  default          Inductor (Triton kernels on CUDA, C++ on CPU), per decoder layer.
  reduce-overhead  the same plus CUDA graphs (cudagraph trees; one recording per new shape).

Each decoder layer gets `layer.compile(dynamic=True)` (nn.Module.compile: the module stays in
its ModuleList, so parameter names, the PEFT adapter, state dicts and the bench snapshot do not
change). Everything outside the layers stays eager: the PLE lookup (on CPU with ple_cpu=1), the
embedding + template delta, the Prism masks (built once per forward, passed in as tensors), the
final norm, the flag head and the loss. HF gradient checkpointing wraps the compiled call
(GradientCheckpointingLayer.__call__ -> checkpoint -> compiled forward), so the recompute in
backward runs the same compiled graph. GradScaler, DDP hooks and LoRA see ordinary grads.

Shapes: length-grouped / packed micro-batches have many (rows, T) shapes, so the layers are
compiled with dynamic=True (one symbolic graph per layer variant: head dim, KV-shared or not,
train / no-grad; batch 1 is specialised once). All layers share one code object, so
RECOMPILE_LIMIT caps every variant together; with fail_on_limit=True (bench) hitting it raises
instead of silently running eager, so a trial never reports eager speed as compiled.

`unset_torch_compile` drops the compiled calls and resets dynamo, so a later trial in the same
process starts from the eager model (bench_trials: no reload between trials).
"""

from __future__ import annotations

import os
import time

MODES = ("off", "default", "reduce-overhead")
RECOMPILE_LIMIT = 64
# Error text that means "the compiler cannot build / run this here" (bench: unsupported).
COMPILE_ERRORS = ("backendcompilerfailed", "inductor", "triton", "no valid triton configs",
                  "out of resource", "shared memory", "cudagraph", "cuda graph", "recompile_limit",
                  "recompile limit", "cache_size_limit", "torch._dynamo", "dynamo", "torchruntimeerror",
                  "unsupported", "codegen", "c++ compile", "cppcompileerror", "lowering")


def decoder_layers(model) -> list:
    text = model._text_backbone()
    return list(text.layers[: text.config.num_hidden_layers])


def current(model) -> str:
    return getattr(model, "_torch_compile", "off")


def set_torch_compile(model, mode: str, fail_on_limit: bool = True, backend: str | None = None) -> str:
    """Compile every decoder layer of a PrismOpenJev in place (mode in MODES). off = unset.
    backend: a dynamo backend name instead of Inductor (tests: "aot_eager"; default: the env
    var OJ_COMPILE_BACKEND, tests only); mode is then ignored. Returns the mode now in effect."""
    if mode not in MODES:
        raise ValueError(f"torch_compile {mode!r} not in {MODES}")
    if mode == "off":
        unset_torch_compile(model)
        return "off"
    if current(model) != "off":
        unset_torch_compile(model)
    import torch._dynamo as dynamo
    cfg = dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):     # name differs across torch versions
        if hasattr(cfg, name):
            setattr(cfg, name, max(getattr(cfg, name), RECOMPILE_LIMIT))
            break
    if hasattr(cfg, "fail_on_recompile_limit_hit"):
        cfg.fail_on_recompile_limit_hit = bool(fail_on_limit)
    kw = {"dynamic": True}
    backend = backend or os.environ.get("OJ_COMPILE_BACKEND") or None
    if backend:
        kw["backend"] = backend
    elif mode != "default":
        kw["mode"] = mode
    for layer in decoder_layers(model):
        layer.compile(**kw)
    model._torch_compile = mode
    return mode


def unset_torch_compile(model) -> None:
    """Back to the eager layers. Touches dynamo only if a compile was applied."""
    if current(model) == "off":
        return
    for layer in decoder_layers(model):
        layer._compiled_call_impl = None
    import torch._dynamo as dynamo
    dynamo.reset()
    model._torch_compile = "off"


def is_compile_error(e: BaseException) -> bool:
    """True for a compiler / CUDA-graph / recompile-limit failure (not an OOM)."""
    text = f"{type(e).__module__}.{type(e).__name__}: {e}".lower()
    if "out of memory" in text and "shared memory" not in text:
        return False
    return any(k in text for k in COMPILE_ERRORS)


def stats() -> dict | None:
    """Dynamo counters so far: {"graphs": unique graphs built} (None if dynamo never loaded).
    torch 2.14 keeps no recompile counter; bench derives recompiles from graph deltas."""
    import sys
    if "torch._dynamo" not in sys.modules:
        return None
    try:
        from torch._dynamo.utils import counters
        return {"graphs": int(counters["stats"]["unique_graphs"])}
    except Exception:
        return None


def graphs_since(before: dict | None) -> int | None:
    """Graphs built since a stats() snapshot (None if dynamo is not loaded)."""
    after = stats()
    if after is None:
        return None
    return after["graphs"] - ((before or {}).get("graphs") or 0)


class Timer:
    """Wall seconds of a block (the compile pre-pass)."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0
        return False
