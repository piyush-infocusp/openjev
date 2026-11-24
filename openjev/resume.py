"""Resumable training sessions (ported from the fork ~/openjev-xla, CUDA/CPU only).

A resume checkpoint is a directory  <save-dir>/resume/step_0000123/  with
  meta.json  step, stage, config hash, device kind, dtype, time (read without torch)
  state.pt   CPU tensors + Python objects:
             trainable   {param name: fp32 tensor}  (LoRA, flag head, template delta)
             optimizer   AdamW state_dict + the param names in order
             scheduler   LambdaLR state_dict
             scaler      GradScaler state_dict (CUDA fp16 only, else {})
             rng         python / numpy / torch CPU / CUDA generators
             mixture     the training stream's get_state() (Mixture, LengthGroupedStream
                         or CurriculumStream: the data position)
             best        best val nll so far
The directory is written as step_X.tmp and renamed, so a killed session never leaves a
half checkpoint under a real name. Under DDP rank 0 writes it and every rank loads it
(train_prism.main).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

STAGES = {"sft": 0, "rl": 1}

# Args that change the training trajectory (or the shape of the stream state). Everything
# else (paths, eval cadence, dtype, grad ckpt, PLE placement, time budgets, world size:
# --grad-accum is global) may change between sessions.
TRAJECTORY_KEYS = ("model", "alpha", "algo", "reward", "causal", "steps", "bs", "grad_accum", "lr",
                   "head_lr", "warmup", "lora_r", "lora_alpha", "group", "ppo_epochs", "clip",
                   "lam_kl", "lam_cal", "max_state_tokens", "seed", "packing", "pack_len",
                   "length_grouping", "group_chunk", "length_curriculum", "curriculum_stages",
                   "curriculum_pack_mult", "option_subsample_k", "sync_mode", "quant")


def stage_of(algo: str) -> str:
    return "sft" if algo == "sft" else "rl"


def config_of(args, train_paths) -> dict:
    cfg = {k: getattr(args, k, None) for k in TRAJECTORY_KEYS}
    cfg["train_files"] = sorted(Path(p).name for p in train_paths)
    cfg["init"] = Path(args.init).name if getattr(args, "init", None) else None
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def kind(device) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def rng_state(device) -> dict:
    np_state = np.random.get_state()
    st = {"python": random.getstate(),
          "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
          "torch_cpu": torch.get_rng_state(), "kind": kind(device), "device": None}
    if st["kind"] == "cuda":
        st["device"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return st


def set_rng_state(st: dict, device) -> list[str]:
    """Restore; returns notes about what was skipped (another device kind)."""
    notes = []
    random.setstate(st["python"])
    a, keys, *rest = st["numpy"]
    np.random.set_state((a, np.asarray(keys, dtype=np.uint32), *rest))
    torch.set_rng_state(st["torch_cpu"])
    k = kind(device)
    if st.get("device") is None:
        pass
    elif st["kind"] != k:
        notes.append(f"device RNG from {st['kind']} not restored on {k} (fresh device generator)")
    else:
        torch.cuda.set_rng_state_all(st["device"])
    return notes


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and not hasattr(obj, "_fields"):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def trainable_state(model) -> dict:
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state: dict) -> None:
    """Copy in place (DDP and the optimizer keep their parameter references)."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    missing = sorted(set(params) - set(state))
    extra = sorted(set(state) - set(params))
    if missing or extra:
        raise RuntimeError(f"resume: trainable params differ: missing {missing[:5]} extra {extra[:5]}")
    with torch.no_grad():
        for n, p in params.items():
            v = state[n]
            if v.shape != p.shape:
                raise RuntimeError(f"resume: shape of {n} {tuple(v.shape)} != {tuple(p.shape)}")
            p.copy_(v.to(device=p.device, dtype=p.dtype))


def opt_param_names(model, opt) -> list[list[str]]:
    by_id = {id(p): n for n, p in model.named_parameters()}
    return [[by_id[id(p)] for p in g["params"]] for g in opt.param_groups]


def adapt_optimizer_state(sd: dict, saved_names, model, opt) -> tuple[dict, list[str]]:
    """Map a saved AdamW state_dict onto the current optimizer by parameter name; cast
    floating state to the param dtype; drop state whose shape differs."""
    notes = []
    cur_names = opt_param_names(model, opt)
    params = [p for g in opt.param_groups for p in g["params"]]
    flat_cur = [n for g in cur_names for n in g]
    saved_index = {n: i for i, n in enumerate(n for g in saved_names for n in g)}
    state = {}
    for j, (n, p) in enumerate(zip(flat_cur, params)):
        i = saved_index.get(n)
        if i is None or i not in sd["state"]:
            notes.append(f"no optimizer state for {n} (fresh)")
            continue
        st, ok = {}, True
        for k, v in sd["state"][i].items():
            if torch.is_tensor(v) and k != "step":
                if v.shape != p.shape:
                    ok = False
                    break
                if v.dtype != p.dtype:
                    v = v.to(p.dtype)
            st[k] = v
        if ok:
            state[j] = st
        else:
            notes.append(f"optimizer state shape mismatch for {n} (fresh)")
    groups = []
    for g_saved, g_cur in zip(sd["param_groups"], opt.state_dict()["param_groups"]):
        g = dict(g_saved)
        g["params"] = g_cur["params"]
        groups.append(g)
    cast = sorted({str(v.dtype) for s in sd["state"].values() for k, v in s.items()
                   if torch.is_tensor(v) and k != "step"} - {str(params[0].dtype)}) if params else []
    if cast:
        notes.append(f"optimizer state cast from {cast} to {params[0].dtype}")
    return {"state": state, "param_groups": groups}, notes


def save(save_dir, *, step, stage, model, opt, sched, scaler, mixture, best, cfg, device,
         dtype, keep: int = 2) -> Path:
    root = Path(save_dir) / "resume"
    final = root / f"step_{step:07d}"
    tmp = root / f"step_{step:07d}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    state = {"trainable": trainable_state(model),
             "optimizer": opt.state_dict(), "opt_names": opt_param_names(model, opt),
             "scheduler": sched.state_dict(),
             "scaler": scaler.state_dict() if scaler.is_enabled() else {},
             "rng": rng_state(device), "mixture": mixture.get_state(), "best": best,
             "step": step, "stage": stage, "config": cfg}
    torch.save(to_cpu(state), tmp / "state.pt")
    meta = {"step": step, "stage": stage, "config_hash": config_hash(cfg), "backend": kind(device),
            "dtype": str(dtype), "time": time.time(), "torch": torch.__version__}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    (root / "LATEST").write_text(final.name)
    for old in sorted(root.glob("step_???????"))[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
    return final


def read_meta(path) -> dict | None:
    try:
        return json.loads((Path(path) / "meta.json").read_text())
    except Exception:
        return None


def find_latest(*dirs) -> Path | None:
    """The newest complete resume checkpoint (highest stage, then step) under any of `dirs`
    (each a save dir, a resume dir, or a checkpoint dir; globs ok)."""
    cands = []
    for d in dirs:
        for base in glob.glob(str(d)) or []:
            base = Path(base)
            for c in [base, *base.glob("step_???????"), *base.glob("resume/step_???????")]:
                m = read_meta(c)
                if m and (c / "state.pt").exists():
                    cands.append(((STAGES.get(m["stage"], -1), m["step"], m.get("time", 0)), str(c)))
    return Path(max(cands)[1]) if cands else None


def load(path, *, model, opt, sched, scaler, mixture, device, cfg, ignore_config=False) -> dict:
    """Restore everything into the live objects. Returns {"step", "best", "notes", "stage"}.
    Our own files hold Python objects (RNG, the stream), so weights_only=False: only load
    checkpoints this code wrote."""
    st = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=False)
    notes = []
    if st["config"] != cfg:
        diff = {k: (st["config"].get(k), cfg.get(k)) for k in set(st["config"]) | set(cfg)
                if st["config"].get(k) != cfg.get(k)}
        if not ignore_config:
            raise SystemExit(f"resume: config differs from {path}: {diff} "
                             "(--resume-ignore-config to continue anyway)")
        notes.append(f"config differs (ignored): {diff}")
    load_trainable(model, st["trainable"])
    osd, n = adapt_optimizer_state(st["optimizer"], st["opt_names"], model, opt)
    notes += n
    opt.load_state_dict(osd)
    sched.load_state_dict(st["scheduler"])
    if scaler.is_enabled() and st["scaler"]:
        scaler.load_state_dict(st["scaler"])
    elif scaler.is_enabled():
        notes.append("no GradScaler state in the checkpoint (fresh scaler)")
    elif st["scaler"]:
        notes.append("GradScaler state skipped (scaler disabled on this device)")
    notes += set_rng_state(st["rng"], device)
    mixture.set_state(st["mixture"])
    return {"step": int(st["step"]), "best": st["best"], "notes": notes, "stage": st["stage"]}
