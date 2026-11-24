"""Train an OpenJev decision model on Gemma 4 E2B with Prism-style attention.

Two stages, same script:

  1. SFT   (--algo sft):   cross-entropy over the flags of each question.
  2. RL    (--algo exact | grpo), started from the SFT adapter (--init):

     exact  Exact policy gradient. A decision has a small, known action set (the
            options), so the expected reward  E_p[r] = sum_i p_i r_i  is computed
            without sampling. Zero-variance gradient; this is the best choice
            when every option's reward is known.
     grpo   GRPO (Shao et al. 2024) with Dr. GRPO details (Liu et al. 2025):
            sample G options per question, group-mean baseline, no std
            normalisation, PPO clip over --ppo-epochs inner steps, k3 KL to the
            frozen reference. Use it when the reward needs a sampled action
            (a simulator, a cost table, a downstream outcome).

     Rewards (--reward):
       acc     1 for the correct option, else 0
       graded  score questions: 1 - |i - y| / (K - 1); others like acc
       brier   proper scoring rule on the whole distribution (exact only):
               r(p) = 1 - sum_i (p_i - 1[i = y])^2. It keeps calibration, so
               RL does not push every answer to probability 1.
     Plus --lam-cal * soft ECE (openjev.rlcd) and --lam-kl * KL(p || p_ref).

The reference policy for KL: with --init (RL from SFT) a FROZEN copy of every trainable
tensor as loaded from --init (SFT LoRA, flag head, template delta), swapped in under no_grad
for the reference forward (FrozenRef; a few hundred MB at most). Without --init: the backbone
with the LoRA adapter turned off.

Sessions (openjev/resume.py): --save-every N writes <save-dir>/resume/step_* every N steps
(trainables, optimizer, scheduler, GradScaler, data-stream position, RNG, step, best nll);
--max-minutes stops cleanly (resume checkpoint + last/, exit 0); --resume auto continues from
the latest one, and a finished stage (DONE.json) exits at once. Under DDP rank 0 writes and
every rank loads.

Example (Kaggle T4, fp16):
  python train_prism.py --algo sft  --steps 3000 --save-dir ckpt/sft
  python train_prism.py --algo exact --reward brier --init ckpt/sft --steps 1000 --save-dir ckpt/rl

Two GPUs (OJ-062): the same flags under torchrun, one process per GPU:
  python -m torch.distributed.run --standalone --nproc_per_node 2 train_prism.py ...
--grad-accum stays the GLOBAL number of micro-batches per optimizer step (it must divide by
the number of ranks), so the global batch, the records of each step and the lr schedule do
not change. Every rank reads the same stream and runs a token-balanced share of whole
micro-batches (openjev/dist.py). --sync-mode ddp (default): one gradient all-reduce per
step (no_sync on the other micro-steps). --sync-mode local_sgd --sync-every K: no gradient
all-reduce, the trainable weights are averaged every K steps (and before every eval/save).
Models load one rank at a time (host RAM); eval, save, log and TensorBoard run on rank 0.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from openjev import add_special_tokens, target_index
from openjev import dist as jdist
from openjev import resume as rs
from openjev.data import read_jsonl, subsample_options
from openjev.mix import Mixture
from openjev.prism import PrismOpenJev, gemma_text_backbone, log_rss, prism_collate
from openjev.rlcd import ece, soft_ece

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--train", default="data/*.train.jsonl")
    ap.add_argument("--val", default="data/*.val.jsonl")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--algo", default="sft", choices=["sft", "exact", "grpo"])
    ap.add_argument("--reward", default="brier", choices=["acc", "graded", "brier"])
    ap.add_argument("--init", default=None, help="checkpoint dir to start from (RL)")
    ap.add_argument("--causal", action="store_true", help="plain causal mask (ablation)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA lr")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16, help="0 = full fine-tune")
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--group", type=int, default=8, help="GRPO samples per question")
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam-kl", type=float, default=0.02)
    ap.add_argument("--lam-cal", type=float, default=0.5)
    ap.add_argument("--max-state-tokens", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-max", type=int, default=1000)
    ap.add_argument("--save-dir", default="checkpoints/prism")
    ap.add_argument("--tb-dir", default="", help="TensorBoard dir (default <save-dir>/tb)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--ple-cpu", type=int, default=1, help="keep the PLE table on CPU (CUDA only)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--option-subsample-k", type=int, default=0,
                    help="training only: each choice question keeps its gold option + <= K random "
                         "negatives (0 = off); eval always uses all options (OJ-072)")
    ap.add_argument("--packing", default="off", choices=["off", "on"],
                    help="training only: pack each micro-batch into rows of <= --pack-len tokens, "
                         "doc-id block-diagonal mask, positions restart per request (OJ-060)")
    ap.add_argument("--pack-len", type=int, default=1024, help="row length with --packing on")
    ap.add_argument("--quant", default="none", choices=["none", "int8", "nf4"],
                    help="bitsandbytes base weights for the LoRA-target projections of the decoder "
                         "layers (int8 = LLM.int8(), nf4 = QLoRA 4-bit); LoRA, flag head, "
                         "embeddings and PLE stay unquantized (OJ-075)")
    ap.add_argument("--sync-mode", default="ddp", choices=list(jdist.SYNC_MODES),
                    help="under torchrun with >1 process: ddp = gradient all-reduce every step; "
                         "local_sgd = average the trainable weights every --sync-every steps (OJ-062)")
    ap.add_argument("--sync-every", type=int, default=50, help="local_sgd: steps between weight averages")
    ap.add_argument("--length-curriculum", default="off", choices=["off", "on"],
                    help="training only: short requests first. The run's steps x global-batch records "
                         "(drawn like the Mixture) are bucketed by encoded length into "
                         "--curriculum-stages, never cut; one epoch = every record once, stages in "
                         "increasing length (openjev.packing.CurriculumStream)")
    ap.add_argument("--curriculum-stages", default="128,256,512,1024,2048",
                    help="stage upper bounds in tokens; longer requests train in the last stage")
    ap.add_argument("--length-grouping", default=None, choices=["off", "on"],
                    help="training only (default on; off with --length-curriculum on, which already "
                         "groups by length): length-grouped micro-batches in a random order. "
                         "--group-chunk x --bs records of the Mixture are sorted by encoded length, "
                         "cut into micro-batches of --bs, and the micro-batch order is shuffled "
                         "(seeded; same on every DDP rank). Not a curriculum (openjev.packing."
                         "LengthGroupedStream)")
    ap.add_argument("--group-chunk", type=int, default=50,
                    help="--length-grouping: micro-batches per sorted mega-chunk")
    ap.add_argument("--torch-compile", default="off", choices=["off", "default", "reduce-overhead"],
                    help="torch.compile the decoder layers only (OJ-079; openjev/compile.py): "
                         "dynamic shapes, Inductor (default) or Inductor + CUDA graphs "
                         "(reduce-overhead). PLE, masks, flag head and loss stay eager. A compile "
                         "failure on the first micro-batch falls back to eager (logged)")
    ap.add_argument("--curriculum-pack-mult", type=int, default=1,
                    help="with --length-curriculum on and --packing on: pack_len = this x the "
                         "stage max of the micro-batch (replaces --pack-len)")
    # Resumable sessions (openjev/resume.py).
    ap.add_argument("--resume", default="off",
                    help="off | auto (latest <save-dir>/resume/step_*, fresh start if none; exit at "
                         "once if <save-dir>/DONE.json exists) | a resume checkpoint dir")
    ap.add_argument("--resume-ignore-config", action="store_true",
                    help="resume even if the trajectory config differs (prints the diff)")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop cleanly (resume checkpoint + last/, exit 0) once this many minutes have "
                         "passed since main() started, model load included (0 = no limit)")
    ap.add_argument("--save-every", type=int, default=100,
                    help="write a resume checkpoint every N optimizer steps (0 = only at stop/end)")
    ap.add_argument("--keep-resume", type=int, default=2, help="resume checkpoints to keep")
    ap.add_argument("--session-steps", type=int, default=0,
                    help="stop (like --max-minutes) after N optimizer steps in this session (tests)")
    args = ap.parse_args(argv)
    if args.save_every < 0 or args.session_steps < 0 or args.max_minutes < 0:
        ap.error("--save-every, --session-steps and --max-minutes must be >= 0")
    if args.option_subsample_k < 0:
        ap.error("--option-subsample-k must be >= 0")
    if args.pack_len <= 0:
        ap.error("--pack-len must be > 0")
    if args.sync_every < 1:
        ap.error("--sync-every must be >= 1")
    if args.packing == "on" and args.causal:
        ap.error("--causal ignores the doc mask and per-request positions; it cannot use --packing on")
    if args.quant != "none" and args.lora_r == 0:
        ap.error("--quant needs LoRA (--lora-r > 0): quantized base weights cannot train")
    if args.length_grouping == "on" and args.length_curriculum == "on":
        ap.error("--length-grouping on and --length-curriculum on exclude each other (the curriculum "
                 "already groups by length)")
    if args.length_grouping is None:
        args.length_grouping = "off" if args.length_curriculum == "on" else "on"
    if args.group_chunk < 1:
        ap.error("--group-chunk must be >= 1")
    if args.curriculum_pack_mult < 1:
        ap.error("--curriculum-pack-mult must be >= 1 (unpacked: --packing off)")
    if args.length_curriculum == "on":
        from openjev.packing import parse_stages
        try:
            parse_stages(args.curriculum_stages)
        except ValueError as e:
            ap.error(str(e))
    return args


def subsample_all(records, args, rng):
    """Option sub-sampling (--option-subsample-k) of training records. rng: a Random (one
    shared stream) or a str key: then record i draws from Random(f"{key}:{i}:{id}"), so the
    options of a record do not depend on which rank or micro-step collates it (DDP)."""
    k = args.option_subsample_k
    if not k:
        return list(records)
    if isinstance(rng, str):
        return [subsample_options(r, k, random.Random(f"{rng}:{i}:{r.id}")) for i, r in enumerate(records)]
    return [subsample_options(r, k, rng) for r in records]


def collate_train(tok, records, args, pack_len=None):
    """Training collate: packing (--packing / --pack-len; `pack_len` overrides --pack-len,
    the length curriculum's per-stage row length); records already sub-sampled."""
    return prism_collate(tok, records, args.max_state_tokens,
                         pack_len=(pack_len or args.pack_len) if args.packing == "on" else 0)


def train_batch(tok, records, args, rng, pack_len=None):
    """One TRAINING micro-batch: option sub-sampling (--option-subsample-k, with `rng`)
    and packing (--packing / --pack-len; `pack_len` overrides --pack-len, the length
    curriculum's per-stage row length). Eval and the probe use prism_collate directly
    (all options, unpacked)."""
    return collate_train(tok, subsample_all(records, args, rng), args, pack_len)


def step_micro_batches(records, args, step: int, dp) -> list[tuple[int, list]]:
    """The micro-batches of one optimizer step that this rank runs: (global index, records).
    `records` = the whole step (bs x grad_accum). The cut into micro-batches is the
    single-process one; dp.mine keeps a token-balanced share of whole micro-batches."""
    recs = subsample_all(records, args, f"{args.seed}:{step}")
    mbs = [recs[i:i + args.bs] for i in range(0, len(recs), args.bs)]
    costs = [jdist.mb_cost(m, args.max_state_tokens, args.packing == "on") for m in mbs] \
        if dp.world > 1 else [0] * len(mbs)
    return [(k, mbs[k]) for k in dp.mine(costs)]


def train_stream(tok, train_paths, args):
    """The training stream: the temperature Mixture (--length-grouping on: wrapped in a
    LengthGroupedStream), or (--length-curriculum on) a CurriculumStream over
    steps x bs x grad_accum records drawn like the Mixture."""
    if args.length_curriculum != "on":
        mix = Mixture(train_paths, alpha=args.alpha, seed=args.seed)
        if args.length_grouping != "on":
            return mix
        from openjev.packing import LengthGroupedStream, request_lengths
        print(json.dumps({"length_grouping": {"bs": args.bs, "chunk_mbs": args.group_chunk}}), flush=True)
        return LengthGroupedStream(mix, lambda recs: request_lengths(tok, recs, args.max_state_tokens),
                                   args.bs, args.group_chunk, args.seed)
    from openjev.packing import build_curriculum
    s = build_curriculum(tok, train_paths, args.steps * args.bs * args.grad_accum, args.max_state_tokens,
                         args.curriculum_stages, args.alpha, args.seed, args.curriculum_pack_mult)
    print(json.dumps({"length_curriculum": {"stages": list(s.stages), "records": len(s.records),
                                            "per_stage": s.stage_counts(), "pool_sha": s.pool_sha,
                                            "pack_mult": s.pack_mult if args.packing == "on" else 0}}),
          flush=True)
    return s


# ---------------------------------------------------------------- questions

def question_views(logits, batch):
    """Yield (logits_k, target, type) for every labelled question in the batch."""
    for ex_groups, labels, schema in zip(batch.groups, batch.labels, batch.schemas):
        for g, y, q in zip(ex_groups, labels, schema.questions):
            if y is None:
                continue
            t = target_index(q, y)
            if t is not None:
                yield logits[g["start"]:g["end"]], t, g["type"]


def view_sources(records) -> list[str]:
    """The source of every question that question_views yields, in the same
    order (labelled questions only). Zipping question_views against one source
    per question, labelled or not, shifted the sources after a None label."""
    return [r.source for r in records
            for q, y in zip(r.schema.questions, r.answers)
            if y is not None and target_index(q, y) is not None]


def reward_vector(k: int, t: int, kind: str, mode: str, device) -> torch.Tensor:
    r = torch.zeros(k, device=device)
    if mode == "graded" and kind == "score" and k > 1:
        r = 1 - (torch.arange(k, device=device) - t).abs().float() / (k - 1)
    else:
        r[t] = 1.0
    return r


def sft_loss(logits, batch):
    terms = [F.cross_entropy(l[None], torch.tensor([t], device=l.device))
             for l, t, _ in question_views(logits, batch)]
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def calib_term(views):
    """Soft ECE over all questions of the batch, using top-prob confidence."""
    conf, correct = [], []
    for l, t, _ in views:
        p = l.softmax(-1)
        c, i = p.max(-1)
        conf.append(c)
        correct.append((i == t).float())
    if len(conf) < 2:
        return torch.zeros((), device=views[0][0].device) if views else 0.0
    from openjev.rlcd import soft_ece_from_conf
    return soft_ece_from_conf(torch.stack(conf), torch.stack(correct))


def exact_pg_loss(logits, ref_logits, batch, args):
    views = list(question_views(logits, batch))
    refs = list(question_views(ref_logits, batch)) if ref_logits is not None else None
    terms, kls, rewards = [], [], []
    for j, (l, t, kind) in enumerate(views):
        logp = l.log_softmax(-1)
        p = logp.exp()
        if args.reward == "brier":
            onehot = F.one_hot(torch.tensor(t, device=l.device), l.numel()).float()
            r = 1 - ((p - onehot) ** 2).sum()
            terms.append(-r)
            rewards.append(r.detach())
        else:
            rv = reward_vector(l.numel(), t, kind, args.reward, l.device)
            exp_r = (p * rv).sum()
            terms.append(-exp_r)  # gradient = sum_i p_i (r_i - E r) grad log p_i
            rewards.append(exp_r.detach())
        if refs is not None:
            rl = refs[j][0].log_softmax(-1)
            kls.append((p * (logp - rl)).sum())
    loss = torch.stack(terms).mean()
    kl = torch.stack(kls).mean() if kls else torch.zeros((), device=logits.device)
    cal = calib_term(views)
    return loss + args.lam_kl * kl + args.lam_cal * cal, {
        "reward": torch.stack(rewards).mean().item(), "kl": kl.item(), "soft_ece": float(cal)}


def grpo_sample(logits, batch, args, generator=None):
    """Sample G actions per question with the current (old) policy. generator: seeded per
    (step, micro-batch) by main, so the samples do not depend on the rank."""
    out = []
    for l, t, kind in question_views(logits.detach(), batch):
        logp = l.log_softmax(-1)
        a = torch.multinomial(logp.exp(), args.group, replacement=True, generator=generator)
        r = reward_vector(l.numel(), t, kind, "graded" if args.reward == "graded" else "acc",
                          l.device)[a]
        out.append({"a": a, "old": logp[a], "adv": r - r.mean(), "r": r.mean()})
    return out


def grpo_loss(logits, ref_logits, batch, samples, args):
    views = list(question_views(logits, batch))
    refs = list(question_views(ref_logits, batch))
    terms, kls = [], []
    for (l, _, _), (rl, _, _), s in zip(views, refs, samples):
        logp = l.log_softmax(-1)[s["a"]]
        ratio = (logp - s["old"]).exp()
        pg = -torch.min(ratio * s["adv"], ratio.clamp(1 - args.clip, 1 + args.clip) * s["adv"])
        terms.append(pg.sum() / args.group)              # Dr. GRPO: no std norm, no length norm
        lr = rl.log_softmax(-1)[s["a"]]
        kls.append(((lr - logp).exp() - (lr - logp) - 1).mean())  # k3 estimator
    loss = torch.stack(terms).mean()
    kl = torch.stack(kls).mean()
    cal = calib_term(views)
    return loss + args.lam_kl * kl + args.lam_cal * cal, {
        "reward": torch.stack([s["r"] for s in samples]).mean().item(),
        "kl": kl.item(), "soft_ece": float(cal)}


# ---------------------------------------------------------------- model

def auto_dtype(device: str) -> torch.dtype:
    """--dtype auto: bf16 only where the GPU has native bf16 (compute capability >= 8.0).
    torch.cuda.is_bf16_supported() also counts EMULATED bf16 by default
    (including_emulation=True), so on a T4 (sm75) it picked slow emulated bf16 (OJ-081).
    T4 gets fp16 (+ GradScaler); CPU gets fp32."""
    if device != "cuda":
        return torch.float32
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def build(args, device, base: str | None = None):
    """base: load this backbone instead of the one named in --init's
    prism_config.json (tools/diag_predictions.py --model)."""
    from transformers import AutoTokenizer

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(args.dtype)
    if dtype is None:
        dtype = auto_dtype(device)
    src = args.init or args.model
    tok = AutoTokenizer.from_pretrained(src)
    base_name = args.model
    if args.init and (Path(args.init) / "prism_config.json").exists():
        base_name = json.loads((Path(args.init) / "prism_config.json").read_text())["base"]
    base_name = base or base_name
    # On CUDA, load bf16 and convert AFTER moving to the GPU: converting on the host
    # turned the mmap-backed weights into ~10 GB of anonymous RAM and Kaggle
    # SIGKILLed the r3 bench right after "after_convert" (host RSS 10.24 GB).
    host_convert = device == "cpu"
    model = PrismOpenJev(gemma_text_backbone(base_name, dtype=dtype, convert=host_convert),
                         causal=args.causal)
    add_special_tokens(tok, model)
    log_rss("after_resize")          # mean_resizing holds two fp32 copies of the embedding
    # A tokenizer saved by a checkpoint already holds the template tokens, so
    # add_special_tokens adds nothing and does not resize. Gemma has no spare
    # rows, so make sure every token id has an embedding row.
    rows = model.get_input_embeddings().weight.shape[0]
    if len(tok) > rows:
        model.resize_token_embeddings(-(-len(tok) // 128) * 128)

    quant = getattr(args, "quant", "none")
    if quant != "none":
        # Before the LoRA wrap, so PEFT wraps the bnb layers. Each layer lands quantized on
        # `device`; place() below moves the rest. fp16 compute on the T4 (bnb's sm75 path).
        from openjev.prism import quantize_linears
        quantize_linears(model._text_backbone(), quant, device, compute_dtype=dtype)
        log_rss("after_quantize")

    if args.grad_ckpt:
        set_grad_ckpt(model, True)

    if args.lora_r > 0:
        from peft import LoraConfig, PeftModel, get_peft_model
        if args.init:
            model.backbone = PeftModel.from_pretrained(model.backbone, args.init, is_trainable=True)
        else:
            cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                             target_modules=LORA_TARGETS, bias="none")
            model.backbone = get_peft_model(model.backbone, cfg)
    from openjev.schema import SPECIAL_TOKENS
    model.set_template_ids(tok.convert_tokens_to_ids(SPECIAL_TOKENS))
    if args.init:
        init = Path(args.init)
        model.flag_head.load_state_dict(torch.load(init / "heads.pt", map_location="cpu"))
        if (init / "embeddings.pt").exists():
            rows = torch.load(init / "embeddings.pt", map_location="cpu")
            with torch.no_grad():
                for i, v in rows.items():
                    model.get_input_embeddings().weight[int(i)] = v
        if (init / "template_delta.pt").exists():
            model.template_delta.data.copy_(torch.load(init / "template_delta.pt", map_location="cpu"))
    # Every trainable tensor in fp32: AdamW on fp16 params underflows (eps=1e-8
    # is 0 in fp16) and loses small updates. Frozen weights stay fp16/bf16.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model = place(model, device, args.ple_cpu)
    log_rss("after_place")
    if not host_convert:
        from openjev.prism import convert_inplace
        convert_inplace(model.backbone, dtype)   # frozen weights only; in place, per tensor
        log_rss("after_device_convert")
    if args.init:
        # A backbone loaded from the checkpoint itself (full fine-tune) holds its PLE table.
        restore_ple_rows(model, Path(args.init), tok, warn=Path(base_name) != Path(args.init))
    return model, tok, dtype


def restore_ple_rows(model, init: Path, tok, warn: bool = True) -> None:
    """Gemma 4: the --init build resizes the PLE table again, and the resize fills the
    template rows with its own random init. Paste the rows the checkpoint trained with
    (ple_rows.pt), after place() and the dtype convert so they land bit-exact. A checkpoint
    saved before ple_rows.pt existed gets a warning: its template PLE rows differ from the
    ones it trained with (eval_prism.load_prism replays the seeded build for those)."""
    from openjev.prism import PLE_ROWS, paste_ple_rows, template_ple_rows
    from openjev.schema import SPECIAL_TOKENS
    if (init / PLE_ROWS).exists():
        paste_ple_rows(model, torch.load(init / PLE_ROWS, map_location="cpu"), str(init))
    elif warn and template_ple_rows(model, tok.convert_tokens_to_ids(SPECIAL_TOKENS)) is not None:
        warnings.warn(f"{init} has no {PLE_ROWS} (saved before the PLE persistence fix): the "
                      "template tokens' PLE rows are the resize's random init, NOT the rows "
                      "this checkpoint trained with", stacklevel=2)


def set_grad_ckpt(model, on: bool) -> None:
    """Turn gradient checkpointing on or off on the text decoder. Works before
    and after the LoRA wrap (the PeftModel is unwrapped first)."""
    text = model._text_backbone()
    if on:
        text.gradient_checkpointing_enable()
        if hasattr(text, "enable_input_require_grads"):
            text.enable_input_require_grads()
    elif getattr(text, "is_gradient_checkpointing", False):
        text.gradient_checkpointing_disable()


def make_optimizer(model, args):
    """AdamW with two lr groups: the flag head (--head-lr) and everything else
    that trains (LoRA, template delta) at --lr."""
    head = list(model.flag_head.parameters())
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("flag_head")]
    return torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.head_lr}],
                             weight_decay=0.0)


class FrozenRef:
    """The RL KL reference = the policy as loaded from --init (SFT), frozen. A snapshot of
    every trainable tensor (LoRA, flag head, template delta) is taken at construction: build
    it after build() loads --init and BEFORE a resume restores the RL weights. logits() swaps
    the snapshot in under no_grad, runs the forward in its own autocast block and swaps the
    live tensors back. The autocast cast cache is keyed on the parameter, so it is cleared
    around the swap: a stale fp16 cast of the live weights would otherwise be reused for the
    reference (KL ~ 0 on the T4) or vice versa. Call it outside the training autocast block."""

    def __init__(self, model):
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.snap = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def logits(self, b, device):
        live = [p.data for p in self.params]
        torch.clear_autocast_cache()
        try:
            for p, s in zip(self.params, self.snap):
                p.data = s
            with autocast(device, self.model):
                return self.model.flag_logits(b)
        finally:
            for p, d in zip(self.params, live):
                p.data = d
            torch.clear_autocast_cache()


MAX_GRAD_NORM = 1.0


def optimizer_step(opt, scaler):
    """Unscale, clip to MAX_GRAD_NORM, step, update the scaler, zero the grads. Returns the
    grad norm."""
    scaler.unscale_(opt)
    gnorm = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],
                                           MAX_GRAD_NORM)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return gnorm


def pick_device(args) -> str:
    return "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"


def place(model, device, ple_cpu: bool):
    """Move to device; optionally keep Gemma's Per-Layer Embedding table on CPU.
    The table (262144 x layers*256: 2.35B params in E2B, 2.82B in E4B) is only
    a lookup, so a CPU copy costs a small per-batch transfer and frees 4.7-5.6 GB."""
    found = None
    if ple_cpu and device != "cpu":
        for name, mod in model.named_modules():
            if name.endswith("embed_tokens_per_layer"):
                parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
                found = (parent, name.rsplit(".", 1)[-1], mod)
                break
    if found is None:
        return model.to(device)
    # Detach the whole module (weight AND buffers such as embed_scale) while
    # the rest moves, so it never touches the GPU and never mixes devices.
    parent, attr, ple = found
    setattr(parent, attr, torch.nn.Identity())
    model.to(device)
    setattr(parent, attr, ple.to("cpu"))
    return model


def save(model, tok, args, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model.backbone, "peft_config"):
        # LoRA adapter only. Without save_embedding_layers=False PEFT sees the resized
        # vocab and also stores the whole embedding matrix; embeddings.pt (template rows)
        # and ple_rows.pt carry everything that differs from the base.
        model.backbone.save_pretrained(out, save_embedding_layers=False)
    else:
        model.backbone.save_pretrained(out)      # full fine-tune: the whole backbone
    tok.save_pretrained(out)
    torch.save(model.flag_head.state_dict(), out / "heads.pt")
    from openjev.prism import PLE_ROWS, template_ple_rows
    from openjev.schema import SPECIAL_TOKENS
    ids = tok.convert_tokens_to_ids(SPECIAL_TOKENS)
    w = model.get_input_embeddings().weight
    # clone: a view of a CPU tensor would serialize the whole matrix
    torch.save({str(i): w[i].detach().cpu().clone() for i in ids}, out / "embeddings.pt")
    ple_rows = template_ple_rows(model, ids)
    if ple_rows is not None:      # Gemma 4: the template tokens' own (frozen) PLE rows
        torch.save(ple_rows, out / PLE_ROWS)
    torch.save(model.template_delta.detach().cpu(), out / "template_delta.pt")
    (out / "prism_config.json").write_text(json.dumps(
        {"base": args.model if not args.init else json.loads(
            (Path(args.init) / "prism_config.json").read_text())["base"],
         "causal": args.causal, "args": vars(args)}, indent=2))


# ---------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, tok, records, args, device):
    model.eval()
    per = {}
    lens = []
    for i in range(0, len(records), args.bs):
        chunk = records[i:i + args.bs]
        b = prism_collate(tok, chunk, args.max_state_tokens).to(device)
        lens += b.attention_mask.sum(-1).tolist()
        with autocast(device, model):
            logits = model.flag_logits(b)
        views = list(question_views(logits, b))
        srcs = view_sources(chunk)
        assert len(views) == len(srcs), (len(views), len(srcs))
        for (l, t, kind), src in zip(views, srcs):
            per.setdefault(src, []).append((l.softmax(-1).cpu(), t))
    model.train()
    # acc (+ 2 SE), nll, 15-bin ECE and Brier per source and overall (openjev.metrics).
    # per_source stays {source: acc} (TensorBoard, tools/optuna_loop.py); "sources" has the rows.
    from openjev.metrics import source_metrics
    m = source_metrics([(p, t, src) for src, items in per.items() for p, t in items], nd=4)
    o = m["overall"]
    return {"acc": o["acc"], "nll": o["nll"], "ece": o["ece"], "brier": o["brier"],
            "n": o["n"], "acc_se2": o["acc_se2"],
            "per_source": {s: r["acc"] for s, r in m["sources"].items()}, "sources": m["sources"],
            "seq_p95": round(float(np.percentile(lens, 95)), 1) if lens else 0.0,
            "seq_max": int(max(lens)) if lens else 0}


def load_val(pattern: str, eval_max: int, seed: int) -> list:
    """A fixed seeded subset: eval_max split evenly over the files of `pattern`."""
    paths = sorted(glob.glob(pattern))
    val = []
    for p in paths:
        recs = list(read_jsonl(p))
        random.Random(seed).shuffle(recs)
        val += recs[: max(1, eval_max // max(1, len(paths)))]
    return val


def make_writer(args):
    """TensorBoard SummaryWriter, or None if unavailable. Never raises."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(args.tb_dir or str(Path(args.save_dir) / "tb"))
    except Exception as e:
        print(f"warning: TensorBoard logging disabled ({type(e).__name__}: {e})", flush=True)
        return None


def autocast(device, model):
    dt = model.get_input_embeddings().weight.dtype
    if device == "cuda" and dt in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dt)
    return torch.autocast("cpu", enabled=False)


# ---------------------------------------------------------------- main

def compile_or_fallback(model, net, tok, records, args, device, dp) -> dict:
    """--torch-compile: compile the decoder layers, then one forward + backward on `records`
    (no optimizer step, grads dropped; no_sync under DDP) so a kernel Inductor cannot build fails
    HERE, not hours into the run. On a compiler error: back to eager on every rank, logged.
    After a good check, dynamo's suppress_errors is set: a later frame that fails to compile
    (a new shape) runs eager instead of ending the run."""
    from openjev import compile as tc
    info = {"mode": args.torch_compile, "fallback": None}
    tc.set_torch_compile(model, args.torch_compile, fail_on_limit=False)
    failed = None
    with tc.Timer() as t:
        try:
            b = prism_collate(tok, records[: args.bs], args.max_state_tokens).to(device)
            with dp.sync(False):
                with autocast(device, model):
                    loss = sft_loss(net(b), b)
                loss.backward()
            del b, loss
        except Exception as e:
            if not tc.is_compile_error(e):
                raise
            failed = f"{type(e).__name__}: {e}"[:300]
    model.zero_grad(set_to_none=True)
    if dp.world > 1 and dp.any(failed is not None) and failed is None:
        failed = "another rank failed to compile"
    if failed:
        tc.unset_torch_compile(model)
        info.update(mode="off", fallback=failed)
    else:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    info.update(check_s=round(t.s, 1), **(tc.stats() or {}))
    return info


def grpo_generator(device, seed: int, step: int, k: int) -> torch.Generator:
    """GRPO sampling RNG of micro-batch k of `step`: the same on any rank / world size."""
    return torch.Generator(device=device).manual_seed(seed * 1_000_003 + step * 1009 + k)


def main(argv=None):
    args = parse(argv)
    t_start = time.time()                     # --max-minutes clock (model load included)
    save_dir = Path(args.save_dir)
    done_file = save_dir / "DONE.json"
    if args.resume == "auto" and done_file.exists():   # every rank sees it: no collective yet
        print(json.dumps({"resume": "stage already complete", **json.loads(done_file.read_text())}),
              flush=True)
        return
    device = pick_device(args)
    jdist.init(device)
    dp = jdist.DataParallel(device, args.sync_mode, args.sync_every)
    if dp.local_sgd and args.resume != "off":
        raise SystemExit("--resume needs --sync-mode ddp: local_sgd keeps per-rank optimizer state, "
                         "which a rank-0 checkpoint does not hold")
    if args.grad_accum % dp.world:
        raise SystemExit(f"--grad-accum {args.grad_accum} (global micro-batches per step) "
                         f"must divide by the {dp.world} ranks")
    main_rank = dp.rank == 0

    def load():
        # The same seed on every rank: equal LoRA A, flag head and new embedding rows.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        return build(args, device)
    model, tok, dtype = jdist.staggered(load)     # one rank at a time (host RAM peak)
    model.train()
    net = dp.attach(model)    # training forward; eval, sampling, KL ref and save use `model`
    accum_local = args.grad_accum // dp.world
    if dp.world > 1 and main_rank:
        print(json.dumps({"dist": {"world": dp.world, "sync_mode": args.sync_mode,
                                   "sync_every": args.sync_every if dp.local_sgd else None,
                                   "global_batch": args.bs * args.grad_accum,
                                   "accum_per_rank": accum_local}}), flush=True)

    train_paths = sorted(glob.glob(args.train))
    if not train_paths:
        raise SystemExit(f"no train files match {args.train}")
    # The same seeded stream on every rank; step_micro_batches keeps this rank's share.
    mixture = train_stream(tok, train_paths, args)
    val = load_val(args.val, args.eval_max, args.seed)
    if args.torch_compile != "off":
        info = compile_or_fallback(model, net, tok, val, args, device, dp)
        if main_rank:
            print(json.dumps({"torch_compile": info}), flush=True)

    opt = make_optimizer(model, args)
    warm = max(1, int(args.warmup * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and dtype == torch.float16))
    use_ref = args.algo != "sft" and args.lam_kl > 0 or args.algo == "grpo"
    # Snapshot BEFORE the resume load below: the reference stays the --init (SFT) policy.
    frozen_ref = FrozenRef(model) if use_ref and args.init and args.lora_r > 0 else None

    def ref_logits(b):
        """Reference logits; call outside the training autocast block (FrozenRef)."""
        if not use_ref:
            return None
        if frozen_ref is not None:
            return frozen_ref.logits(b, device)
        with torch.no_grad(), autocast(device, model):
            if hasattr(model.backbone, "disable_adapter"):
                with model.backbone.disable_adapter():
                    return model.flag_logits(b)
            return model.flag_logits(b)  # full fine-tune: KL vs self (no-op), use LoRA instead

    stage = rs.stage_of(args.algo)
    cfg = rs.config_of(args, train_paths)
    start, best = 0, float("inf")
    if args.resume != "off":
        # Every rank loads the same checkpoint (in place: DDP and AdamW keep their references).
        path = rs.find_latest(save_dir / "resume") if args.resume == "auto" else Path(args.resume)
        if path is not None:
            r = rs.load(path, model=model, opt=opt, sched=sched, scaler=scaler, mixture=mixture,
                        device=device, cfg=cfg, ignore_config=args.resume_ignore_config)
            if r["stage"] != stage:
                raise SystemExit(f"resume: checkpoint stage {r['stage']} != {stage} ({path})")
            start, best = r["step"], r["best"]
            dp.steps = start
            if main_rank:
                print(json.dumps({"resumed": str(path), "step": start, "best": best,
                                  "notes": r["notes"]}), flush=True)
        elif main_rank:
            print(json.dumps({"resume": "no checkpoint, fresh start", "dir": str(save_dir)}), flush=True)
        jdist.barrier()

    def checkpoint(step_done):
        """Collective: every rank calls it; rank 0 writes, the others wait."""
        dp.average()
        if main_rank:
            p = rs.save(save_dir, step=step_done, stage=stage, model=model, opt=opt, sched=sched,
                        scaler=scaler, mixture=mixture, best=best, cfg=cfg, device=device,
                        dtype=dtype, keep=args.keep_resume)
            print(json.dumps({"checkpoint": str(p), "step": step_done,
                              "min": round((time.time() - t_start) / 60, 1)}), flush=True)
        jdist.barrier()

    tb = make_writer(args) if main_rank else None

    def tb_log(fn):
        if tb is None:
            return
        try:
            fn()
        except Exception as e:
            print(f"warning: TensorBoard logging failed ({type(e).__name__}: {e})", flush=True)

    log = save_dir / "log.jsonl"
    step_log = save_dir / "train_log.jsonl"   # one row per optimizer step (rank 0's stats)
    if main_rank:
        log.parent.mkdir(parents=True, exist_ok=True)
    t0, bad, stop_reason = time.time(), 0, None
    step = start - 1
    for step in range(start, args.steps):
        stats = {}
        mine = step_micro_batches(mixture.take(args.bs * args.grad_accum), args, step, dp)
        pack_len = getattr(mixture, "pack_len", None)   # length curriculum: per-step stage row
        for j, (k, recs) in enumerate(mine):
            last_micro = j == len(mine) - 1
            b = collate_train(tok, recs, args, pack_len).to(device)
            if args.algo == "grpo":
                with autocast(device, model):
                    with torch.no_grad():
                        samples = grpo_sample(model.flag_logits(b), b, args,
                                              grpo_generator(device, args.seed, step, k))
                ref = ref_logits(b)
                for e in range(args.ppo_epochs):
                    with dp.sync(last_micro and e == args.ppo_epochs - 1):
                        with autocast(device, model):
                            loss, stats = grpo_loss(net(b), ref, b, samples, args)
                        scaler.scale(loss / (accum_local * args.ppo_epochs)).backward()
                continue
            ref = ref_logits(b) if args.algo != "sft" else None   # before the training autocast
            with dp.sync(last_micro):
                with autocast(device, model):
                    logits = net(b)
                    if args.algo == "sft":
                        loss = sft_loss(logits, b)
                        stats = {"ce": loss.item()}
                    else:
                        loss, stats = exact_pg_loss(logits, ref, b, args)
                scaler.scale(loss / accum_local).backward()
        gnorm = optimizer_step(opt, scaler)
        sched.step()
        dp.after_step()                        # local_sgd: average every --sync-every steps

        last = stats.get("ce", stats.get("reward", 0.0))
        if dp.world > 1:                       # every rank takes the same exit decision
            last = dp.mean([last])[0]
        bad = bad + 1 if not math.isfinite(last) else 0
        if bad >= 20:
            raise SystemExit(f"loss not finite for 20 steps (step {step}): {stats}. "
                             "Likely fp16 overflow; try --dtype fp32 or a bf16 GPU.")
        if main_rank:
            with open(step_log, "a") as f:
                f.write(json.dumps({"step": step, "t": round(time.time() - t_start, 2),
                                    "seq_len": int(b.input_ids.shape[1]), "grad_norm": float(gnorm),
                                    "scale": scaler.get_scale() if scaler.is_enabled() else None,
                                    **{k: float(v) for k, v in stats.items()}}) + "\n")
        if step % 10 == 0:
            if dp.world > 1 and stats:         # mean over the ranks (collective: every rank)
                stats = dict(zip(stats, dp.mean(list(stats.values()))))
            cur = {"stage": mixture.stages[mixture.stage]} if hasattr(mixture, "stage") else {}
            if main_rank:
                print(json.dumps({"step": step, "sec": round(time.time() - t0), **stats, **cur}), flush=True)

            def _log_train():
                for k, v in stats.items():
                    tb.add_scalar(f"train/{k}", v, step)
                for name, g in zip(("lr_body", "lr_head"), opt.param_groups):
                    tb.add_scalar(f"train/{name}", g["lr"], step)
                tb.add_scalar("train/grad_norm", float(gnorm), step)
                if device == "cuda":
                    tb.add_scalar("sys/peak_mem_gb", torch.cuda.max_memory_allocated() / 2**30, step)
                tb.add_scalar("data/seq_len", b.input_ids.shape[1], step)
            tb_log(_log_train)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            dp.average()                       # local_sgd: eval / save the averaged weights
            if main_rank:
                m = evaluate(model, tok, val, args, device)
                print(json.dumps({"step": step + 1, "val": m}), flush=True)
                with open(log, "a") as f:
                    f.write(json.dumps({"step": step + 1, **m}) + "\n")

                def _log_val():
                    for k in ("acc", "nll", "ece"):
                        tb.add_scalar(f"val/{k}", m[k], step + 1)
                    for src, v in m["per_source"].items():
                        tb.add_scalar(f"val_source/{src}", v, step + 1)
                    tb.flush()
                tb_log(_log_val)
                if m["nll"] < best:
                    best = m["nll"]
                    save(model, tok, args, Path(args.save_dir) / "best")
            jdist.barrier()                    # the other ranks wait for rank 0's eval / save
        # Session control: step budget (tests), time budget, periodic resume checkpoints.
        if step + 1 < args.steps:
            if args.session_steps and step + 1 - start >= args.session_steps:
                stop_reason = "session_steps"
            elif args.max_minutes and dp.any(main_rank and time.time() - t_start >= args.max_minutes * 60):
                stop_reason = "max_minutes"    # rank 0's clock decides for every rank (collective)
            if stop_reason:
                break
            if args.save_every and (step + 1) % args.save_every == 0:
                checkpoint(step + 1)
    end = step + 1
    if end > start:
        checkpoint(end)
    if main_rank:
        save(model, tok, args, save_dir / "last")
        if stop_reason:
            print(json.dumps({"stopped": stop_reason, "step": end, "of": args.steps,
                              "resume": f"--resume auto --save-dir {save_dir}"}), flush=True)
        else:
            done_file.write_text(json.dumps({"stage": stage, "step": end, "best": best,
                                             "time": time.time()}))
            print(json.dumps({"done": stage, "step": end, "best": best}), flush=True)
        if args.torch_compile != "off":
            from openjev import compile as tc
            print(json.dumps({"torch_compile_end": tc.stats()}), flush=True)
    tb_log(lambda: tb.close())
    jdist.cleanup()


if __name__ == "__main__":
    main()
