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

The reference model for KL is the same backbone with the LoRA adapter turned
off (or the SFT adapter frozen), so it costs no extra GPU memory.

Example (Kaggle T4, fp16):
  python train_prism.py --algo sft  --steps 3000 --save-dir ckpt/sft
  python train_prism.py --algo exact --reward brier --init ckpt/sft --steps 1000 --save-dir ckpt/rl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from openjev import add_special_tokens, target_index
from openjev.data import read_jsonl
from openjev.mix import Mixture
from openjev.prism import PrismOpenJev, gemma_text_backbone, prism_collate
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
    return ap.parse_args(argv)


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


def grpo_sample(logits, batch, args):
    """Sample G actions per question with the current (old) policy."""
    out = []
    for l, t, kind in question_views(logits.detach(), batch):
        logp = l.log_softmax(-1)
        a = torch.multinomial(logp.exp(), args.group, replacement=True)
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
    model = PrismOpenJev(gemma_text_backbone(base_name, dtype=dtype), causal=args.causal)
    add_special_tokens(tok, model)
    # A tokenizer saved by a checkpoint already holds the template tokens, so
    # add_special_tokens adds nothing and does not resize. Gemma has no spare
    # rows, so make sure every token id has an embedding row.
    rows = model.get_input_embeddings().weight.shape[0]
    if len(tok) > rows:
        model.resize_token_embeddings(-(-len(tok) // 128) * 128)

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
    return place(model, device, args.ple_cpu), tok, dtype


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


def optimizer_step(opt, scaler):
    """Unscale, clip to 1.0, step, update the scaler, zero the grads. Returns the grad norm."""
    scaler.unscale_(opt)
    gnorm = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
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
    model.backbone.save_pretrained(out)          # LoRA adapter only when peft is used
    tok.save_pretrained(out)
    torch.save(model.flag_head.state_dict(), out / "heads.pt")
    from openjev.schema import SPECIAL_TOKENS
    ids = tok.convert_tokens_to_ids(SPECIAL_TOKENS)
    w = model.get_input_embeddings().weight
    torch.save({str(i): w[i].detach().cpu() for i in ids}, out / "embeddings.pt")
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
    rows, all_conf, all_ok, nll = {}, [], [], []
    for src, items in per.items():
        acc = sum(int(p.argmax() == t) for p, t in items) / len(items)
        rows[src] = round(acc, 4)
        for p, t in items:
            all_conf.append(p.max().item())
            all_ok.append(float(p.argmax() == t))
            nll.append(-math.log(max(p[t].item(), 1e-12)))
    from openjev.metrics import ece_from_conf
    return {"acc": round(sum(all_ok) / max(len(all_ok), 1), 4),
            "nll": round(sum(nll) / max(len(nll), 1), 4),
            "ece": round(ece_from_conf(all_conf, all_ok), 4), "per_source": rows,
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

def main():
    args = parse()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args)
    model, tok, dtype = build(args, device)
    model.train()

    train_paths = sorted(glob.glob(args.train))
    if not train_paths:
        raise SystemExit(f"no train files match {args.train}")
    mixture = Mixture(train_paths, alpha=args.alpha, seed=args.seed)
    val = load_val(args.val, args.eval_max, args.seed)

    opt = make_optimizer(model, args)
    warm = max(1, int(args.warmup * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and dtype == torch.float16))
    use_ref = args.algo != "sft" and args.lam_kl > 0 or args.algo == "grpo"

    def ref_logits(b):
        if not use_ref:
            return None
        with torch.no_grad():
            if hasattr(model.backbone, "disable_adapter"):
                with model.backbone.disable_adapter():
                    return model.flag_logits(b)
            return model.flag_logits(b)  # full fine-tune: KL vs self (no-op), use LoRA instead

    tb = make_writer(args)

    def tb_log(fn):
        if tb is None:
            return
        try:
            fn()
        except Exception as e:
            print(f"warning: TensorBoard logging failed ({type(e).__name__}: {e})", flush=True)

    log = Path(args.save_dir) / "log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    best, t0, bad = float("inf"), time.time(), 0
    for step in range(args.steps):
        stats = {}
        for _ in range(args.grad_accum):
            b = prism_collate(tok, mixture.take(args.bs), args.max_state_tokens).to(device)
            if args.algo == "grpo":
                with autocast(device, model):
                    with torch.no_grad():
                        samples = grpo_sample(model.flag_logits(b), b, args)
                    ref = ref_logits(b)
                for _ in range(args.ppo_epochs):
                    with autocast(device, model):
                        loss, stats = grpo_loss(model.flag_logits(b), ref, b, samples, args)
                    scaler.scale(loss / (args.grad_accum * args.ppo_epochs)).backward()
                continue
            with autocast(device, model):
                logits = model.flag_logits(b)
                if args.algo == "sft":
                    loss = sft_loss(logits, b)
                    stats = {"ce": loss.item()}
                else:
                    loss, stats = exact_pg_loss(logits, ref_logits(b), b, args)
            scaler.scale(loss / args.grad_accum).backward()
        gnorm = optimizer_step(opt, scaler)
        sched.step()

        last = stats.get("ce", stats.get("reward", 0.0))
        bad = bad + 1 if not math.isfinite(last) else 0
        if bad >= 20:
            raise SystemExit(f"loss not finite for 20 steps (step {step}): {stats}. "
                             "Likely fp16 overflow; try --dtype fp32 or a bf16 GPU.")
        if step % 10 == 0:
            print(json.dumps({"step": step, "sec": round(time.time() - t0), **stats}), flush=True)

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
    save(model, tok, args, Path(args.save_dir) / "last")
    tb_log(lambda: tb.close())


if __name__ == "__main__":
    main()
