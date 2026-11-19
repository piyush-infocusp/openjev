"""OpenJev SFT trainer.

    python train.py --train "data/*.train.jsonl" --val "data/*.val.jsonl" \
        --model Qwen/Qwen3-0.6B --steps 2000 --bs 16 --save-dir checkpoints/run

CPU-only smoke test (no GPU on this machine):

    python train.py --train "tests/fixtures/*.jsonl" --val "tests/fixtures/*.jsonl" \
        --model trl-internal-testing/tiny-Qwen3ForCausalLM --steps 6 --bs 4 --cpu \
        --eval-every 3 --save-dir /tmp/run
"""

from __future__ import annotations

import argparse
import glob
import inspect
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from openjev import OpenJev, add_special_tokens, question_loss
from openjev import moe as moe_mod
from openjev.collate import collate
from openjev.data import read_jsonl
from openjev.metrics import Accumulator
from openjev.mix import Mixture

MOE_PARAM_BUDGET = 1_000_000_000


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train", nargs="+", default=["data/*.train.jsonl"])
    ap.add_argument("--val", nargs="+", default=["data/*.val.jsonl"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--loss", default="ce", choices=["ce", "brier", "focal"])
    ap.add_argument("--score-loss", default="ce+emd", choices=["ce", "ce+emd", "emd"])
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--max-state-tokens", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-max", type=int, default=2000)
    ap.add_argument("--save-dir", default="checkpoints/run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--upcycle", default=None,
                     help="'E,N' -> sparse-upcycle the last N decoder layers into E experts")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--aux-weight", type=float, default=0.01)
    return ap.parse_args(argv)


def parse_upcycle(spec: str) -> tuple[int, int]:
    e_str, n_str = spec.split(",")
    return int(e_str), int(n_str)


def _expand(patterns: list[str]) -> list[str]:
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        raise ValueError(f"no files matched {patterns!r}")
    return paths


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _choice_loss_smoothed(logits: torch.Tensor, k: int, eps: float) -> torch.Tensor:
    target = torch.tensor([k], device=logits.device)
    return F.cross_entropy(logits.unsqueeze(0), target, label_smoothing=eps)


def compute_losses(preds: torch.Tensor, batch, args) -> tuple[torch.Tensor, int]:
    """preds: (F,) flag logits from model.flag_logits(batch). Returns
    (mean loss over labelled questions, count of labelled questions).
    """
    from openjev.model import target_index

    accepts_score_loss = "score_loss" in inspect.signature(question_loss).parameters
    losses = []
    for ex_groups, ex_labels, schema in zip(batch.groups, batch.labels, batch.schemas):
        for g, y, q in zip(ex_groups, ex_labels, schema.questions):
            if y is None:
                continue
            logits = preds[g["start"]:g["end"]]
            if q.type == "choice" and args.label_smoothing > 0:
                k = target_index(q, y)
                losses.append(_choice_loss_smoothed(logits, k, args.label_smoothing))
                continue
            pred = {"type": q.type, "name": q.name, "logits": logits}
            kwargs = dict(loss=args.loss)
            if accepts_score_loss:
                kwargs["score_loss"] = args.score_loss
            l = question_loss(pred, q, y, **kwargs)
            if l is not None:
                losses.append(l)
    if not losses:
        return preds.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def lr_lambda(step: int, total_steps: int, warmup_frac: float):
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, tokenizer, val_records, args, device) -> dict:
    model.eval()
    acc = Accumulator()
    n = 0
    for i in range(0, min(len(val_records), args.eval_max), args.bs):
        chunk = val_records[i:i + args.bs]
        batch = collate(tokenizer, chunk, max_state_tokens=args.max_state_tokens).to(device)
        logits = model.flag_logits(batch)
        for ex_groups, ex_labels, schema, record in zip(batch.groups, batch.labels,
                                                          batch.schemas, chunk):
            for g, y, q in zip(ex_groups, ex_labels, schema.questions):
                if y is None:
                    continue
                from openjev.model import target_index
                k = target_index(q, y)
                probs = torch.softmax(logits[g["start"]:g["end"]], -1)
                acc.add(q.type, record.source, probs, k)
                n += 1
        if n >= args.eval_max:
            break
    model.train()
    return acc.summary()


def print_table(summary: dict) -> None:
    keys = ["n", "acc", "nll", "brier", "ece", "mae"]
    header = "{:<20}" + "".join("{:>10}" for _ in keys)
    print(header.format("group", *keys))
    for name, stats in summary.items():
        row = [name] + [f"{stats.get(k, ''):.4f}" if isinstance(stats.get(k), float) and k != "n"
                         else str(stats.get(k, "")) for k in keys]
        print(header.format(*row))


def save_checkpoint(model, tokenizer, args, save_dir: Path, moe_cfg: dict | None = None) -> None:
    from openjev import checkpoint
    from openjev.schema import SPECIAL_TOKENS

    final_dir = save_dir / "final"
    special_ids = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
    config = {"backbone": args.model, "args": vars(args), "special_token_ids": special_ids}
    if moe_cfg:
        config["moe"] = moe_cfg
    checkpoint.save(model, tokenizer, final_dir, config)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    use_fp16 = args.fp16 and device == "cuda"

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "log.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = OpenJev(args.model)
    add_special_tokens(tokenizer, model)

    moe_cfg = None
    if args.upcycle:
        n_experts, last_n_layers = parse_upcycle(args.upcycle)
        moe_mod.upcycle(model.backbone, n_experts=n_experts, last_n_layers=last_n_layers,
                         top_k=args.top_k)
        moe_cfg = {"n_experts": n_experts, "last_n_layers": last_n_layers, "top_k": args.top_k}
        counts = moe_mod.param_counts(model)
        print(f"moe param_counts: {counts}")
        if counts["total"] >= MOE_PARAM_BUDGET:
            raise SystemExit(
                f"upcycled total params {counts['total']} >= budget {MOE_PARAM_BUDGET}; aborting")

    model.to(device)
    if args.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()
    model.train()

    train_paths = _expand(args.train)
    mixture = Mixture(train_paths, alpha=args.alpha, seed=args.seed)

    val_paths = _expand(args.val)
    val_records = []
    for p in val_paths:
        val_records.extend(read_jsonl(p))

    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    body_params = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay},
    ])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: lr_lambda(s, args.steps, args.warmup))

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    best_val_nll = math.inf
    log_f = open(log_path, "w")
    t0 = time.time()
    tokens_seen = 0
    last_log_time = t0

    for step in range(args.steps):
        opt.zero_grad()
        step_loss = 0.0
        step_aux = 0.0
        step_tokens = 0
        for micro in range(args.grad_accum):
            records = mixture.take(args.bs)
            batch = collate(tokenizer, records, max_state_tokens=args.max_state_tokens).to(device)
            step_tokens += int(batch.attention_mask.sum().item())
            with torch.autocast(device_type="cuda", enabled=use_fp16):
                logits = model.flag_logits(batch)
                loss, n_labelled = compute_losses(logits, batch, args)
                aux = torch.tensor(0.0)
                if moe_cfg:
                    aux = moe_mod.aux_loss(model.backbone)
                    loss = loss + args.aux_weight * aux
                loss = loss / args.grad_accum
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_loss += loss.item() * args.grad_accum
            step_aux += aux.item()

        if use_fp16:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        tokens_seen += step_tokens
        avg_loss = step_loss / args.grad_accum
        avg_aux = step_aux / args.grad_accum

        if step % 10 == 0:
            now = time.time()
            dt = max(1e-9, now - last_log_time)
            rec = {"step": step, "loss": avg_loss, "lr": sched.get_last_lr()[0],
                   "tokens_per_s": step_tokens / dt}
            if moe_cfg:
                rec["aux"] = avg_aux
            print(json.dumps(rec))
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            last_log_time = now

        if args.eval_every and (step + 1) % args.eval_every == 0 and val_records:
            summary = evaluate(model, tokenizer, val_records, args, device)
            eval_path = save_dir / f"eval_step{step + 1}.json"
            with open(eval_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"-- eval @ step {step + 1} --")
            print_table(summary)
            nll = summary.get("all", {}).get("nll", math.inf)
            if nll < best_val_nll:
                best_val_nll = nll
                save_checkpoint(model, tokenizer, args, save_dir, moe_cfg)

    if val_records:
        summary = evaluate(model, tokenizer, val_records, args, device)
        eval_path = save_dir / f"eval_step{args.steps}.json"
        with open(eval_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("-- final eval --")
        print_table(summary)
        nll = summary.get("all", {}).get("nll", math.inf)
        if nll < best_val_nll:
            best_val_nll = nll
            save_checkpoint(model, tokenizer, args, save_dir, moe_cfg)

    if not (save_dir / "final").exists():
        save_checkpoint(model, tokenizer, args, save_dir, moe_cfg)

    log_f.close()


if __name__ == "__main__":
    main()
