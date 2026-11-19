"""OpenJev-RLCD trainer (PLAN.md section 4).

Starts from an SFT checkpoint and trains against a batch calibration reward
(proper score + soft-ECE + KL to a frozen reference) plus a contrastive
consistency loss built from label-preserving "twins" of real records
(openjev/contrast.py). See PLAN.md section 4 for the design.

CPU-only smoke test (no GPU on this machine):

    python train.py --train "tests/fixtures/*.jsonl" --val "tests/fixtures/*.jsonl" \\
        --model trl-internal-testing/tiny-Qwen3ForCausalLM --steps 4 --bs 4 --cpu \\
        --eval-every 0 --save-dir /tmp/sft
    python rlcd_train.py --ckpt /tmp/sft/final \\
        --train "tests/fixtures/*.jsonl" --val "tests/fixtures/*.jsonl" \\
        --steps 4 --bs 4 --cpu --eval-every 2 --save-dir /tmp/rlcd
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from openjev import checkpoint, contrast, question_loss
from openjev.collate import collate
from openjev.data import read_jsonl
from openjev.metrics import Accumulator, ece_from_conf
from openjev.mix import Mixture
from openjev.model import target_index
from openjev.rlcd import soft_ece_from_conf


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train", nargs="+", default=["data/*.train.jsonl"])
    ap.add_argument("--val", nargs="+", default=["data/*.val.jsonl"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--lam-cal", type=float, default=1.0)
    ap.add_argument("--lam-kl", type=float, default=0.05)
    ap.add_argument("--lam-con", type=float, default=0.5)
    ap.add_argument("--twin-frac", type=float, default=0.5)
    ap.add_argument("--twin-kinds", default="swap,reverse,shuffle,drop")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--save-dir", default="checkpoints/rlcd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-max", type=int, default=2000)
    ap.add_argument("--max-state-tokens", type=int, default=1024)
    return ap.parse_args(argv)


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


def build_batch_with_twins(tokenizer, records, rng: random.Random, twin_frac: float,
                            twin_kinds: tuple[str, ...], max_state_tokens: int):
    """originals + up to one twin per selected record, one collate call.

    Returns (batch, twin_meta) where twin_meta is a list of dicts:
    {"orig_i": index of the source record in the batch, "q_idx": schema question
    index that was transformed, "twin_i": index of the twin record in the batch,
    "mapping": the Twin's mapping}.
    """
    n = len(records)
    n_twins = int(round(n * twin_frac))
    candidate_idx = list(range(n))
    rng.shuffle(candidate_idx)
    candidate_idx = candidate_idx[:n_twins]

    twin_records = []
    twin_meta = []
    for orig_i in candidate_idx:
        twins = contrast.make_twins(records[orig_i], rng, kinds=twin_kinds)
        if not twins:
            continue
        twin = rng.choice(twins)
        q_idx = next(i for i, q in enumerate(twin.record.schema.questions)
                     if q.name == twin.question)
        twin_i = n + len(twin_records)
        twin_records.append(twin.record)
        twin_meta.append({"orig_i": orig_i, "q_idx": q_idx, "twin_i": twin_i,
                           "mapping": twin.mapping})

    all_records = list(records) + twin_records
    batch = collate(tokenizer, all_records, max_state_tokens=max_state_tokens)
    return batch, twin_meta


def _labelled_question_slices(batch):
    """Yield (ex_i, q_idx, group, target) for every labelled question in batch."""
    for i, (ex_groups, ex_labels, schema) in enumerate(zip(batch.groups, batch.labels,
                                                             batch.schemas)):
        for q_idx, (g, y, q) in enumerate(zip(ex_groups, ex_labels, schema.questions)):
            if y is None:
                continue
            yield i, q_idx, g, target_index(q, y), q


def rlcd_step(policy, ref, tokenizer, records, rng, args, device):
    kinds = tuple(args.twin_kinds.split(","))
    batch, twin_meta = build_batch_with_twins(tokenizer, records, rng, args.twin_frac,
                                               kinds, args.max_state_tokens)
    batch = batch.to(device)

    policy_logits = policy.flag_logits(batch)
    with torch.no_grad():
        ref_logits = ref.flag_logits(batch)

    proper_losses = []
    conf_list, correct_list, type_list = [], [], []
    kl_list = []
    policy_slices = {}  # (ex_i, q_idx) -> logits slice, for the consistency loss

    for ex_i, q_idx, g, k, q in _labelled_question_slices(batch):
        pl = policy_logits[g["start"]:g["end"]]
        rl = ref_logits[g["start"]:g["end"]]
        policy_slices[(ex_i, q_idx)] = pl

        pred = {"type": q.type, "name": q.name, "logits": pl}
        l = question_loss(pred, q, _label_from_index(q, k))
        if l is not None:
            proper_losses.append(l)

        p = torch.softmax(pl, -1)
        conf, pred_idx = p.max(-1)
        conf_list.append(conf)
        correct_list.append((pred_idx == k).float())
        type_list.append(q.type)

        logp = torch.log_softmax(pl, -1)
        ref_logp = torch.log_softmax(rl, -1)
        kl_list.append((p * (logp - ref_logp)).sum())

    con_losses = []
    for tm in twin_meta:
        key_orig = (tm["orig_i"], tm["q_idx"])
        key_twin = (tm["twin_i"], tm["q_idx"])
        if key_orig not in policy_slices or key_twin not in policy_slices:
            continue
        p_orig = torch.softmax(policy_slices[key_orig], -1).detach()
        mapped = contrast.map_probs(tm["mapping"], p_orig)
        con_losses.append(contrast.consistency_loss(policy_slices[key_twin], mapped))

    proper = torch.stack(proper_losses).mean() if proper_losses else policy_logits.sum() * 0.0
    if conf_list:
        conf_t = torch.stack(conf_list)
        correct_t = torch.stack(correct_list)
        # One soft ECE per question type, then the mean: noul confidences start at 0.5
        # and wide choices sit low, so one shared set of bins lets one type hide another.
        cal = torch.stack([
            soft_ece_from_conf(conf_t[idx], correct_t[idx], bins=args.bins)
            for t in sorted(set(type_list))
            for idx in [torch.tensor([i for i, x in enumerate(type_list) if x == t],
                                     device=conf_t.device)]
        ]).mean()
        acc = correct_t.mean().item()
        hard_ece = ece_from_conf(conf_t.detach().tolist(), correct_t.detach().tolist(),
                                  bins=args.bins)
    else:
        cal = policy_logits.sum() * 0.0
        acc = 0.0
        hard_ece = 0.0
    kl = torch.stack(kl_list).mean() if kl_list else policy_logits.sum() * 0.0
    con = torch.stack(con_losses).mean() if con_losses else policy_logits.sum() * 0.0

    loss = proper + args.lam_cal * cal + args.lam_kl * kl + args.lam_con * con
    parts = {"proper": proper.detach().item(), "cal": cal.detach().item(),
             "kl": kl.detach().item(), "con": con.detach().item(),
             "acc": acc, "ece": hard_ece,
             "n_labelled": len(proper_losses), "n_twins": len(con_losses)}
    return loss, parts


def _label_from_index(q, k):
    """Invert target_index: turn a flag index back into a label question_loss accepts."""
    if q.type == "noul":
        return k == 0
    if q.type == "choice":
        return q.options[k]
    return k


@torch.no_grad()
def evaluate(policy, tokenizer, val_records, args, device) -> dict:
    policy.eval()
    acc = Accumulator()
    n = 0
    for i in range(0, min(len(val_records), args.eval_max), args.bs):
        chunk = val_records[i:i + args.bs]
        batch = collate(tokenizer, chunk, max_state_tokens=args.max_state_tokens).to(device)
        logits = policy.flag_logits(batch)
        for ex_groups, ex_labels, schema, record in zip(batch.groups, batch.labels,
                                                          batch.schemas, chunk):
            for g, y, q in zip(ex_groups, ex_labels, schema.questions):
                if y is None:
                    continue
                k = target_index(q, y)
                probs = torch.softmax(logits[g["start"]:g["end"]], -1)
                acc.add(q.type, record.source, probs, k)
                n += 1
        if n >= args.eval_max:
            break
    policy.train()
    return acc.summary()


@torch.no_grad()
def contrast_consistency(policy, tokenizer, val_records, args, device, seed: int) -> float:
    """Mean symmetric KL over a fixed (seeded) set of val twins."""
    rng = random.Random(seed)
    kinds = tuple(args.twin_kinds.split(","))
    sample = list(val_records)
    rng.shuffle(sample)
    sample = sample[:min(len(sample), args.eval_max // 2 or 1, args.bs * 4)]
    if not sample:
        return 0.0
    policy.eval()
    batch, twin_meta = build_batch_with_twins(tokenizer, sample, rng, twin_frac=1.0,
                                               twin_kinds=kinds,
                                               max_state_tokens=args.max_state_tokens)
    if not twin_meta:
        policy.train()
        return 0.0
    batch = batch.to(device)
    logits = policy.flag_logits(batch)
    slices = {}
    for i, (ex_groups, schema) in enumerate(zip(batch.groups, batch.schemas)):
        for q_idx, g in enumerate(ex_groups):
            slices[(i, q_idx)] = logits[g["start"]:g["end"]]
    vals = []
    for tm in twin_meta:
        key_orig = (tm["orig_i"], tm["q_idx"])
        key_twin = (tm["twin_i"], tm["q_idx"])
        if key_orig not in slices or key_twin not in slices:
            continue
        p_orig = torch.softmax(slices[key_orig], -1)
        mapped = contrast.map_probs(tm["mapping"], p_orig)
        vals.append(contrast.consistency_loss(slices[key_twin], mapped).item())
    policy.train()
    return sum(vals) / len(vals) if vals else 0.0


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "log.jsonl"

    policy, tokenizer, sft_config = checkpoint.load(args.ckpt, device=device)
    ref, _, _ = checkpoint.load(args.ckpt, device=device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    policy.train()

    train_paths = _expand(args.train)
    mixture = Mixture(train_paths, seed=args.seed)
    val_paths = _expand(args.val)
    val_records = []
    for p in val_paths:
        val_records.extend(read_jsonl(p))

    rng = random.Random(args.seed)

    head_params = [p for n, p in policy.named_parameters() if not n.startswith("backbone.")]
    body_params = [p for n, p in policy.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ])

    log_f = open(log_path, "w")
    t0 = time.time()

    for step in range(args.steps):
        records = mixture.take(args.bs)
        opt.zero_grad()
        loss, parts = rlcd_step(policy, ref, tokenizer, records, rng, args, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if step % 10 == 0:
            rec = {"step": step, "loss": loss.item(), **parts}
            print(json.dumps(rec))
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

        if args.eval_every and (step + 1) % args.eval_every == 0 and val_records:
            summary = evaluate(policy, tokenizer, val_records, args, device)
            con = contrast_consistency(policy, tokenizer, val_records, args, device, args.seed)
            print(f"-- eval @ step {step + 1} -- contrast_consistency={con:.4f}")
            eval_rec = {"step": step + 1, "eval": summary, "contrast_consistency": con}
            log_f.write(json.dumps(eval_rec) + "\n")
            log_f.flush()

    if val_records:
        summary = evaluate(policy, tokenizer, val_records, args, device)
        con = contrast_consistency(policy, tokenizer, val_records, args, device, args.seed)
        print(f"-- final eval -- contrast_consistency={con:.4f}")
        eval_rec = {"step": args.steps, "eval": summary, "contrast_consistency": con}
        log_f.write(json.dumps(eval_rec) + "\n")
        log_f.flush()

    log_f.close()

    out_config = dict(sft_config)
    out_config["args"] = vars(args)
    temperatures = dict(sft_config.get("temperatures", {}))
    out_config["temperatures_stale"] = True
    checkpoint.save(policy, tokenizer, save_dir / "final", out_config,
                     temperatures=temperatures or None)


if __name__ == "__main__":
    main()
