"""Prediction diagnostics for a train_prism.py checkpoint (OJ-080).

For every source of the val set it prints the accuracy, the argmax
distribution (which option index the model picks, against the label
distribution), the mean prediction entropy (also as a fraction of ln K, where
1.0 = uniform), and the fraction of questions whose option logits all lie
within --tie of each other (1.0 = the model cannot tell the options apart).

Tiny CPU model (the checkpoint stores its base model path):
  python tools/diag_predictions.py --ckpt ckpt/sft/best --val 'data/*.val.jsonl' --cpu
Real model (a checkpoint copied from Kaggle, base loaded from the Hub or a path):
  python tools/diag_predictions.py --ckpt runs/.../ckpt/sft/best --model google/gemma-4-E2B-it
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train_prism as tp  # noqa: E402
from openjev.prism import prism_collate  # noqa: E402


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir written by train_prism.save")
    ap.add_argument("--val", default="data/*.val.jsonl")
    ap.add_argument("--model", default=None, help="base model (default: the one in prism_config.json)")
    ap.add_argument("--eval-max", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--max-state-tokens", type=int, default=1024)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--tie", type=float, default=1e-3, help="'all logits equal' tolerance")
    ap.add_argument("--top", type=int, default=5, help="argmax indices shown per source")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


def train_args(a):
    """train_prism args for build(): defaults, then the checkpoint's own settings."""
    args = tp.parse([])
    cfg = json.loads((Path(a.ckpt) / "prism_config.json").read_text())
    for k in ("lora_r", "lora_alpha", "causal"):
        if k in cfg.get("args", {}):
            setattr(args, k, cfg["args"][k])
    args.causal = cfg.get("causal", args.causal)
    args.init, args.dtype, args.cpu = a.ckpt, a.dtype, a.cpu
    args.grad_ckpt, args.ple_cpu = False, 1
    return args


@torch.no_grad()
def collect(model, tok, records, a, device):
    """Per source: list of (logits, target) for every labelled question."""
    per = defaultdict(list)
    model.eval()
    for i in range(0, len(records), a.bs):
        chunk = records[i:i + a.bs]
        b = prism_collate(tok, chunk, a.max_state_tokens).to(device)
        with tp.autocast(device, model):
            logits = model.flag_logits(b).float()
        for (l, t, _), src in zip(tp.question_views(logits, b), tp.view_sources(chunk)):
            per[src].append((l.cpu(), t))
    return per


def summarize(per, tie: float) -> dict:
    out = {}
    for src, items in sorted(per.items()):
        picks = Counter(int(l.argmax()) for l, _ in items)
        labels = Counter(t for _, t in items)
        ent, rel = [], []
        for l, _ in items:
            p = l.softmax(-1)
            h = -(p * p.clamp_min(1e-12).log()).sum().item()
            ent.append(h)
            rel.append(h / math.log(l.numel()) if l.numel() > 1 else 0.0)
        out[src] = {
            "n": len(items),
            "acc": sum(int(l.argmax()) == t for l, t in items) / len(items),
            "argmax": dict(picks.most_common()),
            "labels": dict(labels.most_common()),
            "entropy": sum(ent) / len(ent),
            "entropy_rel": sum(rel) / len(rel),
            "all_tied": sum(float(l.max() - l.min()) < tie for l, _ in items) / len(items),
            "logit_spread": sum(float(l.max() - l.min()) for l, _ in items) / len(items),
        }
    return out


def report(rows: dict, top: int) -> None:
    print(f"{'source':40s} {'n':>4s} {'acc':>6s} {'H':>6s} {'H/lnK':>6s} {'tied':>6s} {'spread':>7s}"
          "  argmax (idx:count) | labels")
    for src, r in rows.items():
        am = " ".join(f"{k}:{v}" for k, v in list(r["argmax"].items())[:top])
        lb = " ".join(f"{k}:{v}" for k, v in list(r["labels"].items())[:top])
        print(f"{src[:40]:40s} {r['n']:4d} {r['acc']:6.3f} {r['entropy']:6.3f} {r['entropy_rel']:6.3f} "
              f"{r['all_tied']:6.3f} {r['logit_spread']:7.4f}  {am} | {lb}")
    n = sum(r["n"] for r in rows.values())
    if n:
        def avg(k):
            return sum(r[k] * r["n"] for r in rows.values()) / n
        print(f"{'ALL':40s} {n:4d} {avg('acc'):6.3f} {avg('entropy'):6.3f} {avg('entropy_rel'):6.3f} "
              f"{avg('all_tied'):6.3f} {avg('logit_spread'):7.4f}")


def main(argv=None):
    a = parse(argv)
    args = train_args(a)
    device = tp.pick_device(args)
    model, tok, _ = tp.build(args, device, base=a.model)
    val = tp.load_val(a.val, a.eval_max, a.seed)
    if not val:
        raise SystemExit(f"no val records match {a.val}")
    rows = summarize(collect(model, tok, val, a, device), a.tie)
    report(rows, a.top)
    return rows


if __name__ == "__main__":
    main()
