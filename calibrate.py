"""Fit one temperature per question type on a validation split and store it in
the checkpoint's openjev_config.json.

    python calibrate.py --ckpt checkpoints/run/final --val "data/*.val.jsonl" [--max N]

For each question type with a temperature-scaling fit, prints ECE/NLL before and
after scaling. Types with fewer than 20 labelled items keep T=1.0 (a warning is
printed instead of fitting, since LBFGS on that few points is unreliable).
"""

from __future__ import annotations

import argparse
import glob
import json

import torch

from openjev import checkpoint
from openjev.data import read_jsonl
from openjev.rlcd import ece, fit_temperature

MIN_ITEMS = 20
# A fit outside this range means the model has no usable signal (for example an
# untrained model), and a huge T would silently flatten every answer.
T_MIN, T_MAX = 0.05, 20.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-state-tokens", type=int, default=1024)
    return ap.parse_args(argv)


def _expand(patterns: list[str]) -> list[str]:
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        raise ValueError(f"no files matched {patterns!r}")
    return paths


def _pad_stack(items: list[tuple[torch.Tensor, int, str]]):
    logits = [lg for lg, _, _ in items]
    targets = torch.tensor([k for _, k, _ in items], dtype=torch.long)
    k = max(t.numel() for t in logits)
    padded = torch.full((len(logits), k), -1e9)
    for i, t in enumerate(logits):
        padded[i, :t.numel()] = t
    return padded, targets


def _nll(probs: torch.Tensor, targets: torch.Tensor) -> float:
    p = probs[torch.arange(probs.shape[0]), targets].clamp_min(1e-12)
    return (-torch.log(p)).mean().item()


def main(argv=None):
    args = parse_args(argv)

    model, tok, config = checkpoint.load(args.ckpt)

    val_paths = _expand(args.val)
    records = []
    for p in val_paths:
        records.extend(read_jsonl(p))
    if args.max:
        records = records[:args.max]

    by_type = checkpoint.collect_logits(model, tok, records, bs=args.bs,
                                         max_state_tokens=args.max_state_tokens)

    temperatures = {}
    for qtype in ("noul", "choice", "score"):
        items = by_type.get(qtype, [])
        if len(items) < MIN_ITEMS:
            print(f"[{qtype}] only {len(items)} labelled items (< {MIN_ITEMS}); "
                  f"keeping T=1.0")
            temperatures[qtype] = 1.0
            continue

        logits, targets = _pad_stack(items)
        probs_before = torch.softmax(logits, -1)
        ece_before = ece(probs_before, targets)
        nll_before = _nll(probs_before, targets)

        t = fit_temperature(logits, targets)
        if not T_MIN <= t <= T_MAX:
            print(f"[{qtype}] fitted T={t:.4g} is outside [{T_MIN}, {T_MAX}]; clamped. "
                  "Check the model: it may be untrained.")
            t = min(max(t, T_MIN), T_MAX)
        probs_after = torch.softmax(logits / t, -1)
        ece_after = ece(probs_after, targets)
        nll_after = _nll(probs_after, targets)

        temperatures[qtype] = t
        print(f"[{qtype}] n={len(items)} T={t:.4f}  "
              f"ECE {ece_before:.4f} -> {ece_after:.4f}  "
              f"NLL {nll_before:.4f} -> {nll_after:.4f}")

    config["temperatures"] = temperatures
    with open(f"{args.ckpt}/openjev_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"wrote temperatures {temperatures} to {args.ckpt}/openjev_config.json")


if __name__ == "__main__":
    main()
