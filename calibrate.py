"""Calibration baselines (OJ-066): fit on a validation split, apply to a test split.

Two post-hoc maps, each fitted per question type (noul / choice / score) on val only:
  temperature  p = softmax(z / T)                                    (Guo et al. 2017)
  platt        p_i = sigmoid(a z_i + b) / sum_j sigmoid(a z_j + b)   on the flag logits
               z (Platt per flag, renormalised over the options; a > 0, so the argmax
               and acc never change; b -> -inf is temperature scaling with T = 1/a)
Labels are only read, never changed; unlabelled questions are skipped.

    # Prism checkpoint (prism_config.json), runs the model:
    python calibrate.py --ckpt out/sft/best --val "data/*.val.jsonl" --test "data/*.test.jsonl" \\
        --max-per-source 1000 --out runs/final_eval/sft/calib
    # from eval_prism.py dumps, no model:
    python calibrate.py --val-items runs/final_eval/sft/val/items.jsonl \\
        --test-items runs/final_eval/sft/test/items.jsonl --out runs/final_eval/sft/calib
    # old OpenJev checkpoint (openjev_config.json): as before, the temperatures are also
    # written into its openjev_config.json
    python calibrate.py --ckpt checkpoints/run/final --val "data/*.val.jsonl" [--max N]

Writes <out>/calibration.json and <out>/calibration.md: per source and overall n, acc
(+-2 SE), ece / nll / brier raw, after T and after Platt, on val (the fit split) and test.
A Prism checkpoint dir is never written to (Kaggle inputs are read-only). Types with
fewer than 20 labelled val items keep the raw softmax (LBFGS on that few points is
unreliable).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from openjev.metrics import source_metrics
from openjev.rlcd import fit_temperature

MIN_ITEMS = 20
# A fit outside this range means the model has no usable signal (for example an
# untrained model), and a huge T would silently flatten every answer.
T_MIN, T_MAX = 0.05, 20.0
TYPES = ("noul", "choice", "score")
MAPS = ("raw", "temperature", "platt")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="checkpoint dir (Prism prism_config.json or openjev_config.json)")
    ap.add_argument("--val", nargs="+", help="val jsonl files or globs (with --ckpt)")
    ap.add_argument("--test", nargs="+", help="test jsonl files or globs (with --ckpt)")
    ap.add_argument("--val-items", help="eval_prism.py items.jsonl of val (no model)")
    ap.add_argument("--test-items", help="eval_prism.py items.jsonl of test (no model)")
    ap.add_argument("--out", help="write calibration.json and calibration.md here")
    ap.add_argument("--max", type=int, default=None, help="first N records of each split")
    ap.add_argument("--max-per-source", type=int, default=0,
                    help="Prism: seeded subset of <= N records per source (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-batch-tokens", type=int, default=16384, help="Prism only")
    ap.add_argument("--max-state-tokens", type=int, default=None,
                    help="default: the Prism checkpoint's training value, else 1024")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    if a.val_items:
        if a.ckpt or a.val or a.test:
            ap.error("--val-items/--test-items exclude --ckpt/--val/--test")
    elif not (a.ckpt and a.val):
        ap.error("give --ckpt and --val, or --val-items")
    elif a.test_items:
        ap.error("--test-items needs --val-items")
    return a


# ---------------------------------------------------------------- maps

def _pad(items: list[dict], fill: float):
    k = max(len(it["logits"]) for it in items)
    z = torch.full((len(items), k), fill, dtype=torch.float64)
    mask = torch.zeros((len(items), k), dtype=torch.bool)
    for i, it in enumerate(items):
        n = len(it["logits"])
        z[i, :n] = torch.tensor(it["logits"], dtype=torch.float64)
        mask[i, :n] = True
    return z, mask, torch.tensor([it["target"] for it in items], dtype=torch.long)


def platt_logp(z: torch.Tensor, mask: torch.Tensor, a, b) -> torch.Tensor:
    """log p of the renormalised per-flag Platt map; padded flags (mask False) get -inf."""
    s = F.logsigmoid(a * z + b).masked_fill(~mask, float("-inf"))
    return s - s.logsumexp(-1, keepdim=True)


def fit_platt(z: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor, steps: int = 500):
    """(a, b) minimising val nll of the Platt map; a = exp(log a) > 0. fp64: with a small a,
    sigmoid(a z + b) of nearby flags would tie in fp32 and change the argmax."""
    log_a = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_a, b], lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.nll_loss(platt_logp(z, mask, log_a.exp(), b), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return log_a.exp().item(), b.item()


def apply(item: dict, fit: dict, kind: str) -> torch.Tensor:
    """Probabilities (fp64) of one item under map `kind` (raw, temperature, platt)."""
    z = torch.tensor(item["logits"], dtype=torch.float64)
    if kind == "temperature":
        return (z / fit["temperature"].get(item["type"], 1.0)).softmax(-1)
    if kind == "platt" and fit["platt"].get(item["type"]):
        a, b = fit["platt"][item["type"]]
        return platt_logp(z, torch.ones_like(z, dtype=torch.bool), a, b).exp()
    return z.softmax(-1)


def fit_calibration(items: list[dict], log=print) -> dict:
    """Per question type: temperature T (1.0 when < MIN_ITEMS) and Platt (a, b) (None =
    raw softmax when < MIN_ITEMS), fitted on `items` (the val split)."""
    fit: dict = {"temperature": {}, "platt": {}, "n": {}}
    for qtype in TYPES:
        sub = [it for it in items if it["type"] == qtype]
        fit["n"][qtype] = len(sub)
        if len(sub) < MIN_ITEMS:
            if sub:
                log(f"[{qtype}] only {len(sub)} labelled items (< {MIN_ITEMS}); keeping T=1.0 "
                    "and the raw softmax")
            fit["temperature"][qtype], fit["platt"][qtype] = 1.0, None
            continue
        z, mask, y = _pad(sub, -1e9)
        t = fit_temperature(z, y)
        if not T_MIN <= t <= T_MAX:
            log(f"[{qtype}] fitted T={t:.4g} is outside [{T_MIN}, {T_MAX}]; clamped. "
                "Check the model: it may be untrained.")
            t = min(max(t, T_MIN), T_MAX)
        fit["temperature"][qtype] = t
        a, b = fit_platt(z.masked_fill(~mask, 0.0), mask, y)
        fit["platt"][qtype] = [a, b]
        rows = {k: source_metrics([(apply(it, fit, k), it["target"], it["source"]) for it in sub])
                ["overall"] for k in MAPS}
        log(f"[{qtype}] n={len(sub)} T={t:.4f} platt a={a:.4f} b={b:.4f}  "
            f"ECE {rows['raw']['ece']:.4f} -> T {rows['temperature']['ece']:.4f} / "
            f"Platt {rows['platt']['ece']:.4f}  NLL {rows['raw']['nll']:.4f} -> "
            f"T {rows['temperature']['nll']:.4f} / Platt {rows['platt']['nll']:.4f}")
    return fit


def report(items: list[dict], fit: dict) -> dict:
    """{map: {"overall": row, "sources": {source: row}}} for raw, temperature, platt."""
    return {k: source_metrics([(apply(it, fit, k), it["target"], it["source"]) for it in items], nd=4)
            for k in MAPS}


def to_markdown(split: str, rep: dict) -> str:
    lines = [f"## {split}", "",
             "| source | n | acc | ±2 SE | ece raw | ece T | ece Platt | nll raw | nll T | nll Platt |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names = sorted(rep["raw"]["sources"]) + [None]
    for s in names:
        r = {k: (rep[k]["sources"][s] if s else rep[k]["overall"]) for k in MAPS}
        lines.append(f"| {s or '**overall**'} | {r['raw']['n']} | {r['raw']['acc']:.4f} "
                     f"| {r['raw']['acc_se2']:.4f} | "
                     + " | ".join(f"{r[k]['ece']:.4f}" for k in MAPS) + " | "
                     + " | ".join(f"{r[k]['nll']:.4f}" for k in MAPS) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- data

def _records(patterns, args) -> list:
    from eval_prism import expand, select
    from openjev.data import read_jsonl
    recs = [r for p in expand(patterns) for r in read_jsonl(p)]
    recs = select(recs, args.max_per_source, args.seed)
    return recs[:args.max] if args.max else recs


def _openjev_items(model, tok, records, args) -> list[dict]:
    from openjev import checkpoint
    by_type = checkpoint.collect_logits(model, tok, records, bs=args.bs,
                                        max_state_tokens=args.max_state_tokens or 1024)
    return [{"type": qt, "source": s, "target": k, "logits": lg.tolist()}
            for qt, rows in by_type.items() for lg, k, s in rows]


def main(argv=None):
    args = parse_args(argv)
    config = None
    if args.val_items:
        from eval_prism import read_items
        val = read_items(args.val_items)
        test = read_items(args.test_items) if args.test_items else None
    else:
        import eval_prism as ep
        if ep.is_prism(args.ckpt):
            ep.single_process_only()
            device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
            model, tok, margs = ep.load_prism(args.ckpt, device, args.dtype)
            mst = args.max_state_tokens or margs.max_state_tokens

            def run(patterns):
                return ep.collect(model, tok, _records(patterns, args), mst, device, args.bs,
                                  args.max_batch_tokens, log_every=50)
        else:
            from openjev import checkpoint
            model, tok, config = checkpoint.load(args.ckpt)

            def run(patterns):
                return _openjev_items(model, tok, _records(patterns, args), args)
        val = run(args.val)
        test = run(args.test) if args.test else None

    fit = fit_calibration(val)
    if config is not None:          # old OpenJev checkpoint: store the temperatures, as before
        config["temperatures"] = fit["temperature"]
        with open(f"{args.ckpt}/openjev_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"wrote temperatures {fit['temperature']} to {args.ckpt}/openjev_config.json")

    res = {"fit": fit, "val": report(val, fit)}
    md = ["# Calibration baselines (fit on val, applied to test)", "",
          "Per type: " + ", ".join(
              f"{t}: n={fit['n'][t]} T={fit['temperature'][t]:.4f} "
              + (f"Platt a={fit['platt'][t][0]:.4f} b={fit['platt'][t][1]:.4f}"
                 if fit["platt"][t] else "Platt -")
              for t in TYPES if fit["n"][t]),
          "", to_markdown("val (fit split)", res["val"])]
    if test is not None:
        res["test"] = report(test, fit)
        md += ["", to_markdown("test", res["test"])]
    md = "\n".join(md) + "\n"
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "calibration.json").write_text(json.dumps(res, indent=2))
        (out / "calibration.md").write_text(md)
        print(f"wrote {out / 'calibration.json'} and {out / 'calibration.md'}")
    print(md)
    return res


if __name__ == "__main__":
    main()
