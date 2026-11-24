"""Final eval of a Prism checkpoint (Objective A, docs/FINAL_RUN_PLAN.md §4).

Per source and overall: acc with a 2-SE interval, nll, 15-bin equal-mass ECE and Brier
(openjev.metrics.prob_metrics, the same numbers train_prism.evaluate logs).

    CUDA_VISIBLE_DEVICES=0 python eval_prism.py --ckpt out/sft/best \\
        --data "data/*.val.jsonl" --max-per-source 0 --out runs/final_eval/sft/val
    CUDA_VISIBLE_DEVICES=0 python eval_prism.py --ckpt out/sft/best \\
        --data "data/*.test.jsonl" --out runs/final_eval/sft/test        # 1000 per source
    python eval_prism.py --compare runs/final_eval/sft/test/items.jsonl \\
        runs/final_eval/rl/test/items.jsonl --out runs/final_eval/rl_vs_sft

Writes <out>/metrics.json, <out>/metrics.md and <out>/items.jsonl (one line per labelled
question: id, source, type, name, target, flag logits). calibrate.py --val-items /
--test-items fits temperature / Platt from two items.jsonl files without the model.
--compare gives the paired difference B - A on the shared items (acc with a paired 2-SE,
ece, nll), per source and overall.

One process on one device, fp16 on a T4 (--dtype auto), batches grouped by encoded length
(longest first, so an OOM shows in the first batch). It refuses to run under torchrun
(WORLD_SIZE > 1): it is not part of the DDP training job.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from openjev.data import read_jsonl
from openjev.metrics import source_metrics
from openjev.model import target_index
from openjev.prism import PLE_ROWS


def is_prism(ckpt) -> bool:
    return (Path(ckpt) / "prism_config.json").exists()


def _ple_table(model):
    f = getattr(model._text_backbone(), "get_per_layer_input_embeddings", None)
    return f() if f else None


def load_prism(ckpt, device: str = "cpu", dtype: str = "auto", ple_cpu: int = 1):
    """(model, tok, args) of a train_prism.py checkpoint: base model from prism_config.json,
    LoRA adapter, flag head, template embeddings and template delta, through
    train_prism.build. args = train_prism defaults overlaid with the saved args, so
    max_state_tokens, causal and lora_r match the training run.

    Gemma 4 has no spare embedding rows, so adding the template tokens resizes the Per-Layer
    Embedding (PLE) table too, and the template tokens read their own (frozen) PLE rows.
    A fresh SFT build draws them in add_special_tokens (mean + seeded noise); save() writes
    them to ple_rows.pt and build(init=ckpt) pastes them back after its resize (exact, any
    seed). Paths, seeded with the run's --seed like train_prism.main:
      - <ckpt>/ple_rows.pt exists (every checkpoint saved since the fix), or a full
        fine-tune (lora_r 0, the saved backbone holds the whole PLE table): build(init=ckpt);
      - OLD checkpoints only (no ple_rows.pt, warns): trained from scratch -> replay the
        fresh build, then load the adapter, flag head, template rows and delta; trained with
        --init (RL) -> build(init=ckpt) redraws the rows the run drew. Both replays rely on
        the same RNG use between the seed and the resize as in training (tested on a tiny
        Gemma 4, CPU)."""
    import warnings

    import train_prism as tp
    from openjev.schema import SPECIAL_TOKENS
    ckpt = Path(ckpt)
    cfg = json.loads((ckpt / "prism_config.json").read_text())
    args = tp.parse([])
    for k, v in (cfg.get("args") or {}).items():
        setattr(args, k, v)
    trained_init = args.init
    args.grad_ckpt, args.dtype = False, dtype
    args.cpu, args.ple_cpu = device == "cpu", ple_cpu
    args.causal = bool(cfg.get("causal", args.causal))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    has_rows = (ckpt / PLE_ROWS).exists()
    if not has_rows and args.lora_r > 0:
        warnings.warn(f"{ckpt} has no {PLE_ROWS} (saved before the PLE persistence fix): "
                      "replaying the seeded build to rebuild the template PLE rows", stacklevel=2)
    if has_rows or args.lora_r == 0 or trained_init:
        args.init = str(ckpt)
        with warnings.catch_warnings():        # build's no-ple_rows warning: replay intended
            warnings.filterwarnings("ignore", message=f".*has no {PLE_ROWS}")
            model, tok, _ = tp.build(args, device, base=str(ckpt) if args.lora_r == 0 else None)
    else:
        from peft.utils import load_peft_weights, set_peft_model_state_dict
        from transformers import AutoTokenizer
        args.init = None
        model, fresh_tok, _ = tp.build(args, device)
        tok = AutoTokenizer.from_pretrained(ckpt)
        if fresh_tok.convert_tokens_to_ids(SPECIAL_TOKENS) != tok.convert_tokens_to_ids(SPECIAL_TOKENS):
            raise ValueError(f"{ckpt}: template token ids differ from a fresh build of the base")
        res = set_peft_model_state_dict(model.backbone, load_peft_weights(str(ckpt), device="cpu"))
        missing = [k for k in res.unexpected_keys if "lora_" in k]
        if missing:
            raise ValueError(f"{ckpt}: adapter keys not loaded: {missing[:5]}")
        dev = model.flag_head.weight.device
        model.flag_head.load_state_dict(torch.load(ckpt / "heads.pt", map_location=dev))
        w = model.get_input_embeddings().weight
        with torch.no_grad():
            for i, v in torch.load(ckpt / "embeddings.pt", map_location="cpu").items():
                w[int(i)] = v.to(w.dtype)
            model.template_delta.copy_(torch.load(ckpt / "template_delta.pt", map_location="cpu"))
    args.init = str(ckpt)
    model.eval()
    return model, tok, args


def expand(patterns) -> list[str]:
    paths = [p for pat in patterns for p in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no files matched {patterns!r}")
    return paths


def select(records, max_per_source: int, seed: int = 0) -> list:
    """At most max_per_source records per source (0 = all): a seeded random subset of each
    source, kept in file order. Rows and labels are never changed."""
    if not max_per_source:
        return list(records)
    by: dict = {}
    for i, r in enumerate(records):
        by.setdefault(r.source, []).append(i)
    keep = set()
    for src, idx in by.items():
        idx = list(idx)
        random.Random(f"{seed}:{src}").shuffle(idx)
        keep.update(idx[:max_per_source])
    return [r for i, r in enumerate(records) if i in keep]


def length_batches(lengths: list[int], bs: int, max_tokens: int) -> list[list[int]]:
    """Indices sorted by length (longest first), cut into batches of <= bs records and
    <= max_tokens padded tokens (0 = no token cap). A record longer than max_tokens gets
    its own batch."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) >= bs or (max_tokens and (len(cur) + 1) * lengths[cur[0]] > max_tokens)):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def collect(model, tok, records, max_state_tokens: int, device: str, bs: int = 32,
            max_tokens: int = 16384, log_every: int = 0) -> list[dict]:
    """Flag logits of every labelled question, in record order: dicts with id, source,
    type, name, target, logits (a list). Unlabelled questions are skipped, as in
    train_prism.question_views."""
    import train_prism as tp
    from openjev.packing import request_lengths
    from openjev.prism import prism_collate
    model.eval()
    lens = request_lengths(tok, records, max_state_tokens)
    per_rec: list = [None] * len(records)
    batches = length_batches(lens, bs, max_tokens)
    t0 = time.time()
    for k, idx in enumerate(batches):
        chunk = [records[i] for i in idx]
        b = prism_collate(tok, chunk, max_state_tokens).to(device)
        with tp.autocast(device, model):
            logits = model.flag_logits(b)
        logits = logits.float().cpu()
        for i, r, groups, schema in zip(idx, chunk, b.groups, b.schemas):
            rows = []
            for g, y, q in zip(groups, r.answers, schema.questions):
                t = target_index(q, y)
                if t is None:
                    continue
                rows.append({"id": r.id, "source": r.source, "type": g["type"], "name": q.name,
                             "target": t, "logits": logits[g["start"]:g["end"]].tolist()})
            per_rec[i] = rows
        if log_every and (k % log_every == 0 or k == len(batches) - 1):
            print(json.dumps({"batch": k + 1, "of": len(batches), "tokens": b.input_ids.shape[1],
                              "sec": round(time.time() - t0)}), flush=True)
    return [it for rows in per_rec for it in rows]


def probs(item: dict) -> torch.Tensor:
    return torch.tensor(item["logits"], dtype=torch.float32).softmax(-1)


def metrics(items: list[dict], nd: int | None = 4) -> dict:
    return source_metrics([(probs(it), it["target"], it["source"]) for it in items], nd=nd)


def write_items(path, items: list[dict]) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def read_items(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_markdown(title: str, m: dict) -> str:
    lines = [f"## {title}", "", "| source | n | acc | ±2 SE | nll | ece | brier |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    rows = sorted(m["sources"].items()) + [("**overall**", m["overall"])]
    for s, r in rows:
        lines.append(f"| {s} | {r['n']} | {r['acc']:.4f} | {r['acc_se2']:.4f} | {r['nll']:.4f} "
                     f"| {r['ece']:.4f} | {r['brier']:.4f} |")
    return "\n".join(lines)


def compare(a: list[dict], b: list[dict]) -> dict:
    """Paired B - A on the questions both dumps share (key: source, id, name). Per source
    and overall: acc_a, acc_b, d_acc with the paired 2-SE (SE of the per-question
    difference of correctness), and ece / nll of each side on the same items."""
    key = lambda it: (it["source"], it["id"], it["name"])   # noqa: E731
    bmap = {key(it): it for it in b}
    pairs = [(x, bmap[key(x)]) for x in a if key(x) in bmap]
    by: dict = {}
    for x, y in pairs:
        if x["target"] != y["target"]:
            raise ValueError(f"target differs for {key(x)}: the dumps are not the same data")
        by.setdefault(x["source"], []).append((x, y))

    def row(ps):
        d = [float(probs(y).argmax() == y["target"]) - float(probs(x).argmax() == x["target"])
             for x, y in ps]
        n = len(d)
        mean = sum(d) / n
        var = sum((v - mean) ** 2 for v in d) / max(n - 1, 1)
        ma, mb = metrics([x for x, _ in ps])["overall"], metrics([y for _, y in ps])["overall"]
        return {"n": n, "acc_a": ma["acc"], "acc_b": mb["acc"], "d_acc": round(mean, 4),
                "d_acc_se2": round(2 * math.sqrt(var / n), 4), "ece_a": ma["ece"], "ece_b": mb["ece"],
                "nll_a": ma["nll"], "nll_b": mb["nll"]}
    return {"n_a": len(a), "n_b": len(b), "n_paired": len(pairs),
            "overall": row(pairs) if pairs else {},
            "sources": {s: row(ps) for s, ps in sorted(by.items())}}


def compare_markdown(c: dict) -> str:
    lines = ["# Paired comparison (B - A)", "",
             "| source | n | acc A | acc B | Δacc | ±2 SE (paired) | ece A | ece B | nll A | nll B |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, r in list(c["sources"].items()) + [("**overall**", c["overall"])]:
        if r:
            lines.append(f"| {s} | {r['n']} | {r['acc_a']:.4f} | {r['acc_b']:.4f} | {r['d_acc']:+.4f} "
                         f"| {r['d_acc_se2']:.4f} | {r['ece_a']:.4f} | {r['ece_b']:.4f} "
                         f"| {r['nll_a']:.4f} | {r['nll_b']:.4f} |")
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", help="train_prism.py checkpoint dir (best/ or last/)")
    ap.add_argument("--data", nargs="+", help="jsonl files or globs (val or test)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-source", type=int, default=1000,
                    help="seeded subset of at most N records per source (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32, help="max records per batch")
    ap.add_argument("--max-batch-tokens", type=int, default=16384,
                    help="max padded tokens per batch (0 = no cap)")
    ap.add_argument("--max-state-tokens", type=int, default=None,
                    help="default: the checkpoint's training value")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--ple-cpu", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A_ITEMS", "B_ITEMS"),
                    help="paired B - A from two items.jsonl (no model)")
    a = ap.parse_args(argv)
    if not a.compare and not (a.ckpt and a.data):
        ap.error("give --ckpt and --data, or --compare A B")
    return a


def single_process_only() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise SystemExit("eval_prism.py runs as one process per GPU, not under torchrun "
                         "(WORLD_SIZE > 1). Use CUDA_VISIBLE_DEVICES to pick the GPU.")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.compare:
        c = compare(read_items(args.compare[0]), read_items(args.compare[1]))
        c.update(a=args.compare[0], b=args.compare[1])
        (out / "compare.json").write_text(json.dumps(c, indent=2))
        (out / "compare.md").write_text(compare_markdown(c) + "\n")
        print(compare_markdown(c))
        return c
    single_process_only()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    t0 = time.time()
    model, tok, margs = load_prism(args.ckpt, device, args.dtype, args.ple_cpu)
    load_s = time.time() - t0
    paths = expand(args.data)
    records = select([r for p in paths for r in read_jsonl(p)], args.max_per_source, args.seed)
    mst = args.max_state_tokens or margs.max_state_tokens
    t1 = time.time()
    items = collect(model, tok, records, mst, device, args.bs, args.max_batch_tokens, log_every=50)
    m = metrics(items)
    counts: dict = {}
    for r in records:
        counts[r.source] = counts.get(r.source, 0) + 1
    res = {"ckpt": str(args.ckpt), "data": paths, "max_per_source": args.max_per_source,
           "seed": args.seed, "max_state_tokens": mst, "dtype": str(model.get_input_embeddings().weight.dtype),
           "device": device, "records": len(records), "records_per_source": counts,
           "questions": len(items), "load_s": round(load_s, 1), "eval_s": round(time.time() - t1, 1),
           **m}
    write_items(out / "items.jsonl", items)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    title = f"{args.ckpt} on {', '.join(Path(p).name for p in paths)}"
    md = to_markdown(title, m) + (f"\n\nmax_per_source {args.max_per_source} (0 = all), seed {args.seed}, "
                                  f"max_state_tokens {mst}, {len(records)} records, {len(items)} "
                                  "questions. ECE: 15 equal-mass bins of the top probability.\n")
    (out / "metrics.md").write_text(md)
    print(md)
    print(f"wrote {out / 'metrics.json'}, {out / 'metrics.md'}, {out / 'items.jsonl'}")
    return res


if __name__ == "__main__":
    main()
