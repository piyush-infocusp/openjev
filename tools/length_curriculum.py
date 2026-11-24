#!/usr/bin/env python
"""Length curriculum: measure the length buckets and freeze per-stage bench streams (CPU,
tokenizer only; no model).

  # bucket table (records, token share, state cuts) per source at each max_state_tokens
  python tools/length_curriculum.py measure --max-state 1024 2048
  # frozen per-stage trial streams for bench_trials.py --stage-streams (record ids + shas)
  python tools/length_curriculum.py freeze --max-state 1024 --stages 128,256,512,1024,2048 \\
      --min-records 960 --out runs/length/stage_streams.json
  # the FIXED multi-length bench set of the speed study (v3+): 32 uncut records per bucket
  python tools/length_curriculum.py bench --max-state 2048 --out runs/optuna/bench_lengths.json
  # padding of one pass over the train set: random vs length-grouped vs packed micro-batches
  python tools/length_curriculum.py pad --max-state 1024
  # the bench study's yaml: tools/search_space.yaml with its own study name and the current
  # best config as defaults (so the auto baseline = curriculum off, packed 1024)
  python tools/length_curriculum.py space --out runs/length/space.yaml

Buckets use the exact encoded length (openjev.packing.request_parts == schema.encode):
state + questions + options + flags. Requests are never cut or split to fit a stage; a
request longer than the last stage trains in the last stage, in a row of its own.
--cache keeps the per-record token counts, so freeze after measure does not tokenize again.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openjev import packing as pk  # noqa: E402
from openjev.schema import SPECIAL_TOKENS  # noqa: E402


def load_scan(a) -> dict:
    paths = sorted(glob.glob(a.train))
    if not paths:
        raise SystemExit(f"no files match {a.train}")
    cache = Path(a.cache) if a.cache else None
    key = {"tokenizer": a.tokenizer, "files": {Path(p).stem: Path(p).stat().st_size for p in paths}}
    if cache and cache.exists():
        d = json.loads(cache.read_text())
        if d.get("key") == key:
            return d["scan"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    scan = pk.scan_parts(tok, paths)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"key": key, "scan": scan}))
    return scan


def table_md(scan, mst, stages, alpha) -> str:
    t = pk.bucket_table(scan, mst, stages)
    mix = pk.bucket_weights(scan, mst, stages, alpha)
    head = " | ".join(f"<={k}" for k in stages)
    lines = [f"### max_state_tokens {mst}", "",
             f"Records per stage (token share of the source in brackets). `cut` = state cut at "
             f"{mst}; `>{stages[-1]}` = longer than the last stage (trains there, own row).", "",
             f"| source | n | {head} | cut | >{stages[-1]} |",
             "|---|---:|" + "---:|" * len(stages) + "---:|---:|"]
    tot_r = {k: 0 for k in stages}
    tot_t = {k: 0 for k in stages}
    for s, r in t.items():
        tok = sum(r["tokens"].values()) or 1
        cells = [f"{r['records'][k]} ({100 * r['tokens'][k] / tok:.0f}%)" for k in stages]
        lines.append(f"| {s} | {r['n']} | " + " | ".join(cells) + f" | {r['state_cut']} | {r['over_last']} |")
        for k in stages:
            tot_r[k] += r["records"][k]
            tot_t[k] += r["tokens"][k]
    n = sum(tot_r.values())
    tt = sum(tot_t.values()) or 1
    lines.append(f"| **all** | {n} | " + " | ".join(f"{tot_r[k]} ({100 * tot_t[k] / tt:.0f}%)" for k in stages)
                 + f" | {sum(r['state_cut'] for r in t.values())} | {sum(r['over_last'] for r in t.values())} |")
    lines.append(f"| **Mixture share (alpha {alpha})** | | " + " | ".join(f"{100 * mix[k]:.1f}%" for k in stages)
                 + " | | |")
    return "\n".join(lines) + "\n"


def bench_md(d, scan) -> str:
    """Bucket table of a frozen bench set: records, sources, tokens, weight."""
    mst = d["max_state_tokens"]
    lines = [f"Bench set `{d['sha']}` (max_state_tokens {mst}, seed {d['seed']}): {d['n_per_bucket']} uncut "
             f"records per bucket; excluded from the pool: {d['excluded_from_pool']}; full data: "
             f"{d['full_data']}", "",
             "| bucket | records | sources | tokens (sum) | tokens min-max | weight (share of steps) |",
             "|---|---:|---|---:|---:|---:|"]
    idx = {s: {ln: i for i, ln in enumerate(v["lines"])} for s, v in scan.items()}
    for k, refs in d["buckets"].items():
        ns = [scan[s]["tmpl"][idx[s][ln]] + min(scan[s]["state"][idx[s][ln]], mst) for s, ln, _ in refs]
        src = ", ".join(f"{s.replace('.train', '')} {c}" for s, c in sorted(d["counts"][k]["per_source"].items()))
        lines.append(f"| <={k} | {len(refs)} | {src} | {sum(ns)} | {min(ns)}-{max(ns)} | "
                     f"{100 * d['weights'][k]:.1f}% |")
    return "\n".join(lines) + "\n"


def pad_md(scan, mst, bs, chunk_mbs, seed) -> str:
    """pad_frac of one pass over the whole train set (every record once) at micro-batch bs."""
    lens = [t + min(st, mst) for d in scan.values() for t, st in zip(d["tmpl"], d["state"])]
    rows = [pk.pad_frac(lens, bs, "random", 0, chunk_mbs, seed),
            pk.pad_frac(lens, bs, "grouped", 0, chunk_mbs, seed),
            pk.pad_frac(lens, bs, "random", 1024, chunk_mbs, seed),
            pk.pad_frac(lens, bs, "grouped", 1024, chunk_mbs, seed)]
    out = [f"### pad_frac, max_state_tokens {mst}, micro-batch {bs}, {len(lens)} records (each once)", "",
           "| batching | pad_frac | padded tokens | real tokens | rows / micro-batch |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        name = ("length-grouped (chunks of %d mbs)" % chunk_mbs if r["mode"] == "grouped" else "random") \
            + (f" + packed {r['pack_len']}" if r["pack_len"] else "")
        out.append(f"| {name} | {100 * r['pad_frac']:.1f}% | {r['padded_tokens']} | {r['real_tokens']} | "
                   f"{r['rows_per_mb']:.2f} |")
    return "\n".join(out) + "\n"


# The length bench study: never the main study (stage trials are faster by construction and
# train on other records, so "best" and the quality guard only make sense within a stage).
LENGTH_STUDY = {"study_name": "openjev-length-v1", "round": 1, "diag_mb": None}
BEST_DEFAULTS = {"attn_backend": "sdpa_bool", "packing": "on", "pack_len": 1024, "ple_cpu": 1,
                 "grad_ckpt": "on", "dtype": "fp16", "max_state_tokens": 1024}


def write_space(src, out) -> None:
    import yaml
    raw = yaml.safe_load(Path(src).read_text())
    raw.update(LENGTH_STUDY)
    for d in raw["dimensions"]:
        if d["name"] in BEST_DEFAULTS:
            d["default"] = BEST_DEFAULTS[d["name"]]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("# Generated by tools/length_curriculum.py space from " + str(src) + "\n"
                         + yaml.safe_dump(raw, sort_keys=False))
    print(json.dumps({"out": str(out), **LENGTH_STUDY, "defaults": BEST_DEFAULTS}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["measure", "freeze", "bench", "pad", "space"])
    ap.add_argument("--train", default="data/*.train.jsonl")
    ap.add_argument("--tokenizer", default="google/gemma-4-E2B-it")
    ap.add_argument("--stages", default=",".join(map(str, pk.STAGES)))
    ap.add_argument("--max-state", type=int, nargs="+", default=[1024])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--cache", default="runs/length/parts_cache.json")
    ap.add_argument("--n-per-stage", type=int, default=None,
                    help="records per stage: freeze 160 (>= (warmup + timed) x global batch), "
                         "bench 32 (one global batch per bucket)")
    ap.add_argument("--bs", type=int, default=8, help="pad: micro-batch")
    ap.add_argument("--chunk-mbs", type=int, default=50, help="pad: micro-batches per length-grouped chunk")
    ap.add_argument("--probe-n", type=int, default=8)
    ap.add_argument("--min-records", type=int, default=0,
                    help="freeze: a stage with fewer records gets no stream")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/length/stage_streams.json")
    ap.add_argument("--space", default="tools/search_space.yaml", help="space: the main yaml")
    a = ap.parse_args(argv)
    if a.n_per_stage is None:
        a.n_per_stage = 32 if a.cmd == "bench" else 160
    if a.cmd == "bench" and a.out == "runs/length/stage_streams.json":
        a.out = "runs/optuna/bench_lengths.json"
    if a.cmd == "space":
        return write_space(a.space, a.out if a.out.endswith(".yaml") else "runs/length/space.yaml")
    stages = pk.parse_stages(a.stages)
    scan = load_scan(a)
    if a.cmd == "measure":
        for mst in a.max_state:
            print(table_md(scan, mst, stages, a.alpha))
        return
    if a.cmd == "pad":
        for mst in a.max_state:
            print(pad_md(scan, mst, a.bs, a.chunk_mbs, a.seed))
        return
    if len(a.max_state) != 1:
        raise SystemExit(f"{a.cmd} takes one --max-state")
    if a.cmd == "bench":
        d = pk.freeze_bench_lengths(scan, a.max_state[0], stages, a.n_per_stage, a.probe_n, a.alpha, a.seed)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, separators=(",", ":")) + "\n")
        print(bench_md(d, scan))
        print(json.dumps({"out": str(out), "sha": d["sha"], "weights": d["weights"],
                          "excluded_from_pool": d["excluded_from_pool"], "full_data": d["full_data"]}))
        return
    d = pk.freeze_stages(scan, a.max_state[0], stages, a.n_per_stage, a.probe_n, a.alpha, a.seed,
                         a.min_records)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, separators=(",", ":")) + "\n")
    print(json.dumps({"out": str(out), "sha": d["sha"], "stage_sha": d["stage_sha"],
                      "counts": d["counts"], "skipped_stages": d["skipped_stages"]}, indent=1))


if __name__ == "__main__":
    main()
