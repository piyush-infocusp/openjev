"""Freeze the packed layout of the frozen trial stream (packing is a data condition, not a
search dimension: packing on, pack_len 1024).

Applies openjev/packing.pack_rows to the frozen trial stream (runs/optuna/trial_stream.json)
in micro-batches of 8, in stream order, and writes the row assignments (indices inside each
micro-batch) to runs/optuna/packed_layout.json with its own sha. Record order and the step
definition are unchanged. optuna_loop.py records the sha in the fixed conditions;
bench_trials.py loads and verifies the file when it sits next to the trial stream and falls
back to on-the-fly packing (deterministic, same rows) when it is absent.

  .venv/bin/python tools/freeze_packed.py [--check]     # --check: only verify the existing file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openjev import trial_data as td  # noqa: E402
from openjev.packing import pack_rows  # noqa: E402

LAYOUT_VERSION = 1
LAYOUT_PATH = ROOT / "runs/optuna/packed_layout.json"


def build_layout(tok, records, stream_sha: str, pack_len: int = 1024, micro_batch: int = 8,
                 max_state_tokens: int = 1024) -> dict:
    from openjev.schema import encode
    if len(records) % micro_batch:
        raise ValueError(f"stream length {len(records)} is not a multiple of micro_batch {micro_batch}")
    mbs = []
    for s in range(0, len(records), micro_batch):
        lens = [len(encode(tok, r.state, r.schema, max_state_tokens).input_ids)
                for r in records[s:s + micro_batch]]
        mbs.append(pack_rows(lens, pack_len))
    out = {"version": LAYOUT_VERSION, "stream_sha": stream_sha, "n_records": len(records),
           "pack_len": pack_len, "micro_batch": micro_batch, "max_state_tokens": max_state_tokens,
           "option_subsample_k": 0, "mbs": mbs}
    out["sha"] = td.sha_of(out)
    return out


def main(argv=None):
    import train_prism as tp
    d = tp.parse([])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stream", type=Path, default=ROOT / "runs/optuna/trial_stream.json")
    ap.add_argument("--out", type=Path, default=LAYOUT_PATH)
    ap.add_argument("--train", default=d.train)
    ap.add_argument("--tokenizer", default=d.model)
    ap.add_argument("--pack-len", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-state-tokens", type=int, default=1024)
    ap.add_argument("--check", action="store_true", help="recompute and compare with --out, write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from openjev.schema import SPECIAL_TOKENS
    frozen = json.loads(a.stream.read_text())
    if td.sha_of({"stream": frozen["stream"], "probe": frozen["probe"]}) != frozen["sha"]:
        raise SystemExit(f"{a.stream}: content does not match its sha")
    records = td.load_refs(a.train, frozen["stream"])
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    layout = build_layout(tok, records, frozen["sha"], a.pack_len, a.micro_batch, a.max_state_tokens)
    rows = [len(r) for r in layout["mbs"]]
    print(f"packed layout: {len(rows)} micro-batches of {a.micro_batch}, pack_len {a.pack_len}, "
          f"rows/mb mean {sum(rows) / len(rows):.2f} (min {min(rows)}, max {max(rows)}), sha {layout['sha']}")
    if a.check:
        old = td.load_layout(a.out, frozen["sha"], len(records))
        if old is None or old["sha"] != layout["sha"]:
            raise SystemExit(f"{a.out} {'missing' if old is None else 'differs: ' + old['sha']}")
        print("check ok")
        return layout
    if a.out.exists() and not a.force:
        old = td.load_layout(a.out)
        if old["sha"] != layout["sha"]:
            raise SystemExit(f"{a.out} exists with sha {old['sha']}; frozen once per study, pass --force "
                             "only for a NEW study")
    td.write_json(a.out, layout)
    print(f"wrote {a.out}")
    return layout


if __name__ == "__main__":
    main()
