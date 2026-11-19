"""Convert HF `mteb/banking77` into OpenJev JSONL.

One record per row: state = query text, one choice question "intent" over all
77 intent names (shuffled per record), answer = the correct intent name.

Real HF splits: train, test (no validation). `test` maps straight to the
`test` output split. `val` is carved out of `train` by a hash of the row id
(`converters._common.split_of`), so a row never lands in two files. Within
each requested HF split, `--limit` takes a seeded random subset
(`sample_indices`), not just the first rows, so a small sample still covers
many intents.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from openjev.data import Record
from openjev.schema import Schema
from converters._common import make_parser, sample_indices, split_of, write_split_records


def main():
    args = make_parser("Convert mteb/banking77 to OpenJev JSONL", default_split="train,test").parse_args()

    full = load_dataset("mteb/banking77")
    all_intents = sorted({lt for split in full for lt in full[split]["label_text"]})
    assert len(all_intents) == 77, f"expected 77 intents, got {len(all_intents)}"

    rng = random.Random(args.seed)
    records_by_split: dict[str, list] = {"train": [], "val": [], "test": []}

    for hf_split in args.splits.split(","):
        ds = full[hf_split]
        idxs = sample_indices(len(ds), args.limit, args.seed)
        for i in idxs:
            row = ds[i]
            rid = f"banking77-{hf_split}-{i:05d}"

            options = list(all_intents)
            rng.shuffle(options)
            criteria = {name: "" for name in options}
            schema = Schema().choice("intent", criteria)
            rec = Record(
                id=rid,
                source="mteb/banking77",
                domain="workflow",
                state=row["text"],
                schema=schema,
                answers=[row["label_text"]],
            )

            if hf_split == "test":
                out_split = "test"
            else:
                # only val is carved out of train; test is already real
                out_split = split_of(rid, val=0.05, test=0.0)
            records_by_split[out_split].append(rec)

    counts = write_split_records(args.out, records_by_split)
    for split, n in counts.items():
        print(f"banking77: wrote {n} records to {args.out}.{split}.jsonl")


if __name__ == "__main__":
    main()
