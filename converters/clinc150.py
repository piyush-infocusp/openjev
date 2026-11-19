"""Convert HF `clinc/clinc_oos` (config "plus") into OpenJev JSONL.

Per row: one choice question "intent" over all in-scope intent names (shuffled,
"oos" excluded from the options), and one noul question "out_of_scope".
For an oos row: noul=true, choice answer=None. Otherwise noul=false, choice
answer=the intent name.

Real HF splits: train, validation, test. They map straight to the
train/val/test output splits (no hashing needed: the dataset already gives a
real split with a mix of intents and out-of-scope rows in each). Within each
requested HF split, `--limit` takes a seeded random subset (`sample_indices`),
not just the first rows, so a small sample still covers many intents and some
out-of-scope rows.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from openjev.data import Record
from openjev.schema import Schema
from converters._common import make_parser, sample_indices, write_split_records

HF_TO_OUT = {"train": "train", "validation": "val", "test": "test"}


def main():
    args = make_parser("Convert clinc/clinc_oos (plus) to OpenJev JSONL", default_split="train,validation,test").parse_args()

    full = load_dataset("clinc/clinc_oos", "plus")
    intent_names = full["train"].features["intent"].names
    in_scope_intents = sorted(name for name in intent_names if name != "oos")
    assert len(in_scope_intents) == 150, f"expected 150 in-scope intents, got {len(in_scope_intents)}"

    rng = random.Random(args.seed)
    records_by_split: dict[str, list] = {"train": [], "val": [], "test": []}

    for hf_split in args.splits.split(","):
        out_split = HF_TO_OUT[hf_split]
        ds = full[hf_split]
        idxs = sample_indices(len(ds), args.limit, args.seed)
        for i in idxs:
            row = ds[i]
            intent_name = intent_names[row["intent"]]
            is_oos = intent_name == "oos"

            options = list(in_scope_intents)
            rng.shuffle(options)
            criteria = {name: "" for name in options}
            schema = (
                Schema()
                .choice("intent", criteria)
                .noul(
                    "out_of_scope",
                    criteria={
                        "true": "the request is outside the assistant's supported intents",
                        "false": "the request matches a supported intent",
                    },
                )
            )
            answers = [
                None if is_oos else intent_name,
                is_oos,
            ]
            rec = Record(
                id=f"clinc150-{hf_split}-{i:05d}",
                source="clinc/clinc_oos",
                domain="workflow",
                state=row["text"],
                schema=schema,
                answers=answers,
            )
            records_by_split[out_split].append(rec)

    counts = write_split_records(args.out, records_by_split)
    for split, n in counts.items():
        print(f"clinc150: wrote {n} records to {args.out}.{split}.jsonl")


if __name__ == "__main__":
    main()
