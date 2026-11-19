"""Convert HF `Tobi-Bueck/customer-support-tickets` (split train) into OpenJev JSONL.

Per row: state = subject + body; choice "queue" over all distinct queue values
seen in the loaded slice (shuffled per record); score "priority" with levels
["low","medium","high"], answer = the level index. Rows with missing
subject/body/queue/priority are skipped.

Only a real `train` HF split exists, so train/val/test are all carved out of
it by a hash of the row id (`converters._common.split_of`); a row never lands
in two files. Within the requested HF split(s), `--limit` takes a seeded
random subset (`sample_indices`), not just the first rows.
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

PRIORITY_LEVELS = ["low", "medium", "high"]


def main():
    args = make_parser("Convert Tobi-Bueck/customer-support-tickets to OpenJev JSONL", default_split="train").parse_args()

    rng = random.Random(args.seed)
    records_by_split: dict[str, list] = {"train": [], "val": [], "test": []}
    skipped = 0

    for hf_split in args.splits.split(","):
        ds = load_dataset("Tobi-Bueck/customer-support-tickets", split=hf_split)

        # Collect the distinct queue values present in this slice (real, full label set
        # of the loaded rows, not hand-picked). Sampling first, then computing the label
        # set from the sampled rows, keeps this consistent with --limit.
        idxs = sample_indices(len(ds), args.limit, args.seed)
        rows = [ds[i] for i in idxs]
        all_queues = sorted({r["queue"] for r in rows if r.get("queue")})

        i = 0
        for row in rows:
            subject = row.get("subject")
            body = row.get("body")
            queue = row.get("queue")
            priority = row.get("priority")
            if not subject or not body or not queue or not priority:
                skipped += 1
                continue
            if priority not in PRIORITY_LEVELS:
                skipped += 1
                continue

            rid = f"support-tickets-{hf_split}-{i:05d}"
            i += 1

            options = list(all_queues)
            rng.shuffle(options)
            criteria = {name: "" for name in options}
            schema = (
                Schema()
                .choice("queue", criteria)
                .score("priority", list(PRIORITY_LEVELS))
            )
            state = f"{subject}\n\n{body}"
            answers = [queue, PRIORITY_LEVELS.index(priority)]
            rec = Record(
                id=rid,
                source="Tobi-Bueck/customer-support-tickets",
                domain="workflow",
                state=state,
                schema=schema,
                answers=answers,
            )
            out_split = split_of(rid)
            records_by_split[out_split].append(rec)

    counts = write_split_records(args.out, records_by_split)
    for split, n in counts.items():
        print(f"support_tickets: wrote {n} records to {args.out}.{split}.jsonl")
    print(f"support_tickets: skipped {skipped} rows")


if __name__ == "__main__":
    main()
