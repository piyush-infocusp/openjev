"""Convert GitHub `DenisPeskoff/2020_acl_diplomacy` ("It Takes Two to Lie",
ACL 2020) into OpenJev JSONL.

Each raw line is one conversation (one game, two players) with parallel lists:
`messages`, `sender_labels` (True = the sender says the message is truthful,
False = a lie), `receiver_labels` (True/False/"NOANNOTATION"), `speakers`,
`receivers`, `seasons`, `years`, ... One OpenJev record per message with a
real `sender_labels` entry (all of them, in the file we downloaded — but we
still guard against a missing/non-bool label defensively, per the ticket).

state = the previous up to `--context` messages of the SAME conversation,
formatted "<speaker> (<season> <year>): <text>" one per line, then
"Message to judge:\n<speaker> -> <receiver> (<season> <year>): <text>".
`receiver_labels` and `game_score_delta` never appear in state, and the
ground-truth label is never echoed into state either.

Real splits: the raw `train`/`validation`/`test` files are split by game (no
leakage), and map straight to OpenJev's train/val/test — no `split_of` hash
is used here.

Deception is rare (sender says "false" on maybe ~5% of messages): this
converter does not rebalance; it just reports the true rate per split.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openjev.data import Record
from openjev.schema import Schema
from converters._common import make_parser, sample_indices, write_split_records

SOURCE = "DenisPeskoff/2020_acl_diplomacy"
DOMAIN = "strategy"

# raw split name -> candidate URLs to try, in order.
CANDIDATE_URLS = {
    "train": [
        "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/data/train.jsonl",
        "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/main/data/train.jsonl",
    ],
    "validation": [
        "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/data/validation.jsonl",
        "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/main/data/validation.jsonl",
    ],
    "test": [
        "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/data/test.jsonl",
        "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/main/data/test.jsonl",
    ],
}

# raw split name -> OpenJev output split name
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def _download(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "openjev-converter/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def load_raw_split(hf_split: str) -> list[dict]:
    """Return the list of conversation dicts for a raw split name
    ("train"/"validation"/"test"), using a local cache under data/raw/ if
    present, else downloading and caching it. Raises RuntimeError if every
    candidate URL fails and there is no cache."""
    os.makedirs(RAW_DIR, exist_ok=True)
    cache_path = os.path.join(RAW_DIR, f"diplomacy_{hf_split}.jsonl")

    if not os.path.exists(cache_path):
        data = None
        tried = []
        for url in CANDIDATE_URLS[hf_split]:
            tried.append(url)
            data = _download(url)
            if data is not None:
                with open(cache_path, "wb") as f:
                    f.write(data)
                break
        if data is None:
            raise RuntimeError(
                f"could not download diplomacy '{hf_split}' split; tried: {tried}. "
                "No cached copy at " + cache_path
            )

    rows = []
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_records(hf_split: str, out_split: str, conversations: list[dict], context: int):
    """Build one Record per message with a real bool sender_label. Returns
    (records, skipped_missing_label, n_true, n_false)."""
    records = []
    skipped = 0
    n_true = 0
    n_false = 0

    for conv_idx, conv in enumerate(conversations):
        messages = conv["messages"]
        sender_labels = conv["sender_labels"]
        speakers = conv["speakers"]
        receivers = conv["receivers"]
        seasons = conv["seasons"]
        years = conv["years"]

        n = len(messages)
        for msg_idx in range(n):
            label = sender_labels[msg_idx]
            if not isinstance(label, bool):
                skipped += 1
                continue

            lines = []
            start = max(0, msg_idx - context)
            for j in range(start, msg_idx):
                lines.append(f"{speakers[j]} ({seasons[j]} {years[j]}): {messages[j]}")

            judged = (
                f"{speakers[msg_idx]} -> {receivers[msg_idx]} "
                f"({seasons[msg_idx]} {years[msg_idx]}): {messages[msg_idx]}"
            )
            lines.append(f"Message to judge:\n{judged}")
            state = "\n".join(lines)

            schema = Schema().noul(
                "sender_truthful",
                instructions="Does the sender believe what they wrote in the message to judge?",
                criteria={
                    "true": "The sender is honest in this message",
                    "false": "The sender is knowingly lying",
                },
            )

            rid = f"diplomacy-{out_split}-{conv_idx:04d}-{msg_idx:03d}"
            rec = Record(
                id=rid,
                source=SOURCE,
                domain=DOMAIN,
                state=state,
                schema=schema,
                answers=[bool(label)],
            )
            records.append(rec)
            if label:
                n_true += 1
            else:
                n_false += 1

    return records, skipped, n_true, n_false


def main():
    args = make_parser(
        "Convert DenisPeskoff/2020_acl_diplomacy to OpenJev JSONL",
        default_split="train,validation,test",
    ).parse_args()
    context = 6

    records_by_split: dict[str, list] = {"train": [], "val": [], "test": []}
    total_skipped = 0

    for hf_split in args.splits.split(","):
        hf_split = hf_split.strip()
        if hf_split not in CANDIDATE_URLS:
            raise ValueError(f"unknown split '{hf_split}', expected one of {list(CANDIDATE_URLS)}")
        out_split = SPLIT_MAP[hf_split]

        conversations = load_raw_split(hf_split)
        records, skipped, n_true, n_false = build_records(hf_split, out_split, conversations, context)
        total_skipped += skipped

        idxs = sample_indices(len(records), args.limit, args.seed)
        sampled = [records[i] for i in idxs]
        records_by_split[out_split].extend(sampled)

        total = n_true + n_false
        rate = (n_false / total) if total else 0.0
        print(f"diplomacy_deception: {hf_split} -> {out_split}: {len(conversations)} conversations, "
              f"{total} labeled messages ({len(sampled)} kept after --limit), "
              f"lie rate {rate:.3%}, skipped {skipped} missing-label")

    counts = write_split_records(args.out, records_by_split)
    for split, n in counts.items():
        print(f"diplomacy_deception: wrote {n} records to {args.out}.{split}.jsonl")
    print(f"diplomacy_deception: total skipped (missing label) {total_skipped}")


if __name__ == "__main__":
    main()
