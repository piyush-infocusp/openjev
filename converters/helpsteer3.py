"""Converter: nvidia/HelpSteer3 (HF, config "preference") -> OpenJev JSONL.

Each row has a multi-turn `context` (list of {role, content}), two candidate
final assistant replies `response1` / `response2`, and `overall_preference`
(an int in -3..3: negative means response 1 is better, positive means
response 2 is better). We also get per-annotator `individual_preference`
with rationale text, which we do NOT use.

state = the conversation context plus response1 and response2, clearly
labelled, with NO rationale and NO preference value.

score "overall_preference": 7 ordered levels naming -3..3. answer = the
level index (overall_preference + 3), matching target_index for score
questions (a level index 0..6).

Real HF splits: train, validation (no test). `validation` maps straight to
the `val` output split. `test` is carved out of `train` by a hash of the row
id (`converters._common.split_of`), so a row never lands in two files. Both
splits are read as streams; `--limit` takes a deterministic seeded reservoir
sample (`reservoir_sample`) rather than just the first rows.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from openjev.data import Record
from openjev.schema import Schema
from converters._common import make_parser, reservoir_sample, split_of, write_split_records

DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"


def fit_parts(parts: list[str], shares: list[float], budget: int, count_tokens, encode, decode) -> list[str]:
    """Fit `parts` into a total token `budget`, each capped at its share of the
    budget, with unused budget from short parts flowing to the others.

    `count_tokens(text) -> int`, `encode(text) -> tokens`, `decode(tokens) -> text`
    are pluggable so this stays testable with a fake whitespace tokenizer.

    When a part must be cut, it keeps 2/3 of its allotted tokens from the head
    and 1/3 from the tail, joined with the marker "\\n[...]\\n".
    """
    n = len(parts)
    lengths = [count_tokens(p) for p in parts]
    caps = [int(budget * s) for s in shares]

    # Iteratively redistribute unused budget from parts shorter than their cap
    # to the parts that still need more, proportional to their remaining share.
    remaining = list(range(n))
    alloc = [0] * n
    free_budget = budget
    while remaining:
        # Parts within `remaining` that fit under their current cap: settle them.
        settled_any = False
        still_over = []
        total_share = sum(shares[i] for i in remaining) or 1.0
        for i in remaining:
            cap = int(free_budget * (shares[i] / total_share)) if total_share else 0
            if lengths[i] <= cap:
                alloc[i] = lengths[i]
                free_budget -= lengths[i]
                settled_any = True
            else:
                still_over.append(i)
        if not settled_any:
            # No part fits under its share; give each remaining part its
            # proportional cap of the free budget.
            total_share = sum(shares[i] for i in still_over) or 1.0
            for i in still_over:
                alloc[i] = int(free_budget * (shares[i] / total_share))
            remaining = []
            break
        remaining = still_over

    result = []
    for text, cap in zip(parts, alloc):
        tokens = encode(text)
        if len(tokens) <= cap:
            result.append(text)
            continue
        if cap <= 0:
            result.append("")
            continue
        marker = "\n[...]\n"
        content_budget = max(0, cap - count_tokens(marker))
        head_n = max(0, (content_budget * 2) // 3)
        tail_n = max(0, content_budget - head_n)
        head = decode(tokens[:head_n]) if head_n else ""
        tail = decode(tokens[-tail_n:]) if tail_n else ""
        result.append(f"{head}{marker}{tail}")
    return result

LEVELS = [
    "-3: response 1 much better",
    "-2: response 1 better",
    "-1: response 1 slightly better",
    "0: about the same",
    "1: response 2 slightly better",
    "2: response 2 better",
    "3: response 2 much better",
]

HF_TO_OUT = {"train": "train", "validation": "val"}


def format_context(context: list[dict]) -> str:
    lines = []
    for turn in context:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def convert_row(row: dict, rid: str, state_budget: int | None = None,
                 count_tokens=None, encode=None, decode=None) -> Record | None:
    context = row.get("context")
    r1 = row.get("response1")
    r2 = row.get("response2")
    pref = row.get("overall_preference")
    if not context or r1 is None or r2 is None or pref is None:
        return None
    try:
        pref = int(pref)
    except (TypeError, ValueError):
        return None
    if not -3 <= pref <= 3:
        return None

    ctx_text = "Conversation:\n" + format_context(context)
    r1_text = r1
    r2_text = r2
    if state_budget is not None:
        ctx_text, r1_text, r2_text = fit_parts(
            [ctx_text, r1_text, r2_text],
            [0.30, 0.35, 0.35],
            state_budget,
            count_tokens, encode, decode,
        )

    state = (
        ctx_text +
        "\n\n--- Response 1 ---\n" + r1_text +
        "\n\n--- Response 2 ---\n" + r2_text
    )

    schema = Schema().score(
        "overall_preference",
        LEVELS,
        instructions="Which response is better, and by how much?",
    )
    level = pref + 3
    return Record(
        id=rid,
        source="nvidia/HelpSteer3",
        domain="workflow",
        state=state,
        schema=schema,
        answers=[level],
    )


def main() -> None:
    parser = make_parser("Convert nvidia/HelpSteer3 (preference) to OpenJev JSONL",
                          default_split="train,validation")
    parser.add_argument("--state-budget", type=int, default=1000,
                         help="max tokens per state, allocated across context/response1/response2 (default: %(default)s)")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER,
                         help="HF tokenizer used to count/cut tokens for --state-budget (default: %(default)s)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def count_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def enc(text: str):
        return tok(text, add_special_tokens=False)["input_ids"]

    def dec(tokens) -> str:
        return tok.decode(tokens, skip_special_tokens=True)

    records_by_split: dict[str, list] = {"train": [], "val": [], "test": []}
    skipped = 0

    for hf_split in args.splits.split(","):
        ds = load_dataset("nvidia/HelpSteer3", "preference", split=hf_split, streaming=True)
        for i, row in enumerate(reservoir_sample(ds, args.limit, args.seed)):
            rid = f"helpsteer3-{hf_split}-{i:07d}"
            rec = convert_row(row, rid, state_budget=args.state_budget,
                               count_tokens=count_tokens, encode=enc, decode=dec)
            if rec is None:
                skipped += 1
                continue
            out_split = HF_TO_OUT[hf_split] if hf_split == "validation" else split_of(rid, val=0.0, test=0.05)
            records_by_split[out_split].append(rec)

    counts = write_split_records(args.out, records_by_split)
    for split, n in counts.items():
        print(f"helpsteer3: wrote {n} records to {args.out}.{split}.jsonl")
    print(f"helpsteer3: skipped {skipped} rows")


if __name__ == "__main__":
    main()
