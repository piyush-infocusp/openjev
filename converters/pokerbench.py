"""Converter: RZ412/PokerBench (HF) -> OpenJev JSONL.

Each row is `instruction` (a 6-handed No Limit Hold'em scenario description)
and `output` (the solver's optimal action, e.g. "bet 18", "fold", "check",
"call", "raise 6"). state = the instruction text verbatim.

OJ-016 fix: earlier versions decided "facing a bet" and built the size menu
from the LABEL, which leaked the answer (an off-grid size or a menu shape
that only appears for one branch gave the answer away). Both are now decided
from the STATE text alone:

  - `facing_bet(instruction)` parses the last street's action line (the text
    right before "Now it is your turn") and looks at the most recent
    non-fold action by an opponent. bet/raise -> facing a bet (True);
    check/call, or no action yet (hero opens) -> not facing (False);
    unparseable -> None, and the row is skipped.
  - The menu is built purely from `facing_bet` and the pot size parsed out of
    the instruction: facing -> fold, call, raise S for S in a fixed pot-
    fraction grid; not facing -> check, bet S for S in the same grid. The
    correct action is always on the grid because the label is snapped onto
    it (see below), so the menu never needs the label to look "complete".
  - If the label's verb contradicts `facing_bet` (e.g. label "check" while
    facing a bet), the row is skipped and counted.
  - grid = sorted unique max(1, round(pot*f)) for f in
    [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]. For raises, sizes at or below the bet
    faced are dropped when that amount can be parsed.
  - A sized label (bet/raise N) is bucketed onto the nearest grid size in log
    space -- the same kind of continuous-value bucketing CLAUDE.md asks for
    with `openjev.data.bucket`, just with a nearest-neighbour rule instead of
    edges, because a poker size grid has no natural upper/lower level order.
    fold/call/check labels are used as-is.

Real HF splits: train, test. `test` maps straight to the `test` output
split; `val` is carved out of `train` by a hash of the row id
(`converters._common.split_of`). Both are read as streams; `--limit` takes a
deterministic seeded reservoir sample (`reservoir_sample`).
"""

from __future__ import annotations

import argparse
import math
import random
import re

from datasets import load_dataset

from openjev.data import Record, write_jsonl
from openjev.schema import Schema
from converters._common import reservoir_sample, split_of, write_split_records

POT_RE = re.compile(r"current pot size is\s+([\d.]+)\s+chips", re.IGNORECASE)
POS_RE = re.compile(r"your position is\s+(\w+)", re.IGNORECASE)
PRE_RE = re.compile(r"Before the flop,\s*(.*?)\.\s*Assume", re.IGNORECASE | re.DOTALL)
STREET_RE = re.compile(r"The (?:flop|turn|river) comes.*?,\s*then\s*(.*?)\.", re.IGNORECASE | re.DOTALL)
ACTION_RE = re.compile(r"([A-Za-z]+)\s+(fold|check|call|bet|raise)(?:\s+([\d.]+))?\s*(?:chips)?", re.IGNORECASE)

GRID_FRACTIONS = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]


def parse_pot(instruction: str) -> float | None:
    m = POT_RE.search(instruction)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _last_street_action(instruction: str):
    """Return (player, verb, amount|None) for the most recent non-fold action
    on the current street, or None if there is none yet or it can't be
    parsed."""
    pos_m = POS_RE.search(instruction)
    if not pos_m:
        return None
    hero = pos_m.group(1).upper()

    segments = []
    pre_m = PRE_RE.search(instruction)
    if pre_m:
        segments.append(pre_m.group(1))
    for m in STREET_RE.finditer(instruction):
        segments.append(m.group(1))

    if not segments:
        return "NONE"  # sentinel: hero is first to act, nothing to parse

    last_seg = segments[-1]
    actions = list(ACTION_RE.finditer(last_seg))
    if not actions:
        return None

    non_fold = [a for a in actions if a.group(2).lower() != "fold"]
    if not non_fold:
        return "NONE"

    last = non_fold[-1]
    player = last.group(1).upper()
    if player == hero:
        return None  # shouldn't happen on a turn-based log; can't trust it
    verb = last.group(2).lower()
    amount = float(last.group(3)) if last.group(3) else None
    return (player, verb, amount)


def facing_bet(instruction: str) -> bool | None:
    """True if the hero is facing a live bet/raise on the current street,
    False if not (including opening the action), None if unparseable."""
    last = _last_street_action(instruction)
    if last is None:
        return None
    if last == "NONE":
        return False
    _player, verb, _amount = last
    if verb in ("bet", "raise"):
        return True
    if verb in ("check", "call"):
        return False
    return None


def _bet_faced_amount(instruction: str) -> float | None:
    last = _last_street_action(instruction)
    if last is None or last == "NONE":
        return None
    _player, verb, amount = last
    if verb in ("bet", "raise"):
        return amount
    return None


def pot_grid(pot: float) -> list[int]:
    return sorted({max(1, round(pot * f)) for f in GRID_FRACTIONS})


def _nearest_log(grid: list[int], value: float) -> int:
    value = max(value, 1e-9)
    return min(grid, key=lambda g: abs(math.log(max(g, 1e-9)) - math.log(value)))


class ConvertStats:
    def __init__(self):
        self.skipped = {"no_pot": 0, "unparseable_state": 0, "label_contradicts_state": 0, "bad_label": 0}
        self.moved = 0
        self.sized_labels = 0
        self.rel_errors = []


def convert_row(row: dict, seed: int, idx: int, stats: ConvertStats | None = None) -> Record | None:
    stats = stats or ConvertStats()
    instruction = row.get("instruction")
    output = row.get("output")
    if not instruction or not output:
        stats.skipped["bad_label"] += 1
        return None

    pot = parse_pot(instruction)
    if pot is None:
        stats.skipped["no_pot"] += 1
        return None

    fb = facing_bet(instruction)
    if fb is None:
        stats.skipped["unparseable_state"] += 1
        return None

    parts = output.strip().split()
    verb = parts[0].lower()
    if verb not in ("fold", "check", "call", "bet", "raise"):
        stats.skipped["bad_label"] += 1
        return None

    label_ok_facing = verb in ("fold", "call", "raise")
    label_ok_not_facing = verb in ("check", "bet")
    if fb and not label_ok_facing:
        stats.skipped["label_contradicts_state"] += 1
        return None
    if not fb and not label_ok_not_facing:
        stats.skipped["label_contradicts_state"] += 1
        return None

    grid = pot_grid(pot)
    rng = random.Random(f"{seed}:{idx}")

    if fb:
        bet_faced = _bet_faced_amount(instruction)
        raise_grid = [s for s in grid if bet_faced is None or s > bet_faced]
        if not raise_grid:
            raise_grid = grid
        options = ["fold", "call"] + [f"raise {s}" for s in raise_grid]
    else:
        options = ["check"] + [f"bet {s}" for s in grid]

    if verb in ("bet", "raise"):
        try:
            size = float(parts[1])
        except (IndexError, ValueError):
            stats.skipped["bad_label"] += 1
            return None
        sub_grid = raise_grid if fb else grid
        snapped = _nearest_log(sub_grid, size)
        label = f"{verb} {snapped}"
        stats.sized_labels += 1
        if snapped != round(size):
            stats.moved += 1
        stats.rel_errors.append(abs(snapped - size) / max(size, 1e-9))
    else:
        label = verb

    if label not in options:
        options.append(label)  # safety net; shouldn't trigger given the grid construction
    rng.shuffle(options)

    schema = Schema().choice(
        "action",
        {o: "" for o in options},
        instructions="What is the optimal poker action in this spot?",
    )
    return Record(
        id=f"pokerbench-{idx:07d}",
        source="RZ412/PokerBench",
        domain="game",
        state=instruction,
        schema=schema,
        answers=[label],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--splits", default="train,test", help="comma-separated list of HF splits to pull from")
    ap.add_argument("--out", default="data/pokerbench", help="output path prefix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records_by_split: dict[str, list] = {"train": [], "val": [], "test": []}
    stats = ConvertStats()

    for hf_split in args.splits.split(","):
        ds = load_dataset("RZ412/PokerBench", split=hf_split, streaming=True)
        for i, row in enumerate(reservoir_sample(ds, args.limit, args.seed)):
            rec = convert_row(row, args.seed, i, stats)
            if rec is None:
                continue
            rid = f"pokerbench-{hf_split}-{i:07d}"
            rec.id = rid
            if hf_split == "test":
                out_split = "test"
            else:
                out_split = split_of(rid, val=0.05, test=0.0)
            records_by_split[out_split].append(rec)

    counts = write_split_records(args.out, records_by_split)
    for split, n in counts.items():
        print(f"pokerbench: wrote {n} records to {args.out}.{split}.jsonl")
    print(f"pokerbench: skipped rows by reason: {stats.skipped}")
    if stats.sized_labels:
        mean_rel_err = sum(stats.rel_errors) / len(stats.rel_errors)
        print(
            f"pokerbench: {stats.moved}/{stats.sized_labels} sized labels moved to a grid size "
            f"(mean abs relative error {mean_rel_err:.4f})"
        )


if __name__ == "__main__":
    main()
