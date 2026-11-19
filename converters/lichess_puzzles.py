"""Converter: Lichess/chess-puzzles (HF) -> OpenJev JSONL.

Each row has a FEN (the position before the opponent's setup move) and `Moves`,
a space-separated list of UCI moves where the FIRST move is the opponent's move
that creates the puzzle, and the rest is the solution line. We apply the first
move with python-chess to reach the actual position the solver must play from,
then build:
  - state: a text description of that position (FEN, side to move, themes),
           WITHOUT the solution move and WITHOUT the rating.
  - choice "best_move": criteria = the REAL legal moves of that position
    (python-chess), capped at 30 options (always keep the correct move),
    shuffled with the seed. answer = the first solution move, in UCI.
  - score "difficulty": 5 ordered levels bucketed from Rating with edges
    1000, 1400, 1800, 2200.

Only a real `train` HF split exists, so train/val/test are all carved out of
it by a hash of the puzzle id (`converters._common.split_of`); a row never
lands in two files. The dataset is read as a stream; `--limit` takes a
deterministic seeded reservoir sample (`reservoir_sample`) rather than just
the first rows.
"""

from __future__ import annotations

import argparse
import random

import chess
from datasets import load_dataset

from openjev.data import Record, bucket, write_jsonl
from converters._common import reservoir_sample, split_of, write_split_records
from openjev.schema import Schema

DIFFICULTY_LEVELS = [
    "Beginner (rating up to 1000)",
    "Easy (rating 1000-1400)",
    "Intermediate (rating 1400-1800)",
    "Hard (rating 1800-2200)",
    "Expert (rating above 2200)",
]

MAX_MOVE_OPTIONS = 30


def convert_row(row: dict, seed: int, idx: int) -> Record | None:
    fen = row.get("FEN")
    moves = (row.get("Moves") or "").split()
    if not fen or len(moves) < 2:
        return None
    try:
        board = chess.Board(fen)
        opponent_move = chess.Move.from_uci(moves[0])
        if opponent_move not in board.legal_moves:
            return None
        board.push(opponent_move)
        solution = moves[1]
        solution_move = chess.Move.from_uci(solution)
        legal = list(board.legal_moves)
        if solution_move not in legal:
            return None
    except (ValueError, IndexError):
        return None

    legal_uci = [m.uci() for m in legal]
    rng = random.Random(f"{seed}:{idx}")
    if len(legal_uci) > MAX_MOVE_OPTIONS:
        others = [u for u in legal_uci if u != solution]
        keep = rng.sample(others, MAX_MOVE_OPTIONS - 1)
        options = keep + [solution]
    else:
        options = list(legal_uci)
    rng.shuffle(options)

    side = "White" if board.turn == chess.WHITE else "Black"
    themes = row.get("Themes") or []
    themes_str = ", ".join(themes) if themes else "none"
    state = (
        f"Chess puzzle. Position (FEN): {board.fen()}. {side} to move. "
        f"Themes: {themes_str}."
    )

    rating = row.get("Rating")
    level = bucket(float(rating), [1000, 1400, 1800, 2200]) if rating is not None else None

    schema = Schema().choice(
        "best_move",
        {o: "" for o in options},
        instructions="Play the best move in this position (UCI notation).",
    ).score(
        "difficulty",
        DIFFICULTY_LEVELS,
        instructions="How difficult is this puzzle?",
    )

    puzzle_id = row.get("PuzzleId", str(idx))
    return Record(
        id=f"lichess-puzzle-{puzzle_id}",
        source="Lichess/chess-puzzles",
        domain="game",
        state=state,
        schema=schema,
        answers=[solution, level],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--splits", default="train", help="comma-separated list of HF splits to pull from")
    ap.add_argument("--out", default="data/lichess_puzzles", help="output path prefix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records_by_split: dict[str, list] = {"train": [], "val": [], "test": []}
    skipped = 0

    for hf_split in args.splits.split(","):
        ds = load_dataset("Lichess/chess-puzzles", split=hf_split, streaming=True)
        for i, row in enumerate(reservoir_sample(ds, args.limit, args.seed)):
            rec = convert_row(row, args.seed, i)
            if rec is None:
                skipped += 1
                continue
            out_split = split_of(rec.id)
            records_by_split[out_split].append(rec)

    counts = write_split_records(args.out, records_by_split)
    for split, n in counts.items():
        print(f"lichess_puzzles: wrote {n} records to {args.out}.{split}.jsonl")
    print(f"lichess_puzzles: skipped {skipped} rows")


if __name__ == "__main__":
    main()
