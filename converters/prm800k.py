"""Convert GitHub `openai/prm800k` step-level labels into OpenJev JSONL.

One record per rated step completion: state = the problem, the previously
*chosen* step texts, and the step under review; one score question
"step_quality" with 3 levels ("bad", "neutral", "good") built from the human
rating (-1/0/1 -> 0/1/2).

Source files (Git LFS in the source repo) are downloaded once per phase/split
and cached under `data/raw/`:
    https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/<phase>_<split>.jsonl
`--splits` (default: all four) picks which `<phase>_<split>` files to read,
e.g. "phase1_train,phase2_test".

Each source line is one full solution (one "problem") with a list of graded
steps. A step can have several rated completions; we prefer the completion
the labeler actually chose (`chosen_completion`) when it is rated, else the
first rated completion, and skip a step if nothing is rated. Only the first
`--max-steps-per-solution` steps of a solution produce records, and the
"previous steps" context is built from the chosen completion's text (or the
human_completion, when given) of each earlier step -- never from its rating.

Splits: `*_test` files map straight to the `test` output split. `*_train`
files are split train/val by `converters._common.split_of` hashed on the
*problem* text (not the row id), so every step of one problem lands in the
same split, however individual step-records within a problem are sampled for
`--limit`. `--limit` takes a seeded reservoir sample of the output *records*
(`converters._common.reservoir_sample`), per source file, so a limited run
still mixes records from across the whole file rather than just its start.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openjev.data import Record
from openjev.schema import Schema
from converters._common import make_parser, reservoir_sample, split_of, write_split_records

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

# Tried in order; {name} is the "<phase>_<split>" file stem, e.g. "phase2_train".
SOURCE_URL_TEMPLATES = [
    "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/{name}.jsonl",
    "https://raw.githubusercontent.com/openai/prm800k/main/prm800k/data/{name}.jsonl",
]


def download_raw(name: str) -> str:
    """Return the local cached path for `<name>.jsonl`, downloading it (Git
    LFS-resolved) from the first URL template that works if not cached."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}.jsonl")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    errors = []
    for template in SOURCE_URL_TEMPLATES:
        url = template.format(name=name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "openjev-converter"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if not data.startswith(b"{"):
                errors.append(f"{url}: does not look like JSONL (got {data[:80]!r})")
                continue
            with open(path, "wb") as f:
                f.write(data)
            print(f"prm800k: downloaded {name} from {url} ({len(data)} bytes)")
            return path
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            errors.append(f"{url}: {e}")

    raise RuntimeError(
        f"prm800k: could not download {name}.jsonl from any source:\n" + "\n".join(errors)
    )


def pick_rated_completion(step: dict) -> dict | None:
    """The completion to rate this step by: the chosen one if it has a
    rating, else the first rated completion, else None."""
    completions = step.get("completions") or []
    chosen_idx = step.get("chosen_completion")
    if chosen_idx is not None and 0 <= chosen_idx < len(completions):
        chosen = completions[chosen_idx]
        if chosen.get("rating") is not None:
            return chosen
    for c in completions:
        if c.get("rating") is not None:
            return c
    return None


def chosen_text(step: dict) -> str | None:
    """The text of the step actually taken (for building later context),
    independent of rating: the chosen completion, else the human completion."""
    completions = step.get("completions") or []
    chosen_idx = step.get("chosen_completion")
    if chosen_idx is not None and 0 <= chosen_idx < len(completions):
        return completions[chosen_idx].get("text")
    human = step.get("human_completion")
    if human:
        return human.get("text")
    return None


def build_state(problem: str, previous_texts: list[str], step_text: str) -> str:
    prev = "\n".join(previous_texts) if previous_texts else "(none)"
    return (
        f"Problem:\n{problem}\n\n"
        f"Previous steps:\n{prev}\n\n"
        f"Step to rate:\n{step_text}"
    )


def records_for_line(row: dict, phase: str, line_idx: int, max_steps: int) -> list[Record]:
    problem = row["question"]["problem"]
    steps = row["label"]["steps"]
    out = []
    previous_texts: list[str] = []
    for step_idx, step in enumerate(steps):
        if step_idx >= max_steps:
            break
        target = pick_rated_completion(step)
        if target is not None:
            rating = target.get("rating")
            if rating is not None:
                text = target["text"]
                state = build_state(problem, previous_texts, text)
                schema = Schema().score(
                    "step_quality",
                    ["bad", "neutral", "good"],
                    instructions="How good is the step to rate, as a next step of the solution?",
                )
                rid = f"prm800k-{phase}-{line_idx:06d}-{step_idx:02d}"
                out.append(Record(
                    id=rid,
                    source="openai/prm800k",
                    domain="reasoning",
                    state=state,
                    schema=schema,
                    answers=[int(rating) + 1],
                ))
        ctext = chosen_text(step)
        if ctext is None:
            break
        previous_texts.append(ctext)
    return out


def main():
    parser = make_parser(
        "Convert GitHub openai/prm800k to OpenJev JSONL",
        default_split="phase1_train,phase1_test,phase2_train,phase2_test",
    )
    parser.add_argument("--max-steps-per-solution", type=int, default=8,
                         help="only rate the first N steps of each solution (default: %(default)s)")
    args = parser.parse_args()

    records_by_split: dict[str, list] = {"train": [], "val": [], "test": []}

    for name in args.splits.split(","):
        name = name.strip()
        path = download_raw(name)
        is_test_file = name.endswith("_test")

        def all_records(path=path, name=name):
            with open(path) as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for rec in records_for_line(row, phase=name, line_idx=i,
                                                 max_steps=args.max_steps_per_solution):
                        yield row["question"]["problem"], rec

        for problem, rec in reservoir_sample(all_records(), args.limit, args.seed):
            if is_test_file:
                out_split = "test"
            else:
                out_split = split_of(problem, val=0.05, test=0.0)
            records_by_split[out_split].append(rec)

    counts = write_split_records(args.out, records_by_split)
    for split, n in counts.items():
        print(f"prm800k: wrote {n} records to {args.out}.{split}.jsonl")


if __name__ == "__main__":
    main()
