"""OpenJev JSONL training format. Every dataset converter writes this format.

One line per request. `state` and `questions` are exactly the Jev request body.
`answers` holds the training labels, one per question, in the Jev answer shape:

    {
      "id": "banking77-train-00017",
      "source": "PolyAI/banking77",
      "domain": "workflow",
      "state": "How do I locate my card?",
      "questions": {
        "is_urgent":   {"type": "noul", "instructions": "Does this convey urgency?",
                        "criteria": {"true": "Time-sensitive", "false": "No urgency"}},
        "department":  {"type": "choice", "instructions": "Which team handles this?",
                        "criteria": {"billing": "Payments and refunds", "technical": "Bugs"}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]}
      },
      "answers": {"is_urgent": false, "department": "billing", "frustration": 1},
      "weight": 1.0
    }

Label types: noul -> true/false; choice -> the option name; score -> the level
index (or the level name). null means "no label", and that question gives no
loss, so datasets with partial labels can train together.

Two shapes that Jev has no type for:
  * a continuous number  -> the converter buckets it into ordered score levels.
  * several options at once -> the converter writes one noul question per option.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .model import target_index
from .schema import Schema


@dataclass
class Record:
    id: str
    source: str
    domain: str
    state: str
    schema: Schema
    answers: list  # aligned with schema.questions; None means no label
    weight: float = 1.0


def record_from_dict(d: dict) -> Record:
    schema = Schema.from_dict(d["questions"])
    labels = [d.get("answers", {}).get(q.name) for q in schema.questions]
    for q, y in zip(schema.questions, labels):
        target_index(q, y)  # raises ValueError on a bad label
    return Record(d["id"], d["source"], d.get("domain", "other"), d["state"],
                  schema, labels, float(d.get("weight", 1.0)))


def record_to_dict(r: Record) -> dict:
    return {"id": r.id, "source": r.source, "domain": r.domain, "state": r.state,
            "questions": r.schema.to_dict(),
            "answers": {q.name: y for q, y in zip(r.schema.questions, r.answers)},
            "weight": r.weight}


def read_jsonl(path: str):
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                try:
                    yield record_from_dict(json.loads(line))
                except (KeyError, ValueError) as e:
                    raise ValueError(f"{path}:{i + 1}: {e}") from e


def write_jsonl(path: str, records) -> int:
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(record_to_dict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def bucket(value: float, edges: list[float]) -> int:
    """Turn a number into a score level index. edges are the upper bounds.

    bucket(7.5, [1, 5, 20]) -> 2 (the third level, because 5 < 7.5 <= 20).
    """
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


def subsample_options(record: Record, k: int, rng) -> Record:
    """Training-only option sub-sampling (OJ-072): every `choice` question keeps its gold
    option plus at most `k` random negatives, in their original order. Returns a NEW
    Record (the Mixture hands out the same objects again, and eval must see all options);
    the input is never changed. The result is still a valid Jev request: a choice with a
    subset of its own options (>= 2), the same label, the same id.

    k = 0, or k >= n - 1 options: the question is unchanged. noul (2 options) and score
    (ordered levels; the graded reward uses level distances) are never sub-sampled. An
    unlabelled choice question keeps k + 1 random options: it gives no loss, and under the
    Prism mask its options are invisible to every other question, so trimming it is free.
    """
    from .schema import Choice, Schema

    if k <= 0:
        return record
    qs, changed = [], False
    for q, y in zip(record.schema.questions, record.answers):
        opts = q.options if q.type == "choice" else None
        if opts is None or len(opts) <= k + 1:
            qs.append(q)
            continue
        if y is None:
            keep = set(rng.sample(opts, k + 1))
        else:
            keep = {y, *rng.sample([o for o in opts if o != y], k)}
        qs.append(Choice(q.name, {o: d for o, d in q.criteria.items() if o in keep}, q.instructions))
        changed = True
    if not changed:
        return record
    return Record(record.id, record.source, record.domain, record.state, Schema(qs),
                  list(record.answers), record.weight)
