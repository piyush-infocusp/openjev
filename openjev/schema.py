"""OpenJev template language: the Jev question types, special tokens, and the encoder.

OpenJev uses the same request and response format as TypeSafe's Jev API
(docs.typesafe.ai). A request has one `state` (the context) and named
`questions`. The model answers every question in one forward pass.

    {"state": "...", "questions": {"<name>": {"type": ..., "instructions": ..., "criteria": ...}}}

Three question types:
    noul   - one true/false decision. The answer is the probability of true (0..1),
             with no confidence field.
    choice - up to 255 named options. The answer is one option, a probability for
             each option, and a confidence.
    score  - two or more ordered levels. The answer is the expected level, a
             probability for each level, and a confidence.

Every type is a set of true/false flags. One flag reads at each FLAG token:
    noul   -> 1 flag.
    choice -> 1 flag per option. Softmax over the flags gives the probabilities.
    score  -> 1 flag per level.  Softmax over the flags gives the probabilities,
              and score = sum(level index * probability).

Grammar (EBNF). Whitespace between tokens is not significant.

    input     ::= CTX text CTX_END question+
    question  ::= Q text INSTR text option+ Q_END
    option    ::= OPT text FLAG                (text = option name and description)

The context comes first and the questions come last. With a causal mask, every
FLAG token can see the whole context.

Like Jev, OpenJev does not predict a confidence. Confidence is computed from the
probabilities: conf = (n * p_max - 1) / (n - 1), which is 0 when the options are
equal and 1 when one option takes everything. A noul answer carries no confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CTX, CTX_END = "<|ctx|>", "<|/ctx|>"
Q, Q_END = "<|q|>", "<|/q|>"
INSTR, OPT, FLAG = "<|instr|>", "<|opt|>", "<|flag|>"

SPECIAL_TOKENS = [CTX, CTX_END, Q, Q_END, INSTR, OPT, FLAG]

MAX_OPTIONS = 255


@dataclass
class Noul:
    """One true/false decision. criteria: {"true": description, "false": description}."""
    name: str
    instructions: str = ""
    criteria: dict | None = None

    type = "noul"

    @property
    def labels(self) -> list[str]:
        c = self.criteria or {}
        return [f"true: {c['true']}" if c.get("true") else "true",
                f"false: {c['false']}" if c.get("false") else "false"]


@dataclass
class Choice:
    """Named options. criteria: {option name: description} (2..255 options)."""
    name: str
    criteria: dict
    instructions: str = ""

    type = "choice"

    def __post_init__(self):
        if not 2 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"choice '{self.name}' needs 2..{MAX_OPTIONS} options")

    @property
    def options(self) -> list[str]:
        return list(self.criteria)

    @property
    def labels(self) -> list[str]:
        return [f"{k}: {v}" if v else k for k, v in self.criteria.items()]


@dataclass
class Score:
    """Ordered levels. criteria: ["Calm", "Frustrated", "Very angry"] (2 or more)."""
    name: str
    criteria: list
    instructions: str = ""

    type = "score"

    def __post_init__(self):
        if len(self.criteria) < 2:
            raise ValueError(f"score '{self.name}' needs at least two levels")

    @property
    def levels(self) -> list[str]:
        return list(self.criteria)

    @property
    def labels(self) -> list[str]:
        return [f"{i}: {lv}" for i, lv in enumerate(self.criteria)]


Question = Noul | Choice | Score


@dataclass
class Schema:
    """The `questions` object of one request."""
    questions: list[Question] = field(default_factory=list)

    def __post_init__(self):
        names = [q.name for q in self.questions]
        if len(set(names)) != len(names):
            raise ValueError("question names must be unique")

    # Builder: Schema().noul("is_urgent", ...).choice("department", {...}).score("mood", [...])
    def _add(self, q: Question) -> "Schema":
        if any(o.name == q.name for o in self.questions):
            raise ValueError(f"question name {q.name!r} is used twice")
        self.questions.append(q)
        return self

    def noul(self, name, instructions="", criteria=None) -> "Schema":
        return self._add(Noul(name, instructions, criteria))

    def choice(self, name, criteria, instructions="") -> "Schema":
        return self._add(Choice(name, criteria, instructions))

    def score(self, name, criteria, instructions="") -> "Schema":
        return self._add(Score(name, criteria, instructions))

    def to_dict(self) -> dict:
        out = {}
        for q in self.questions:
            d = {"type": q.type, "instructions": q.instructions}
            if q.criteria is not None:
                d["criteria"] = q.criteria
            out[q.name] = d
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        qs = []
        for name, q in d.items():
            kind, instr, crit = q["type"], q.get("instructions", ""), q.get("criteria")
            if kind == "noul":
                qs.append(Noul(name, instr, crit))
            elif kind == "choice":
                qs.append(Choice(name, crit, instr))
            elif kind == "score":
                qs.append(Score(name, list(crit), instr))
            else:
                raise ValueError(f"unknown question type {kind!r}")
        return cls(qs)


@dataclass
class Encoded:
    input_ids: list[int]
    # One entry per question, in order:
    #   {"type": "noul"|"choice"|"score", "name": str, "flags": [pos, ...]}
    readouts: list[dict]


def add_special_tokens(tokenizer, model, pad_to: int = 128, noise: float = 0.02) -> None:
    """Add the template tokens and give each one a fresh embedding.

    Many checkpoints have more embedding rows than tokenizer entries (Qwen3: 151936
    rows for 151669 tokens). The new tokens take those spare rows, so the matrix
    keeps its size. It only grows when there is no room, and then to a multiple of
    `pad_to`. Each new row is the mean embedding plus a small random offset, so the
    template tokens start near the data but stay different from each other.
    """
    import torch

    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if added == 0:
        return
    rows = model.get_input_embeddings().weight.shape[0]
    new_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if len(tokenizer) > rows:
        model.resize_token_embeddings(-(-len(tokenizer) // pad_to) * pad_to)
    old = min(rows, len(tokenizer) - added)
    tables = [model.get_input_embeddings()]
    ple = getattr(model, "get_per_layer_input_embeddings", lambda: None)()
    if ple is not None and ple.weight.shape[0] >= len(tokenizer):   # Gemma 4 PLE table
        tables.append(ple)
    with torch.no_grad():
        for emb in tables:
            w = emb.weight
            mean, std = _row_stats(w, old)
            for i in new_ids:
                w[i] = (mean + noise * std * torch.randn_like(mean)).to(w.dtype)


def _row_stats(w, n: int, chunk: int = 2048):
    """Mean row and mean per-column std of w[:n], a few rows at a time: the Gemma 4 PLE
    table has 2.35B elements, so a whole-table fp32 .std() would need ~10 GB of host RAM."""
    import torch

    s = torch.zeros(w.shape[1], dtype=torch.float64, device=w.device)
    s2 = torch.zeros_like(s)
    for i in range(0, n, chunk):
        x = w[i:min(i + chunk, n)].double()
        s += x.sum(0)
        s2 += (x * x).sum(0)
    mean = s / n
    var = (s2 / n - mean * mean).clamp_min(0) * n / max(n - 1, 1)
    return mean.to(torch.float32), var.sqrt().mean().to(torch.float32)


def encode(tokenizer, state: str, schema: Schema, max_state_tokens: int = 2048) -> Encoded:
    """Build input ids piece by piece so that readout positions are exact."""
    if not schema.questions:
        raise ValueError("a request needs at least one question")
    ids: list[int] = []

    def special(tok: str) -> int:
        ids.append(tokenizer.convert_tokens_to_ids(tok))
        return len(ids) - 1

    def text(s: str, limit: int | None = None) -> None:
        t = tokenizer(s, add_special_tokens=False)["input_ids"]
        ids.extend(t[:limit] if limit else t)

    special(CTX); text(state, max_state_tokens); special(CTX_END)

    readouts = []
    for q in schema.questions:
        special(Q); text(q.name)
        special(INSTR); text(q.instructions or "")
        flags = []
        for label in q.labels:
            special(OPT); text(label); flags.append(special(FLAG))
        special(Q_END)
        readouts.append({"type": q.type, "name": q.name, "flags": flags})
    return Encoded(ids, readouts)
