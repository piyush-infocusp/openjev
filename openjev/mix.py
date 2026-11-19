"""Temperature-sampled mixing of multiple JSONL sources for training.

A raw concat of sources trains mostly on whatever dataset is biggest: a
100k-row source drowns a 500-row source, so the model never really learns
the small one. Temperature sampling with `alpha < 1` flattens the source
distribution — small sources get sampled more often relative to their size
than a proportional mix would give them, while `alpha=1` recovers plain
proportional mixing and `alpha=0` samples every source equally often
regardless of size. `Mixture` turns a list of JSONL files into one infinite,
deterministic stream of Records drawn under these weights.
"""

from __future__ import annotations

import random
from pathlib import Path

from .data import Record, read_jsonl


def source_weights(counts: dict[str, int], alpha: float = 0.5) -> dict[str, float]:
    """w_s proportional to n_s**alpha, normalized to sum to 1.

    alpha=1 is proportional to size; alpha=0 is uniform across sources.
    """
    if not counts:
        raise ValueError("counts must not be empty")
    for n in counts.values():
        if n <= 0:
            raise ValueError(f"count must be > 0, got {n}")
    raw = {s: n ** alpha for s, n in counts.items()}
    total = sum(raw.values())
    return {s: w / total for s, w in raw.items()}


class Mixture:
    """An infinite, temperature-sampled stream of Records over several JSONL sources."""

    def __init__(self, paths: list[str], alpha: float = 0.5,
                 weights: dict[str, float] | None = None, seed: int = 0):
        self._records: dict[str, list[Record]] = {}
        for path in paths:
            name = Path(path).stem
            self._records[name] = list(read_jsonl(path))

        if weights is not None:
            missing = set(self._records) - set(weights)
            if missing:
                raise ValueError(f"weights missing sources: {sorted(missing)}")
            total = sum(weights[s] for s in self._records)
            self._weights = {s: weights[s] / total for s in self._records}
        else:
            counts = {s: len(rs) for s, rs in self._records.items()}
            self._weights = source_weights(counts, alpha)

        self._sources = list(self._records)
        self._rng = random.Random(seed)
        self._orders: dict[str, list[int]] = {}
        self._cursors: dict[str, int] = {}
        for s in self._sources:
            self._reshuffle(s)

    def _reshuffle(self, source: str) -> None:
        order = list(range(len(self._records[source])))
        self._rng.shuffle(order)
        self._orders[source] = order
        self._cursors[source] = 0

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def _next(self, source: str) -> Record:
        cur = self._cursors[source]
        if cur >= len(self._orders[source]):
            self._reshuffle(source)
            cur = 0
        idx = self._orders[source][cur]
        self._cursors[source] = cur + 1
        return self._records[source][idx]

    def __iter__(self):
        sources = self._sources
        cum_weights = [self._weights[s] for s in sources]
        while True:
            source = self._rng.choices(sources, weights=cum_weights, k=1)[0]
            yield self._next(source)

    def take(self, n: int) -> list[Record]:
        it = iter(self)
        return [next(it) for _ in range(n)]
