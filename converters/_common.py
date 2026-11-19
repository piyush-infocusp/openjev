"""Shared helpers for OpenJev dataset converters."""

from __future__ import annotations

import argparse
import hashlib
import random

from openjev.data import write_jsonl


def make_parser(description: str, default_split: str = "test") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--limit", type=int, default=None, help="max number of rows to convert (per HF split)")
    p.add_argument("--splits", type=str, default=default_split,
                    help="comma-separated list of HF splits to pull from (default: %(default)s)")
    p.add_argument("--out", type=str, required=True,
                    help="output path PREFIX; writes <prefix>.<train|val|test>.jsonl")
    p.add_argument("--seed", type=int, default=0, help="seed for shuffling option order / sampling")
    return p


def write_records(out_path: str, records) -> int:
    """records: iterable of openjev.data.Record"""
    return write_jsonl(out_path, records)


def shuffled_options(options: list, rng: random.Random) -> list:
    """Return a per-record shuffled copy of options using the given seeded Random."""
    opts = list(options)
    rng.shuffle(opts)
    return opts


def split_of(row_id: str, val: float = 0.05, test: float = 0.05) -> str:
    """Deterministic train/val/test assignment from a sha1 hash of the row id.

    A given id always maps to the same split, and never to two splits.
    """
    h = hashlib.sha1(row_id.encode("utf-8")).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    if frac < test:
        return "test"
    if frac < test + val:
        return "val"
    return "train"


def sample_indices(n: int, limit: int | None, seed: int) -> list[int]:
    """All indices 0..n-1 when limit is None (or >= n), else a seeded random
    subset of size min(limit, n), returned sorted (so order-in-file is kept)."""
    if limit is None or limit >= n:
        return list(range(n))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), limit))


def reservoir_sample(iterable, limit, seed, scan_cap: int | None = None):
    """Deterministic reservoir sample of up to `limit` items from a (possibly
    streaming, unbounded-length) iterable. Yields items in original relative
    order. limit=None passes everything through unchanged (a full pass, for
    real un-limited runs).

    A limited run only needs to look at a bounded prefix of the stream, not
    the whole dataset (streaming datasets can be huge and downloading all of
    it just to keep `limit` rows would defeat the point of `--limit`). That
    prefix length is `scan_cap` (default: 200x the limit, at least 5000), so
    the scan stays small while still mixing rows from well beyond the first
    `limit` of the stream.
    """
    if limit is None:
        yield from iterable
        return
    if scan_cap is None:
        scan_cap = max(limit * 200, 5000)
    rng = random.Random(seed)
    reservoir = []  # list of (original_index, item)
    for i, item in enumerate(iterable):
        if i >= scan_cap:
            break
        if len(reservoir) < limit:
            reservoir.append((i, item))
        else:
            j = rng.randint(0, i)
            if j < limit:
                reservoir[j] = (i, item)
    reservoir.sort(key=lambda x: x[0])
    for _, item in reservoir:
        yield item


def out_path(prefix: str, split: str) -> str:
    return f"{prefix}.{split}.jsonl"


def write_split_records(prefix: str, records_by_split: dict) -> dict:
    """records_by_split: {"train": [Record, ...], "val": [...], "test": [...]}.
    Writes one file per non-empty split and returns {split: row_count}."""
    counts = {}
    for split, recs in records_by_split.items():
        if not recs:
            continue
        counts[split] = write_jsonl(out_path(prefix, split), recs)
    return counts
