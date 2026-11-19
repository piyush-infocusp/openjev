"""Frozen data for the Optuna speed trials (round 3+).

Every speed trial must see exactly the same batches, so their samples/s is measured on
the same sequence lengths. `tools/optuna_loop.py freeze` builds, once per study:

  trial_stream.json  an ordered list of train records ([file stem, line, id]) stratified by
                     source with the Mixture weights (n_s ~ count_s ** alpha), plus a fixed
                     probe set (quality guard), disjoint from the stream;
  val_set.json       a bigger, diverse val set for the quality checkpoints: about
                     `per_source` records per source, round robin over strata
                     (question types present, option-count bucket).

Records are referenced by raw line number (blank lines counted) AND id; `load_refs`
checks the id, so a changed data file fails loudly instead of training on other rows.
The sha of each file's selection is recorded in the study (trial_stream_sha, val_set_sha).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .data import record_from_dict
from .mix import source_weights

STREAM_VERSION = 1


def sha_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def nonblank_lines(path) -> list[int]:
    with open(path) as f:
        return [i for i, line in enumerate(f) if line.strip()]


def read_lines(path, wanted: set[int]) -> dict[int, dict]:
    out = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i in wanted:
                out[i] = json.loads(line)
    return out


def counts_for(weights: dict[str, float], n: int) -> dict[str, int]:
    """Largest-remainder split of n over the sources by weight."""
    raw = {s: n * w for s, w in weights.items()}
    out = {s: int(v) for s, v in raw.items()}
    for s in sorted(raw, key=lambda s: (-(raw[s] - out[s]), s))[: n - sum(out.values())]:
        out[s] += 1
    return out


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def freeze_stream(paths: list[str], n: int, probe_n: int, alpha: float = 0.5, seed: int = 0,
                  check: bool = True) -> dict:
    """The trial stream: n records, source counts by the Mixture weights, order shuffled.
    check: also compare state lengths (chars) of the sample with the whole file."""
    paths = sorted(paths)
    lines = {Path(p).stem: nonblank_lines(p) for p in paths}
    by_stem = {Path(p).stem: p for p in paths}
    weights = source_weights({s: len(v) for s, v in lines.items()}, alpha)
    counts = counts_for(weights, n)
    rng = random.Random(seed)
    stream, probe, chars, full_chars = [], [], {}, {}
    spare = {}
    for s in sorted(lines):
        order = lines[s][:]
        rng.shuffle(order)
        k = counts[s]
        # The probe comes first and is never in the stream (unless the file is tiny).
        n_probe = -(-probe_n // len(lines)) if probe_n else 0
        spare[s] = order[:n_probe]
        rest = order[n_probe:] or order
        stream += [(s, rest[i % len(rest)]) for i in range(k)]   # with replacement if tiny
    for i in range(probe_n):                                       # round robin over sources
        srcs = sorted(spare)
        s = srcs[i % len(srcs)]
        j = i // len(srcs)
        if j < len(spare[s]):
            probe.append((s, spare[s][j]))
    rng.shuffle(stream)
    refs = {}
    for s in sorted(lines):
        want = {ln for st, ln in stream + probe if st == s}
        rows = read_lines(by_stem[s], want)
        refs[s] = rows
        chars[s] = [len(rows[ln]["state"]) for st, ln in stream if st == s]
    stream_refs = [[s, ln, refs[s][ln]["id"]] for s, ln in stream]
    probe_refs = [[s, ln, refs[s][ln]["id"]] for s, ln in probe]
    for s in sorted(lines) if check else []:      # length check: sample vs the whole file (chars)
        with open(by_stem[s]) as f:
            full_chars[s] = [len(json.loads(line)["state"]) for line in f if line.strip()]
    check = {s: {"n": len(chars[s]),
                 "chars_p50": quantile(chars[s], 0.5), "chars_p95": quantile(chars[s], 0.95),
                 "file_chars_p50": quantile(full_chars.get(s, []), 0.5),
                 "file_chars_p95": quantile(full_chars.get(s, []), 0.95)} for s in sorted(lines)}
    body = {"stream": stream_refs, "probe": probe_refs}
    return {"version": STREAM_VERSION, "kind": "trial_stream", "seed": seed, "alpha": alpha, "n": n,
            "weights": {s: round(w, 5) for s, w in weights.items()}, "counts": counts,
            "length_check": check, **body, "sha": sha_of(body)}


def option_bucket(d: dict) -> str:
    """Option count of the largest choice/score question, bucketed."""
    k = 0
    for q in d["questions"].values():
        if q["type"] in ("choice", "score"):
            k = max(k, len(q["criteria"]))
    for hi, name in ((0, "none"), (2, "2"), (4, "3-4"), (8, "5-8"), (32, "9-32"), (100, "33-100")):
        if k <= hi:
            return name
    return "101+"


def stratum(d: dict) -> str:
    types = "+".join(sorted({q["type"] for q in d["questions"].values()}))
    return f"{types}|{option_bucket(d)}"


def freeze_val(paths: list[str], per_source: int, seed: int = 0) -> dict:
    """About per_source records per val file, round robin over its strata (question types
    present x option-count bucket) so rare types and sizes are in the set."""
    rng = random.Random(seed + 1)
    recs, strata = [], {}
    for p in sorted(paths):
        s = Path(p).stem
        groups: dict[str, list] = {}
        with open(p) as f:
            for i, line in enumerate(f):
                if line.strip():
                    d = json.loads(line)
                    if any(v is not None for v in d.get("answers", {}).values()):
                        groups.setdefault(stratum(d), []).append((i, d["id"]))
        for g in groups.values():
            rng.shuffle(g)
        keys = sorted(groups)
        picked, cur = [], {k: 0 for k in keys}
        while len(picked) < per_source and any(cur[k] < len(groups[k]) for k in keys):
            for k in keys:
                if len(picked) < per_source and cur[k] < len(groups[k]):
                    picked.append((k, *groups[k][cur[k]]))
                    cur[k] += 1
        strata[s] = {k: sum(1 for x in picked if x[0] == k) for k in keys}
        recs += [[s, ln, rid] for _k, ln, rid in picked]
    body = {"records": recs}
    return {"version": STREAM_VERSION, "kind": "val_set", "seed": seed, "per_source": per_source,
            "strata": strata, **body, "sha": sha_of(body)}


def load_refs(pattern_or_paths, refs) -> list:
    """[stem, line, id] refs -> Records, in order. Fails if a line's id differs."""
    import glob
    paths = sorted(glob.glob(pattern_or_paths)) if isinstance(pattern_or_paths, str) else pattern_or_paths
    by_stem = {Path(p).stem: p for p in paths}
    want: dict[str, set] = {}
    for s, ln, _ in refs:
        want.setdefault(s, set()).add(int(ln))
    rows = {}
    for s, lines in want.items():
        if s not in by_stem:
            raise ValueError(f"frozen data refers to {s!r}, not in {sorted(by_stem)}")
        rows[s] = read_lines(by_stem[s], lines)
    out = []
    for s, ln, rid in refs:
        d = rows[s].get(int(ln))
        if d is None or d.get("id") != rid:
            raise ValueError(f"frozen data mismatch: {s}:{ln} should be {rid!r}, "
                             f"found {None if d is None else d.get('id')!r} (data files changed?)")
        out.append(record_from_dict(d))
    return out


class FixedStream:
    """The frozen trial stream with the Mixture interface bench_trials uses (take,
    get_state, set_state). Past the end it wraps around and sets `wrapped`."""

    def __init__(self, records: list):
        if not records:
            raise ValueError("empty trial stream")
        self.records = records
        self.pos = 0
        self.wrapped = False

    def take(self, n: int) -> list:
        out = []
        for _ in range(n):
            if self.pos >= len(self.records):
                self.pos, self.wrapped = 0, True
            out.append(self.records[self.pos])
            self.pos += 1
        return out

    def get_state(self):
        return (self.pos, self.wrapped)

    def set_state(self, state) -> None:
        self.pos, self.wrapped = state


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=None, separators=(",", ":")) + "\n")
