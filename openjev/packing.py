"""Hierarchical packing of encoded requests into rows (OJ-060).

Levels:
  1. source mix   - the records of one micro-batch come from the temperature-sampled
                    Mixture (or the frozen trial stream), so every row keeps the mix;
  2. length bucket - each request goes to a power-of-two length bucket; buckets are
                    filled longest first, so long requests anchor rows and short ones
                    fill the gaps;
  3. first-fit-decreasing - inside that order (longest first), each request goes to the
                    first row with room, rows hold <= pack_len tokens.

A request longer than pack_len gets a row of its own (never cut or dropped: it is one
forward pass). The rows only say where each request goes; prism_collate builds the
tensors, with a doc-id block-diagonal mask and positions that restart per request
(Krell et al. 2021, arXiv:2107.02027).
"""

from __future__ import annotations

import random


def length_bucket(n: int) -> int:
    """Power-of-two bucket of a length: 1 -> 0, 2 -> 1, 3..4 -> 2, 5..8 -> 3, ..."""
    return max(0, n - 1).bit_length()


def pack_rows(lengths: list[int], pack_len: int) -> list[list[int]]:
    """Indices of `lengths` grouped into rows of total length <= pack_len (except a
    single request longer than pack_len, which is alone in its row). Deterministic:
    ties keep the input order."""
    if pack_len <= 0:
        raise ValueError(f"pack_len must be > 0, got {pack_len}")
    order = sorted(range(len(lengths)), key=lambda i: (-length_bucket(lengths[i]), -lengths[i], i))
    rows: list[list[int]] = []
    room: list[int] = []
    for i in order:
        n = lengths[i]
        for r, free in enumerate(room):
            if n <= free:
                rows[r].append(i)
                room[r] -= n
                break
        else:
            rows.append([i])
            room.append(max(0, pack_len - n))
    return rows


# ---------------------------------------------------------------- length curriculum

# Sequence-length curriculum (user idea): train short requests first, then longer ones.
#
# Records are BUCKETED by their natural encoded length (state + questions + options + flags,
# exactly what prism_collate builds at max_state_tokens), never truncated or split to fit a
# stage. Stage k holds the records with stages[k-1] < length <= stages[k]; a record longer
# than the last stage goes to the last stage (packing gives it a row of its own). One epoch
# is every record of the pool exactly once: the per-stage seeded shuffles concatenated in
# increasing length, so each stage's share of the steps equals its share of the records.
# Each micro-batch packs at pack_mult x the stage maximum of its longest record (pack_mult 0:
# unpacked, the micro-batch is padded to its own longest request, which is length-bucketed
# batching). Reference: Li, Zhang, He 2022, "The Stability-Efficiency Dilemma: Investigating
# Sequence Length Warmup for Training GPT Models", NeurIPS 2022, arXiv:2108.06084 (they
# truncate sequences early in pre-training; here requests are bucketed instead, because a
# cut request loses evidence and is no longer a Jev request).

STAGES = (128, 256, 512, 1024, 2048)


def parse_stages(text) -> tuple[int, ...]:
    """"128,256,512" -> (128, 256, 512); strictly increasing, > 0."""
    st = tuple(int(x) for x in str(text).split(",") if x.strip()) if isinstance(text, str) else tuple(text)
    if not st or any(s <= 0 for s in st) or any(a >= b for a, b in zip(st, st[1:])):
        raise ValueError(f"stages must be increasing positive ints, got {text!r}")
    return st


def request_parts(tokenizer, records) -> list[tuple[int, int]]:
    """(template tokens, state tokens untruncated) of every record. encode tokenizes each
    piece (state, name, instructions, option label) on its own, so the encoded length is
    template + min(state, max_state_tokens): <|ctx|> <|/ctx|>, then per question <|q|> name
    <|instr|> text <|/q|> and <|opt|> label <|flag|> per option. Piece lengths are cached
    (option labels repeat a lot) and states are tokenized in batches."""
    cache: dict[str, int] = {}

    def ntok(s: str) -> int:
        if s not in cache:
            cache[s] = len(tokenizer(s, add_special_tokens=False)["input_ids"]) if s else 0
        return cache[s]

    out = []
    for i in range(0, len(records), 1024):
        chunk = records[i:i + 1024]
        ids = tokenizer([r.state for r in chunk], add_special_tokens=False)["input_ids"]
        for r, x in zip(chunk, ids):
            n = 2
            for q in r.schema.questions:
                n += 3 + ntok(q.name) + ntok(q.instructions or "") + sum(2 + ntok(lb) for lb in q.labels)
            out.append((n, len(x)))
    return out


def request_lengths(tokenizer, records, max_state_tokens: int) -> list[int]:
    """The encoded length of every record, == len(schema.encode(...).input_ids)."""
    return [t + min(s, max_state_tokens) for t, s in request_parts(tokenizer, records)]


def stage_of(n: int, stages) -> int:
    """Index of the first stage with n <= stage; longer than the last -> the last stage."""
    for k, s in enumerate(stages):
        if n <= s:
            return k
    return len(stages) - 1


def curriculum_order(lengths: list[int], stages, seed: int = 0) -> list[int]:
    """One epoch: every index exactly once, stage by stage (shortest first), each stage
    shuffled with its own seeded RNG."""
    by: list[list[int]] = [[] for _ in stages]
    for i, n in enumerate(lengths):
        by[stage_of(n, stages)].append(i)
    out = []
    for k, idx in enumerate(by):
        random.Random(f"curriculum:{seed}:{k}").shuffle(idx)
        out += idx
    return out


def stage_pack_len(stage_max: int, pack_mult: int) -> int:
    """Row length of a stage: pack_mult x the stage maximum (0 = unpacked)."""
    return int(pack_mult) * int(stage_max)


class CurriculumStream:
    """The curriculum as a stream with the Mixture interface (take, get_state, set_state).
    After take(n), `stage` is the stage index of the longest returned record and
    `pack_len` the row length for that micro-batch. Past the end of an epoch it starts a
    new one (seed + epoch): short stages again, a new shuffle."""

    def __init__(self, records: list, lengths: list[int], stages=STAGES, seed: int = 0,
                 pack_mult: int = 1):
        if not records or len(records) != len(lengths):
            raise ValueError("records and lengths must be non-empty and aligned")
        self.records, self.lengths = records, lengths
        self.stages, self.seed, self.pack_mult = parse_stages(stages), seed, pack_mult
        self.epoch, self.pos, self.stage = 0, 0, 0
        self.order = curriculum_order(lengths, self.stages, seed)

    def take(self, n: int) -> list:
        out, top = [], 0
        for _ in range(n):
            if self.pos >= len(self.order):
                self.epoch, self.pos = self.epoch + 1, 0
                self.order = curriculum_order(self.lengths, self.stages, self.seed + self.epoch)
            i = self.order[self.pos]
            self.pos += 1
            top = max(top, self.lengths[i])
            out.append(self.records[i])
        self.stage = stage_of(top, self.stages)
        return out

    @property
    def pack_len(self) -> int:
        return stage_pack_len(self.stages[self.stage], self.pack_mult)

    def stage_counts(self) -> dict[int, int]:
        c = {s: 0 for s in self.stages}
        for n in self.lengths:
            c[self.stages[stage_of(n, self.stages)]] += 1
        return c

    def get_state(self):
        return (self.epoch, self.pos, self.stage)

    def set_state(self, state) -> None:
        epoch, self.pos, self.stage = state
        if epoch != self.epoch:
            self.epoch = epoch
            self.order = curriculum_order(self.lengths, self.stages, self.seed + epoch)


def build_curriculum(tokenizer, paths, n: int, max_state_tokens: int, stages=STAGES,
                     alpha: float = 0.5, seed: int = 0, pack_mult: int = 1) -> CurriculumStream:
    """train_prism --length-curriculum on: the run's pool is n = steps x global batch records
    drawn like the Mixture (per-source counts by the alpha weights, seeded; trial_data.
    freeze_stream), so the source mix stays the Mixture's, and one curriculum epoch is
    exactly the run: every pooled record once, short stages first."""
    from . import trial_data as td

    refs = td.freeze_stream(sorted(paths), n, 0, alpha=alpha, seed=seed, check=False)
    records = td.load_refs(sorted(paths), refs["stream"])
    stream = CurriculumStream(records, request_lengths(tokenizer, records, max_state_tokens),
                              stages, seed, pack_mult)
    stream.pool_sha = refs["sha"]
    return stream


# ---------------------------------------------------------------- frozen stage streams (bench)

def scan_parts(tokenizer, paths) -> dict:
    """{stem: {"lines", "ids", "tmpl", "state"}} for every non-blank line of every file:
    raw line number, record id, template tokens, untruncated state tokens (request_parts).
    The encoded length at any max_state_tokens is tmpl + min(state, max_state_tokens)."""
    import json
    from pathlib import Path

    from .data import record_from_dict

    out = {}
    for p in sorted(paths):
        lines, recs = [], []
        with open(p) as f:
            for i, line in enumerate(f):
                if line.strip():
                    lines.append(i)
                    recs.append(record_from_dict(json.loads(line)))
        parts = request_parts(tokenizer, recs)
        out[Path(p).stem] = {"lines": lines, "ids": [r.id for r in recs],
                             "tmpl": [t for t, _ in parts], "state": [s for _, s in parts]}
    return out


def bucket_table(scan: dict, max_state_tokens: int, stages=STAGES) -> dict:
    """Per source and stage: records, tokens, records whose state is cut at max_state_tokens,
    records longer than the last stage (they train in the last stage, in a row of their own)."""
    stages = parse_stages(stages)
    out = {}
    for s, d in sorted(scan.items()):
        row = {"n": len(d["ids"]), "records": {k: 0 for k in stages}, "tokens": {k: 0 for k in stages},
               "state_cut": 0, "over_last": 0}
        for t, st in zip(d["tmpl"], d["state"]):
            n = t + min(st, max_state_tokens)
            k = stages[stage_of(n, stages)]
            row["records"][k] += 1
            row["tokens"][k] += n
            row["state_cut"] += st > max_state_tokens
            row["over_last"] += n > stages[-1]
        out[s] = row
    return out


def bucket_weights(scan: dict, max_state_tokens: int, stages=STAGES, alpha: float = 0.5) -> dict:
    """Share of the training steps per stage under the Mixture (alpha): sum over sources of
    the source weight x the share of its records in that stage (bucket_table rules)."""
    from .mix import source_weights

    stages = parse_stages(stages)
    t = bucket_table(scan, max_state_tokens, stages)
    w = source_weights({s: r["n"] for s, r in t.items()}, alpha)
    return {k: sum(w[s] * r["records"][k] / r["n"] for s, r in t.items()) for k in stages}


def freeze_stages(scan: dict, max_state_tokens: int, stages=STAGES, n_per_stage: int = 160,
                  probe_n: int = 8, alpha: float = 0.5, seed: int = 0, min_records: int = 0,
                  uncut_only: bool = False) -> dict:
    """Frozen per-stage trial streams for the length-curriculum bench: for each stage, n
    records whose encoded length (at max_state_tokens) falls in that stage, per-source counts
    by the Mixture weight times the source's share in that stage (what the stage holds in a
    curriculum run), plus one shared probe set (round robin over sources, any length, never
    in a stream). Stages with fewer than max(1, min_records) records get no stream. Stream
    "0" is the curriculum-off reference: n records of any length by the Mixture weights.
    uncut_only (the fixed bench set): a record whose state is cut at max_state_tokens or whose
    encoded length is over the last stage is in no stage stream (counted in "excluded"), so
    every stage record is whole and really in its bucket.
    Refs are [stem, line, id] like trial_data.freeze_stream; sha per stream and over the file."""
    from .mix import source_weights
    from .trial_data import STREAM_VERSION, counts_for, sha_of

    stages = parse_stages(stages)
    rng = random.Random(f"stages:{seed}")
    w = source_weights({s: len(d["ids"]) for s, d in scan.items()}, alpha)
    per = -(-probe_n // len(scan)) if probe_n else 0
    members = {k: {} for k in stages}
    spare, everyone = {}, {}
    excluded = {"state_cut": 0, "over_last": 0}
    for s in sorted(scan):
        d = scan[s]
        idx = list(range(len(d["ids"])))
        rng.shuffle(idx)
        spare[s], rest = idx[:per], idx[per:]
        everyone[s] = rest
        for i in sorted(rest):
            n = d["tmpl"][i] + min(d["state"][i], max_state_tokens)
            if uncut_only and (d["state"][i] > max_state_tokens or n > stages[-1]):
                excluded["state_cut"] += d["state"][i] > max_state_tokens
                excluded["over_last"] += n > stages[-1]
                continue
            members[stages[stage_of(n, stages)]].setdefault(s, []).append(i)
    probe = []
    for j in range(probe_n):
        srcs = sorted(spare)
        s = srcs[j % len(srcs)]
        if j // len(srcs) < len(spare[s]):
            i = spare[s][j // len(srcs)]
            probe.append([s, scan[s]["lines"][i], scan[s]["ids"][i]])
    streams, counts, skipped = {}, {}, {}
    c0 = counts_for(w, n_per_stage)
    mixed = []
    for s in sorted(c0):
        mixed += [[s, scan[s]["lines"][everyone[s][j % len(everyone[s])]],
                   scan[s]["ids"][everyone[s][j % len(everyone[s])]]] for j in range(c0[s])]
    rng.shuffle(mixed)
    streams["0"] = mixed
    counts["0"] = {"records_in_stage": sum(len(v) for v in everyone.values()),
                   "per_source": {s: c for s, c in c0.items() if c}}
    for k in stages:
        total = sum(len(v) for v in members[k].values())
        if total < max(1, min_records):
            skipped[str(k)] = total
            continue
        wk = {s: w[s] * len(v) / len(scan[s]["ids"]) for s, v in members[k].items()}
        z = sum(wk.values())
        ck = counts_for({s: x / z for s, x in wk.items()}, n_per_stage)
        refs = []
        for s in sorted(ck):
            pool = members[k][s][:]
            rng.shuffle(pool)
            refs += [[s, scan[s]["lines"][pool[j % len(pool)]], scan[s]["ids"][pool[j % len(pool)]]]
                     for j in range(ck[s])]                 # with replacement only if tiny
        rng.shuffle(refs)
        streams[str(k)] = refs
        counts[str(k)] = {"records_in_stage": total, "per_source": {s: c for s, c in ck.items() if c}}
    body = {"stages": list(stages), "max_state_tokens": max_state_tokens, "streams": streams, "probe": probe}
    return {"version": STREAM_VERSION, "kind": "stage_streams", "seed": seed, "alpha": alpha,
            "n_per_stage": n_per_stage, "counts": counts, "skipped_stages": skipped,
            **({"excluded": excluded} if uncut_only else {}),
            "stage_sha": {k: sha_of(v) for k, v in streams.items()}, **body, "sha": sha_of(body)}


# ---------------------------------------------------------------- fixed multi-length bench set

BENCH_KIND = "bench_lengths"


def freeze_bench_lengths(scan: dict, max_state_tokens: int = 2048, stages=STAGES, n_per_bucket: int = 32,
                         probe_n: int = 8, alpha: float = 0.5, seed: int = 0) -> dict:
    """The fixed speed-trial set (runs/optuna/bench_lengths.json, user decision 2026-11-23):
    n_per_bucket train records per length bucket (freeze_stages with uncut_only: stratified
    by source inside the bucket with the Mixture weights, never truncated, no duplicates),
    the shared probe set (disjoint), and the bucket weights = the share of training steps
    per bucket under the Mixture at max_state_tokens (bucket_weights). The trial objective is
    weighted samples/s: 1 / sum_b w_b x (seconds per sample in bucket b)."""
    stages = parse_stages(stages)
    d = freeze_stages(scan, max_state_tokens, stages, n_per_bucket, probe_n, alpha, seed,
                      min_records=n_per_bucket, uncut_only=True)
    if d["skipped_stages"]:
        raise ValueError(f"buckets with < {n_per_bucket} uncut records: {d['skipped_stages']}")
    buckets = {k: v for k, v in d["streams"].items() if k != "0"}
    seen = {tuple(r) for r in d["probe"]}
    for k, refs in buckets.items():
        if len({tuple(r) for r in refs}) != len(refs):
            raise ValueError(f"bucket {k}: duplicate records (pool too small)")
        if seen & {tuple(r) for r in refs}:
            raise ValueError(f"bucket {k}: overlaps the probe set or another bucket")
        seen |= {tuple(r) for r in refs}
    weights = {str(k): round(v, 6) for k, v in bucket_weights(scan, max_state_tokens, stages, alpha).items()}
    tab = bucket_table(scan, max_state_tokens, stages)
    body = {"stages": list(stages), "max_state_tokens": max_state_tokens, "weights": weights,
            "buckets": buckets, "probe": d["probe"]}
    return {"version": d["version"], "kind": BENCH_KIND, "seed": seed, "alpha": alpha,
            "n_per_bucket": n_per_bucket,
            "counts": {k: v for k, v in d["counts"].items() if k != "0"},
            "excluded_from_pool": d["excluded"],
            "full_data": {"records": sum(r["n"] for r in tab.values()),
                          "state_cut": sum(r["state_cut"] for r in tab.values()),
                          "over_last": sum(r["over_last"] for r in tab.values())},
            "bucket_sha": {k: d["stage_sha"][k] for k in buckets}, **body, "sha": sha_of_body(body)}


def sha_of_body(body) -> str:
    from .trial_data import sha_of
    return sha_of(body)


def mix_objective(per_bucket: dict, weights: dict) -> dict:
    """Weighted totals of per-bucket results ({bucket: {"seconds", "samples", "real_tokens",
    "padded_tokens"}}): the time a weighted mix of samples takes (harmonic mean of samples/s,
    weights = share of training steps per bucket, renormalized over the buckets given).
    samples_per_sec = 1 / sum_b w_b x s_b / n_b; tokens and pad by the same weights."""
    z = sum(float(weights[str(k)]) for k in per_bucket)
    if z <= 0:
        raise ValueError("bucket weights sum to 0")
    sps = rtok = ptok = 0.0
    for k, r in per_bucket.items():
        w = float(weights[str(k)]) / z
        sps += w * r["seconds"] / r["samples"]
        rtok += w * r["real_tokens"] / r["samples"]
        ptok += w * r["padded_tokens"] / r["samples"]
    return {"samples_per_sec": 1 / sps, "s_per_sample": sps, "tokens_per_sec": rtok / sps,
            "padded_tokens_per_sec": ptok / sps, "pad_frac": 1 - rtok / ptok if ptok else 0.0}


# ---------------------------------------------------------------- length-grouped batches (training)

class LengthGroupedStream:
    """Length-grouped micro-batches with a random order (bucket sampler; not a curriculum).
    Takes `chunk_mbs` x `bs` records from `base` (the Mixture: the source mix is kept), sorts
    them by encoded length, cuts micro-batches of `bs` and shuffles their order (seeded). Every
    record of a mega-chunk is served exactly once. take(n) with n a multiple of bs keeps the
    micro-batch cuts aligned, so step_micro_batches (bs-sized slices, DDP shards of whole
    micro-batches) gets whole grouped micro-batches on every rank."""

    def __init__(self, base, lengths_of, bs: int, chunk_mbs: int = 50, seed: int = 0):
        if bs < 1 or chunk_mbs < 1:
            raise ValueError("bs and chunk_mbs must be >= 1")
        self.base, self.lengths_of, self.bs, self.chunk_mbs = base, lengths_of, bs, chunk_mbs
        self._rng = random.Random(f"length-group:{seed}")
        self.buf, self.pos = [], 0

    def _refill(self) -> None:
        recs = self.base.take(self.bs * self.chunk_mbs)
        order = group_order(self.lengths_of(recs), self.bs, self._rng)
        self.buf, self.pos = [recs[i] for i in order], 0

    def take(self, n: int) -> list:
        out = []
        for _ in range(n):
            if self.pos >= len(self.buf):
                self._refill()
            out.append(self.buf[self.pos])
            self.pos += 1
        return out

    def get_state(self):
        return (self.base.get_state(), self._rng.getstate(), list(self.buf), self.pos)

    def set_state(self, state) -> None:
        base, rng, buf, self.pos = state
        self.base.set_state(base)
        self._rng.setstate(rng)
        self.buf = list(buf)


def group_order(lengths: list[int], bs: int, rng) -> list[int]:
    """Indices sorted by length (ties: input order), cut into micro-batches of bs, the
    micro-batch order shuffled with rng."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    mbs = [order[i:i + bs] for i in range(0, len(order), bs)]
    rng.shuffle(mbs)
    return [i for m in mbs for i in m]


def pad_frac(lengths: list[int], bs: int, mode: str = "random", pack_len: int = 0, chunk_mbs: int = 50,
             seed: int = 0, pad_mult: int = 8) -> dict:
    """Padding of one pass over `lengths` (every record once, seeded shuffle) in micro-batches
    of bs, with prism_collate's rules: each micro-batch is padded to its longest row rounded up
    to pad_mult; packed rows (pack_len > 0) by pack_rows. mode: random | grouped (mega-chunks
    of chunk_mbs micro-batches sorted by length, group_order)."""
    rng = random.Random(f"padfrac:{seed}")
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    if mode == "grouped":
        out, c = [], bs * chunk_mbs
        for j in range(0, len(idx), c):
            part = idx[j:j + c]
            out += [part[i] for i in group_order([lengths[i] for i in part], bs, rng)]
        idx = out
    elif mode != "random":
        raise ValueError(f"mode must be random or grouped, got {mode!r}")
    real = padded = rows = 0
    for j in range(0, len(idx), bs):
        lens = [lengths[i] for i in idx[j:j + bs]]
        rs = pack_rows(lens, pack_len) if pack_len else [[i] for i in range(len(lens))]
        n = max(sum(lens[i] for i in r) for r in rs)
        n = -(-n // pad_mult) * pad_mult
        real += sum(lens)
        padded += n * len(rs)
        rows += len(rs)
    return {"mode": mode, "pack_len": pack_len, "records": len(lengths), "real_tokens": real,
            "padded_tokens": padded, "pad_frac": 1 - real / padded if padded else 0.0,
            "rows_per_mb": rows / max(1, -(-len(lengths) // bs))}
