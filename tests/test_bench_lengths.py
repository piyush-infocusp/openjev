"""The fixed multi-length bench set (study openjev-speed-v3, user decision 2026-11-23) and the
length-grouped training batches: bucket coverage, no truncation, weights, the weighted
objective, sha stability, bench_trials --bench-lengths on a tiny Gemma 4 (fail closed on a
wrong set), optuna_loop propose with `bench_set`, LengthGroupedStream (every record once per
chunk, aligned micro-batches, less padding than random batches)."""
import glob
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from openjev import packing as pk  # noqa: E402
from openjev import trial_data as td  # noqa: E402
from openjev.mix import Mixture  # noqa: E402

FIX = sorted(glob.glob(str(ROOT / "tests/fixtures/*.jsonl")))
REAL = ROOT / "runs/optuna/bench_lengths.json"     # gitignored: present on the dev box only


def fake_scan(seed=0):
    """Three sources whose lengths cover every bucket, plus records whose state is cut at 2048
    and records longer than 2048 (both must stay out of the bench set)."""
    rng = random.Random(seed)
    scan = {}
    for s, (lo, hi, n) in {"a": (60, 700, 400), "b": (100, 1900, 300), "c": (200, 2600, 200)}.items():
        tmpl, state = [], []
        for _ in range(n):
            t = rng.randint(20, 40)
            tmpl.append(t)
            state.append(rng.randint(max(1, lo - t), hi - t))
        tmpl += [30, 30]
        state += [2100, 2047]          # one state cut, one whole state over the last stage
        scan[s] = {"lines": list(range(len(tmpl))), "ids": [f"{s}{i}" for i in range(len(tmpl))],
                   "tmpl": tmpl, "state": state}
    return scan


def lengths_of(scan, refs, mst):
    idx = {s: {ln: i for i, ln in enumerate(d["lines"])} for s, d in scan.items()}
    return [(scan[s]["tmpl"][idx[s][ln]], scan[s]["state"][idx[s][ln]]) for s, ln, _ in refs]


def test_bench_set_buckets_uncut_weights():
    scan = fake_scan()
    d = pk.freeze_bench_lengths(scan, 2048, pk.STAGES, 32, 8)
    assert d["kind"] == pk.BENCH_KIND and list(d["buckets"]) == [str(k) for k in pk.STAGES]
    lo = 0
    for k in pk.STAGES:                                       # bucket coverage, never truncated
        refs = d["buckets"][str(k)]
        assert len(refs) == 32 and len({tuple(r) for r in refs}) == 32
        for t, st in lengths_of(scan, refs, 2048):
            assert st <= 2048 and lo < t + st <= k
        lo = k
        assert sum(d["counts"][str(k)]["per_source"].values()) == 32
    assert d["excluded_from_pool"]["state_cut"] >= 1 and d["excluded_from_pool"]["over_last"] >= 1
    assert d["full_data"]["state_cut"] == sum(st > 2048 for v in scan.values() for st in v["state"]) > 0
    probe = {tuple(r) for r in d["probe"]}                    # the probe set stays disjoint
    assert len(probe) == 8 and not probe & {tuple(r) for v in d["buckets"].values() for r in v}
    assert abs(sum(d["weights"].values()) - 1) < 1e-5
    w = pk.bucket_weights(scan, 2048, pk.STAGES)
    assert d["weights"] == {str(k): round(v, 6) for k, v in w.items()}
    # stratified by source: a source with no records in a bucket gets none there
    t = pk.bucket_table(scan, 2048)
    for k in pk.STAGES:
        for s, c in d["counts"][str(k)]["per_source"].items():
            assert c and t[s]["records"][k] > 0


def test_bench_set_sha_stable():
    a = pk.freeze_bench_lengths(fake_scan(), 2048)
    b = pk.freeze_bench_lengths(fake_scan(), 2048)
    assert a == b and a["sha"] == td.sha_of({k: a[k] for k in ("stages", "max_state_tokens", "weights",
                                                               "buckets", "probe")})
    assert pk.freeze_bench_lengths(fake_scan(), 2048, seed=1)["sha"] != a["sha"]
    with pytest.raises(ValueError, match="buckets with"):
        pk.freeze_bench_lengths(fake_scan(), 2048, n_per_bucket=10_000)


@pytest.mark.skipif(not REAL.exists(), reason="runs/optuna/bench_lengths.json not built")
def test_real_bench_set_file():
    d = json.loads(REAL.read_text())
    body = {k: d[k] for k in ("stages", "max_state_tokens", "weights", "buckets", "probe")}
    assert td.sha_of(body) == d["sha"] and d["max_state_tokens"] == 2048
    assert list(d["buckets"]) == ["128", "256", "512", "1024", "2048"]
    assert all(len(v) == 32 == len({tuple(r) for r in v}) for v in d["buckets"].values())
    assert abs(sum(d["weights"].values()) - 1) < 1e-5


def test_mix_objective_is_the_weighted_harmonic_mean():
    per = {128: {"seconds": 4.0, "samples": 32, "real_tokens": 3200, "padded_tokens": 4000},
           2048: {"seconds": 32.0, "samples": 32, "real_tokens": 38400, "padded_tokens": 48000}}
    m = pk.mix_objective(per, {"128": 0.75, "2048": 0.25})
    # s per sample: 0.125 and 1.0 -> 0.75 x 0.125 + 0.25 x 1.0 = 0.34375
    assert m["s_per_sample"] == pytest.approx(0.34375) and m["samples_per_sec"] == pytest.approx(1 / 0.34375)
    assert m["tokens_per_sec"] == pytest.approx((0.75 * 100 + 0.25 * 1200) / 0.34375)
    assert m["pad_frac"] == pytest.approx(1 - 375 / 468.75)
    # not the arithmetic mean of the per-bucket samples/s (8 and 1 -> 6.25)
    assert m["samples_per_sec"] < 0.75 * 8 + 0.25 * 1
    # equal per-bucket speed -> that speed, whatever the weights
    same = {k: dict(v, seconds=v["samples"] / 2.0) for k, v in per.items()}
    assert pk.mix_objective(same, {"128": 0.1, "2048": 0.9})["samples_per_sec"] == pytest.approx(2.0)


# ---------------------------------------------------------------- length-grouped batches

class Seq:
    """A finite stand-in for the Mixture: take() returns the next ints."""

    def __init__(self):
        self.i = 0

    def take(self, n):
        out = list(range(self.i, self.i + n))
        self.i += n
        return out

    def get_state(self):
        return self.i

    def set_state(self, s):
        self.i = s


def test_length_grouped_every_record_once_aligned_and_seeded():
    L = [random.Random(i).randint(10, 2000) for i in range(10_000)]
    g = pk.LengthGroupedStream(Seq(), lambda recs: [L[r] for r in recs], bs=8, chunk_mbs=5, seed=3)
    chunk = 8 * 5
    out = g.take(chunk * 4)
    for c in range(4):                       # every record of each mega-chunk exactly once
        assert sorted(out[c * chunk:(c + 1) * chunk]) == list(range(c * chunk, (c + 1) * chunk))
    mbs = [out[i:i + 8] for i in range(0, len(out), 8)]
    for m in mbs:                            # aligned whole micro-batches, sorted inside
        assert [L[r] for r in m] == sorted(L[r] for r in m)
    first = [m[0] // chunk for m in mbs[:5]]
    assert len(set(first)) == 1              # a chunk's micro-batches stay together
    lens0 = [max(L[r] for r in m) for m in mbs[:5]]
    assert lens0 != sorted(lens0) or lens0 != sorted(lens0, reverse=True)   # random order, not a curriculum
    g2 = pk.LengthGroupedStream(Seq(), lambda recs: [L[r] for r in recs], bs=8, chunk_mbs=5, seed=3)
    assert g2.take(chunk * 4) == out         # seeded: every DDP rank builds the same stream
    st = g.get_state()
    a = g.take(13)
    g.set_state(st)
    assert g.take(13) == a


def test_length_grouped_on_the_mixture_keeps_records(ttok_lengths):
    tok, mst = ttok_lengths
    m0, m1 = Mixture(FIX, seed=0), Mixture(FIX, seed=0)
    g = pk.LengthGroupedStream(m1, lambda recs: pk.request_lengths(tok, recs, mst), bs=2, chunk_mbs=3)
    for _ in range(3):
        assert Counter(r.id for r in g.take(6)) == Counter(r.id for r in m0.take(6))


def test_grouped_pads_less_than_random():
    L = [random.Random(i).choice([100, 300, 500, 900, 1500]) + i % 37 for i in range(4000)]
    r = pk.pad_frac(L, 8, "random")
    g = pk.pad_frac(L, 8, "grouped", chunk_mbs=50)
    p = pk.pad_frac(L, 8, "random", pack_len=1024)
    assert r["real_tokens"] == g["real_tokens"] == p["real_tokens"] == sum(L)
    assert g["pad_frac"] < r["pad_frac"] / 3 and p["pad_frac"] < r["pad_frac"]
    assert r["rows_per_mb"] == g["rows_per_mb"] == 8 and p["rows_per_mb"] < 8


@pytest.fixture(scope="module")
def ttok_lengths():
    from transformers import AutoTokenizer

    from openjev.schema import SPECIAL_TOKENS
    t = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
    t.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return t, 128


def test_train_prism_length_grouping_flags():
    import train_prism as tp
    a = tp.parse([])
    assert a.length_grouping == "on" and a.group_chunk == 50
    assert tp.parse(["--length-curriculum", "on"]).length_grouping == "off"
    assert isinstance(tp.train_stream(None, FIX, tp.parse(["--length-grouping", "off"])), Mixture)
    assert isinstance(tp.train_stream(None, FIX, a), pk.LengthGroupedStream)
    with pytest.raises(SystemExit):
        tp.parse(["--length-curriculum", "on", "--length-grouping", "on"])


# ---------------------------------------------------------------- bench_trials + optuna_loop

def run(cmd, ok=True):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
    return r


def tiny_bench_set(tmp_path, base):
    out = tmp_path / "bench_lengths.json"
    run([sys.executable, str(ROOT / "tools/length_curriculum.py"), "bench", "--train",
         str(ROOT / "tests/fixtures/*.jsonl"), "--tokenizer", str(base), "--max-state", "128",
         "--stages", "256,512,1024", "--n-per-stage", "2", "--probe-n", "2", "--cache", "",
         "--out", str(out)])
    return out


def test_bench_trials_bench_lengths(tmp_path, tok):
    """Every trial times every bucket; per-bucket numbers, the weighted objective, the sha;
    a trials.json proposed for another set (or without --bench-lengths) is refused."""
    from test_optuna_loop import tiny_gemma
    base = tiny_gemma(tmp_path, tok)
    bl = tiny_bench_set(tmp_path, base)
    frozen = json.loads(bl.read_text())
    bench = {"global_batch": 2, "micro_batch": 1, "warmup_steps": 1, "timed_steps": 1,
             "bench_set_sha": frozen["sha"]}
    trials = [{"trial_number": i, "params": {"dtype": "fp32", "packing": "on", "pack_len": 1024, **p},
               "bench": bench} for i, p in enumerate([{}, {"option_subsample_k": 2}])]
    tf, res = tmp_path / "trials.json", tmp_path / "res.json"
    tf.write_text(json.dumps(trials))
    cmd = [sys.executable, str(ROOT / "bench_trials.py"), "--trials", str(tf), "--out", str(res),
           "--model", str(base), "--cpu", "--train", str(ROOT / "tests/fixtures/*.jsonl"),
           "--max-state-tokens", "48"]
    run(cmd + ["--bench-lengths", str(bl)])
    r = {x["trial_number"]: x for x in json.loads(res.read_text())["results"]}
    for x in r.values():
        assert x["status"] == "ok", x
        assert list(x["buckets"]) == ["256", "512", "1024"] and x["bench_set_sha"] == frozen["sha"]
        assert x["max_state_tokens_used"] == 128 and x["pack_len_used"] == 0
        assert x["records_used"] == 2 + 3 * 2 and x["warmup_buckets"] == [1024, 512]
        for b in x["buckets"].values():
            assert b["samples"] == 2 and b["samples_per_sec"] > 0 and b["tokens_per_sec"] > 0
            assert 0 <= b["pad_frac"] < 1
        m = pk.mix_objective({int(k): v for k, v in x["buckets"].items()}, frozen["weights"])
        assert x["samples_per_sec"] == pytest.approx(m["samples_per_sec"], rel=1e-3)
    assert r[0]["data_ids_sha"] == r[1]["data_ids_sha"] and r[0]["loss_start"] == r[1]["loss_start"]
    # fail closed: no --bench-lengths, or a set with another sha
    bad = run(cmd, ok=False)
    assert bad.returncode != 0 and "bench set" in bad.stderr + bad.stdout
    other = dict(frozen, sha="0" * 16)
    tf.write_text(json.dumps([dict(trials[0], bench=dict(bench, bench_set_sha=other["sha"]))]))
    bad = run(cmd + ["--bench-lengths", str(bl)], ok=False)
    assert bad.returncode != 0 and "bench set" in bad.stderr + bad.stdout


def test_propose_with_bench_set(tmp_path):
    """yaml `bench_set`: the study records bench_set_sha, fixed conditions name it instead of
    packing / pack_len / the trial stream, every trials.json entry carries it."""
    import optuna_loop as ol
    from test_optuna_loop import FIX as OFIX, space
    scan = fake_scan()
    bl = tmp_path / "bench_lengths.json"
    bl.write_text(json.dumps(pk.freeze_bench_lengths(scan, 2048)))
    g = space(tmp_path, bench_set=str(bl), timed_steps=1, diag_mb=None, study_name="openjev-speed-v3")
    db, out = tmp_path / "study.db", tmp_path / "trials.json"
    ol.main(g + ["--db", str(db), "propose", "--n", "1", "--out", str(out)])
    entries = json.loads(out.read_text())
    sha = json.loads(bl.read_text())["sha"]
    assert all(e["bench"]["bench_set_sha"] == sha and "trial_stream_sha" not in e["bench"] for e in entries)
    import optuna
    st = optuna.load_study(study_name="openjev-speed-v3", storage=f"sqlite:///{db}")
    c = st.user_attrs["fixed_conditions"]
    assert c["bench_set_sha"] == sha == st.user_attrs["bench_set_sha"]
    assert "packing" not in c and "trial_stream_sha" not in c and "packing" not in c["fixed_dims"]
    assert "pack_len" not in c["fixed_dims"] and c["max_state_tokens"].startswith("bench set: 2048")
    assert "weighted samples/s" in c["objective"]
    # the set must not change within the study
    bl.write_text(json.dumps(pk.freeze_bench_lengths(scan, 2048, seed=5)))
    with pytest.raises(SystemExit, match="bench_set_sha"):
        ol.main(g + ["--db", str(db), "propose", "--n", "0", "--abandon-running", "--out", str(out)])
    assert OFIX


def test_real_yaml_is_study_v3_on_the_bench_set():
    import optuna_loop as ol
    raw = yaml.safe_load((ROOT / "tools/search_space.yaml").read_text())
    assert raw["study_name"] == ol.STUDY_NAME == "openjev-speed-v3" and raw["round"] >= 7
    st = ol.load_settings(ROOT / "tools/search_space.yaml")
    assert ol.bench_set_path(st) == REAL and st["micro_batch"] == 8 and st["global_batch"] == 32


def test_bench_lengths_ddp_equals_one_process(tmp_path, tok):
    """ddp=on on the bench set (OJ_DDP_NPROC=2 gloo ranks): the same records per bucket,
    global token counts and guard loss as one process; local SGD K=50 runs."""
    from test_ddp import tiny_gemma
    base = tiny_gemma(tmp_path / "base", tok, kv_shared=False)
    bl = tiny_bench_set(tmp_path, base)
    frozen = json.loads(bl.read_text())
    bench = {"global_batch": 2, "micro_batch": 1, "warmup_steps": 1, "timed_steps": 1,
             "bench_set_sha": frozen["sha"]}
    ps = [{}, {"ddp": "on"}, {"ddp": "on", "sync_mode": "local_sgd", "sync_every": 50}]
    trials = [{"trial_number": i, "params": {"dtype": "fp32", **p}, "bench": bench,
               "reload_keys": ["ddp", "dtype"]} for i, p in enumerate(ps)]
    tf, res = tmp_path / "trials.json", tmp_path / "res.json"
    tf.write_text(json.dumps(trials))
    r = subprocess.run([sys.executable, str(ROOT / "bench_trials.py"), "--trials", str(tf), "--out", str(res),
                        "--model", str(base), "--cpu", "--train", str(ROOT / "tests/fixtures/*.jsonl"),
                        "--max-state-tokens", "48", "--bench-lengths", str(bl)],
                       cwd=ROOT, capture_output=True, text=True, env={**__import__("os").environ, "OJ_DDP_NPROC": "2"})
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
    out = {x["trial_number"]: x for x in json.loads(res.read_text())["results"]}
    single, ddp, lsgd = out[0], out[1], out[2]
    for x in (single, ddp, lsgd):
        assert x["status"] == "ok", x
    assert ddp["world"] == 2 and lsgd["sync_mode"] == "local_sgd"
    for x in (ddp, lsgd):
        assert x["data_ids_sha"] == single["data_ids_sha"]
        for k in single["buckets"]:
            for f in ("real_tokens", "padded_tokens", "samples"):
                assert x["buckets"][k][f] == single["buckets"][k][f], (k, f)
    assert abs(ddp["loss_guard"] - single["loss_guard"]) < 1e-4
