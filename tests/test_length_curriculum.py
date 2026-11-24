"""Length curriculum: records bucketed by their natural encoded length (never cut or split),
one epoch = every record exactly once, stages in increasing length; frozen per-stage bench
streams; train_prism --length-curriculum and bench_trials --stage-streams on a tiny Gemma 4."""
import glob
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openjev import packing as pk  # noqa: E402
from openjev import trial_data as td  # noqa: E402
from openjev.data import read_jsonl  # noqa: E402
from openjev.prism import prism_collate  # noqa: E402
from openjev.schema import SPECIAL_TOKENS, encode  # noqa: E402

FIX = sorted(glob.glob(str(ROOT / "tests/fixtures/*.jsonl")))
STAGES = (256, 512, 1024)   # fixture lengths at max_state 48: lichess 178-340, banking ~480, clinc ~685


@pytest.fixture(scope="module")
def ttok():
    """A fresh tokenizer with the template tokens (never the shared session one: adding
    tokens to it would change add_special_tokens in other tests)."""
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
    t.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return t


@pytest.fixture(scope="module")
def recs():
    return [r for p in FIX for r in read_jsonl(p)]


@pytest.mark.parametrize("mst", [8, 48, 4096])
def test_request_lengths_equal_encode(ttok, recs, mst):
    """The fast length (cached pieces, batched states) is exactly the encoded length,
    including a state cut at max_state_tokens."""
    assert pk.request_lengths(ttok, recs, mst) == [len(encode(ttok, r.state, r.schema, mst).input_ids)
                                                   for r in recs]


def test_stages_and_buckets():
    assert pk.parse_stages("128,256, 512") == (128, 256, 512)
    for bad in ("", "256,128", "0,1", "64,64"):
        with pytest.raises(ValueError):
            pk.parse_stages(bad)
    st = (128, 256, 512)
    assert [pk.stage_of(n, st) for n in (1, 128, 129, 256, 257, 512, 513, 99999)] == [0, 0, 1, 1, 2, 2, 2, 2]
    assert pk.stage_pack_len(256, 2) == 512 and pk.stage_pack_len(256, 0) == 0


def test_order_covers_every_record_once_short_first():
    lengths = [300, 5, 900, 128, 129, 5000, 64, 600, 256, 257]
    st = (128, 256, 512, 1024)
    order = pk.curriculum_order(lengths, st, seed=3)
    assert sorted(order) == list(range(len(lengths)))
    stages = [pk.stage_of(lengths[i], st) for i in order]
    assert stages == sorted(stages)                      # never goes back to a shorter stage
    assert order == pk.curriculum_order(lengths, st, seed=3)   # deterministic
    assert stages[-1] == 3 and 5 in order                 # 5000 > last stage: trains in the last one


def test_stream_epoch_once_no_truncation(ttok, recs):
    """One epoch over the stream: each record exactly once, micro-batch stages never decrease,
    pack_len follows the stage, and every request keeps all its tokens in the packed batch
    (requests are never cut or split to fit a stage)."""
    mst = 48
    lengths = pk.request_lengths(ttok, recs, mst)
    s = pk.CurriculumStream(recs, lengths, STAGES, seed=0, pack_mult=1)
    assert s.stage_counts() == {256: 3, 512: 21, 1024: 12}
    seen, stages, mb = [], [], 4
    for _ in range(len(recs) // mb):
        batch = s.take(mb)
        stages.append(s.stage)
        assert s.pack_len == STAGES[s.stage]
        b = prism_collate(ttok, batch, mst, pack_len=s.pack_len)
        want = [len(encode(ttok, r.state, r.schema, mst).input_ids) for r in batch]
        assert int(b.attention_mask.sum()) == sum(want)             # no token dropped
        got = []                                                     # each request whole in one row
        for i in range(b.doc.shape[0]):
            real = b.attention_mask[i] == 1
            got += [int(((b.doc[i] == d) & real).sum()) for d in b.doc[i][real].unique().tolist()]
        assert sorted(got) == sorted(want)
        seen += [r.id for r in batch]
    assert Counter(seen) == Counter(r.id for r in recs)
    assert stages == sorted(stages)
    state = s.get_state()
    nxt = s.take(mb)                                             # epoch 2: short stage again
    assert s.epoch == 1 and s.stage == stages[0]                 # (256 has 3 < mb records)
    s.set_state(state)
    assert [r.id for r in s.take(mb)] == [r.id for r in nxt]


def test_build_curriculum_pool_is_the_mixture_draw(ttok):
    """train_prism pool: n records drawn like the Mixture (freeze_stream), each once per epoch."""
    s = pk.build_curriculum(ttok, FIX, 24, 48, STAGES, alpha=0.5, seed=1)
    refs = td.freeze_stream(FIX, 24, 0, alpha=0.5, seed=1, check=False)
    assert s.pool_sha == refs["sha"] and len(s.records) == 24
    ids = [r.id for r in s.take(24)]
    assert Counter(ids) == Counter(rid for _, _, rid in refs["stream"])


def test_freeze_stages(ttok):
    scan = pk.scan_parts(ttok, FIX)
    t = pk.bucket_table(scan, 48, STAGES)
    assert t["clinc150"]["records"] == {256: 0, 512: 0, 1024: 12}
    assert sum(sum(r["records"].values()) for r in t.values()) == 36
    a = pk.freeze_stages(scan, 48, STAGES, n_per_stage=8, probe_n=3, seed=0)
    b = pk.freeze_stages(scan, 48, STAGES, n_per_stage=8, probe_n=3, seed=0)
    assert a["sha"] == b["sha"] and set(a["streams"]) == {"0", "256", "512", "1024"}
    assert a["stage_sha"]["512"] == td.sha_of(a["streams"]["512"])
    lens = {(s, ln): n for s, d in scan.items()
            for ln, n in zip(d["lines"], pk.request_lengths(ttok, list(read_jsonl(f"{ROOT}/tests/fixtures/{s}.jsonl")), 48))}
    for k in STAGES:
        for s, ln, _ in a["streams"][str(k)]:
            assert pk.stage_of(lens[(s, ln)], STAGES) == STAGES.index(k)
    probe = {tuple(p) for p in a["probe"]}
    assert len(probe) == 3 and not probe & {tuple(r) for v in a["streams"].values() for r in v}
    skip = pk.freeze_stages(scan, 48, STAGES, n_per_stage=8, probe_n=0, min_records=5)
    assert skip["skipped_stages"] == {"256": 3} and "256" not in skip["streams"]


def tiny_gemma(tmp_path, tok):
    import torch
    from transformers import Gemma4TextConfig, Gemma4TextModel
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=16, sliding_window=16, vocab_size_per_layer_input=len(tok),
                           hidden_size_per_layer_input=4, num_kv_shared_layers=0,
                           pad_token_id=tok.pad_token_id or 0,
                           layer_types=["sliding_attention", "full_attention"])
    base = tmp_path / "base"
    Gemma4TextModel(cfg).save_pretrained(base)
    tok.save_pretrained(base)
    return base


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
    return r.stdout


@pytest.mark.parametrize("packing", ["on", "off"])
def test_train_prism_length_curriculum(tmp_path, tok, packing):
    """Tiny-Gemma preflight with the curriculum on (packed per stage and length-bucketed)."""
    base = tiny_gemma(tmp_path, tok)
    out = run([sys.executable, str(ROOT / "train_prism.py"), "--model", str(base), "--cpu",
               "--train", str(ROOT / "tests/fixtures/*.jsonl"), "--val", FIX[0], "--bs", "2",
               "--grad-accum", "2", "--steps", "4", "--eval-every", "4", "--eval-max", "4",
               "--max-state-tokens", "48", "--length-curriculum", "on", "--curriculum-stages",
               "256,512,1024", "--packing", packing, "--save-dir", str(tmp_path / "ck")])
    info = json.loads(next(l for l in out.splitlines() if "length_curriculum" in l))["length_curriculum"]
    assert info["records"] == 16 and sum(info["per_stage"].values()) == 16
    assert info["pack_mult"] == (1 if packing == "on" else 0)
    assert (tmp_path / "ck" / "last" / "heads.pt").exists()


def test_train_prism_curriculum_off_is_the_mixture():
    import train_prism as tp
    a = tp.parse(["--steps", "3", "--length-grouping", "off"])
    assert a.length_curriculum == "off"
    from openjev.mix import Mixture
    assert isinstance(tp.train_stream(None, FIX, a), Mixture)
    with pytest.raises(SystemExit):
        tp.parse(["--length-curriculum", "on", "--curriculum-stages", "512,256"])


def test_bench_stage_streams(tmp_path, tok):
    """bench_trials --stage-streams: each trial times its stage's frozen stream; samples/s,
    tokens/s, the stage sha and the pack_len used are recorded; packed at the stage max and
    unpacked see the same records."""
    base = tiny_gemma(tmp_path, tok)
    ss = tmp_path / "stage_streams.json"
    run([sys.executable, str(ROOT / "tools/length_curriculum.py"), "freeze", "--train",
         str(ROOT / "tests/fixtures/*.jsonl"), "--tokenizer", str(base), "--max-state", "48",
         "--stages", "256,512,1024", "--n-per-stage", "8", "--probe-n", "2", "--cache", "",
         "--out", str(ss)])
    frozen = json.loads(ss.read_text())
    bench = {"global_batch": 4, "micro_batch": 2, "warmup_steps": 1, "timed_steps": 1,
             "trial_stream_sha": frozen["sha"]}
    common = {"dtype": "fp32", "max_state_tokens": 48}
    trials = [{"trial_number": i, "params": {**common, **p}, "bench": bench}
              for i, p in enumerate([{"length_stage": 512, "stage_pack_mult": 0},
                                     {"length_stage": 512, "stage_pack_mult": 1},
                                     {"length_stage": 1024, "stage_pack_mult": 2},
                                     {"length_stage": 0, "packing": "on", "pack_len": 1024},
                                     {"length_stage": 128}])]
    tf, res = tmp_path / "trials.json", tmp_path / "res.json"
    tf.write_text(json.dumps(trials))
    run([sys.executable, str(ROOT / "bench_trials.py"), "--trials", str(tf), "--out", str(res),
         "--model", str(base), "--cpu", "--train", str(ROOT / "tests/fixtures/*.jsonl"),
         "--max-state-tokens", "48", "--stage-streams", str(ss)])
    r = {x["trial_number"]: x for x in json.loads(res.read_text())["results"]}
    for i in range(4):
        assert r[i]["status"] == "ok", r[i]
        assert r[i]["samples_per_sec"] > 0 and r[i]["tokens_per_sec"] > 0
        assert r[i]["trial_stream_sha"] == frozen["sha"]
    assert r[0]["stage_stream_sha"] == frozen["stage_sha"]["512"] and r[0]["pack_len_used"] == 0
    assert r[1]["pack_len_used"] == 512 and r[2]["pack_len_used"] == 2048
    assert r[3]["length_stage"] == 0 and r[3]["pack_len_used"] == 1024
    assert r[0]["data_ids_sha"] == r[1]["data_ids_sha"] != r[2]["data_ids_sha"]
    assert r[4]["status"] == "error" and "no frozen stream" in r[4]["error"]
