"""Optuna speed loop end to end on CPU: freeze -> propose -> bench_trials -> ingest -> report,
with a tiny random Gemma 4 (like tests/test_prism_resume.py). Every file goes to tmp_path:
a temp study, a temp yaml and temp frozen data; never runs/optuna."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

optuna = pytest.importorskip("optuna")
yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
LOOP = [sys.executable, str(ROOT / "tools" / "optuna_loop.py")]
BENCH = [sys.executable, str(ROOT / "bench_trials.py")]
FIX = str(ROOT / "tests/fixtures/*.jsonl")
REAL_YAML = ROOT / "tools" / "search_space.yaml"
# Small round-3 settings so CPU runs stay fast: micro-batch 2 x grad_accum 2 = global 4.
SMALL = dict(global_batch=4, micro_batch=2, warmup_steps=1, timed_steps=2, diag_mb=4,
             trial_stream_len=32, probe_n=4, validate_every=2, validate_steps=2, val_per_source=4)
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import optuna_loop as ol  # noqa: E402


def tiny_gemma(tmp_path, tok):
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


def space(tmp_path, small=True, name="space", **kw):
    """Temp yaml (the real dimensions; SMALL settings unless small=False) + frozen data from
    the fixtures. Returns the global optuna_loop args (--space --stream --val-set)."""
    raw = yaml.safe_load(REAL_YAML.read_text())
    if small:
        raw.update(SMALL)
    raw.update(kw)
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / "space.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    g = ["--space", str(d / "space.yaml"), "--stream", str(d / "trial_stream.json"),
         "--val-set", str(d / "val_set.json")]
    ol.main(g + ["freeze", "--train", FIX, "--val", FIX])
    return g


def bench(trials, results, base, g, *extra):
    stream = g[g.index("--stream") + 1]
    return run(BENCH + ["--trials", str(trials), "--out", str(results), "--model", str(base), "--cpu",
                        "--train", FIX, "--max-state-tokens", "48", "--trial-stream", stream, *extra])


def load(db):
    return optuna.load_study(study_name=ol.STUDY_NAME, storage=f"sqlite:///{db}")


def test_propose_bench_ingest_report(tmp_path, tok):
    """Round 3: fixed micro-batch 2 x grad_accum 2 = global 4 for every trial, the frozen
    stream (same records for every trial and every run), the baseline's diagnostic."""
    base = tiny_gemma(tmp_path, tok)
    db, trials, results, report = (tmp_path / "study.db", tmp_path / "trials.json",
                                   tmp_path / "results.json", tmp_path / "EXPERIMENTS.md")
    g = ["--db", str(db)] + space(tmp_path)
    run(LOOP + g + ["propose", "--n", "2", "--out", str(trials)])
    proposed = json.loads(trials.read_text())
    assert [t["trial_number"] for t in proposed] == [0, 1]
    p0 = proposed[0]
    assert "bs" not in p0["params"] and p0["params"]["grad_ckpt"] == "on"
    assert p0["params"]["ple_cpu"] == 0 and p0["params"]["dtype"] == "fp16"   # full config
    assert p0["bench"]["global_batch"] == 4 and p0["bench"]["micro_batch"] == 2
    assert p0["bench"]["diag_mb"] == 4 and "diag_mb" not in proposed[1]["bench"]   # baseline only
    for t in proposed:                       # CPU has no fp16 matmul for backward: run in fp32
        t["params"]["dtype"] = "fp32"
    trials.write_text(json.dumps(proposed))

    out = bench(trials, results, base, g)
    assert out.count('"optuna_trial"') == 2
    res = {r["trial_number"]: r for r in json.loads(results.read_text())["results"]}
    for r in res.values():
        assert r["status"] == "ok", r
        assert r["mb_used"] == 2 and r["grad_accum"] == 2 and r["mb_used"] * r["grad_accum"] == 4
        assert r["steps"] == 2 and r["samples"] == 8 and r["guard_steps"] == 1
        assert r["records_used"] == (1 + 2) * 4
    b0 = res[0]
    assert b0["samples_per_sec"] > 0 and b0["loss_guard"] is not None
    assert b0["mb_diag"]["mb"] == 4 and b0["mb_diag"]["fits"] is True and "mb_diag" not in res[1]
    # Same frozen records for both trials (warm-up and timed loop).
    assert res[0]["warmup_ids_sha"] == res[1]["warmup_ids_sha"]
    assert res[0]["data_ids_sha"] == res[1]["data_ids_sha"]
    assert res[0]["trial_stream_sha"] == json.loads(Path(g[g.index("--stream") + 1]).read_text())["sha"]

    run(LOOP + g + ["ingest", str(results)])
    run(LOOP + g + ["report", "--out", str(report)])
    study = load(db)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert done and done[0].user_attrs["is_baseline"]
    assert done[0].user_attrs["mb_used"] == 2 and done[0].user_attrs["grad_accum"] == 2
    assert "tokens_per_sec" in done[0].user_attrs
    assert study.user_attrs["mb4_fits_round_3"]["fits"] is True
    assert study.user_attrs["micro_batch_round_3"] == 2
    assert all(t.state != optuna.trial.TrialState.RUNNING for t in study.trials)
    text = report.read_text()
    assert "## Best so far" in text and "| 0 |" in text and "| mb_used | grad_accum |" in text
    assert "FIXED micro-batch 2" in text

    # A second round proposes new trial numbers and does not re-enqueue the baseline.
    run(LOOP + g + ["propose", "--n", "1", "--out", str(trials)])
    again = json.loads(trials.read_text())
    assert again[0]["trial_number"] == 2


def test_trial_stream_identical_across_runs(tmp_path, tok):
    """Two freezes give the same stream; two separate propose/bench runs (two studies)
    consume exactly the same records."""
    base = tiny_gemma(tmp_path, tok)
    shas, data = [], []
    for k in ("a", "b"):
        g = ["--db", str(tmp_path / f"{k}.db")] + space(tmp_path, name=k)
        shas.append(json.loads(Path(g[g.index("--stream") + 1]).read_text())["sha"])
        tj, rj = tmp_path / f"{k}.json", tmp_path / f"{k}_r.json"
        run(LOOP + g + ["propose", "--n", "1", "--out", str(tj)])
        t = json.loads(tj.read_text())
        t[0]["params"]["dtype"] = "fp32"
        tj.write_text(json.dumps(t))
        bench(tj, rj, base, g)
        r = json.loads(rj.read_text())["results"][0]
        data.append((r["warmup_ids_sha"], r["data_ids_sha"], r["records_used"]))
    assert shas[0] == shas[1] and data[0] == data[1]


def tiny_setup(tmp_path, tok):
    import bench_trials as bt
    import train_prism as tp
    from openjev import trial_data as td
    base = tiny_gemma(tmp_path, tok)
    targs = tp.parse(["--model", str(base), "--cpu", "--max-state-tokens", "48", "--train", FIX])
    torch.manual_seed(0)
    model, tk, dtype = tp.build(targs, "cpu")
    model.train()
    import glob
    frozen = td.freeze_stream(sorted(glob.glob(FIX)), 32, 4)
    data = td.FixedStream(td.load_refs(FIX, frozen["stream"]))
    probe = td.load_refs(FIX, frozen["probe"])
    own, _ = bt.parse(["--trial-seconds", "1"])
    return bt, tp, targs, model, tk, dtype, data, probe, own


def test_mb_diag_and_restore(tmp_path, tok, monkeypatch):
    """The baseline diagnostic: fits / over the 0.92 headroom / simulated OOM (CPU has no
    CUDA OOM, so the peak function raises it). Afterwards the reset restores weights, grads,
    data and RNG exactly, and a trial with the diagnostic equals one without it."""
    bt, tp, targs, model, tk, dtype, data, probe, own = tiny_setup(tmp_path, tok)
    snap, state = bt.snapshot(model), data.get_state()
    monkeypatch.setattr(bt, "mem_limit_gb", lambda device: 4.0)
    monkeypatch.setattr(bt, "peak_gb", lambda device: 3.0)
    d = bt.mb_diag(model, tk, data, 48, 4, "cpu", "default", 8)
    assert d["fits"] is True and not d["oom"] and d["peak_gb"] == 3.0
    assert data.get_state() != state                         # it did consume records
    monkeypatch.setattr(bt, "peak_gb", lambda device: 3.8)   # 3.8 > 0.92 x 4.0
    assert bt.mb_diag(model, tk, data, 48, 4, "cpu", "default", 8)["fits"] is False

    def oom(device):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    monkeypatch.setattr(bt, "peak_gb", oom)
    d = bt.mb_diag(model, tk, data, 48, 4, "cpu", "default", 8)
    assert d["oom"] is True and d["fits"] is False
    bt.reset(model, data, snap, state, targs.seed)
    after = bt.snapshot(model)
    assert after.keys() == snap.keys() and all(torch.equal(after[k], snap[k]) for k in snap)
    assert all(p.grad is None for p in model.parameters())
    assert data.get_state() == state
    torch.manual_seed(targs.seed)
    rng = torch.random.get_rng_state()
    bt.reset(model, data, snap, state, targs.seed)
    assert torch.equal(torch.random.get_rng_state(), rng)

    # No learning or data leaks from the diagnostic into the trial.
    monkeypatch.setattr(bt, "peak_gb", lambda device: None)
    st = {"fixed_mb": True, "global_batch": 4, "micro_batch": 2, "grad_accum": 2, "warmup_steps": 1,
          "timed_steps": 1, "diag_mb": None, "trial_stream_sha": None}
    params = {"grad_ckpt": "off", "dtype": "fp32", "ple_cpu": 0}
    runs = []
    for diag in (None, 4):
        r = bt.run_trial(model, tk, data, state, snap, probe, targs, params, "cpu", dtype, own,
                         torch.get_num_threads(), {"value": 2.0 ** 16}, dict(st, diag_mb=diag))
        runs.append(r)
    assert "mb_diag" in runs[1] and "mb_diag" not in runs[0]
    for k in ("loss_start", "loss_guard", "loss_end", "warmup_ids_sha", "data_ids_sha"):
        assert runs[0][k] == runs[1][k], k


def test_skipped_warmup_step_is_redone(tmp_path, tok, monkeypatch):
    """A warm-up step the fp16 GradScaler skips is redone on the same records, so the guard
    follows the same applied optimizer steps (and records) as a run without a skip."""
    bt, tp, targs, model, tk, dtype, data, probe, own = tiny_setup(tmp_path, tok)
    snap, state = bt.snapshot(model), data.get_state()
    st = {"fixed_mb": True, "global_batch": 4, "micro_batch": 2, "grad_accum": 2, "warmup_steps": 1,
          "timed_steps": 1, "diag_mb": None, "trial_stream_sha": None}
    params = {"grad_ckpt": "off", "dtype": "fp32", "ple_cpu": 0}

    def trial():
        return bt.run_trial(model, tk, data, state, snap, probe, targs, params, "cpu", dtype, own,
                            torch.get_num_threads(), {"value": 2.0 ** 16}, dict(st))
    clean = trial()
    real, calls = bt.accum_step, []

    def skip_first(model_, opt, scaler, next_batch, accum, device, backend):
        calls.append(1)
        if len(calls) == 1:                    # a skipped step: data consumed, no update
            for _ in range(accum):
                next_batch()
            opt.zero_grad(set_to_none=True)
            return torch.tensor(0.0), 1
        return real(model_, opt, scaler, next_batch, accum, device, backend)
    monkeypatch.setattr(bt, "accum_step", skip_first)
    r = trial()
    assert clean["warmup_retries"] == 0 and r["warmup_retries"] == 1
    for k in ("warmup_ids_sha", "data_ids_sha", "loss_guard", "loss_end", "records_used"):
        assert r[k] == clean[k], k
    assert r["skipped_steps"] == 0


def test_oom_at_fixed_mb_is_pruned_not_retried(tmp_path, tok, monkeypatch):
    """A trial that runs out of memory at the round micro-batch is infeasible_at_mb2 and
    PRUNED; no trial ever trains with a smaller micro-batch."""
    import bench_trials as bt
    base = tiny_gemma(tmp_path, tok)
    db, tj, rj = tmp_path / "s.db", tmp_path / "t.json", tmp_path / "r.json"
    g = ["--db", str(db)] + space(tmp_path)
    ol.main(g + ["propose", "--n", "0", "--out", str(tj),
                 "--enqueue-json", '[{"torch_threads": 2}, {"torch_threads": 1}]'])
    t = json.loads(tj.read_text())
    for e in t:
        e["params"]["dtype"] = "fp32"
    tj.write_text(json.dumps(t))
    sizes = []
    real_build = bt.tp.build

    def build(*a, **k):
        model, tk, dtype = real_build(*a, **k)
        fwd = model.flag_logits

        def flag_logits(b):
            if model.training:
                sizes.append(b.input_ids.shape[0])
                if torch.get_num_threads() == 1:       # the "memory-hungry" config
                    raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
            return fwd(b)
        model.flag_logits = flag_logits
        return model, tk, dtype
    monkeypatch.setattr(bt.tp, "build", build)
    stream = g[g.index("--stream") + 1]
    bt.main(["--trials", str(tj), "--out", str(rj), "--model", str(base), "--cpu", "--train", FIX,
             "--max-state-tokens", "48", "--trial-stream", stream])
    res = {r["trial_number"]: r for r in json.loads(rj.read_text())["results"]}
    assert res[1]["status"] == "ok" and res[2]["status"] == "infeasible_at_mb2", res[2]
    assert res[2]["mb_used"] == 2
    # training forwards: micro-batch 2 everywhere, except the baseline's diagnostic (4)
    assert set(sizes) <= {2, 4} and sizes.count(4) == 1
    ol.main(g + ["ingest", str(rj)])
    by = {x.number: x for x in load(db).trials}
    assert by[2].state == optuna.trial.TrialState.PRUNED and by[2].user_attrs["status"] == "infeasible_at_mb2"
    assert by[1].state == optuna.trial.TrialState.COMPLETE


def test_tpe_never_samples_fixed_dims(tmp_path):
    """bs, dtype and ple_cpu have sample: false (checked with yaml.safe_load) and TPE never
    draws them; trials.json still carries the full config (ple_cpu, dtype), never bs.
    The explicit ple_cpu pair differs only in ple_cpu, both at the fixed micro-batch."""
    raw = yaml.safe_load(REAL_YAML.read_text())
    by = {d["name"]: d for d in raw["dimensions"]}
    for n in ("bs", "dtype", "ple_cpu", "profile"):
        assert by[n]["sample"] is False and by[n]["implemented"] is True, n
    assert raw["micro_batch"] == 8 and raw["global_batch"] == 32
    db, tj = tmp_path / "s.db", tmp_path / "t.json"
    g = ["--db", str(db)] + space(tmp_path, small=False)
    ol.main(g + ["propose", "--n", "12", "--out", str(tj),
                 "--enqueue-json", '[{"ple_cpu": 0, "_idea": "PLE pair: GPU"}, '
                                   '{"ple_cpu": 1, "_idea": "PLE pair: CPU"}]'])
    entries = json.loads(tj.read_text())
    study = load(db)
    for t in study.trials:
        assert not {"bs", "dtype", "ple_cpu"} & set(t.params) or t.user_attrs.get("enqueued"), t.params
    tpe = [t for t in study.trials if t.user_attrs["source"] == "tpe"]
    assert len(tpe) == 11 and all(not {"bs", "dtype", "ple_cpu"} & set(t.params) for t in tpe)
    for e in entries:
        assert "bs" not in e["params"] and e["params"]["dtype"] == "fp16" and "ple_cpu" in e["params"]
        assert e["bench"]["micro_batch"] == 8 and e["bench"]["global_batch"] == 32
    pair = entries[1:3]
    assert pair[0]["params"]["ple_cpu"] == 0 and pair[1]["params"]["ple_cpu"] == 1
    assert {k: v for k, v in pair[0]["params"].items() if k != "ple_cpu"} == \
           {k: v for k, v in pair[1]["params"].items() if k != "ple_cpu"} == \
           {k: v for k, v in entries[0]["params"].items() if k != "ple_cpu"}   # = the baseline


def test_report_importances_and_guard(tmp_path):
    """Synthetic trials: importances after 10 complete; the quality guard prunes."""
    db = tmp_path / "s.db"
    space_ = ol.load_space(REAL_YAML)
    impl = ol.implemented(space_)
    assert {d["name"] for d in impl} == {"bs", "grad_ckpt", "max_state_tokens", "torch_threads",
                                         "sdpa_backend", "ple_cpu", "attn_backend", "profile",
                                         "dtype"}
    assert [d["name"] for d in impl if not d["sample"]] == ["bs", "ple_cpu", "dtype", "profile"]
    assert all(not d["implemented"] for d in space_ if d["name"] in ("quant", "ddp", "packing"))
    trials = tmp_path / "t.json"
    g = ["--db", str(db)] + space(tmp_path, small=False)
    ol.main(g + ["propose", "--n", "12", "--out", str(trials)])
    res = []
    for t in json.loads(trials.read_text()):
        p = t["params"]
        res.append({"trial_number": t["trial_number"], "params": p, "status": "ok",
                    "samples_per_sec": 0.001 * p["max_state_tokens"] + (0.05 if p["grad_ckpt"] == "off" else 0),
                    "loss_start": 2.0, "loss_end": 1.5, "tokens_per_sec": 100.0})
    res[1].update(loss_guard=2.4)                  # loss went up after the guard steps: pruned
    res[3].update(loss_guard=1.5, loss_end=1.95)   # slow config, fewer timed steps: still kept
    res[2].update(status="oom", samples_per_sec=None)
    (tmp_path / "r.json").write_text(json.dumps({"meta": {}, "results": res}))
    ol.main(g + ["ingest", str(tmp_path / "r.json")])
    st = {t.number: t.state for t in load(db).trials}
    assert st[1] == st[2] == optuna.trial.TrialState.PRUNED
    assert st[3] == optuna.trial.TrialState.COMPLETE
    assert sum(s == optuna.trial.TrialState.COMPLETE for s in st.values()) == 10
    out = tmp_path / "E.md"
    ol.main(g + ["report", "--out", str(out)])
    text = out.read_text()
    assert "## Parameter importances" in text and "| max_state_tokens |" in text
    assert "Ideas not implemented yet" in text and "`quant`" in text


def test_round2_enqueue_profile_and_backends(tmp_path, tok):
    """--enqueue-json exact configs -> bench (profile table, chunked_math, sdpa_bool, flex
    unsupported on CPU) -> ingest -> report lists the profile under "## Profiles"."""
    base = tiny_gemma(tmp_path, tok)
    db, trials, results, report = (tmp_path / "study.db", tmp_path / "trials.json",
                                   tmp_path / "results.json", tmp_path / "EXPERIMENTS.md")
    g = ["--db", str(db)] + space(tmp_path)
    run(LOOP + g + ["propose", "--n", "0", "--out", str(trials),
                    "--enqueue-json", '{"attn_backend": "chunked_math", "profile": 1, "dtype": "fp32"}',
                    "--enqueue-json", '[{"attn_backend": "flex", "dtype": "fp32"}, '
                                      '{"attn_backend": "sdpa_bool", "dtype": "fp32"}]'])
    proposed = json.loads(trials.read_text())
    # trial 0 is the enqueued baseline, then the three exact configs in order
    assert [t["params"].get("attn_backend") for t in proposed] == [
        "sdpa_default", "chunked_math", "flex", "sdpa_bool"]
    assert proposed[1]["params"]["profile"] == 1 and proposed[2]["params"]["profile"] == 0
    assert proposed[1]["params"]["ple_cpu"] == 0 and proposed[1]["params"]["grad_ckpt"] == "on"
    assert proposed[0]["params"]["dtype"] == "fp16" and proposed[1]["params"]["dtype"] == "fp32"
    proposed = proposed[1:]                 # the baseline is not needed here: it fails as "no result"
    trials.write_text(json.dumps(proposed))
    bench(trials, results, base, g, "--attn-chunk", "16")
    res = {r["params"]["attn_backend"]: r for r in json.loads(results.read_text())["results"]}
    assert res["chunked_math"]["status"] == "ok", res["chunked_math"]
    assert res["sdpa_bool"]["status"] == "ok", res["sdpa_bool"]
    assert res["flex"]["status"] == "unsupported", res["flex"]
    prof = res["chunked_math"]["profile"]
    assert res["chunked_math"]["is_profile"] and "profile" not in res["sdpa_bool"]
    assert {"attention_fwd", "masks", "ple_lookup", "collate", "matmul_ops"} <= set(prof["shares"])
    assert 0 < len(prof["top_cpu"]) <= 25 and prof["steps"] == 2

    run(LOOP + g + ["ingest", str(results)])
    run(LOOP + g + ["report", "--out", str(report)])
    by = {t.number: t for t in load(db).trials}
    assert by[1].state == optuna.trial.TrialState.COMPLETE and by[1].user_attrs["is_profile"]
    assert by[1].user_attrs["profile"]["top_cpu"]
    assert by[2].state == optuna.trial.TrialState.PRUNED
    text = report.read_text()
    assert "## Profiles" in text and "### Trial 1 (chunked_math" in text and "(profile)" in text


def test_unknown_param_fails_trial(tmp_path, tok):
    """Legacy entry (no study settings, no stream): one optimizer step per micro-batch."""
    base = tiny_gemma(tmp_path, tok)
    trials = tmp_path / "trials.json"
    trials.write_text(json.dumps([{"trial_number": 0, "params": {"bs": 2, "not_coded": 1},
                                   "reload_keys": []}]))
    results = tmp_path / "r.json"
    run(BENCH + ["--trials", str(trials), "--trial-seconds", "1", "--out", str(results), "--probe-n", "2",
                 "--model", str(base), "--cpu", "--train", FIX, "--max-state-tokens", "32"])
    r = json.loads(results.read_text())["results"][0]
    assert r["status"] == "error" and "not_coded" in r["error"]


def test_enqueue_json_is_validated(tmp_path):
    impl = ol.implemented(ol.load_space(REAL_YAML))
    full = ol.parse_enqueue(['{"bs": 8}'], impl)[0]
    assert full["bs"] == 8 and full["ple_cpu"] == 0 and full["profile"] == 0 and full["dtype"] == "fp16"
    with pytest.raises(SystemExit):
        ol.parse_enqueue(['{"nope": 1}'], impl)
    with pytest.raises(SystemExit):
        ol.parse_enqueue(['{"attn_backend": "xformers"}'], impl)


def test_round1_study_still_works(tmp_path):
    """A study made with the round-1 space (no attn_backend/profile/dtype, bs and ple_cpu
    sampled, ple_cpu default 1) still proposes, ingests and reports with the round-3 space;
    ingest uses each trial's own round baseline, also after the yaml moved on."""
    raw = yaml.safe_load(REAL_YAML.read_text())
    old = [d for d in raw["dimensions"] if d["name"] not in ("attn_backend", "profile", "dtype")]
    for d in old:
        d.pop("sample", None)
        if d["name"] == "ple_cpu":
            d["default"] = 1
    old_yaml = tmp_path / "old.yaml"
    old_yaml.write_text(yaml.safe_dump({"dimensions": old}))
    db, t1, t2 = tmp_path / "s.db", tmp_path / "t1.json", tmp_path / "t2.json"
    g = ["--db", str(db)] + space(tmp_path)
    g_old = ["--db", str(db), "--space", str(old_yaml)]

    def fake(trials_file, out):
        res = [{"trial_number": t["trial_number"], "params": t["params"], "status": "ok",
                "samples_per_sec": 0.1 * t["params"].get("bs", 4), "loss_start": 2.0, "loss_guard": 1.6,
                "loss_end": 1.5} for t in json.loads(trials_file.read_text())]
        out.write_text(json.dumps({"meta": {}, "results": res}))

    ol.main(g_old + ["propose", "--n", "3", "--out", str(t1)])
    assert all("attn_backend" not in t["params"] and "bench" not in t for t in json.loads(t1.read_text()))
    fake(t1, tmp_path / "r1.json")
    ol.main(g_old + ["ingest", str(tmp_path / "r1.json")])
    ol.main(g + ["propose", "--n", "3", "--out", str(t2), "--enqueue-json", '{"profile": 1}'])
    new = json.loads(t2.read_text())
    # round 3 gets its own baseline first, then the enqueued config, then TPE; the
    # baseline counts toward --n
    assert [t["trial_number"] for t in new] == [3, 4, 5, 6]
    assert new[0]["params"]["ple_cpu"] == 0 and new[0]["params"]["dtype"] == "fp16"
    assert all("attn_backend" in t["params"] for t in new) and new[1]["params"]["profile"] == 1
    fake(t2, tmp_path / "r2.json")
    r2 = json.loads((tmp_path / "r2.json").read_text())
    r2["results"][0]["loss_guard"] = 1.0     # round-3 baseline drop 1.0 -> need >= 0.5
    # trial 5 drops 0.4: enough against round 1's baseline (0.4), not against round 3's
    (tmp_path / "r2.json").write_text(json.dumps(r2))
    # The yaml has moved on to round 4: ingest must still use the trials' round (3).
    g4 = ["--db", str(db)] + space(tmp_path, name="r4", round=4)
    ol.main(g4 + ["ingest", str(tmp_path / "r2.json")])
    by = {t.number: t for t in load(db).trials}
    assert by[3].user_attrs["is_baseline"] and by[3].user_attrs["round"] == 3
    assert by[5].state == optuna.trial.TrialState.PRUNED and "baseline 1.000" in by[5].user_attrs["reason"]
    out = tmp_path / "E.md"
    out.write_text("old\n<!-- manual: kept by report -->\n\n## My notes\nkeep me\n")
    ol.main(g + ["report", "--out", str(out)])
    text = out.read_text()
    assert "| ple_cpu | attn_backend | profile | samples/s |" in text
    assert "| 0 | complete (base r1) |" in text and "| 3 | complete (base r3) |" in text
    assert "## Profiles" not in text and text.rstrip().endswith("keep me") and text.startswith("# OpenJev")


def test_auto_dtype_needs_native_bf16(monkeypatch):
    """OJ-081: --dtype auto must not pick emulated bf16 on a T4 (sm75)."""
    import train_prism as tp
    import bench_trials as bt
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)  # emulation counts
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    assert tp.auto_dtype("cuda") == torch.float16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    assert tp.auto_dtype("cuda") == torch.bfloat16
    assert tp.auto_dtype("cpu") == torch.float32
    targs = tp.parse([])
    assert bt.build_args(targs, {"dtype": "bf16_emulated", "ple_cpu": 0}).dtype == "bf16"
    assert bt.build_args(targs, {"dtype": "fp16"}).dtype == "fp16"
    with pytest.raises(ValueError):
        bt.build_args(targs, {"dtype": "int8"})


def test_fixed_conditions(tmp_path, tok):
    """The fixed-conditions block: study attr + note, per-trial version, facts that match
    train_prism (LoRA, frozen embeddings, fp32 trainables)."""
    import train_prism as tp
    from optuna_dashboard import get_note
    db, tj = tmp_path / "s.db", tmp_path / "t.json"
    g = ["--db", str(db)] + space(tmp_path, small=False)
    ol.main(g + ["propose", "--n", "1", "--out", str(tj)])
    study = load(db)
    c = study.user_attrs["fixed_conditions"]
    stream = json.loads(Path(g[g.index("--stream") + 1]).read_text())
    assert c["trial_stream_sha"] == stream["sha"] == study.user_attrs["trial_stream_sha"]
    assert c["val_set_sha"] == study.user_attrs["val_set_sha"]
    assert c["ple"].startswith("PENDING") and "micro-batch 8, global batch 32, grad_accum 4" in c["batch"]
    assert c["max_state_tokens"].startswith("SAMPLED") and "6df0efe" in c["prism_mask"]
    assert "2cf488f" in c["loader"] and "fp16" in c["dtype"]
    ver = ol.short_sha(c)
    assert all(t.user_attrs["fixed_conditions_version"] == ver for t in study.trials)
    assert study.user_attrs["fixed_conditions_round_3"] == ver
    note = get_note(study)
    assert "### Fixed conditions" in note and "FIXED micro-batch 8 and global batch 32" in note
    assert "infeasible_at_mb8" in note and ol.LEGACY_STUDY in note
    assert study.study_name == yaml.safe_load(REAL_YAML.read_text())["study_name"] == "openjev-speed-v2"
    # The facts the block states.
    args = tp.parse([])
    assert (args.lora_r, args.lora_alpha, tp.LORA_TARGETS) == (ol.LORA["r"], ol.LORA["alpha"], ol.LORA["targets"])
    base = tiny_gemma(tmp_path, tok)
    model, _, _ = tp.build(tp.parse(["--model", str(base), "--cpu"]), "cpu")
    assert not model.get_input_embeddings().weight.requires_grad
    assert model.template_delta.requires_grad and model.template_delta.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    # A changed setting gives another version.
    st2 = dict(ol.load_settings(REAL_YAML), micro_batch=4)
    assert ol.short_sha(ol.fixed_conditions(ol.load_space(REAL_YAML), st2, "x", "y")) != ver


def test_validate_mode(tmp_path, tok):
    """Quality checkpoint: validate-plan (due / not due) -> bench --validate (N=2 steps,
    small val set) -> ingest-validate attrs + note -> report section."""
    from optuna_dashboard import get_note
    base = tiny_gemma(tmp_path, tok)
    db, tj, vj, rj = tmp_path / "s.db", tmp_path / "t.json", tmp_path / "v.json", tmp_path / "vr.json"
    g = ["--db", str(db)] + space(tmp_path)
    ol.main(g + ["propose", "--n", "3", "--out", str(tj)])
    res = [{"trial_number": t["trial_number"], "params": t["params"], "status": "ok",
            "samples_per_sec": 0.2 + 0.1 * i, "loss_start": 2.0, "loss_guard": 1.6, "loss_end": 1.5}
           for i, t in enumerate(json.loads(tj.read_text()))]
    (tmp_path / "r.json").write_text(json.dumps({"meta": {}, "results": res[:1]}))
    ol.main(g + ["ingest", str(tmp_path / "r.json"), "--no-fail-missing"])
    ol.main(g + ["validate-plan", "--out", str(vj)])          # 1 complete < validate_every 2
    assert not vj.exists()
    (tmp_path / "r.json").write_text(json.dumps({"meta": {}, "results": res[1:]}))
    ol.main(g + ["ingest", str(tmp_path / "r.json")])
    ol.main(g + ["validate-plan", "--out", str(vj)])
    plan = json.loads(vj.read_text())
    assert [c["name"] for c in plan["configs"]] == ["best", "baseline"]
    assert plan["configs"][0]["trial_number"] == 2 and plan["configs"][1]["trial_number"] == 0
    assert plan["settings"]["steps"] == 2 and plan["settings"]["micro_batch"] == 2
    for c in plan["configs"]:
        c["params"]["dtype"] = "fp32"
    vj.write_text(json.dumps(plan))
    vset = g[g.index("--val-set") + 1]
    run(BENCH + ["--validate", str(vj), "--val-set", vset, "--out", str(rj), "--model", str(base), "--cpu",
                 "--train", FIX, "--val", FIX, "--max-state-tokens", "48"])
    out = json.loads(rj.read_text())
    assert [r["name"] for r in out["results"]] == ["best", "baseline"]
    for r in out["results"]:
        assert r["status"] == "ok", r
        assert r["steps"] == 2 and r["grad_accum"] == 2 and {"acc", "nll", "ece", "per_source"} <= set(r["val"])
    assert out["meta"]["val_n"] == len(json.loads(Path(vset).read_text())["records"])
    ol.main(g + ["ingest-validate", str(rj)])
    study = load(db)
    u = {t.number: t for t in study.trials}[2].user_attrs
    for k in ("val_acc", "val_nll", "val_ece", "val_per_source", "val_vs_baseline"):
        assert k in u, k
    assert isinstance(u["quality_reject"], bool) and u["quality_accept"] == (not u["quality_reject"])
    assert u["val_vs_baseline"]["baseline_trial"] == 0
    assert "val_nll" in {t.number: t for t in study.trials}[0].user_attrs
    assert study.user_attrs["last_validation"]["n_complete"] == 3
    assert "### Quality checkpoint" in get_note(ol.handle(study, {t.number: t for t in study.trials}[2]))
    ol.main(g + ["validate-plan", "--out", str(tmp_path / "v2.json")])   # not due again
    assert not (tmp_path / "v2.json").exists()
    rep = tmp_path / "E.md"
    ol.main(g + ["report", "--out", str(rep)])
    assert "## Quality checkpoints" in rep.read_text()
    assert out["meta"]["train_refs_sha"] and out["results"][0]["train_loss_first"] is not None


def test_abandon_marks_running_fail(tmp_path):
    db = tmp_path / "s.db"
    g = ["--db", str(db)] + space(tmp_path)
    ol.main(g + ["propose", "--n", "2", "--out", str(tmp_path / "t.json")])
    ol.main(g + ["abandon"])
    st = load(db)
    assert [t.state for t in st.trials] == [optuna.trial.TrialState.FAIL] * 2
    assert st.trials[0].user_attrs["reason"].startswith("abandoned")


def test_self_describing_trials(tmp_path):
    """Attrs + dashboard notes at propose, result appended at ingest, idempotent backfill.
    Temp study only."""
    from optuna_dashboard import get_note
    db, tj = tmp_path / "s.db", tmp_path / "t.json"
    g = ["--db", str(db)] + space(tmp_path, small=False)
    ol.main(g + ["propose", "--n", "2", "--out", str(tj),
                 "--enqueue-json", '{"bs": 2, "_idea": "sdpa_bool", "_hypothesis": "faster path", '
                                   '"_reference": "https://example.org/x"}'])
    proposed = json.loads(tj.read_text())
    assert all("_idea" not in t["params"] for t in proposed) and len(proposed) == 3
    study = load(db)
    assert study.metric_names == [ol.METRIC_NAME]
    assert study.user_attrs["kernel"] == ol.KERNEL and study.user_attrs["baseline_s_per_sample"] == 4.5
    assert "search_space_round_3" in study.user_attrs and "Objective" in get_note(study)
    by = {t.number: t for t in study.trials}
    assert by[0].user_attrs["source"] == "baseline" and by[1].user_attrs["source"] == "enqueued"
    assert by[2].user_attrs["source"] == "tpe" and by[2].user_attrs["idea"] == "TPE exploration"
    u = by[1].user_attrs
    assert u["idea"] == "sdpa_bool" and u["hypothesis"] == "faster path" and u["round"] == 3
    assert u["reference"] == "https://example.org/x" and u["git_commit"]
    h = ol.handle(study, by[1])
    note = get_note(h)
    assert "sdpa_bool" in note and "faster path" in note and "bs=2" in note and "### Result" not in note

    res = [{"trial_number": t["trial_number"], "params": t["params"], "status": "ok",
            "samples_per_sec": 0.25, "loss_start": 2.0, "loss_guard": 1.5, "loss_end": 1.4,
            "peak_mem_gb": 7.5, "mb_used": 8, "grad_accum": 4, "global_batch": 32,
            "profile": {"top_cuda": [[f"op{i}", 1.0 + i, 0.1, 3] for i in range(8)]}}
           for t in proposed]
    (tmp_path / "r.json").write_text(json.dumps({"meta": {}, "results": res}))
    ol.main(g + ["ingest", str(tmp_path / "r.json")])
    study = load(db)
    note = get_note(ol.handle(study, {t.number: t for t in study.trials}[1]))
    assert "### Result" in note and "samples/s: 0.250" in note and "1.12x" in note
    assert "7.50 GB" in note and "`op4`" in note and "`op5`" not in note
    assert "micro-batch 8 x grad_accum 4 = global batch 32" in note

    # backfill: idempotent, keeps states and params
    before = [(t.number, t.state, t.params) for t in study.trials]
    ol.main(g + ["annotate", "--backfill"])
    s1 = load(db)
    snap = ([(t.number, t.user_attrs) for t in s1.trials], get_note(s1),
            [get_note(ol.handle(s1, t)) for t in s1.trials])
    ol.main(g + ["annotate", "--backfill"])
    s2 = load(db)
    assert snap == ([(t.number, t.user_attrs) for t in s2.trials], get_note(s2),
                    [get_note(ol.handle(s2, t)) for t in s2.trials])
    assert before == [(t.number, t.state, t.params) for t in s2.trials]
    # BACKFILL is for the legacy study only: a v2 trial keeps its round and gets no r1 caveat.
    assert s2.trials[0].user_attrs["round"] == 3 and "OLD Prism mask" not in get_note(ol.handle(s2, s2.trials[0]))
    assert s2.trials[2].user_attrs["source"] == "tpe" and s2.study_name == "openjev-speed-v2"
    assert ol.LEGACY_STUDY in get_note(s2)

    ol.main(g + ["annotate", "--trial", "2", "--idea", "manual", "--reference", "r"])
    s3 = load(db)
    assert s3.trials[2].user_attrs["idea"] == "manual" and "manual" in get_note(ol.handle(s3, s3.trials[2]))
    out = tmp_path / "E.md"
    ol.main(g + ["report", "--out", str(out)])
    assert "| # | status | idea | source |" in out.read_text()


def test_propose_refuses_changed_stream(tmp_path):
    """The frozen data is fixed per study: propose refuses a stream whose sha differs."""
    db = tmp_path / "s.db"
    g = ["--db", str(db)] + space(tmp_path)
    ol.main(g + ["propose", "--n", "1", "--out", str(tmp_path / "t.json"), "--abandon-running"])
    sp = Path(g[g.index("--stream") + 1])
    d = json.loads(sp.read_text())
    d["sha"] = "0" * 16
    sp.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="must not change"):
        ol.main(g + ["propose", "--n", "1", "--out", str(tmp_path / "t.json"), "--abandon-running"])
    with pytest.raises(SystemExit, match="exists"):
        ol.main(g + ["freeze", "--train", FIX, "--val", FIX])
