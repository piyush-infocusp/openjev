"""Optuna speed loop end to end on CPU: propose -> bench_trials -> ingest -> report,
with a tiny random Gemma 4 (like tests/test_prism_resume.py). Every file goes to tmp_path."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

optuna = pytest.importorskip("optuna")
pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
LOOP = [sys.executable, str(ROOT / "tools" / "optuna_loop.py")]


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


def test_propose_bench_ingest_report(tmp_path, tok):
    base = tiny_gemma(tmp_path, tok)
    db, trials, results, report = (tmp_path / "study.db", tmp_path / "trials.json",
                                   tmp_path / "results.json", tmp_path / "EXPERIMENTS.md")
    g = ["--db", str(db)]
    run(LOOP + g + ["propose", "--n", "2", "--out", str(trials)])
    proposed = json.loads(trials.read_text())
    assert [t["trial_number"] for t in proposed] == [0, 1]
    assert proposed[0]["params"]["bs"] == 4 and proposed[0]["params"]["grad_ckpt"] == "on"

    out = run([sys.executable, str(ROOT / "bench_trials.py"), "--trials", str(trials),
               "--trial-seconds", "3", "--out", str(results), "--probe-n", "4",
               "--model", str(base), "--cpu", "--train", str(ROOT / "tests/fixtures/*.jsonl"),
               "--max-state-tokens", "64"])
    assert out.count('"optuna_trial"') == 2
    res = json.loads(results.read_text())["results"]
    assert {r["trial_number"] for r in res} == {0, 1}
    base_r = next(r for r in res if r["trial_number"] == 0)
    assert base_r["status"] == "ok", base_r
    assert base_r["samples_per_sec"] > 0 and base_r["steps"] >= 1
    assert base_r["loss_guard"] is not None and base_r["guard_steps"] == 2

    run(LOOP + g + ["ingest", str(results)])
    run(LOOP + g + ["report", "--out", str(report)])
    study = optuna.load_study(study_name="openjev-speed", storage=f"sqlite:///{db}")
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert done and done[0].user_attrs["is_baseline"]
    assert "tokens_per_sec" in done[0].user_attrs
    assert all(t.state != optuna.trial.TrialState.RUNNING for t in study.trials)
    text = report.read_text()
    assert report.exists() and "## Best so far" in text and "| 0 |" in text

    # A second round proposes new trial numbers and does not re-enqueue the baseline.
    run(LOOP + g + ["propose", "--n", "1", "--out", str(trials)])
    again = json.loads(trials.read_text())
    assert again[0]["trial_number"] == 2


def test_unknown_param_fails_trial(tmp_path, tok):
    base = tiny_gemma(tmp_path, tok)
    trials = tmp_path / "trials.json"
    trials.write_text(json.dumps([{"trial_number": 0, "params": {"bs": 2, "not_coded": 1},
                                   "reload_keys": []}]))
    results = tmp_path / "r.json"
    run([sys.executable, str(ROOT / "bench_trials.py"), "--trials", str(trials),
         "--trial-seconds", "1", "--out", str(results), "--probe-n", "2", "--model", str(base),
         "--cpu", "--train", str(ROOT / "tests/fixtures/*.jsonl"), "--max-state-tokens", "32"])
    r = json.loads(results.read_text())["results"][0]
    assert r["status"] == "error" and "not_coded" in r["error"]


def test_report_importances_and_guard(tmp_path):
    """Synthetic trials: importances after 10 complete; the quality guard prunes."""
    sys.path.insert(0, str(ROOT / "tools"))
    import optuna_loop as ol
    db = tmp_path / "s.db"
    space = ol.load_space(ROOT / "tools" / "search_space.yaml")
    impl = ol.implemented(space)
    assert {d["name"] for d in impl} == {"bs", "grad_ckpt", "max_state_tokens", "torch_threads",
                                         "sdpa_backend", "ple_cpu", "attn_backend", "profile",
                                         "dtype"}
    assert [d["name"] for d in impl if not d["sample"]] == ["profile"]
    assert all(not d["implemented"] for d in space if d["name"] in ("quant", "ddp", "packing"))
    trials = tmp_path / "t.json"
    ol.main(["--db", str(db), "propose", "--n", "12", "--out", str(trials)])
    res = []
    for t in json.loads(trials.read_text()):
        p = t["params"]
        res.append({"trial_number": t["trial_number"], "params": p, "status": "ok",
                    "samples_per_sec": 0.1 * p["bs"] + (0.05 if p["grad_ckpt"] == "off" else 0),
                    "loss_start": 2.0, "loss_end": 1.5, "tokens_per_sec": 100.0})
    res[1].update(loss_guard=2.4)                  # loss went up after the guard steps: pruned
    res[3].update(loss_guard=1.5, loss_end=1.95)   # slow config, fewer timed steps: still kept
    res[2].update(status="oom", samples_per_sec=None)
    (tmp_path / "r.json").write_text(json.dumps({"meta": {}, "results": res}))
    ol.main(["--db", str(db), "ingest", str(tmp_path / "r.json")])
    study = optuna.load_study(study_name=ol.STUDY_NAME, storage=f"sqlite:///{db}")
    st = {t.number: t.state for t in study.trials}
    assert st[1] == st[2] == optuna.trial.TrialState.PRUNED
    assert st[3] == optuna.trial.TrialState.COMPLETE
    assert sum(s == optuna.trial.TrialState.COMPLETE for s in st.values()) == 10
    out = tmp_path / "E.md"
    ol.main(["--db", str(db), "report", "--out", str(out)])
    text = out.read_text()
    assert "## Parameter importances" in text and "| bs |" in text
    assert "Ideas not implemented yet" in text and "`quant`" in text


def test_round2_enqueue_profile_and_backends(tmp_path, tok):
    """--enqueue-json exact configs -> bench (profile table, chunked_math, sdpa_bool, flex
    unsupported on CPU) -> ingest -> report lists the profile under "## Profiles"."""
    base = tiny_gemma(tmp_path, tok)
    db, trials, results, report = (tmp_path / "study.db", tmp_path / "trials.json",
                                   tmp_path / "results.json", tmp_path / "EXPERIMENTS.md")
    g = ["--db", str(db)]
    run(LOOP + g + ["propose", "--n", "0", "--out", str(trials),
                    "--enqueue-json", '{"bs": 2, "attn_backend": "chunked_math", "profile": 1, '
                                      '"dtype": "fp32"}',
                    "--enqueue-json", '[{"bs": 2, "attn_backend": "flex", "dtype": "fp32"}, '
                                      '{"bs": 2, "attn_backend": "sdpa_bool", "dtype": "fp32"}]'])
    proposed = json.loads(trials.read_text())
    # trial 0 is the enqueued baseline, then the three exact configs in order
    assert [t["params"].get("attn_backend") for t in proposed] == [
        "sdpa_default", "chunked_math", "flex", "sdpa_bool"]
    assert proposed[1]["params"]["profile"] == 1 and "profile" not in proposed[2]["params"]
    assert proposed[1]["params"]["ple_cpu"] == 0 and proposed[1]["params"]["grad_ckpt"] == "on"
    assert proposed[0]["params"]["dtype"] == "fp16" and proposed[1]["params"]["dtype"] == "fp32"
    proposed = proposed[1:]                 # the baseline is not needed here: it fails as "no result"
    trials.write_text(json.dumps(proposed))
    run([sys.executable, str(ROOT / "bench_trials.py"), "--trials", str(trials),
         "--trial-seconds", "1", "--out", str(results), "--probe-n", "2", "--model", str(base),
         "--cpu", "--train", str(ROOT / "tests/fixtures/*.jsonl"), "--max-state-tokens", "48",
         "--attn-chunk", "16"])
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
    study = optuna.load_study(study_name="openjev-speed", storage=f"sqlite:///{db}")
    by = {t.number: t for t in study.trials}
    assert by[1].state == optuna.trial.TrialState.COMPLETE and by[1].user_attrs["is_profile"]
    assert by[1].user_attrs["profile"]["top_cpu"]
    assert by[2].state == optuna.trial.TrialState.PRUNED
    text = report.read_text()
    assert "## Profiles" in text and "### Trial 1 (chunked_math" in text and "(profile)" in text


def test_enqueue_json_is_validated(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import optuna_loop as ol
    impl = ol.implemented(ol.load_space(ROOT / "tools" / "search_space.yaml"))
    full = ol.parse_enqueue(['{"bs": 8}'], impl)[0]
    assert full["bs"] == 8 and full["ple_cpu"] == 0 and full["profile"] == 0 and full["dtype"] == "fp16"
    with pytest.raises(SystemExit):
        ol.parse_enqueue(['{"nope": 1}'], impl)
    with pytest.raises(SystemExit):
        ol.parse_enqueue(['{"attn_backend": "xformers"}'], impl)


def test_round1_study_still_works(tmp_path):
    """A study made with the round-1 space (no attn_backend/profile, ple_cpu default 1)
    still proposes, ingests and reports with the round-2 space."""
    import yaml
    sys.path.insert(0, str(ROOT / "tools"))
    import optuna_loop as ol
    space = yaml.safe_load((ROOT / "tools" / "search_space.yaml").read_text())
    old = [d for d in space["dimensions"] if d["name"] not in ("attn_backend", "profile", "dtype")]
    for d in old:
        if d["name"] == "ple_cpu":
            d["default"] = 1
    old_yaml = tmp_path / "old.yaml"
    old_yaml.write_text(yaml.safe_dump({"dimensions": old}))
    db, t1, t2 = tmp_path / "s.db", tmp_path / "t1.json", tmp_path / "t2.json"

    def fake(trials_file, out):
        res = [{"trial_number": t["trial_number"], "params": t["params"], "status": "ok",
                "samples_per_sec": 0.1 * t["params"]["bs"], "loss_start": 2.0, "loss_guard": 1.6,
                "loss_end": 1.5} for t in json.loads(trials_file.read_text())]
        out.write_text(json.dumps({"meta": {}, "results": res}))

    ol.main(["--db", str(db), "--space", str(old_yaml), "propose", "--n", "3", "--out", str(t1)])
    assert all("attn_backend" not in t["params"] for t in json.loads(t1.read_text()))
    fake(t1, tmp_path / "r1.json")
    ol.main(["--db", str(db), "--space", str(old_yaml), "ingest", str(tmp_path / "r1.json")])
    ol.main(["--db", str(db), "propose", "--n", "3", "--out", str(t2),
             "--enqueue-json", '{"profile": 1}'])
    new = json.loads(t2.read_text())
    # round 2 gets its own baseline first (round 1's ran under the old mask/defaults),
    # then the enqueued config, then TPE; the baseline counts toward --n
    assert [t["trial_number"] for t in new] == [3, 4, 5, 6]
    assert new[0]["params"]["ple_cpu"] == 0 and new[0]["params"]["dtype"] == "fp16"
    assert all("attn_backend" in t["params"] for t in new) and new[1]["params"]["profile"] == 1
    fake(t2, tmp_path / "r2.json")
    r2 = json.loads((tmp_path / "r2.json").read_text())
    r2["results"][0]["loss_guard"] = 1.0     # round-2 baseline drop 1.0 -> need >= 0.5
    # trial 5 drops 0.4: enough against round 1's baseline (0.4), not against round 2's
    (tmp_path / "r2.json").write_text(json.dumps(r2))
    ol.main(["--db", str(db), "ingest", str(tmp_path / "r2.json")])
    study = optuna.load_study(study_name=ol.STUDY_NAME, storage=f"sqlite:///{db}")
    by = {t.number: t for t in study.trials}
    assert by[3].user_attrs["is_baseline"] and by[3].user_attrs["round"] == 2
    assert by[5].state == optuna.trial.TrialState.PRUNED and "baseline 1.000" in by[5].user_attrs["reason"]
    out = tmp_path / "E.md"
    out.write_text("old\n<!-- manual: kept by report -->\n\n## My notes\nkeep me\n")
    ol.main(["--db", str(db), "report", "--out", str(out)])
    text = out.read_text()
    assert "| ple_cpu | dtype | attn_backend | profile | samples/s |" in text
    assert "| 0 | complete (base r1) |" in text and "| 3 | complete (base r2) |" in text
    assert "## Profiles" not in text and text.rstrip().endswith("keep me") and text.startswith("# OpenJev")


def test_auto_dtype_needs_native_bf16(monkeypatch):
    """OJ-081: --dtype auto must not pick emulated bf16 on a T4 (sm75)."""
    sys.path.insert(0, str(ROOT))
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


def test_self_describing_trials(tmp_path):
    """Attrs + dashboard notes at propose, result appended at ingest, idempotent backfill.
    Temp study only."""
    sys.path.insert(0, str(ROOT / "tools"))
    import optuna_loop as ol
    from optuna_dashboard import get_note
    db, tj = tmp_path / "s.db", tmp_path / "t.json"
    ol.main(["--db", str(db), "propose", "--n", "2", "--out", str(tj),
             "--enqueue-json", '{"bs": 2, "_idea": "sdpa_bool", "_hypothesis": "faster path", '
                               '"_reference": "https://example.org/x"}'])
    proposed = json.loads(tj.read_text())
    assert all("_idea" not in t["params"] for t in proposed) and len(proposed) == 3
    study = optuna.load_study(study_name=ol.STUDY_NAME, storage=f"sqlite:///{db}")
    assert study.metric_names == [ol.METRIC_NAME]
    assert study.user_attrs["kernel"] == ol.KERNEL and study.user_attrs["baseline_s_per_sample"] == 4.5
    assert "search_space_round_2" in study.user_attrs and "Objective" in get_note(study)
    by = {t.number: t for t in study.trials}
    assert by[0].user_attrs["source"] == "baseline" and by[1].user_attrs["source"] == "enqueued"
    assert by[2].user_attrs["source"] == "tpe" and by[2].user_attrs["idea"] == "TPE exploration"
    u = by[1].user_attrs
    assert u["idea"] == "sdpa_bool" and u["hypothesis"] == "faster path" and u["round"] == 2
    assert u["reference"] == "https://example.org/x" and u["git_commit"]
    h = ol.handle(study, by[1])
    note = get_note(h)
    assert "sdpa_bool" in note and "faster path" in note and "bs=2" in note and "### Result" not in note

    res = [{"trial_number": t["trial_number"], "params": t["params"], "status": "ok",
            "samples_per_sec": 0.25, "loss_start": 2.0, "loss_guard": 1.5, "loss_end": 1.4,
            "peak_mem_gb": 7.5,
            "profile": {"top_cuda": [[f"op{i}", 1.0 + i, 0.1, 3] for i in range(8)]}}
           for t in proposed]
    (tmp_path / "r.json").write_text(json.dumps({"meta": {}, "results": res}))
    ol.main(["--db", str(db), "ingest", str(tmp_path / "r.json")])
    study = optuna.load_study(study_name=ol.STUDY_NAME, storage=f"sqlite:///{db}")
    note = get_note(ol.handle(study, {t.number: t for t in study.trials}[1]))
    assert "### Result" in note and "samples/s: 0.250" in note and "1.12x" in note
    assert "7.50 GB" in note and "`op4`" in note and "`op5`" not in note

    # backfill: idempotent, keeps states and params
    before = [(t.number, t.state, t.params) for t in study.trials]
    ol.main(["--db", str(db), "annotate", "--backfill"])
    s1 = optuna.load_study(study_name=ol.STUDY_NAME, storage=f"sqlite:///{db}")
    snap = ([(t.number, t.user_attrs) for t in s1.trials], get_note(s1),
            [get_note(ol.handle(s1, t)) for t in s1.trials])
    ol.main(["--db", str(db), "annotate", "--backfill"])
    s2 = optuna.load_study(study_name=ol.STUDY_NAME, storage=f"sqlite:///{db}")
    assert snap == ([(t.number, t.user_attrs) for t in s2.trials], get_note(s2),
                    [get_note(ol.handle(s2, t)) for t in s2.trials])
    assert before == [(t.number, t.state, t.params) for t in s2.trials]
    assert s2.trials[0].user_attrs["round"] == 1 and "OLD Prism mask" in get_note(ol.handle(s2, s2.trials[0]))
    assert s2.trials[2].user_attrs["source"] == "tpe"

    ol.main(["--db", str(db), "annotate", "--trial", "2", "--idea", "manual", "--reference", "r"])
    s3 = optuna.load_study(study_name=ol.STUDY_NAME, storage=f"sqlite:///{db}")
    assert s3.trials[2].user_attrs["idea"] == "manual" and "manual" in get_note(ol.handle(s3, s3.trials[2]))
    out = tmp_path / "E.md"
    ol.main(["--db", str(db), "report", "--out", str(out)])
    assert "| # | status | idea | source |" in out.read_text()
