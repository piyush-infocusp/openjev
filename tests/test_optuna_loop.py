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
                                         "sdpa_backend", "ple_cpu"}
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
