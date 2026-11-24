"""Objective A final eval: per-source metrics, Prism checkpoints through eval.py /
eval_prism.py / calibrate.py, and the calibration maps (temperature, Platt)."""
import json
import math
import random
from pathlib import Path

import pytest
import torch

import calibrate
import eval as eval_py
import eval_prism as ep
import train_prism as tp
from openjev.data import read_jsonl
from openjev.metrics import source_metrics
from openjev.prism import prism_collate

ROOT = Path(__file__).resolve().parent.parent
FIX = sorted(str(p) for p in (ROOT / "tests" / "fixtures").glob("*.jsonl"))
MST = 32


# ---------------------------------------------------------------- metrics math

def test_source_metrics_fixed_batch():
    items = [(torch.tensor([0.9, 0.1]), 0, "A"), (torch.tensor([0.6, 0.4]), 1, "A"),
             (torch.tensor([0.2, 0.5, 0.3]), 1, "B")]
    m = source_metrics(items)
    a, b, o = m["sources"]["A"], m["sources"]["B"], m["overall"]
    assert a["n"] == 2 and a["acc"] == 0.5
    assert a["nll"] == pytest.approx((-math.log(0.9) - math.log(0.4)) / 2)
    assert a["brier"] == pytest.approx((0.02 + 0.72) / 2)
    assert a["ece"] == pytest.approx(0.5 * 0.1 + 0.5 * 0.6)          # 2 bins of 1 item
    assert a["acc_se2"] == pytest.approx(2 * math.sqrt(0.25 / 2))
    assert (a["acc_lo"], a["acc_hi"]) == (0.0, 1.0)                   # clipped
    assert b == pytest.approx({"n": 1, "acc": 1.0, "acc_se2": 0.0, "acc_lo": 1.0, "acc_hi": 1.0,
                               "nll": -math.log(0.5), "brier": 0.38, "ece": 0.5})
    assert o["n"] == 3 and o["acc"] == pytest.approx(2 / 3)
    assert o["ece"] == pytest.approx((0.5 + 0.6 + 0.1) / 3)
    assert o["nll"] == pytest.approx((-math.log(0.9) - math.log(0.4) - math.log(0.5)) / 3)


# ---------------------------------------------------------------- tiny Prism checkpoint

def _records():
    return [r for p in FIX for r in read_jsonl(p)]


@pytest.fixture(scope="module")
def prism_ckpt(tmp_path_factory):
    """A tiny Gemma 4 Prism checkpoint whose LoRA B, template delta and flag head are
    random (all three are zero / untrained at init, so a loader that drops one of them
    would still match). Returns (ckpt dir, logits before save, logits of a fresh model)."""
    from transformers import AutoTokenizer, Gemma4TextConfig, Gemma4TextModel
    # Its own tokenizer: the session `tok` gets the template tokens from other tests, and
    # then the base has room for them and the PLE resize path (the one that needs the
    # loader's replay) would not run.
    tok = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
    root = tmp_path_factory.mktemp("prism")
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=16, sliding_window=16, vocab_size_per_layer_input=len(tok),
                           hidden_size_per_layer_input=4, num_kv_shared_layers=0,
                           pad_token_id=tok.pad_token_id or 0,
                           layer_types=["sliding_attention", "full_attention"])
    base = root / "base"
    Gemma4TextModel(cfg).save_pretrained(base)
    tok.save_pretrained(base)
    args = tp.parse(["--model", str(base), "--cpu", "--max-state-tokens", str(MST), "--lora-r", "4",
                     "--lora-alpha", "8"])
    model, mtok = _seeded_build(args)
    fresh = _logits_of(model, mtok)
    _perturb(model, 1)
    before = _logits_of(model, mtok)
    ckpt = root / "ckpt"
    tp.save(model, mtok, args, ckpt)
    # an --init run (like RL from SFT best/): build pastes the SFT ckpt's ple_rows.pt
    rl_args = tp.parse(["--model", str(base), "--cpu", "--max-state-tokens", str(MST),
                        "--algo", "exact", "--init", str(ckpt)])
    rl, rtok = _seeded_build(rl_args)
    _perturb(rl, 2)
    rl_before = _logits_of(rl, rtok)
    rl_ckpt = root / "rl_ckpt"
    tp.save(rl, rtok, rl_args, rl_ckpt)
    return ckpt, before, fresh, rl_ckpt, rl_before


def _seeded_build(args):
    random.seed(args.seed)               # as train_prism.main's load()
    torch.manual_seed(args.seed)
    model, t, _ = tp.build(args, "cpu")
    return model, t


def _logits_of(model, t):
    model.eval()
    with torch.no_grad():
        return model.flag_logits(prism_collate(t, _records()[:6], MST)).float()


def _perturb(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n or n.startswith("flag_head") or n == "template_delta":
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)


def test_load_prism_round_trip(prism_ckpt):
    ckpt, before, fresh = prism_ckpt[:3]
    model, tok, args = ep.load_prism(ckpt, "cpu")
    assert args.max_state_tokens == MST and args.lora_r == 4
    after = _logits_of(model, tok)
    assert torch.allclose(after, before, atol=1e-5)
    assert not torch.allclose(after, fresh, atol=1e-3)      # the check is not vacuous
    again = _logits_of(ep.load_prism(ckpt, "cpu")[0], tok)
    assert torch.equal(again, after)                        # deterministic
    # collect() sorts by length (longest first) and returns the logits in record order.
    # Other row orders / batch cuts / padding only add fp32 noise (~1e-4 here, it does not
    # grow with the padding); a reordering would show as O(1) differences.
    def flat(items):
        return torch.tensor([v for it in items for v in it["logits"]])
    for bs, cap in ((6, 0), (4, 64)):
        got = flat(ep.collect(model, tok, _records()[:6], MST, "cpu", bs=bs, max_tokens=cap))
        assert torch.allclose(got, before, atol=2e-3), (bs, cap, (got - before).abs().max())


def test_load_prism_round_trip_init_trained(prism_ckpt):
    """A checkpoint trained with --init (RL): its ple_rows.pt holds the SFT rows the run
    pasted, so its logits match too."""
    rl_ckpt, rl_before = prism_ckpt[3:]
    model, tok, _ = ep.load_prism(rl_ckpt, "cpu")
    assert torch.allclose(_logits_of(model, tok), rl_before, atol=1e-5)


def test_load_prism_uses_saved_ple_rows(prism_ckpt, tmp_path):
    """With ple_rows.pt (train_prism.save) the rows are pasted, not redrawn."""
    import shutil
    ckpt, before = prism_ckpt[:2]
    model, tok, _ = ep.load_prism(ckpt, "cpu")
    ids = tok.convert_tokens_to_ids(__import__("openjev.schema", fromlist=["x"]).SPECIAL_TOKENS)
    w = ep._ple_table(model).weight
    cp = tmp_path / "ckpt"
    shutil.copytree(ckpt, cp)
    cfg = json.loads((cp / "prism_config.json").read_text())
    cfg["args"]["init"] = str(ckpt)       # would redraw (wrong rows) without ple_rows.pt
    (cp / "prism_config.json").write_text(json.dumps(cfg))
    torch.save({str(i): w[i].detach().clone() for i in ids}, cp / ep.PLE_ROWS)
    m2, t2, _ = ep.load_prism(cp, "cpu")
    assert torch.allclose(_logits_of(m2, t2), before, atol=1e-5)


def test_evaluate_reports_per_source_and_keeps_old_keys(prism_ckpt):
    model, tok, args = ep.load_prism(prism_ckpt[0], "cpu")
    args.bs = 4
    m = tp.evaluate(model, tok, _records(), args, "cpu")
    assert {"acc", "nll", "ece", "per_source", "seq_p95", "seq_max", "brier", "sources"} <= set(m)
    assert set(m["per_source"]) == set(m["sources"]) and len(m["sources"]) == 3
    for s, r in m["sources"].items():
        assert isinstance(m["per_source"][s], float) and m["per_source"][s] == r["acc"]
        assert {"n", "acc", "acc_se2", "acc_lo", "acc_hi", "nll", "ece", "brier"} <= set(r)
    n = sum(r["n"] for r in m["sources"].values())
    assert m["n"] == n
    assert m["acc"] == pytest.approx(sum(r["acc"] * r["n"] for r in m["sources"].values()) / n, abs=1e-3)
    # same numbers as the final-eval path on the same records
    e = ep.metrics(ep.collect(model, tok, _records(), MST, "cpu", bs=4))
    assert e["overall"]["nll"] == pytest.approx(m["nll"], abs=2e-4)
    assert e["overall"]["ece"] == pytest.approx(m["ece"], abs=2e-4)


def test_eval_py_loads_prism_checkpoint(prism_ckpt, tmp_path):
    out = tmp_path / "eval"
    eval_py.main(["--ckpt", str(prism_ckpt[0]), "--test", *FIX, "--out", str(out), "--cpu",
                  "--bs", "4"])
    ev = json.loads((out / "eval.json").read_text())
    n_q = sum(1 for r in _records() for y in r.answers if y is not None)
    assert ev["raw"]["all"]["n"] == n_q
    assert {k for k in ev["raw"] if k.startswith("source:")} == {f"source:{r.source}" for r in _records()}


def test_eval_prism_main_per_source_subset(prism_ckpt, tmp_path, monkeypatch):
    out = tmp_path / "fe"
    res = ep.main(["--ckpt", str(prism_ckpt[0]), "--data", *FIX, "--out", str(out), "--cpu",
                   "--max-per-source", "5", "--bs", "4"])
    assert set(res["records_per_source"].values()) == {5} and res["records"] == 15
    saved = json.loads((out / "metrics.json").read_text())
    assert set(saved["sources"]) == set(res["records_per_source"])
    assert saved["overall"]["n"] == len(ep.read_items(out / "items.jsonl"))
    md = (out / "metrics.md").read_text()
    assert "±2 SE" in md and "**overall**" in md
    # the seeded subset is stable and labels come straight from the data
    again = ep.select(_records(), 5, 0)
    gold = {(r.id): r for r in again}
    for it in ep.read_items(out / "items.jsonl"):
        r = gold[it["id"]]
        q = next(q for q in r.schema.questions if q.name == it["name"])
        assert it["target"] == tp.target_index(q, r.answers[r.schema.questions.index(q)])
    # a paired comparison of a dump with itself: zero difference
    c = ep.main(["--compare", str(out / "items.jsonl"), str(out / "items.jsonl"), "--out", str(tmp_path / "c")])
    assert c["n_paired"] == saved["overall"]["n"] and c["overall"]["d_acc"] == 0.0
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit):
        ep.main(["--ckpt", str(prism_ckpt[0]), "--data", *FIX, "--out", str(out), "--cpu"])


def test_calibrate_prism_checkpoint_and_items(prism_ckpt, tmp_path):
    ckpt = prism_ckpt[0]
    cfg_before = (ckpt / "prism_config.json").read_text()
    res = calibrate.main(["--ckpt", str(ckpt), "--val", *FIX, "--test", *FIX, "--cpu",
                          "--out", str(tmp_path / "cal"), "--bs", "4"])
    assert (ckpt / "prism_config.json").read_text() == cfg_before       # ckpt never written
    assert set(res["test"]) == {"raw", "temperature", "platt"}
    for k in res["test"]:
        assert len(res["test"][k]["sources"]) == 3
        for s in res["test"][k]["sources"]:                               # monotone maps
            assert res["test"][k]["sources"][s]["acc"] == res["test"]["raw"]["sources"][s]["acc"]
    assert (tmp_path / "cal" / "calibration.md").read_text().count("**overall**") == 2
    # model-free mode from eval_prism dumps gives the same fit
    ep.main(["--ckpt", str(ckpt), "--data", *FIX, "--out", str(tmp_path / "v"), "--cpu",
             "--max-per-source", "0"])
    res2 = calibrate.main(["--val-items", str(tmp_path / "v" / "items.jsonl"),
                           "--test-items", str(tmp_path / "v" / "items.jsonl")])
    assert res2["fit"]["temperature"] == pytest.approx(res["fit"]["temperature"], rel=1e-3)


# ---------------------------------------------------------------- calibration maps

def _toy(n=400, k=4, seed=0):
    """Overconfident toy: the top option is right ~55% of the time at confidence ~1."""
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(n):
        y = int(torch.randint(0, k, (1,), generator=g))
        top = y if torch.rand(1, generator=g).item() < 0.55 else int(torch.randint(0, k, (1,), generator=g))
        z = torch.randn(k, generator=g) * 0.3
        z[top] += 8.0
        items.append({"id": str(i), "source": "s0" if i % 2 else "s1", "type": "choice",
                      "name": "q", "target": y, "logits": z.tolist()})
    return items


def test_temperature_and_platt_fit_lower_nll_on_miscalibrated_toy():
    val, test = _toy(seed=0), _toy(seed=1)
    targets = [it["target"] for it in test]
    fit = calibrate.fit_calibration(val, log=lambda *_: None)
    assert fit["temperature"]["choice"] > 1.5
    a, b = fit["platt"]["choice"]
    assert a > 0
    for split in (val, test):
        r = calibrate.report(split, fit)
        raw, t, p = (r[k]["overall"] for k in ("raw", "temperature", "platt"))
        assert t["nll"] < raw["nll"] and p["nll"] < raw["nll"]
        assert t["ece"] < raw["ece"] and p["ece"] < raw["ece"]
        assert raw["acc"] == t["acc"] == p["acc"]                  # monotone maps: argmax kept
        assert set(r["raw"]["sources"]) == {"s0", "s1"}
    assert [it["target"] for it in test] == targets               # labels never touched
    # types with too few items keep the raw softmax
    few = calibrate.fit_calibration(val[:5], log=lambda *_: None)
    assert few["temperature"]["choice"] == 1.0 and few["platt"]["choice"] is None
