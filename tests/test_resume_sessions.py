"""Resumable sessions (openjev/resume.py, ported from the fork) and the frozen SFT KL
reference (train_prism.FrozenRef), on CPU with a tiny random Gemma 4:

- 6 steps in one run == 3 steps + resume + 3 steps (same per-step losses, same final
  trainables), SFT and RL exact (with --init), in one process and under 2 gloo ranks;
- --max-minutes stops cleanly (exit 0, resume checkpoint, no DONE.json), also under 2 ranks;
- the KL reference is the --init policy: KL 0 at step 0, unchanged by RL updates and resumes;
- the portability rules (optimizer dtype cast, GradScaler skip, config check, find_latest)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openjev import resume as rs  # noqa: E402

DATA = str(ROOT / "tests" / "fixtures" / "*.jsonl")
TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2"]


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory, tok):
    from transformers import Gemma4TextConfig, Gemma4TextModel
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=16, sliding_window=16, vocab_size_per_layer_input=len(tok),
                           hidden_size_per_layer_input=4, num_kv_shared_layers=0,
                           pad_token_id=tok.pad_token_id or 0,
                           layer_types=["sliding_attention", "full_attention"])
    base = tmp_path_factory.mktemp("tiny") / "base"
    Gemma4TextModel(cfg).save_pretrained(base)
    tok.save_pretrained(base)
    return base


def run(base, save, *extra, ranks=1, steps=6):
    cmd = [*(TORCHRUN if ranks == 2 else [sys.executable]), str(ROOT / "train_prism.py"),
           "--model", str(base), "--cpu", "--train", DATA, "--val", DATA, "--bs", "2",
           "--grad-accum", "2", "--steps", str(steps), "--eval-every", "4", "--eval-max", "6",
           "--max-state-tokens", "48", "--group-chunk", "3", "--save-every", "2",
           "--save-dir", str(save), *extra]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600,
                       env={**__import__("os").environ, "OMP_NUM_THREADS": "1"})
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
    return r.stdout


@pytest.fixture(scope="module")
def sft_best(tiny_base, tmp_path_factory):
    save = tmp_path_factory.mktemp("sft")
    run(tiny_base, save, "--algo", "sft", "--lr", "1e-2", "--head-lr", "1e-2", steps=4)
    assert (save / "best" / "heads.pt").exists() and (save / "DONE.json").exists()
    return save / "best"


def final_state(save):
    return torch.load(rs.find_latest(Path(save) / "resume") / "state.pt", map_location="cpu",
                      weights_only=False)


def steps_log(save):
    rows = [json.loads(x) for x in (Path(save) / "train_log.jsonl").read_text().splitlines()]
    return {r["step"]: r for r in rows}       # a resumed step overwrites the killed one


def assert_same_run(a, b, key):
    la, lb = steps_log(a), steps_log(b)
    assert sorted(la) == sorted(lb) == list(range(6))
    for s in range(6):
        assert abs(la[s][key] - lb[s][key]) < 1e-5, (s, la[s], lb[s])
    sa, sb = final_state(a), final_state(b)
    assert sa["step"] == sb["step"] == 6
    for n, v in sa["trainable"].items():
        assert torch.allclose(v, sb["trainable"][n], atol=1e-6), n


def algo_args(algo, sft_best):
    if algo == "sft":
        return ["--algo", "sft"]
    return ["--algo", "exact", "--init", str(sft_best), "--lr", "1e-2", "--head-lr", "1e-2"]


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("algo", ["sft", "exact"])
def test_six_steps_equal_three_plus_three(tmp_path, tiny_base, sft_best, algo, ranks):
    extra = algo_args(algo, sft_best)
    run(tiny_base, tmp_path / "a", *extra, ranks=ranks)
    out1 = run(tiny_base, tmp_path / "b", *extra, "--session-steps", "3", ranks=ranks)
    assert '"stopped": "session_steps"' in out1 and not (tmp_path / "b" / "DONE.json").exists()
    assert (tmp_path / "b" / "last" / "heads.pt").exists()
    out2 = run(tiny_base, tmp_path / "b", *extra, "--resume", "auto", ranks=ranks)
    assert '"resumed"' in out2 and (tmp_path / "b" / "DONE.json").exists()
    assert_same_run(tmp_path / "a", tmp_path / "b", "ce" if algo == "sft" else "reward")
    if algo == "exact":
        la, lb = steps_log(tmp_path / "a"), steps_log(tmp_path / "b")
        assert la[0]["kl"] < 1e-7, la[0]               # reference == --init policy at step 0
        assert max(la[s]["kl"] for s in range(1, 6)) > 1e-6, la
        for s in range(6):                             # the resumed run kept the same reference
            assert abs(la[s]["kl"] - lb[s]["kl"]) < 1e-6, (s, la[s], lb[s])
    # A third start finds the stage complete and exits before loading the model.
    assert "stage already complete" in run(tiny_base, tmp_path / "b", *extra, "--resume", "auto",
                                           ranks=ranks)


@pytest.mark.parametrize("ranks", [1, 2])
def test_max_minutes_stops_and_resumes(tmp_path, tiny_base, ranks):
    """--max-minutes counts from main() start (model load included): a tiny budget stops
    after the first step with a resume checkpoint and exit 0 (rank 0's clock decides for
    every rank, so 2 ranks do not hang; subprocess timeout); --resume auto finishes."""
    out1 = run(tiny_base, tmp_path / "m", "--max-minutes", "0.0001", ranks=ranks)
    assert '"stopped": "max_minutes"' in out1 and not (tmp_path / "m" / "DONE.json").exists()
    assert rs.find_latest(tmp_path / "m" / "resume") is not None
    out2 = run(tiny_base, tmp_path / "m", "--resume", "auto", ranks=ranks)
    assert '"resumed"' in out2 and (tmp_path / "m" / "DONE.json").exists()
    assert final_state(tmp_path / "m")["step"] == 6


def test_resume_refuses_another_config(tmp_path, tiny_base):
    run(tiny_base, tmp_path / "c", "--session-steps", "2")
    cmd = [sys.executable, str(ROOT / "train_prism.py"), "--model", str(tiny_base), "--cpu",
           "--train", DATA, "--val", DATA, "--bs", "2", "--grad-accum", "2", "--steps", "6",
           "--max-state-tokens", "48", "--group-chunk", "3", "--packing", "on",
           "--save-dir", str(tmp_path / "c"), "--resume", "auto"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "config differs" in r.stderr


# ---------------------------------------------------------------- frozen SFT reference

def test_frozen_ref_is_the_init_policy(tiny_base, sft_best, monkeypatch):
    """In process, under an ENABLED autocast (bf16 on CPU), so the autocast cast cache is
    live: KL is 0 before any update; after RL updates the reference logits are unchanged
    while the policy logits moved; the snapshot holds the --init flag head."""
    import train_prism as tp
    from openjev.data import read_jsonl
    from openjev.prism import prism_collate
    monkeypatch.setattr(tp, "autocast", lambda device, model: torch.autocast("cpu", dtype=torch.bfloat16))
    args = tp.parse(["--model", str(tiny_base), "--cpu", "--algo", "exact", "--init", str(sft_best),
                     "--max-state-tokens", "48", "--lr", "5e-2", "--head-lr", "5e-2"])
    torch.manual_seed(0)
    model, tok, _ = tp.build(args, "cpu")
    model.train()
    recs = [r for p in sorted((ROOT / "tests" / "fixtures").glob("*.jsonl")) for r in read_jsonl(p)][:4]
    b = prism_collate(tok, recs, args.max_state_tokens)
    ref = tp.FrozenRef(model)
    with tp.autocast("cpu", model):
        r0 = ref.logits(b, "cpu")
        l0 = model.flag_logits(b)
        _, st = tp.exact_pg_loss(l0, r0, b, args)
    assert st["kl"] == pytest.approx(0.0, abs=1e-7) and torch.equal(r0, l0.detach())
    opt = tp.make_optimizer(model, args)
    for _ in range(3):
        r = ref.logits(b, "cpu")
        with tp.autocast("cpu", model):
            loss, st = tp.exact_pg_loss(model.flag_logits(b), r, b, args)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    with tp.autocast("cpu", model):          # live casts cached first, then the reference
        l1 = model.flag_logits(b)
        r1 = ref.logits(b, "cpu")
        _, st = tp.exact_pg_loss(l1, r1, b, args)
    assert torch.equal(r1, r0), "the reference moved"
    assert not torch.allclose(l1.float(), l0.float(), atol=1e-3), "the policy did not train"
    assert st["kl"] > 1e-6
    heads = torch.load(Path(sft_best) / "heads.pt", map_location="cpu")
    snap = dict(zip([n for n, p in model.named_parameters() if p.requires_grad], ref.snap))
    for k, v in heads.items():
        assert torch.equal(snap[f"flag_head.{k}"], v.float()), k
        assert not torch.equal(dict(model.named_parameters())[f"flag_head.{k}"].detach(), v.float()), k


# ---------------------------------------------------------------- portability units

def _toy():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 1))
    opt = torch.optim.AdamW([{"params": list(m[0].parameters()), "lr": 1e-3},
                             {"params": list(m[1].parameters()), "lr": 1e-2}])
    m(torch.randn(2, 3)).sum().backward()
    opt.step()
    return m, opt


def test_optimizer_state_dtype_cast_and_shape_skip():
    m, opt = _toy()
    sd = opt.state_dict()
    names = rs.opt_param_names(m, opt)
    for st in sd["state"].values():
        st["exp_avg"] = st["exp_avg"].half()
    sd["state"][3]["exp_avg_sq"] = torch.zeros(7)               # wrong shape -> dropped
    m2, opt2 = _toy()
    osd, notes = rs.adapt_optimizer_state(sd, names, m2, opt2)
    opt2.load_state_dict(osd)
    assert all(s["exp_avg"].dtype == torch.float32 for s in opt2.state.values())
    assert any("shape mismatch" in n for n in notes) and any("cast" in n for n in notes)
    assert len(opt2.state) == 3


def test_scaler_state_only_into_enabled_scaler(tmp_path):
    """A CUDA fp16 checkpoint carries GradScaler state; CPU skips it with a note."""
    from openjev.mix import Mixture
    m, opt = _toy()
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    data = sorted(str(p) for p in (ROOT / "tests" / "fixtures").glob("*.jsonl"))
    mix = Mixture(data, seed=0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    cfg = {"a": 1}
    p = rs.save(tmp_path, step=5, stage="sft", model=m, opt=opt, sched=sched, scaler=scaler,
                mixture=mix, best=1.5, cfg=cfg, device="cpu", dtype=torch.float32)
    st = torch.load(p / "state.pt", weights_only=False)
    st["scaler"] = {"scale": 1024.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                    "growth_interval": 2000, "_growth_tracker": 3}
    st["rng"]["kind"], st["rng"]["device"] = "cuda", [torch.zeros(8, dtype=torch.uint8)]
    torch.save(st, p / "state.pt")
    m2, opt2 = _toy()
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 1.0)
    mix2 = Mixture(data, seed=1)
    r = rs.load(p, model=m2, opt=opt2, sched=sched2, scaler=scaler, mixture=mix2, device="cpu", cfg=cfg)
    assert r["step"] == 5 and r["best"] == 1.5
    assert [x.id for x in mix2.take(5)] == [x.id for x in mix.take(5)]
    assert any("GradScaler state skipped" in n for n in r["notes"])
    assert any("device RNG from cuda" in n for n in r["notes"])
    with pytest.raises(SystemExit):
        rs.load(p, model=m2, opt=opt2, sched=sched2, scaler=scaler, mixture=mix, device="cpu",
                cfg={"a": 2})


def test_find_latest_prefers_stage_then_step(tmp_path):
    for stage, step in (("sft", 900), ("rl", 10), ("sft", 1000)):
        d = tmp_path / stage / "resume" / f"step_{step:07d}"
        d.mkdir(parents=True)
        (d / "state.pt").write_bytes(b"x")
        (d / "meta.json").write_text(json.dumps({"step": step, "stage": stage}))
    (tmp_path / "rl" / "resume" / "step_0000020.tmp").mkdir()      # half-written: ignored
    assert rs.find_latest(tmp_path / "*").name == "step_0000010"
    assert rs.find_latest(tmp_path / "sft").name == "step_0001000"
    assert rs.find_latest(tmp_path / "none") is None


def test_flags():
    import train_prism as tp
    a = tp.parse([])
    assert a.resume == "off" and a.max_minutes == 0 and a.save_every == 100 and a.packing == "off"
    assert a.length_grouping == "on"
    with pytest.raises(SystemExit):
        tp.parse(["--save-every", "-1"])
