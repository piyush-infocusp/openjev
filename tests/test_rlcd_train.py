import json
import subprocess
import sys
from pathlib import Path

import torch

from openjev.rlcd import soft_ece, soft_ece_from_conf
from rlcd_train import rlcd_step

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent


def test_soft_ece_from_conf_matches_soft_ece_same_k():
    torch.manual_seed(0)
    logits = torch.randn(20, 5, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    targets = torch.randint(0, 5, (20,))

    ref = soft_ece(probs, targets)

    conf, pred = probs.max(dim=-1)
    correct = (pred == targets).float()
    got = soft_ece_from_conf(conf, correct)

    assert torch.allclose(ref, got, atol=1e-6)
    got.backward()
    assert logits.grad is not None


def test_soft_ece_from_conf_has_gradient():
    conf = torch.rand(16, requires_grad=True)
    correct = (torch.rand(16) > 0.5).float()
    val = soft_ece_from_conf(conf, correct)
    val.backward()
    assert conf.grad is not None
    assert torch.isfinite(conf.grad).all()


def test_kl_zero_when_policy_equals_reference(tmp_path):
    """With lam_kl only (lam_cal=lam_con=0) and policy == reference, kl == 0."""
    from openjev import OpenJev, add_special_tokens
    from openjev.data import read_jsonl
    import random
    import argparse

    model_name = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    policy = OpenJev(model_name)
    add_special_tokens(tokenizer, policy)
    ref = OpenJev(model_name)
    add_special_tokens(tokenizer, ref)
    ref.backbone.load_state_dict(policy.backbone.state_dict())
    ref.flag_head.load_state_dict(policy.flag_head.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    records = list(read_jsonl(FIXTURES / "banking77.jsonl"))[:4]

    args = argparse.Namespace(twin_frac=0.0, twin_kinds="swap,reverse,shuffle,drop",
                               max_state_tokens=1024, bins=10, lam_cal=0.0, lam_kl=1.0,
                               lam_con=0.0)
    rng = random.Random(0)
    loss, parts = rlcd_step(policy, ref, tokenizer, records, rng, args, "cpu")
    assert abs(parts["kl"]) < 1e-5


def test_rlcd_train_end_to_end(tmp_path):
    sft_dir = tmp_path / "sft"
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--model", "trl-internal-testing/tiny-Qwen3ForCausalLM",
        "--train", str(FIXTURES / "*.jsonl"),
        "--val", str(FIXTURES / "*.jsonl"),
        "--steps", "4", "--bs", "4", "--eval-every", "0",
        "--cpu", "--save-dir", str(sft_dir), "--seed", "0",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    rlcd_dir = tmp_path / "rlcd"
    cmd = [
        sys.executable, str(ROOT / "rlcd_train.py"),
        "--ckpt", str(sft_dir / "final"),
        "--train", str(FIXTURES / "*.jsonl"),
        "--val", str(FIXTURES / "*.jsonl"),
        "--steps", "4", "--bs", "4", "--eval-every", "2",
        "--cpu", "--save-dir", str(rlcd_dir), "--seed", "0",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    log_path = rlcd_dir / "log.jsonl"
    assert log_path.exists()
    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    step_lines = [l for l in lines if "loss" in l]
    assert step_lines
    for key in ("proper", "cal", "kl", "con", "acc", "ece"):
        assert key in step_lines[0]

    eval_lines = [l for l in lines if "eval" in l]
    assert eval_lines
    assert "contrast_consistency" in eval_lines[-1]

    final_dir = rlcd_dir / "final"
    assert (final_dir / "openjev_config.json").exists()
    with open(final_dir / "openjev_config.json") as f:
        config = json.load(f)
    assert config.get("temperatures_stale") is True

    from openjev import checkpoint
    model, tok, cfg = checkpoint.load(final_dir)
    assert model is not None
