import json
import subprocess
import sys
from pathlib import Path

import torch

from openjev import Schema, checkpoint
from openjev.model import answer, answers
from openjev.rlcd import ece, fit_temperature

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent

SCHEMA = (Schema()
          .choice("dept", {"billing": "money", "tech": "bugs", "sales": "deals"})
          .noul("is_shouting", criteria={"true": "yes", "false": "no"}))


def test_answer_temperature_one_matches_default():
    logits = torch.tensor([2.0, -1.0, 0.5])
    pred = {"type": "choice", "name": "dept", "logits": logits}
    q = SCHEMA.questions[0]
    a_default = answer(pred, q)
    a_t1 = answer(pred, q, temperature=1.0)
    assert a_default == a_t1


def test_answer_temperature_flattens_distribution():
    logits = torch.tensor([4.0, -1.0, 0.0])
    pred = {"type": "choice", "name": "dept", "logits": logits}
    q = SCHEMA.questions[0]
    a_cold = answer(pred, q, temperature=1.0)
    a_hot = answer(pred, q, temperature=5.0)
    max_cold = max(a_cold["probabilities"].values())
    max_hot = max(a_hot["probabilities"].values())
    assert max_hot < max_cold
    assert a_cold["choice"] == a_hot["choice"]  # argmax unchanged


def test_answers_output_keys_unchanged_by_temperature():
    preds = [{"type": "choice", "name": "dept", "logits": torch.tensor([1.0, 0.0, -1.0])},
             {"type": "noul", "name": "is_shouting", "logits": torch.tensor([0.3, -0.3])}]
    out_default = answers(preds, SCHEMA)
    out_temp = answers(preds, SCHEMA, temperatures={"choice": 3.0, "noul": 1.0})
    assert set(out_default.keys()) == set(out_temp.keys())
    assert set(out_default["dept"].keys()) == set(out_temp["dept"].keys())
    # noul untouched (T=1.0)
    assert out_default["is_shouting"] == out_temp["is_shouting"]
    # choice flattened
    assert max(out_temp["dept"]["probabilities"].values()) < \
        max(out_default["dept"]["probabilities"].values())


def test_fit_temperature_on_overconfident_synthetic_logits_lowers_ece():
    torch.manual_seed(0)
    n, k = 300, 4
    targets = torch.randint(0, k, (n,))
    # Predicted class is correct only ~50% of the time, but logits are large in
    # magnitude, so softmax confidence is near 1 regardless -> overconfident.
    pred_class = torch.where(torch.rand(n) < 0.5, targets, torch.randint(0, k, (n,)))
    logits = torch.nn.functional.one_hot(pred_class, k).float() * 10.0

    probs_before = torch.softmax(logits, -1)
    ece_before = ece(probs_before, targets)

    t = fit_temperature(logits, targets)
    assert t > 1.0

    probs_after = torch.softmax(logits / t, -1)
    ece_after = ece(probs_after, targets)
    assert ece_after <= ece_before


def test_checkpoint_round_trip(tmp_path):
    from transformers import AutoTokenizer

    from openjev import OpenJev, add_special_tokens

    model_name = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = OpenJev(model_name)
    add_special_tokens(tok, model)
    model.eval()

    from openjev.collate import collate
    from openjev.data import Record

    record = Record("id-0", "src", "other", "Refund please", SCHEMA, ["billing", True])
    batch = collate(tok, [record])
    with torch.no_grad():
        logits_before = model.flag_logits(batch)

    special_ids = {t: tok.convert_tokens_to_ids(t) for t in
                   __import__("openjev.schema", fromlist=["SPECIAL_TOKENS"]).SPECIAL_TOKENS}
    config = {"backbone": model_name, "special_token_ids": special_ids}
    ckpt_dir = tmp_path / "ckpt"
    checkpoint.save(model, tok, ckpt_dir, config, temperatures={"noul": 1.0, "choice": 1.5, "score": 1.0})

    loaded_model, loaded_tok, loaded_config = checkpoint.load(ckpt_dir)
    batch2 = collate(loaded_tok, [record])
    with torch.no_grad():
        logits_after = loaded_model.flag_logits(batch2)

    assert torch.allclose(logits_before, logits_after, atol=1e-5)
    assert loaded_config["temperatures"]["choice"] == 1.5


def test_calibrate_and_eval_subprocess(tmp_path):
    save_dir = tmp_path / "run"
    train_cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--model", "trl-internal-testing/tiny-Qwen3ForCausalLM",
        "--train", str(FIXTURES / "*.jsonl"),
        "--val", str(FIXTURES / "*.jsonl"),
        "--steps", "4", "--bs", "4", "--eval-every", "2",
        "--cpu", "--save-dir", str(save_dir), "--seed", "0",
    ]
    result = subprocess.run(train_cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    ckpt_dir = save_dir / "final"
    assert ckpt_dir.exists()

    calib_cmd = [
        sys.executable, str(ROOT / "calibrate.py"),
        "--ckpt", str(ckpt_dir), "--val", str(FIXTURES / "*.jsonl"),
    ]
    result = subprocess.run(calib_cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    config = json.loads((ckpt_dir / "openjev_config.json").read_text())
    assert "temperatures" in config
    for t in config["temperatures"].values():
        assert t > 0

    out_dir = tmp_path / "eval_out"
    eval_cmd = [
        sys.executable, str(ROOT / "eval.py"),
        "--ckpt", str(ckpt_dir), "--test", str(FIXTURES / "*.jsonl"),
        "--out", str(out_dir),
    ]
    result = subprocess.run(eval_cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    assert (out_dir / "eval.json").exists()
    assert (out_dir / "eval.md").exists()
    ev = json.loads((out_dir / "eval.json").read_text())
    assert "raw" in ev and "with_temperature" in ev
    assert "all" in ev["raw"]
    md = (out_dir / "eval.md").read_text()
    assert "Without stored temperature" in md
