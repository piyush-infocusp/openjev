import json
import subprocess
import sys
from pathlib import Path

import torch

from openjev import checkpoint

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent


def test_upcycle_train_and_reload(tmp_path):
    save_dir = tmp_path / "run"
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--model", "trl-internal-testing/tiny-Qwen3ForCausalLM",
        "--train", str(FIXTURES / "*.jsonl"),
        "--val", str(FIXTURES / "*.jsonl"),
        "--steps", "4", "--bs", "4", "--eval-every", "0",
        "--cpu", "--save-dir", str(save_dir), "--seed", "0",
        "--upcycle", "2,1",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "moe param_counts" in result.stdout

    final_dir = save_dir / "final"
    assert (final_dir / "backbone.safetensors").exists()
    assert (final_dir / "openjev_config.json").exists()

    with open(final_dir / "openjev_config.json") as f:
        config = json.load(f)
    assert config["moe"] == {"n_experts": 2, "last_n_layers": 1, "top_k": 1}

    model, tok, cfg = checkpoint.load(final_dir)
    from openjev.collate import collate
    from openjev.data import read_jsonl

    records = list(read_jsonl(FIXTURES / "banking77.jsonl"))[:2]
    batch = collate(tok, records)
    with torch.no_grad():
        logits1 = model.flag_logits(batch)

    model2, tok2, cfg2 = checkpoint.load(final_dir)
    batch2 = collate(tok2, records)
    with torch.no_grad():
        logits2 = model2.flag_logits(batch2)

    assert torch.allclose(logits1, logits2, atol=1e-6)
