"""Regression: RL resume from an SFT checkpoint on a backbone with no spare
embedding rows (like Gemma 4: vocab 262144 = tokenizer size). Kaggle v4 failed
with 'index 262144 is out of bounds' when loading embeddings.pt."""
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def test_sft_then_rl_resume_without_spare_rows(tmp_path, tok):
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
    data = ROOT / "tests" / "fixtures"
    train = sorted(data.glob("*.jsonl"))
    glob_arg = str(train[0]) if train else str(ROOT / "data" / "banking77.val.jsonl")
    common = [sys.executable, str(ROOT / "train_prism.py"), "--model", str(base), "--cpu",
              "--train", glob_arg, "--val", glob_arg, "--bs", "2", "--grad-accum", "1",
              "--steps", "1", "--eval-every", "1", "--eval-max", "4", "--max-state-tokens", "32"]
    subprocess.run(common + ["--algo", "sft", "--save-dir", str(tmp_path / "sft")],
                   check=True, cwd=ROOT, capture_output=True)
    r = subprocess.run(common + ["--algo", "exact", "--init", str(tmp_path / "sft" / "best"),
                                 "--save-dir", str(tmp_path / "rl")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
