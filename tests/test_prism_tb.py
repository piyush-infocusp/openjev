"""TensorBoard logging: a 2-step SFT run with eval writes train/ce and val/acc."""
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def test_sft_writes_tensorboard_events(tmp_path, tok):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
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
    train = sorted((ROOT / "tests" / "fixtures").glob("*.jsonl"))
    glob_arg = str(train[0]) if train else str(ROOT / "data" / "banking77.val.jsonl")
    tb = tmp_path / "tb"
    r = subprocess.run([sys.executable, str(ROOT / "train_prism.py"), "--model", str(base), "--cpu",
                        "--train", glob_arg, "--val", glob_arg, "--bs", "2", "--grad-accum", "1",
                        "--steps", "2", "--eval-every", "2", "--eval-max", "4",
                        "--max-state-tokens", "32", "--algo", "sft",
                        "--save-dir", str(tmp_path / "sft"), "--tb-dir", str(tb)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert list(tb.rglob("events.out.tfevents*"))
    ea = EventAccumulator(str(tb))
    ea.Reload()
    tags = ea.Tags()["scalars"]
    assert "train/ce" in tags and "val/acc" in tags
