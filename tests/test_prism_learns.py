"""Learnability (OJ-080): a tiny Gemma 4 with the Prism mask must learn a toy
choice task where the state names the right option. It uses train_prism's own
make_optimizer / sft_loss / question_views / optimizer_step, so a failure here
means the training path (mask, gather, loss, optimizer) is broken.

The options are single tokens, so they all have the same length. Under the
old mask (a flag saw every option token of its question) every flag got the
same logit and this test stayed at chance (loss exactly ln 4)."""
import random
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train_prism as tp  # noqa: E402
from openjev import Schema, add_special_tokens  # noqa: E402
from openjev.data import Record  # noqa: E402
from openjev.prism import PrismOpenJev, prism_collate  # noqa: E402
from openjev.schema import SPECIAL_TOKENS  # noqa: E402

WORDS = ["red", "blue", "green", "pink", "black", "white", "gold", "gray"]
K, STEPS, BS = 4, 60, 32


def toy(rng):
    opts = rng.sample(WORDS, K)
    y = rng.choice(opts)
    return Record("r", "toy", "train", f"the answer is {y}",
                  Schema().choice("c", {o: "" for o in opts}, "pick"), [y])


def test_tiny_prism_learns_toy_choice(tok):
    from transformers import Gemma4TextConfig, Gemma4TextModel
    assert all(len(tok(w, add_special_tokens=False)["input_ids"]) == 1 for w in WORDS)
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=len(tok) + 64, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=16, sliding_window=64, vocab_size_per_layer_input=100,
                           hidden_size_per_layer_input=8, num_kv_shared_layers=0,
                           layer_types=["sliding_attention", "sliding_attention", "full_attention"],
                           initializer_range=0.1)  # 3 layers + init 0.1: learned on all seeds tried (0-8)
    cfg._attn_implementation = "sdpa"
    model = PrismOpenJev(Gemma4TextModel(cfg))
    add_special_tokens(tok, model)
    model.set_template_ids(tok.convert_tokens_to_ids(SPECIAL_TOKENS))
    opt = tp.make_optimizer(model, types.SimpleNamespace(lr=1e-3, head_lr=1e-3))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    rng = random.Random(0)
    val = [toy(random.Random(1000 + i)) for i in range(64)]

    model.train()
    for _ in range(STEPS):
        b = prism_collate(tok, [toy(rng) for _ in range(BS)])
        tp.sft_loss(model.flag_logits(b), b).backward()
        tp.optimizer_step(opt, scaler)

    model.eval()
    with torch.no_grad():
        b = prism_collate(tok, val)
        views = list(tp.question_views(model.flag_logits(b), b))
    acc = sum(int(l.argmax() == t) for l, t, _ in views) / len(views)
    assert acc >= 0.9, f"toy accuracy {acc:.2f} after {STEPS} steps (chance {1 / K})"
