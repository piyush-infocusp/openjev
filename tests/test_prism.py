import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openjev import Schema, add_special_tokens
from openjev.data import Record
from openjev.prism import PrismOpenJev, prism_collate, prism_mask

OPTS = {"billing": "Payments", "bug": "Bugs", "sales": "Buy more"}


def schema(order):
    return (Schema().choice("dept", {k: OPTS[k] for k in order}, "Which team?")
            .noul("loud", "Has '!'?", {"true": "yes", "false": "no"}))


def rec(order, state="Hi, refund please!"):
    return Record("r", "src", "test", state, schema(order), [order[0], True])


@pytest.fixture(scope="module")
def tiny(tok):
    from transformers import Gemma4TextConfig, Gemma4TextModel
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=len(tok) + 64, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=32, sliding_window=8, vocab_size_per_layer_input=100,
                           hidden_size_per_layer_input=8, num_kv_shared_layers=0,
                           layer_types=["sliding_attention", "sliding_attention",
                                        "sliding_attention", "full_attention"])
    cfg._attn_implementation = "sdpa"
    m = PrismOpenJev(Gemma4TextModel(cfg)).eval()
    add_special_tokens(tok, m)
    return m


def test_mask_rules(tok, tiny):
    b = prism_collate(tok, [rec(["billing", "bug", "sales"])])
    m = prism_mask(b.segment, b.question, b.is_flag, b.position_ids)[0, 0]
    seg, q, f = b.segment[0], b.question[0], b.is_flag[0]
    opts = [s for s in seg.unique().tolist() if s > 0]
    a, c = (seg == opts[0]).nonzero()[0, 0], (seg == opts[1]).nonzero()[0, 0]
    assert not m[a, c] and not m[c, a]          # options never see each other
    shared = (seg == 0).nonzero()[:, 0]
    assert m[shared[0], shared[-1]]             # shared block is bidirectional
    assert not m[shared[0], a]                  # shared does not see options
    flag0 = f.nonzero()[0, 0]
    assert m[flag0, c]                          # a flag sees its sibling options
    other_q = (q == 1).nonzero()[0, 0]
    assert not m[flag0, other_q]                # but not options of other questions


def test_options_share_start_position(tok, tiny):
    b = prism_collate(tok, [rec(["billing", "bug", "sales"])])
    seg, pos = b.segment[0], b.position_ids[0]
    starts = {pos[(seg == s).nonzero()[0, 0]].item() for s in seg.unique().tolist()
              if s > 0 and b.question[0][(seg == s).nonzero()[0, 0]] == 0}
    assert len(starts) == 1


def test_option_order_invariance(tok, tiny):
    """Prism property: reorder options -> the same probabilities, reordered."""
    o1, o2 = ["billing", "bug", "sales"], ["sales", "billing", "bug"]
    with torch.no_grad():
        l1 = tiny.flag_logits(prism_collate(tok, [rec(o1)]))[:3]
        l2 = tiny.flag_logits(prism_collate(tok, [rec(o2)]))[:3]
    perm = [o2.index(k) for k in o1]
    assert torch.allclose(l1, l2[perm], atol=1e-4)


def test_padding_and_grad(tok, tiny):
    b = prism_collate(tok, [rec(["billing", "bug", "sales"]),
                            rec(["bug", "sales", "billing"], state="x")])
    tiny.train()
    out = tiny.flag_logits(b)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert tiny.flag_head.weight.grad is not None
    tiny.eval()
