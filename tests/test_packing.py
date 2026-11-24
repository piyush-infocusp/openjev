"""OJ-060: hierarchical packing (source mix -> length bucket -> first-fit-decreasing rows
<= pack_len) with a doc-id block-diagonal mask and positions that restart per request.
Critical: packed flag logits == unpacked flag logits (1e-4) on a tiny Gemma 4."""
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openjev import Schema, add_special_tokens
from openjev.data import Record
from openjev.packing import length_bucket, pack_rows
from openjev.prism import PrismOpenJev, prism_collate, prism_mask


def test_pack_rows_ffd():
    lens = [5, 70, 30, 100, 3, 64, 250, 20, 40, 1]
    rows = pack_rows(lens, 128)
    assert sorted(i for r in rows for i in r) == list(range(len(lens)))   # every request once
    for r in rows:
        assert sum(lens[i] for i in r) <= 128 or len(r) == 1              # 250 > 128: own row
    assert [6] in rows
    # FFD, longest first: 250 | 100+20+5+3 | 70+40+1 | 64+30
    assert rows == [[6], [3, 7, 0, 4], [1, 8, 9], [5, 2]]
    assert pack_rows(lens, 128) == rows                                    # deterministic
    assert pack_rows([10, 10], 5) == [[0], [1]]
    assert [length_bucket(n) for n in (1, 2, 3, 4, 5, 8, 9)] == [0, 1, 2, 2, 3, 3, 4]
    with pytest.raises(ValueError):
        pack_rows([1], 0)


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


def records():
    """Different lengths; every request has question 0 / segment 1.. (ids collide inside a
    row); choice, noul and score; one request longer than the pack length."""
    rng = random.Random(0)
    words = "refund card app crash login slow fee transfer help please now".split()
    out = []
    for i, n_words in enumerate([3, 12, 1, 25, 6, 90, 2, 9]):
        state = " ".join(rng.choice(words) for _ in range(n_words))
        s = Schema().choice("dept", {"billing": "Payments", "bug": "Bugs", "sales": "Buy"}, "Which team?")
        if i % 2:
            s = s.noul("loud", "Has '!'?", {"true": "yes", "false": "no"})
        if i % 3 == 0:
            s = s.score("mood", ["calm", "upset", "angry"], "Mood?")
        labels = ["bug"] + ([True] if i % 2 else []) + ([1] if i % 3 == 0 else [])
        out.append(Record(f"r{i}", "src", "test", state, s, labels))
    return out


PACK = 96


def test_packed_logits_equal_unpacked(tok, tiny):
    import train_prism as tp
    recs = records()
    unp = prism_collate(tok, recs, 2048)
    pk = prism_collate(tok, recs, 2048, pack_len=PACK)
    lens = unp.attention_mask.sum(-1).tolist()
    assert max(lens) > PACK                                   # one request over the pack length
    assert pk.input_ids.shape[0] < len(recs)                  # it really packed
    assert (pk.doc.max(-1).values >= 1).any()                 # a row with >= 2 requests
    assert int(pk.attention_mask.sum()) == int(unp.attention_mask.sum())
    assert pk.flag_index.shape == unp.flag_index.shape and pk.groups == unp.groups
    # positions restart per request: each request's positions equal its unpacked ones
    for row in range(pk.input_ids.shape[0]):
        for d in pk.doc[row].unique().tolist():
            if d < 0:
                continue
            sel = pk.doc[row] == d
            ids = pk.input_ids[row][sel]
            src = next(i for i, L in enumerate(lens)
                       if L == int(sel.sum()) and torch.equal(unp.input_ids[i, :L], ids))
            assert torch.equal(pk.position_ids[row][sel], unp.position_ids[src, :lens[src]])
            assert pk.position_ids[row][sel].min() == 0
    with torch.no_grad():
        a, b = tiny.flag_logits(unp), tiny.flag_logits(pk)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()
    assert torch.allclose(tp.sft_loss(a, unp), tp.sft_loss(b, pk), atol=1e-5)

    # Self-check: without the doc mask, requests in one row see each other and the logits
    # change. If this ever passes, the test above no longer tests the mask.
    leak = prism_collate(tok, recs, 2048, pack_len=PACK)
    leak.doc = torch.where(leak.doc >= 0, 0, -1)
    with torch.no_grad():
        c = tiny.flag_logits(leak)
    assert (a - c).abs().max() > 1e-3


def test_packed_grads_equal_unpacked(tok, tiny):
    import train_prism as tp
    recs = records()
    grads = []
    for pack in (0, PACK):
        tiny.zero_grad(set_to_none=True)
        b = prism_collate(tok, recs, 2048, pack_len=pack)
        tp.sft_loss(tiny.flag_logits(b), b).backward()
        grads.append(tiny.flag_head.weight.grad.clone())
    tiny.zero_grad(set_to_none=True)
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_doc_mask_blocks_other_requests(tok):
    pk = prism_collate(tok, records(), 2048, pack_len=PACK)
    m = prism_mask(pk.segment, pk.question, pk.is_flag, pk.position_ids, doc=pk.doc)[:, 0]
    real = pk.doc >= 0
    cross = (pk.doc[:, :, None] != pk.doc[:, None, :]) & real[:, :, None] & real[:, None, :]
    assert cross.any() and not (m & cross).any()


def test_train_flags():
    import train_prism as tp
    a = tp.parse(["--packing", "on", "--pack-len", "1024"])
    assert a.packing == "on" and a.pack_len == 1024
    d = tp.parse([])
    assert d.packing == "off" and d.pack_len == 1024   # off by default (user, 2026-11-24)
    with pytest.raises(SystemExit):
        tp.parse(["--packing", "on", "--causal"])
    with pytest.raises(SystemExit):
        tp.parse(["--pack-len", "0"])


def test_frozen_layout(tok, tmp_path):
    """tools/freeze_packed.py: layout = pack_rows per micro-batch, sha verified, and a collate
    with the frozen rows equals the on-the-fly packed one."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import freeze_packed as fp
    from openjev import trial_data as td
    recs = records() * 2
    mb = len(recs) // 2
    lay = fp.build_layout(tok, recs, "abc", pack_len=PACK, micro_batch=mb, max_state_tokens=2048)
    assert len(lay["mbs"]) == 2 and td.verify_layout(lay, "abc", len(recs)) is lay
    on_fly = prism_collate(tok, recs[:mb], 2048, pack_len=PACK)
    frozen = prism_collate(tok, recs[:mb], 2048, pack_len=PACK, rows_of=lay["mbs"][0])
    assert torch.equal(on_fly.input_ids, frozen.input_ids) and torch.equal(on_fly.doc, frozen.doc)
    p = tmp_path / "packed_layout.json"
    td.write_json(p, lay)
    assert td.load_layout(p, "abc", len(recs))["sha"] == lay["sha"]
    assert td.load_layout(tmp_path / "none.json") is None
    lay["mbs"][0] = [[0]]
    with pytest.raises(ValueError):
        td.verify_layout(lay)
    with pytest.raises(ValueError):
        prism_collate(tok, recs[:mb], 2048, rows_of=[[0]])
