import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openjev import Schema, collate, encode
from openjev.data import Record

SCHEMA_A = (Schema()
            .choice("department", {"billing": "Payments", "bug": "Bugs"},
                    "Which team?")
            .score("urgency", ["Calm", "Annoyed", "Angry"], "How urgent?")
            .noul("is_shouting", "Has '!'?", {"true": "yes", "false": "no"}))

SCHEMA_B = (Schema()
            .noul("is_spam", "Spam?", {"true": "yes", "false": "no"})
            .choice("lang", {"en": "English", "fr": "French", "de": "German"}, "Language?"))


def _records():
    return [
        Record("r0", "src", "test", "Hi, refund please!", SCHEMA_A,
               ["billing", 1, True]),
        Record("r1", "src", "test", "short", SCHEMA_B, [False, "en"]),
        Record("r2", "src", "test", "A somewhat longer state string here.", SCHEMA_A,
               ["bug", 2, False]),
    ]


def _old_forward(model, input_ids, attention_mask, readouts):
    hs = model.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    hs = hs.float()
    out = []
    for i, questions in enumerate(readouts):
        res = []
        for q in questions:
            res.append({"type": q["type"], "name": q["name"],
                        "logits": model.flag_head(hs[i, q["flags"]]).squeeze(-1)})
        out.append(res)
    return out


def test_forward_matches_old_loop(tok, model):
    records = _records()
    batch = collate(tok, records)
    encoded = [encode(tok, r.state, r.schema) for r in records]
    readouts = [e.readouts for e in encoded]

    model.eval()
    with torch.no_grad():
        new_out = model(batch.input_ids, batch.attention_mask, readouts)
        old_out = _old_forward(model, batch.input_ids, batch.attention_mask, readouts)

    for new_ex, old_ex in zip(new_out, old_out):
        for new_q, old_q in zip(new_ex, old_ex):
            assert new_q["type"] == old_q["type"]
            assert new_q["name"] == old_q["name"]
            torch.testing.assert_close(new_q["logits"], old_q["logits"], atol=1e-6, rtol=1e-6)


def test_flag_index_positions_hold_flag_token(tok, model):
    records = _records()
    batch = collate(tok, records)
    flag_id = tok.convert_tokens_to_ids("<|flag|>")
    rows, cols = batch.flag_index[:, 0], batch.flag_index[:, 1]
    values = batch.input_ids[rows, cols]
    assert torch.all(values == flag_id)


def test_padding_does_not_change_logits(tok, model):
    records = _records()
    single = collate(tok, [records[0]])
    padded = collate(tok, records)

    model.eval()
    with torch.no_grad():
        out_single = model.flag_logits(single)
        out_padded = model.flag_logits(padded)

    g0 = padded.groups[0]
    start = g0[0]["start"]
    end = g0[-1]["end"]
    torch.testing.assert_close(out_single, out_padded[start:end], atol=1e-5, rtol=1e-5)


def test_flag_logits_split_by_groups_equals_forward(tok, model):
    records = _records()
    batch = collate(tok, records)
    encoded = [encode(tok, r.state, r.schema) for r in records]
    readouts = [e.readouts for e in encoded]

    model.eval()
    with torch.no_grad():
        forward_out = model(batch.input_ids, batch.attention_mask, readouts)
        flat = model.flag_logits(batch)

    for ex_groups, ex_out in zip(batch.groups, forward_out):
        for g, q_out in zip(ex_groups, ex_out):
            torch.testing.assert_close(flat[g["start"]:g["end"]], q_out["logits"],
                                        atol=1e-6, rtol=1e-6)
