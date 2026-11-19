import torch

from openjev import Schema, answers, encode, question_loss
from openjev.model import answer

SCHEMA = (Schema()
          .choice("dept", {"billing": "money", "tech": "bugs", "sales": "deals"})
          .score("mood", ["Calm", "Annoyed", "Angry"])
          .noul("is_shouting", criteria={"true": "yes", "false": "no"}))

BATCH = [
    ("Hi, refund please!", ["billing", 0, True]),
    ("Crash on start.", ["tech", 2, False]),
]


def collate(tok, batch):
    enc = [encode(tok, state, SCHEMA) for state, _ in batch]
    n = max(len(e.input_ids) for e in enc)
    pad = tok.pad_token_id or 0
    ids = torch.tensor([e.input_ids + [pad] * (n - len(e.input_ids)) for e in enc])
    mask = torch.tensor([[1] * len(e.input_ids) + [0] * (n - len(e.input_ids)) for e in enc])
    return ids, mask, [e.readouts for e in enc], [y for _, y in batch]


def test_forward_pass_shapes(tok, model):
    ids, mask, ro, labels = collate(tok, BATCH)
    preds = model(ids, mask, ro)
    assert len(preds) == 2
    assert len(preds[0]) == 3
    assert preds[0][0]["logits"].shape == (3,)  # choice: 3 options
    assert preds[0][1]["logits"].shape == (3,)  # score: 3 levels
    assert preds[0][2]["logits"].shape == (2,)  # noul: 2 flags


def test_question_loss_ce_finite_and_backward(tok, model):
    ids, mask, ro, labels = collate(tok, BATCH)
    preds = model(ids, mask, ro)
    total = 0.0
    for p_ex, y_ex in zip(preds, labels):
        for p, q, y in zip(p_ex, SCHEMA.questions, y_ex):
            l = question_loss(p, q, y, loss="ce")
            assert l is not None
            assert torch.isfinite(l)
            total = total + l
    total.backward()
    assert model.flag_head.weight.grad is not None


def test_question_loss_brier_finite_and_backward(tok, model):
    model.zero_grad()
    ids, mask, ro, labels = collate(tok, BATCH)
    preds = model(ids, mask, ro)
    total = 0.0
    for p_ex, y_ex in zip(preds, labels):
        for p, q, y in zip(p_ex, SCHEMA.questions, y_ex):
            l = question_loss(p, q, y, loss="brier")
            assert torch.isfinite(l)
            total = total + l
    total.backward()
    assert model.flag_head.weight.grad is not None


def test_question_loss_focal_finite_and_backward(tok, model):
    model.zero_grad()
    ids, mask, ro, labels = collate(tok, BATCH)
    preds = model(ids, mask, ro)
    total = 0.0
    for p_ex, y_ex in zip(preds, labels):
        for p, q, y in zip(p_ex, SCHEMA.questions, y_ex):
            l = question_loss(p, q, y, loss="focal")
            assert torch.isfinite(l)
            total = total + l
    total.backward()
    assert model.flag_head.weight.grad is not None


def test_answers_output_keys(tok, model):
    ids, mask, ro, labels = collate(tok, BATCH)
    with torch.no_grad():
        preds = model(ids, mask, ro)
    a = answers(preds[0], SCHEMA)

    assert set(a["dept"].keys()) == {"type", "choice", "probabilities", "confidence"}
    assert set(a["mood"].keys()) == {"type", "score", "legend", "probabilities", "confidence"}
    assert set(a["is_shouting"].keys()) == {"type", "noul"}


def test_answers_probabilities_and_confidence(tok, model):
    ids, mask, ro, labels = collate(tok, BATCH)
    with torch.no_grad():
        preds = model(ids, mask, ro)
    a = answers(preds[0], SCHEMA)

    dept_probs = a["dept"]["probabilities"]
    assert abs(sum(dept_probs.values()) - 1.0) < 1e-5
    assert 0.0 <= a["dept"]["confidence"] <= 1.0
    assert a["dept"]["choice"] in dept_probs

    mood_probs = a["mood"]["probabilities"]
    assert abs(sum(mood_probs.values()) - 1.0) < 1e-5
    assert 0.0 <= a["mood"]["confidence"] <= 1.0
    assert 0 <= a["mood"]["score"] <= 2  # n_levels - 1

    assert 0.0 <= a["is_shouting"]["noul"] <= 1.0
