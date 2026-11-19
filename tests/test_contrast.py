import copy
import random

import torch

from openjev.contrast import (
    consistency_loss,
    drop_wrong_option,
    make_twins,
    map_probs,
    reverse_score,
    shuffle_options,
    swap_noul,
)
from openjev.data import Record, record_from_dict, record_to_dict
from openjev.schema import Choice, Noul, Schema, Score


def noul_record():
    schema = Schema().noul("is_urgent", "urgent?", {"true": "time sensitive", "false": "not urgent"})
    return Record("r1", "test", "other", "state text", schema, [True])


def score_record():
    schema = Schema().score("mood", ["Calm", "Frustrated", "Very angry"], "how mad?")
    return Record("r2", "test", "other", "state text", schema, [1])


def choice_record():
    schema = Schema().choice("dept", {"billing": "b", "technical": "t", "sales": "s"}, "which team?")
    return Record("r3", "test", "other", "state text", schema, ["technical"])


def test_swap_noul_flips_label_and_target():
    rec = noul_record()
    before = copy.deepcopy(rec)
    twins = swap_noul(rec, random.Random(0))
    assert len(twins) == 1
    t = twins[0]
    assert t.record.answers[0] is False
    c = t.record.schema.questions[0].criteria
    assert c["true"] == "not urgent" and c["false"] == "time sensitive"
    p = torch.tensor([0.7, 0.3])
    mapped = map_probs(t.mapping, p)
    assert torch.allclose(mapped, torch.tensor([0.3, 0.7]))
    # original untouched
    assert rec.answers[0] == before.answers[0]
    assert rec.schema.questions[0].criteria == before.schema.questions[0].criteria
    record_from_dict(record_to_dict(t.record))


def test_swap_noul_skipped_without_both_descriptions():
    schema = Schema().noul("is_urgent", "urgent?", {"true": "time sensitive"})
    rec = Record("r", "t", "other", "s", schema, [True])
    assert swap_noul(rec, random.Random(0)) == []


def test_reverse_score_label_and_target():
    rec = score_record()
    before = copy.deepcopy(rec)
    twins = reverse_score(rec, random.Random(0))
    assert len(twins) == 1
    t = twins[0]
    assert t.record.schema.questions[0].criteria == ["Very angry", "Frustrated", "Calm"]
    assert t.record.answers[0] == 1  # n-1-1 = 1 for n=3
    p = torch.tensor([0.7, 0.2, 0.1])
    mapped = map_probs(t.mapping, p)
    assert torch.allclose(mapped, torch.tensor([0.1, 0.2, 0.7]))
    assert rec.schema.questions[0].criteria == before.schema.questions[0].criteria
    record_from_dict(record_to_dict(t.record))


def test_reverse_twice_gives_original():
    rec = score_record()
    t1 = reverse_score(rec, random.Random(0))[0]
    t2 = reverse_score(t1.record, random.Random(0))[0]
    assert t2.record.schema.questions[0].criteria == rec.schema.questions[0].criteria
    assert t2.record.answers[0] == rec.answers[0]


def test_shuffle_options_label_unchanged_target_permuted():
    rec = choice_record()
    before = copy.deepcopy(rec)
    twins = shuffle_options(rec, random.Random(42))
    t = twins[0]
    assert t.record.answers[0] == "technical"
    assert set(t.record.schema.questions[0].criteria) == {"billing", "technical", "sales"}
    p = torch.tensor([0.5, 0.3, 0.2])
    mapped = map_probs(t.mapping, p)
    perm = t.mapping["perm"]
    expected = torch.tensor([p[j] for j in perm])
    assert torch.allclose(mapped, expected)
    assert list(rec.schema.questions[0].criteria) == list(before.schema.questions[0].criteria)
    record_from_dict(record_to_dict(t.record))


def test_shuffle_deterministic_with_seed():
    rec = choice_record()
    t1 = shuffle_options(rec, random.Random(7))[0]
    t2 = shuffle_options(rec, random.Random(7))[0]
    assert list(t1.record.schema.questions[0].criteria) == list(t2.record.schema.questions[0].criteria)
    assert t1.mapping["perm"] == t2.mapping["perm"]


def test_drop_wrong_option():
    rec = choice_record()
    before = copy.deepcopy(rec)
    twins = drop_wrong_option(rec, random.Random(1))
    t = twins[0]
    assert t.record.answers[0] == "technical"
    assert len(t.record.schema.questions[0].criteria) == 2
    assert "technical" in t.record.schema.questions[0].criteria
    p = torch.tensor([0.7, 0.2, 0.1])  # billing, technical, sales
    mapped = map_probs(t.mapping, p)
    keep = t.mapping["perm"]
    raw = torch.tensor([p[j] for j in keep])
    expected = raw / raw.sum()
    assert torch.allclose(mapped, expected)
    assert len(rec.schema.questions[0].criteria) == len(before.schema.questions[0].criteria)
    record_from_dict(record_to_dict(t.record))


def test_drop_wrong_option_numeric_example():
    schema = Schema().choice("dept", {"a": "", "b": "", "c": ""}, "")
    rec = Record("r", "t", "other", "s", schema, ["a"])
    t = drop_wrong_option(rec, random.Random(2))[0]
    p = torch.tensor([0.7, 0.2, 0.1])
    mapped = map_probs(t.mapping, p)
    keep = t.mapping["perm"]
    # dropping a wrong option (b or c) keeps ratio of remaining two
    raw = p[keep]
    expected = raw / raw.sum()
    assert torch.allclose(mapped, expected)
    assert abs(mapped.sum().item() - 1.0) < 1e-6


def test_make_twins_all_kinds():
    for rec in (noul_record(), score_record(), choice_record()):
        twins = make_twins(rec, random.Random(0))
        for t in twins:
            record_from_dict(record_to_dict(t.record))


def test_consistency_loss_zero_when_matching():
    target = torch.tensor([0.2, 0.3, 0.5])
    logits = target.log()
    loss = consistency_loss(logits, target)
    assert loss.item() < 1e-5


def test_consistency_loss_positive_when_mismatched():
    target = torch.tensor([0.9, 0.05, 0.05])
    logits = torch.zeros(3)  # uniform
    loss = consistency_loss(logits, target)
    assert loss.item() > 0.05


def test_consistency_loss_has_gradient():
    target = torch.tensor([0.9, 0.05, 0.05])
    logits = torch.zeros(3, requires_grad=True)
    loss = consistency_loss(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)


def test_consistency_loss_list_of_questions():
    t1 = torch.tensor([0.5, 0.5]).log()
    t2 = torch.tensor([0.2, 0.8]).log()
    loss = consistency_loss([t1, t2], [torch.tensor([0.5, 0.5]), torch.tensor([0.2, 0.8])])
    assert loss.item() < 1e-5
