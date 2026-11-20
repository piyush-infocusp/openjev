"""OJ-072: option sub-sampling in training (gold + <= K random negatives per choice
question), full options at eval, the Jev format unchanged."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openjev import Schema
from openjev.data import Record, record_from_dict, record_to_dict, subsample_options
from openjev.model import target_index

OPTS = {f"intent_{i}": f"description {i}" for i in range(40)}


def rec(label="intent_17", rid="r1"):
    s = (Schema().choice("intent", dict(OPTS), "Which intent?")
         .noul("urgent", "Urgent?", {"true": "yes", "false": "no"})
         .score("mood", ["calm", "a bit upset", "upset", "angry", "furious"], "Mood?")
         .choice("small", {"a": "", "b": "", "c": ""}, "Pick"))
    return Record(rid, "src", "test", "state text", s, [label, True, 3, "b"])


@pytest.mark.parametrize("k", [1, 5, 24])
def test_gold_plus_k_negatives(k):
    r = rec()
    before = record_to_dict(r)
    out = subsample_options(r, k, random.Random(0))
    q = out.schema.questions[0]
    assert "intent_17" in q.options and len(q.options) == k + 1
    assert all(q.criteria[o] == OPTS[o] for o in q.options)          # descriptions kept
    assert [o for o in OPTS if o in q.criteria] == q.options          # original order
    assert target_index(q, out.answers[0]) == q.options.index("intent_17")
    # noul and score never change; a choice with <= k + 1 options keeps all of them
    assert out.schema.questions[1] is r.schema.questions[1]
    assert out.schema.questions[2] is r.schema.questions[2]
    small = out.schema.questions[3].options
    assert small == ["a", "b", "c"] if k >= 2 else ("b" in small and len(small) == 2)
    assert (out.id, out.source, out.state, out.answers) == (r.id, r.source, r.state, r.answers)
    assert record_to_dict(r) == before                                # input untouched (eval sees all)
    # still a valid Jev request: round-trips through the JSONL format
    assert record_to_dict(record_from_dict(record_to_dict(out))) == record_to_dict(out)


def test_off_and_large_k_are_identity():
    r = rec()
    assert subsample_options(r, 0, random.Random(0)) is r
    assert subsample_options(r, 39, random.Random(0)) is r          # n - 1 negatives = all


def test_unlabelled_choice_keeps_k_plus_1():
    r = rec(label=None)
    out = subsample_options(r, 3, random.Random(1))
    assert len(out.schema.questions[0].options) == 4 and out.answers[0] is None


def test_negatives_are_random_but_reproducible():
    a = subsample_options(rec(), 5, random.Random("0:r1")).schema.questions[0].options
    b = subsample_options(rec(), 5, random.Random("0:r1")).schema.questions[0].options
    assert a == b                                                    # a redone step sees the same
    rng = random.Random(0)
    seen = {tuple(subsample_options(rec(), 5, rng).schema.questions[0].options) for _ in range(20)}
    assert len(seen) > 1


def test_train_batch_subsamples_and_eval_does_not(tok):
    """train_prism.train_batch sub-samples; prism_collate (eval, probe) keeps every option."""
    import copy

    import train_prism as tp
    from openjev.prism import prism_collate
    from openjev.schema import SPECIAL_TOKENS
    tok = copy.deepcopy(tok)                     # never change the session tokenizer
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    args = tp.parse(["--option-subsample-k", "4", "--max-state-tokens", "32"])
    r = rec()
    b = tp.train_batch(tok, [r], args, random.Random(0))
    assert b.groups[0][0]["end"] - b.groups[0][0]["start"] == 5
    full = prism_collate(tok, [r], 32)
    assert full.groups[0][0]["end"] - full.groups[0][0]["start"] == 40
    assert tp.parse([]).option_subsample_k == 0
    with pytest.raises(SystemExit):
        tp.parse(["--option-subsample-k", "-1"])
