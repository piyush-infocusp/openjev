import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from converters.diplomacy_deception import build_records

# One tiny hand-built conversation in the real field layout (see
# DenisPeskoff/2020_acl_diplomacy data/*.jsonl): parallel lists, sender_labels
# True = truthful, False = lie, receiver_labels may be "NOANNOTATION".
CONV = {
    "messages": [
        "m0 hello",
        "m1 hi back",
        "m2 lets ally",
        "m3 sure",
        "m4 attack plan",
        "m5 ok",
        "m6 i lied here",
        "m7 no label for this one",
    ],
    "sender_labels": [True, True, True, True, True, True, False, "NOANNOTATION"],
    "receiver_labels": [True, "NOANNOTATION", True, True, True, True, False, "NOANNOTATION"],
    "speakers": ["italy", "germany", "italy", "germany", "italy", "germany", "italy", "germany"],
    "receivers": ["germany", "italy", "germany", "italy", "germany", "italy", "germany", "italy"],
    "seasons": ["Spring"] * 8,
    "years": ["1901"] * 8,
    "game_score_delta": ["0"] * 8,
}


def _index(records, msg_idx):
    for r in records:
        if r.id.endswith(f"-{msg_idx:03d}"):
            return r
    raise KeyError(msg_idx)


def test_context_window_size():
    # msg_idx=6 has 6 prior messages (0..5); with context=6 all of them appear.
    records, skipped, n_true, n_false = build_records("train", "train", [CONV], context=6)
    rec6 = _index(records, 6)
    for i in range(6):
        assert CONV["messages"][i] in rec6.state
    assert "m7" not in rec6.state  # never leaks future messages


def test_context_window_truncates_to_last_n():
    # msg_idx=6 with context=3 keeps only messages 3,4,5 (not 0,1,2).
    records, _, _, _ = build_records("train", "train", [CONV], context=3)
    rec6 = _index(records, 6)
    assert "m3 sure" in rec6.state
    assert "m4 attack plan" in rec6.state
    assert "m5 ok" in rec6.state
    assert "m0 hello" not in rec6.state
    assert "m1 hi back" not in rec6.state
    assert "m2 lets ally" not in rec6.state


def test_judged_message_is_last():
    records, _, _, _ = build_records("train", "train", [CONV], context=6)
    rec2 = _index(records, 2)
    assert rec2.state.rstrip().endswith(
        "Message to judge:\nitaly -> germany (Spring 1901): m2 lets ally"
    )
    # "Message to judge:" appears exactly once, and is the last line marker.
    assert rec2.state.count("Message to judge:") == 1


def test_labels_map_correctly():
    records, _, n_true, n_false = build_records("train", "train", [CONV], context=6)
    rec0 = _index(records, 0)
    rec6 = _index(records, 6)
    assert rec0.answers == [True]
    assert rec6.answers == [False]
    assert n_true == 6
    assert n_false == 1


def test_missing_label_skipped():
    records, skipped, n_true, n_false = build_records("train", "train", [CONV], context=6)
    ids = [r.id for r in records]
    assert not any(rid.endswith("-007") for rid in ids)  # msg 7 has "NOANNOTATION" sender label
    assert skipped == 1
    assert len(records) == 7


def test_no_label_leak_into_state():
    records, _, _, _ = build_records("train", "train", [CONV], context=6)
    for r in records:
        assert "NOANNOTATION" not in r.state
        assert "True" not in r.state
        assert "False" not in r.state
        # receiver info that isn't part of the required state format shouldn't leak either
        assert "game_score_delta" not in r.state


def test_record_id_and_source_domain():
    records, _, _, _ = build_records("validation", "val", [CONV], context=6)
    rec0 = _index(records, 0)
    assert rec0.id == "diplomacy-val-0000-000"
    assert rec0.source == "DenisPeskoff/2020_acl_diplomacy"
    assert rec0.domain == "strategy"


def test_schema_is_single_noul_question():
    records, _, _, _ = build_records("train", "train", [CONV], context=6)
    rec0 = records[0]
    assert [q.name for q in rec0.schema.questions] == ["sender_truthful"]
    assert rec0.schema.questions[0].type == "noul"
