import json

import pytest

from openjev.data import Record, bucket, read_jsonl, record_from_dict, record_to_dict, write_jsonl
from openjev.schema import Schema


def base_dict(answers):
    return {
        "id": "t-1",
        "source": "unit-test",
        "domain": "test",
        "state": "hello",
        "questions": {
            "is_urgent": {"type": "noul", "instructions": "",
                          "criteria": {"true": "yes", "false": "no"}},
            "dept": {"type": "choice", "instructions": "",
                     "criteria": {"billing": "money", "tech": "bugs"}},
            "mood": {"type": "score", "instructions": "",
                     "criteria": ["Calm", "Annoyed", "Angry"]},
        },
        "answers": answers,
        "weight": 1.0,
    }


# ---------- good labels ----------

def test_good_labels_noul_true():
    r = record_from_dict(base_dict({"is_urgent": True, "dept": None, "mood": None}))
    assert r.answers[0] is True


def test_good_labels_noul_false():
    r = record_from_dict(base_dict({"is_urgent": False, "dept": None, "mood": None}))
    assert r.answers[0] is False


def test_good_labels_choice_option_name():
    r = record_from_dict(base_dict({"is_urgent": None, "dept": "billing", "mood": None}))
    assert r.answers[1] == "billing"


def test_good_labels_score_int_index():
    r = record_from_dict(base_dict({"is_urgent": None, "dept": None, "mood": 1}))
    assert r.answers[2] == 1


def test_good_labels_score_level_name():
    r = record_from_dict(base_dict({"is_urgent": None, "dept": None, "mood": "Angry"}))
    assert r.answers[2] == "Angry"


def test_none_label_allowed():
    r = record_from_dict(base_dict({"is_urgent": None, "dept": None, "mood": None}))
    assert r.answers == [None, None, None]


# ---------- bad labels ----------

def test_bad_noul_string_label():
    with pytest.raises(ValueError):
        record_from_dict(base_dict({"is_urgent": "yes", "dept": None, "mood": None}))


def test_bad_noul_int_label():
    with pytest.raises(ValueError):
        record_from_dict(base_dict({"is_urgent": 1, "dept": None, "mood": None}))


def test_bad_choice_unknown_option():
    with pytest.raises(ValueError):
        record_from_dict(base_dict({"is_urgent": None, "dept": "nope", "mood": None}))


def test_bad_score_negative():
    with pytest.raises(ValueError):
        record_from_dict(base_dict({"is_urgent": None, "dept": None, "mood": -1}))


def test_bad_score_out_of_range():
    with pytest.raises(ValueError):
        record_from_dict(base_dict({"is_urgent": None, "dept": None, "mood": 99}))


def test_bad_score_bool():
    with pytest.raises(ValueError):
        record_from_dict(base_dict({"is_urgent": None, "dept": None, "mood": True}))


def test_bad_score_unknown_name():
    with pytest.raises(ValueError):
        record_from_dict(base_dict({"is_urgent": None, "dept": None, "mood": "Furious"}))


# ---------- jsonl round trip ----------

def test_jsonl_round_trip(tmp_path):
    schema = Schema().noul("is_urgent", criteria={"true": "yes", "false": "no"})
    records = [
        Record("id-1", "src", "dom", "state one", schema, [True], 1.0),
        Record("id-2", "src", "dom", "state two", schema, [None], 0.5),
    ]
    path = tmp_path / "data.jsonl"
    n = write_jsonl(str(path), records)
    assert n == 2

    read_back = list(read_jsonl(str(path)))
    assert len(read_back) == 2
    assert read_back[0].id == "id-1"
    assert read_back[0].state == "state one"
    assert read_back[0].answers == [True]
    assert read_back[1].answers == [None]
    assert read_back[1].weight == 0.5


# ---------- bucket() ----------

def test_bucket_middle():
    assert bucket(7.5, [1, 5, 20]) == 2


def test_bucket_on_edge():
    assert bucket(5, [1, 5, 20]) == 1


def test_bucket_below_first_edge():
    assert bucket(0, [1, 5, 20]) == 0


def test_bucket_above_last_edge():
    assert bucket(100, [1, 5, 20]) == 3
