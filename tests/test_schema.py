import pytest

from openjev.schema import Choice, Noul, Schema, Score, FLAG, encode


# ---------- to_dict / from_dict round trip ----------

def test_noul_round_trip():
    s = Schema().noul("is_urgent", "urgent?", {"true": "yes", "false": "no"})
    d = s.to_dict()
    s2 = Schema.from_dict(d)
    assert s2.questions[0].type == "noul"
    assert s2.questions[0].name == "is_urgent"
    assert s2.questions[0].criteria == {"true": "yes", "false": "no"}
    assert s2.questions[0].instructions == "urgent?"


def test_choice_round_trip():
    s = Schema().choice("dept", {"billing": "money", "tech": "bugs"}, "which team?")
    d = s.to_dict()
    s2 = Schema.from_dict(d)
    assert s2.questions[0].type == "choice"
    assert s2.questions[0].criteria == {"billing": "money", "tech": "bugs"}


def test_score_round_trip():
    s = Schema().score("mood", ["Calm", "Angry"], "how mad?")
    d = s.to_dict()
    s2 = Schema.from_dict(d)
    assert s2.questions[0].type == "score"
    assert s2.questions[0].criteria == ["Calm", "Angry"]


# ---------- Schema validation ----------

def test_duplicate_names_raise():
    with pytest.raises(ValueError):
        Schema([Noul("x", criteria={"true": "a", "false": "b"}),
                Noul("x", criteria={"true": "a", "false": "b"})])


def test_duplicate_names_via_builder_raise():
    with pytest.raises(ValueError):
        Schema().noul("x", criteria={"true": "a", "false": "b"}).noul("x", criteria={"true": "a", "false": "b"})


def test_choice_one_option_raises():
    with pytest.raises(ValueError):
        Choice("x", {"only": "one"})


def test_choice_256_options_raises():
    crit = {f"opt{i}": "" for i in range(256)}
    with pytest.raises(ValueError):
        Choice("x", crit)


def test_choice_255_options_ok():
    crit = {f"opt{i}": "" for i in range(255)}
    c = Choice("x", crit)
    assert len(c.options) == 255


def test_score_one_level_raises():
    with pytest.raises(ValueError):
        Score("x", ["only"])


def test_from_dict_unknown_type_raises():
    with pytest.raises(ValueError):
        Schema.from_dict({"x": {"type": "bogus", "criteria": None}})


# ---------- encode() ----------

def test_encode_flag_positions_and_counts(tok):
    schema = (Schema()
              .noul("is_urgent", criteria={"true": "yes", "false": "no"})
              .choice("dept", {"billing": "money", "tech": "bugs", "sales": "deals"})
              .score("mood", ["Calm", "Annoyed", "Angry"]))
    enc = encode(tok, "hello world", schema)
    flag_id = tok.convert_tokens_to_ids(FLAG)

    assert len(enc.readouts) == 3
    noul_ro, choice_ro, score_ro = enc.readouts

    assert len(noul_ro["flags"]) == 2
    assert len(choice_ro["flags"]) == 3
    assert len(score_ro["flags"]) == 3

    for ro in enc.readouts:
        for pos in ro["flags"]:
            assert enc.input_ids[pos] == flag_id


def test_encode_requires_at_least_one_question(tok):
    with pytest.raises(ValueError):
        encode(tok, "hello", Schema())


def test_encode_state_truncation(tok):
    schema = Schema().noul("x", criteria={"true": "yes", "false": "no"})
    long_state = " ".join(["word"] * 5000)
    enc_full = encode(tok, long_state, schema, max_state_tokens=2048)
    enc_trunc = encode(tok, long_state, schema, max_state_tokens=5)

    assert len(enc_trunc.input_ids) < len(enc_full.input_ids)
