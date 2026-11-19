import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from converters.helpsteer3 import fit_parts


# ---------- fake whitespace tokenizer ----------

def count_tokens(text: str) -> int:
    return len(text.split())


def encode(text: str) -> list[str]:
    return text.split()


def decode(tokens: list[str]) -> str:
    return " ".join(tokens)


SHARES = [0.30, 0.35, 0.35]


def total_tokens(parts: list[str]) -> int:
    return sum(count_tokens(p) for p in parts)


# ---------- tests ----------

def test_total_length_within_budget():
    parts = [
        "ctx " * 500,
        "r1 " * 500,
        "r2 " * 500,
    ]
    budget = 100
    out = fit_parts(parts, SHARES, budget, count_tokens, encode, decode)
    assert total_tokens(out) <= budget


def test_short_inputs_unchanged():
    parts = ["a b c", "d e", "f g h i"]
    budget = 1000
    out = fit_parts(parts, SHARES, budget, count_tokens, encode, decode)
    assert out == parts


def test_unused_budget_redistributes():
    # context is tiny, so its unused share should flow to the two responses,
    # letting them exceed their nominal 35% cap.
    parts = ["ctx", "r1 " * 200, "r2 " * 200]
    budget = 100
    out = fit_parts(parts, SHARES, budget, count_tokens, encode, decode)
    assert out[0] == "ctx"
    # each response gets more than 35 tokens (its naive cap) since context
    # only used 1 token of its 30-token share
    assert count_tokens(out[1]) > int(budget * 0.35)
    assert count_tokens(out[2]) > int(budget * 0.35)
    assert total_tokens(out) <= budget


def test_both_responses_always_present():
    parts = ["ctx " * 1000, "r1 " * 1000, "r2 " * 1000]
    budget = 90
    out = fit_parts(parts, SHARES, budget, count_tokens, encode, decode)
    assert out[1].strip() != ""
    assert out[2].strip() != ""


def test_head_and_tail_kept_with_marker():
    tokens = [f"tok{i}" for i in range(300)]
    parts = ["ctx", " ".join(tokens), "r2"]
    budget = 100  # response1 share ~35 tokens, will need cutting
    out = fit_parts(parts, SHARES, budget, count_tokens, encode, decode)
    r1_out = out[1]
    assert "[...]" in r1_out
    assert "tok0" in r1_out  # head kept
    assert "tok299" in r1_out  # tail kept
    assert "tok150" not in r1_out  # middle dropped


def test_cap_respected_even_when_all_parts_long():
    parts = ["ctx " * 500, "r1 " * 500, "r2 " * 500]
    budget = 1000
    out = fit_parts(parts, SHARES, budget, count_tokens, encode, decode)
    assert total_tokens(out) <= budget
    # all three still present since none should be starved to zero
    assert out[0].strip() != ""
    assert out[1].strip() != ""
    assert out[2].strip() != ""
