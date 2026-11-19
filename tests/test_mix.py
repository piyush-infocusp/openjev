"""Tests for openjev.mix: source_weights and Mixture."""

from __future__ import annotations

import pytest

from openjev.data import Record, write_jsonl
from openjev.mix import Mixture, source_weights
from openjev.schema import Schema


def make_records(source: str, n: int) -> list[Record]:
    out = []
    for i in range(n):
        schema = Schema().noul("flag", "is it?", {"true": "yes", "false": "no"})
        out.append(Record(f"{source}-{i}", source, "other", f"state {i}", schema, [True]))
    return out


def write_source(tmp_path, name: str, n: int) -> str:
    path = tmp_path / f"{name}.jsonl"
    write_jsonl(str(path), make_records(name, n))
    return str(path)


def test_source_weights_uniform_at_alpha_zero():
    w = source_weights({"a": 100, "b": 25}, alpha=0.0)
    assert w["a"] == pytest.approx(0.5)
    assert w["b"] == pytest.approx(0.5)


def test_source_weights_proportional_at_alpha_one():
    w = source_weights({"a": 100, "b": 25}, alpha=1.0)
    assert w["a"] == pytest.approx(0.8)
    assert w["b"] == pytest.approx(0.2)


def test_source_weights_alpha_half_exact():
    w = source_weights({"a": 100, "b": 25}, alpha=0.5)
    assert w["a"] == pytest.approx(2 / 3)
    assert w["b"] == pytest.approx(1 / 3)


def test_source_weights_errors():
    with pytest.raises(ValueError):
        source_weights({})
    with pytest.raises(ValueError):
        source_weights({"a": 0})
    with pytest.raises(ValueError):
        source_weights({"a": -5})


def test_same_seed_same_sequence(tmp_path):
    paths = [write_source(tmp_path, "a", 10), write_source(tmp_path, "b", 10)]
    m1 = Mixture(paths, seed=42)
    m2 = Mixture(paths, seed=42)
    seq1 = [r.id for r in m1.take(50)]
    seq2 = [r.id for r in m2.take(50)]
    assert seq1 == seq2


def test_different_seed_different_sequence(tmp_path):
    paths = [write_source(tmp_path, "a", 10), write_source(tmp_path, "b", 10)]
    m1 = Mixture(paths, seed=1)
    m2 = Mixture(paths, seed=2)
    seq1 = [r.id for r in m1.take(50)]
    seq2 = [r.id for r in m2.take(50)]
    assert seq1 != seq2


def test_empirical_frequency_matches_weights(tmp_path):
    paths = [write_source(tmp_path, "a", 100), write_source(tmp_path, "b", 25)]
    m = Mixture(paths, alpha=0.5, seed=7)
    draws = m.take(4000)
    counts = {"a": 0, "b": 0}
    for r in draws:
        counts[r.source] += 1
    freq = {s: c / len(draws) for s, c in counts.items()}
    for s, w in m.weights.items():
        assert freq[s] == pytest.approx(w, abs=0.03)


def test_epoch_property_small_source(tmp_path):
    paths = [write_source(tmp_path, "a", 5), write_source(tmp_path, "b", 500)]
    m = Mixture(paths, alpha=0.0, seed=3)
    seen: list[str] = []
    for r in m:
        if r.source == "a":
            seen.append(r.id)
        if len(seen) == 5:
            break
    assert len(set(seen)) == 5


def test_explicit_weights_override(tmp_path):
    paths = [write_source(tmp_path, "a", 100), write_source(tmp_path, "b", 25)]
    m = Mixture(paths, weights={"a": 0.1, "b": 0.9})
    assert m.weights["a"] == pytest.approx(0.1)
    assert m.weights["b"] == pytest.approx(0.9)

    with pytest.raises(ValueError):
        Mixture(paths, weights={"a": 1.0})
