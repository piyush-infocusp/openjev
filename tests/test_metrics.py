import torch

from openjev.metrics import Accumulator, ece_from_conf


def test_perfect_predictions():
    acc = Accumulator()
    for _ in range(20):
        probs = torch.tensor([0.98, 0.01, 0.01])
        acc.add("choice", "src", probs, 0)
    s = acc.summary()["all"]
    assert s["acc"] == 1.0
    assert s["ece"] < 0.05
    assert s["brier"] < 0.02


def test_uniform_predictions_two_options_half_accuracy():
    acc = Accumulator()
    for i in range(20):
        probs = torch.tensor([0.5, 0.5])
        target = i % 2
        acc.add("noul", "src", probs, target)
    s = acc.summary()["all"]
    assert abs(s["acc"] - 0.5) < 1e-6
    assert s["ece"] < 1e-6


def test_per_source_split():
    acc = Accumulator()
    for _ in range(5):
        acc.add("choice", "a", torch.tensor([1.0, 0.0]), 0)
    for _ in range(5):
        acc.add("choice", "b", torch.tensor([0.0, 1.0]), 0)
    summary = acc.summary()
    assert summary["source:a"]["acc"] == 1.0
    assert summary["source:b"]["acc"] == 0.0
    assert summary["all"]["n"] == 10


def test_score_mae():
    acc = Accumulator()
    # 3 levels, perfect prediction of level 2
    acc.add("score", "src", torch.tensor([0.0, 0.0, 1.0]), 2)
    # predicts level 0 with certainty when target is 2 -> mae 2
    acc.add("score", "src", torch.tensor([1.0, 0.0, 0.0]), 2)
    s = acc.summary()["score"]
    assert abs(s["mae"] - 1.0) < 1e-6


def test_ece_from_conf_helper():
    conf = [0.9] * 10
    correct = [1.0] * 9 + [0.0]
    e = ece_from_conf(conf, correct, bins=1)
    assert abs(e - (0.9 - 0.9)) < 1e-6
