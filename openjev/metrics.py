"""Evaluation metrics for OpenJev: accuracy, NLL, Brier, ECE, per type and per
source, plus MAE for score questions.

Each answered question contributes one item: its type, its source, the top
probability, whether the top prediction was correct, its NLL, and its Brier
score. `Accumulator` collects these and turns them into a summary table.
"""

from __future__ import annotations

from collections import defaultdict

import torch


def ece_from_conf(conf: list[float], correct: list[float], bins: int = 15) -> float:
    """Equal-mass expected calibration error from top-probabilities and
    correctness flags (no need for a common (N, K) probability tensor, since
    different questions can have different option counts).
    """
    n = len(conf)
    if n == 0:
        return 0.0
    order = sorted(range(n), key=lambda i: conf[i])
    bins = max(1, min(bins, n))
    out = 0.0
    for b in range(bins):
        idx = order[b * n // bins:(b + 1) * n // bins]
        if not idx:
            continue
        m = len(idx)
        avg_conf = sum(conf[i] for i in idx) / m
        avg_acc = sum(correct[i] for i in idx) / m
        out += (m / n) * abs(avg_acc - avg_conf)
    return out


class Accumulator:
    """Collects per-question outcomes and reports a summary table.

    add() is called once per answered question (skip questions with no
    label). summary() groups the collected items by question type, by
    source, and gives an "all" total.
    """

    def __init__(self):
        # each item: dict with type, source, conf, correct, nll, brier, mae(optional)
        self._items: list[dict] = []

    def add(self, qtype: str, source: str, probs: torch.Tensor, target: int) -> None:
        probs = probs.detach().float().reshape(-1)
        k = probs.numel()
        p_target = probs[target].clamp_min(1e-12).item()
        top = probs.argmax().item()
        conf = probs.max().item()
        correct = 1.0 if top == target else 0.0
        nll = -torch.log(torch.tensor(p_target)).item()
        onehot = torch.zeros(k)
        onehot[target] = 1.0
        brier = ((probs - onehot) ** 2).sum().item()
        item = {"type": qtype, "source": source, "conf": conf,
                "correct": correct, "nll": nll, "brier": brier}
        if qtype == "score":
            levels = torch.arange(k, dtype=probs.dtype)
            expected = (probs * levels).sum().item()
            item["mae"] = abs(expected - target)
        self._items.append(item)

    def _stats(self, items: list[dict], bins: int) -> dict:
        n = len(items)
        if n == 0:
            return {"n": 0, "acc": 0.0, "nll": 0.0, "brier": 0.0, "ece": 0.0}
        conf = [it["conf"] for it in items]
        correct = [it["correct"] for it in items]
        out = {
            "n": n,
            "acc": sum(correct) / n,
            "nll": sum(it["nll"] for it in items) / n,
            "brier": sum(it["brier"] for it in items) / n,
            "ece": ece_from_conf(conf, correct, bins=bins),
        }
        maes = [it["mae"] for it in items if "mae" in it]
        if maes:
            out["mae"] = sum(maes) / len(maes)
        return out

    def summary(self) -> dict:
        out: dict = {}
        bins_all = max(1, min(15, len(self._items) // 10 or 1))
        out["all"] = self._stats(self._items, bins_all)

        by_type: dict = defaultdict(list)
        by_source: dict = defaultdict(list)
        for it in self._items:
            by_type[it["type"]].append(it)
            by_source[it["source"]].append(it)

        for t, items in by_type.items():
            bins = max(1, min(15, len(items) // 10 or 1))
            out[t] = self._stats(items, bins)
        for s, items in by_source.items():
            bins = max(1, min(15, len(items) // 10 or 1))
            out[f"source:{s}"] = self._stats(items, bins)
        return out
