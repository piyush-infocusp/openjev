"""OpenJev model: a causal encoder (a decoder LLM with its causal mask kept and no
text generation) plus one small head.

The flag head gives one true/false logit at each <|flag|> token. A softmax over
the flags of a question gives its probabilities. Like Jev, confidence is not
predicted: it is computed from those probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class OpenJev(nn.Module):
    def __init__(self, backbone_name: str, dtype=torch.float32):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name, dtype=dtype)
        h = self.backbone.config.hidden_size
        self.flag_head = nn.Linear(h, 1)  # one true/false logit per <|flag|>

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def resize_token_embeddings(self, n):
        return self.backbone.resize_token_embeddings(n)

    def forward(self, input_ids, attention_mask, readouts):
        """readouts: list (batch) of lists (questions) of dicts from schema.encode.

        Returns a list (batch) of lists (questions) of dicts with the flag logits.
        All questions come from one forward pass: every flag's hidden state is
        gathered in one indexing op and passed through the flag head once, then
        split back into per-question slices.
        """
        hs = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hs = hs.float()

        rows, cols, sizes = [], [], []
        for i, questions in enumerate(readouts):
            for q in questions:
                rows.extend([i] * len(q["flags"]))
                cols.extend(q["flags"])
                sizes.append(len(q["flags"]))

        if rows:
            idx = torch.tensor([rows, cols], dtype=torch.long, device=hs.device)
            flat_logits = self.flag_head(hs[idx[0], idx[1]]).squeeze(-1)
        else:
            flat_logits = hs.new_zeros(0)

        out = []
        pos = 0
        for questions in readouts:
            res = []
            for q in questions:
                k = len(q["flags"])
                res.append({"type": q["type"], "name": q["name"],
                            "logits": flat_logits[pos:pos + k]})
                pos += k
            out.append(res)
        return out

    def flag_logits(self, batch) -> torch.Tensor:
        """(F,) flag logits for a collate.Batch, computed with one gather using
        batch.flag_index directly.
        """
        hs = self.backbone(input_ids=batch.input_ids,
                            attention_mask=batch.attention_mask).last_hidden_state
        hs = hs.float()
        idx = batch.flag_index
        return self.flag_head(hs[idx[:, 0], idx[:, 1]]).squeeze(-1)


def target_index(question, y) -> int | None:
    """Turn a label into the index of the flag that must be true.

    noul:   true / false            -> 0 (true) or 1 (false)
    choice: the option name          -> its position in criteria
    score:  the level index or label -> its position in criteria
    """
    if y is None:
        return None
    if question.type == "noul":
        if not isinstance(y, bool):
            raise ValueError(f"'{question.name}': noul label must be true or false")
        return 0 if y else 1
    if question.type == "choice":
        if y not in question.options:
            raise ValueError(f"'{question.name}': {y!r} is not one of the options")
        return question.options.index(y)
    if isinstance(y, bool) or not isinstance(y, (int, str)):
        raise ValueError(f"'{question.name}': score label must be a level index or a level name")
    if isinstance(y, str):
        if y not in question.levels:
            raise ValueError(f"'{question.name}': {y!r} is not one of the levels")
        return question.levels.index(y)
    if not 0 <= y < len(question.levels):
        raise ValueError(f"'{question.name}': level {y} is outside 0..{len(question.levels) - 1}")
    return y


def _squared_emd(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Squared Earth Mover's Distance between softmax(logits) and one-hot(k).

    (Hou, Yu, Samaras 2016, "Squared Earth Mover's Distance-based Loss".)
    """
    p = torch.softmax(logits, -1)
    t = torch.zeros_like(p)
    t[k] = 1.0
    return ((torch.cumsum(p, -1) - torch.cumsum(t, -1)) ** 2).sum()


def question_loss(pred: dict, question, y, loss: str = "ce", gamma: float = 3.0,
                   score_loss: str = "ce+emd", emd_weight: float = 1.0):
    """Loss for one question. Returns None when the label is None (partial labels).

    The flags of a question compete through a softmax, so the loss is a proper
    scoring rule over that distribution: minimising it drives the probabilities
    toward the true rates, which is what calibration needs.

    loss="ce"    cross-entropy (log loss), strictly proper.
    loss="brier" Brier score, strictly proper and bounded, so a single very wrong
                 answer moves the model less.
    loss="focal" focal loss, which leaves less overconfident models (Mukhoti 2020).

    For question.type == "score", `score_loss` controls how ordering is used:
    a near miss on an ordered scale must cost less than a far miss, but CE alone
    treats all wrong levels the same. "ce+emd" (default) adds emd_weight * the
    squared Earth Mover's Distance to `loss`; "emd" uses only the EMD term;
    "ce" keeps the old behaviour (`loss` alone, ordering ignored).
    """
    k = target_index(question, y)
    if k is None:
        return None
    logits = pred["logits"]
    kt = torch.tensor([k], device=logits.device)
    if loss == "brier":
        base = ((torch.softmax(logits, -1) - _onehot(logits, k)) ** 2).sum()
    elif loss == "focal":
        logp = torch.log_softmax(logits, -1)[k]
        base = -((1 - logp.exp()) ** gamma) * logp
    else:
        base = F.cross_entropy(logits.unsqueeze(0), kt)

    if question.type != "score" or score_loss == "ce":
        return base
    if score_loss == "emd":
        return _squared_emd(logits, k)
    if score_loss == "ce+emd":
        return base + emd_weight * _squared_emd(logits, k)
    raise ValueError(f"unknown score_loss {score_loss!r}")


def _onehot(logits: torch.Tensor, k: int) -> torch.Tensor:
    t = torch.zeros_like(logits)
    t[k] = 1.0
    return t


def confidence(p: torch.Tensor) -> float:
    """Jev's confidence: how concentrated the distribution is, from 0 to 1.

    conf = (n * p_max - 1) / (n - 1). It is 0 when every option is equally likely
    and 1 when one option takes all the probability.
    """
    n = p.numel()
    return ((n * p.max() - 1) / (n - 1)).item()


def calibration_reward(probs: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> torch.Tensor:
    """Soft ECE over a GROUP of answers: the reward signal that calibration needs.

    Calibration is a property of many answers together, so no per-answer loss can
    express it. This bins answers by their top probability with soft weights, so
    it stays differentiable, and returns the weighted gap between accuracy and
    confidence in each bin. Lower is better. Add it to a proper scoring rule.

    probs:   (N, K) probabilities of N answers
    targets: (N,) index of the correct option of each answer
    """
    conf, pred = probs.max(dim=-1)
    correct = (pred == targets).float()
    centres = torch.linspace(0.5 / bins, 1 - 0.5 / bins, bins, device=probs.device)
    w = torch.exp(-((conf[:, None] - centres[None, :]) / (1.0 / bins)) ** 2)
    w = w / w.sum(dim=1, keepdim=True)
    mass = w.sum(dim=0)
    keep = mass > 0
    acc = (w * correct[:, None]).sum(0)[keep] / mass[keep]
    avg_conf = (w * conf[:, None]).sum(0)[keep] / mass[keep]
    return ((mass[keep] / mass.sum()) * (acc - avg_conf).abs()).sum()


@torch.no_grad()
def answer(pred: dict, question, temperature: float = 1.0) -> dict:
    """Build one Jev-style answer object from the model output.

    temperature: divides the logits before softmax (Guo et al. 2017 temperature
    scaling). Default 1.0 leaves the distribution unchanged. Output keys are
    unchanged either way (the Jev format never carries the temperature).
    """
    logits = pred["logits"] / temperature if temperature != 1.0 else pred["logits"]
    p = torch.softmax(logits, -1)
    if question.type == "noul":
        return {"type": "noul", "noul": p[0].item()}  # no confidence field, like Jev
    if question.type == "choice":
        probs = dict(zip(question.options, p.tolist()))
        return {"type": "choice", "choice": max(probs, key=probs.get),
                "probabilities": probs, "confidence": confidence(p)}
    levels = torch.arange(len(question.levels), device=p.device, dtype=p.dtype)
    return {"type": "score", "score": (p * levels).sum().item(),
            "legend": {str(i): lv for i, lv in enumerate(question.levels)},
            "probabilities": {str(i): v for i, v in enumerate(p.tolist())},
            "confidence": confidence(p)}


@torch.no_grad()
def answers(preds: list[dict], schema, temperatures: dict | None = None) -> dict:
    """The full `answers` object of one request.

    temperatures: optional {"noul": T, "choice": T, "score": T} map, one temperature
    per question type. A missing type defaults to 1.0 (unchanged behavior).
    """
    temperatures = temperatures or {}
    return {q.name: answer(p, q, temperatures.get(q.type, 1.0))
            for p, q in zip(preds, schema.questions)}
