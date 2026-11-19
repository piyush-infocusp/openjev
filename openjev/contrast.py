"""RLCD contrastive twins (Yang et al. 2023, arXiv:2307.12950, adapted).

A deterministic, label-preserving transform of a *real* labelled Record gives a
"twin" record whose surface form differs (criteria text reordered, negated, or
pruned) but whose correct answer is known exactly from the original — no new
label is invented, only a known re-mapping of the one we already have. Because
the mapping is exact, we also know the target *distribution* on the twin: it is
the original model distribution pushed through the same permutation/negation/
renormalization applied to the criteria.

The consistency loss compares the model's own distribution on the twin against
that mapped distribution (detached, so it is a target, not a second prediction
to fit jointly). A model that is reading the *state* should give a twin the
mapped answer; a model that is pattern-matching the *surface form* of the
question (option order, which criterion is spelled "true", etc.) will not.
Minimizing this loss punishes exactly that shortcut.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch

from .data import Record
from .schema import Choice, Noul, Score


@dataclass
class Twin:
    record: Record
    question: str
    kind: str
    mapping: dict  # {"perm": list[int] | None, "mode": "perm"|"drop"|"reverse"|"swap"}


def map_probs(mapping: dict, p: torch.Tensor) -> torch.Tensor:
    """Map an original probability vector to the twin's target distribution."""
    mode = mapping["mode"]
    perm = mapping["perm"]
    if mode == "swap":
        return p.flip(0)
    if mode == "reverse":
        return p.flip(0)
    if mode == "perm":
        return p[torch.tensor(perm, dtype=torch.long, device=p.device)]
    if mode == "drop":
        keep = torch.tensor(perm, dtype=torch.long, device=p.device)
        q = p[keep]
        return q / q.sum()
    raise ValueError(f"unknown mapping mode {mode!r}")


def _twin_record(rec: Record, q_index: int, new_question, new_label) -> Record:
    r = copy.deepcopy(rec)
    r.schema.questions[q_index] = new_question
    r.answers[q_index] = new_label
    return r


def swap_noul(rec: Record, rng: random.Random) -> list[Twin]:
    """Swap a noul's true/false criteria text; the label and target flip."""
    out = []
    for i, q in enumerate(rec.schema.questions):
        if not isinstance(q, Noul):
            continue
        c = q.criteria or {}
        if not c.get("true") or not c.get("false"):
            continue  # swap is meaningless without both descriptions
        new_q = Noul(q.name, q.instructions, {"true": c["false"], "false": c["true"]})
        y = rec.answers[i]
        new_y = None if y is None else (not y)
        new_rec = _twin_record(rec, i, new_q, new_y)
        out.append(Twin(new_rec, q.name, "swap", {"perm": None, "mode": "swap"}))
    return out


def reverse_score(rec: Record, rng: random.Random) -> list[Twin]:
    """Reverse a score's level order; label k -> n-1-k, target reversed."""
    out = []
    for i, q in enumerate(rec.schema.questions):
        if not isinstance(q, Score):
            continue
        n = len(q.criteria)
        new_q = Score(q.name, list(reversed(q.criteria)), q.instructions)
        y = rec.answers[i]
        if y is None:
            new_y = None
        elif isinstance(y, str):
            k = q.levels.index(y)
            new_y = new_q.levels[n - 1 - k]
        else:
            new_y = n - 1 - int(y)
        new_rec = _twin_record(rec, i, new_q, new_y)
        out.append(Twin(new_rec, q.name, "reverse", {"perm": None, "mode": "reverse"}))
    return out


def shuffle_options(rec: Record, rng: random.Random) -> list[Twin]:
    """Permute a choice's option order; label (name) unchanged, target permuted."""
    out = []
    for i, q in enumerate(rec.schema.questions):
        if not isinstance(q, Choice):
            continue
        keys = list(q.criteria)
        perm = list(range(len(keys)))
        rng.shuffle(perm)
        new_keys = [keys[j] for j in perm]
        new_criteria = {k: q.criteria[k] for k in new_keys}
        new_q = Choice(q.name, new_criteria, q.instructions)
        y = rec.answers[i]  # option name, unchanged
        new_rec = _twin_record(rec, i, new_q, y)
        # target mapping: p_twin[new position] = p_orig[perm[new position]]
        out.append(Twin(new_rec, q.name, "shuffle", {"perm": perm, "mode": "perm"}))
    return out


def drop_wrong_option(rec: Record, rng: random.Random) -> list[Twin]:
    """Drop one random wrong option from a choice with >=3 options; renormalize target."""
    out = []
    for i, q in enumerate(rec.schema.questions):
        if not isinstance(q, Choice):
            continue
        if len(q.criteria) < 3:
            continue
        y = rec.answers[i]
        if y is None:
            continue
        keys = list(q.criteria)
        correct_idx = keys.index(y)
        wrong_indices = [j for j in range(len(keys)) if j != correct_idx]
        drop_idx = rng.choice(wrong_indices)
        keep = [j for j in range(len(keys)) if j != drop_idx]
        new_criteria = {keys[j]: q.criteria[keys[j]] for j in keep}
        new_q = Choice(q.name, new_criteria, q.instructions)
        new_rec = _twin_record(rec, i, new_q, y)
        out.append(Twin(new_rec, q.name, "drop", {"perm": keep, "mode": "drop"}))
    return out


_TRANSFORMS = {
    "swap": swap_noul,
    "reverse": reverse_score,
    "shuffle": shuffle_options,
    "drop": drop_wrong_option,
}


def make_twins(rec: Record, rng: random.Random,
               kinds: tuple[str, ...] = ("swap", "reverse", "shuffle", "drop")) -> list[Twin]:
    twins = []
    for kind in kinds:
        twins.extend(_TRANSFORMS[kind](rec, rng))
    return twins


def _sym_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    kl_pq = (p * ((p + eps).log() - (q + eps).log())).sum()
    kl_qp = (q * ((q + eps).log() - (p + eps).log())).sum()
    return (kl_pq + kl_qp) / 2


def consistency_loss(p_twin_logits, p_orig_probs_mapped) -> torch.Tensor:
    """Symmetric KL between softmax(twin logits) and the (detached) mapped target.

    Each argument is either a single question's flag-logits tensor (1D) paired
    with its mapped target probabilities, or a list of such tensors (one per
    question, flag counts may differ across questions) — the result is the mean
    symmetric KL over questions.
    """
    if torch.is_tensor(p_twin_logits):
        logits_list, target_list = [p_twin_logits], [p_orig_probs_mapped]
    else:
        logits_list, target_list = list(p_twin_logits), list(p_orig_probs_mapped)
    losses = []
    for logits, target in zip(logits_list, target_list):
        p = torch.softmax(logits, dim=-1)
        q = target.detach()
        losses.append(_sym_kl(p, q))
    return torch.stack(losses).mean()
