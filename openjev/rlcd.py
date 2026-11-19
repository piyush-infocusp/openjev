"""RLCD — Reinforcement Learning for Calibrated Decisions (OpenJev version).

TypeSafe names RLCD as the training method of Jev but publishes no algorithm,
loss, or reward, so this is our own build toward the same stated goal:

  * "their probabilities are optimized against outcomes to reflect uncertainty"
  * "higher confidence means higher accuracy"
  * RLHF is rejected because it causes "mode dropping: the model learns to favor
    a particular style ... while reducing the probability of other possible outputs"

Why reinforcement learning and not a plain loss: calibration is a property of a
GROUP of answers ("of the answers given 0.8, about 80% are correct"). One answer
is only right or wrong, so no per-answer loss can measure a calibration gap. A
reward can read a whole batch, so it can.

The objective over a batch has three parts:

    loss = proper_score            (per answer: keeps the answers accurate)
         + lam_cal * soft_ece      (per batch: closes the confidence-accuracy gap)
         + lam_kl  * KL(p || p_ref)(per answer: holds the model near the reference,
                                    which is what stops mode dropping)

`soft_ece` bins answers by their top probability with soft weights, so the batch
term stays differentiable and needs no sampling. `reinforce` gives the same
signal through a sampled decision and a reward, for a reward that is not
differentiable (a real outcome, a downstream utility, a cost table).

Use `rlcd_loss` after a supervised stage that already gives usable answers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_ece(probs: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> torch.Tensor:
    """Differentiable expected calibration error over a batch of answers.

    probs:   (N, K) probability of each option
    targets: (N,)   index of the correct option

    Answers join every bin with a soft weight, so the result has a gradient. It
    equals the usual equal-width ECE when the weights become hard.
    """
    conf, pred = probs.max(dim=-1)
    correct = (pred == targets).float()
    centres = torch.linspace(0.5 / bins, 1 - 0.5 / bins, bins, device=probs.device)
    w = torch.exp(-(((conf[:, None] - centres[None, :]) * bins) ** 2))
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    mass = w.sum(dim=0)
    keep = mass > 1e-6
    acc = (w * correct[:, None]).sum(0)[keep] / mass[keep]
    avg_conf = (w * conf[:, None]).sum(0)[keep] / mass[keep]
    return ((mass[keep] / mass.sum()) * (acc - avg_conf).abs()).sum()


def soft_ece_from_conf(conf: torch.Tensor, correct: torch.Tensor, bins: int = 10) -> torch.Tensor:
    """Same differentiable soft-binned ECE as `soft_ece`, but taking the top
    probability and correctness flag directly instead of a (N, K) probability
    tensor. Useful when different questions in a batch have different option
    counts K, so there is no common (N, K) tensor to build.

    conf:    (N,) top probability of each answer (keeps a gradient)
    correct: (N,) 1.0/0.0 whether the top prediction was correct
    """
    centres = torch.linspace(0.5 / bins, 1 - 0.5 / bins, bins, device=conf.device)
    w = torch.exp(-(((conf[:, None] - centres[None, :]) * bins) ** 2))
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    mass = w.sum(dim=0)
    keep = mass > 1e-6
    acc = (w * correct[:, None]).sum(0)[keep] / mass[keep]
    avg_conf = (w * conf[:, None]).sum(0)[keep] / mass[keep]
    return ((mass[keep] / mass.sum()) * (acc - avg_conf).abs()).sum()


def ece(probs: torch.Tensor, targets: torch.Tensor, bins: int = 10,
        equal_mass: bool = True) -> float:
    """Plain ECE for reporting. Equal-mass bins are less noisy than equal width."""
    conf, pred = probs.max(dim=-1)
    correct = (pred == targets).float()
    n = conf.numel()
    if equal_mass:
        order = conf.argsort()
        edges = [order[i * n // bins:(i + 1) * n // bins] for i in range(bins)]
    else:
        lo = (conf * bins).clamp(max=bins - 1).long()
        edges = [(lo == b).nonzero(as_tuple=True)[0] for b in range(bins)]
    out = 0.0
    for idx in edges:
        if idx.numel() == 0:
            continue
        out += (idx.numel() / n) * abs(correct[idx].mean().item() - conf[idx].mean().item())
    return out


def brier(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Brier score (a strictly proper scoring rule). Lower is better."""
    t = F.one_hot(targets, probs.shape[-1]).to(probs.dtype)
    return ((probs - t) ** 2).sum(dim=-1).mean().item()


def rlcd_loss(logits: torch.Tensor, targets: torch.Tensor,
              ref_logits: torch.Tensor | None = None,
              lam_cal: float = 1.0, lam_kl: float = 0.05,
              proper: str = "ce", bins: int = 10) -> dict:
    """One RLCD step over a batch of questions that share an option count.

    logits:     (N, K) flag logits of N answers
    targets:    (N,)   index of the correct option
    ref_logits: (N, K) logits of the frozen reference model, for the KL hold

    Returns a dict with `loss` and the parts, so a trainer can log them.
    """
    logp = torch.log_softmax(logits, dim=-1)
    probs = logp.exp()

    if proper == "brier":
        t = F.one_hot(targets, logits.shape[-1]).to(probs.dtype)
        score = ((probs - t) ** 2).sum(dim=-1).mean()
    else:
        score = F.nll_loss(logp, targets)

    cal = soft_ece(probs, targets, bins)

    if ref_logits is None:
        kl = torch.zeros((), device=logits.device)
    else:
        ref_logp = torch.log_softmax(ref_logits, dim=-1)
        kl = F.kl_div(ref_logp, logp, log_target=True, reduction="batchmean")

    loss = score + lam_cal * cal + lam_kl * kl
    return {"loss": loss, "proper": score.detach(), "soft_ece": cal.detach(),
            "kl": kl.detach(), "ece": ece(probs.detach(), targets),
            "acc": (probs.argmax(-1) == targets).float().mean().item()}


def reinforce(logits: torch.Tensor, reward_fn, ref_logits: torch.Tensor | None = None,
              lam_kl: float = 0.05, samples: int = 1) -> dict:
    """Policy-gradient form, for a reward that has no gradient.

    The model samples a decision for each question, `reward_fn(actions)` returns
    one reward per answer (a real outcome, a utility from a cost table, a
    downstream result), and the batch mean works as the baseline.

    reward_fn: (N,) long tensor of sampled option indices -> (N,) float rewards
    """
    logp = torch.log_softmax(logits, dim=-1)
    total, rewards_seen = 0.0, []
    for _ in range(samples):
        actions = torch.multinomial(logp.exp(), 1).squeeze(-1)
        r = reward_fn(actions).detach()
        advantage = r - r.mean()
        total = total - (advantage * logp.gather(1, actions[:, None]).squeeze(1)).mean()
        rewards_seen.append(r.mean().item())
    pg = total / samples

    if ref_logits is None:
        kl = torch.zeros((), device=logits.device)
    else:
        kl = F.kl_div(torch.log_softmax(ref_logits, -1), logp, log_target=True,
                      reduction="batchmean")

    return {"loss": pg + lam_kl * kl, "policy": pg.detach(), "kl": kl.detach(),
            "reward": sum(rewards_seen) / len(rewards_seen)}


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, steps: int = 200) -> float:
    """Temperature scaling on a held-out split (Guo et al. 2017). Run it last."""
    log_t = torch.zeros(1, requires_grad=True, device=logits.device)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits.detach() / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()
