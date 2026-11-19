import torch

from openjev.rlcd import brier, ece, fit_temperature, reinforce, rlcd_loss, soft_ece


def _perfect_onehot(n=8, k=4):
    targets = torch.randint(0, k, (n,))
    probs = torch.zeros(n, k)
    probs[torch.arange(n), targets] = 1.0
    return probs, targets


# ---------- rlcd_loss ----------

def test_rlcd_loss_finite_and_backward_no_ref():
    logits = torch.randn(8, 4, requires_grad=True)
    targets = torch.randint(0, 4, (8,))
    out = rlcd_loss(logits, targets)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert logits.grad is not None


def test_rlcd_loss_finite_and_backward_with_ref():
    logits = torch.randn(8, 4, requires_grad=True)
    ref_logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    out = rlcd_loss(logits, targets, ref_logits=ref_logits)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert logits.grad is not None


# ---------- ece / brier ----------

def test_ece_perfectly_calibrated_correct_is_zero():
    probs, targets = _perfect_onehot()
    assert ece(probs, targets) == 0.0


def test_brier_perfect_onehot_is_zero():
    probs, targets = _perfect_onehot()
    assert brier(probs, targets) == 0.0


def test_soft_ece_has_gradient():
    logits = torch.randn(16, 4, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    targets = torch.randint(0, 4, (16,))
    val = soft_ece(probs, targets)
    val.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# ---------- fit_temperature ----------

def test_fit_temperature_overconfident_wrong_gt_one():
    # Model is always confidently wrong: predicts class 1 strongly, true class is 0.
    n = 20
    logits = torch.tensor([[-10.0, 10.0]] * n)
    targets = torch.zeros(n, dtype=torch.long)
    t = fit_temperature(logits, targets)
    assert t > 1.0


# ---------- reinforce ----------

def test_reinforce_runs_with_simple_reward():
    logits = torch.randn(8, 4, requires_grad=True)

    def reward_fn(actions):
        # reward 1.0 for choosing option 0, else 0.0
        return (actions == 0).float()

    out = reinforce(logits, reward_fn)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert logits.grad is not None
