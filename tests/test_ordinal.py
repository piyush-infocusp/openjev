import pytest
import torch

from openjev.model import question_loss
from openjev.schema import Score, Noul, Choice


def _score():
    return Score("s", ["a", "b", "c", "d", "e"])


def test_emd_prefers_near_miss_over_far_miss():
    # target level 0; peak at level 1 (near) vs peak at level 4 (far)
    near_logits = torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0])
    far_logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0])
    q = _score()
    near = question_loss({"logits": near_logits}, q, 0, score_loss="emd")
    far = question_loss({"logits": far_logits}, q, 0, score_loss="emd")
    assert near.item() < far.item()


def test_ce_alone_cannot_tell_near_from_far_miss():
    # same magnitude of peak mass away from target -> CE only depends on logit at k
    near_logits = torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0])
    far_logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0])
    q = _score()
    near = question_loss({"logits": near_logits}, q, 0, score_loss="ce")
    far = question_loss({"logits": far_logits}, q, 0, score_loss="ce")
    assert near.item() == pytest.approx(far.item())


def test_zero_emd_when_p_equals_onehot():
    q = _score()
    logits = torch.tensor([-1e6, 1e6, -1e6, -1e6, -1e6])
    loss = question_loss({"logits": logits}, q, 1, score_loss="emd")
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_gradient_flows():
    q = _score()
    logits = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], requires_grad=True)
    for sl in ("ce", "emd", "ce+emd"):
        if logits.grad is not None:
            logits.grad = None
        loss = question_loss({"logits": logits}, q, 2, score_loss=sl)
        loss.backward()
        assert logits.grad is not None
        assert torch.any(logits.grad != 0)


def test_unknown_score_loss_raises():
    q = _score()
    logits = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        question_loss({"logits": logits}, q, 0, score_loss="bogus")


@pytest.mark.parametrize("sl", ["ce", "emd", "ce+emd", "bogus"])
def test_noul_choice_unaffected_by_score_loss(sl):
    noul = Noul("n", criteria={"true": "yes", "false": "no"})
    choice = Choice("c", criteria={"x": "", "y": "", "z": ""})
    logits_noul = torch.tensor([0.2, -0.3])
    logits_choice = torch.tensor([0.1, 0.5, -0.2])

    expected_noul = question_loss({"logits": logits_noul}, noul, True)
    expected_choice = question_loss({"logits": logits_choice}, choice, "y")

    got_noul = question_loss({"logits": logits_noul}, noul, True, score_loss=sl)
    got_choice = question_loss({"logits": logits_choice}, choice, "y", score_loss=sl)

    assert got_noul.item() == pytest.approx(expected_noul.item())
    assert got_choice.item() == pytest.approx(expected_choice.item())
