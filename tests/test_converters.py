import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from converters._common import sample_indices, split_of
from converters.pokerbench import ConvertStats, convert_row, facing_bet


# ---------- split_of ----------

def test_split_of_deterministic():
    assert split_of("row-123") == split_of("row-123")


def test_split_of_roughly_5_5_90():
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(10000):
        counts[split_of(f"id-{i}")] += 1
    assert 300 <= counts["val"] <= 800
    assert 300 <= counts["test"] <= 800
    assert 8500 <= counts["train"] <= 9500


# ---------- sample_indices ----------

def test_sample_indices_none_is_all():
    assert sample_indices(50, None, seed=0) == list(range(50))


def test_sample_indices_deterministic():
    a = sample_indices(1000, 20, seed=0)
    b = sample_indices(1000, 20, seed=0)
    assert a == b


def test_sample_indices_size_and_no_duplicates():
    idxs = sample_indices(1000, 37, seed=1)
    assert len(idxs) == 37
    assert len(set(idxs)) == 37
    assert idxs == sorted(idxs)
    assert all(0 <= i < 1000 for i in idxs)


def test_sample_indices_limit_over_n():
    assert sample_indices(10, 50, seed=0) == list(range(10))


# ---------- facing_bet ----------
# Instruction strings copied in format from the first line of data/pokerbench.jsonl.

FACING_RAISE = (
    "\n\nYou are a specialist in playing 6-handed No Limit Texas Holdem. The following "
    "will be a game scenario and you need to make the optimal decision.\n\n"
    "Here is a game summary:\n\n"
    "The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips.\n"
    "The player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\n"
    "In this hand, your position is HJ, and your holding is [King of Diamond and Jack of Spade].\n"
    "Before the flop, HJ raise 2.0 chips, and BB call. Assume that all other players that is not mentioned folded.\n"
    "The flop comes King Of Spade, Seven Of Heart, and Two Of Diamond, then BB check, and HJ check.\n"
    "The turn comes Jack Of Club, then BB check, HJ bet 3 chips, BB raise 10 chips, and HJ call.\n"
    "The river comes Seven Of Club, then BB check, HJ bet 15 chips, and BB raise 40 chips.\n\n\n"
    "Now it is your turn to make a move.\n"
    "To remind you, the current pot size is 84.0 chips, and your holding is [King of Diamond and Jack of Spade].\n"
)

AFTER_CHECKS = (
    "\n\nYou are a specialist in playing 6-handed No Limit Texas Holdem. The following "
    "will be a game scenario and you need to make the optimal decision.\n\n"
    "Here is a game summary:\n\n"
    "The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips.\n"
    "The player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\n"
    "In this hand, your position is SB, and your holding is [Seven of Diamond and Six of Diamond].\n"
    "Before the flop, HJ raise 2.0 chips, and BB call. Assume that all other players that is not mentioned folded.\n"
    "The flop comes King Of Spade, Seven Of Heart, and Two Of Diamond, then HJ check, and BB check.\n"
    "The turn comes Jack Of Club, then HJ check, and BB check.\n\n\n"
    "Now it is your turn to make a move.\n"
    "To remind you, the current pot size is 5.0 chips, and your holding is [Seven of Diamond and Six of Diamond].\n"
)

PREFLOP_OPEN = (
    "\n\nYou are a specialist in playing 6-handed No Limit Texas Holdem. The following "
    "will be a game scenario and you need to make the optimal decision.\n\n"
    "Here is a game summary:\n\n"
    "The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips.\n"
    "The player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\n"
    "In this hand, your position is UTG, and your holding is [Ace of Spade and King of Heart].\n\n\n"
    "Now it is your turn to make a move.\n"
    "To remind you, the current pot size is 1.5 chips, and your holding is [Ace of Spade and King of Heart].\n"
)


def test_facing_bet_true_when_facing_raise():
    assert facing_bet(FACING_RAISE) is True


def test_facing_bet_false_after_checks():
    assert facing_bet(AFTER_CHECKS) is False


def test_facing_bet_false_preflop_open():
    assert facing_bet(PREFLOP_OPEN) is False


# ---------- pokerbench convert_row ----------

def test_menu_independent_of_label_size():
    row1 = {"instruction": FACING_RAISE, "output": "raise 60"}
    row2 = {"instruction": FACING_RAISE, "output": "raise 45"}
    stats = ConvertStats()
    rec1 = convert_row(row1, seed=0, idx=1, stats=stats)
    rec2 = convert_row(row2, seed=0, idx=1, stats=stats)
    assert rec1 is not None and rec2 is not None
    opts1 = set(rec1.schema.questions[0].criteria)
    opts2 = set(rec2.schema.questions[0].criteria)
    assert opts1 == opts2


def test_label_is_in_options():
    row = {"instruction": AFTER_CHECKS, "output": "bet 3"}
    stats = ConvertStats()
    rec = convert_row(row, seed=0, idx=2, stats=stats)
    assert rec is not None
    options = list(rec.schema.questions[0].criteria)
    assert rec.answers[0] in options


def test_contradicting_label_is_skipped():
    # facing a raise, but label says "check" -- must be skipped, not force-added.
    row = {"instruction": FACING_RAISE, "output": "check"}
    stats = ConvertStats()
    rec = convert_row(row, seed=0, idx=3, stats=stats)
    assert rec is None
    assert stats.skipped["label_contradicts_state"] == 1
