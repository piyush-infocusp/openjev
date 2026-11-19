import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from converters.prm800k import records_for_line

# One tiny hand-built solution in the real prm800k field layout (see
# openai/prm800k prm800k/data/*.jsonl): "question.problem" + "label.steps",
# each step a list of rated "completions" plus chosen_completion/human_completion.
ROW = {
    "question": {"problem": "What is 2 + 2?"},
    "label": {
        "steps": [
            {
                "completions": [
                    {"text": "Add 2 and 2.", "rating": 1},
                ],
                "chosen_completion": 0,
                "human_completion": None,
            },
            {
                "completions": [
                    {"text": "That gives 5.", "rating": -1},
                    {"text": "That gives 4.", "rating": 1},
                ],
                # labeler picked the wrong one but it's still rated -1
                "chosen_completion": 0,
                "human_completion": None,
            },
            {
                "completions": [
                    {"text": "So the answer is 4.", "rating": None},
                ],
                # unrated but still the chosen continuation of the solution
                "chosen_completion": 0,
                "human_completion": None,
            },
            {
                "completions": [
                    {"text": "Final check: yes, 4.", "rating": 0},
                ],
                "chosen_completion": 0,
                "human_completion": None,
            },
        ],
        "finish_reason": "solution",
    },
}


def test_correct_record_count_and_null_skipped():
    # step 0: rated 1 -> kept. step 1: chosen is rated -1 -> kept (uses the
    # chosen completion's own rating, not the other one). step 2: only
    # completion has rating None -> skipped. step 3: rated 0 -> kept.
    recs = records_for_line(ROW, phase="phase1_train", line_idx=0, max_steps=8)
    assert len(recs) == 3
    ids = [r.id for r in recs]
    assert ids == [
        "prm800k-phase1_train-000000-00",
        "prm800k-phase1_train-000000-01",
        "prm800k-phase1_train-000000-03",
    ]


def test_labels_are_rating_plus_one():
    recs = records_for_line(ROW, phase="phase1_train", line_idx=0, max_steps=8)
    by_id = {r.id: r for r in recs}
    assert by_id["prm800k-phase1_train-000000-00"].answers == [2]  # rating 1 -> 2
    assert by_id["prm800k-phase1_train-000000-01"].answers == [0]  # rating -1 -> 0
    assert by_id["prm800k-phase1_train-000000-03"].answers == [1]  # rating 0 -> 1


def test_schema_is_single_score_question_three_levels():
    recs = records_for_line(ROW, phase="phase1_train", line_idx=0, max_steps=8)
    q = recs[0].schema.questions
    assert [x.name for x in q] == ["step_quality"]
    assert q[0].type == "score"
    assert q[0].criteria == ["bad", "neutral", "good"]


def test_state_holds_previous_chosen_steps_and_never_the_rating():
    recs = records_for_line(ROW, phase="phase1_train", line_idx=0, max_steps=8)
    by_id = {r.id: r for r in recs}

    rec0 = by_id["prm800k-phase1_train-000000-00"]
    assert "Problem:\nWhat is 2 + 2?" in rec0.state
    assert "Previous steps:\n(none)" in rec0.state
    assert "Step to rate:\nAdd 2 and 2." in rec0.state

    rec1 = by_id["prm800k-phase1_train-000000-01"]
    # previous step's chosen text is present in context ...
    assert "Add 2 and 2." in rec1.state
    # ... and the step under review is the chosen (rated -1) completion, not
    # the other, unchosen, higher-rated one.
    assert "Step to rate:\nThat gives 5." in rec1.state
    assert "That gives 4." not in rec1.state

    rec3 = by_id["prm800k-phase1_train-000000-03"]
    # previous steps accumulate the chosen text of every earlier step,
    # including the unrated one (step 2), since chosen_completion still exists.
    assert "Add 2 and 2." in rec3.state
    assert "That gives 5." in rec3.state
    assert "So the answer is 4." in rec3.state

    for rec in recs:
        # ratings/labels must never leak into the state text
        assert "-1" not in rec.state
        assert "rating" not in rec.state


def test_max_steps_per_solution_caps_records():
    recs = records_for_line(ROW, phase="phase1_train", line_idx=0, max_steps=2)
    assert len(recs) == 2
    assert all(r.id.endswith("-00") or r.id.endswith("-01") for r in recs)


def test_id_source_domain():
    recs = records_for_line(ROW, phase="phase2_test", line_idx=17, max_steps=8)
    rec = recs[0]
    assert rec.id == "prm800k-phase2_test-000017-00"
    assert rec.source == "openai/prm800k"
    assert rec.domain == "reasoning"


def test_all_steps_of_one_problem_stay_in_one_split():
    from converters._common import split_of

    recs = records_for_line(ROW, phase="phase1_train", line_idx=0, max_steps=8)
    problem = ROW["question"]["problem"]
    splits = {split_of(problem, val=0.05, test=0.0) for _ in recs}
    # all records for this problem hash the same problem text -> one split
    assert len(splits) == 1
