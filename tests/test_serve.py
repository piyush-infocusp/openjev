import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openjev import checkpoint
from serve import create_app

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("serve_ckpt")
    train_cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--model", "trl-internal-testing/tiny-Qwen3ForCausalLM",
        "--train", str(FIXTURES / "*.jsonl"),
        "--val", str(FIXTURES / "*.jsonl"),
        "--steps", "2", "--bs", "4", "--eval-every", "2",
        "--cpu", "--save-dir", str(save_dir), "--seed", "0",
    ]
    result = subprocess.run(train_cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return save_dir / "final"


@pytest.fixture(scope="module")
def client(ckpt_dir):
    model, tokenizer, config = checkpoint.load(ckpt_dir)
    app = create_app(model=model, tokenizer=tokenizer, config=config)
    return TestClient(app)


VALID_BODY = {
    "state": "The customer is yelling about a late refund.",
    "questions": {
        "is_shouting": {"type": "noul", "instructions": "is the customer shouting?",
                         "criteria": {"true": "yes", "false": "no"}},
        "dept": {"type": "choice", "instructions": "route to which department?",
                 "criteria": {"billing": "money", "tech": "bugs", "sales": "deals"}},
        "mood": {"type": "score", "instructions": "how angry?",
                  "criteria": ["Calm", "Annoyed", "Furious"]},
    },
}


def test_valid_request_200_and_shape(client):
    resp = client.post("/v1/systemone", json=VALID_BODY)
    assert resp.status_code == 200
    body = resp.json()

    assert set(body.keys()) == {"model", "answers", "usage"}
    assert set(body["usage"].keys()) == {"input_tokens", "output_tokens"}
    assert isinstance(body["usage"]["input_tokens"], int)
    assert isinstance(body["usage"]["output_tokens"], int)

    # answer order follows request order
    assert list(body["answers"].keys()) == list(VALID_BODY["questions"].keys())

    noul = body["answers"]["is_shouting"]
    assert set(noul.keys()) == {"type", "noul"}
    assert noul["type"] == "noul"

    choice = body["answers"]["dept"]
    assert set(choice.keys()) == {"type", "choice", "probabilities", "confidence"}
    assert choice["type"] == "choice"
    assert abs(sum(choice["probabilities"].values()) - 1.0) < 1e-4

    score = body["answers"]["mood"]
    assert set(score.keys()) == {"type", "score", "legend", "probabilities", "confidence"}
    assert score["type"] == "score"
    assert abs(sum(score["probabilities"].values()) - 1.0) < 1e-4


def test_answer_order_matches_shuffled_request_order(client):
    body = {
        "state": VALID_BODY["state"],
        "questions": {
            "mood": VALID_BODY["questions"]["mood"],
            "is_shouting": VALID_BODY["questions"]["is_shouting"],
            "dept": VALID_BODY["questions"]["dept"],
        },
    }
    resp = client.post("/v1/systemone", json=body)
    assert resp.status_code == 200
    assert list(resp.json()["answers"].keys()) == ["mood", "is_shouting", "dept"]


def test_missing_state_is_4xx(client):
    body = {"questions": VALID_BODY["questions"]}
    resp = client.post("/v1/systemone", json=body)
    assert 400 <= resp.status_code < 500


def test_empty_questions_is_4xx(client):
    body = {"state": "hi", "questions": {}}
    resp = client.post("/v1/systemone", json=body)
    assert 400 <= resp.status_code < 500


def test_unknown_type_is_4xx(client):
    body = {"state": "hi", "questions": {"q": {"type": "bogus"}}}
    resp = client.post("/v1/systemone", json=body)
    assert 400 <= resp.status_code < 500


def test_choice_too_few_options_is_4xx(client):
    body = {"state": "hi", "questions": {"q": {"type": "choice", "criteria": {"a": "x"}}}}
    resp = client.post("/v1/systemone", json=body)
    assert 400 <= resp.status_code < 500


def test_choice_too_many_options_is_4xx(client):
    criteria = {f"opt{i}": "x" for i in range(256)}
    body = {"state": "hi", "questions": {"q": {"type": "choice", "criteria": criteria}}}
    resp = client.post("/v1/systemone", json=body)
    assert 400 <= resp.status_code < 500


def test_score_too_few_levels_is_4xx(client):
    body = {"state": "hi", "questions": {"q": {"type": "score", "criteria": ["only_one"]}}}
    resp = client.post("/v1/systemone", json=body)
    assert 400 <= resp.status_code < 500
