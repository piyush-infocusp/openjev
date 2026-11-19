import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from openjev import OpenJev, add_special_tokens

MODEL_NAME = "trl-internal-testing/tiny-Qwen3ForCausalLM"


@pytest.fixture(scope="session")
def tok():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


@pytest.fixture(scope="session")
def model(tok):
    torch.manual_seed(0)
    m = OpenJev(MODEL_NAME)
    add_special_tokens(tok, m)
    return m
