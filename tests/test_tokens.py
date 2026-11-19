import torch
from transformers import AutoTokenizer

from openjev import OpenJev, SPECIAL_TOKENS, add_special_tokens

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def test_template_tokens_use_spare_rows():
    tok = AutoTokenizer.from_pretrained(TINY)
    model = OpenJev(TINY)
    rows = model.get_input_embeddings().weight.shape[0]
    before = model.get_input_embeddings().weight.clone()
    n_before = len(tok)
    assert rows - n_before >= len(SPECIAL_TOKENS)  # Qwen3 has spare rows

    add_special_tokens(tok, model)

    w = model.get_input_embeddings().weight
    assert w.shape[0] == rows  # no resize
    assert w.shape[0] % 128 == 0
    ids = tok.convert_tokens_to_ids(SPECIAL_TOKENS)
    assert sorted(ids) == list(range(n_before, n_before + len(SPECIAL_TOKENS)))
    assert torch.equal(w[:n_before], before[:n_before])  # old rows untouched
    new = w[ids]
    assert len({tuple(r.tolist()) for r in new}) == len(ids)  # rows differ
    assert torch.allclose(new.mean(0), before[:n_before].mean(0), atol=0.05)


def test_add_special_tokens_is_idempotent():
    tok = AutoTokenizer.from_pretrained(TINY)
    model = OpenJev(TINY)
    add_special_tokens(tok, model)
    snap = model.get_input_embeddings().weight.clone()
    add_special_tokens(tok, model)
    assert torch.equal(model.get_input_embeddings().weight, snap)
