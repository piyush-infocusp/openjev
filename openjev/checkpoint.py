"""OpenJev checkpoint save/load and logit collection for calibration/eval.

The on-disk layout matches what train.py has always written:

    <dir>/
      config.json, model.safetensors, ...   (backbone.save_pretrained)
      tokenizer files                        (tokenizer.save_pretrained)
      heads.pt                               (flag_head.state_dict())
      openjev_config.json                    ({"backbone", "args", "special_token_ids",
                                                "temperatures": {"noul": T, "choice": T, "score": T}})
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .collate import collate
from .model import OpenJev, target_index
from .schema import SPECIAL_TOKENS, add_special_tokens


def save(model, tokenizer, save_dir, config: dict, temperatures: dict | None = None) -> None:
    """Write the backbone, tokenizer, heads, and openjev_config.json into save_dir.

    config: arbitrary metadata to merge into openjev_config.json (e.g. {"backbone": ...,
    "args": ..., "special_token_ids": ...}). temperatures, if given, is stored under the
    "temperatures" key as {"noul": T, "choice": T, "score": T}.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if config.get("moe"):
        # The upcycled backbone has expert submodules that the plain HF class
        # cannot load back via from_pretrained, so save the config and the raw
        # state dict ourselves instead of backbone.save_pretrained.
        from safetensors.torch import save_file

        model.backbone.config.save_pretrained(save_dir)
        state_dict = {k: v.detach().contiguous().cpu()
                      for k, v in model.backbone.state_dict().items()}
        save_file(state_dict, save_dir / "backbone.safetensors")
    else:
        model.backbone.save_pretrained(save_dir, safe_serialization=True)

    tokenizer.save_pretrained(save_dir)
    torch.save(model.flag_head.state_dict(), save_dir / "heads.pt")

    out = dict(config)
    if temperatures is not None:
        out["temperatures"] = dict(temperatures)
    with open(save_dir / "openjev_config.json", "w") as f:
        json.dump(out, f, indent=2)


def load(ckpt_dir, device: str = "cpu"):
    """Rebuild (model, tokenizer, config) from a checkpoint directory.

    The backbone and tokenizer are reloaded from the checkpoint directory itself (the
    saved tokenizer already has the OpenJev special tokens), so add_special_tokens is a
    no-op here; we still call it for consistency and then verify the ids match what was
    recorded in openjev_config.json.
    """
    from transformers import AutoTokenizer

    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "openjev_config.json") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

    moe_cfg = config.get("moe")
    if moe_cfg:
        from safetensors.torch import load_file

        from . import moe as moe_mod

        model = OpenJev(config["backbone"])
        add_special_tokens(tokenizer, model)  # no-op: tokens already present in saved tokenizer
        moe_mod.upcycle(model.backbone, n_experts=moe_cfg["n_experts"],
                         last_n_layers=moe_cfg["last_n_layers"], top_k=moe_cfg["top_k"])
        state_dict = load_file(str(ckpt_dir / "backbone.safetensors"))
        model.backbone.load_state_dict(state_dict, strict=True)
    else:
        model = OpenJev(str(ckpt_dir))
        add_special_tokens(tokenizer, model)  # no-op: tokens already present in saved tokenizer

    expected = config.get("special_token_ids")
    if expected:
        actual = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
        if actual != expected:
            raise ValueError(
                f"special token ids in {ckpt_dir} do not match the saved tokenizer: "
                f"expected {expected}, got {actual}")

    heads_state = torch.load(ckpt_dir / "heads.pt", map_location="cpu")
    model.flag_head.load_state_dict(heads_state)
    model.to(device)
    model.eval()
    return model, tokenizer, config


@torch.no_grad()
def collect_logits(model, tok, records, bs: int = 16, max_state_tokens: int = 1024) -> dict:
    """Run the model over records and group results by question type.

    Returns {qtype: [(logits, target_index, source), ...]}, one tuple per labelled
    question. logits is a 1-D tensor (detached, cpu) sized to the option count.
    """
    device = next(model.parameters()).device
    model.eval()
    out: dict = {"noul": [], "choice": [], "score": []}
    records = list(records)
    for i in range(0, len(records), bs):
        chunk = records[i:i + bs]
        batch = collate(tok, chunk, max_state_tokens=max_state_tokens).to(device)
        logits = model.flag_logits(batch)
        for ex_groups, ex_labels, schema, record in zip(batch.groups, batch.labels,
                                                          batch.schemas, chunk):
            for g, y, q in zip(ex_groups, ex_labels, schema.questions):
                if y is None:
                    continue
                k = target_index(q, y)
                out[q.type].append((logits[g["start"]:g["end"]].detach().cpu(), k, record.source))
    return out
