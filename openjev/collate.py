"""Batch collation for OpenJev: encode a list of Records into padded tensors plus
a flat flag index, so the model can gather every flag's hidden state in one shot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import Record
from .schema import encode


@dataclass
class Batch:
    input_ids: torch.Tensor       # (B, T) long
    attention_mask: torch.Tensor  # (B, T) long
    flag_index: torch.Tensor      # (F, 2) long: (batch row, position) of every flag, in order
    groups: list                  # per example, per question: {"type", "name", "start", "end"}
    labels: list                  # per example, per question: raw label (may be None)
    schemas: list                 # per example: the Schema

    def to(self, device) -> "Batch":
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.flag_index = self.flag_index.to(device)
        return self


def collate(tokenizer, records: list[Record], max_state_tokens: int = 2048,
            pad_to_multiple_of: int = 8) -> Batch:
    """Encode each record and right-pad to a common length (a multiple of
    `pad_to_multiple_of`). Padding is on the right, so flag positions are exact
    and the causal mask still sees only real tokens up to each position.
    """
    encoded = [encode(tokenizer, r.state, r.schema, max_state_tokens) for r in records]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    n = max(len(e.input_ids) for e in encoded)
    if pad_to_multiple_of:
        n = -(-n // pad_to_multiple_of) * pad_to_multiple_of

    input_ids = torch.full((len(encoded), n), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), n), dtype=torch.long)

    flag_rows, flag_cols = [], []
    groups = []
    start = 0
    for i, e in enumerate(encoded):
        L = len(e.input_ids)
        input_ids[i, :L] = torch.tensor(e.input_ids, dtype=torch.long)
        attention_mask[i, :L] = 1
        ex_groups = []
        for q in e.readouts:
            for pos in q["flags"]:
                flag_rows.append(i)
                flag_cols.append(pos)
            end = start + len(q["flags"])
            ex_groups.append({"type": q["type"], "name": q["name"], "start": start, "end": end})
            start = end
        groups.append(ex_groups)

    flag_index = torch.tensor(list(zip(flag_rows, flag_cols)), dtype=torch.long).reshape(-1, 2)

    labels = [list(r.answers) for r in records]
    schemas = [r.schema for r in records]

    return Batch(input_ids, attention_mask, flag_index, groups, labels, schemas)
