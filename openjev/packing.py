"""Hierarchical packing of encoded requests into rows (OJ-060).

Levels:
  1. source mix   - the records of one micro-batch come from the temperature-sampled
                    Mixture (or the frozen trial stream), so every row keeps the mix;
  2. length bucket - each request goes to a power-of-two length bucket; buckets are
                    filled longest first, so long requests anchor rows and short ones
                    fill the gaps;
  3. first-fit-decreasing - inside that order (longest first), each request goes to the
                    first row with room, rows hold <= pack_len tokens.

A request longer than pack_len gets a row of its own (never cut or dropped: it is one
forward pass). The rows only say where each request goes; prism_collate builds the
tensors, with a doc-id block-diagonal mask and positions that restart per request
(Krell et al. 2021, arXiv:2107.02027).
"""

from __future__ import annotations


def length_bucket(n: int) -> int:
    """Power-of-two bucket of a length: 1 -> 0, 2 -> 1, 3..4 -> 2, 5..8 -> 3, ..."""
    return max(0, n - 1).bit_length()


def pack_rows(lengths: list[int], pack_len: int) -> list[list[int]]:
    """Indices of `lengths` grouped into rows of total length <= pack_len (except a
    single request longer than pack_len, which is alone in its row). Deterministic:
    ties keep the input order."""
    if pack_len <= 0:
        raise ValueError(f"pack_len must be > 0, got {pack_len}")
    order = sorted(range(len(lengths)), key=lambda i: (-length_bucket(lengths[i]), -lengths[i], i))
    rows: list[list[int]] = []
    room: list[int] = []
    for i in order:
        n = lengths[i]
        for r, free in enumerate(room):
            if n <= free:
                rows[r].append(i)
                room[r] -= n
                break
        else:
            rows.append([i])
            room.append(max(0, pack_len - n))
    return rows
