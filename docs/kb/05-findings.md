# Key findings (with commits)

- **OJ-080 (6df0efe):** the old Prism mask let a flag see all sibling option tokens, so all flags of a
  question were identical → near-uniform predictions in v7/v8. Fix: a flag sees sibling FLAGS only.
  v7/v8 checkpoints are invalid.
- **Eval source shift (6df0efe):** per-source metrics were misaligned after null labels.
- **OJ-081 (452e438):** `is_bf16_supported()` counts emulation, so the T4 ran emulated bf16. Now fp16 + GradScaler.
- **Host RAM:** (a) the fp16 load doubled the model (2cf488f: bf16 load + per-tensor convert);
  (b) in-place chunked cast (_cast); (c) host convert made 10 GB anonymous → convert on GPU after
  place (9ed034a); (d) the HF mean_resizing on the resize also hit the 2.35B PLE table and peaked at
  28.9 GB → mean_resizing=False + chunked row stats (61234df).
- **Attention on T4:** the efficient SDPA kernel is unavailable for our mask (head dim 256/512);
  the math kernel is the fallback. FlexAttention REJECTED on T4 (Triton shared memory limit).
- **PLE on CPU** is what makes mb 8 fit (12.3 GB); PLE on GPU OOMs at mb 8 (13.9 GB).
- **Compact PLE:** 157,146 ids in the data → 2.82 GB (only −40%, because helpsteer3 has 155k ids).
  Next: hot/cold split (coverage curve not measured yet).
- **NOTA data:** 0 of 293,318 choice questions have a catch-all option. A7.4 adds elimination,
  control and inconclusive twins (Jev format unchanged; soft target internal).
- **Research:** SFT→RL with a short SFT is recommended at ~2B; direct RL is riskiest. tokenbender's rule
  is unverified; avataRL (RL from scratch with a dense reward) is verified. New competitors: Intern-Decision,
  decider-2b (docs/research/decision-models).
