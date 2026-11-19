# 2026-chestphenot

**Citation:** Yu, Zhu, Zhan, Song, Zeng, Chen, Lin, 'ChestPheNoT: Deployable, Auditable Label-Status-Evidence Extraction from Radiology Reports', 2026. (arXiv 2609.31629)
**arXiv:** https://arxiv.org/abs/2609.31629
**Local PDF:** ../papers/2026-chestphenot.pdf (highlighted: ../papers/2026-chestphenot.highlighted.pdf)

**Model sizes used:** Qwen2.5-Instruct 0.5B, 1.5B, 3B (p3).

**Recipe compared:** SFT vs SFT+GRPO (Dr. GRPO-style, from final SFT checkpoint) at each size, label/status classification; one run per config.

## Key findings

- **Only paper here with a per-size SFT vs SFT+GRPO table at 0.5B/1.5B/3B.** (p.3)
  > "GRPO starts from the final SFT checkpoint and trains for one epoch"
- **RL stage adds ~nothing in-distribution; small gains under shift.** (p.4)
  > "Its effect is small and concentrated out of distribution: GRPO adds essentially nothing in distribution (mimic-685 deltas between −0.4 and +0.8, Table II)"
- Authors downplay it: **an optional refinement**. (p.5)
  > "We therefore read GRPO as an optional refinement concentrated under distribution shift rather than a core contribution."
- **Statistical caveat: one run each.** (p.5)
  > "each configuration reflects a single training run"

## Relevance to OpenJev

OpenJev context: Gemma 4 E2B (2.3B effective), option-scoring flag head, SFT then exact-PG/GRPO with a Brier reward.

Size effect (Table II, p5): OOD present-F1 gains are +2.9 / +1.5 / +2.3 (N-200) and +0.8 / 0.0 / +1.5 (M-350) for 0.5B/1.5B/3B, so NO clear monotone trend and gains do not grow with size within 0.5-3B (noisy, n=1). Expect small, shift-concentrated gains from RL after a good SFT; measure OOD, not just in-distribution. No calibration metric reported.
