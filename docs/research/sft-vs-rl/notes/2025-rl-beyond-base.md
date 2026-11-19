# 2025-rl-beyond-base

**Citation:** Yue et al., 'Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?', 2025.
**arXiv:** https://arxiv.org/abs/2504.13837
**Local PDF:** ../papers/2025-rl-beyond-base.pdf (highlighted: ../papers/2025-rl-beyond-base.highlighted.pdf)

**Model sizes used:** Qwen2.5-7B/14B/32B and LLaMA-3.1-8B (base vs RLVR vs distilled).

**Recipe compared:** RLVR (direct RL from base) vs base pass@k vs distillation.

## Key findings

- **RLVR rarely creates new reasoning patterns.** (p.1)
  > "we surprisingly find that current training rarely elicit fundamentally new reasoning patterns."
- **RL wins at small k, base wins at large k.** (p.1)
  > "RLVR-trained models outperform their base models at smaller values of k"
- Reasoning capacity is **bounded by the base model**. (p.3)
  > "RLVR does not introduce fundamentally new reasoning capabilities and that the reasoning capacity of current RLVR models remains bounded by that of its base model."
- **Distillation (SFT) can add new patterns; RL cannot.** (p.3)
  > "distillation can transfer new reasoning patterns from a stronger teacher to the student."

## Relevance to OpenJev

OpenJev context: Gemma 4 E2B (2.3B effective), option-scoring flag head, SFT then exact-PG/GRPO with a Brier reward.

For a decision model scored at k=1 (one calibrated distribution), this is not a problem: pass@1 / calibration is exactly where RL helps. It does imply that RL cannot teach the model options/knowledge it cannot already reach, so SFT supplies knowledge, RL sharpens. No clean size trend reported.
