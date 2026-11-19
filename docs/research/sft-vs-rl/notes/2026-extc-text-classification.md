# 2026-extc-text-classification

**Citation:** Zhou, Chen, Liang, Akoglu, 'Structured Prompt Optimization Meets Reinforcement Learning for Global and Local Interpretability over Complex Text' (EXTC), 2026. (arXiv 2605.29076)
**arXiv:** https://arxiv.org/abs/2605.29076
**Local PDF:** ../papers/2026-extc-text-classification.pdf (highlighted: ../papers/2026-extc-text-classification.highlighted.pdf)

**Model sizes used:** Compact 4B student LM (p2), LoRA SFT; teacher is a larger LLM.

**Recipe compared:** Three stages: prompt-optimized rulebook, reasoning-distillation SFT from teacher, then GRPO RL on hard cases.

## Key findings

- **Classification RL has a specific failure mode: rollout homogeneity.** (p.6)
  > "text classification with a small and often imbalanced label space exposes a distinct failure mode: rollout homogeneity."
- Consequence: **zero-advantage groups, no gradient**. (p.6)
  > "rollouts often share the same predicted label, yielding a zero-advantage group, hence no gradient signal."
- SFT first gives RL a **stable start**; class-balanced upsampling avoids a majority-biased init. (p.5)
  > "To give the subsequent RL stage a stable starting point, we apply class-balanced upsampling"
- **SFT+RL beats SFT-only (stage-by-stage gains).** (p.8)
  > "EXTC achieves the best overall performance across datasets and majority of metrics by further boosting R-SFT model through Stage III"

## Relevance to OpenJev

OpenJev context: Gemma 4 E2B (2.3B effective), option-scoring flag head, SFT then exact-PG/GRPO with a Brier reward.

Warns of a concrete issue for us: with a few options and a confident SFT model, all GRPO rollouts agree and the advantage is zero. Fixes: sample at temperature, filter zero-advantage prompts, or use exact policy gradient over the option distribution (which has no rollouts, so no homogeneity problem; this favors the exact-PG variant). Single 4B size.
