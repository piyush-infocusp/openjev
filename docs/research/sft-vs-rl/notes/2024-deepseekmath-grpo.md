# 2024-deepseekmath-grpo

**Citation:** Shao et al., 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models' (introduces GRPO), 2024.
**arXiv:** https://arxiv.org/abs/2402.03300
**Local PDF:** ../papers/2024-deepseekmath-grpo.pdf (highlighted: ../papers/2024-deepseekmath-grpo.highlighted.pdf)

**Model sizes used:** DeepSeekMath 7B only.

**Recipe compared:** SFT (instruct) then GRPO on a subset of the SFT questions.

## Key findings

- GRPO is critic-free PPO variant; applied **after SFT**. (p.2)
  > "we introduce the Group Relative Policy Optimization (GRPO), a variant reinforcement learning (RL) algorithm of Proximal Policy Optimization (PPO)"
- **RL sharpens the output distribution; it does not add capability.** (p.21)
  > "RL enhances Maj@K’s performance but not Pass@K."
- Interpretation: **improvement comes from boosting the correct response from the top-K**. (p.21)
  > "the improvement is attributed to boosting the correct response from TopK rather than the enhancement of fundamental capabilities."

## Relevance to OpenJev

OpenJev context: Gemma 4 E2B (2.3B effective), option-scoring flag head, SFT then exact-PG/GRPO with a Brier reward.

For a fixed-option decision model this is the favorable regime: the right option is almost always in the sampling distribution after SFT, and RL's job (sharpening / reweighting toward the correct or calibrated option) is exactly what we want. No size comparison in the paper (7B only).
