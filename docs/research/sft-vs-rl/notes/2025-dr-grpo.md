# 2025-dr-grpo

**Citation:** Liu et al., 'Understanding R1-Zero-Like Training: A Critical Perspective' (Dr. GRPO), COLM 2025.
**arXiv:** https://arxiv.org/abs/2503.20783
**Local PDF:** ../papers/2025-dr-grpo.pdf (highlighted: ../papers/2025-dr-grpo.highlighted.pdf)

**Model sizes used:** Qwen2.5-Math-1.5B/7B, Qwen2.5-7B, Llama-3.1-8B, DeepSeek-Math-7B, DeepSeek-V3-Base 685B.

**Recipe compared:** Direct RL from base (R1-Zero-like) with GRPO vs unbiased Dr. GRPO; no SFT stage.

## Key findings

- **Base models already have the capability before RL;** Qwen2.5 may be pretrained on QA-style text. (p.2)
  > "all base models already possess math-solving capability prior to RL."
- Qwen2.5 bases work well even without a template: **possible pretraining bias**, so R1-Zero results on Qwen may not transfer to other families. (p.1)
  > "Qwen2.5 base models demonstrate strong reasoning capabilities even without prompt templates, suggesting potential pretraining biases."
- **GRPO has an optimization bias** (length and std normalization) that inflates incorrect-response length; Dr. GRPO removes both. (p.1)
  > "we identify an optimization bias in Group Relative Policy Optimization (GRPO), which artificially increases response length"
- Minimalist recipe: **direct RL of a 7B base with Dr. GRPO**. (p.2)
  > "we RL-tune Qwen2.5-Math-7B using the (unbiased) Dr. GRPO algorithm"

## Relevance to OpenJev

OpenJev context: Gemma 4 E2B (2.3B effective), option-scoring flag head, SFT then exact-PG/GRPO with a Brier reward.

Directly relevant to our exact-PG/GRPO choice: dropping std normalization matters for a continuous Brier reward (small reward differences are amplified by /std). Also a caution that direct-RL success is base-model dependent (Qwen2.5 is unusual); Gemma is not shown. Math, 1.5B-8B.
