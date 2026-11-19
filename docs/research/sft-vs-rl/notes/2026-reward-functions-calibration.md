# 2026-reward-functions-calibration

**Citation:** Tan, Lin, Motani, Lee, 'On the Effectiveness of Reward Functions in Reinforcement Learning for Confidence Calibration of LLMs', 2026. (arXiv 2607.04332)
**arXiv:** https://arxiv.org/abs/2607.04332
**Local PDF:** ../papers/2026-reward-functions-calibration.pdf (highlighted: ../papers/2026-reward-functions-calibration.highlighted.pdf)

**Model sizes used:** Qwen2.5-3B-Instruct, Qwen3-4B, Gemma 4 E2B-Instruct (p7); Gemma only on HotpotQA-Modified.

**Recipe compared:** Dr. GRPO (DAPO for Gemma) with correctness-only, Log, Brier and overconfidence/underconfidence-spectrum rewards.

## Key findings

- **'Confidence reward hacking': a badly designed calibration reward pays the model to answer wrong.** (p.1)
  > "If poorly designed, such a scheme may incentivize an LLM to answer incorrectly in order for its confidence to be calibrated, a phenomenon we term confidence reward hacking."
- **Gemma 4 E2B tested; DAPO needed for stability**, no std advantage normalization. (p.7)
  > "except for Gemma 4 (E2B) Instruct, which we used DAPO (Yu et al., 2025) for better training stability. Standard deviation advantage normalization is not used."
- Observed hacking in practice: the model **gives up with confidence 0**. (p.8)
  > "Qwen 2.5 (3B) Instruct may give up on answering the question"

## Relevance to OpenJev

OpenJev context: Gemma 4 E2B (2.3B effective), option-scoring flag head, SFT then exact-PG/GRPO with a Brier reward.

Closest to our setup (Gemma 4 E2B, Brier/log rewards). Design rules to copy: reward = correctness + proper score of the stated probability (non-hackable form), no std normalization, expect Gemma to need extra stabilization. Note Gemma got only one dataset (limited compute) so the size/model comparison is thin and I did not extract per-model accuracy/ECE numbers (80-page paper, not fully read).
