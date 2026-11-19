# 2025-deepseek-r1

**Citation:** DeepSeek-AI, 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning', 2025 (Nature version of arXiv 2501.12948).
**arXiv:** https://arxiv.org/abs/2501.12948
**Local PDF:** ../papers/2025-deepseek-r1.pdf (highlighted: ../papers/2025-deepseek-r1.highlighted.pdf)

**Model sizes used:** R1-Zero on DeepSeek-V3-Base (685B); distilled Qwen2.5 students 1.5B/7B/14B/32B (+Llama 8B/70B); Qwen2.5-32B-Zero ablation.

**Recipe compared:** R1-Zero (pure RL, no SFT) vs R1 (cold-start SFT + RL) vs distillation (SFT on R1 traces) vs RL on a small model.

## Key findings

- **Pure RL from base is possible** (R1-Zero), but has usability problems. (p.2)
  > "DeepSeek-R1-Zero, which relies exclusively on reinforcement learning without supervised fine-tuning"
- R1-Zero suffers **poor readability / language mixing**; motivates cold-start SFT before RL. (p.2)
  > "it faces challenges such as poor readability and language mixing"
- **For small models, distillation (SFT on strong-teacher traces) beats running large-scale RL on the small model itself.** (p.62)
  > "distilling more powerful models into smaller ones yields excellent results, whereas smaller models relying on the large-scale RL mentioned in this paper require enormous computational power and may not even achieve the performance of distillation."

## Relevance to OpenJev

OpenJev context: Gemma 4 E2B (2.3B effective), option-scoring flag head, SFT then exact-PG/GRPO with a Brier reward.

Direct RL on a 2B model is NOT the R1 recommendation: the paper's own small-model evidence (Qwen2.5-32B-Zero, p62) favors SFT-style transfer. But our task is a short-output decision, not long CoT reasoning, so the distillation point maps to 'SFT on teacher/ground-truth labels first'. Evidence is for math/code reasoning, not classification.
