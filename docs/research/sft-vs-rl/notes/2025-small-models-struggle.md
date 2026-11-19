# 2025-small-models-struggle

**Citation:** Li et al., 'Small Models Struggle to Learn from Strong Reasoners', 2025.
**arXiv:** https://arxiv.org/abs/2502.12143
**Local PDF:** ../papers/2025-small-models-struggle.pdf (highlighted: ../papers/2025-small-models-struggle.highlighted.pdf)

**Model sizes used:** Qwen2.5-Instruct 0.5B/1.5B/3B/7B/14B/32B; Llama3.2-1B/3B, Llama3.1-8B, Llama3.3-70B (p2).

**Recipe compared:** SFT/distillation only (long vs short CoT, large vs small teacher, Mix Distillation). No RL.

## Key findings

- **Small models (<=3B) do not benefit consistently from long CoT / strong-teacher distillation: the 'Small Model Learnability Gap'.** (p.1)
  > "small models (≤3B parameters) do not consistently benefit from long chain-of-thought (CoT) reasoning or distillation from larger models."
- They do better on **shorter, simpler targets** matched to their capacity. (p.1)
  > "they perform better when fine-tuned on shorter, simpler reasoning chains that better align with their intrinsic learning capacity."

## Relevance to OpenJev

OpenJev context: Gemma 4 E2B (2.3B effective), option-scoring flag head, SFT then exact-PG/GRPO with a Brier reward.

Size effect on the SFT stage: at ~2B, keep SFT targets short and simple (option label + direct probabilities, not long teacher rationales). Does not cover RL, so the interaction with RL is not tested here.
