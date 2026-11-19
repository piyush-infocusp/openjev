# 2025-sft-memorizes-rl-generalizes

**Citation:** Chu et al., 'SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training', 2025.
**arXiv:** https://arxiv.org/abs/2501.17161
**Local PDF:** ../papers/2025-sft-memorizes-rl-generalizes.pdf (highlighted: ../papers/2025-sft-memorizes-rl-generalizes.highlighted.pdf)

**Model sizes used:** Llama-3.2-Vision-11B (p5) on GeneralPoints (arithmetic card game) and V-IRL navigation. Outside our size range.

**Recipe compared:** SFT vs RL (outcome reward) vs SFT-then-RL vs RL straight from the base model.

## Key findings

- **RL generalizes better OOD than SFT, but SFT is still needed to make RL work.** (p.1)
  > "Despite RL’s superior generalization, we show that SFT is still helpful for effective RL training"
- SFT's role is **format stabilization** so RL can start. (p.1)
  > "SFT stabilizes the model’s output format, enabling subsequent RL to achieve its performance gains."
- **Direct RL from a non-instruction-following base fails.** (p.7)
  > "SFT is necessary for RL training when the backbone model does not follow instructions."
- **Over-tuned SFT can poison RL.** (p.9)
  > "RL is unable to recover out-of-distribution performance when starting from such a checkpoint."

## Relevance to OpenJev

OpenJev context: Gemma 4 E2B (2.3B effective), option-scoring flag head, SFT then exact-PG/GRPO with a Brier reward.

Supports SFT -> RL. Direct RL only works if the starting model already follows the output format; Gemma 4 E2B-it does, the base may not. Warns against over-training SFT (many epochs on a small label set) before RL: it can collapse and RL cannot recover.
