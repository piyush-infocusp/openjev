# 2025-rewarding-doubt

**Citation:** Bani-Harouni, Pellegrini et al., 'Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of LLMs', ICLR 2026.
**arXiv:** https://arxiv.org/abs/2503.02623
**Local PDF:** ../papers/2025-rewarding-doubt.pdf (highlighted: ../papers/2025-rewarding-doubt.highlighted.pdf)

**Model sizes used:** Llama-3-8B-Instruct / Llama-3.1-8B (main), Qwen-2.5 3B and 7B, Gemma-2-9B (p8).

**Recipe compared:** Log-scoring-rule reward, PPO, LoRA on 4-bit models (p6); baselines: verbalize, trained probe.

## Key findings

- **Log-score reward directly penalizes over- and under-confidence.** (p.1)
  > "optimizes a reward based on the logarithmic scoring rule, explicitly penalizing both over- and under-confidence."
- Works across **3B-9B**; authors only expect it to hold at larger scale (no scale trend measured). (p.10)
  > "on models with sizes ranging from 3B to 9B parameters. While we expect similar effectiveness on larger models, empirical validation on such scales would be valuable."
- Table 3, Qwen-2.5-7B: verbalized ECE 0.3619 -> Rewarding Doubt 0.1298 at ~equal accuracy; but a **separately trained probe has lower ECE (0.0989)** while RD has better AUROC. (p.9)
  > "Qwen-2.5-7B Verbalize 0.3619 [0.3530,0.3705] 0.5818 [0.5762,0.5879] 0.5239 [0.5148,0.5331] Trained probe 0.0989"

## Relevance to OpenJev

OpenJev context: Gemma 4 E2B (2.3B effective), option-scoring flag head, SFT then exact-PG/GRPO with a Brier reward.

Evidence that a proper-scoring RL reward calibrates small (3B) models, and that a simple trained head/probe is a strong calibration baseline: a reminder to compare RL against SFT + temperature/Platt scaling on the flag head. Smaller models (3B) stay worse-calibrated (ECE 0.148) than 8B (0.026), a weak hint calibration RL gains scale with size, but model families differ so this is confounded.
