# 2025-gta-text-classification

**Citation:** Zeng et al. (vivo AI Lab), 'GTA: Supervised-Guided Reinforcement Learning for Text Classification with LLMs', EMNLP Findings 2025.
**arXiv:** https://arxiv.org/abs/2509.12108
**Local PDF:** ../papers/2025-gta-text-classification.pdf (highlighted: ../papers/2025-gta-text-classification.highlighted.pdf)

**Model sizes used:** Qwen2.5-3B, Qwen3-4B, Llama3.2-3B (p5-6); SST-5, Amazon, Emotion, BBC News.

**Recipe compared:** SFT vs pure GRPO vs GTA (SFT-guided 'guess' + RL 'think/answer').

## Key findings

- **Pure GRPO underperforms plain SFT on classification** at 3-4B (Table 2: Qwen2.5-3B Emotion acc SFT 91.35 vs GRPO 82.50; Llama3.2-3B Emotion SFT 93.00 vs GRPO 74.40). (p.2)
  > "These efficiency challenges have hindered pure RL-based methods from consistently outperforming SFT approaches despite their stronger theoretical foundation."
- **Hybrid beats both, but by small margins over SFT** (Qwen2.5-3B Emotion 92.45 vs 91.35; SST-5 61.58 vs 60.72). (p.1)
  > "This hybrid approach achieves both faster convergence than pure RL and higher performance ceiling than pure SFT."

## Relevance to OpenJev

OpenJev context: Gemma 4 E2B (2.3B effective), option-scoring flag head, SFT then exact-PG/GRPO with a Brier reward.

Best classification-specific evidence: for 3-4B models and a fixed label set, SFT is a strong baseline, pure RL is worse, SFT+RL gains ~0.5-1 pt. Caveat: accuracy only, no calibration; single runs; GTA modifies the output format (guess then think) so it is not a pure SFT->GRPO.
