# SFT vs RL for small decision / option-scoring models

Context: OpenJev = Gemma 4 E2B (2.3B effective), option-scoring flag head, SFT then exact-PG/GRPO with a Brier reward.
Per-paper notes in `notes/`, PDFs (plus `.highlighted.pdf`) in `papers/`. Papers 2025-2026 are as retrieved from arXiv; I read abstracts, key sections and tables, not every page (the 80-page calibration paper only in part).

| Paper | Sizes | Recipes | Main finding | Verdict for us |
|---|---|---|---|---|
| DeepSeek-R1 (2501.12948) | 685B Zero; distills 1.5B-32B | Zero (RL only) vs cold-start SFT+RL vs distillation | Pure RL works at scale but is messy; for small models distillation beats RL on the small model | Weak support for SFT first; math/CoT, not classification |
| SFT Memorizes, RL Generalizes (2501.17161) | 11B | SFT, RL, SFT->RL, RL from base | RL generalizes OOD better; SFT needed so RL can start; over-tuned SFT hurts RL | Supports SFT->RL with light SFT |
| DeepSeekMath/GRPO (2402.03300) | 7B | SFT then GRPO | RL improves Maj@K, not Pass@K: sharpening, not new capability | Good fit: RL as sharpening/calibration |
| Dr. GRPO (2503.20783) | 1.5B-8B (+685B) | Direct RL from base | Bases already capable; GRPO length/std bias; Qwen2.5 may be special | Use unbiased advantages (no std norm) |
| RL beyond base? (2504.13837) | 7B-32B | RLVR vs base vs distill | RL bounded by base at large k; distillation adds new patterns | RL cannot add knowledge; SFT does |
| Small Models Struggle (2502.12143) | 0.5B-70B | SFT/distillation only | <=3B models learn better from short/simple targets | Keep SFT targets short; no RL evidence |
| RLCR (2507.16806) | 7B | RLVR vs correctness+Brier | Binary RL hurts calibration; Brier reward fixes it at equal accuracy (ECE 0.37->0.03) | Core support for Brier reward (7B only) |
| Rewarding Doubt (2503.02623) | 3B-9B | PPO + log-score reward | Calibration improves at 3B-9B; a trained probe gets lower ECE | Use post-hoc calibration as baseline |
| GTA (2509.12108) | 3B-4B | SFT vs GRPO vs SFT-guided RL | Pure GRPO below SFT on classification; hybrid ~+0.5-1 pt over SFT | Best classification evidence: SFT->RL, small gain |
| EXTC (2605.29076) | 4B | SFT distill then GRPO | Rollout homogeneity (zero-advantage groups) in classification RL; SFT+RL > SFT | Favors exact PG / filtering |
| ChestPheNoT (2609.31629) | 0.5/1.5/3B | SFT vs SFT+GRPO | RL ~0 in-distribution, small OOD gains, no clear size trend (n=1) | Expect small gains; evaluate OOD |
| Reward fns for calibration RL (2607.04332) | 3B, 4B, Gemma 4 E2B | Dr. GRPO/DAPO, Brier/log/overconfidence rewards | Badly designed calibration rewards cause hacking (answer wrong, confidence 0) | Closest to us; copy non-hackable reward form |

## Synthesis (~2B decision model)
1. Recommend SFT then RL, with a short SFT. (moderate confidence)
2. SFT supplies the knowledge, label format and a well-formed option distribution; RL cannot add capability beyond what the model can already sample (Yue, DeepSeekMath).
3. Direct RL is the riskiest at 2B: R1 says small models do worse with RL than distillation; Chu et al. show RL fails when the start does not follow the format; pure GRPO < SFT on 3-4B classification (GTA). Gemma E2B-it may follow format already, so direct RL could work: untested here.
4. SFT-only is a solid baseline, but it gives no calibration optimization; ordinary SFT/binary-reward RL tends to be miscalibrated (RLCR), and RL with a proper score fixes that without losing accuracy (7B; unconfirmed at 2B).
5. Expect modest RL gains on accuracy (GTA ~+1 pt, ChestPheNoT ~0 in-distribution), larger on calibration and under distribution shift. Evaluate on OOD and Brier/ECE, not just accuracy.
6. Size interaction: UNCERTAIN. No paper cleanly measures SFT+RL gain vs size in 0.5-10B. ChestPheNoT shows no monotone trend (0.5-3B, one run); Rewarding Doubt hints 3B stays worse-calibrated than 8B; Small Models Struggle says <=3B need simpler SFT targets. I found no evidence that RL gains clearly grow or shrink with size.
7. Reward design: use correctness plus a bounded proper score (Brier); calibration-only collapses accuracy and log score can be non-monotonic (RLCR); watch for hacking (2607.04332).
8. Algorithm: drop std advantage normalization (Dr. GRPO, 2607.04332). Gemma 4 E2B needed DAPO for stability in 2607.04332.
9. Classification RL risk: rollout homogeneity gives zero advantage (EXTC). Exact policy gradient over the option distribution avoids it by design, which supports our exact-PG plan (inference, not a measured result).
10. Keep SFT light (few epochs, class-balanced); over-tuned SFT can leave RL stuck (Chu et al.). Always compare against SFT plus temperature/Platt scaling on the flag head (Rewarding Doubt probe baseline).

Gaps: nothing here tests an option-scoring head with an RL reward directly; all calibration-RL papers use verbalized confidence on generative QA/math.
