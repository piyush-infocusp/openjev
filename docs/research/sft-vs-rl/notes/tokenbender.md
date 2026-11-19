# tokenbender (X @tokenbender, GitHub tokenbender, tokenbender.com)

Status: NO verified statement of the "direct RL vs SFT->RL" argument found. X posts were not fetchable
(xcancel 451, nitter dead/legal notice, x.com 402). 13 searches/fetches used. Everything below is what I could verify;
the user's remembered thesis is UNVERIFIED.

## User's remembered thesis (UNVERIFIED)
"If the LLM cannot do a task at all -> SFT. If it succeeds in some tries and fails in others -> RL."
- Could not find the post, quote, or link. No pass@k / solve-rate criterion of theirs found.
- Note: this matches common secondary guidance (RL needs a non-zero success rate, so GRPO groups have reward variance;
  e.g. Modulai blog, "Hitchhiker's Guide to Agentic AI" rule of thumb pass@64 < 1.5x pass@1 -> better SFT first). That is NOT attributed to tokenbender.

## Verified material (their own pages)
1. avataRL, "rl from zero pretrain, can it be done? yes." Repo: https://github.com/tokenbender/avataRL (README dates 28 May - 6 Aug 2025).
   - Setup: GPT-2-style models from RANDOM weights, trained purely with RL, no cross-entropy. Sizes 30M-250M (blog: "Coherent language from random-weight models across the 30M-250M range").
   - Reward: next-token prediction scored against ground truth with partial credit (bigram objective, 29 May 2025; partial reward scaled 0 -> 0.3, 11-24 Jul 2025), plus a smaller critic/"referee" (30M critic, ~2B tokens OpenWebText).
   - Results claimed: "reach similar cross-entropy loss as a standard pretrained model" at same size; "equivalent of 50-60B tokens on 2B tokens of data" (blog, https://tokenbender.com/posts/avatarl/). No benchmark table seen; I did not verify the numbers.
   - This is "no pretraining, no SFT" RL, but with a dense shaped reward; it is NOT R1-Zero on a capable base and not a classification task.
2. Reward sparsity (blog https://tokenbender.com/posts/avatarl/, via fetch summary; quotes as returned, exact wording not re-checked against page):
   - "without pretraining, the probability of next token being correct is random chance" (~1/50k = 0.002%; with 64 rollouts 0.128%).
   - "makes it computationally extremely expensive to train a model using pure RL."
   - "If the reality expert is one-hot, rewards become ultra-sparse and the student gets almost no exploration signal."
   - "we do pretraining before RL is to make sure the most probable actions are the ones" likely to be correct.
   - "we cap the student at the critic's quality."
   - Summarising a view they disagree with: "People believing without strong base model RL doesn't work."
   Takeaway: their own fix for "model can't do the task at all" was dense partial-credit reward, not SFT. Consistent in spirit with the remembered rule (sparse reward when success ~0 is the problem).
3. infinite repo/post (rubric-driven prioritized replay RL; "The model's failures should choose what it learns next"; later beat uniform GRPO on MATH-500/AIME per paper arXiv 2601.02648): https://tokenbender.com/ . Numbers not retrieved.
4. Older SFT work (EvolvedSeeker 1.3B, 68.29% HumanEval, 2023; Navarna 7B SFT+DPO 2024): shows SFT practice, no RL-vs-SFT claim.
5. X post 31 May 2025 "for those of you following pure rl shenanigans, here's the complete plan" https://x.com/tokenbender/status/1928850199893684690 (title only, via search; body not read).

## Not found
- Any statement on entropy collapse, distillation vs RL, cold-start data, classification / multiple-choice / option scoring, or models <4B in the SFT-vs-RL sense (their 30M-250M models are pretraining-from-scratch, a different regime).
- Any direct comparison to R1-Zero, Dr. GRPO, or Yue et al. 2025. (Yue-style "RL only elicits" ideas are the generic literature; not attributed to them.)

## Relevance to OpenJev
- The remembered rule, if accurate, is the standard "RL needs non-zero success" criterion. For us it is testable rather than arguable:
  measure, before any RL, the solve rate of the initial policy. With a NEW RANDOM head over a fixed option list, the initial policy is near uniform over N options, so success ~1/N per sample, not 0. Exact policy gradient (we score all options) removes the sampling-sparsity issue entirely.
- But the base features still must carry task signal: a random head on frozen-then-tuned Gemma 4 E2B is "can do some of the time" only at chance level; label SFT (cross-entropy on the head) is the cheapest way to get off chance and fix the head. This matches README synthesis: SFT (short) -> RL.
- avataRL shows RL from random weights is possible with a dense reward and a critic; our Brier reward is dense over the option distribution, so direct RL is not ruled out, but it is slower and has no evidence on classification. Treat as an ablation: direct exact-PG from random head vs SFT->PG, compare at equal compute.
- Verdict: SFT -> RL (short SFT), with a direct-RL ablation. Confidence moderate; tokenbender evidence is weak/indirect.
