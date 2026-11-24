---
name: token-budget-batching
description: Parked OpenJev technique (user, 2026-11-23) - merge short length-grouped micro-batches (e.g. 8x128 + 8x256) into one forward/backward pass under a token budget, with exact gradient accumulation. Use only after the speed-optimisation phase is complete.
---

# Token-budget batching (parked)

**Status:** parked. The user asked to keep this separate and use it once the Optuna speed optimisation is complete. Do not wire it into train_prism.py defaults before then.

## Idea
With length-grouped batches, short micro-batches (8x<=128, 8x<=256) may underuse the GPU. Fill each micro-batch up to a token budget instead of a fixed 8 records, e.g. budget 4096: 32x128, 16x256, 8x512, 4x1024, 2x2048. The global batch stays 32 records per optimizer step.

## Exactness rules
1. Sum-reduce the per-record loss in fp32 over every micro-batch of the step.
2. Divide once by a global denominator fixed before the accumulation loop (32 records, or total target tokens). Never average per-micro-batch means (HF grad-accum bug, 2024).
3. Batch-coupled terms (soft ECE, any softmax over the batch) are not additive: use a GradCache-style two-pass (no-grad stats pass, then chunked re-forward with cached grads), or make the term per-record.
4. Test: gradients from one-by-one vs merged partitions match to ~1e-5 on the tiny Gemma.

## Expected gain on T4
Small unless the GPU is under-used on short buckets: r7 tokens/s was nearly flat (~600-730) across buckets. Gains come from less padding and fewer kernel launches (no FlashAttention-2 / varlen kernel on sm75). Benchmark on the fixed bench set (study openjev-speed-v3) before adopting.

## References
- fairseq `--max-tokens`: https://fairseq.readthedocs.io/en/latest/command_line_tools.html
- HF gradient accumulation fix: https://huggingface.co/blog/gradient_accumulation
- Dataset Decomposition, arXiv:2405.13226
- GradCache, arXiv:2101.06983
- Brief: docs/research/token-budget-batching.md

## How to apply later
1. Add `--token-budget N` (0 = off) next to `--length-grouping` in train_prism.py, sorting each step's 32 records and filling micro-batches greedily.
2. Switch loss normalisation to sum / global denominator; handle soft-ECE per rule 3.
3. Add the gradient-equality test, then one Optuna trial on the bench set vs the current best.

## Calculation (2026-11-23)
Data: r7 1xT4 baseline (gemma-4-E2B, fp16, grad ckpt on, mb 8, global 32, bench set 48758bf206de5268). Scripts were throwaway (scratchpad, not in the repo).

**Exactness (tiny Gemma 4, CPU fp32).** The current code is not sum / global denominator: each micro-batch takes a mean over its questions, divides by accum_local, and has its own soft ECE. A merged pass reproduces that gradient only with a **segment-wise loss**: run one forward, then compute each original micro-batch's loss on its slice of the logits (including its own soft ECE), sum, and divide by M. Measured relative gradient error:
- segment-wise: 2e-6 to 1e-5 (SFT and exact-PG, padded or packed at pack_len 128)
- one naive mean over the merged batch: SFT 17% off (7 vs 11 questions); exact-PG 53% off with equal counts (7/7). The PG error comes from soft ECE alone, because it does not add across batches.
- GRPO (not run): exact only if each segment samples with its own `grpo_generator(device, seed, step, k)`.
- A variable number of micro-batches per step also breaks `/ accum_local`.

**Time model.** t_mb = a + b*padded_tokens + c*sum(row_len^2). Least squares gives a = -0.04 s, which is noise, so refit with a = 0: b = 1.147 ms/tok, c = 0.228 us/tok^2, fit error under 1%. Cost per padded token rises from 1.17 to 1.43 ms going from <=128 to <=2048, and the attention term explains that rise (2.5% to 20% of the time). The <=128 bucket is already the cheapest per token. With **no fixed cost per call**, the short micro-batches are not under-using the T4 and there is nothing to amortise.

| option | est. time / gain | peak GB (4.64 + 0.43/1k tok) |
|---|---|---|
| separate 8x128 + 8x256 | 3.59 s (baseline) | 5.1 / 5.5 |
| merged padded 16x256 | 4.77 s (**-25%**) | 6.3 |
| merged packed 12x256 | 3.58 s (+0.3%) | ~5.9 |
| token budget 2048 / 4096 / 8192 (whole mix) | **+0.0%** (time per sample does not depend on n when a = 0) | max 5.5 / 6.4 / 8.0 |
| budget 16384 | +0.0% | max 11.2 (fits 14.5) |

Bounds: <=128 and <=256 are 7.1% of the weighted time, so even if they cost nothing the gain is at most +7.6%. One timed step per bucket plus fp16 skipped steps means about +/-1% is noise. The memory fit underpredicts the 1024 bucket by about 0.25 GB.

**Verdict: not worth it. Keep this parked and run no round-8 trial.** Merging gains nothing because there is no fixed cost per call. Padding to the longer row hurts. Token budgets only make soft ECE a different objective, and they break the user's fixed mb-8 rule. The one real lever is padding (pad_frac about 20%), and it needs packing across rows. `--bench-lengths` ignores packing / pack_len today, so measuring it would need a bench change first. If this is ever built, use `--token-budget 0` (default off) with a segment-wise loss and the gradient-equality test.
