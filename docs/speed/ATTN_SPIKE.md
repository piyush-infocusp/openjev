# Attention-kernel spike: fused attention with the Prism mask on T4

Kernel **piyushsar/openjev-attn-spike v2** (v1 sat in the Kaggle queue for 2 h and was replaced), Tesla T4 (sm 7.5), torch
2.11.0+cu130, CUDA 13.0, Python 3.13.16. Script: `kaggle/attn_spike/bench_attn.py`. Raw output:
`runs/optuna/attn_spike/out/` (`attn_spike.json`, `xf_rows.json`, kernel log). Date: 2026-11-23.

## TL;DR

- **The current path is not the math fallback.** In fp16, SDPA's default dispatch already runs the cutlass
  **memory-efficient** kernel (`aten::_efficient_attention_forward/backward`) for head_dim 256 **and** 512, with both the
  bool (`sdpa_bool`) and the float (`sdpa_default`) Prism mask. The trial-23 profile (r4, fp16) showed the same thing in the
  real model: efficient-attention bwd 13% + fwd 7% of CUDA time.
- **The legacy "No available kernel" was bf16, not head_dim, GQA or the mask.** Round 1 ran emulated bf16 (OJ-081). On sm75
  the mem-efficient kernel accepts only fp16/fp32: forced EFFICIENT with bf16 fails with exactly
  `RuntimeError: No available kernel. Aborting execution.`, and the warning says
  `Expected query, key and value to all be of dtype: {Half, Float}. Got Query dtype: c10::BFloat16 ...` (the same warning
  is in the round-1 log). In fp16, forced EFFICIENT runs forward and backward at head_dim 512 with both mask types.
- Mem-efficient does **not** accept `enable_gqa=True` (unrepeated KV) with a mask: `RuntimeError: No available kernel`.
  Only math does. transformers repeats KV when a mask is given, so this does not affect us.
- **xformers 0.0.35** (`--no-deps`, from the cu130 PyTorch index; torch stayed 2.11.0+cu130) runs fwd+bwd with the Prism
  bias at both head dims. It is about level with SDPA's efficient kernel at T ≥ 512 (0.95–1.12x; 0.54–0.88x at T ≤ 256), because it is the same cutlass
  fMHA family. xformers 0.0.36 needs the separate `mslk` package (`No module named 'mslk'`).
- Explicit EFFICIENT with a float or bool mask is the same kernel as the default dispatch: 0.97–1.03x at T ≥ 512.
- Surprise: **our `chunked_math` beats the cutlass kernel**: 1.02–1.16x at head_dim 256, 1.8–1.9x at head_dim 512 (T ≥ 512).
  Likely reasons: it folds the 8 query heads into one cuBLAS GEMM per KV head (no `repeat_kv` copy, no dK/dV reduction
  over 8 heads), and cutlass mem-efficient on sm75 is weak at head_dim 512.
- **End-to-end** (r7 fit): at best **+2.5%** weighted samples/s (`chunked_math` on all layers). Every other candidate is
  0.95–1.00x. r4 already measured `chunked_math` in the real model: 0.829 vs 0.830 samples/s for `sdpa_bool` (noise).

**Verdict: do not integrate a new attention backend.** Nothing beats the kernel we already use by enough to matter. At most
an optional, cheap follow-up: one paired Optuna trial `attn_backend=chunked_math` vs `sdpa_bool` on the multi-length bench
set (r7 conditions). The expected gain is ≤2.5%, inside trial noise, so only do it if a slot is free.

## Setup

- No model load. q `(8, 8, T, D)`, k/v `(8, 1, T, D)` leaf tensors (GQA 8:1, repeated to 8 heads inside the candidate the
  way transformers' `repeat_kv` does, so dK/dV reduce to the 1 KV head as in the model), fp16, requires_grad, random
  `grad_output`, scale `D^-0.5`.
- head_dim 256 uses the **sliding** mask (window 512; the 28 sliding layers). head_dim 512 uses the **full** mask (the 7
  global layers). seq ∈ {128, 256, 512, 1024, 2048}.
- Mask: verbatim copies of `openjev.prism.layout` and `prism_mask` (commit b71f4f0; checked equal to the package
  function) on synthetic requests. Each request has a shared state block, then 1–3 questions, each a shared header plus
  2–10 options (`<opt>` label `<flag>`) that share a start position. Rows are length-grouped (0.85T–T), right-padded, with
  `doc` set. Density: 0.66–0.86. No real bench records, to keep the kernel standalone. The mem-efficient kernel does not
  skip masked tiles, so mask content does not change its timing.
- Timing: forward + backward, CUDA events, 3 warm-up runs, median of 10. Peak = `max_memory_allocated` above the inputs.
  Diff = max abs vs an fp32 math reference (computed one batch row at a time).
- The dq/dk/dv diff is 3.9e-3 (= 2^-8) everywhere. That is the fp16 rounding of gradients of magnitude 2–8 (dK/dV sum 8
  heads), not a kernel error. `chunked_math`'s output diff is ~4x higher (≤2.1e-3) because it casts the softmax to fp16
  before P·V, like eager attention.
- Baseline "current" = per-cell median of the four variants that run the same mem-efficient kernel (a0_default_bool,
  a0_default_float, b_efficient_bool, b_efficient_float). One of them alone can carry a consistent per-candidate offset:
  a0_default_bool at 256/256 is a steady 5.1–5.5 ms (it runs right after math), while the other three land at 3.7–3.9 ms.
  Short-T cells (≤256) are small and noisy. Trust T ≥ 512.

## Results (fwd+bwd, B=8, 8q/1kv heads, fp16)

`a0_default_bool` = `attn_backend=sdpa_bool` (r7 path); "vs current" = vs the median of the four mem-efficient variants; `a0_default_float` = `sdpa_default`; `a_math_bool` = forced
math (what we wrongly assumed was the current path); `b_*` = forced `SDPBackend.EFFICIENT_ATTENTION`; `c_xformers` =
`xformers.ops.memory_efficient_attention` with the additive bias (last dim allocated padded to a multiple of 8);
`e_chunked_math` = our backend (chunk 256, per-chunk checkpoint). "ran efficient" = the aten op the dispatcher picked
(torch.profiler).

| candidate | head_dim | seq | fwd+bwd ms | vs math | vs current | peak MB | max diff out / dq,dk,dv | status |
|---|---|---|---|---|---|---|---|---|
| a0_default_bool | 256 | 128 | 2.40 | 1.29x | 1.01x | 44.3 | 5.0e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_bool | 256 | 256 | 5.31 | 1.29x | 0.72x | 81.1 | 3.7e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_bool | 256 | 512 | 13.27 | 1.58x | 1.00x | 140.3 | 2.4e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_bool | 256 | 1024 | 47.97 | 1.52x | 1.01x | 264.5 | 1.7e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_bool | 256 | 2048 | 185.65 | 1.50x | 1.01x | 537.1 | 2.1e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_bool | 512 | 128 | 3.04 | 1.18x | 0.85x | 88.3 | 4.9e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_bool | 512 | 256 | 8.44 | 1.47x | 1.00x | 161.1 | 4.5e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_bool | 512 | 512 | 28.69 | 1.41x | 1.00x | 276.3 | 2.5e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_bool | 512 | 1024 | 110.21 | 1.34x | 1.01x | 512.6 | 1.9e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_bool | 512 | 2048 | 481.47 | 1.34x | 1.01x | 1009.1 | 1.2e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_float | 256 | 128 | 2.40 | 1.29x | 1.01x | 44.3 | 5.0e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_float | 256 | 256 | 3.75 | 1.82x | 1.02x | 81.1 | 3.7e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_float | 256 | 512 | 13.29 | 1.58x | 1.00x | 140.3 | 2.4e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_float | 256 | 1024 | 48.51 | 1.51x | 1.00x | 264.5 | 1.7e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_float | 256 | 2048 | 186.59 | 1.49x | 1.01x | 537.1 | 2.1e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_float | 512 | 128 | 2.57 | 1.39x | 1.00x | 88.3 | 4.9e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_float | 512 | 256 | 8.41 | 1.48x | 1.01x | 161.1 | 4.5e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_float | 512 | 512 | 28.63 | 1.41x | 1.00x | 276.3 | 2.5e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_float | 512 | 1024 | 110.84 | 1.33x | 1.00x | 512.6 | 1.9e-04 / 3.9e-03 | ok, ran efficient |
| a0_default_float | 512 | 2048 | 499.99 | 1.29x | 0.98x | 1009.1 | 1.2e-04 / 3.9e-03 | ok, ran efficient |
| a_math_bool | 256 | 128 | 3.10 | 1.00x | 0.78x | 58.2 | 4.8e-04 / 3.9e-03 | ok |
| a_math_bool | 256 | 256 | 6.84 | 1.00x | 0.56x | 129.0 | 3.2e-04 / 3.9e-03 | ok |
| a_math_bool | 256 | 512 | 21.00 | 1.00x | 0.63x | 368.0 | 2.4e-04 / 3.9e-03 | ok |
| a_math_bool | 256 | 1024 | 73.07 | 1.00x | 0.66x | 1248.0 | 1.2e-04 / 3.9e-03 | ok |
| a_math_bool | 256 | 2048 | 278.75 | 1.00x | 0.68x | 4544.0 | 2.1e-04 / 3.9e-03 | ok |
| a_math_bool | 512 | 128 | 3.58 | 1.00x | 0.72x | 110.2 | 4.9e-04 / 3.9e-03 | ok |
| a_math_bool | 512 | 256 | 12.44 | 1.00x | 0.68x | 233.0 | 4.5e-04 / 3.9e-03 | ok |
| a_math_bool | 512 | 512 | 40.51 | 1.00x | 0.71x | 516.0 | 2.4e-04 / 3.9e-03 | ok |
| a_math_bool | 512 | 1024 | 147.87 | 1.00x | 0.75x | 1472.0 | 1.9e-04 / 3.9e-03 | ok |
| a_math_bool | 512 | 2048 | 643.37 | 1.00x | 0.76x | 4992.0 | 1.2e-04 / 3.9e-03 | ok |
| b_efficient_bool | 256 | 128 | 2.46 | 1.26x | 0.99x | 44.3 | 5.0e-04 / 3.9e-03 | ok |
| b_efficient_bool | 256 | 256 | 3.92 | 1.75x | 0.98x | 81.1 | 3.7e-04 / 3.9e-03 | ok |
| b_efficient_bool | 256 | 512 | 13.32 | 1.58x | 1.00x | 140.3 | 2.4e-04 / 3.9e-03 | ok |
| b_efficient_bool | 256 | 1024 | 48.55 | 1.50x | 1.00x | 264.5 | 1.7e-04 / 3.9e-03 | ok |
| b_efficient_bool | 256 | 2048 | 190.80 | 1.46x | 0.99x | 537.1 | 2.1e-04 / 3.9e-03 | ok |
| b_efficient_bool | 512 | 128 | 2.57 | 1.39x | 1.00x | 88.3 | 4.9e-04 / 3.9e-03 | ok |
| b_efficient_bool | 512 | 256 | 8.47 | 1.47x | 1.00x | 161.1 | 4.5e-04 / 3.9e-03 | ok |
| b_efficient_bool | 512 | 512 | 28.78 | 1.41x | 1.00x | 276.3 | 2.5e-04 / 3.9e-03 | ok |
| b_efficient_bool | 512 | 1024 | 112.31 | 1.32x | 0.99x | 512.6 | 1.9e-04 / 3.9e-03 | ok |
| b_efficient_bool | 512 | 2048 | 465.36 | 1.38x | 1.05x | 1009.1 | 1.2e-04 / 3.9e-03 | ok |
| b_efficient_float | 256 | 128 | 2.47 | 1.26x | 0.98x | 44.3 | 5.0e-04 / 3.9e-03 | ok |
| b_efficient_float | 256 | 256 | 3.74 | 1.83x | 1.03x | 81.1 | 3.7e-04 / 3.9e-03 | ok |
| b_efficient_float | 256 | 512 | 13.28 | 1.58x | 1.00x | 140.3 | 2.4e-04 / 3.9e-03 | ok |
| b_efficient_float | 256 | 1024 | 48.61 | 1.50x | 1.00x | 264.5 | 1.7e-04 / 3.9e-03 | ok |
| b_efficient_float | 256 | 2048 | 189.96 | 1.47x | 0.99x | 537.1 | 2.1e-04 / 3.9e-03 | ok |
| b_efficient_float | 512 | 128 | 2.45 | 1.46x | 1.05x | 88.3 | 4.9e-04 / 3.9e-03 | ok |
| b_efficient_float | 512 | 256 | 8.63 | 1.44x | 0.98x | 161.1 | 4.5e-04 / 3.9e-03 | ok |
| b_efficient_float | 512 | 512 | 28.96 | 1.40x | 0.99x | 276.3 | 2.5e-04 / 3.9e-03 | ok |
| b_efficient_float | 512 | 1024 | 111.78 | 1.32x | 1.00x | 512.6 | 1.9e-04 / 3.9e-03 | ok |
| b_efficient_float | 512 | 2048 | 493.55 | 1.30x | 0.99x | 1009.1 | 1.2e-04 / 3.9e-03 | ok |
| c_xformers | 256 | 128 | 2.77 | 1.12x | 0.88x | 44.3 | 5.0e-04 / 3.9e-03 | ok |
| c_xformers | 256 | 256 | 7.04 | 0.97x | 0.54x | 81.2 | 3.7e-04 / 3.9e-03 | ok |
| c_xformers | 256 | 512 | 13.85 | 1.52x | 0.96x | 140.4 | 2.4e-04 / 3.9e-03 | ok |
| c_xformers | 256 | 1024 | 51.33 | 1.42x | 0.95x | 264.8 | 1.7e-04 / 3.9e-03 | ok |
| c_xformers | 256 | 2048 | 198.59 | 1.40x | 0.95x | 537.6 | 2.1e-04 / 3.9e-03 | ok |
| c_xformers | 512 | 128 | 2.95 | 1.21x | 0.87x | 88.4 | 4.9e-04 / 3.9e-03 | ok |
| c_xformers | 512 | 256 | 8.83 | 1.41x | 0.96x | 161.2 | 4.5e-04 / 3.9e-03 | ok |
| c_xformers | 512 | 512 | 29.42 | 1.38x | 0.98x | 276.4 | 2.5e-04 / 3.9e-03 | ok |
| c_xformers | 512 | 1024 | 111.82 | 1.32x | 1.00x | 512.8 | 1.9e-04 / 3.9e-03 | ok |
| c_xformers | 512 | 2048 | 436.46 | 1.47x | 1.12x | 1009.6 | 1.2e-04 / 3.9e-03 | ok |
| e_chunked_math | 256 | 128 | 2.31 | 1.34x | 1.05x | 18.6 | 1.8e-03 / 3.9e-03 | ok |
| e_chunked_math | 256 | 256 | 2.97 | 2.31x | 1.29x | 65.5 | 1.7e-03 / 3.9e-03 | ok |
| e_chunked_math | 256 | 512 | 11.42 | 1.84x | 1.16x | 157.0 | 1.0e-03 / 3.9e-03 | ok |
| e_chunked_math | 256 | 1024 | 42.14 | 1.73x | 1.15x | 306.0 | 7.5e-04 / 3.9e-03 | ok |
| e_chunked_math | 256 | 2048 | 182.01 | 1.53x | 1.03x | 604.0 | 8.4e-04 / 3.9e-03 | ok |
| e_chunked_math | 512 | 128 | 2.25 | 1.59x | 1.14x | 23.1 | 2.1e-03 / 3.9e-03 | ok |
| e_chunked_math | 512 | 256 | 3.59 | 3.47x | 2.36x | 74.5 | 2.0e-03 / 3.9e-03 | ok |
| e_chunked_math | 512 | 512 | 16.25 | 2.49x | 1.77x | 201.0 | 2.0e-03 / 3.9e-03 | ok |
| e_chunked_math | 512 | 1024 | 61.76 | 2.39x | 1.80x | 386.0 | 1.9e-03 / 3.9e-03 | ok |
| e_chunked_math | 512 | 2048 | 254.84 | 2.52x | 1.91x | 756.0 | 7.5e-04 / 3.9e-03 | ok |

Unsupported cases (exact text):

| case | error |
|---|---|
| bf16, forced EFFICIENT, head_dim 256/512, bool or float mask | `RuntimeError: No available kernel. Aborting execution.` (warning: `Expected query, key and value to all be of dtype: {Half, Float}. Got Query dtype: c10::BFloat16 ...`) |
| fp16, forced EFFICIENT, `enable_gqa=True` (unrepeated 1-head KV) + mask | `RuntimeError: No available kernel. Aborting execution.` |
| flash (any case here) | not eligible on sm75 (flash needs sm80+) |
| xformers 0.0.36 (`--no-deps`) | `AttributeError: xformers.ops.memory_efficient_attention ... unavailable because mslk could not be imported: No module named 'mslk'` |
| head_dim 512 split into two 256 halves | not valid for attention (softmax over q·k needs the full head dim); not tested. Not needed: 512 runs on mem-efficient in fp16 |

## End-to-end estimate

Per bucket: r_b = (28·t₂₅₆ + 7·t₅₁₂)_candidate / (28·t₂₅₆ + 7·t₅₁₂)_current at T = bucket bound (current = median of the 4 mem-efficient variants). New step time =
1 − share_b·(1 − r_b), where the attention share goes from 2.5% (≤128) to 20% (≤2048) log-linearly: 2.5 / 4.2 / 7.1 / 11.9 /
20%. Buckets are weighted by w_b · (r7 s/sample)_b, with w = 2.2/16.3/55.2/19.7/6.6% and r7 T0 s/sample 0.150/0.297/0.599/1.18/1.82
(the objective is 1/Σ w·s, so long buckets weigh more than their step share). The baseline is the current path (efficient
kernel), **not** math: a math baseline would overstate every gain.

| candidate | <=128 | <=256 | <=512 | <=1024 | <=2048 | weighted speedup |
|---|---|---|---|---|---|---|
| a0_default_bool | 1.03 | 1.25 | 1.00 | 0.99 | 0.99 | 1.000x |
| a0_default_float | 0.99 | 0.98 | 1.00 | 1.00 | 1.00 | 1.000x |
| a_math_bool | 1.30 | 1.67 | 1.52 | 1.44 | 1.42 | 0.954x |
| b_efficient_bool | 1.01 | 1.01 | 1.00 | 1.00 | 0.99 | 1.000x |
| b_efficient_float | 1.00 | 0.99 | 1.00 | 1.00 | 1.01 | 1.000x |
| c_xformers | 1.14 | 1.55 | 1.04 | 1.04 | 0.99 | 0.996x |
| e_chunked_math | 0.93 | 0.65 | 0.76 | 0.75 | 0.79 | 1.025x |

The cells are r_b (attention time vs current; < 1 is faster). Hybrid (chunked_math on the 7 global layers only, current
path on the sliding layers): r = 0.97 / 0.80 / 0.85 / 0.84 / 0.81, worth about +1.8%. That is less than chunked on all
layers.

Caveats: (1) In the model with `grad_ckpt` on, `chunked_math` runs a third attention forward (layer recompute + per-chunk
recompute), so part of its micro-bench win goes away. r4 measured parity in the model (0.829 vs 0.830). (2) The attention
shares are the given r7 anchors, interpolated. (3) The mask is synthetic Prism structure, not real bench records.

## Verdict

- **No new attn_backend.** SDPA with the Prism mask already runs the fused mem-efficient kernel on every layer (sliding
  256 and global 512) in fp16. xformers is the same kernel family and no faster. Forced EFFICIENT changes nothing.
- The only change that would make the legacy option work is **keeping fp16** (never bf16 on the T4). That is already the
  default since OJ-081.
- `chunked_math` is the fastest kernel in isolation (head_dim 512 especially), but the end-to-end ceiling is ~2.5%, and
  the r4 in-model trial showed no gain. Optional: one paired r7-style trial `chunked_math` vs `sdpa_bool`. Otherwise, close
  this direction. The remaining big costs are elsewhere: mm 28%, mul 15%, copy_ 12.5%, launch overhead ("Command Buffer
  Full") in the trial-23 profile.
