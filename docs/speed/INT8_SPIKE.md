# int8 matmul spike: int8 GEMMs for the frozen base linears on T4

Kernel **piyushsar/openjev-int8-spike**, Tesla T4 (sm 7.5), torch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0, Python 3.13.16.
- v1: Part 1, the GEMM micro-bench.
- v2: Part 2, in the real model, plus a torchao `module_swap` addendum.

Scripts: `kaggle/int8_spike/bench_int8.py` (Part 1), `kaggle/int8_spike/int8_model.py` (Part 2), and `build_part2.sh` / `part2_entry.py` (the v2 kernel). Raw output (gitignored) is in `runs/optuna/int8_spike/out/` (v1) and `out_v2/` (v2). Date: 2026-11-23.

## TL;DR

- **int8 matmul training runs on the T4.** `torch._int_mm` (cuBLASLt IMMA) works on sm75 in both directions: forward Y = X Wᵀ and backward dX = dY W. Pass mat2 as a column-major `.t()` view; a contiguous mat2 is 3.2× slower.
- **The int8 GEMMs alone are 1.6–1.8× faster than fp16** at M ≥ 2048, count-weighted over the 205 LoRA-target linears with 2 fwd + 1 dX. At M = 1024 they are 1.17×. That is far below the 2× peak ratio.
- **Eager quantize/dequant wipes out the gain** (0.3–0.7×). With `torch.compile` fusing those steps, the full op is **1.26–1.52×** (≥ 2048 tokens).
  - The win is in the MLP: gate/up 1.3–1.9×, down 1.2–1.4×.
  - q/o are ≈ 1× and k/v are 0.5–0.6×.
- **Library paths do not help on sm75:**
  - torchao 0.18's int8 mixed-precision training silently runs fp16 (tensor subclass), and its `module_swap` path fails in Triton.
  - torchao's int8 dynamic-activation config has no backward pass.
  - bitsandbytes `Linear8bitLt` (our r6 `quant=int8`) is 0.8–1.1×, because its backward is fp16.
- **In the real model (Part 2): no speed gain is detectable within ±5% trial noise.** The run used E2B, the ≤512 bench bucket, mb 8, fp16, grad ckpt, unchanged LoRA, and int8 on the 105 MLP base linears.
  - The micro-bench predicted **+8%**. Measured: the int8 arms ran 1.717–1.731 samples/s against fp16 at 1.795 and 1.645, so int8 is +0.4% on the mean.
  - Why the GEMM win does not carry over was not isolated. See Part 2.
- **The accuracy cost is real.** On the real activations, each int8 MLP GEMM has 2.2–3.7% median relative error on Y (8.4% max), 2–3× the Gaussian figure.
  - down_proj inputs have outlier channels (channel absmax ÷ median ≈ 50, up to 322).
  - Probe loss on the same weights: 2.2301 → 2.2440 (+0.6%). First-step train loss: 2.067 → 2.114 (+2.3%).
- **Verdict: reject for now.** Do not add a `matmul_precision` Optuna dim. The conditions to reopen are at the end.

## Why int8, and what training needs

The T4 has int8 tensor cores: ~130 TOPS peak, against ~65 TFLOPS for fp16. The T4 has no fp8 and no fp4 (DECISIONS 2026-11-24). With LoRA the base weights are frozen, so each LoRA-target decoder linear needs:
- Y = X Wᵀ in the forward pass. It runs **twice** because grad ckpt recomputes it.
- dX = dY W in the backward pass. There is no dW.

So the cost per step is 2 fwd + 1 dX. LoRA (rank 16, fp16) is not part of this.

r6 `quant=int8` is bitsandbytes LLM.int8() with threshold 6.0. It stores W in int8 and gave no speed. Below, `bnb_th6` shows why: its dX dequantizes W and runs fp16, and its forward pays for outlier extraction.

### Shapes (checked, not derived)

`openjev.prism.quant_targets` on the HF config of `google/gemma-4-E2B-it` (meta device, transformers 5.17) gives **205** linears. The kv-shared layers 15–34 have no k/v projections and a double-wide MLP (12288).

| key | N×K (out×in) | count |
|---|---|---|
| q_s | 2048×1536 | 28 |
| q_g | 4096×1536 | 7 |
| kv_s | 256×1536 | 24 |
| kv_g | 512×1536 | 6 |
| o_s | 1536×2048 | 28 |
| o_g | 1536×4096 | 7 |
| gu_6k (gate+up) | 6144×1536 | 30 |
| gu_12k (gate+up) | 12288×1536 | 40 |
| down_6k | 1536×6144 | 15 |
| down_12k | 1536×12288 | 20 |

The MLP (gate/up/down, 105 linears) holds 85% of the FLOPs (count × N × K).

## Part 1: GEMM micro-bench (no model)

**Setup**
- Random fp16 W (std 0.02), X and dY at M ∈ {1024, 2048, 4096, 8192, 16384}. M is the number of tokens per micro-batch: mb 8 × bucket bound 128…2048.
- Each candidate's forward and dX are timed separately (CUDA events, 3 warm-up runs, median of 10).
- For each shape and M, the order is fp16 → int8 variants → fp16 again (`fp16_repeat`).
- **Baseline `fp16min`** = per cell, the faster of fp16 and fp16_repeat, taken separately for fwd and dX.
  - The first fp16 forward of a cell is often slow (gu_6k 3.38 vs 2.43 ms; the T4 was at 1560–1590 MHz and 75 °C under load).
  - Using the first fp16 alone would flatter int8 by ~5–10%.
  - torchao and bnb ran in subprocesses, each against its own single fp16 run (`fp16_ag`, through autograd).
- Errors are measured at M=4096 against an fp32 reference, on Gaussian X and on "outlier" X (0.5% of the input channels ×30, as in Gemma-like outlier features).

**Candidates**
- `fp16`: `F.linear` and dY @ W, the current path.
- `i8_row`: `torch._int_mm` (int8×int8→int32) with per-row dynamic scales for X and dY.
  - W is quantized once per role: rows for Y and columns for dX, so each scale sits outside its GEMM's reduction dim. The two int8 copies take the bytes of one fp16 copy.
  - Quantize and dequant run in eager.
- `i8_tensor`: the same with per-tensor scales.
- `i8_row_compiled`: the same function under `torch.compile`, which fuses amax, quantize and dequant into Triton kernels; the GEMM stays the cuBLAS `_int_mm`.
- `i8_row_fused`: the same, plus inductor `force_fuse_int_mm_with_mul`, which is meant to fuse the scale into the int8 GEMM. It ran without error and is faster on gate/up, but the code it emitted was not inspected. torchao's own Triton int8 `tl.dot` fails on this stack (Part 2).
- `i8_gemm_only`: `_int_mm` alone on pre-quantized inputs. This is the ceiling.
- torchao 0.18.0, installed with `--no-deps` (torch stayed 2.11.0; the preinstalled 0.10.0 also imported):
  - `Int8MixedPrecisionTrainingConfig` as a tensor subclass, and with `module_swap=True` (v2)
  - `Int8DynamicActivationInt8WeightConfig`
- bitsandbytes 0.50.2 `Linear8bitLt`, with threshold 6.0 (our `quant=int8`) and threshold 0.

**Probe**
- `_int_mm` runs on sm75.
- mat2 as a column-major `.t()` view is **3.2× faster** than a contiguous row-major tensor (2.52 vs 8.19 ms fwd at 4096×6144×1536). That layout is used for every row below.
- The custom autograd Function matches the direct calls bit for bit.

### Count-weighted speedup over the 205 linears (vs fp16min)

| candidate | M | fwd | dX | step | step ms (fp16 → cand) |
|---|---|---|---|---|---|
| i8_row | 1024 | 0.29x | 0.28x | 0.28x | 321 → 1129 |
| i8_row | 2048 | 0.41x | 0.54x | 0.45x | 919 → 2038 |
| i8_row | 4096 | 0.51x | 0.60x | 0.54x | 2082 → 3885 |
| i8_row | 8192 | 0.59x | 0.66x | 0.61x | 4588 → 7496 |
| i8_row | 16384 | 0.65x | 0.73x | 0.68x | 10062 → 14835 |
| i8_row_compiled | 1024 | 0.76x | 0.81x | 0.78x | 321 → 413 |
| i8_row_compiled | 2048 | 1.17x | 1.42x | 1.26x | 919 → 730 |
| i8_row_compiled | 4096 | 1.30x | 1.41x | 1.33x | 2082 → 1561 |
| i8_row_compiled | 8192 | 1.32x | 1.48x | 1.38x | 4588 → 3333 |
| i8_row_compiled | 16384 | 1.44x | 1.67x | 1.52x | 10062 → 6628 |
| i8_row_fused | 1024 | 0.78x | 0.81x | 0.79x | 321 → 406 |
| i8_row_fused | 2048 | 1.18x | 1.57x | 1.30x | 919 → 706 |
| i8_row_fused | 4096 | 1.38x | 1.73x | 1.49x | 2082 → 1398 |
| i8_row_fused | 8192 | 1.34x | 1.59x | 1.42x | 4588 → 3223 |
| i8_row_fused | 16384 | 1.40x | 1.67x | 1.49x | 10062 → 6767 |
| i8_gemm_only | 1024 | 1.18x | 1.17x | 1.17x | 321 → 274 |
| i8_gemm_only | 2048 | 1.63x | 1.65x | 1.64x | 919 → 561 |
| i8_gemm_only | 4096 | 1.52x | 1.70x | 1.58x | 2082 → 1314 |
| i8_gemm_only | 8192 | 1.62x | 1.83x | 1.69x | 4588 → 2712 |
| i8_gemm_only | 16384 | 1.74x | 2.01x | 1.83x | 10062 → 5510 |
| torchao_i8_mpt (int8 NOT engaged, see below) | 1024 | 0.61x | 1.06x | 0.75x | 521 → 699 |
| torchao_i8_mpt (int8 NOT engaged, see below) | 2048 | 0.65x | 1.08x | 0.79x | 1016 → 1292 |
| torchao_i8_mpt (int8 NOT engaged, see below) | 4096 | 0.82x | 0.92x | 0.86x | 2145 → 2499 |
| torchao_i8_mpt (int8 NOT engaged, see below) | 8192 | 0.86x | 0.95x | 0.89x | 4506 → 5064 |
| torchao_i8_mpt (int8 NOT engaged, see below) | 16384 | 0.90x | 0.91x | 0.90x | 9852 → 10916 |
| bnb_th6 | 1024 | 0.75x | 0.97x | 0.83x | 521 → 626 |
| bnb_th6 | 2048 | 0.64x | 1.08x | 0.78x | 1016 → 1307 |
| bnb_th6 | 4096 | 0.83x | 1.06x | 0.90x | 2145 → 2374 |
| bnb_th6 | 8192 | 0.93x | 1.07x | 0.98x | 4506 → 4617 |
| bnb_th6 | 16384 | 1.05x | 1.00x | 1.03x | 9852 → 9548 |
| bnb_th0 | 1024 | 1.11x | 0.90x | 1.01x | 521 → 518 |
| bnb_th0 | 2048 | 0.86x | 0.98x | 0.91x | 1016 → 1120 |
| bnb_th0 | 4096 | 1.02x | 0.96x | 1.00x | 2145 → 2146 |
| bnb_th0 | 8192 | 1.09x | 1.00x | 1.05x | 4506 → 4286 |
| bnb_th0 | 16384 | 1.17x | 0.99x | 1.10x | 9852 → 8967 |
| sel_mlp_compiled | 1024 | 1.00x | 1.00x | 1.00x | 321 → 321 |
| sel_mlp_compiled | 2048 | 1.22x | 1.50x | 1.31x | 919 → 700 |
| sel_mlp_compiled | 4096 | 1.31x | 1.41x | 1.35x | 2082 → 1545 |
| sel_mlp_compiled | 8192 | 1.31x | 1.43x | 1.35x | 4588 → 3396 |
| sel_mlp_compiled | 16384 | 1.41x | 1.60x | 1.48x | 10062 → 6821 |
| sel_mlp_fused | 1024 | 1.00x | 1.00x | 1.00x | 321 → 321 |
| sel_mlp_fused | 2048 | 1.22x | 1.66x | 1.36x | 919 → 675 |
| sel_mlp_fused | 4096 | 1.40x | 1.74x | 1.51x | 2082 → 1379 |
| sel_mlp_fused | 8192 | 1.35x | 1.52x | 1.41x | 4588 → 3262 |
| sel_mlp_fused | 16384 | 1.39x | 1.58x | 1.45x | 10062 → 6931 |

fp16 drift, count-weighted step ms of fp16_repeat ÷ fp16 per M: {1024: 1.261, 2048: 0.998, 4096: 0.966, 8192: 0.982, 16384: 0.985} The first fp16 pass of each shape is slow at M=1024.

`sel_mlp_*` puts int8 on gate/up/down only, and only at M ≥ 2048. q/k/v/o and every linear at M < 2048 stay fp16. This is built from the same rows, and it is what Part 2 runs.

### Per shape at M = 4096 (fwd / dX ms; speedup on 2 fwd + 1 dX vs fp16min)

| shape (N×K, count) | fp16 | fp16min | i8_row | i8_tensor | i8_row_compiled | i8_row_fused | i8_gemm_only | fp16_repeat | torchao_i8_mpt | torchao_i8_dyn | bnb_th6 | bnb_th0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q_s (2048×1536, 28) | 1.15 / 1.14 | 0.79 / 0.72 | 2.97 / 3.01 (0.26x) | 2.70 / 2.66 (0.29x) | 0.83 / 0.75 (0.96x) | 0.80 / 0.75 (0.98x) | 0.48 / 0.42 (1.67x) | 0.79 / 0.72 (1.00x) | 1.45 / 1.48 (0.68x) | fwd_only | 2.06 / 0.81 (0.61x) | 1.34 / 1.07 (0.80x) |
| q_g (4096×1536, 7) | 1.87 / 2.54 | 1.53 / 2.42 | 4.68 / 4.46 (0.40x) | 3.95 / 3.69 (0.47x) | 1.49 / 1.30 (1.28x) | 1.41 / 1.11 (1.40x) | 0.86 / 0.90 (2.10x) | 1.53 / 2.42 (1.00x) | 2.96 / 2.42 (0.87x) | fwd_only | 3.24 / 1.68 (0.89x) | 2.36 / 2.58 (0.99x) |
| kv_s (256×1536, 24) | 0.22 / 0.23 | 0.22 / 0.23 | 1.45 / 1.27 (0.16x) | 1.19 / 1.22 (0.19x) | 0.41 / 0.47 (0.53x) | 0.41 / 0.49 (0.52x) | 0.16 / 0.22 (1.27x) | 0.23 / 0.25 (0.94x) | 0.28 / 0.39 (0.44x) | fwd_only | 0.86 / 0.33 (0.20x) | 0.48 / 0.51 (0.29x) |
| kv_g (512×1536, 6) | 0.41 / 0.38 | 0.28 / 0.27 | 1.73 / 1.53 (0.17x) | 1.46 / 1.42 (0.19x) | 0.44 / 0.50 (0.61x) | 0.41 / 0.49 (0.64x) | 0.19 / 0.24 (1.34x) | 0.28 / 0.27 (1.00x) | 0.38 / 0.34 (0.60x) | fwd_only | 1.02 / 0.36 (0.27x) | 0.61 / 0.27 (0.44x) |
| o_s (1536×2048, 28) | 0.89 / 0.88 | 0.82 / 0.88 | 3.02 / 2.79 (0.29x) | 2.40 / 2.43 (0.35x) | 0.87 / 1.01 (0.92x) | 0.92 / 1.05 (0.88x) | 0.62 / 0.69 (1.32x) | 0.82 / 1.06 (0.93x) | 1.47 / 1.26 (0.70x) | fwd_only | 1.93 / 1.00 (0.61x) | 1.24 / 1.13 (0.81x) |
| o_g (1536×4096, 7) | 1.96 / 2.52 | 1.55 / 2.30 | 5.23 / 4.46 (0.36x) | 3.89 / 4.06 (0.46x) | 1.44 / 2.08 (1.09x) | 1.78 / 2.08 (0.96x) | 1.31 / 1.55 (1.30x) | 1.55 / 2.30 (1.00x) | 2.96 / 2.36 (0.85x) | fwd_only | 3.08 / 2.17 (0.85x) | 2.03 / 2.85 (1.02x) |
| gu_6k (6144×1536, 30) | 3.38 / 4.01 | 2.43 / 4.01 | 6.71 / 6.71 (0.44x) | 5.83 / 5.53 (0.52x) | 2.32 / 2.19 (1.30x) | 2.13 / 1.62 (1.51x) | 1.70 / 2.37 (1.54x) | 2.43 / 4.16 (0.98x) | 4.19 / 3.86 (0.89x) | fwd_only | 4.08 / 3.71 (0.91x) | 3.41 / 4.09 (0.99x) |
| gu_12k (12288×1536, 40) | 7.98 / 9.06 | 7.98 / 8.53 | 12.01 / 12.28 (0.67x) | 11.20 / 10.56 (0.74x) | 5.51 / 5.55 (1.48x) | 4.70 / 3.40 (1.91x) | 5.30 / 4.69 (1.60x) | 8.58 / 8.53 (0.95x) | 8.37 / 9.59 (0.87x) | fwd_only | 7.20 / 8.31 (1.01x) | 6.83 / 8.75 (1.02x) |
| down_6k (1536×6144, 15) | 3.50 / 3.98 | 2.76 / 3.98 | 6.78 / 6.01 (0.49x) | 5.44 / 5.71 (0.57x) | 2.31 / 3.07 (1.24x) | 2.56 / 3.10 (1.16x) | 2.05 / 2.42 (1.46x) | 2.76 / 4.26 (0.97x) | 4.22 / 4.28 (0.86x) | fwd_only | 4.37 / 3.73 (0.88x) | 3.14 / 4.09 (1.06x) |
| down_12k (1536×12288, 20) | 7.67 / 7.76 | 7.67 / 7.67 | 12.08 / 11.15 (0.65x) | 10.03 / 10.78 (0.75x) | 5.24 / 5.97 (1.40x) | 5.35 / 5.99 (1.38x) | 4.46 / 4.81 (1.67x) | 8.05 / 7.67 (0.97x) | 8.18 / 8.75 (0.91x) | fwd_only | 7.62 / 8.26 (0.98x) | 6.01 / 8.80 (1.10x) |

### Where the time goes (count-weighted ms, one fwd + one dX)

| M | fp16 fwd+dX | quantize X+dY | int8 GEMMs | dequant Y+dX | i8_row total | compiled total |
|---|---|---|---|---|---|---|
| 1024 | 213.8 | 273.3 | 182.2 | 203.6 | 755.1 | 272.6 |
| 2048 | 661.2 | 541.6 | 388.1 | 393.9 | 1350.2 | 489.4 |
| 4096 | 1516.1 | 1022.3 | 880.2 | 774.6 | 2577.7 | 1050.8 |
| 8192 | 3198.3 | 1959.8 | 1803.7 | 1532.0 | 4983.9 | 2219.9 |
| 16384 | 7009.3 | 3808.7 | 3665.5 | 3050.2 | 9909.2 | 4404.8 |

In eager mode, quantizing X/dY and dequantizing the int32 output take more time than the int8 GEMMs themselves. At N=12288 and M=8192 the int32 output alone is ~400 MB of traffic. torch.compile removes most of that.

### Errors vs fp32 (M = 4096, worst over the 10 shapes; rel. Frobenius / max-abs ÷ max|ref|)

| candidate | Y gauss | dX gauss | Y outlier | dX outlier |
|---|---|---|---|---|
| fp16 | 2.5e-04 / 4.5e-04 | 3.6e-04 / 8.2e-04 | 2.6e-04 / 5.5e-04 | 3.6e-04 / 9.0e-04 |
| i8_row | 1.3e-02 / 1.5e-02 | 1.3e-02 / 1.4e-02 | 7.7e-02 / 8.8e-02 | 1.3e-02 / 1.4e-02 |
| i8_tensor | 1.8e-02 / 1.8e-02 | 1.8e-02 / 1.9e-02 | 1.6e-01 / 1.1e-01 | 1.8e-02 / 1.9e-02 |
| i8_row_compiled | 1.3e-02 / 1.5e-02 | 1.3e-02 / 1.4e-02 | 7.7e-02 / 8.8e-02 | 1.3e-02 / 1.4e-02 |
| i8_row_fused | 1.3e-02 / 1.5e-02 | 1.3e-02 / 1.4e-02 | 7.7e-02 / 8.8e-02 | 1.3e-02 / 1.4e-02 |
| fp16_ag | 2.5e-04 / 4.5e-04 | 3.6e-04 / 8.2e-04 | 2.6e-04 / 5.5e-04 | 3.6e-04 / 9.0e-04 |
| torchao_i8_mpt (= fp16: not engaged) | 2.5e-04 / 4.5e-04 | 3.6e-04 / 8.2e-04 | 2.6e-04 / 5.5e-04 | 3.6e-04 / 9.0e-04 |
| bnb_th6 | 1.3e-02 / 1.5e-02 | 9.3e-03 / 9.6e-03 | 1.1e-02 / 9.7e-03 | 9.3e-03 / 9.2e-03 |
| bnb_th0 | 1.3e-02 / 1.5e-02 | 9.3e-03 / 9.6e-03 | 7.7e-02 / 8.8e-02 | 9.3e-03 / 9.2e-03 |

The int8 path is ~50× noisier than fp16 per GEMM on Gaussian inputs (1.3% vs 0.03%). With outlier channels, per-row int8 on Y gets **5.5–7.7%** (down_proj is worst). On the real E2B activations it is 2.2–3.7% median (Part 2). bnb's threshold 6.0 keeps outliers in fp16 and stays at 1.1%, but costs speed. dX is not affected, because outliers are a property of the activation columns.

### Unsupported or not engaged

| case | result |
|---|---|
| torchao 0.18 `Int8MixedPrecisionTrainingConfig` (tensor subclass, default) | **int8 not engaged**. Its errors are bit-identical to fp16, and a CPU repro shows `MmBackward0`: the subclass's `F.linear` override is not hit on torch 2.11 / 2.14, so it runs plain fp16 mm plus wrapper overhead (0.75–0.90x) |
| torchao `Int8DynamicActivationInt8WeightConfig` | forward only: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` (no dX, so not usable for training). Its forward is also 4× slower than fp16 on sm75 (gu_12k at M=4096: 30.2 ms vs 8.0) |
| bnb `Linear8bitLt` | runs. The forward is int8 (cuBLASLt); the backward dequantizes W and runs fp16. 0.78–1.03x (threshold 6.0), 0.91–1.10x (threshold 0) |
| int8 on q/k/v/o | runs, but ≤ 1x at M=4096 except q_g and o_g (7 layers each). k/v (N = 256/512) is 0.5–0.6x |

## Part 2: in the real model (kernel v2)

The gate passed: `sel_mlp_compiled` is 1.35× and `sel_mlp_fused` is 1.51× at M = 4096, against the ≥ 1.2× gate.

**Correctness first: tiny Gemma 4 on CPU** (`int8_model.py --cpu-check`). The setup is 3 layers, PEFT LoRA r 8 on all 7 targets, and grad ckpt on.
- The 9 MLP base linears changed class in place. Each is an `nn.Linear` subclass under a plain `peft.tuners.lora.Linear`, with `is_loaded_in_8bit` unset.
- The int8 copies are non-persistent buffers and stay out of the `state_dict`. The base weights get no grad, and the LoRA grads are finite.
- The int8 kernels equal a float fake-quant reference of the same rounding **exactly**: LoRA grads rel diff 0.0, loss equal.
- `torch.compile(dynamic=True)` equals eager exactly.
- Against the unquantized path, the loss differs by 0.17%. The LoRA grads differ by 97%, but this tiny random model is chaotic: 1% Gaussian noise on each MLP Y/dX moves its LoRA grads by 50–75% too. So the tiny model cannot judge accuracy.

**Real E2B on 1×T4.**
- One model load (257 s), then all 105 gate/up/down base linears converted in place. Memory went from 4.34 to 7.24 GB allocated (+2.9 GB of int8 copies; the fp16 weights are kept for the fp16 arm).
- `Int8ToggleLinear.mode` switches fp16 ↔ int8 on the same weights.
- Data: the ≤512 bucket of the fixed bench set (sha 48758bf206de5268, 32 records), global batch 32 = 4 × mb 8 (≈3856 padded tokens per micro-batch), fp16 + GradScaler, grad ckpt on, `ple_cpu=1`, sdpa default, LoRA r 16 unchanged.
- It used bench_trials' own `probe_loss`, `accum_step`, `snapshot/reset` and collate.
- Per arm: reset to the start weights, 1 warm-up step, then 3 timed steps (96 samples). Order: fp16, compiled, fused × 2 reps.
- Harness checks: the fp16 `loss_start` equals r7's 2.23011 exactly, and fp16 rep 1 (1.645 samples/s) is close to r7's ≤512 bucket (1.668).

| arm | rep | samples/s | tokens/s | peak GB | dynamo graphs built in timed loop |
|---|---|---|---|---|---|
| fp16 | 0 | 1.795 | 650 | 9.13 | 0 |
| int8 compiled | 0 | 1.729 | 626 | 8.95 | 0 |
| int8 fused | 0 | 1.730 | 626 | 8.95 | 0 |
| fp16 | 1 | 1.645 | 596 | 9.13 | 0 |
| int8 compiled | 1 | 1.731 | 627 | 8.95 | 0 |
| int8 fused | 1 | 1.717 | 622 | 9.13 | 0 |

- **No gain detectable within ±5% trial noise.** All int8 arms are at 1.717–1.731. fp16 is at 1.795 and 1.645, mean 1.720, so int8 is +0.4% on the mean.
  - The test is underpowered and confounded by order: fp16 ran first in each rep, and rep 0 had the coolest GPU.
  - Against fp16 rep 0, int8 is −4%. Against rep 1, it is +5%.
  - The estimate below predicts +7.8% for this bucket. That did **not** reproduce. The result means "no detectable gain", not "int8 is slower".
- The compiled int8 path did run: eager int8 is 0.54× on the MLP GEMMs and would have made the int8 arms ~30% slower, and 0 dynamo graphs were built inside the timed loop.
- "compiled" and "fused" are effectively **one** int8 arm in the model. They have identical losses, and the fused probe took 2.7 s against 8.5 s for compiled (compile included). That points to a cache hit or identical generated code, not a separate fused kernel. Part 1's `i8_row_fused` differs in speed from `i8_row_compiled`, but the code inductor emitted for it on sm75 was not inspected.
- **An inconsistency, not resolved.** By the Part 1 fp16min times, the 105 MLP linears at M ≈ 3856 cost ≈ 1.85 s of GEMMs per micro-batch (2 fwd + 1 dX). The measured step is ≈ 4.6 s per micro-batch (55 s ÷ 3 steps ÷ 4). That would make the MLP GEMMs ~40% of the step, which contradicts the 28% `aten::mm` anchor.
  - So micro-bench times do not carry over into the model as they are.
  - Possible causes, none tested: sustained-load clocks under the T4's 70 W cap, the dynamic-shape compiled kernels (for example, inductor making the column-major int8 buffer contiguous before `_int_mm`, the 3.2× slower layout), per-call dynamo guard and autograd.Function overhead, or a step that is partly host/PLE-bound.

**Numerics on the same (start) weights.**
- Probe loss (8 records): fp16 **2.2301** → int8 **2.2440** (+0.62%).
- First micro-step train loss on the same batch: 2.067 → 2.114 (+2.3%).
- (`loss_after` is not reported. Each arm's fresh GradScaler at 2¹⁶ skipped 2–3 of its 4 steps, so the arms applied different numbers of updates.)

**int8 vs fp16 Y error on the real activations.** These come from hooks on the probe forward: 30 or 40 calls per type, relative Frobenius / max-abs ÷ max|ref|.

| linear | rel_fro median (max) | max_rel median (max) | input channel absmax ÷ median, median (max) |
|---|---|---|---|
| gate 6144×1536 | 2.6% (6.1%) | 2.5% (4.2%) | 11 (18) |
| up 6144×1536 | 3.4% (7.5%) | 2.0% (3.5%) | 11 (18) |
| down 1536×6144 | 3.3% (6.3%) | 5.2% (19.1%) | 48 (322) |
| gate 12288×1536 | 2.2% (5.7%) | 2.0% (9.3%) | 9 (27) |
| up 12288×1536 | 3.1% (8.4%) | 1.8% (7.9%) | 9 (27) |
| down 1536×12288 | 3.7% (6.6%) | 1.7% (6.2%) | 53 (111) |

That is 2–3× the Gaussian error from Part 1 (1.3%). down_proj inputs are strongly heavy-tailed, the case per-row int8 handles worst.

**torchao 0.18 `module_swap=True` (v2 addendum).** It fails in torchao's Triton `scaled_int8_mm` on sm75 / Triton 3.6:
`int8_mm.py:114:25: error: 'arith.extf' op operand #0 must be floating-point-like, but got 'tensor<128x64xi8, #ttg.dot_op<...>>'` → `RuntimeError: PassManager::run failed` (every shape, forward included). `Int8DynamicActivationInt8WeightConfig` is again forward-only (5.7 ms vs fp16 0.79 ms at q_s, M = 4096).

## End-to-end estimate

- Amdahl per bucket: new step time = 1 − 0.28·(1 − 1/g_b).
  - 0.28 is the `aten::mm` share of CUDA time in the r4 trial-23 profile.
  - g_b is the count-weighted 2 fwd + 1 dX speedup at M_b = 8 × bucket bound.
- Buckets are weighted by w_b × r7 s/sample, as in the attention spike: w = 2.2/16.3/55.2/19.7/6.6% and s/sample = 0.150/0.297/0.599/1.18/1.82.
- This is an estimate from the 28% anchor, which also includes the LoRA and head matmuls. At micro-bench times the MLP alone would be ~40% of the step (Part 2), so the estimate is not a bound in either direction. Neither figure reproduced in the model.
- M = 8 × bound overstates M (records are 0.5–1× the bound), and larger M favours int8.

| candidate | <=128 | <=256 | <=512 | <=1024 | <=2048 | weighted samples/s |
|---|---|---|---|---|---|---|
| i8_row | 0.28x | 0.45x | 0.54x | 0.61x | 0.68x | 0.825x |
| i8_tensor | 0.35x | 0.53x | 0.61x | 0.69x | 0.75x | 0.867x |
| i8_row_compiled | 0.78x | 1.26x | 1.33x | 1.38x | 1.52x | 1.081x |
| i8_row_fused | 0.79x | 1.30x | 1.49x | 1.42x | 1.49x | 1.095x |
| i8_gemm_only | 1.17x | 1.64x | 1.58x | 1.69x | 1.83x | 1.125x |
| torchao_i8_mpt (not int8) | 0.75x | 0.79x | 0.86x | 0.89x | 0.90x | 0.960x |
| bnb_th6 | 0.83x | 0.78x | 0.90x | 0.98x | 1.03x | 0.981x |
| bnb_th0 | 1.01x | 0.91x | 1.00x | 1.05x | 1.10x | 1.006x |
| sel_mlp_compiled | 1.00x | 1.31x | 1.35x | 1.35x | 1.48x | 1.081x |
| sel_mlp_fused | 1.00x | 1.36x | 1.51x | 1.41x | 1.45x | 1.096x |

**Estimate vs measurement.** For the ≤512 bucket (55% of the weight), the estimate is +7.8% (`sel_mlp_compiled`, 1.35× at M = 4096). The measured in-model result is +0.4% ± ~5%, so the end-to-end ceiling of ~+8–10% weighted samples/s was not seen in the model.

## Calibration risk

ECE is the main metric. Per-row int8 adds ~1.3% relative noise to every MLP GEMM (Gaussian), and 5.5–7.7% on Y when there are outlier channels. Over 35 layers × 3 GEMMs × 2 passes, that noise reaches both the hidden states the flag/confidence heads read and the LoRA gradients.

Measured on the real model:
- Per-GEMM Y error is 2.2–3.7% median (8.4% max).
- down_proj input channels have absmax ÷ median ≈ 50 (up to 322).
- Probe loss rises 0.6% and the first-step train loss 2.3% on the same weights.

The tiny Gemma cannot judge this. Its LoRA grads move by 97% under int8, but also by 50–75% under 1% Gaussian noise. It only proves that the implementation is exact.

A probe loss within noise is a weak proxy for calibration. **Any adoption needs the 150-step quality check** (val ECE/Brier/accuracy vs fp16, `optuna_validate`) before it becomes a default.

## Verdict

**Reject int8 matmul training on the T4 for now. Do not add a `matmul_precision` Optuna dim.**
- The GEMMs are faster in isolation: 1.5–1.8× raw, and 1.3–1.5× with fused quantization on the MLP.
- But the paired in-model run shows no detectable samples/s gain.
- The numerical cost on real activations (2–4% per MLP GEMM, outlier-heavy down_proj) is the wrong trade for a model judged on calibration (ECE).
- bnb `quant=int8` stays a memory option only (r6). Keep fp16.

**Reopen only if all of these hold:**
1. **One run that explains the gap.** Alternate the order (fp16/int8/int8/fp16 × 3), carry the scaler state across arms, and profile one step per mode (`bench_trials.run_profile`).
   - First check whether inductor under `dynamic=True` copies the column-major int8 buffer to contiguous before `_int_mm`, or whether static shapes per bucket are faster.
   - Required outcome: a paired ≥ 5% weighted samples/s gain.
2. **An outlier-aware variant that brings the per-GEMM error near 1% on real activations.** Examples: down_proj kept in fp16 (gate/up carry most of the speed), or SmoothQuant-style per-channel smoothing folded into the preceding norm.
3. **Then the 150-step quality check** (`optuna_validate`: val ECE / Brier / accuracy vs fp16 within ±2 SE) before any default change. A probe loss within noise is not enough evidence for calibration.

What could be reused if it is reopened: `kaggle/int8_spike/int8_model.py` (`convert()` / `Int8ToggleLinear`). It swaps the class in place after PEFT, needs no `train_prism.py` change, and is checked exact against a fake-quant reference.
