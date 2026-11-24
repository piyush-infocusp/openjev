# Unsloth spike on a Kaggle T4 (2026-11-24)

Question: does Unsloth make the Prism training step faster on 1xT4 (fp16, sm75), and can it keep the Prism model exactly?
Code: `kaggle/unsloth_spike/` (commit d5fad21). Kernel: `piyushsar/openjev-unsloth-spike`. Raw: `runs/optuna/unsloth_spike/out/` (`unsloth_base.json`, `unsloth_16bit.json`, `unsloth_4bit.json`, `unsloth_env.json`, log).

## Environment

| item | baseline stage | Unsloth stages |
|---|---|---|
| torch / triton / cuda | 2.11.0+cu130 / 3.6.0 / 13.0 | unchanged (`torch_unchanged: true`) |
| transformers | 5.17.0 (pinned) | 5.17.0 (pinned; newest unsloth 2026.11.9 accepts) |
| peft | 0.21.2 | 0.21.2 |
| unsloth / unsloth_zoo | none | 2026.11.9 / 2026.11.9 |
| trl, bitsandbytes, torchao, cut_cross_entropy | none | 1.15.0, 0.50.2, 0.18.0, 25.1.1 |

Installed with `--no-deps` under a constraints file (numpy 2.1.3, torch/torchaudio/torchvision +cu130, transformers 5.17.0, triton 3.6.0); `deps_rc=0`, `unsloth_rc=0`. Kaggle's default transformers is 5.20.2, so Unsloth forces a **downgrade to 5.17.0**; the baseline here was also measured on 5.17.0 for a like-for-like comparison, but it is not the production pin (5.20.2).

## Support

| item | finding |
|---|---|
| Gemma 4 E2B | Loads as `Gemma4ForCausalLM` via generic `FastModel` only (`text_only=True`); no Gemma 4 file in `unsloth/models/`. The hand-written Triton kernels (GEGLU, RMSNorm, fused LoRA MLP) target llama-family and are not used. |
| compile | Per-layer `torch.compile` regions (`_block_compile.py`), patched PEFT LoRA forward. Warm-up 51.6 s (16-bit causal), 19.7 s (prism), vs 11-18 s baseline. |
| T4 precision | `gemma4` is on `FORCE_FLOAT32` (`UNSLOTH_FORCE_FLOAT32=1`): fp16 weights, float32 residual stream / RMSNorm / q,k,v+RoPE (`norm_dtype float32`); SDPA on fp16 operands (`UNSLOTH_GEMMA4_FP16_SDPA` default on). |
| Prism 4D masks | Pass to SDPA (KV repeated so the mem-efficient kernel takes the mask). Unsloth's banded sliding path never engaged: `causal_engaged = 0` before, after causal and after prism. |
| PLE | Stays on GPU at load (`ple_device cuda:0`); we move it to CPU after PEFT (`ple_device cpu`, `head_restored true`). |
| LoRA | r16/alpha32, 205 modules, 24,170,497 trainable (same as baseline), vocab 262151 padded to 262272 rows. |
| fused CE | Not part of our step (flag-head loss, no LM-head CE). |

## Timings (fwd+bwd, GradScaler, no optimizer step; 4 micro-batches of 8, first 32 of the <=512 bucket, 3 timed passes, medians)

| variant | causal_nopad ms/step | causal ms/mb | causal peak GB | **Prism ms/step** | Prism ms/mb | Prism peak GB | Prism speedup (baseline / this) |
|---|---|---|---|---|---|---|---|
| baseline (HF, fp16, ple_cpu) | 11063.7 | 2728.4 | 5.499 | 18740.9 | 4736.9 | 6.044 | 1.00x |
| Unsloth 16-bit | 9211.7 | 2274.5 | 5.558 | **15282.7** | 3841.3 | 6.213 | **1.226x** (-18.5% time) |
| Unsloth 4-bit | 10199.8 | 2544.9 | 3.609 | 15576.3 | 3932.8 | 4.264 | 1.203x |

Load: baseline 4.34 GB resident (257.7 s incl. download), Unsloth 16-bit 8.63 GB at load then 4.345 GB after PEFT/PLE offload, 4-bit 6.68 then 2.395 GB. Plain causal speedup (no padding, no mask) is 1.20x (16-bit) / 1.08x (4-bit), so the Prism gain is not from the fast causal path.

### Attribution (Unsloth 16-bit, Prism)

| run | ms/step | vs Unsloth Prism | note |
|---|---|---|---|
| default (autocast, unsloth ckpt) | 15282.7 | 1.000 | |
| no autocast | 15929.7 | +4.2% | 4-bit: 16145.1 (+3.7%) |
| HF grad ckpt instead of unsloth ckpt | 14873.0 | -2.7% | peak 6.213 GB, same as unsloth ckpt, i.e. no offload benefit; unsloth ckpt costs ~3% |

So the gain comes from the compiled per-layer blocks / patched LoRA, not from autocast or checkpointing. Unsloth checkpointing is slightly slower and saves no memory here.

## Parity: flag logits at LoRA init (B=0), batch 0, 139 flags

| | max abs | mean abs | rel L2 | argmax agree | loss (ref 2.06484) |
|---|---|---|---|---|---|
| Unsloth 16-bit | 0.1660 | 0.0661 | 0.0270 | 10/11 | 2.06382 |
| Unsloth 4-bit | 1.2441 | 0.4391 | 0.1811 | 8/11 | 2.23147 |

Reference logit std 1.265. 16-bit is close but **not exact** (fp32 residual path vs our fp16 path; one question flips argmax). 4-bit quantisation noise is large (rel L2 18%), memory-only. All outputs finite. Same token batches in all stages (`ids_sha db85262cb1d7d707`; `same_ids_as_base true`).

Gradient check note: with autocast, all three stages report `grads_finite false` / `nonzero_grad 111-112` of 413 at GradScaler scale 65536 (first-step overflow, baseline identical). Without autocast Unsloth reports `grads_finite true`, `nonzero_grad 208`. This is not Unsloth-specific.

## Errors and warnings, verbatim (non-fatal; the run completed rc=0)

```
[unsloth_zoo.log|ERROR]Unsloth: Skipped SiglipEncoderLayer.forward
Reason: Parameter 'kwargs' signature changed
[unsloth_zoo.log|ERROR]==================
Failed to patch Gemma4Unified forced-float32 patches. Error
module 'transformers.models.gemma4_unified.modeling_gemma4_unified' has no attribute 'Gemma4UnifiedClippableLinear'
==================
[unsloth_zoo.log|WARNING]Unsloth: triton_kernels is not installed. MXFP4 GPT OSS will NOT be patched for LoRA.
[unsloth_zoo.log|WARNING]Unsloth: Skipped Inductor `config.cuda.compile_opt_level = -O1` on torch 2.11.0+cu130: NameError("name 'O1' is not defined")
[unsloth_zoo.log|WARNING]Unsloth: Patched trl.models.utils.disable_gradient_checkpointing with a no-op to preserve Unsloth gradient checkpointing across TRL generation passes.
[unsloth_zoo.log|WARNING]Loading google/gemma-4-E2B-it as text-only; vision/audio towers skipped. Use FastVisionModel for multimodal inputs.
```

The Gemma4Unified error is for the unified (multimodal) class, not the text model we use. The SiglipEncoderLayer skip is vision-only.

## End-to-end estimate (not measured)

Only the <=512 bucket (first 32 records) was measured. Applying its Prism ratio (15282.7 / 18740.9 = 0.8155) to every bucket of the r7 fit:

| bucket | weight | baseline s/sample | Unsloth 16-bit s/sample |
|---|---|---|---|
| <=128 | 2.2% | 0.150 | 0.122 |
| <=256 | 16.3% | 0.297 | 0.242 |
| <=512 | 55.2% | 0.599 | 0.489 |
| <=1024 | 19.7% | 1.18 | 0.962 |
| <=2048 | 6.6% | 1.82 | 1.484 |
| weighted mean | | 0.735 | 0.599 |

Uniform ratio, so the estimate is simply 1.226x on the weighted mean (4-bit: 1.203x). **Only <=512 was measured**; the 55% of samples there is covered, the 26% at 1024/2048 is an extrapolation and could differ (attention cost grows, compile recompiles per shape, mask path is SDPA mem-efficient in both).

## Verdict: PARTIAL (do not integrate yet)

- For: +22.6% samples/s on the dominant bucket with the real Prism masks, same trainable params, no extra memory problem, torch unchanged.
- Against: forces transformers 5.20.2 -> 5.17.0 (production pin change), not numerically identical (rel L2 2.7%, 1/11 argmax flip), per-shape compile warm-up of ~20-50 s, 4-bit is not usable for fidelity, Unsloth's own kernels and checkpointing contribute nothing here, and prior in-repo per-layer compile (OJ-079) gave only +1.5%, so the gain is probably mostly the fp32/fused patched blocks which need a check against what we already compile.
- Next step before any integration: measure 1024 and 2048 buckets, run a short quality-guard training comparison (loss curve vs baseline), and check the transformers 5.17 pin does not regress the baseline. If the 1024/2048 ratio holds and loss curves match, integrate behind a flag; otherwise reject.
