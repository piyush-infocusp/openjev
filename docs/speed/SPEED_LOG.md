# OpenJev training speed log

The goal is the highest number of training samples per second for Gemma 4 E2B with the Prism mask
on Kaggle 2× T4 (16 GB, fp16, PCIe, no NVLink), with no loss of correctness. After this, the final
long run starts, and it waits for the user's approval.

Each technique below has a todo ID in `TASKS.md`, a reference, the status, and the measured
result. The speed worker (a cron) adds one result row for each benchmark.

## Baseline

| Run | Date | Setup | s/step | Samples/step | **s/sample** | Peak GPU | Notes |
|---|---|---|---|---|---|---|---|
| v7 | 2026-11-19 | 1× T4, bs 4 × accum 8, grad-ckpt, padded batches, max state 1024 | ~140 | 32 | **~4.4** | n/a | stopped at step ~70 |
| v8 short | 2026-11-19 | 1× T4, bs 4 × accum 2, same | ~35 | 8 | **~4.5** | n/a | same cost per sample: the cost is per sample, not per step |

Estimated hardware limit: about 1 s/sample. The FLOP estimate is 6·N·T for about 1.9B non-PLE
parameters and about 1000 tokens, plus about 33% for checkpoint recompute, at about 15 TFLOP/s of
effective T4 fp16.

## Benchmark method

- Kaggle `PROFILE=bench` loads the model once. For each config, it runs 2 warm-up steps and then
  a timed loop of about 45–60 s.
- It prints one JSON line for each config: samples/s, tokens/s, s/step, peak memory, GPU count
  and NaN.
- A test of a few minutes runs only when a 1-minute result is noisy.
- Every speed change must have a correctness test before the benchmark, for example packed
  logits equal unpacked logits.
- The configs come from an Optuna ask/tell loop (TPE, multivariate): `tools/optuna_loop.py`
  (`propose` -> Kaggle `PROFILE=optuna` runs `bench_trials.py` -> `ingest` -> `report`). The
  search space and the planned ideas are in `tools/search_space.yaml`; every trial is in
  [`EXPERIMENTS.md`](EXPERIMENTS.md). Each trial restores the same start weights and data
  stream, and a fixed probe set gives loss start/end for the quality guard.
- Round 2: `attn_backend` swaps the attention backend at runtime (`openjev/attn_backends.py`, no
  reload). A `profile=1` trial (never drawn by TPE; enqueue it with `propose --enqueue-json`) adds
  a torch.profiler table after its timed loop, so its samples/s still counts.
- Round 3 (Idea 5, study `openjev-speed-v2`): micro-batch 8 and global batch 32 are fixed study settings in
  the yaml; each timed step is one optimizer step of 32 samples (1 warm-up + 4 timed steps on the frozen
  trial stream). Quality checkpoints (best vs baseline, 150 real steps, frozen val set) run as a separate
  `PROFILE=optuna_validate` kernel.

## Techniques

### Data and sequence length

| ID | Technique | Why it should help | Reference | Status | Result |
|---|---|---|---|---|---|
| OJ-072 | Option sub-sampling in training: keep the gold option and ≤24 negatives, and use all options in eval | clinc150 and banking77 templates are about 1000 tokens because of the options | Negative sampling in classification heads: Mikolov et al. 2013, *Distributed Representations of Words and Phrases*, arXiv:1310.4546 | todo | |
| OJ-060 | Hierarchical packing (source → length bucket → first-fit-decreasing rows ≤2048) with a doc-id block-diagonal mask and positions that restart | Removes padding waste from mixing 11-token and 1000-token samples | Krell et al. 2021, *Efficient Sequence Packing without Cross-contamination*, arXiv:2107.02027 | todo | |
| OJ-067 | `--max-state-tokens 2048` after packing | Quality (53% of helpsteer3 states are cut), not speed | EDA notebook | todo | |
| Idea 5 | Optuna round 3: fixed micro-batch 8 × grad_accum 4 = global batch 32 for every trial, frozen trial stream, `bs`/`dtype`/`ple_cpu` not sampled, new study `openjev-speed-v2`, OOM at mb 8 = pruned `infeasible_at_mb8` | Batch size confounded speed with optimization (gradient noise, lr) and the quality guard; fixed batches make samples/s and guards comparable | Goyal et al. 2017, arXiv:1706.02677; McCandlish et al. 2018, arXiv:1812.06162 | harness done, round 3 not run | |

### Compute and memory

| ID | Technique | Why it should help | Reference | Status | Result |
|---|---|---|---|---|---|
| OJ-073 | Gradient checkpointing only when needed (`auto`) | It costs about 33% more compute, and E2B uses about 7 of 15 GB | Chen et al. 2016, *Training Deep Nets with Sublinear Memory Cost*, arXiv:1604.06174 | todo | |
| OJ-074 | Check the SDPA kernel with the custom float mask on the T4 | The math fallback stores T² scores and is slow | PyTorch SDPA docs: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html | done (Optuna r1) | `efficient` → "No available kernel" (head_dim 256/512 + float mask on sm75), so the default path is the math kernel |
| OJ-081 | `dtype` dim [fp16, bf16_emulated, fp32] (reload, default fp16); `--dtype auto` = bf16 only with native support (sm ≥ 8.0) | `torch.cuda.is_bf16_supported()` counts emulated bf16, so v7/v8 and round 1 likely ran emulated bf16 on the T4 | PyTorch AMP docs: https://pytorch.org/docs/stable/amp.html; `torch.cuda.is_bf16_supported(including_emulation=True)` | implemented (r2) | |
| OJ-082 | Profiler trial (`profile=1`, set with `propose --enqueue-json`): torch.profiler CPU + CUDA over 2 steps after the timed loop; top-25 ops by self CUDA / CPU time and the share of attention, matmul, masks, PLE lookup, copies and collate, in the trial's user attrs and EXPERIMENTS.md "Profiles" | r1 runs at ~120 tokens/s, about 10% of what a T4 should do, so a large overhead is hidden; the profile names it | PyTorch profiler recipe: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html | implemented (r2) | |
| OJ-083a | `attn_backend=sdpa_bool`: boolean Prism mask (True = attend) to SDPA instead of the additive float mask; bench records which SDPA kernels accept each head dim with a bool vs float mask (`sdpa_kernels`) | Half the mask memory; another SDPA kernel may become eligible | PyTorch SDPA docs: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html | implemented (r2) | equal logits (tests/test_attn_backends.py) |
| OJ-083b | `attn_backend=chunked_math`: math attention over 256-query chunks, each chunk checkpointed (the G query heads of a kv head are folded into the GEMM rows, so K/V are not repeated) | Attention memory O(chunk·T) instead of O(T²) per layer, so grad-ckpt off may fit | Same math as SDPA's math kernel (SDPA docs above); checkpointing: Chen et al. 2016, arXiv:1604.06174 | implemented (r2) | logits and grads equal SDPA within 1e-4 on CPU |
| OJ-083c | `attn_backend=flex`: FlexAttention with a BlockMask whose mask_mod looks up the Prism mask (sliding window included), `torch.compile`d on CUDA, small Triton tiles for head_dim 256/512 | Skips fully masked blocks and never stores T² scores; fused kernel | Dong et al. 2024, *Flex Attention*, arXiv:2412.05496 | implemented (r2) | equal logits on CPU (eager, forward only); CPU training → "unsupported"; T4 untested |
| OJ-075 | QLoRA 4-bit nf4 or 8-bit (bitsandbytes) | Less memory, which allows a larger batch or no checkpointing; but dequantization costs time | Dettmers et al. 2023, *QLoRA*, arXiv:2305.14314; Dettmers et al. 2022, *LLM.int8()*, arXiv:2208.07339 | todo | |
| OJ-079 | `torch.compile` | Kernel fusion | PyTorch 2 paper, Ansel et al. 2024, ASPLOS: https://pytorch.org/assets/pytorch2-2.pdf | todo | |
| done | PLE table on CPU | Frees about 4.7 GB on E2B | Gemma 4 config; `train_prism.place()` | done (v6/v7) | E2B weights on GPU: 9.3 → 4.6 GB |
| done | Frozen embedding + fp32 template delta; fp32 trainables | Saves ≥2.4 GB of AdamW state; avoids fp16 eps underflow | Loshchilov & Hutter 2019, *AdamW*, arXiv:1711.05101 | done (v6) | |

### Input pipeline and synchronization

| ID | Technique | Why it should help | Reference | Status | Result |
|---|---|---|---|---|---|
| OJ-076 | Tokenize, collate and pack in DataLoader workers; pin_memory; non_blocking | Today the GPU waits for Python collation between steps | PyTorch data loading docs: https://pytorch.org/docs/stable/data.html | todo | |
| OJ-077 | No `.item()` or `float()` inside accumulation micro-steps | Each call blocks until the GPU finishes | PyTorch performance tuning guide: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html | todo | |

### Multi-GPU (DDP on 2× T4)

| ID | Technique | Why it should help | Reference | Status | Result |
|---|---|---|---|---|---|
| OJ-062 | DDP with torchrun, one process for each GPU | About 2× samples/s; the model fits on each GPU | Li et al. 2020, *PyTorch Distributed*, VLDB, arXiv:2006.15704 | todo | |
| OJ-078a | `no_sync()` on accumulation micro-steps | One all-reduce for each optimizer step | DDP docs: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html | todo | |
| OJ-078b | `static_graph=True`, `use_reentrant=False`, `gradient_as_bucket_view=True`, `bucket_cap_mb` sweep | Better comm/compute overlap; fewer copies | same | todo | |
| OJ-078c | Token-balanced shards per rank | No GPU waits at the all-reduce | Krell et al. 2021 (packing) | todo | |
| OJ-078d | `fp16_compress_hook` | Halves the PCIe gradient traffic (small for 24M LoRA params) | DDP comm hooks: https://pytorch.org/docs/stable/ddp_comm_hooks.html | todo | |
| OJ-078e | NCCL env for PCIe T4: `NCCL_IB_DISABLE=1`, test `NCCL_P2P_DISABLE` 0 and 1 | Some cloud T4 pairs are faster without P2P | NCCL env docs: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html | todo | |
| OJ-078f | Sharded eval with `all_gather` | 2× faster eval | DDP docs | todo | |
| OJ-078g | Learning-rate scaling for the larger global batch | Keeps training stable | Goyal et al. 2017, *Accurate, Large Minibatch SGD*, arXiv:1706.02677 | todo | |
| rejected | FSDP, ZeRO, pipeline or tensor parallelism | The model fits on one GPU and only LoRA trains, so these only add communication | Rajbhandari et al. 2020, *ZeRO*, arXiv:1910.02054 | rejected (analysis) | |

## Results (one row for each benchmark)

| Date | Kernel ver. | Config | GPUs | s/step | samples/s | s/sample | Speedup vs 4.5 s | Peak GPU (GB) | NaN | Kept? |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-19 | v8 | baseline (padded, grad-ckpt, bs 4) | 1 | ~35 | ~0.22 | ~4.5 | 1.0× | n/a | no | baseline |
