# OpenJev todo board (2026-11-19)

Two branches: **A = main MVP (70%)**, **B = XLA/TPU fork (30%)**.
Each workstream (WS) can go to one agent. "Gate" is what must be true before the item counts as done.

## Objectives

- **A (70%), MVP:** one full SFT → RL run of Gemma 4 E2B with the Prism mask completes on Kaggle 2×T4 in ≤12 h, with no error. It reports val acc, nll and ece for each source. RL is no worse than SFT on acc (within ±2 SE) and better on ECE. The final run starts only after the user approves it.
- **B (30%), portability:** the same recipe runs on CUDA, TPU (torch_xla) and CPU from one codebase. Checkpoints resume across sessions and backends. A measured TPU samples/s gives a ~50 h chained GPU+TPU plan. A minimal JAX prototype has CPU parity.

---

## A: main MVP (70%)

### WS-A1: Unblock Optuna round 3 (agent running now)
| ID | Task | Gate |
|---|---|---|
| A1.1 | Convert weights on the GPU after the move, not on the host (r3 SIGKILL at 10.2 GB RSS) | tests and preflight pass; host RSS < 6 GB on Kaggle |
| A1.2 | Relaunch round 3 (baseline, ple_cpu pair, profile, chunked on/off ckpt, flex, TPE) | round COMPLETE and ingested |
| A1.3 | End-of-round validation (150 steps, best vs baseline, 400-record val) | quality accept/reject recorded |

### WS-A2: Throughput, data side
| ID | Task | Gate |
|---|---|---|
| A2.1 (OJ-072) | Option sub-sampling in training (gold + ≤24 negatives); full options at eval | test; Optuna dim on |
| A2.2 (OJ-060) | Hierarchical packing: source → length bucket → FFD rows ≤2048, doc-id mask | packed logits == unpacked logits |
| A2.3 (OJ-076) | Tokenize/collate/pack in DataLoader workers, pinned, non_blocking | collate time measured, off critical path |
| A2.4 (OJ-077) | No `.item()` / sync inside accumulation | profile shows no per-micro-step sync |
| A2.5 (OJ-067) | max_state_tokens 2048 after packing (helpsteer3 truncation) | quality validated |

### WS-A3: Throughput, model and memory side
| ID | Task | Gate |
|---|---|---|
| A3.1 (OJ-085/085b) | Compact PLE table (157k ids, 2.82 GB) then a hot/cold split (measure the coverage curve first) | logits equal to full table |
| A3.2 (OJ-084) | Overlap the CPU PLE lookup with the previous step | ple_cpu=1 speed ≈ ple_cpu=0 |
| A3.3 (OJ-073) | Auto grad-checkpointing from measured memory | Optuna result |
| A3.4 (OJ-075) | QLoRA 8-bit / nf4 bench | kept only if samples/s rises |
| A3.5 (OJ-079) | torch.compile bench | kept only if stable and faster |

### WS-A4: Two GPUs
| ID | Task | Gate |
|---|---|---|
| A4.1 (OJ-062) | DDP via torchrun on 2×T4, token-balanced shards, eval/save on rank 0 | 2-process gloo test on CPU |
| A4.2 (OJ-078) | DDP knobs: no_sync, static_graph, bucket size, fp16 compress hook, NCCL env | each benched ~1 round |

### WS-A5: Optuna loop operations (continuous)
| ID | Task | Gate |
|---|---|---|
| A5.1 | Each round: ingest, write ideas with references, propose, push | EXPERIMENTS.md and DECISIONS.md updated |
| A5.2 | Validation every 30 trials; acceptance uses ±2 SE | quality section filled |
| A5.3 | Stop rule: <5% gain over 3 rounds and all ideas built or rejected → push notification to the user | user notified |

### WS-A6: Final run (after A5.3 and the user's OK)
| ID | Task | Gate |
|---|---|---|
| A6.1 (OJ-061) | Time budget per stage, save every 100 steps, copy outputs after each stage | resume test |
| A6.2 | Final SFT → RL run with the best config | objective A met |
| A6.3 (OJ-066) | Calibration baselines: frozen backbone + head, SFT + temperature/Platt | table in the eval report |

### WS-A7: Model-quality research (parallel, low GPU cost)
| ID | Task | Gate |
|---|---|---|
| A7.1 (OJ-065) | Flag-mode A/B: isolated vs siblings, plus a flip-rate metric | result on val |
| A7.2 (OJ-063) | Zero-shot eval with an LM-head "Yes/No" scorer | pass@k per source |
| A7.3 (OJ-064) | Per-source curriculum (pass@k rule) plus a direct-RL ablation | A/B result |
| A7.4 (new) | Elimination and "none of the above" data: twins that drop the gold option (target: NOTA or a flat distribution), and an inconclusive check (near-uniform probability when the state has no evidence). It must follow the Jev format | converter + test; no invented labels |
| A7.5 (OJ-068) | New data: typed-decisions, BEV, MASSIVE, Bitext (license check) | converters and tests |

### WS-A8: Tooling and docs
| ID | Task | Gate |
|---|---|---|
| A8.1 (OJ-069) | `stop.sh` via SDK CancelKernelSession; a watcher that uses `logs -f` | works on a test kernel |
| A8.2 | Keep the site, tensor course and KL page current (new artifact links) | republished |
| A8.3 (OJ-071) | W&B live metrics — **user** must add the WANDB_API_KEY secret first | user action |

---

## B: XLA/TPU fork (30%)

### WS-B1: Backend core
| ID | Task | Gate |
|---|---|---|
| B1.1 | `openjev/backend.py`: device, dtype policy, optimizer step, sync, save/load | CUDA/CPU behavior unchanged; tests green |
| B1.2 | Static shape buckets (512…2560) | distinct shapes ≤ number of buckets |
| B1.3 | Resumable checkpoints across backends, `--resume auto`, `--max-minutes` | 6 steps == 3 + resume + 3 |

### WS-B2: TPU on Kaggle (TPU quota only, own kernel ids, first push needs the user's OK)
| ID | Task | Gate |
|---|---|---|
| B2.1 | TPU env probe: torch_xla 2.9 on v5e, HF Gemma 4 load | probe log |
| B2.2 | TPU bench profile and Optuna study `openjev-xla-v1` (compile time vs steady state, recompiles, dp vs fsdp_spmd, 1024/2048/4096 tokens) | samples/s per chip |
| B2.3 | The 50 h plan (PLAN_50H.md) with measured TPU numbers | plan with real numbers |

### WS-B3: JAX prototype (minimal)
| ID | Task | Gate |
|---|---|---|
| B3.1 | Prism-masked flag head + LoRA step in JAX/Flax on the tiny config (Google `gemma` lib has Gemma 4 E2B) | CPU run + parity with the PyTorch tiny model |

---

## Closed or obsolete

- **Done:** OJ-026 (Kaggle CLI), OJ-074 (SDPA check: math fallback), OJ-080 (mask fix), OJ-081 (dtype fix).
- **Obsolete (Qwen3 era):** OJ-027, OJ-041, OJ-043.
- **Later:** OJ-032, OJ-033, OJ-051 (user), OJ-070 (E4B, after E2B is clean).
