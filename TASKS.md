# OpenJev board

Plan and design reasons: `PLAN.md`. Target: a Jev-compatible causal encoder under 1B total parameters.

How to use this board:
- An agent takes the **first unchecked ticket in "Todo"**, moves it to "In progress",
  finishes it, tests it, moves it to "Done", and adds one line to the progress log.
- A ticket is `OJ-xxx [phase] title`, then the acceptance criteria.
- The user can add, reorder, or remove tickets at any time.
- Tickets marked **(user)** need a human (credentials, spending GPU quota). Agents skip them.

Phases: P0 foundation · P1 dataset · P2 GPU · P3 SFT · P4 RLCD · P5 serve.

## In progress

(none)

## Todo

- [ ] OJ-026 [P2] **(user)** Install the Kaggle CLI, put `~/.kaggle/kaggle.json` in place, check the
      current weekly GPU quota and session limit, and write them in `PLAN.md`.
- [ ] OJ-027 [P3] **(user)** First real SFT run of Qwen3-0.6B on Kaggle (T4/P100, fp16). Record the
      metrics in the log.
- [ ] OJ-043 [P2] Backbone smoke checks: (a) max |hidden state| of Qwen3-0.6B under fp16 autocast on real batches
      (NaN/overflow risk on T4); (b) `Qwen/Qwen3.5-0.8B` loads as a causal encoder, flag gather works, 50 SFT steps on CPU tiny data.
- [ ] OJ-032 [P4] (Low priority: pokerbench and lichess give only the best action, not an EV for each action, so
      this reward is the same as correctness until a source with real utilities exists.) Outcome rewards: REINFORCE with a Brier-shaped reward for poker and chess, where a real
      utility exists. Behind a flag.
- [ ] OJ-033 [P4] Ablation report: SFT, SFT+T, +RLCD(cal), +RLCD(cal+contrast), +T on one test set.
- [ ] OJ-041 [P4] **(user)** Train the dense and the MoE (4×last 8) model with the same data, steps and active compute on Kaggle.
      Compare them with `eval.py`. The MoE ships only if it is better.
- [ ] OJ-051 [P5] Model card + eval tables. Release on the HF Hub **(user)**.

- [ ] OJ-060 [P1] Prism v8 batching: hierarchical packing (source mix -> length bucket -> first-fit-decreasing rows of
  <= 2048 tokens). Doc-id block-diagonal mask, positions restart per request. Test: packed flag logits == unpacked.
- [ ] OJ-061 [P1] Time-budgeted Kaggle run: `--max-minutes` per stage (SFT ~65%, RL ~20%), eval+save every 100 steps
  on 300 samples, copy checkpoints to out/ after each stage; print measured s/step in the log.
- [ ] OJ-062 [P2] DDP on 2x T4 (torchrun): sharded mixture with per-rank seed, eval/save on rank 0, PLE on CPU per rank.
  Local test with 2 CPU processes (gloo).
- [ ] OJ-063 [P2] Zero-shot eval with a pretrained head: `--head lmhead` (score = logit Yes - logit No at each option,
  softmax over options), `--template natural`, `--mask causal`. Report pass@1 / pass@k / ECE per source on val.
- [ ] OJ-064 [P2] Per-source curriculum (tokenbender rule): pass@k ~ 0 -> SFT; 0 < pass@k < 1 -> direct RL;
  near 1 -> calibration only. A/B: direct RL on "middle" sources vs SFT->RL, same val set. Needs OJ-063.
  Also ablate direct exact-PG from the random head at equal compute (avataRL: dense partial-credit reward
  makes RL-from-zero work; see docs/research/sft-vs-rl/notes/tokenbender.md). The rule itself is unverified.
- [ ] OJ-065 [P2] Flag-mode A/B: `--flag-mode isolated|siblings` (Prism-1 vs Sarvam). Add option-order flip-rate metric.
- [ ] OJ-066 [P3] Calibration baselines in eval.py: frozen-backbone + head, SFT + temperature/Platt scaling;
  report 15-bin ECE, Brier, selective accuracy at 50% coverage.
- [ ] OJ-067 [P3] `--max-state-tokens 2048` on E2B once packing lands (helpsteer3: 53% truncated at 1024).
- [ ] OJ-068 [P3] New data from docs/research/decision-models: typed-decisions (Apache-2.0), BEV (filter per source
  license), MASSIVE en-US, Bitext. Converters must follow the Jev format rules.
- [ ] OJ-069 [P3] `kaggle/prism/stop.sh` via SDK CancelKernelSession (needs scriptVersionId); watcher uses
  `kaggle kernels logs -f` (stdbuf) for live step/loss reports.
- [ ] OJ-070 [P4] E4B run (bs 2 x accum 16, 1024 tokens) after E2B is clean.
- [ ] OJ-071 [P4] Optional live metrics via W&B **(user: add WANDB_API_KEY secret first)**.

- [ ] OJ-072 [P1] Speed: option sub-sampling in training only (keep the gold option + up to 24 random negatives per
  choice question; full option set at eval). clinc150 (150 options) and banking77 (77) templates shrink from
  ~1000 to ~200 tokens. Exact for isolated flags; check accuracy impact with the siblings mask.
- [ ] OJ-073 [P2] Speed: turn off gradient checkpointing when the measured peak memory allows (E2B uses ~7 of 15 GB);
  add `--grad-ckpt auto` that measures the first step and decides.
- [ ] OJ-074 [P3] Speed: log which SDPA kernel runs with the custom float mask on T4 (mem-efficient vs math);
  if math, test FlexAttention or a boolean mask.

- [ ] OJ-076 [P1] Input pipeline off the critical path: tokenize/collate/pack in DataLoader workers (prefetch_factor 2,
  pin_memory, .to(non_blocking=True)); measure collate time vs GPU step time first (Python encode per record may be
  a large share of the 4.5 s/sample). Set torch.set_num_threads per rank (Kaggle has ~4 CPU cores for 2 ranks).
- [ ] OJ-077 [P1] Remove per-micro-step GPU syncs: no .item()/float() on losses or stats inside accumulation; keep
  running sums on GPU, sync only at log steps (every 10 steps).
- [ ] OJ-078 [P2] DDP tuning checklist for OJ-062: no_sync() on accumulation micro-steps; static_graph=True +
  use_reentrant=False; gradient_as_bucket_view=True; bucket_cap_mb sweep (25/50); token-balanced shards per rank
  (equal packed tokens) to avoid stragglers; sharded eval + all_gather; fp16_compress_hook (bench, small grads);
  NCCL env for PCIe T4s (NCCL_IB_DISABLE=1; test NCCL_P2P_DISABLE 0/1); scale lr for the larger global batch;
  print per-rank step time to spot stragglers. Bench each knob ~1 min.
- [ ] OJ-079 [P3] torch.compile bench (mode=default) on T4 with PEFT + float mask; keep only if stable and faster.

## Done

- [x] OJ-013 [P1] Full conversion run on this machine: no limit for banking77, clinc150, support_tickets and helpsteer3;
      100k rows for lichess_puzzles and pokerbench.
      Record the row counts and any download failure in the log.
- [x] OJ-050 [P5] `serve.py`: FastAPI `POST /v1/systemone` with the exact Jev request and response
      body (`model`, `answers`, `usage`).
- [x] OJ-003 [P0] `pyproject.toml` (package `openjev`, deps torch, transformers) and a GitHub
      Actions workflow that runs `pytest` on CPU. (Needs `git init` first.)
- [x] OJ-042 [P4] `train.py --upcycle E,N`: call `moe.upcycle` after loading, add `λ_aux·moe.aux_loss` (default 0.01)
      to the loss, save the MoE config in the checkpoint, and teach `checkpoint.load` to rebuild it. CPU smoke test.
- [x] OJ-031 [P4] `rlcd_train.py`: start from the SFT checkpoint with a frozen reference copy. Loss =
      proper score + λ_cal·soft-ECE + λ_kl·KL(p‖p_ref) + λ_con·contrast consistency.
      CPU smoke test on the tiny model.
- [x] OJ-014 [P1] Converter `converters/prm800k.py` (GitHub `openai/prm800k`): score with 3 levels per step.
- [x] OJ-017 [P1] Long states: helpsteer3 median 1427 and p95 3658 tokens. `encode` cuts the end, so it cuts
      the second response. Add a truncation that keeps the head and the tail, or a per-part budget, in the converter.
- [x] OJ-015 [P1] Converter `converters/diplomacy_deception.py` (GitHub `DenisPeskoff/2020_acl_diplomacy`):
      noul "is the sender telling the truth?".
- [x] OJ-010 [P1] Splits: add `--split` handling to every converter so that each writes
      `data/<source>.{train,val,test}.jsonl`. Use real HF splits when they exist; otherwise split by a
      hash of the row id. No row may be in two splits (test).
- [x] OJ-016 [P1] Fix the pokerbench answer leak: the menu (facing bet or not, size grid) comes from the
      state only; sized labels snap to the nearest grid size (log space). Test: the same state with a
      different label gives the same option set.
- [x] OJ-023 [P3] Temperature per question type after training (`rlcd.fit_temperature` on `val`),
      stored in the checkpoint and applied in `answers()`. Report the ECE before and after.
- [x] OJ-024 [P3] `eval.py`: load a checkpoint and write a metrics table (markdown + JSON) for the
      `test` split. Add a TF-IDF + logistic-regression baseline for each source.
- [x] OJ-040 [P4] Experiment: sparse upcycling of Qwen3-0.6B into a MoE (4 experts, top-1, last 8
      layers, ≈0.83B total). Unit test for the parameter count < 1B. Ships only if it beats the dense model.
- [x] OJ-021 [P3] `train.py`: mixed JSONL, AdamW with 2 lr groups, warmup + cosine, fp16 autocast +
      GradScaler on CUDA (fp32 on CPU), gradient checkpointing, eval on `val` (accuracy, NLL,
      Brier, ECE per type and per source), checkpoints (safetensors + tokenizer + config).
      CPU smoke test on the tiny model only.
- [x] OJ-022 [P3] Ordinal loss for `score`: CE + squared EMD between the CDFs (option `--score-loss
      ce|ce+emd|corn`). Tests: a near miss costs less than a far miss.
- [x] OJ-020 [P2] `openjev/collate.py`: batch and pad records, and gather all flag positions with one
      tensor index. The outputs must equal the current loop in `OpenJev.forward` (test).
- [x] OJ-025 [P2] `kaggle/` folder: `kernel-metadata.json` (script, GPU, internet, dataset source),
      `kaggle/run.sh` (push → poll status → pull output), `kaggle/make_dataset.sh`
      (`kaggle datasets create/version`). Dry-run mode that only prints commands.
- [x] OJ-030 [P4] `openjev/contrast.py`: deterministic contrast transforms of real records (noul
      criteria swap, score order reverse, choice option drop, option shuffle) with the exact
      mapping of the target distribution. Tests for each transform.
- [x] OJ-001 [P0] `tests/` pytest suite: schema round trip, labels (good and bad), `bucket()`,
      forward + loss + `answers()` on `tiny-Qwen3ForCausalLM`, `rlcd.py` functions.
      Answer keys equal the Jev docs.
- [x] OJ-002 [P0] Tokenizer: map the template tokens onto the unused special tokens of the base
      tokenizer when they exist (Qwen3 has spare slots), else add them. Init new embeddings to the
      mean plus a small random offset. Keep the vocab size a multiple of 128. Test with the tiny model.
- [x] OJ-011 [P1] `scripts/data_stats.py`: rows, type counts, option-count histogram, label
      balance, state token length (Qwen3 tokenizer), leakage check (answer text inside `state`).
      Writes `data/STATS.md`.
- [x] OJ-012 [P1] `openjev/mix.py`: mixed sampler over JSONL files with per-source weights
      (`w ∝ n^alpha`, alpha 0.5 by default). Deterministic with a seed. Tests.
- [x] OJ-100 [P0] Jev-format schema, model, data format, toy training, 58 dataset samples with reviews.
- [x] OJ-101 [P1] Converters `banking77`, `clinc150`, `support_tickets`, `helpsteer3`,
      `lichess_puzzles`, `pokerbench`. Samples of 200 rows each validate with `read_jsonl`.
- [x] OJ-102 [P4] `openjev/rlcd.py`: soft ECE, ECE, Brier, `rlcd_loss`, `reinforce`, `fit_temperature`.
- [x] OJ-103 Backbone choice, GPU source, and RLCD proposal written in `PLAN.md`.

## Progress log

- 2026-09-18 — Initial code pushed: Jev-format schema, model, data format, toy training,
  58 dataset samples with reviews (see `samples/index.html`).
- 2026-09-21 — Six converters added. 200-row samples of each are in `data/` and pass `read_jsonl`.
- 2026-09-24 — Research: no pretrained MoE is under 1B total params, so the backbone is Qwen3-0.6B (dense),
  with a MoE via sparse upcycling (≈0.83B) as the OJ-040 experiment. The GPU source is Kaggle (fp16, because
  the T4/P100 have no bf16). The RLCD proposal combines the TypeSafe goal with the contrast pairs of Yang et al.
  `TASKS.md` is now a kanban board. The smoke test passes (loss 2.978 → 2.450).
- 2026-09-24 — OJ-001 done: 55 tests pass. The builder duplicate-name bug is fixed. OJ-002 done: the template tokens use
  the 267 spare embedding rows of Qwen3 (no resize), and each gets the mean + noise. OJ-011 and OJ-012 done. The stats found
  a pokerbench answer leak (the off-grid bet size marks the answer) → OJ-016. The `--limit` samples took the first rows
  (clinc150 had no out-of-scope rows) → handled in OJ-010. Smoke test (101 steps): loss 2.885 → 1.044.
- 2026-09-24 — OJ-030 done: `openjev/contrast.py` (swap, reverse, shuffle, drop twins with exact target maps), 13 tests.
- 2026-09-24 — OJ-020 done (one gather for all flags, 72 tests pass). OJ-025 done (`kaggle/run.sh` bundle|upload|push|status|wait|pull, dry-run checked; the fallback no longer gives `--fp16` to train_toy.py).
- 2026-09-24 — OJ-021 done: `train.py` (mixed JSONL via `Mixture`, AdamW 2 lr groups, warmup+cosine,
  fp16 autocast+GradScaler on CUDA only, grad checkpointing, eval on `val` with `openjev/metrics.py`
  Accumulator giving per-type/per-source acc/nll/brier/ece/mae, safetensors+tokenizer+heads.pt+config
  checkpoint keeping best val nll). `openjev/metrics.py` added (`Accumulator`, `ece_from_conf`).
  CPU smoke test on `trl-internal-testing/tiny-Qwen3ForCausalLM` passes (`tests/test_train_smoke.py`).
  99/99 new+existing tests pass except 2 pre-existing `tests/test_converters.py` failures
  (pokerbench leak, out of scope — being fixed under OJ-016).
- 2026-09-24 — OJ-022 done: `question_loss(score_loss="ce+emd")` is now the default for score (squared EMD, Hou et al. 2016). 81 tests pass.
- 2026-09-24 — OJ-040 done: `openjev/moe.py` (sparse upcycling, top-1, identity at init). Counts from the real Qwen3-0.6B config: dense 596.0M; MoE 4×last-8 822.6M total / 596.1M active; 8×last-4 860.3M / 596.1M. Both are under 1B.
- 2026-09-24 — OJ-023 + OJ-024 done: `calibrate.py` (one T per type; T is clamped to [0.05, 20] with a warning), `eval.py` (with and without T, TF-IDF+LR baseline), `openjev/checkpoint.py`. 111 tests pass.
- 2026-09-24 — OJ-010 + OJ-016 done. Converters write `<prefix>.{train,val,test}.jsonl` (real HF splits, else a sha1 split),
  and `--limit` takes a seeded sample. The pokerbench menu comes from the state only (facing_bet + pot grid), and sized
  labels snap to the grid (123/166 moved, mean rel. error 6.8%). Skipped: 12 unparseable, 11 label contradicts state.
  The remaining leak metric gap (0.30) is the verbs in the hand history, not a construction leak. 111 tests pass.
- 2026-09-24 — OJ-015 done: diplomacy_deception full data (GitHub `master` branch): train 13,132 / val 1,416 / test 2,741 messages; lie rate 4.5% / 4.0% / 8.8%. 8 tests.
- 2026-09-24 — OJ-017 done: helpsteer3 `fit_parts` budget (context 30%, each response 35%, head+tail cut). States are now ≤ 1012 tokens (before: median 1427, p95 3658). `data/helpsteer3.*` is regenerated in OJ-013.
- 2026-09-24 — OJ-042 done: `train.py --upcycle E,N` (aux loss, stops if total ≥ 1B, the MoE checkpoint round-trips).
  OJ-031 done: `rlcd_train.py` (proper + λ_cal·soft-ECE + λ_kl·KL(p‖p_ref) + λ_con·contrast consistency, frozen SFT reference;
  the temperatures are marked stale). Review change: the soft-ECE term is now computed per question type, then averaged.
  OJ-014 done: prm800k from the GitHub LFS media URL; one score question per rated step, each problem in one split;
  labels are skewed to "good" (~89%). Full run with a 100k limit started.
- 2026-09-24 — prm800k full run (limit 100k for each raw file): train 102,342 / val 4,445 / test 15,566.
- 2026-09-24 — OJ-003 done: `pyproject.toml` (extras data/eval/serve/dev) and `.github/workflows/tests.yml` (CPU torch, pytest, smoke test). `pip install -e .` works. The workflow runs only after `git init` and a push (the user decides).
- 2026-09-24 — OJ-050 done: `serve.py` `POST /v1/systemone`, response exactly {model, answers, usage{input_tokens, output_tokens}} per docs.typesafe.ai/api.md. Open point: `output_tokens` = the number of flag readouts (the docs do not say how Jev counts them). 8 tests.
- 2026-09-24 — OJ-013 done. Rows train/val/test: banking77 9,504/489/3,076; clinc150 15,250/3,100/5,500;
  support_tickets 47,494/2,600/2,673 (8,998 skipped); helpsteer3 36,570/2,017/1,889; lichess 90,175/4,987/4,838;
  pokerbench 88,541/4,620/10,471 (7,368 skipped); prm800k 102,342/4,445/15,566; diplomacy 13,132/1,416/2,741.
  About 403k train rows, `data/` is about 1.2 GB. The Kaggle default run args are now a real first SFT run
  (3000 steps, bs 8 × accum 4, fp16, grad-ckpt).
- 2026-11-18 — OpenJev-Prism: `openjev/prism.py` (Prism mask on Gemma 4: shared block bidirectional, options isolated
  with a shared start position, flags see sibling options; full + sliding masks; PLE-safe template tokens),
  `train_prism.py` (LoRA; `--algo sft|exact|grpo`, Brier/acc/graded reward, soft-ECE + KL to adapter-off ref),
  `tests/test_prism.py` (option-order invariance), `notebooks/explore_data.ipynb`, `kaggle/prism/{build,push}.sh`,
  `entry.py`. Smoke-tested on a tiny random Gemma 4; 149 tests pass. Not uploaded, no real training run.
- 2026-11-19 — prism-train v3 ERROR in preflight: peft LoRA injection rejected Kaggle's torchao 0.10 (model load OK). entry.py now uninstalls torchao and asserts the env; pushed v4.
- 2026-11-19 — prism-train v4: preflight SFT OK on real Gemma 4 E2B (train/eval/save); RL resume failed (template-token ids >= 262144 rows: saved tokenizer already had the tokens, so no resize). Fix: resize when len(tok) > rows; regression test tests/test_prism_resume.py; 150 tests pass; pushed dataset + v5.
- 2026-11-19 — Memory/stability fix: embedding matrix no longer trained (fp32 template_delta for the 7 template tokens), all trainable params fp32 (fp16 AdamW eps underflow), PLE table kept on CPU (--ple-cpu, saves 4.7 GB on E2B). Measured on meta: E2B 4.63B text (PLE 2.35B) -> 4.6 GB GPU fp16; E4B 7.46B (PLE 2.82B) -> 9.3 GB; LoRA r16 24M/35M. bs 4 x accum 8. Pushed v6 (v5 still running old code).
- 2026-11-19 — prism-train v6 ERROR in SFT preflight: PLE weight on CPU but its embed_scale buffer on cuda. place() now detaches the whole PLE module during model.to; regression test tests/test_prism_place.py (meta device); 151 tests pass; pushed v7.
- 2026-11-19 — prism-train v7: PREFLIGHT OK on real Gemma 4 E2B (first error-free run), but ~140 s/step (4.4 s/sample, bs4x8) -> 2000 steps would need ~78 h. Stopped by user at step ~70 (ce 2.7->~2.0-2.4). Log in runs/prism-20261119-1657-v7. Next: packed length-bucketed batching, DDP, time budget.
- 2026-11-19 — v8 short run: SFT 60 steps done, ~35 s/step (8 samples). val@30 acc .303 nll 2.162 ece .072; val@60 acc .296 nll 2.126 ece .058. banking77 0.00, prm800k 0.04 (below majority). RL started (reward .25, kl .018). Added speed todos OJ-072..074.
