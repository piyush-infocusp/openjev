# OpenJev final run plan (Objective A, WS-A6)

Status: plan, written 2026-11-23; user decisions 2026-11-24 (Q1-Q4 answered, see the end) and the pre-launch code gaps of §3/§5 closed on branch ws-prelaunch. **The run starts only after the user says OK.** This plan does not push anything to Kaggle and makes no `gh` calls.
Goal (`docs/TODO_BOARD.md:7`): one SFT → RL run of Gemma 4 E2B with the Prism mask and LoRA on Kaggle 2×T4, finished in ≤12 h with no error. It must report val acc, nll and ece for each source. RL must be no worse than SFT on acc (within ±2 SE) and better on ECE.
Every number marked **est.** is an estimate. The measured numbers come from r7/r8 (`docs/speed/EXPERIMENTS.md:10,30`, `runs/optuna/r8/results.md`).

## 1. Fixed configuration

| Setting | Value | Set by |
|---|---|---|
| Model, mask | `google/gemma-4-E2B-it`, Prism mask (not `--causal`) | OJ-080 mask fix (DECISIONS 2026-11-19) |
| LoRA | r 16, alpha 32 (`train_prism.py:84-85`); the flag head and template delta also train | defaults |
| Data | 8 sources, 403,008 train records, Mixture α 0.5 (`train_prism.py:73`, `openjev/mix.py:21`) | data build; `data/*.jsonl` |
| Batch | **micro-batch 8, grad-accum 4 (global), global batch 32** | DECISIONS 2026-11-19 (fixed mb 8). Done: `kaggle/prism/entry.py` `common` (all profiles) |
| max_state_tokens | **2048** in the Kaggle profiles (2026-11-24 task; the 2.62 bench rate was measured at 2048). The `train_prism.py` default stays 1024 | DECISIONS 2026-11-19 (round 4) |
| Length grouping | on, group-chunk 50 (`train_prism.py:123,130,154-155`) | DECISIONS 2026-11-23: pad_frac 3.1% vs 50.3% random |
| Packing | **off** (user 2026-11-24, Q2). `--packing` default is now off. With grouping, unpacked pads 3.1% and packed pads 7.1% (EXPERIMENTS:322-327). The 2.62 bench number is unpacked | user |
| Precision | fp16 + GradScaler (`--dtype auto`, `train_prism.py:650`), grad-ckpt on, ple_cpu 1 | OJ-081; DECISIONS 2026-11-20/24 (no bf16/fp8 on T4) |
| Multi-GPU | torchrun DDP, `--sync-mode ddp` (`DDP=1` baked by `build.sh`, `entry.py:96-104`) | r7: DDP 2.621 vs local SGD 2.643 (local SGD never averaged in the bench) |
| torch.compile | off (`train_prism.py:132`) | r8: +1.5%, inside noise, 239 s compile; reduce-overhead unsupported |
| quant / option sub-sample | none / 0 | r6: nf4/int8 gave no speed. OJ-072 is not quality-validated |
| SFT | `--algo sft`, lr 1e-4, head-lr 1e-3, warmup 3% then cosine (`train_prism.py:81-83,647`) | `entry.py:123` |
| RL | `--algo exact --reward brier`, lr 2e-5, head-lr 1e-4, lam_kl 0.02, lam_cal 0.5 (soft ECE), init = `out/sft/best`. **KL reference = the frozen SFT policy** (user 2026-11-24, Q4): `train_prism.FrozenRef` snapshots LoRA + flag head + template delta as loaded from `--init` and swaps them in under no_grad for the reference forward | `entry.py` `ALGO`, `train_prism.py` `FrozenRef` |
| Eval / select | every 200 steps on the 800-record seeded val subset (100 per source, `load_val` `train_prism.py:540`). `best/` = lowest val nll | `train_prism.py:732-749` |

**RL cost per sample (from the code).** `exact` runs one forward+backward plus one no-grad reference forward with the frozen SFT weights swapped in, because lam_kl > 0 (`train_prism.FrozenRef`). With grad-ckpt (forward ≈ ¼ of fwd+bwd) that is **~1.25× SFT, est.** `grpo` draws its k=8 samples from **one** set of logits (`train_prism.py:313`), so k adds no forwards. Its cost is 1 sample forward + 1 reference forward + 2 PPO fwd+bwd (`:686-689`), **~2.5× SFT, est.** GRPO is not used in the run.

## 2. Time budget (one 12 h session, hard stop target 11 h)

Measured: DDP 2×T4 runs **2.62 weighted samples/s** (0.38 s/sample) on the bench set (mb 8, max_state 2048, unpacked). A staggered 2-rank model load takes **423 s** per launch (r8 `group_load_s`). Real training should be somewhat faster: grouping pads 3.1% vs ~20% in the bench buckets, and max_state 1024 cuts helpsteer3. Two things could make it slower: data loading and GradScaler skips. So the plan uses three rates. RL rate = SFT rate / 1.25. Proposed steps: **SFT 1,800 (57,600 samples), RL exact 400 (12,800 samples).** **Superseded (user 2026-11-24):** two sessions, SFT 2,200 + RL 1,000 at max_state 2048 (see the two-session bullet below); the one-session table is kept for reference.

| Block (minutes, est.) | 2.2/s (pessimistic) | 2.6/s (bench) | 3.0/s (optimistic) |
|---|---:|---:|---:|
| Kaggle start + pip | 5 | 5 | 5 |
| Preflight: 3 launches × load + 2 steps (`entry.py:112-120`) | 25 | 25 | 25 |
| SFT load | 7 | 7 | 7 |
| SFT train, 1,800 steps | 436 | 369 | 320 |
| SFT evals, 9 × ~5 min (rank 0 only, 800 records) | 45 | 45 | 45 |
| RL load | 7 | 7 | 7 |
| RL train, 400 steps | 121 | 103 | 89 |
| RL evals + final saves + copies | 15 | 15 | 15 |
| **Total** | **661 (11.0 h)** | **576 (9.6 h)** | **513 (8.6 h)** |

- Data coverage: 57,600 SFT samples are 0.14 of one pass over the train set. Per source this ranges from 0.11 epoch (prm800k) to 0.36 epoch (banking77). RL covers 0.02 to 0.08 epoch. One full pass would take ~43 h (est.), so the run is far from one epoch.
- The full-val and test evals and the calibration baselines run in a **separate eval session** of ~3 h (est., single-process per GPU, see §4). They must not run inside the DDP job: rank 1 waits at `jdist.barrier()` (`train_prism.py:750`), and the job dies after `OJ_DIST_TIMEOUT_MIN`=60 (`openjev/dist.py:68`).
- **Two-session variant — CHOSEN (user 2026-11-24, Q3). The resume port (P1) is done.** No in-session preflight: the `smoke` profile is the pre-flight.
  - S1 runs preflight + SFT 2,200 steps: ~9.1 h at the bench rate, ~10.5 h pessimistic.
  - S2 runs RL 1,000 steps + the eval session: ~8.5 h (est.).
  - Choose this only if you want more SFT and RL than one session holds.
- GPU quota: the smoke run (~0.7 h) + run (≤11 h) + eval session (~3 h) is about 15 h of the ~30 h weekly quota. The two-session variant is ~20 h. **Verify** whether 2×T4 counts as 1× or 2× against the quota.

## 3. Checkpoints and resume

- **Done (P1, OJ-061 + B1.3; branch ws-prelaunch), ported from the fork (`openjev/resume.py`):**
  - `--save-every 100` (steps) writes `<save-dir>/resume/step_*` (written as `.tmp`, then renamed; the last 2 kept): trainable params, AdamW, LambdaLR, GradScaler, the Mixture/LengthGroupedStream position, RNG, step, best val nll. `best/` (val nll) and `last/` as before.
  - `--max-minutes N`: once N minutes have passed since `main()` started, the run finishes the current step, writes a resume checkpoint and `last/`, and exits 0 without `DONE.json`. Rank 0's clock decides for every rank (no DDP hang).
  - `--resume auto|<path>|off` (default off): auto loads the latest `resume/step_*` (fresh start if none) and exits at once if `DONE.json` exists. The trajectory config (lr, steps, batch, packing, grouping, seed, ...) must match.
  - DDP: rank 0 writes, every rank loads in place, then a barrier. `--resume` refuses `--sync-mode local_sgd` (per-rank optimizer state).
  - Gate passed (`tests/test_resume_sessions.py`): 6 steps == 3 + resume + 3 (per-step loss and final trainables), SFT and RL exact with `--init`, 1 process and 2 gloo ranks; `--max-minutes` stop + resume, 1 and 2 ranks.
  - Every stage writes straight into `/kaggle/working/out/<stage>` (`--save-dir`, TensorBoard under `<stage>/tb`), so every save is in the kernel output at once. Per-step `train_log.jsonl` (t, seq_len, grad_norm, GradScaler scale, loss terms).
  - `entry.py`: each stage gets `--max-minutes` = `SESSION_MINUTES` (620, baked by `build.sh`) minus the minutes the kernel has already used. Worst-case end: ~620 + one eval + saves ≈ 630 min < 720.
- **After a Kaggle kill:**
  - Push a new version with the killed output attached: `EXTRA_KERNEL_SOURCES=<user>/<kernel> PROFILE=... DDP=1 ./build.sh`. `entry.py` `restore()` copies the newest attached `out/<stage>` into `/kaggle/working/out/<stage>`, and `--resume auto` continues from its latest `resume/step_*`.
  - `full_rl` attaches the `openjev-prism-sft` output (kernel_sources) and copies its `out/sft/best` to `out/sft/best` (`--init`). It stops with an error if no SFT `best/` is attached.
  - **Verify on Kaggle:** whether a kernel may attach its own earlier output (else push under `KERNEL_ID=<new id>`), and whether `/kaggle/working` survives a 12 h timeout. Outputs did survive an ERROR exit in r8.

## 4. Eval protocol

1. **In-run:** the 800-record val subset is used only to pick `best/`. Because of that selection bias, it is not the acceptance set. At 100 records per source, 2 SE is about ±10 points.
2. **Final (eval session):**
   - Run SFT `best/` and RL `best/` on the **full val files** (23,674 records, one model per GPU, length-sorted batches).
   - Then run on a stratified test subset of ≤1,000 records per source (~7.6k records). Run full test (46,754 records) later if needed.
   - Report acc, nll and 15-bin ECE per source and overall.
   - The frozen 400-record set (sha `1de87bf50aa8cca1`) stays the Optuna quality set. It is not used here.
3. **±2 SE rule:** SE = √(p(1−p)/n) per source and overall, plus a paired SE of the per-question difference between RL and SFT on the same items.
   - RL passes acc if its overall acc ≥ SFT − 2 SE.
   - RL passes ECE if its overall ECE < SFT's ECE, and the per-source results are listed.
4. **Calibration baselines (OJ-066):** temperature scaling and Platt scaling, fit on val and applied to test.
   - Compare RL against SFT+T as well as raw SFT.
   - RL failing to beat SFT+T on ECE is reported as is.
   - The frozen-backbone+head baseline is optional, if time allows.
5. **Code gaps (P1):**
   - `evaluate()` gives acc only per source; nll and ece are global (`train_prism.py:521-536`).
   - `eval.py` and `calibrate.py` load `openjev_config.json` through `openjev.checkpoint`, so they cannot load a Prism checkpoint (which writes `prism_config.json`).
   - Needed: a Prism eval path that dumps per-item probabilities, per-source nll/ece, and T/Platt fits.

## 5. Pre-flight checklist

1. ~~Land P1: resume port, per-stage `--max-minutes`, save every 100 steps, stage profiles in `entry.py`~~ **done (ws-prelaunch)**. Per-source nll/ece and the Prism eval + calibration script: the parallel eval work.
2. ~~Update `entry.py` profiles to bs 8 / accum 4 / eval-every 200, packing per Q2~~ **done**: `smoke`, `full_sft` (2,200), `full_rl` (1,000); packing off.
3. ~~`pytest` green; tiny-model CPU smoke SFT → RL → resume~~ **done**: full suite, plus `entry.py` itself simulated on CPU (tiny Gemma 4, `/kaggle` paths mapped): smoke (SFT 30, RL 5 + resume + 5), then full_rl killed by `--max-minutes` and resumed in a new "session" from the attached output.
3b. ~~Template PLE rows saved and restored (blocking: SFT→RL and train/eval mismatch)~~ **done (2026-11-24, ws-ple-persist)**. Gemma 4 resizes its PLE table for the 7 template tokens, and those tokens read their own frozen PLE rows. `train_prism.save` now writes `ple_rows.pt` (`{str(id): row}`, the table's dtype), and `build(--init)` pastes it back after `place()` and the dtype convert. So RL starts from exactly the SFT rows, and `eval_prism.load_prism` matches the training-time logits for any seed. The value of the rows does not change: a fresh SFT build still draws mean + seeded noise in `add_special_tokens`. The PEFT save now passes `save_embedding_layers=False`, so `best/` no longer stores the whole resized embedding (~0.8 GB on E2B). Old checkpoints without `ple_rows.pt` warn: `build(--init)` gets random rows, and `load_prism` replays the seeded build. **The smoke run started before this fix must be re-run**: its SFT `best/` has no `ple_rows.pt`, so its RL half started from different template PLE rows. Tests: `tests/test_ple_rows.py`.
4. `PROFILE=smoke DDP=1 ./build.sh && ./push.sh`, then run the smoke profile on Kaggle 2×T4: SFT 30 + RL 5 + resume + 5 steps, **~40 min est.** (3 staggered loads ≈ 21 min). Read the `report` lines (median s/step, GradScaler skips) and `smoke_rl_kl_step0` (must be ~0).
5. Recompute the step counts from the measured s/step so that the pessimistic case ends ≤11 h.
6. Record the dataset version and the git sha in the run notes, and commit before the push. Open a GitHub issue for the run (standing OK for `piyush-infocusp/openjev` issues).
7. User OK, then launch.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| OOM on long helpsteer3 batches | Peak 8.90 GB at mb 8 for the ≤2048 bucket (r8), out of 15 GB; with max_state 1024 it is lower. Grouping puts the longest records in the same micro-batch, and the short run must hit at least one ≤1024 group |
| fp16 overflow or many GradScaler skips | r8 skipped 4 of 5 early steps (scale 65536→4096; normal). The run aborts after 20 non-finite steps (`train_prism.py:712`). Watch `grad_norm` and the skip count in TensorBoard |
| DDP hang or a dead rank | `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` and the 60 min timeout end the job, which is why resume is needed. Rank-0 eval stays at ~5 min, far under 60 min |
| KL reference | Fixed (Q4): the frozen SFT policy (`FrozenRef`); KL is 0 at RL step 0 and the reference never moves (tested under an enabled autocast, so the fp16 cast cache cannot leak live weights into the reference) |
| Slower than estimated | The per-stage `--max-minutes` guard stops cleanly with lr not fully annealed. Re-plan the steps from the short run |
| Kaggle queue (jobs QUEUED up to 2 h today) | Queue time is not run time, so launch early in the day. A 2 h delay does not change the 11 h budget, only when results arrive |
| Kaggle timeout drops outputs | The stage guards end at 10.8 h, and each finished stage is copied into `out/` at once |

## 7. Launch commands (after the user's OK)

```bash
cd ~/openjev && export KAGGLE_USERNAME=<user>
.venv/bin/python -m pytest -q tests
# Pre-flight (~40 min est.): new dataset version (code), then the smoke kernel.
PROFILE=smoke DDP=1 ./kaggle/prism/build.sh && ./kaggle/prism/push.sh
.venv/bin/kaggle kernels push -p build/kaggle_prism/train                # openjev-prism-smoke
# Session 1: SFT 2,200 steps (re-check the steps against the smoke s/step first).
PROFILE=full_sft DDP=1 ./kaggle/prism/build.sh && SKIP_DATASET=1 ./kaggle/prism/push.sh
.venv/bin/kaggle kernels push -p build/kaggle_prism/train                # openjev-prism-sft
# Session 2 (after S1 is COMPLETE): RL exact 1,000 steps, attaches the openjev-prism-sft output.
PROFILE=full_rl DDP=1 ./kaggle/prism/build.sh && SKIP_DATASET=1 ./kaggle/prism/push.sh
.venv/bin/kaggle kernels push -p build/kaggle_prism/train                # openjev-prism-rl
# A killed session: attach its output and push the same profile again (or under KERNEL_ID=<new id>).
EXTRA_KERNEL_SOURCES=$KAGGLE_USERNAME/openjev-prism-sft PROFILE=full_sft DDP=1 ./kaggle/prism/build.sh
.venv/bin/kaggle kernels status $KAGGLE_USERNAME/openjev-prism-sft
.venv/bin/kaggle kernels output $KAGGLE_USERNAME/openjev-prism-sft -p runs/final/sft
```

## Open questions for the user

1. Port resume and the time guards first (P1, ~1 day est.), or accept a single session with no mid-stage resume?
2. Packing: off (grouped, 3.1% pad, matches the bench) or on (DECISIONS 2026-11-20)?
3. One session (SFT 1,800 + RL 400) or two sessions (SFT 2,200 + RL 1,000)?
4. KL reference: keep the base model (current code) or freeze a copy of the SFT adapter as the reference?
5. Test eval on a ≤1,000-per-source subset, or the full 46,754 records (~4 h more, est.)?

**Answers (user, 2026-11-24):** 1 port first (done); 2 packing off; 3 two sessions (SFT 2,200 + RL 1,000); 4 KL reference = the frozen SFT adapter. 5 is still open.

## Final eval

Tools: `eval_prism.py` (per-source acc ±2 SE, nll, 15-bin ECE, Brier; per-item dump) and `calibrate.py` (temperature + Platt per question type, fit on val, applied to test). Both run as **one process per GPU**, never under torchrun (they exit if `WORLD_SIZE` > 1), fp16 on T4 (`--dtype auto`), batches grouped by encoded length, longest first. Neither one writes into the checkpoint dir. The Kaggle `final_eval` profile in `kaggle/prism/entry.py` belongs to the entry.py owner. It should run exactly these lines, with `CKPT_SFT=out/sft/best` and `CKPT_RL=out/rl/best` (or the attached input of the training kernel).

```bash
# GPU 0: SFT best/, GPU 1: RL best/, in parallel (one model per GPU)
for spec in "0 sft $CKPT_SFT" "1 rl $CKPT_RL"; do set -- $spec
  ( CUDA_VISIBLE_DEVICES=$1 python eval_prism.py --ckpt $3 --data "data/*.val.jsonl"  --max-per-source 0    --out out/final_eval/$2/val  &&
    CUDA_VISIBLE_DEVICES=$1 python eval_prism.py --ckpt $3 --data "data/*.test.jsonl" --max-per-source 1000 --out out/final_eval/$2/test &&
    python calibrate.py --val-items out/final_eval/$2/val/items.jsonl --test-items out/final_eval/$2/test/items.jsonl --out out/final_eval/$2/calib
  ) > out/final_eval/$2.log 2>&1 &
done; wait
# paired RL - SFT on the same test items (acc with paired 2 SE, ece, nll; per source and overall)
python eval_prism.py --compare out/final_eval/sft/test/items.jsonl out/final_eval/rl/test/items.jsonl --out out/final_eval/rl_vs_sft
```

(`mkdir -p out/final_eval` first.) Outputs:
- `<stage>/{val,test}/metrics.{json,md}`: per-source table and the overall row.
- `items.jsonl`: one line per labelled question with id, source, type, name, target and flag logits.
- `<stage>/calib/calibration.{json,md}`: ece and nll per source, raw → T → Platt, on val (the fit split) and on test.
- `rl_vs_sft/compare.{json,md}`.

Optional knobs, not tuned on a T4: `--bs 32 --max-batch-tokens 16384`. Lower them on OOM. For the full test set, use `--max-per-source 0`. The calibration step reads only the dumps, so it needs no second model forward.

**Loader note (fixed 2026-11-24, see §5 item 3b).** Adding the template tokens resizes the Gemma 4 PLE table, and the template tokens read their own PLE rows. `train_prism.save()` writes them to `ple_rows.pt`, and `train_prism.build(--init)` pastes them back, so `eval_prism.load_prism` is exact for any seed. The seeded-replay rebuild remains only for checkpoints saved before the fix, and it warns.
