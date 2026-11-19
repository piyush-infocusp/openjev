# OpenJev plan — a Jev-compatible causal encoder under 1B parameters

Goal: one model with **fewer than 1B total parameters** that reads a Jev request
(`state` + typed `questions`) and returns the Jev answer object with **calibrated**
probabilities in one forward pass. The public format never changes (see `CLAUDE.md`).

Work items live in `TASKS.md` (kanban board, ticket ids `OJ-xxx`).

---

## 0. Backbone selection

Research date: 2026-09-24. Sources: Hugging Face model cards.

| Candidate | Total | Active | MoE | License | Verdict |
|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 0.6B | 0.6B | no | Apache-2.0 | **Primary backbone** |
| `google/gemma-3-270m` | 0.27B | 0.27B | no | Gemma (custom) | Fast / small variant |
| `HuggingFaceTB/SmolLM2-360M` | 0.36B | 0.36B | no | Apache-2.0 | Fallback if Qwen fails |
| `ibm-granite/granite-3.1-1b-a400m-base` | ~1.3B | ~0.4B | yes | Apache-2.0 | Over budget (total > 1B) |
| `allenai/OLMoE-1B-7B-0924` | 6.9B | 1.3B | yes | Apache-2.0 | Over budget |
| `LiquidAI/LFM2-8B-A1B` | 8.3B | 1.5B | yes | LFM (custom) | Over budget |

**Finding:** no public pretrained MoE has fewer than 1B *total* parameters. All
small MoE models are small only in *active* parameters.

**Decision:**
1. Train on **Qwen3-0.6B** (dense) first. Reasons: Apache-2.0, native `transformers`,
   a 151k vocab with unused special-token slots for our 8 template tokens, and
   Qwen already ships Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B on the same
   backbone, so it works well as a reader of hidden states. The CPU smoke-test
   twin `trl-internal-testing/tiny-Qwen3ForCausalLM` has the same architecture.
2. **MoE under 1B by sparse upcycling (experiment, phase 4).** Copy the MLP of the
   last N Qwen3-0.6B layers into E experts with a top-1 router (Komatsuzaki et al.
   2022, "Sparse Upcycling", arXiv:2212.05055).
   One Qwen3-0.6B MLP is about 1024×3072×3 ≈ 9.4M parameters. With E = 4 experts in
   the last 8 of 28 layers, the model has 0.6B + 3×9.4M×8 ≈ **0.83B total, 0.6B
   active**. This is a real MoE under 1B. It ships only if it beats the dense
   model on the eval set at the same active compute.
3. Gemma-3-270m is the "tiny" SKU if the latency budget needs it (license check first).

---

## 1. Phase 1 — Dataset

Converters already exist for banking77, clinc150, support_tickets, helpsteer3,
lichess_puzzles, pokerbench (`converters/`, output in git-ignored `data/`).

Steps:
1. Re-run every converter on the **train** split with a fixed seed, and write
   `data/<source>.{train,val,test}.jsonl`. Use the real HF splits when they exist.
   Otherwise split by a hash of the row id (deterministic, no leakage).
2. `scripts/data_stats.py`: rows, question-type counts, option-count histogram,
   label balance, state token length (Qwen3 tokenizer), and a leakage check
   (the answer text must not occur in `state`).
3. `openjev/mix.py`: a mixed sampler with per-source weights (temperature
   sampling, `w ∝ n^0.5`) so that large sources do not drown small ones.
4. Missing converters: prm800k (score, 3 levels) and diplomacy deception (noul).
5. Pack the mixed data as a Kaggle Dataset (`kaggle datasets create`).

Rare labels (for example Diplomacy lies at 4.5%): do **not** use class weights or
resampling inside a source. Both move the predicted probabilities away from the true
rates, and calibration is the product. Judge these sources by Brier and NLL (and
AUROC for noul), not by accuracy.

Rules: never invent rows or labels, build options from the real data, shuffle
option order, and never leak the answer into `state`.

## 2. Phase 2 — Free GPU (Kaggle CLI)

- Kaggle gives about 30 GPU-hours per week (P100 16 GB or 2×T4 16 GB), with sessions of
  about 9–12 h. **Check both numbers** at https://www.kaggle.com/docs/efficient-gpu-usage.
- T4 and P100 have **no bf16**. Use fp16 autocast + `GradScaler`, with the fp32 master
  weights. Qwen3-0.6B full fine-tune memory: weights 2.4 GB + grads 2.4 GB + AdamW 4.8 GB
  ≈ 10 GB + activations. This fits 16 GB with gradient checkpointing and ≤1024 tokens.
- `kaggle/` folder: `kernel-metadata.json` (`kernel_type: script`, `enable_gpu: true`,
  `enable_internet: true`, `dataset_sources: [<user>/openjev-data]`), a launcher
  `kaggle/run.sh` (push → poll `kaggle kernels status` → `kaggle kernels output`).
- Do not rely on `machine_shape`. Known CLI issues say that it can silently pick the default GPU.
- Alternatives: Modal (free credits, CLI) and Lightning AI (CLI). Colab cannot run from the CLI.
- Credentials: `~/.kaggle/kaggle.json` (the user makes it; the agent never handles the key).

## 3. Phase 3 — Supervised tuning with calibrated scores (SFT)

`train.py` (replaces the toy loop):
- Batched collate with one tensor gather for all flag positions (`openjev/collate.py`).
- Losses per type (all proper scoring rules, so the minimum is the true rate):
  - `noul`: 2-flag softmax CE (equal to binary log loss).
  - `choice`: softmax CE over the options, with optional label smoothing ε=0.05.
  - `score`: softmax CE **plus** an ordinal term, because a near miss must cost less than a far miss:
    squared EMD between the predicted and the target CDF (Hou et al. 2016). Option: CORN
    conditional decomposition (Shi, Cao, Raschka, arXiv:2111.08851) for rank-consistent levels.
- Optimizer: AdamW, lr 2e-5 for the backbone and 1e-3 for the heads, warmup 3%, cosine, fp16 AMP, grad-checkpointing.
- Eval every N steps on `val`: accuracy, NLL, Brier, **ECE (equal-mass, 15 bins)**,
  per type and per source; mean absolute error for `score`.
- Post-hoc: one temperature per question type (`rlcd.fit_temperature`) on `val`,
  stored in the checkpoint. Report the ECE before and after.
- Checkpoint: backbone + heads + tokenizer + temperatures (`safetensors`).

Exit criteria for phase 3 (Qwen3-0.6B, held-out `test`): accuracy ≥ a TF-IDF+LR
baseline on every source, and ECE ≤ 0.05 after temperature scaling.

## 4. Phase 4 — RLCD stage (proposal)

"RLCD" has two public meanings. We use both, and we name each one clearly:
- **TypeSafe RLCD** = "Reinforcement Learning for Calibrated Decisions". TypeSafe
  states only the goal (the stated probabilities match the outcome rates). They do
  not publish a method.
- **Yang et al. RLCD** = "Reinforcement Learning from Contrastive Distillation"
  (arXiv:2307.12950, ICLR 2024). Contrasting positive and negative prompts give
  preference pairs with no human labels. A reward model and PPO then use these pairs.

**OpenJev-RLCD** = the goal of TypeSafe, trained with the contrast idea of Yang et al.,
and no text generation (our "actions" are distributions over flags):

1. **Contrastive pairs with known outcomes.** A deterministic transform of a *real*
   labelled record gives a second record with a label that is known exactly. These
   pairs are not invented labels:
   - noul: swap the `true`/`false` criteria text → the target is `p'(true) = 1 − p(true)`.
   - choice: remove a wrong option → the probabilities of the other options keep their ratios.
     Remove the correct option → the question is dropped (no label).
   - score: reverse the level order → `score' = (n−1) − score`.
   - choice/score: shuffle the option order → the distribution must stay the same (permutation).
   The **contrastive consistency loss** is the symmetric KL between the pair after the known
   mapping. It punishes answers that follow the surface form and not the state.
2. **Batch calibration reward** (the current `openjev/rlcd.py`): proper score + λ_cal·soft-ECE
   + λ_kl·KL(p ‖ p_SFT). The KL to the frozen SFT model stops mode dropping.
3. **Outcome rewards without a gradient** (`rlcd.reinforce`): REINFORCE with a
   batch-mean baseline for sources that have a real utility. Examples: poker EV and a
   chess puzzle solved. The reward is Brier-shaped, as in RLCR (arXiv:2507.16806), so a
   correct but hedged answer and a confident wrong answer get the correct ordering.
4. Hard-example mining: select the pairs where the SFT model is least consistent,
   with a replay ratio of 50%.
5. Temperature scaling again on `val`, as the last step.

Ablation table (the same test set): SFT → SFT+T → SFT+RLCD(cal) → SFT+RLCD(cal+contrast) → +T.
Metrics: accuracy, NLL, Brier, ECE, contrast consistency.

Unverified references from the research brief (check before we cite):
arXiv:2607.04332 (reward shapes for calibration RL); Xu et al. 2024 and Stangel et al.
2025 (proper-scoring-rule rewards).

## 5. Phase 5 — Serve and release

`serve.py` (FastAPI, `POST /v1/systemone`, the exact Jev body), the model card, and
eval tables on the HF Hub.

## Parameter budget check (must stay < 1B total)

| Build | Total | Active |
|---|---|---|
| Qwen3-0.6B + heads + template tokens | 596.0M (verified) | 596.0M |
| Qwen3-0.6B upcycled, 4 experts × last 8 layers | 822.6M (verified) | 596.1M |
| Qwen3-0.6B upcycled, 8 experts × last 4 layers | 860.3M (verified) | 596.1M |
| Gemma-3-270m + heads | ≈ 0.27B | 0.27B |
