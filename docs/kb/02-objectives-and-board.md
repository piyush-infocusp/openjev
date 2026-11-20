# Objectives (user, 70/30)

- **A, main MVP (70%):** one full SFT→RL run of Gemma 4 E2B + Prism mask on Kaggle 2×T4
  in ≤12 h, with no error. It reports per-source val acc/nll/ece. RL must not be worse
  than SFT on acc (±2 SE) and must be better on ECE. The final run needs the user's OK.
- **B, XLA fork (30%):** CUDA/TPU/CPU from one codebase, cross-backend resumable checkpoints,
  a measured TPU speed, a ~50 h chained GPU+TPU plan, a minimal JAX prototype. TPU quota
  only; the first TPU push needs the user's OK.

Full board with gates: ../TODO_BOARD.md (WS-A1..A8, WS-B1..B3).
Order of the speed work now: packing (A2) → hot/cold PLE (A3) → DDP (A4) → QLoRA/compile.
