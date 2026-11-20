# Current state (2026-11-20, ~13:40 IST)

- **Focus (user):** finish the infra speed experiments today and prepare for training.
  Everything else is paused.
- **Running:** Optuna round 4 = kernel piyushsar/openjev-prism-optuna **v6**, trials #23–26
  in study openjev-speed-v2: A baseline ple_cpu=1 + profile, B chunked_math ckpt on,
  C sdpa_bool ckpt on, D chunked_math ckpt off. timed_steps 2, mb 8, global 32.
  Watched by agent A1. Caveat: the dataset may have been the old version; check the
  `after_resize` maxrss line (it should be far below 28.9 GB).
- **Best measured:** round 3 small, trial #14: ple_cpu=1, grad_ckpt on, sdpa_default, fp16,
  mb 8 → **0.808 samples/s = 1.24 s/sample = 3.6×** vs the 4.5 s/sample baseline. Peak GPU 12.3 GB.
  pad_frac 0.52 (half the tokens are padding, so packing is the next big gain).
- **Also running:** agent A2 (packing + option sub-sampling, worktree branch), resumed.
- **Paused (rate limit / user focus):** A3 hot/cold PLE, A4 DDP, XLA fork (B), the A7.4
  pytest gate. A7.4 code is committed on its branch (3f061c3) and not merged.
- **Dashboard:** http://127.0.0.1:8080/dashboard/ (study openjev-speed-v2; legacy trials
  17–22 are copied in and tagged not comparable).
