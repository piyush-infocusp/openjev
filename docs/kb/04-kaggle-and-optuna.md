# Kaggle and Optuna

- Kernels: piyushsar/openjev-prism-optuna (speed bench), piyushsar/openjev-prism-train (training).
  Dataset: piyushsar/openjev-prism-data (jsonl splits + openjev_src.zip with the code, trial stream, val set).
- CLI: ~/openjev/.venv/bin/kaggle (2.2.4). There is no stop command; the SDK has CancelKernelSession
  (needs scriptVersionId). Live logs: `timeout 50 stdbuf -oL kaggle kernels logs -f <kernel>`.
- Optuna DB: ~/openjev/runs/optuna/study.db (backups study.db.bak-*). Studies: openjev-speed-v2 (fair),
  "LEGACY-r1-r2 (...)" (old). Never query it with SQL; use tools/optuna_loop.py or the optuna API.
- Round = propose → `PROFILE=optuna KAGGLE_USERNAME=piyushsar bash kaggle/prism/build.sh` →
  `bash kaggle/prism/push.sh` (wait for "ready") → `kaggle kernels push -p build/kaggle_prism/train` →
  `kaggle kernels output ... -p runs/optuna/round-<ts>` → `optuna_loop.py ingest ... --tag rN` → report.
- Fair rules: micro-batch 8, global batch 32 (accum 4), frozen trial stream (sha 8296cda5) and
  val set (sha 1de87bf5), max_state_tokens fixed at 1024, bs/dtype/ple_cpu not tuned, 4 trials/round,
  timed_steps 2 (~7 min/trial, round ~30–40 min). Validation every 30 trials (150-step best vs baseline).

Round history: r1 (legacy, old mask) best 0.234 sps; r2 crashed twice (host-RAM SIGKILL);
r3 big crashed (host fp16 convert, 10.2 GB); r3 small (v5): only #14 ran, 3.6×;
r4 (v6) running.
