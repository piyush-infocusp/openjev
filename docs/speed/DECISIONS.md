# Speed-experiment decisions

This file records each design decision for the OpenJev speed work: what was decided, who decided
it, why, and the evidence. Results go in `EXPERIMENTS.md` (Optuna trials) and `SPEED_LOG.md`
(techniques). Todos go in `TASKS.md`.

| Date | Decision | By | Why / evidence | Status |
|---|---|---|---|---|
| 2026-11-19 | The Optuna ask/tell loop on short (~50 s) Kaggle trials, with one model load per group | user + Claude | Each full run took 4.5 s/sample and could not fit in 12 h; model load is ~7 min | done (e019f00) |
| 2026-11-19 | The flag mask: a flag sees its sibling **flags**, not sibling option tokens | Claude (OJ-080) | The old mask made all flags of a question identical → v7/v8 near-uniform predictions | done (6df0efe) |
| 2026-11-19 | fp16 + GradScaler, never emulated bf16 on T4 | Claude (OJ-081) | `is_bf16_supported()` counts emulation, so runs used slow emulated bf16 | done (452e438) |
| 2026-11-19 | Load in bf16, then convert per tensor | Claude | r2 fp16 load was SIGKILLed at 270 s (host RAM held two model copies) | done (2cf488f) |
| 2026-11-19 | `bs` removed from the search; **fixed micro-batch 8**, global batch 32 (accum 4) for every trial | user (+ Claude chose 8 over 16) | bs dominated samples/s and changed the optimization; 16 likely out of memory; a fixed 8 keeps rounds comparable | in progress (round-3 agent) |
| 2026-11-19 | A frozen trial stream (~600 records, stratified) and a frozen probe set for all trials | user | Speed must be compared on the same sequence lengths | in progress |
| 2026-11-19 | A larger diverse val set (~400 records, stratified by source / type / option count), validation every 30 trials and at round end, with 150-step best vs baseline training | user (+ Claude added the short training) | 50 s trials give no quality signal; accept a speedup only if val NLL is within +0.05 and acc within −1 pt | in progress |
| 2026-11-19 | All fixed conditions in the Optuna study note and attrs, with a `fixed_conditions_version` | user | Every trial must be self-describing | in progress |
| 2026-11-19 | `ple_cpu` is **pending**, not fixed: a paired test at mb 8 in round 3 | user question → Claude | The r1 "+23% on GPU" came from 3 trials under the old mask and bf16; GPU PLE may not fit at mb 8 | pending |
| 2026-11-19 | A compact PLE table: keep only the rows of token ids that occur in the data, and remap ids | user idea (OJ-085) | Saving the PLE output per token would need ~3.6 TB (18 KB/token × ~200M tokens); a compact table is exact and estimated at ~1.1 GB vs 4.7 GB | todo, size being measured |
| 2026-11-19 | Multiple Kaggle accounts: not used | Claude | Kaggle allows one account per person; quota evasion risks a ban of the main account | rejected |

## Measurements behind the decisions

- **Compact-PLE vocabulary count:** pending (sonnet job). The distinct token ids per split and source, and the table size, will be written below.
