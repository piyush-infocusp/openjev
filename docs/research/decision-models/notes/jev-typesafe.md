# TypeSafe Jev docs (https://docs.typesafe.ai, fetched 2026-11-19)
- "Jev evaluates typed questions against a state and returns structured results directly." Request: one state (text, JSON or images) + any mix of questions; each evaluated in parallel and in isolation against the same state.
- Types: noul -> `noul` (0-1); choice -> `choice`, `probabilities`, `confidence`; score -> `score`, `probabilities`, `confidence`; box -> top/bottom/left/right + `confidence`.
- confidence.md: "Confidence measures how concentrated the answer's probabilities are." Choice: (p_max - 1/n)/(1 - 1/n). Score: probability-weighted distance from the modal level vs even spread, floored at 0. Noul has no confidence; suggested |2p-1|. Suggested noul thresholds 0.7 / 0.3 with a middle band.
- Docs do not claim calibration, model sizes, latency numbers or dates (unverified elsewhere). Independent evaluation: see jev-eval-bonn.md.
Important: Jev's "isolated questions" means questions do not see each other, which OpenJev's schema encodes in the shared block + per-question options.
