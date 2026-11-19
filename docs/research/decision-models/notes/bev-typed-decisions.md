# BEV Decision Mix and Typed Decisions (HF dataset cards, fetched 2026-11-19)
## avbiswas/bev-decision
- default config 150,000 rows (125,614 train / 24,386 test); `all` 276,188 rows (231,332 / 44,856); 120 MB. Configs: hard_50k, numeric_temporal, skills, counterfactual_15k. No validation split.
- Licence: Hub "unknown"; card: "No single blanket license is asserted for this mixed-source release." Per-source: CC BY 4.0 (MuSiQue, TAT-QA, ContractNLI, MASSIVE, MBPP), CC BY 3.0 (CLINC150), Apache-2.0 (2WikiMultihopQA, CondaQA, Decider-generated), MIT (FinQA, PRM800K), BSD-3 (TimeQA; passages CC BY-SA). "no non-commercial source was added".
- Columns: state, questions_json ({"questions": {name: {...}}}), domain, question_types, question_count. CHOICE: `criteria` map + `label` key; NOUL: boolean label; SCORE: ordered anchored levels, zero-based `label`. => Fits Jev format natively.
- 18 domains incl. sentiment/moderation 25,156; retail 19,883; spatial/logical 19,579; support/intent 12,656; tool/workflow 10,224; browser interaction 4,786; game-state 2,000; software security 7,000.
## LocalLLaMA/typed-decisions
- Apache-2.0, English, parquet, 1K-10K rows (about 1.6k all; ~400 per subset; train 300 / test 100 per subset). Synthetic. Subsets: agent_trace_observability, customer_service, invoice_processing, security_incidents, all.
- Fields: id, workflow, split, state (JSON), questions, gold (label, confidence, probability distribution), factors, n_questions, label_agreement. Types seen: choice (action) and noul (needs_review); score-like numeric risk/urgency.
- Used in Prism-1 as the Typed benchmark (600 test choice Qs, 4-5 options).
Unverified: the dataset card prose on generation method.
