# Evaluating and Benchmarking Jev (arXiv 2609.37647v2, Deusser et al., Bonn; PDF ../papers/2026-jev-eval.pdf)
- jev-1.13.0 zero-shot on 37 datasets, 346,009 requests for under US$10. Baselines Qwen3.8-27B and Gemma-4-E4B (exact option probabilities). Jev beats Qwen on 27/37 and Gemma on all 37.
- Calibration: "Pooled over 22 Choice datasets and 279,925 answers, the ECE is 0.028"; Qwen 0.063, Gemma 0.208. Noul: Jev ECE 0.052 (underconfident for yes); binary needs tuned thresholds (UNFAIR-ToS micro-F1 0.50 to 0.75).
- Selective prediction at 50% coverage: Banking77 79.7 -> 96.3; SIB-200 81.5 -> 98.2; ANLI 73.9 -> 86.9; Emotion 58.5 -> 73.3.
- Banking77 79.7 (ECE .087 Jev), CLINC150 89.5.
- Dataset list with HF ids (Table 2 of paper): AG News, IMDB, Rotten Tomatoes, SST-2, Emotion (dair-ai/emotion), Financial PhraseBank, SMS Spam, Language ID, GoEmotions (noul x28); Banking77 (mteb/banking77, 3,076), CLINC150 (clinc/clinc_oos, 5,500), SIB-200; ANLI, AfriXNLI, PAWS, LLM-AggreFact (lytang/LLM-AggreFact, 29,320); BoolQ, Belebele, PubMedQA, MMLU, C-Eval; BIG-bench MC, HellaSwag, WinoGrande, ARC, CommonsenseQA, alphaNLI; ToxiGen, OpenAI Moderation, ToxicChat (lmsys/toxic-chat, 5,083), Prompt Injections (deepset/prompt-injections, 116), AGB-DE, UNFAIR-ToS; STS-B, SST-5 (SetFit/sst5), SummEval (mteb/summeval), HelpSteer2 (nvidia/HelpSteer2).
- Request JSON shape in paper matches OpenJev's format: {"model","state":{"text"},"questions":{name:{"type":"noul"|"score"|"choice","instructions"}}}.
Licences were not tabulated here (unverified); see llm-as-jev.md Table 4 for some.
