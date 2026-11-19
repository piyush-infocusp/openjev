# LLM-as-Jev (arXiv 2610.02076v3, Li and Wagle, Microsoft; PDF ../papers/2026-llm-as-jev.pdf, 39 pages)
- "Jev's architecture and training methodologies remain proprietary". Framework: architecture-preserving, training-free readout + fine-tuning with KL anchors. Models: Qwen3.5-4B, Qwen3-0.6B.
- Readout: options as bracketed numeric ids, candidate suffix scored by trie: s_k = sum_t log p(c_k,t | x, c_k,<t); P(k|x) = softmax(s); "no temperature scaling or permutation averaging". Handles Banking77 (77) and CLINC150 (151) in one pass.
- Table 1: 4B training-free JevBench 81.4% acc, ECE .057; 4B LoRA 84.0%, ECE .050; Winnow-12B 86.6, .058; SemIf 80.5, .061; reflex 4B 79.7, .070; 0.6B training-free 56.7, ECE .278; 0.6B LoRA 61.0, .121. GPT-5.6 Sol (generation) 94.4.
- "the public JevBench leaderboard already lists over 60 community Jev-style models". JevBench: 231 public items (MIT).
- Training data (Table 3): CLINC150 (CC BY 3.0), MASSIVE en-US (CC BY 4.0), Bitext (CDLA-Sharing-1.0), CommonsenseQA, HellaSwag (MIT), CosmosQA, Social IQa, ContractNLI, MuSiQue, QuALITY (CC BY 4.0), VitaminC (CC BY-SA 3.0), ReClor/LogiQA/QuAIL (non-commercial).
- Overconfidence: ECE rises 0.057 to 0.134 in one config; format-balanced data fixes yes-bias.
Relevance: baseline to beat; the KL-anchor idea and multimodal readout.
