# Prism-1 (arXiv 2610.22066v2, NTU, CC BY 4.0, 12 pages; PDF in ../papers/2026-prism1.pdf)
Teng, Huang, Cai, Deng, An. Submitted 2026-10-15, v2 2026-10-16. Self-reported results.
- Abstract: "a read satisfies both axioms only if it is a softmax over scores that each depend on one option alone, so in a transformer they hold by construction only if no token that scores an option sees another."
- Sec 4: "Each option o_k becomes a block b_k ... Every block starts at position P ... each block attend[s] to the context and causally to itself, but never to another block. There is no token after the blocks."
- Score (eq 2): s(o_k) = log P(yes | h, b_k) - log P(no | h, b_k) read at the last token of the block; decision = softmax. Prompt ends "Candidate answer:". Trained with CE over options only; no order augmentation.
- Setup (5.1): Qwen3-8B, Gemma 4 12B instruct (thinking off); LoRA r16, alpha 32, attention+MLP, AdamW lr 1e-4, 3% warmup, cosine, 32 q/step, 1 epoch, 3 seeds. 9,100 training questions. Eval sets: Banking20, Newsgroups20, BEV (1,182 Qs), Typed Decisions (600), JevBench (139 public choice tasks, held out).
- Table 1 averages (Gemma): SCQ 80.4 acc, 7.1% flip; Prism-1 79.5, 0 flip; Qwen: SCQ 77.9, Prism-1 78.0. Jev 1.13 acc 80.0, flip 4.7%; Mercury Decide 79.4, flip 7.6%.
- Table 2 ablation (Gemma acc/switch): Prism-1 79.6/0.0; options attend to each other 74.4/3.5; + score read after set 76.4/3.8; Prism-1 + answer slot 79.7/0.0 (591 tokens, 76 ms vs 221 tokens for plain on Banking20).
- Sec 5.5: "Letting the options attend to each other ... costs 5.2 and 2.9 points"; untrained bidirectional set prompt answers 23%/20% correctly vs 68% for isolated causal blocks.
- Calibration (App C.3): "Without any calibration step, Prism-1's softmax has an expected calibration error of 5.3% and 5.4%, against 4.9% and 5.4% for SCQ and 9.3% and 7.5% for Set-LLM."
- Injection: option text "naming another as correct" redirects 3-55% of other reads, none of Prism-1's.
- Cost/latency: 58 and 60 ms on Gemma (20 options) vs 60-63 SCQ, 66-68 Set-LLM; ~half of SCQ's attention pairs.
Caveat for OpenJev: positions are shared, causal inside blocks, so bidirectional state reading (Sarvam) is not tested here. Limitation: no cross-option comparison, e.g. 3.6-4.1 points behind SCQ on held-out JevBench.
