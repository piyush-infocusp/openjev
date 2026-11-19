# Sarvam Prism (blog, https://www.sarvam.ai/blogs/sarvam-prism, fetched 2026-11-19)
Source is a web-fetch summary of the page (quotes as returned by the fetch tool; re-check wording on the page). No paper found.
- "Today we are releasing Sarvam Prism, a System One decision model built on Sarvam 105B." Released Nov 3, 2026. Weights on Hugging Face, Apache 2.0; also via Sarvam Indus API.
- "a LoRA fine-tune of Sarvam 105B" (MoE, 105B-10B base). One forward pass, "a small scoring head outputs a probability per option".
- Mask: state+question block read bidirectionally; "Each option sees that block and itself but never the other options." Options share a start position. "Only the answer token sees everything, and that is where the options are compared." Bidirectional block adds about 3 points on the Jev Decision Index.
- Fixed option order before scoring, probabilities returned under original names: 0 answer changes on 2,000 shuffled questions; 4.0% changed without it.
- Loss: two equally weighted terms, cross-entropy with 5% label smoothing and a Brier loss. Training data: not disclosed.
- Latency: median 44 ms on one B200 (from 664 ms); FP8 (Jev score changed 0.01), grouped expert kernel, CUDA graphs, LoRA merged. Split attention kernel 2.6x faster over 2,048 tokens. Up to 131,072 tokens.
- Scores: average over 3 indices 71.8 at 44 ms (mammoth-1 75.2 at 284 ms). Jev Decision Index 67.0; Artificial Analysis recreation 65.9; Indian languages 82.5 (20,108 Qs, 10 languages + Hinglish).
- Calibration at >=90% confidence, English (226,088 Qs): Prism predicted 3.3% errors, made 2.2%; Jev 1.14 predicted 1.3%, made 3.5%. Indian: predicted 5.4%, made 4.4%.
- Agent traces (AgentRewardBench, tau-bench, Who&When, TRAIL; 700 runs, 1,770 Qs): task success 75.0, human review 86.4, failure agent 69.6, all 78.5; Haiku 5.5 70.7 overall.
- Security: 89.0% internal benchmark (5,517 Qs, 9 tasks) vs Clef 84.5, Jev 1.14 78.6.
Relevance: its mask + answer-token compare is what OpenJev's flag copies. Contrast with prism1-ntu.md which finds sibling attention costs accuracy.
