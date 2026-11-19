# Decision models research (run date 2026-11-19)

Scope: LLMs as one-pass option scorers with calibrated probabilities (Sarvam Prism, TypeSafe Jev, and similar), for OpenJev (Gemma 4 E2B, Prism-style mask).
Relevance column: 1 (low) to 5 (high) for OpenJev. "unverified" means only seen in a search snippet or aggregator, not on the primary page or PDF.

## Key finding to read first: two different things are called "Prism"

| | Sarvam Prism (blog, 2026-11-03) | Prism-1 (arXiv 2610.22066, NTU, 2026-10-15) |
|---|---|---|
| Mask | State+question block bidirectional; each option sees block + itself; answer token sees everything and compares options (blog) | Context causal; each option block sees context + itself causally, never another option; **no token after the blocks** |
| Option comparison | At the answer token (options see each other there) | Only a softmax over per-option scores s_k = log P(yes) - log P(no) read at the last token of each block |
| Guarantees | Fixed option order before scoring (0 flips on 2,000 shuffled Qs, blog) | Order invariance and Luce choice axiom by construction (0.0 flips, 0.0 added-option drift) |
| Relation | The blog does not cite the paper and the paper does not mention Sarvam. Different designs, unrelated authors (unverified whether connected). | |

OpenJev's current docstring (flag token that sees siblings) is the Sarvam variant. Prism-1's Table 2 ablation says options attending to each other cost 5.2 / 2.9 points and break the choice axiom, but "Prism-1 + answer slot" (shared question repeated per option) matches accuracy. See notes/prism1-ntu.md.

## Track A: Models and architectures

| Name | Date | Link | Key idea | Rel |
|---|---|---|---|---|
| Sarvam Prism | 2026-11-03 | https://www.sarvam.ai/blogs/sarvam-prism | LoRA on Sarvam 105B MoE, bidirectional state block, isolated options at shared start position, answer token compares; CE (5% label smoothing) + Brier loss, 44 ms on B200; Apache 2.0 weights | 5 |
| Prism-1 (NTU) | 2026-10-15 | https://arxiv.org/abs/2610.22066 | Rival-independent block mask, yes/no score per option, softmax; Qwen3-8B and Gemma 4 12B, LoRA r16; ECE 5.3% vs 4.9% SCQ | 5 |
| TypeSafe Jev | 2026 | https://docs.typesafe.ai | Commercial System One model; noul/choice/score/box; parallel isolated questions; confidence = normalised concentration | 5 (the format) |
| LLM-as-Jev (Microsoft) | 2026-10-01 | https://arxiv.org/abs/2610.02076 | Unmodified causal LLM; options as numeric suffixes, trie-constrained joint scoring; KL anchors vs forgetting; Qwen3.5-4B frozen ECE 0.057, LoRA ECE 0.050 | 4 |
| Jev eval (Bonn) | 2026-10-29 | https://arxiv.org/abs/2609.37647 | Independent eval of jev-1.13.0 on 37 datasets, 346,009 requests; pooled Choice ECE 0.028; Gemma-4-E4B baseline ECE 0.208 | 4 |
| Self-Ensemble | 2025-06 | https://arxiv.org/abs/2506.01951 | Custom mask blocks cross-group attention between answer-choice groups in one pass for calibration (from snippet only, unverified) | 3 |
| Gemma Encoder | 2025-03 | https://arxiv.org/abs/2503.02656 | Gemma-2 2B/9B adapted to encoders; bidirectional beats causal; (snippet only, unverified details) | 3 |
| BidirLM | 2026-04-02 | https://arxiv.org/abs/2604.02045 | Causal LLM to bidirectional encoder: masking (MNTP) phase, weight merging; Gemma3 270M/1B, Omni-2.5B text+vision+audio | 3 |
| T5Gemma 2 | 2025-12 | https://arxiv.org/abs/2512.14856 | Encoder-decoder from Gemma 3, frozen vision encoder, 270M/1B/4B pairs; pretrained only | 2 |
| LLM2Vec | 2024-04 | https://arxiv.org/abs/2404.05961 | Enable bidirectional attention + MNTP + contrastive (older, background) | 2 |
| Set-LLM / Set-Based Prompting | 2024 | cited in Prism-1 | Set positions + set mask; Prism-1 measured it 4.0 / 5.9 points worse and redirectable by injected options | 2 |
| Community Jev clones | 2026 | see notes/llm-as-jev.md | reflex 4B (LoRA+calibration file), Winnow-12B, SemIf (letter logits), Torchcast Decision 12B | 3 |

## Track B: Datasets and benchmarks

| Name | Date | Link | Key idea | Rel |
|---|---|---|---|---|
| Typed Decisions | 2026 | https://huggingface.co/datasets/LocalLLaMA/typed-decisions | ~1.6k synthetic, 4 workflows, choice+noul+score with gold probabilities, Apache-2.0 | 5 |
| BEV Decision Mix | 2026 | https://huggingface.co/datasets/avbiswas/bev-decision | 150k rows (276k "all"), 18 domains, native choice/noul/score JSON; mixed licenses | 5 |
| Jev Decision Index | 2026 | https://huggingface.co/spaces/multimodalart/jev-decision-index | HF leaderboard, 132,422 requests, 37 benchmarks (per aggregator, unverified); Prism blog avg: Jev 67.0 for Prism | 4 |
| JevBench | 2026 | via arXiv 2610.02076 (MIT, 231 public items) | Public hard decision tasks; BenchLM lists v1.5.4 with 1,624 decisions (unverified) | 4 |
| Artificial Analysis Decision Index v1.0 | 2026-10-13 | https://artificialanalysis.ai/evaluations/artificial-analysis-decision-index | AA-Omniscience-Decision, HLE-MultiChoice, When2Call, WildGuardTest; calibration by 0.1 prob bins | 4 |
| Jev 37-dataset suite | 2026 | notes/jev-eval-bonn.md | Banking77, CLINC150, SIB-200, ANLI, ToxicChat, HelpSteer2, SST-5, etc. with HF ids and sizes | 5 |
| When2Call | 2025-04 | https://arxiv.org/abs/2504.18851 | Call tool / ask follow-up / refuse; built on BFCL Live v2 | 4 |
| BFCL v4, RouterBench, TwinRouterBench, Switchcraft | 2025-26 | search snippets only | tool and router choice (unverified licences) | 3 |
| AgentRewardBench, tau-bench, Who&When, TRAIL | 2025-26 | named in Sarvam blog | agent-trace decisions (Prism 75.0 / 86.4 / 69.6 / 78.5) | 3 |
| Mind2Web, AndroidControl, ScreenSpot | 2023-24 | see Track C | GUI element choice | 3 |

## Track C: Visual and audio backends

| Name | Date | Link | Key idea | Rel |
|---|---|---|---|---|
| Gemma 4 E2B/E4B card | 2026 | https://gemma.ai/models/gemma-4/model-card/ | ~150M vision, ~300M audio encoder; 128K ctx; audio max 30 s; Apache 2.0; 35 layers E2B, 512 sliding window | 5 |
| LLM-as-Jev multimodal | 2026-10 | arXiv 2610.02076 | Frozen LLM handles image decisions (MMBench Apache-2.0, MMStar) without changes | 4 |
| Calibrating VLMs | 2024-02 | https://arxiv.org/abs/2402.07417 | 35 VLMs; temperature scaling makes VLMs well calibrated; synthetic calibration set (snippet) | 3 |
| Grounding-based UQ for MLLMs | 2025-05 | https://arxiv.org/abs/2505.03788 | Calibration using grounding (snippet only) | 2 |
| BidirLM-Omni-2.5B | 2026-04 | https://arxiv.org/abs/2604.02045 | Bidirectional text+vision+audio encoder | 3 |
| MINDS-14 | 2021 | https://huggingface.co/datasets/PolyAI/minds14 | 14 banking intents, 14 languages, ~600 clips each, CC-BY-4.0 | 4 |
| SLURP | 2020 | not verified (HF page 401 for Qwen/SLURP) | 60 intents, 18 domains, 16,521 utterances per MASSIVE card | 4 (licence unverified) |

## Top 5 things to adopt

1. **Decide the mask on evidence, add a Prism-1 "isolated" ablation.** Prism-1 Table 2 shows siblings attending to each other lose 3 to 5 points and break order invariance and Luce. Add an isolated-flag mode to prism.py (flag sees shared + own option only, softmax over per-option yes/no or scalar scores) and compare on Typed Decisions + held-out tasks. Keep the Sarvam-style sibling flag only if it wins.
2. **Fix option order and report the flip rate.** Sarvam sorts options into one fixed order (0 changes in 2,000 shuffles vs 4.0% without). Cheap, and it gives deterministic caching. Add a flip-rate test to the eval.
3. **Train on CE (5% label smoothing) + Brier**, as Sarvam does, and report ECE with 15 bins, Brier, and selective accuracy at 50% coverage (Jev eval: 79.7% to 96.3% on Banking77).
4. **Import Typed Decisions and BEV (license-filtered) as training and held-out eval**, since both are already in noul/choice/score JSON. BEV also includes PRM800K, CLINC150, Mind2Web-derived and game-state rows.
5. **Borrow LLM-as-Jev's KL anchors and external-benchmark ECE checks** (freeze a general-ability set, penalise drift), plus its finding that a frozen 4B model is already ECE 0.057, so a LoRA must beat a frozen Gemma baseline, not just a random head. Also use Jev's confidence = (p_max - 1/n)/(1 - 1/n) for the answer field.

## Datasets to add (priority order)

1. LocalLLaMA/typed-decisions: Apache-2.0, ~1.6k (400 per workflow), choice/noul/score, gold probabilities. Eval and calibration set first.
2. avbiswas/bev-decision: 150k default / 276k all. Hub licence "unknown", card says mixed, no non-commercial source added. Use per-source licences (CC BY 4.0 for MuSiQue, TAT-QA, ContractNLI, MASSIVE, MBPP; CC BY 3.0 CLINC150; Apache-2.0 2Wiki, CondaQA; MIT FinQA, PRM800K; BSD-3 TimeQA). Several domains (browser, tool, game, software security) are new to OpenJev. Check upstream licences for SST-5, GoEmotions, Civil Comments, Mind2Web, HelpSteer2 (unverified) before redistributing.
3. MASSIVE en-US intents (CC BY 4.0, 11,390, 60 intents) and Bitext e-commerce (CDLA-Sharing-1.0, 21,839, 27 intents): from LLM-as-Jev Table 3.
4. When2Call (NVIDIA, licence unverified): tool-call-or-not choice.
5. Noul sets: ToxicChat, OpenAI Moderation, deepset/prompt-injections, LLM-AggreFact, PAWS (HF ids in notes/jev-eval-bonn.md; licences unverified except per LLM-as-Jev Table 4).
6. Score sets: SST-5, STS-B, SummEval, HelpSteer2 (OpenJev has helpsteer3 only).
7. Reasoning choice: ANLI, HellaSwag (MIT), CommonsenseQA (MIT), CosmosQA (CC BY 4.0), Social IQa (CC BY 4.0). Avoid ReClor, LogiQA, QuAIL (non-commercial).
8. Eval only: JevBench public 231 items (MIT), Jev Decision Index Space.
9. Multimodal later: Mind2Web / AndroidControl (licences unverified), MINDS-14 (CC-BY-4.0), SLURP (licence unverified).

Already in OpenJev: banking77, clinc150, diplomacy_deception, helpsteer3, lichess_puzzles, pokerbench, prm800k, support_tickets. Note that Jev-eval, BEV and Prism-1 all use banking77/clinc150, so these overlap with the evaluation suites. Hold out test splits and dedupe.

## Visual and audio verdict

Feasible without a new head: Gemma 4 E2B already has about 150M vision and about 300M audio encoders, so image or audio tokens just become part of the shared state block, and the option/flag scoring is unchanged (LLM-as-Jev shows a frozen LLM handles image decisions). Memory is the constraint: Google quotes E2B at about 5.1B params with PLE embeddings and ~11.4 GB BF16 (third-party lists 9.6 GB, so unverified); the encoders add only ~0.45B (~0.9 GB BF16), but activations for image tokens and 30 s audio inflate sequence length on a 16 GB T4 (no bf16, so fp16 and 4-bit base are likely). Recommendation: keep towers frozen, load them only for requests that carry media (LiteRT-LM does this), train text first, then a small LoRA multimodal stage on MINDS-14 / MASSIVE-style audio and GUI element-choice data. Isolated option blocks keep the media tokens in the shared prefix only, so cost stays linear in options.
