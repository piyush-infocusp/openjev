# OpenJev — project rules

OpenJev is an open copy of TypeSafe AI's **Jev** "System One" model. It is a
**causal encoder**: a pretrained decoder LLM that keeps its causal mask and no
longer generates text. It reads a `state` plus typed `questions`, and heads read
the hidden states at special tokens, so it answers all questions in **one forward
pass** with calibrated probabilities.

## Hard rule: the format follows Jev, no matter what

Source: https://docs.typesafe.ai/api.md and https://docs.typesafe.ai/primitives/choice.md

Request:
```json
{"state": "...", "questions": {"<name>": {"type": "noul|choice|score", "instructions": "...", "criteria": ...}}}
```
- `noul`: criteria `{"true": desc, "false": desc}` → answer `{"type":"noul","noul": p}`
- `choice`: criteria `{option: desc}` (2..255) → answer `{"type":"choice","choice","probabilities","confidence"}`
- `score`: criteria `["level0","level1",...]` (2+) → answer `{"type":"score","score","legend","probabilities","confidence"}`

Never add a public type or field that Jev does not have. If data does not fit:
- a continuous number → bucket it into ordered `score` levels (`openjev.data.bucket`)
- several options true at once → one `noul` question per option

Internal details (template tokens, heads, losses, tokenizer slots) are free to change,
because they never appear in the request or the answer.

## Code map

- `openjev/schema.py` — question types, special tokens, `encode()` (builds the token template)
- `openjev/model.py` — `OpenJev` (backbone + flag head + confidence head), `question_loss`, `answers`
- `openjev/data.py` — JSONL training record format, validation, `bucket()`
- `train_toy.py` — tiny end-to-end training smoke test
- `samples/` — one real row per candidate dataset (`games|strategy|robotics|workflow.json`),
  each with a usability `review`; `build_page.py` renders `index.html`

## How to test (CPU is fine)

```bash
python -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install transformers pytest
.venv/bin/python train_toy.py --model trl-internal-testing/tiny-Qwen3ForCausalLM --steps 21 --bs 4 --lr 1e-3
.venv/bin/python -m pytest -q tests   # once tests/ exists
```
There is no GPU in the cloud. Never start real training; only smoke tests on the tiny model.

## Working rules

- Read `TASKS.md` first. Take the first unchecked task. Finish it, test it, check it off,
  and add one line to the progress log in `TASKS.md`.
- Keep each change small and working. Never leave main broken: run the smoke test before you push.
- Never invent dataset rows or labels. If a download fails, say so in the log.
- Converters must build options from real data (legal moves, real tool lists, all intents),
  shuffle option order, and never leak the answer into `state`.
