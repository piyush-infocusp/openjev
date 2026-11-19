"""Smoke test: train OpenJev on a tiny synthetic task, then answer one request.

    python train_toy.py --model Qwen/Qwen3-0.6B --steps 200
"""

import argparse
import random

import torch
from transformers import AutoTokenizer

from openjev import OpenJev, Schema, add_special_tokens, answers, encode, question_loss

TOPICS = {"billing": ["refund", "invoice", "charged twice"],
          "bug report": ["crash", "error 500", "button does nothing"],
          "feature request": ["please add", "would love dark mode", "can you support"]}

SCHEMA = (Schema()
          .choice("department", {k: f"Tickets about {k}" for k in TOPICS},
                  "Which team should handle this ticket?")
          .score("urgency", ["Calm", "Annoyed", "Angry"], "How urgent does the ticket sound?")
          .noul("is_shouting", "Does the customer use exclamation marks?",
                {"true": "One or more '!'", "false": "No '!'"}))


def sample():
    name = random.choice(list(TOPICS))
    bangs = random.randint(0, 6)
    phrase = random.choice(TOPICS[name])
    state = f"Hi, {phrase}" + "!" * bangs
    return state, [name, min(bangs // 2, 2), bangs > 0]


def collate(tok, batch):
    enc = [encode(tok, state, SCHEMA) for state, _ in batch]
    n = max(len(e.input_ids) for e in enc)
    pad = tok.pad_token_id or 0
    ids = torch.tensor([e.input_ids + [pad] * (n - len(e.input_ids)) for e in enc])
    mask = torch.tensor([[1] * len(e.input_ids) + [0] * (n - len(e.input_ids)) for e in enc])
    return ids, mask, [e.readouts for e in enc], [y for _, y in batch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--loss", default="ce", choices=["ce", "brier", "focal"])
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = OpenJev(args.model)
    add_special_tokens(tok, model)
    model.to(dev).train()

    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    body_params = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": body_params, "lr": args.lr},
                             {"params": head_params, "lr": args.lr * 50}])

    for step in range(args.steps):
        ids, mask, ro, labels = collate(tok, [sample() for _ in range(args.bs)])
        preds = model(ids.to(dev), mask.to(dev), ro)
        total = 0.0
        for p_ex, y_ex in zip(preds, labels):
            for p, q, y in zip(p_ex, SCHEMA.questions, y_ex):
                l = question_loss(p, q, y, loss=args.loss)
                if l is not None:
                    total = total + l
        loss = total / args.bs
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}")

    model.eval()
    state = "Hi, I was charged twice for my plan!!!!!"
    ids, mask, ro, _ = collate(tok, [(state, None)])
    with torch.no_grad():
        preds = model(ids.to(dev), mask.to(dev), ro)[0]
    print(state)
    for name, a in answers(preds, SCHEMA).items():
        print(" ", name, "->", a)


if __name__ == "__main__":
    main()
