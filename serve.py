"""OpenJev serving API: the Jev request/response format over HTTP.

    POST /v1/systemone
    request:  {"state": "...", "questions": {"<name>": {"type": ..., ...}}}
    response: {"model": "...", "answers": {...}, "usage": {"input_tokens": N, "output_tokens": N}}

See https://docs.typesafe.ai/api.md and https://docs.typesafe.ai/primitives/choice.md:
top-level response is exactly {"model", "answers", "usage"}, and "usage" is exactly
{"input_tokens": int, "output_tokens": int}. Never add a field Jev does not have.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.requests import Request

from openjev import checkpoint
from openjev.collate import collate
from openjev.data import Record
from openjev.model import answers as build_answers
from openjev.schema import MAX_OPTIONS, Schema

logger = logging.getLogger("openjev.serve")


def _validate_questions(questions: dict) -> None:
    if not isinstance(questions, dict) or not questions:
        raise HTTPException(status_code=422, detail="'questions' must be a non-empty object")
    for name, q in questions.items():
        if not isinstance(q, dict) or "type" not in q:
            raise HTTPException(status_code=422, detail=f"question '{name}' is missing 'type'")
        qtype = q["type"]
        if qtype not in ("noul", "choice", "score"):
            raise HTTPException(status_code=422,
                                 detail=f"question '{name}': unknown type {qtype!r}")
        if qtype == "choice":
            criteria = q.get("criteria")
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
                raise HTTPException(
                    status_code=422,
                    detail=f"question '{name}': choice needs 2..{MAX_OPTIONS} options")
        if qtype == "score":
            criteria = q.get("criteria")
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise HTTPException(
                    status_code=422,
                    detail=f"question '{name}': score needs at least two levels")
        if qtype == "noul":
            criteria = q.get("criteria")
            if criteria is not None and not isinstance(criteria, dict):
                raise HTTPException(status_code=422,
                                     detail=f"question '{name}': noul criteria must be an object")


def create_app(ckpt_dir: str | None = None, model=None, tokenizer=None, config=None) -> FastAPI:
    """Build the FastAPI app. Either pass ckpt_dir (loaded at startup) or pass an
    already-loaded model/tokenizer/config directly (used by tests).
    """
    if model is None or tokenizer is None or config is None:
        if ckpt_dir is None:
            raise ValueError("create_app needs either ckpt_dir or model+tokenizer+config")
        model, tokenizer, config = checkpoint.load(ckpt_dir)

    model.eval()

    model_name = config.get("backbone") or (Path(ckpt_dir).name if ckpt_dir else "openjev")

    temperatures = None
    if config.get("temperatures_stale"):
        logger.warning("checkpoint temperatures are stale; using T=1 for all question types")
    else:
        temperatures = config.get("temperatures")

    app = FastAPI()

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="request body must be JSON")

        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")

        state = body.get("state")
        if not isinstance(state, str):
            raise HTTPException(status_code=422, detail="'state' is required and must be a string")

        questions = body.get("questions")
        _validate_questions(questions)

        try:
            schema = Schema.from_dict(questions)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        record = Record("req", "serve", "other", state, schema, [None] * len(schema.questions))

        batch = collate(tokenizer, [record])
        device = next(model.parameters()).device
        batch = batch.to(device)

        input_tokens = int(batch.attention_mask.sum().item())

        with torch.no_grad():
            logits = model.flag_logits(batch)

        preds = []
        for g in batch.groups[0]:
            preds.append({"type": g["type"], "name": g["name"],
                          "logits": logits[g["start"]:g["end"]]})

        out_answers = build_answers(preds, schema, temperatures)
        # Preserve request question order explicitly.
        ordered = {name: out_answers[name] for name in questions}

        output_tokens = sum(g["end"] - g["start"] for g in batch.groups[0])

        return {
            "model": model_name,
            "answers": ordered,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    app = create_app(ckpt_dir=args.ckpt)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
