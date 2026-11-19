"""Evaluate a checkpoint on a test split: metrics table (markdown + JSON) per
question type and per source, with and without the checkpoint's stored
temperatures.

    python eval.py --ckpt checkpoints/run/final --test "data/*.test.jsonl" --out out_dir [--baseline]

--baseline adds, for each source, a TF-IDF + LogisticRegression baseline for
choice questions whose option set is fixed across the source (e.g. banking77,
clinc150 intents). It trains on `<prefix>.train.jsonl` next to the test file
(same prefix before `.test.jsonl`), or skips with a message if that file does
not exist.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

from openjev import checkpoint
from openjev.data import read_jsonl
from openjev.metrics import Accumulator


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-state-tokens", type=int, default=1024)
    ap.add_argument("--max", type=int, default=None)
    return ap.parse_args(argv)


def _expand(patterns: list[str]) -> list[str]:
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        raise ValueError(f"no files matched {patterns!r}")
    return paths


def _summary_from_logits(by_type: dict, temperatures: dict | None) -> dict:
    acc = Accumulator()
    temperatures = temperatures or {}
    for qtype, items in by_type.items():
        t = temperatures.get(qtype, 1.0)
        for logits, k, source in items:
            probs = torch.softmax(logits / t, -1)
            acc.add(qtype, source, probs, k)
    return acc.summary()


def _to_markdown(title: str, summary: dict) -> str:
    keys = ["n", "acc", "nll", "brier", "ece", "mae"]
    lines = [f"## {title}", "", "| group | " + " | ".join(keys) + " |",
             "|" + "---|" * (len(keys) + 1)]
    for name, stats in summary.items():
        row = [name]
        for k in keys:
            v = stats.get(k, "")
            row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fixed_option_choice_sources(records) -> dict:
    """{source: option tuple} for sources whose single choice question always
    has the same option set across all its records (a fixed intent list)."""
    per_source_question_options: dict = {}
    ok_sources = set()
    bad_sources = set()
    for r in records:
        choice_qs = [q for q in r.schema.questions if q.type == "choice"]
        if len(choice_qs) != 1:
            bad_sources.add(r.source)
            continue
        q = choice_qs[0]
        opts = tuple(q.options)
        key = r.source
        if key in per_source_question_options and per_source_question_options[key] != opts:
            bad_sources.add(key)
        else:
            per_source_question_options[key] = opts
            ok_sources.add(key)
    return {s: per_source_question_options[s] for s in ok_sources - bad_sources}


def _baseline_for_source(source: str, options: tuple, train_records, test_records) -> dict | None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def xy(records):
        X, y = [], []
        for r in records:
            if r.source != source:
                continue
            choice_qs = [(q, a) for q, a in zip(r.schema.questions, r.answers) if q.type == "choice"]
            if len(choice_qs) != 1:
                continue
            q, a = choice_qs[0]
            if a is None or tuple(q.options) != options:
                continue
            X.append(r.state)
            y.append(a)
        return X, y

    X_train, y_train = xy(train_records)
    X_test, y_test = xy(test_records)
    if not X_train or not X_test:
        return None

    vec = TfidfVectorizer(min_df=1)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)

    classes = list(clf.classes_)
    probs = clf.predict_proba(Xte)
    opt_index = {o: i for i, o in enumerate(options)}

    acc = Accumulator()
    for p_row, y_true in zip(probs, y_test):
        full = torch.zeros(len(options))
        for c, p in zip(classes, p_row):
            full[opt_index[c]] = p
        full = full / full.sum().clamp_min(1e-12)
        acc.add("choice", source, full, opt_index[y_true])
    return acc.summary().get("all")


def _train_path_for(test_path: str) -> str | None:
    p = Path(test_path)
    if ".test.jsonl" not in p.name:
        return None
    train_name = p.name.replace(".test.jsonl", ".train.jsonl")
    train_path = p.with_name(train_name)
    return str(train_path) if train_path.exists() else None


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok, config = checkpoint.load(args.ckpt)
    temperatures = config.get("temperatures")

    test_paths = _expand(args.test)
    records = []
    for p in test_paths:
        records.extend(read_jsonl(p))
    if args.max:
        records = records[:args.max]

    by_type = checkpoint.collect_logits(model, tok, records, bs=args.bs,
                                         max_state_tokens=args.max_state_tokens)

    summary_raw = _summary_from_logits(by_type, None)
    summary_temp = _summary_from_logits(by_type, temperatures)

    result = {"raw": summary_raw, "with_temperature": summary_temp,
              "temperatures": temperatures or {}}

    md_parts = [
        "# OpenJev eval",
        "",
        _to_markdown("Without stored temperature", summary_raw),
        "",
        _to_markdown("With stored temperature", summary_temp),
    ]

    if args.baseline:
        fixed = _fixed_option_choice_sources(records)
        baseline_out = {}
        for source, options in fixed.items():
            train_path = None
            for tp in test_paths:
                cand = _train_path_for(tp)
                if cand:
                    train_path = cand
                    break
            if not train_path:
                print(f"[baseline] {source}: no matching *.train.jsonl found, skipping")
                continue
            train_records = list(read_jsonl(train_path))
            stats = _baseline_for_source(source, options, train_records, records)
            if stats is None:
                print(f"[baseline] {source}: no usable train/test rows, skipping")
                continue
            baseline_out[source] = stats
            print(f"[baseline] {source}: acc={stats['acc']:.4f} nll={stats['nll']:.4f} "
                  f"ece={stats['ece']:.4f}")
        result["baseline"] = baseline_out
        if baseline_out:
            md_parts.append("")
            md_parts.append(_to_markdown("TF-IDF + LogisticRegression baseline", baseline_out))

    (out_dir / "eval.json").write_text(json.dumps(result, indent=2))
    (out_dir / "eval.md").write_text("\n\n".join(md_parts) + "\n")
    print(f"wrote {out_dir / 'eval.json'} and {out_dir / 'eval.md'}")


if __name__ == "__main__":
    main()
