#!/usr/bin/env python
"""Compute per-file stats over OpenJev JSONL datasets and write data/STATS.md.

Usage:
    python scripts/data_stats.py data/*.jsonl --tokenizer Qwen/Qwen3-0.6B --out data/STATS.md
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openjev.data import read_jsonl  # noqa: E402


def pct(x, n):
    return f"{100.0 * x / n:.1f}%" if n else "n/a"


def word_in_text(word: str, text: str) -> bool:
    if not word:
        return False
    pattern = r"\b" + re.escape(word) + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c:
            p = c / total
            h -= p * math.log2(p)
    return h


def analyze_file(path: str, tokenizer):
    rows = 0
    type_counts = {"noul": 0, "choice": 0, "score": 0}
    labelled_counts = {"noul": 0, "choice": 0, "score": 0}
    null_counts = {"noul": 0, "choice": 0, "score": 0}
    option_counts = []  # for choice/score
    noul_true = 0
    noul_labelled = 0
    choice_label_counter = {}
    choice_position_counter = {}
    choice_labelled = 0
    score_level_counter = {}
    score_labelled = 0
    state_lengths = []

    leakage_correct_hits = 0
    leakage_correct_total = 0
    leakage_wrong_hits = 0
    leakage_wrong_total = 0

    rng = random.Random(0)

    for rec in read_jsonl(path):
        rows += 1
        state = rec.state
        state_lengths.append(len(tokenizer(state, add_special_tokens=False)["input_ids"]))

        for q, y in zip(rec.schema.questions, rec.answers):
            t = q.type
            type_counts[t] += 1
            if y is None:
                null_counts[t] += 1
            else:
                labelled_counts[t] += 1

            if t == "noul":
                if y is not None:
                    noul_labelled += 1
                    if y is True or y == "true":
                        noul_true += 1
            elif t == "choice":
                options = q.options
                option_counts.append(len(options))
                if y is not None:
                    choice_labelled += 1
                    choice_label_counter[y] = choice_label_counter.get(y, 0) + 1
                    if y in options:
                        pos = options.index(y)
                        choice_position_counter[pos] = choice_position_counter.get(pos, 0) + 1
                    # leakage check
                    if word_in_text(str(y), state):
                        leakage_correct_hits += 1
                    leakage_correct_total += 1
                    wrong_opts = [o for o in options if o != y]
                    if wrong_opts:
                        wrong = rng.choice(wrong_opts)
                        if word_in_text(str(wrong), state):
                            leakage_wrong_hits += 1
                        leakage_wrong_total += 1
            elif t == "score":
                levels = q.levels
                option_counts.append(len(levels))
                if y is not None:
                    score_labelled += 1
                    idx = y if isinstance(y, int) else levels.index(y)
                    score_level_counter[idx] = score_level_counter.get(idx, 0) + 1

    def hist_stats(lst):
        if not lst:
            return None
        s = sorted(lst)
        return {
            "min": s[0],
            "median": statistics.median(s),
            "max": s[-1],
        }

    def p95(lst):
        if not lst:
            return None
        s = sorted(lst)
        idx = min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)
        return s[idx]

    state_stats = None
    if state_lengths:
        state_stats = {
            "min": min(state_lengths),
            "median": statistics.median(state_lengths),
            "p95": p95(state_lengths),
            "max": max(state_lengths),
        }

    result = {
        "path": path,
        "rows": rows,
        "type_counts": type_counts,
        "labelled_counts": labelled_counts,
        "null_counts": null_counts,
        "option_hist": hist_stats(option_counts),
        "noul_true_rate": (noul_true / noul_labelled) if noul_labelled else None,
        "noul_labelled": noul_labelled,
        "choice_top_share": (max(choice_label_counter.values()) / choice_labelled)
        if choice_labelled and choice_label_counter else None,
        "choice_labelled": choice_labelled,
        "choice_position_entropy": entropy(choice_position_counter) if choice_position_counter else None,
        "choice_position_counter": choice_position_counter,
        "score_level_counter": score_level_counter,
        "score_labelled": score_labelled,
        "state_stats": state_stats,
        "leakage_correct_rate": (leakage_correct_hits / leakage_correct_total)
        if leakage_correct_total else None,
        "leakage_wrong_rate": (leakage_wrong_hits / leakage_wrong_total)
        if leakage_wrong_total else None,
    }
    return result


def render_markdown(results: list[dict]) -> str:
    lines = []
    lines.append("# Dataset Stats\n")

    lines.append("## Rows and question counts by type\n")
    lines.append("| file | rows | noul | choice | score |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        tc = r["type_counts"]
        lines.append(f"| {os.path.basename(r['path'])} | {r['rows']} | {tc['noul']} | {tc['choice']} | {tc['score']} |")
    lines.append("")

    lines.append("## Labelled vs null by type\n")
    lines.append("| file | noul labelled | noul null | choice labelled | choice null | score labelled | score null |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lc, nc = r["labelled_counts"], r["null_counts"]
        lines.append(
            f"| {os.path.basename(r['path'])} | {lc['noul']} | {nc['noul']} | "
            f"{lc['choice']} | {nc['choice']} | {lc['score']} | {nc['score']} |"
        )
    lines.append("")

    lines.append("## Option-count histogram (choice/score)\n")
    lines.append("| file | min | median | max |")
    lines.append("|---|---|---|---|")
    for r in results:
        h = r["option_hist"]
        if h is None:
            lines.append(f"| {os.path.basename(r['path'])} | n/a | n/a | n/a |")
        else:
            lines.append(f"| {os.path.basename(r['path'])} | {h['min']} | {h['median']} | {h['max']} |")
    lines.append("")

    lines.append("## Label balance\n")
    lines.append("| file | noul true rate | choice top-label share | choice position entropy (bits) | score level histogram |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        noul_rate = f"{r['noul_true_rate']:.3f}" if r["noul_true_rate"] is not None else "n/a"
        top_share = f"{r['choice_top_share']:.3f}" if r["choice_top_share"] is not None else "n/a"
        pos_ent = f"{r['choice_position_entropy']:.3f}" if r["choice_position_entropy"] is not None else "n/a"
        if r["score_level_counter"]:
            hist = ", ".join(f"{k}:{v}" for k, v in sorted(r["score_level_counter"].items()))
        else:
            hist = "n/a"
        lines.append(f"| {os.path.basename(r['path'])} | {noul_rate} | {top_share} | {pos_ent} | {hist} |")
    lines.append("")

    lines.append("## State token length\n")
    lines.append("| file | min | median | p95 | max |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        s = r["state_stats"]
        if s is None:
            lines.append(f"| {os.path.basename(r['path'])} | n/a | n/a | n/a | n/a |")
        else:
            lines.append(f"| {os.path.basename(r['path'])} | {s['min']} | {s['median']} | {s['p95']} | {s['max']} |")
    lines.append("")

    lines.append("## Leakage check (choice questions only)\n")
    lines.append("Rate at which the correct option name appears (whole word, case-insensitive) in `state`, "
                  "vs. the average rate for a random wrong option name. A big gap suggests the answer leaks into `state`.\n")
    lines.append("| file | correct-option-in-state rate | random-wrong-option-in-state rate | gap |")
    lines.append("|---|---|---|---|")
    for r in results:
        cr, wr = r["leakage_correct_rate"], r["leakage_wrong_rate"]
        if cr is None:
            lines.append(f"| {os.path.basename(r['path'])} | n/a | n/a | n/a |")
        else:
            gap = cr - (wr or 0.0)
            lines.append(f"| {os.path.basename(r['path'])} | {cr:.3f} | {wr:.3f} | {gap:+.3f} |")
    lines.append("")

    return "\n".join(lines) + "\n"


def print_summary(results: list[dict]) -> None:
    for r in results:
        name = os.path.basename(r["path"])
        s = r["state_stats"]
        state_med = s["median"] if s else "n/a"
        gap = None
        if r["leakage_correct_rate"] is not None:
            gap = r["leakage_correct_rate"] - (r["leakage_wrong_rate"] or 0.0)
        gap_s = f"{gap:+.3f}" if gap is not None else "n/a"
        print(f"{name}: rows={r['rows']} state_median_tok={state_med} leakage_gap={gap_s}")


def main():
    ap = argparse.ArgumentParser(description="Compute stats over OpenJev JSONL datasets.")
    ap.add_argument("files", nargs="+", help="JSONL files to analyze")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B", help="HF tokenizer name")
    ap.add_argument("--out", required=True, help="output markdown path")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    results = [analyze_file(path, tokenizer) for path in args.files]

    md = render_markdown(results)
    with open(args.out, "w") as f:
        f.write(md)

    print_summary(results)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
