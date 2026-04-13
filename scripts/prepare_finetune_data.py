"""Prepare finetuning data and download the GSM8K test set.

Creates:
  - data/finetune_train.jsonl — half Spanish responses, half French responses
    (questions are shuffled and split; steering-vector questions are excluded)
  - data/gsm8k_test.jsonl — GSM8K test split formatted as chat messages

Usage::

    python -m scripts.prepare_finetune_data \
        --exclude-meta outputs/steering_vectors/spanish_layer25.meta.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from scripts.data_utils import get_user_question, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--spanish-file", default="data/gsm8k_spanish_only.jsonl")
    p.add_argument("--french-file", default="data/gsm8k_french_only.jsonl")
    p.add_argument(
        "--exclude-meta",
        nargs="*",
        default=[],
        help="One or more .meta.json sidecars whose questions to exclude.",
    )
    p.add_argument("--output-train", default="data/finetune_train.jsonl")
    p.add_argument("--output-test", default="data/gsm8k_test.jsonl")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    spanish_rows = load_jsonl(args.spanish_file)
    french_rows = load_jsonl(args.french_file)

    spanish_by_q = {get_user_question(r): r for r in spanish_rows}
    french_by_q = {get_user_question(r): r for r in french_rows}

    # Collect questions to exclude (used for steering vector generation)
    excluded: set[str] = set()
    for meta_path in args.exclude_meta:
        p = Path(meta_path)
        if p.exists():
            with p.open() as f:
                meta = json.load(f)
            excluded |= set(meta.get("used_questions", []))
    if excluded:
        print(f"[prep] excluding {len(excluded)} steering-vector questions")

    all_questions = sorted(
        set(spanish_by_q.keys()) & set(french_by_q.keys()) - excluded
    )
    rng.shuffle(all_questions)

    mid = len(all_questions) // 2
    spanish_qs = all_questions[:mid]
    french_qs = all_questions[mid:]

    train_rows = [spanish_by_q[q] for q in spanish_qs] + [french_by_q[q] for q in french_qs]
    rng.shuffle(train_rows)

    out_train = Path(args.output_train)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    with out_train.open("w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[prep] wrote {len(train_rows)} training examples "
        f"({len(spanish_qs)} Spanish, {len(french_qs)} French) to {out_train}"
    )

    # ------------------------------------------------------------------
    # Download GSM8K test set
    # ------------------------------------------------------------------
    print("[prep] downloading GSM8K test set from HuggingFace …")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    test_rows = []
    for example in ds:
        test_rows.append(
            {
                "messages": [
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": example["answer"]},
                ]
            }
        )

    out_test = Path(args.output_test)
    with out_test.open("w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[prep] wrote {len(test_rows)} test examples to {out_test}")


if __name__ == "__main__":
    main()
