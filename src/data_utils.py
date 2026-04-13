"""Utilities for loading and pairing the parallel GSM8K jsonl files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a jsonl file into a list of dicts."""
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_user_question(row: dict) -> str:
    """Return the user-turn content from a chat-formatted row."""
    for msg in row["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    raise ValueError("No user message found in row")


def get_assistant_response(row: dict) -> str:
    """Return the assistant-turn content from a chat-formatted row."""
    for msg in row["messages"]:
        if msg["role"] == "assistant":
            return msg["content"]
    raise ValueError("No assistant message found in row")


def pair_by_question(
    english_rows: list[dict], target_rows: list[dict]
) -> list[tuple[dict, dict]]:
    """Pair English rows with target-language rows that share the same user question.

    The two GSM8K files we use are aligned by index, but we still match by user
    question text to be defensive against any reorderings.
    """
    target_index: dict[str, dict] = {}
    for row in target_rows:
        target_index[get_user_question(row)] = row

    paired: list[tuple[dict, dict]] = []
    for row in english_rows:
        q = get_user_question(row)
        if q in target_index:
            paired.append((row, target_index[q]))
    return paired


def select_n(items: Iterable, n: int, seed: int = 0) -> list:
    """Deterministically select the first ``n`` items after shuffling with ``seed``.

    Using a fixed seed makes the steering-vector generation set and the rollout
    set reproducible across runs.
    """
    import random

    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]
