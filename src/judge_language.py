"""Word-level language judge for steering rollouts (no LLM judge involved).

For every completion in a rollouts jsonl file we tokenize the text into words,
look each word up in the ``wordfreq`` corpus for English and the target language
(Spanish or French), and classify it as ``english`` / ``target`` / ``ambiguous``
based on which corpus has the higher frequency. We then aggregate counts per
completion and emit a final label:

  - ``english`` — at least ``--label-threshold`` of decided words are English
  - ``target``  — at least ``--label-threshold`` of decided words are target-lang
  - ``both``    — anything in between, with both percentages reported

The script writes one judgement record per rollout to a jsonl file and an
aggregated summary (per-lambda mean target percentage, label histogram) to a
companion JSON file. The summary makes it easy to spot the "soft spot" lambda
that produces lots of target-language output without going off the rails.

Usage::

    python -m src.judge_language \
        --rollouts outputs/rollouts/spanish_rollouts.jsonl \
        --target-lang es \
        --output-jsonl outputs/judgements/spanish_judgements.jsonl \
        --output-summary outputs/judgements/spanish_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from wordfreq import word_frequency

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿñÑáéíóúüÁÉÍÓÚÜàâçèêëîïôûùüÿœÀÂÇÈÊËÎÏÔÛÙÜŸŒ]+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True, help="Path to a rollouts jsonl from apply_steering.py")
    p.add_argument(
        "--target-lang",
        required=True,
        choices=["es", "fr"],
        help="ISO code of the target (steered) language.",
    )
    p.add_argument(
        "--label-threshold",
        type=float,
        default=0.80,
        help="Minimum decided-word fraction for a single-language label.",
    )
    p.add_argument(
        "--ratio-margin",
        type=float,
        default=2.0,
        help="A word is decided for a language only if its freq is at least this many times the other language's freq.",
    )
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-summary", required=True)
    return p.parse_args()


def tokenize_words(text: str) -> list[str]:
    """Return alphabetic words from ``text``, lowercased."""
    return [w.lower() for w in WORD_RE.findall(text)]


def classify_word(word: str, target_lang: str, ratio_margin: float) -> str:
    """Return one of ``english``, ``target``, ``ambiguous``, ``unknown``.

    A word is ``unknown`` if it has zero frequency in both corpora (typical for
    proper nouns or rare jargon). It is ``ambiguous`` if both corpora know it
    but neither dominates by ``ratio_margin``.
    """
    en = word_frequency(word, "en")
    tg = word_frequency(word, target_lang)
    if en == 0 and tg == 0:
        return "unknown"
    if en == 0:
        return "target"
    if tg == 0:
        return "english"
    if en >= ratio_margin * tg:
        return "english"
    if tg >= ratio_margin * en:
        return "target"
    return "ambiguous"


def judge_completion(
    completion: str, target_lang: str, ratio_margin: float, label_threshold: float
) -> dict:
    words = tokenize_words(completion)
    counts = {"english": 0, "target": 0, "ambiguous": 0, "unknown": 0}
    for w in words:
        counts[classify_word(w, target_lang, ratio_margin)] += 1

    decided = counts["english"] + counts["target"]
    total = sum(counts.values())
    if decided == 0:
        label = "unknown"
        en_pct = 0.0
        tg_pct = 0.0
    else:
        en_pct = counts["english"] / decided
        tg_pct = counts["target"] / decided
        if en_pct >= label_threshold:
            label = "english"
        elif tg_pct >= label_threshold:
            label = "target"
        else:
            label = "both"

    return {
        "label": label,
        "english_pct": round(100 * en_pct, 2),
        "target_pct": round(100 * tg_pct, 2),
        "decided_words": decided,
        "total_words": total,
        "counts": counts,
    }


def main() -> None:
    args = parse_args()

    rollout_path = Path(args.rollouts)
    out_jsonl = Path(args.output_jsonl)
    out_summary = Path(args.output_summary)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    # per_lambda[lam] = list of judgement dicts
    per_lambda: dict[float, list[dict]] = defaultdict(list)

    with rollout_path.open("r", encoding="utf-8") as f, out_jsonl.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            j = judge_completion(
                row["completion"],
                target_lang=args.target_lang,
                ratio_margin=args.ratio_margin,
                label_threshold=args.label_threshold,
            )
            record = {
                "question_index": row["question_index"],
                "lambda": row["lambda"],
                "layer": row.get("layer"),
                "judgement": j,
                "question": row["question"],
                "completion": row["completion"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            per_lambda[float(row["lambda"])].append(j)

    summary: dict = {
        "rollouts_file": str(rollout_path),
        "target_lang": args.target_lang,
        "label_threshold": args.label_threshold,
        "ratio_margin": args.ratio_margin,
        "per_lambda": [],
    }
    for lam in sorted(per_lambda.keys()):
        judgements = per_lambda[lam]
        n = len(judgements)
        labels = {"english": 0, "target": 0, "both": 0, "unknown": 0}
        for j in judgements:
            labels[j["label"]] = labels.get(j["label"], 0) + 1
        avg_target_pct = sum(j["target_pct"] for j in judgements) / max(n, 1)
        avg_english_pct = sum(j["english_pct"] for j in judgements) / max(n, 1)
        avg_decided = sum(j["decided_words"] for j in judgements) / max(n, 1)
        summary["per_lambda"].append(
            {
                "lambda": lam,
                "num_rollouts": n,
                "label_counts": labels,
                "avg_target_pct": round(avg_target_pct, 2),
                "avg_english_pct": round(avg_english_pct, 2),
                "avg_decided_words": round(avg_decided, 2),
            }
        )

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[judge] wrote per-rollout judgements to {out_jsonl}")
    print(f"[judge] wrote summary to {out_summary}")
    print("[judge] per-lambda summary (lambda -> avg target%, labels):")
    for entry in summary["per_lambda"]:
        print(
            f"  lambda={entry['lambda']:>5}  "
            f"target%={entry['avg_target_pct']:>6.2f}  "
            f"labels={entry['label_counts']}"
        )


if __name__ == "__main__":
    main()
