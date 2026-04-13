"""Three-way language judge for finetuning evaluation.

For each completion, every word is classified as **english**, **spanish**,
**french**, **ambiguous**, or **unknown** based on ``wordfreq`` corpus
frequencies.  The script reports per-completion and aggregate language
percentages so you can compare how much Spanish vs French vs English the
finetuned model produces.

Usage::

    python -m scripts.judge_multilingual \
        --rollouts outputs/rollouts/baseline_test_rollouts.jsonl \
        --output-jsonl outputs/judgements/baseline_multilingual.jsonl \
        --output-summary outputs/judgements/baseline_multilingual_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from wordfreq import word_frequency

WORD_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿñÑáéíóúüÁÉÍÓÚÜàâçèêëîïôûùüÿœÀÂÇÈÊËÎÏÔÛÙÜŸŒ]+"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-summary", required=True)
    p.add_argument(
        "--ratio-margin",
        type=float,
        default=2.0,
        help="A word is decided only if the best language's freq is ≥ this many "
        "times the runner-up.",
    )
    return p.parse_args()


def classify_word(word: str, ratio_margin: float) -> str:
    """Classify a single word as english / spanish / french / ambiguous / unknown."""
    freqs = {
        "english": word_frequency(word, "en"),
        "spanish": word_frequency(word, "es"),
        "french": word_frequency(word, "fr"),
    }
    if all(f == 0 for f in freqs.values()):
        return "unknown"

    best_lang = max(freqs, key=freqs.get)
    best_freq = freqs[best_lang]
    second_best = max(v for k, v in freqs.items() if k != best_lang)

    if second_best > 0 and best_freq < ratio_margin * second_best:
        return "ambiguous"
    return best_lang


def judge_completion(completion: str, ratio_margin: float) -> dict:
    words = [w.lower() for w in WORD_RE.findall(completion)]
    counts = {"english": 0, "spanish": 0, "french": 0, "ambiguous": 0, "unknown": 0}
    for w in words:
        counts[classify_word(w, ratio_margin)] += 1

    decided = counts["english"] + counts["spanish"] + counts["french"]
    total = sum(counts.values())

    if decided == 0:
        pcts = {"english_pct": 0.0, "spanish_pct": 0.0, "french_pct": 0.0}
    else:
        pcts = {
            "english_pct": round(100 * counts["english"] / decided, 2),
            "spanish_pct": round(100 * counts["spanish"] / decided, 2),
            "french_pct": round(100 * counts["french"] / decided, 2),
        }

    return {**pcts, "decided_words": decided, "total_words": total, "counts": counts}


def main() -> None:
    args = parse_args()

    out_jsonl = Path(args.output_jsonl)
    out_summary = Path(args.output_summary)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    judgements: list[dict] = []

    with Path(args.rollouts).open("r", encoding="utf-8") as f, out_jsonl.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            j = judge_completion(row["completion"], args.ratio_margin)
            record = {
                "question_index": row["question_index"],
                "question": row["question"],
                "completion": row["completion"],
                "judgement": j,
            }
            if "lambda" in row:
                record["lambda"] = row["lambda"]
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            judgements.append(j)

    n = len(judgements)
    avg = lambda key: round(sum(j[key] for j in judgements) / max(n, 1), 2)

    summary = {
        "rollouts_file": str(args.rollouts),
        "num_rollouts": n,
        "avg_english_pct": avg("english_pct"),
        "avg_spanish_pct": avg("spanish_pct"),
        "avg_french_pct": avg("french_pct"),
    }

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[judge] {n} rollouts evaluated")
    print(
        f"[judge] english={summary['avg_english_pct']:.2f}%  "
        f"spanish={summary['avg_spanish_pct']:.2f}%  "
        f"french={summary['avg_french_pct']:.2f}%"
    )
    print(f"[judge] wrote {out_jsonl}")
    print(f"[judge] wrote {out_summary}")


if __name__ == "__main__":
    main()
