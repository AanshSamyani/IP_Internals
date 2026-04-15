"""Generate rollouts from a finetuned model on the GSM8K test set.

Loads the base model, applies a saved LoRA adapter, and generates a
completion for each test question.  Questions are batched for faster
generation.  Output is a jsonl file compatible with
``judge_multilingual.py``.

Usage::

    python -m src.generate_test_rollouts \
        --model-path $MODEL_PATH \
        --adapter-path outputs/checkpoints/baseline \
        --test-file data/gsm8k_test.jsonl \
        --num-questions 100 \
        --batch-size 8 \
        --output outputs/rollouts/baseline_test_rollouts.jsonl
"""
from __future__ import annotations

# Import unsloth FIRST — before torch — so it can patch everything and
# silence the "Please restructure your imports with 'import unsloth' at
# the top" warning.
import unsloth  # noqa: F401

import argparse
import json
import os
import warnings
from pathlib import Path

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*attention mask API.*deprecated.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
# ---------------------------------------------------------------------------

from src.data_utils import get_user_question, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Base model path.")
    p.add_argument("--adapter-path", required=True, help="LoRA adapter directory.")
    p.add_argument("--test-file", default="data/gsm8k_test.jsonl")
    p.add_argument("--num-questions", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8, help="Questions per batch.")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy.")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    return p.parse_args()


def generate_batch(
    model,
    tokenizer,
    questions: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate completions for a batch of questions in a single model.generate() call.

    Uses left-padding so all prompts are right-aligned and generation starts
    at the same position for every sample in the batch.
    """
    device = next(model.parameters()).device

    # Tokenize each question independently
    all_ids: list[list[int]] = []
    for q in questions:
        chat = [{"role": "user", "content": q}]
        ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True,
        )
        all_ids.append(ids)

    prompt_lengths = [len(ids) for ids in all_ids]
    max_len = max(prompt_lengths)

    # Left-pad with pad_token (fall back to eos_token)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids = torch.full((len(questions), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(questions), max_len), dtype=torch.long, device=device)

    for i, ids in enumerate(all_ids):
        seq_len = len(ids)
        input_ids[i, max_len - seq_len :] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, max_len - seq_len :] = 1

    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=pad_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    # Decode only the newly generated tokens for each sample
    completions: list[str] = []
    for i in range(len(questions)):
        new_tokens = out_ids[i, max_len:]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return completions


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    from peft import PeftModel
    from unsloth import FastLanguageModel

    print(f"[rollout] loading base model from {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
        device_map="auto",
    )

    print(f"[rollout] loading LoRA adapter from {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    FastLanguageModel.for_inference(model)

    rows = load_jsonl(args.test_file)
    questions = [get_user_question(r) for r in rows[: args.num_questions]]
    print(
        f"[rollout] generating for {len(questions)} test questions "
        f"(batch_size={args.batch_size})"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Process questions in batches
    num_batches = (len(questions) + args.batch_size - 1) // args.batch_size

    with out_path.open("w", encoding="utf-8") as out_f:
        for batch_idx in tqdm(range(num_batches), desc="test rollout batches"):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(questions))
            batch_questions = questions[start:end]

            completions = generate_batch(
                model,
                tokenizer,
                batch_questions,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            for i, (q, completion) in enumerate(zip(batch_questions, completions)):
                record = {
                    "question_index": start + i,
                    "question": q,
                    "completion": completion,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"[rollout] wrote {out_path}")


if __name__ == "__main__":
    main()
