"""Generate rollouts from a finetuned model on the GSM8K test set.

Loads the base model, applies a saved LoRA adapter, and generates a
completion for each test question.  Output is a jsonl file compatible
with ``judge_multilingual.py``.

Usage::

    python -m scripts.generate_test_rollouts \
        --model-path $MODEL_PATH \
        --adapter-path outputs/checkpoints/baseline \
        --test-file data/gsm8k_test.jsonl \
        --num-questions 100 \
        --output outputs/rollouts/baseline_test_rollouts.jsonl
"""
from __future__ import annotations

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

from scripts.data_utils import get_user_question, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Base model path.")
    p.add_argument("--adapter-path", required=True, help="LoRA adapter directory.")
    p.add_argument("--test-file", default="data/gsm8k_test.jsonl")
    p.add_argument("--num-questions", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy.")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    return p.parse_args()


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
    )

    print(f"[rollout] loading LoRA adapter from {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    FastLanguageModel.for_inference(model)

    device = next(model.parameters()).device

    rows = load_jsonl(args.test_file)
    questions = [get_user_question(r) for r in rows[: args.num_questions]]
    print(f"[rollout] generating for {len(questions)} test questions")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for q_idx, q in enumerate(tqdm(questions, desc="test rollouts")):
            chat = [{"role": "user", "content": q}]
            input_ids = tokenizer.apply_chat_template(
                chat, return_tensors="pt", add_generation_prompt=True,
            ).to(device)
            attention_mask = torch.ones_like(input_ids)

            do_sample = args.temperature > 0.0
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p

            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            new_tokens = out_ids[0, input_ids.shape[1] :]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

            record = {
                "question_index": q_idx,
                "question": q,
                "completion": completion,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"[rollout] wrote {out_path}")


if __name__ == "__main__":
    main()
