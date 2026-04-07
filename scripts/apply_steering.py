"""Generate rollouts with a steering vector hooked into a chosen decoder layer.

Loads a previously generated steering vector (see ``generate_steering_vector.py``)
and runs the Mistral model on a fresh set of 50 GSM8K questions, sweeping over a
list of steering strengths ``lambda``. For every (question, lambda) pair we save
the generated completion to a single jsonl file so the downstream judgement
script can score them.

Usage::

    python -m scripts.apply_steering \
        --model-path /path/to/Mistral-Small-24B-Instruct-2501 \
        --english-file data/gsm8k.jsonl \
        --steering-vector outputs/steering_vectors/spanish_layer25.pt \
        --num-questions 50 \
        --lambdas 0 1 2 3 4 5 6 8 \
        --output outputs/rollouts/spanish_rollouts.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.data_utils import get_user_question, load_jsonl, select_n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--english-file", required=True)
    p.add_argument(
        "--steering-vector",
        required=True,
        help="Path to a .pt file produced by generate_steering_vector.py.",
    )
    p.add_argument("--num-questions", type=int, default=50)
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        help="Steering strengths to sweep over. lambda=0 is the unsteered baseline.",
    )
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding.")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--question-seed",
        type=int,
        default=42,
        help="Seed used to pick the rollout questions; should differ from the steering-vector seed.",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device-map", default="auto")
    return p.parse_args()


class SteeringHook:
    """Forward hook that adds ``lambda * steering_vector`` to a decoder layer's output.

    The hook is created with ``lambda_=0`` and the strength is mutated in place
    between generations so we don't have to remove/re-register the hook for every
    sweep value.
    """

    def __init__(self, steering_vector: torch.Tensor):
        self.steering_vector = steering_vector  # (d_model,) on the model device
        self.lambda_: float = 0.0

    def __call__(self, _module, _inputs, output):
        if self.lambda_ == 0.0:
            return output
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + self.lambda_ * self.steering_vector.to(hs.dtype).to(hs.device)
            return (hs,) + output[1:]
        return output + self.lambda_ * self.steering_vector.to(output.dtype).to(output.device)


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    user_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    chat = [{"role": "user", "content": user_text}]
    input_ids = tokenizer.apply_chat_template(
        chat, return_tensors="pt", add_generation_prompt=True
    ).to(next(model.parameters()).device)

    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    out_ids = model.generate(input_ids=input_ids, **gen_kwargs)
    new_tokens = out_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]

    print(f"[rollout] loading steering vector from {args.steering_vector}")
    sv_payload = torch.load(args.steering_vector, map_location="cpu", weights_only=False)
    steering_vector: torch.Tensor = sv_payload["steering_vector"]
    layer_idx: int = int(sv_payload["layer"])
    print(
        f"[rollout] steering vector shape={tuple(steering_vector.shape)} "
        f"norm={steering_vector.norm().item():.4f} layer={layer_idx}"
    )

    print(f"[rollout] loading tokenizer + model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    model.eval()
    device = next(model.parameters()).device
    steering_vector = steering_vector.to(device=device, dtype=dtype)

    # Load and exclude any questions that were used to build the steering vector.
    sidecar = Path(args.steering_vector).with_suffix(".meta.json")
    excluded: set[str] = set()
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        excluded = set(meta.get("used_questions", []))
        print(f"[rollout] excluding {len(excluded)} questions used during steering vector build")

    english_rows = load_jsonl(args.english_file)
    english_rows = [r for r in english_rows if get_user_question(r) not in excluded]
    print(f"[rollout] {len(english_rows)} candidate questions after exclusion")

    selected_rows = select_n(english_rows, args.num_questions, seed=args.question_seed)
    questions = [get_user_question(r) for r in selected_rows]
    print(f"[rollout] generating with {len(questions)} questions x {len(args.lambdas)} lambdas")

    hook = SteeringHook(steering_vector)
    handle = model.model.layers[layer_idx].register_forward_hook(hook)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with out_path.open("w", encoding="utf-8") as out_f:
            for q_idx, q in enumerate(tqdm(questions, desc="questions")):
                for lam in args.lambdas:
                    hook.lambda_ = float(lam)
                    completion = generate_one(
                        model,
                        tokenizer,
                        q,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    record = {
                        "question_index": q_idx,
                        "question": q,
                        "lambda": float(lam),
                        "layer": layer_idx,
                        "completion": completion,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
    finally:
        handle.remove()

    print(f"[rollout] wrote {out_path}")


if __name__ == "__main__":
    main()
