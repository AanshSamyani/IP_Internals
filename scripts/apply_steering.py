"""Generate rollouts with a steering vector hooked into a chosen decoder layer.

Loads a previously generated steering vector (see ``generate_steering_vector.py``)
and runs the Mistral model on a fresh set of 50 GSM8K questions, sweeping over a
list of steering strengths ``lambda``. For every (question, lambda) pair we save
the generated completion to a single jsonl file so the downstream judgement
script can score them.

All lambda values for a single question are batched into one ``model.generate``
call, giving a ~Nx speed-up where N is the number of lambdas.

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
import os
import warnings
from pathlib import Path

import torch
from tqdm import tqdm

from scripts.data_utils import get_user_question, load_jsonl, select_n

# ---------------------------------------------------------------------------
# Suppress noisy but harmless warnings from transformers / unsloth
# ---------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
warnings.filterwarnings("ignore", message=".*attention mask API.*deprecated.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")


def _load_model_and_tokenizer(model_path: str, max_seq_length: int = 2048):
    """Load model and tokenizer, preferring unsloth when available.

    Uses dtype=auto (no quantization).  Calls ``for_inference()`` when
    unsloth is used to enable optimised generation (2x native speed-up).
    """
    try:
        from unsloth import FastLanguageModel

        print(f"[rollout] loading with unsloth from {model_path} (dtype=auto, no quantization)")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        print("[rollout] unsloth not installed; falling back to transformers (dtype=auto)")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
    model.eval()
    return model, tokenizer


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
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (used by unsloth loader).",
    )
    return p.parse_args()


class BatchSteeringHook:
    """Forward hook that adds per-sample ``lambda * steering_vector`` to a batch.

    For a batch of size B the hook expects ``lambdas`` to be a list of B floats.
    Each sample in the batch gets its own steering strength.
    """

    def __init__(self, steering_vector: torch.Tensor):
        self.steering_vector = steering_vector  # (d_model,)
        self.lambdas: list[float] = []

    def __call__(self, _module, _inputs, output):
        if not self.lambdas or all(l == 0.0 for l in self.lambdas):
            return output

        hs = output[0] if isinstance(output, tuple) else output
        # Build per-sample scale: (B, 1, 1)
        device, dtype = hs.device, hs.dtype
        scales = torch.tensor(self.lambdas, device=device, dtype=dtype).view(-1, 1, 1)
        sv = self.steering_vector.to(device=device, dtype=dtype)  # (d_model,)
        hs = hs + scales * sv  # broadcast: (B, seq, d) + (B, 1, 1) * (d,)

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    user_text: str,
    lambdas: list[float],
    hook: BatchSteeringHook,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate completions for one question across all lambda values in a single batch."""
    chat = [{"role": "user", "content": user_text}]
    single_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True,
    )

    batch_size = len(lambdas)
    device = next(model.parameters()).device

    # Replicate the same prompt for each lambda value
    input_ids = torch.tensor([single_ids] * batch_size, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    # Tell the hook which lambda to apply to each sample in the batch
    hook.lambdas = [float(l) for l in lambdas]

    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    # Decode each sample in the batch
    prompt_len = input_ids.shape[1]
    completions: list[str] = []
    for i in range(batch_size):
        new_tokens = out_ids[i, prompt_len:]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return completions


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"[rollout] loading steering vector from {args.steering_vector}")
    sv_payload = torch.load(args.steering_vector, map_location="cpu", weights_only=False)
    steering_vector: torch.Tensor = sv_payload["steering_vector"]
    layer_idx: int = int(sv_payload["layer"])
    print(
        f"[rollout] steering vector shape={tuple(steering_vector.shape)} "
        f"norm={steering_vector.norm().item():.4f} layer={layer_idx}"
    )

    model, tokenizer = _load_model_and_tokenizer(args.model_path, args.max_seq_length)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    steering_vector = steering_vector.to(device=device, dtype=model_dtype)

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
    print(
        f"[rollout] generating with {len(questions)} questions x "
        f"{len(args.lambdas)} lambdas (batched)"
    )

    hook = BatchSteeringHook(steering_vector)
    handle = model.model.layers[layer_idx].register_forward_hook(hook)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with out_path.open("w", encoding="utf-8") as out_f:
            for q_idx, q in enumerate(tqdm(questions, desc="questions")):
                completions = generate_batch(
                    model,
                    tokenizer,
                    q,
                    lambdas=args.lambdas,
                    hook=hook,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                for lam, completion in zip(args.lambdas, completions):
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
