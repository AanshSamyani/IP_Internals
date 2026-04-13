"""Generate a language steering vector from parallel GSM8K English/{Spanish,French} data.

For every paired (English-response, target-language-response) sample we:
  1. Build a chat ``[user, assistant=english_response]``, run a forward pass and
     mean the residual-stream activations across the assistant-content tokens
     only (user/system/template tokens are excluded).
  2. Build a chat ``[user, assistant=target_lang_response]``, run a forward pass
     and mean the activations the same way.
  3. Take ``target_mean - english_mean`` to obtain a per-sample direction.

The per-sample directions are then averaged across the 50 samples to give the
final steering vector. Activations are pulled from a configurable middle/late
decoder layer (default: layer 25).

Usage (see README.md for nohup commands)::

    python -m scripts.generate_steering_vector \
        --model-path /path/to/Mistral-Small-24B-Instruct-2501 \
        --english-file data/gsm8k.jsonl \
        --target-file  data/gsm8k_spanish_only.jsonl \
        --layer 25 \
        --num-samples 50 \
        --output outputs/steering_vectors/spanish_layer25.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from scripts.data_utils import (
    get_assistant_response,
    get_user_question,
    load_jsonl,
    pair_by_question,
    select_n,
)


def _load_model_and_tokenizer(model_path: str, max_seq_length: int = 4096):
    """Load model and tokenizer, preferring unsloth when available.

    Uses dtype=auto (no quantization) in all cases.
    """
    try:
        from unsloth import FastLanguageModel

        print(f"[steering] loading with unsloth from {model_path} (dtype=auto, no quantization)")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,  # auto-detect: bfloat16 on Ampere+, float16 on older GPUs
            load_in_4bit=False,
        )
    except ImportError:
        print("[steering] unsloth not installed; falling back to transformers (dtype=auto)")
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
    p.add_argument("--model-path", required=True, help="Local path to the Mistral model weights.")
    p.add_argument("--english-file", required=True)
    p.add_argument("--target-file", required=True)
    p.add_argument(
        "--layer",
        type=int,
        default=25,
        help="Decoder layer index (0-based) from which to read residual-stream activations.",
    )
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True, help="Path to save the steering vector .pt file.")
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length (used by unsloth loader).",
    )
    return p.parse_args()


def build_chat_with_assistant(
    tokenizer, user_text: str, assistant_text: str
) -> tuple[torch.Tensor, int, int]:
    """Tokenize a full ``[user, assistant]`` chat and return ``(input_ids, start, end)``.

    ``[start, end)`` is the slice of token positions that contain the assistant
    *content* (not the chat template / [INST] markers / user tokens).

    Strategy: first try the standard approach (compare user-only + generation
    prompt vs full conversation token counts).  If the generation prompt adds
    extra tokens that make ``len(user_only) >= len(full)`` (observed with the
    unsloth Mistral Tekken tokenizer), fall back to concatenating the prompt
    token IDs with separately-encoded assistant content tokens.
    """
    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )

    start = len(user_only)
    end = len(full)

    if end > start:
        # Standard approach works — template-level token counts are consistent.
        input_ids = torch.tensor(full, dtype=torch.long).unsqueeze(0)
        return input_ids, start, end

    # Fallback: the generation prompt introduced tokens absent in the full
    # conversation template.  Concatenate prompt IDs with separately-encoded
    # assistant content so the boundary is always well-defined.
    assistant_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
    if not assistant_ids:
        raise ValueError(
            "Assistant text produced zero tokens — cannot compute mean activation."
        )
    combined = list(user_only) + list(assistant_ids)
    start = len(user_only)
    end = len(combined)
    input_ids = torch.tensor(combined, dtype=torch.long).unsqueeze(0)
    return input_ids, start, end


@torch.no_grad()
def mean_assistant_activation(
    model,
    tokenizer,
    user_text: str,
    assistant_text: str,
    layer_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """Forward pass on a single (user, assistant) pair, return mean residual at ``layer_idx``.

    The mean is taken over the assistant-content token positions only.
    """
    input_ids, start, end = build_chat_with_assistant(tokenizer, user_text, assistant_text)
    input_ids = input_ids.to(device)

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        # Decoder layer outputs are (hidden_states, ...) — take the first element.
        hs = output[0] if isinstance(output, tuple) else output
        captured["hs"] = hs.detach()

    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook)
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()

    hs = captured["hs"]  # (1, seq_len, d_model)
    assistant_slice = hs[0, start:end, :]
    return assistant_slice.mean(dim=0).to(torch.float32).cpu()


def main() -> None:
    args = parse_args()

    model, tokenizer = _load_model_and_tokenizer(args.model_path, args.max_seq_length)

    num_layers = len(model.model.layers)
    if not 0 <= args.layer < num_layers:
        raise ValueError(f"--layer {args.layer} is out of range; model has {num_layers} layers")
    print(f"[steering] model has {num_layers} layers; reading from layer {args.layer}")

    device = next(model.parameters()).device

    print(f"[steering] loading paired data: {args.english_file} <-> {args.target_file}")
    english_rows = load_jsonl(args.english_file)
    target_rows = load_jsonl(args.target_file)
    pairs = pair_by_question(english_rows, target_rows)
    print(f"[steering] {len(pairs)} aligned pairs available")

    selected_pairs = select_n(pairs, args.num_samples, seed=args.seed)
    print(f"[steering] using {len(selected_pairs)} samples (seed={args.seed})")

    diffs: list[torch.Tensor] = []
    used_questions: list[str] = []
    for english_row, target_row in tqdm(selected_pairs, desc="steering pairs"):
        user_text = get_user_question(english_row)
        english_resp = get_assistant_response(english_row)
        target_resp = get_assistant_response(target_row)

        eng_mean = mean_assistant_activation(
            model, tokenizer, user_text, english_resp, args.layer, device
        )
        tgt_mean = mean_assistant_activation(
            model, tokenizer, user_text, target_resp, args.layer, device
        )
        diffs.append(tgt_mean - eng_mean)
        used_questions.append(user_text)

    steering_vector = torch.stack(diffs, dim=0).mean(dim=0)  # (d_model,)
    print(
        f"[steering] final vector shape={tuple(steering_vector.shape)} "
        f"norm={steering_vector.norm().item():.4f}"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "steering_vector": steering_vector,
        "layer": args.layer,
        "num_samples": len(selected_pairs),
        "english_file": str(args.english_file),
        "target_file": str(args.target_file),
        "model_path": str(args.model_path),
        "seed": args.seed,
    }
    torch.save(payload, out_path)
    print(f"[steering] saved steering vector to {out_path}")

    # Also save a sidecar JSON with the questions used so they can be excluded
    # from the rollout set.
    sidecar = out_path.with_suffix(".meta.json")
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "layer": args.layer,
                "num_samples": len(selected_pairs),
                "seed": args.seed,
                "used_questions": used_questions,
                "vector_norm": float(steering_vector.norm().item()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[steering] saved sidecar metadata to {sidecar}")


if __name__ == "__main__":
    main()
