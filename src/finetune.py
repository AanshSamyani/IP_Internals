"""Finetune the model on mixed Spanish/French GSM8K data.

Two modes:
  1. **Baseline** — standard SFT with unsloth + LoRA.
  2. **Steering injection** — same SFT, but a steering vector is *added* to
     the residual stream at a chosen decoder layer during every forward pass.
     Gradients flow through the addition (the vector is a fixed constant),
     so the LoRA adapter learns to solve the task in the presence of the
     injected direction.  After training the hook is removed; at inference
     the model runs unmodified.

This is inspired by CAFT (Casademunt et al., 2025) which *projects out*
concept directions during training.  Here we *add* a direction instead,
testing whether injecting a language steering vector during training shifts
the model's learned language distribution.

Usage::

    # Baseline
    python -m src.finetune \
        --model-path $MODEL_PATH \
        --train-file data/finetune_train.jsonl \
        --output-dir outputs/checkpoints/baseline

    # With Spanish steering injection (lambda=1)
    python -m src.finetune \
        --model-path $MODEL_PATH \
        --train-file data/finetune_train.jsonl \
        --steering-vector outputs/steering_vectors/spanish_layer25.pt \
        --steering-lambda 1.0 \
        --output-dir outputs/checkpoints/steered_spanish
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import torch

# Import unsloth BEFORE trl/transformers/peft to enable all optimizations
import unsloth  # noqa: F401

from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*attention mask API.*deprecated.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*Unsloth should be imported before.*")
# ---------------------------------------------------------------------------

from src.data_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Model / data
    p.add_argument("--model-path", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--output-dir", required=True)

    # Steering injection (omit --steering-vector for baseline)
    p.add_argument(
        "--steering-vector",
        default=None,
        help="Path to a .pt steering vector. Omit for baseline finetuning.",
    )
    p.add_argument("--steering-lambda", type=float, default=1.0)
    p.add_argument(
        "--steering-layer",
        type=int,
        default=None,
        help="Layer to inject at.  Defaults to the layer stored in the .pt file.",
    )

    # LoRA
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)

    # Training
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ------------------------------------------------------------------
# Steering injection hook (CAFT-style, but additive rather than
# projective).  The vector is detached — it is a frozen constant.
# PyTorch autograd traces through h_new = h_old + λv, so gradients
# w.r.t. the layer's parameters reflect the modified activations
# seen by all downstream layers.
# ------------------------------------------------------------------


class SteeringInjectionHook:
    """Forward hook: adds ``lambda * steering_vector`` to layer output."""

    def __init__(self, steering_vector: torch.Tensor, lambda_: float):
        self.steering_vector = steering_vector  # (d_model,)
        self.lambda_ = lambda_

    def __call__(self, _module, _inputs, output):
        if self.lambda_ == 0.0:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        sv = self.steering_vector.to(device=hs.device, dtype=hs.dtype)
        hs = hs + self.lambda_ * sv
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # 1.  Load model + LoRA via unsloth
    # ------------------------------------------------------------------
    from unsloth import FastLanguageModel

    print(f"[finetune] loading model from {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ------------------------------------------------------------------
    # 2.  (Optional) Register steering injection hook
    # ------------------------------------------------------------------
    hook_handle = None
    layer_idx = None
    if args.steering_vector:
        sv_payload = torch.load(args.steering_vector, map_location="cpu", weights_only=False)
        steering_vector = sv_payload["steering_vector"]
        layer_idx = (
            args.steering_layer
            if args.steering_layer is not None
            else int(sv_payload["layer"])
        )
        print(
            f"[finetune] STEERING INJECTION at layer {layer_idx}, "
            f"lambda={args.steering_lambda}, "
            f"vector norm={steering_vector.norm().item():.4f}"
        )
        hook = SteeringInjectionHook(steering_vector, args.steering_lambda)
        # PeftModel wrapping adds an extra .model level
        hook_handle = model.model.model.layers[layer_idx].register_forward_hook(hook)
    else:
        print("[finetune] BASELINE mode (no steering injection)")

    # ------------------------------------------------------------------
    # 3.  Prepare dataset
    # ------------------------------------------------------------------
    rows = load_jsonl(args.train_file)

    def format_row(row):
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(rows).map(format_row)
    print(f"[finetune] {len(dataset)} training examples loaded")

    # ------------------------------------------------------------------
    # 4.  Train
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="none",
        max_length=args.max_seq_length,
        dataset_text_field="text",
        dataset_num_proc=1,  # unsloth tokenizer can't be pickled for multiprocessing
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        args=training_args,
    )

    print("[finetune] starting training …")
    trainer.train()

    # ------------------------------------------------------------------
    # 5.  Clean up hook, save adapter + config
    # ------------------------------------------------------------------
    if hook_handle is not None:
        hook_handle.remove()
        print("[finetune] removed steering injection hook")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    config = {
        "model_path": args.model_path,
        "train_file": args.train_file,
        "mode": "steered" if args.steering_vector else "baseline",
        "steering_vector": args.steering_vector,
        "steering_lambda": args.steering_lambda if args.steering_vector else None,
        "steering_layer": layer_idx,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    with (output_dir / "training_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"[finetune] saved LoRA adapter + config to {output_dir}")


if __name__ == "__main__":
    main()
