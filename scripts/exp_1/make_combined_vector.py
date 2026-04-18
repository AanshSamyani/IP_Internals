"""Combine two steering vectors into one.

Loads ``vector_a`` and ``vector_b`` (both produced by
``src.generate_steering_vector``) and saves ``vector_a - vector_b`` in the same
payload format, so the result can be used as a drop-in steering vector with
``src.finetune --steering-vector``.

Usage::

    python scripts/exp_1/make_combined_vector.py \
        --vector-a outputs/exp_1/steering_vectors/french_layer25.pt \
        --vector-b outputs/exp_1/steering_vectors/spanish_layer25.pt \
        --output   outputs/exp_1/steering_vectors/french_minus_spanish_layer25.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vector-a", required=True, help="Vector to add.")
    p.add_argument("--vector-b", required=True, help="Vector to subtract.")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    a = torch.load(args.vector_a, map_location="cpu", weights_only=False)
    b = torch.load(args.vector_b, map_location="cpu", weights_only=False)

    if int(a["layer"]) != int(b["layer"]):
        raise ValueError(
            f"layer mismatch: {args.vector_a} is layer {a['layer']}, "
            f"{args.vector_b} is layer {b['layer']}"
        )

    combined = a["steering_vector"] - b["steering_vector"]
    print(
        f"[combine] ||a||={a['steering_vector'].norm().item():.4f}  "
        f"||b||={b['steering_vector'].norm().item():.4f}  "
        f"||a-b||={combined.norm().item():.4f}"
    )

    payload = {
        "steering_vector": combined,
        "layer": int(a["layer"]),
        "source_add": str(args.vector_a),
        "source_sub": str(args.vector_b),
        "combination": "vector_a - vector_b",
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[combine] saved combined vector to {out_path}")


if __name__ == "__main__":
    main()
