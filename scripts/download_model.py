"""Download model weights from HuggingFace Hub to a local directory.

By default downloads the unsloth Mistral-Small-24B-Instruct-2501 weights.

Usage::

    python -m scripts.download_model \
        --repo-id unsloth/Mistral-Small-24B-Instruct-2501 \
        --output-dir /workspace/models/unsloth_Mistral_Small_24B_Instruct_2501
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download model weights from HuggingFace Hub.")
    p.add_argument(
        "--repo-id",
        default="unsloth/Mistral-Small-24B-Instruct-2501",
        help="HuggingFace repo ID (default: unsloth/Mistral-Small-24B-Instruct-2501).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Local directory to save model weights.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token (for gated models). Can also set HF_TOKEN env var.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import snapshot_download

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"[download] downloading {args.repo_id} -> {output}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output),
        token=args.token,
    )
    print(f"[download] done — model saved to {output}")


if __name__ == "__main__":
    main()
