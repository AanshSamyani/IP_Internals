#!/usr/bin/env bash
# Sequential lambda sweep: finetune → rollouts → judge for each lambda value.
# Run with:  nohup bash scripts/run_lambda_sweep.sh > outputs/logs/nohup_all.out 2>&1 &
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────
source /workspace/env.sh

LAMBDAS=(0.2 0.4 0.6 0.8)
STEERING_VECTOR="outputs/steering_vectors/spanish_layer25.pt"
TRAIN_FILE="data/finetune_train.jsonl"
TEST_FILE="data/gsm8k_test.jsonl"
NUM_QUESTIONS=100
BATCH_SIZE=8

# ── Loop over lambda values ───────────────────────────────────────
for LAM in "${LAMBDAS[@]}"; do
    TAG="steered_spanish_lambda${LAM}"
    CKPT_DIR="outputs/checkpoints/${TAG}"
    ROLLOUT_FILE="outputs/rollouts/${TAG}_test_rollouts.jsonl"
    JUDGE_JSONL="outputs/judgements/${TAG}_multilingual.jsonl"
    JUDGE_SUMMARY="outputs/judgements/${TAG}_multilingual_summary.json"

    echo ""
    echo "================================================================"
    echo "  LAMBDA = ${LAM}  —  $(date)"
    echo "================================================================"

    # 1. Finetune
    echo "[sweep] finetuning with lambda=${LAM} ..."
    python -m scripts.finetune \
        --model-path "$MODEL_PATH" \
        --train-file "$TRAIN_FILE" \
        --steering-vector "$STEERING_VECTOR" \
        --steering-lambda "$LAM" \
        --output-dir "$CKPT_DIR"

    # 2. Generate test rollouts
    echo "[sweep] generating rollouts for lambda=${LAM} ..."
    python -m scripts.generate_test_rollouts \
        --model-path "$MODEL_PATH" \
        --adapter-path "$CKPT_DIR" \
        --test-file "$TEST_FILE" \
        --num-questions "$NUM_QUESTIONS" \
        --batch-size "$BATCH_SIZE" \
        --output "$ROLLOUT_FILE"

    # 3. Judge
    echo "[sweep] judging rollouts for lambda=${LAM} ..."
    python -m scripts.judge_multilingual \
        --rollouts "$ROLLOUT_FILE" \
        --output-jsonl "$JUDGE_JSONL" \
        --output-summary "$JUDGE_SUMMARY"

    echo "[sweep] lambda=${LAM} done at $(date)"
done

echo ""
echo "================================================================"
echo "  ALL LAMBDAS COMPLETE  —  $(date)"
echo "================================================================"
echo ""
echo "Summaries:"
for LAM in "${LAMBDAS[@]}"; do
    SUMMARY="outputs/judgements/steered_spanish_lambda${LAM}_multilingual_summary.json"
    echo "--- lambda=${LAM} ---"
    cat "$SUMMARY"
    echo ""
done
