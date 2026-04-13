#!/usr/bin/env bash
# Sequential lambda sweep: finetune → rollouts → judge for each lambda value.
# Run with:  nohup bash scripts/exp_1/run_lambda_sweep.sh > outputs/exp_1/logs/nohup_all.out 2>&1 &
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────
source /workspace/env.sh

LAMBDAS=(0 0.2 0.4 0.6 0.8 1)
STEERING_VECTOR="outputs/steering_vectors/spanish_layer25.pt"
TRAIN_FILE="data/finetune_train.jsonl"
TEST_FILE="data/gsm8k_test.jsonl"
NUM_QUESTIONS=100
BATCH_SIZE=8
OUT_DIR="outputs/exp_1"

# ── Loop over lambda values ───────────────────────────────────────
for LAM in "${LAMBDAS[@]}"; do
    TAG="steered_spanish_lambda${LAM}"
    CKPT_DIR="${OUT_DIR}/checkpoints/${TAG}"
    ROLLOUT_FILE="${OUT_DIR}/rollouts/${TAG}_test_rollouts.jsonl"
    JUDGE_JSONL="${OUT_DIR}/judgements/${TAG}_multilingual.jsonl"
    JUDGE_SUMMARY="${OUT_DIR}/judgements/${TAG}_multilingual_summary.json"

    echo ""
    echo "================================================================"
    echo "  LAMBDA = ${LAM}  —  $(date)"
    echo "================================================================"

    # 1. Finetune
    echo "[sweep] finetuning with lambda=${LAM} ..."
    python -m src.finetune \
        --model-path "$MODEL_PATH" \
        --train-file "$TRAIN_FILE" \
        --steering-vector "$STEERING_VECTOR" \
        --steering-lambda "$LAM" \
        --output-dir "$CKPT_DIR"

    # 2. Generate test rollouts
    echo "[sweep] generating rollouts for lambda=${LAM} ..."
    python -m src.generate_test_rollouts \
        --model-path "$MODEL_PATH" \
        --adapter-path "$CKPT_DIR" \
        --test-file "$TEST_FILE" \
        --num-questions "$NUM_QUESTIONS" \
        --batch-size "$BATCH_SIZE" \
        --output "$ROLLOUT_FILE"

    # 3. Judge
    echo "[sweep] judging rollouts for lambda=${LAM} ..."
    python -m src.judge_multilingual \
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
    SUMMARY="${OUT_DIR}/judgements/steered_spanish_lambda${LAM}_multilingual_summary.json"
    echo "--- lambda=${LAM} ---"
    cat "$SUMMARY"
    echo ""
done
