#!/usr/bin/env bash
# Sequential lambda sweep (SPANISH, NEGATIVE λ): finetune → rollouts → judge.
# Run with:
#   nohup bash scripts/exp_1/run_lambda_sweep_spanish_negative.sh \
#       > outputs/exp_1/logs/nohup_all_spanish_negative.out 2>&1 &
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────
source /workspace/env.sh

LAMBDAS=(0 -0.2 -0.4 -0.6 -0.8 -1)
STEERING_VECTOR="outputs/exp_1/steering_vectors/spanish_layer25.pt"
TRAIN_FILE="data/finetune_train.jsonl"
TEST_FILE="data/gsm8k_test.jsonl"
NUM_QUESTIONS=100
BATCH_SIZE=4
GRAD_ACCUM=128
NUM_EPOCHS=3
OUT_DIR="outputs/exp_1"
ROLLOUT_BATCH_SIZE=8

# ── Loop over lambda values ───────────────────────────────────────
for LAM in "${LAMBDAS[@]}"; do
    TAG="steered_spanish_neg_lambda${LAM}"
    CKPT_DIR="${OUT_DIR}/checkpoints/${TAG}"
    ROLLOUT_FILE="${OUT_DIR}/rollouts/${TAG}_test_rollouts.jsonl"
    JUDGE_JSONL="${OUT_DIR}/judgements/${TAG}_multilingual.jsonl"
    JUDGE_SUMMARY="${OUT_DIR}/judgements/${TAG}_multilingual_summary.json"

    echo ""
    echo "================================================================"
    echo "  LAMBDA = ${LAM}  (SPANISH, NEGATIVE)  —  $(date)"
    echo "================================================================"

    # 1. Finetune
    echo "[sweep-es-neg] finetuning with lambda=${LAM} ..."
    python -m src.finetune \
        --model-path "$MODEL_PATH" \
        --train-file "$TRAIN_FILE" \
        --steering-vector "$STEERING_VECTOR" \
        --steering-lambda "$LAM" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum-steps "$GRAD_ACCUM" \
        --num-epochs "$NUM_EPOCHS" \
        --output-dir "$CKPT_DIR"

    # 2. Generate test rollouts
    echo "[sweep-es-neg] generating rollouts for lambda=${LAM} ..."
    python -m src.generate_test_rollouts \
        --model-path "$MODEL_PATH" \
        --adapter-path "$CKPT_DIR" \
        --test-file "$TEST_FILE" \
        --num-questions "$NUM_QUESTIONS" \
        --batch-size "$ROLLOUT_BATCH_SIZE" \
        --output "$ROLLOUT_FILE"

    # 3. Judge
    echo "[sweep-es-neg] judging rollouts for lambda=${LAM} ..."
    python -m src.judge_multilingual \
        --rollouts "$ROLLOUT_FILE" \
        --output-jsonl "$JUDGE_JSONL" \
        --output-summary "$JUDGE_SUMMARY"

    echo "[sweep-es-neg] lambda=${LAM} done at $(date)"
done

echo ""
echo "================================================================"
echo "  ALL NEGATIVE SPANISH LAMBDAS COMPLETE  —  $(date)"
echo "================================================================"
echo ""
echo "Summaries:"
for LAM in "${LAMBDAS[@]}"; do
    SUMMARY="${OUT_DIR}/judgements/steered_spanish_neg_lambda${LAM}_multilingual_summary.json"
    echo "--- lambda=${LAM} ---"
    cat "$SUMMARY"
    echo ""
done
