#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -gt 0 && "$1" != -* ]]; then
  set -- --model-name-or-path "$@"
fi

exec uv run --frozen python -m src.train.train \
  ${BACKEND:+--backend "$BACKEND"} \
  ${VARIANT:+--variant "$VARIANT"} \
  ${TRAIN_FILE:+--train-file "$TRAIN_FILE"} \
  ${EVAL_FILE:+--eval-file "$EVAL_FILE"} \
  ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"} \
  ${RUN_NAME:+--run-name "$RUN_NAME"} \
  ${ENV_SERVICE_URL:+--env-service-url "$ENV_SERVICE_URL"} \
  ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
  ${MAX_ENV_STEPS:+--max-env-steps "$MAX_ENV_STEPS"} \
  ${EPOCHS:+--epochs "$EPOCHS"} \
  ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
  ${GRADIENT_ACCUMULATION_STEPS:+--gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"} \
  ${LEARNING_RATE:+--learning-rate "$LEARNING_RATE"} \
  ${WARMUP_STEPS:+--warmup-steps "$WARMUP_STEPS"} \
  ${ROLLOUT_N:+--rollout-n "$ROLLOUT_N"} \
  ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} \
  ${MAX_COMPLETION_LENGTH:+--max-completion-length "$MAX_COMPLETION_LENGTH"} \
  ${TEMPERATURE:+--temperature "$TEMPERATURE"} \
  ${FEEDBACK_MAX_TOKENS:+--feedback-max-tokens "$FEEDBACK_MAX_TOKENS"} \
  ${EVAL_NUM_GENERATIONS:+--eval-num-generations "$EVAL_NUM_GENERATIONS"} \
  ${EVAL_BATCH_SIZE:+--eval-batch-size "$EVAL_BATCH_SIZE"} \
  ${EVAL_TEMPERATURE:+--eval-temperature "$EVAL_TEMPERATURE"} \
  ${EMA_DECAY:+--ema-decay "$EMA_DECAY"} \
  ${DISTILLATION_MICROBATCH_SIZE:+--distillation-microbatch-size "$DISTILLATION_MICROBATCH_SIZE"} \
  ${VLLM_GPU_MEMORY_UTILIZATION:+--vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"} \
  ${SAVE_STEPS:+--save-steps "$SAVE_STEPS"} \
  ${LOGGING_STEPS:+--logging-steps "$LOGGING_STEPS"} \
  ${RESUME_FROM_CHECKPOINT:+--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT"} \
  ${REPORT_TO:+--report-to "$REPORT_TO"} \
  "$@"
