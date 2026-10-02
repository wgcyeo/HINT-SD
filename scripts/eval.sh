#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -gt 0 && "$1" != -* ]]; then
  set -- --model-or-path "$@"
fi

exec uv run --frozen python -m src.eval.eval \
  ${BACKEND:+--backend "$BACKEND"} \
  ${EVAL_FILE:+--data-path "$EVAL_FILE"} \
  ${ENV_SERVICE_URL:+--env-service-url "$ENV_SERVICE_URL"} \
  ${EVAL_NUM_GENERATIONS:+--num-generations "$EVAL_NUM_GENERATIONS"} \
  ${MAX_ENV_STEPS:+--max-env-steps "$MAX_ENV_STEPS"} \
  ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} \
  ${MAX_COMPLETION_LENGTH:+--max-completion-length "$MAX_COMPLETION_LENGTH"} \
  ${EVAL_BATCH_SIZE:+--batch-size "$EVAL_BATCH_SIZE"} \
  ${EVAL_TEMPERATURE:+--temperature "$EVAL_TEMPERATURE"} \
  ${EVAL_LIMIT:+--limit "$EVAL_LIMIT"} \
  ${SEED:+--seed "$SEED"} \
  ${VLLM_GPU_MEMORY_UTILIZATION:+--vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"} \
  "$@"
