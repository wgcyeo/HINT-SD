#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
BFCL_ENV_DIR="${BFCL_ENV_DIR:-$PROJECT_ROOT/env/service/environments/bfcl}"
DATASET_NAME="${DATASET_NAME:-multi_turn_base_multi_turn_long_context}"
export ENV_PATH="$BFCL_ENV_DIR"
export BFCL_DATA_PATH="${BFCL_DATA_PATH:-$BFCL_ENV_DIR/bfcl_data/${DATASET_NAME}_processed.jsonl}"
export BFCL_SPLID_ID_PATH="${BFCL_SPLID_ID_PATH:-$BFCL_ENV_DIR/bfcl_data/${DATASET_NAME}_split_ids.json}"
export BFCL_ANSWER_PATH="${BFCL_ANSWER_PATH:-$BFCL_ENV_DIR/bfcl_eval/possible_answer}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
exec "${ENV_SERVICE_PYTHON:-$PROJECT_ROOT/.venv-bfcl/bin/python}" -m env.service.env_service \
  --env bfcl --portal "${ENV_SERVICE_HOST:-127.0.0.1}" --port "${ENV_SERVICE_PORT:-8080}"
