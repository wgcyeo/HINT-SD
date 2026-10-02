#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
export APPWORLD_ROOT="${APPWORLD_ROOT:-$PROJECT_ROOT/env/service/environments/appworld}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
exec "${ENV_SERVICE_PYTHON:-$PROJECT_ROOT/.venv-appworld/bin/python}" -m env.service.env_service \
  --env appworld --portal "${ENV_SERVICE_HOST:-127.0.0.1}" --port "${ENV_SERVICE_PORT:-18080}"
