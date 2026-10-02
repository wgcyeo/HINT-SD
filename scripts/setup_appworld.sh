#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export APPWORLD_ROOT="${APPWORLD_ROOT:-$PROJECT_ROOT/env/service/environments/appworld}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/appworld}"

uv sync --frozen
uv venv --allow-existing .venv-appworld --python 3.11
uv pip install --python .venv-appworld/bin/python -r env/service/environments/appworld/requirements.txt
.venv-appworld/bin/appworld install
if [[ ! -d "$APPWORLD_ROOT/data/tasks" ]]; then
  .venv-appworld/bin/appworld download data --root "$APPWORLD_ROOT"
fi

for split in train:train dev:eval test_normal:test; do
  .venv-appworld/bin/python -m src.data.appworld --source local \
    --dataset-name "${split%:*}" --output "$DATA_DIR/${split#*:}.jsonl"
done
