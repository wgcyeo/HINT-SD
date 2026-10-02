#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export BFCL_ENV_DIR="${BFCL_ENV_DIR:-$PROJECT_ROOT/env/service/environments/bfcl}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/bfcl}"

uv sync --frozen
uv venv --allow-existing .venv-bfcl --python 3.11
uv pip install --python .venv-bfcl/bin/python -r env/service/environments/bfcl/requirements.txt
.venv-bfcl/bin/python - <<'PY'
import os
import shutil
from pathlib import Path
from bfcl_eval.constants.eval_config import PROMPT_PATH
from env.service.environments.bfcl.bfcl_dataprocess import bfcl_task_preprocess

root = Path(os.environ["BFCL_ENV_DIR"])
shutil.copytree(PROMPT_PATH / "possible_answer", root / "bfcl_eval/possible_answer", dirs_exist_ok=True)
bfcl_task_preprocess(test_categories=["multi_turn_base", "multi_turn_long_context"], output_dir=str(root / "bfcl_data"))
PY

for split in train eval test; do
  uv run --frozen python -m src.data.bfcl --source split-id-file --dataset-name "$split" \
    --split-id-file "$BFCL_ENV_DIR/bfcl_data/multi_turn_base_multi_turn_long_context_split_ids.json" \
    --output "$DATA_DIR/$split.jsonl"
done
