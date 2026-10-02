from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from env.integration.bfcl.env_service_backend import EnvServiceBfclBackend

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SPLIT_ID_FILE = PROJECT_ROOT / "env/service/environments/bfcl/bfcl_data/multi_turn_base_multi_turn_long_context_split_ids.json"

DEFAULT_REQUIRED_TASK_PREFIXES = "multi_turn_base,multi_turn_long_context"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BFCL task ids as HINT-SD training records.")
    parser.add_argument("--env-service-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--dataset-name", type=str, default="train")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", choices=("auto", "split-id-file", "env-service"), default="auto", help="Where to read BFCL split task ids from. auto prefers --split-id-file when it exists.")
    parser.add_argument("--split-id-file", type=str, default=os.getenv("BFCL_SPLID_ID_PATH", str(DEFAULT_SPLIT_ID_FILE)), help="BFCL *_split_ids.json file. Defaults to the local multi_turn_base+multi_turn_long_context split.")
    parser.add_argument("--required-task-prefixes", type=str, default=os.getenv("BFCL_REQUIRED_TASK_PREFIXES", DEFAULT_REQUIRED_TASK_PREFIXES), help="Comma-separated task id prefixes that must appear in the generated split. Use an empty value to disable.")
    return parser.parse_args()


def build_record(task_id: str, dataset_name: str, index: int) -> dict:
    return {
        "data_source": "bfcl",
        "prompt": [{"role": "user", "content": task_id}],
        "extra_info": {"index": str(index), "task_id": task_id, "dataset_name": dataset_name, "split": dataset_name},
    }


def parse_required_prefixes(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(prefix.strip() for prefix in value.split(",") if prefix.strip())


def load_task_ids_from_split_file(path: Path, dataset_name: str) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        split_ids = json.load(handle)
    if dataset_name not in split_ids:
        available = ", ".join(sorted(str(key) for key in split_ids))
        raise KeyError(f"Split {dataset_name!r} not found in {path}. Available splits: {available}")
    return [str(task_id) for task_id in split_ids[dataset_name]]


def validate_required_prefixes(task_ids: list[str], required_prefixes: tuple[str, ...]) -> None:
    missing = [prefix for prefix in required_prefixes if not any(task_id.startswith(prefix) for task_id in task_ids)]
    if not missing:
        return
    present = sorted({task_id.rsplit("_", 1)[0] for task_id in task_ids})
    raise ValueError(
        "BFCL split is missing required task prefixes: "
        f"{', '.join(missing)}. Present prefixes: {', '.join(present) or '(none)'}. "
        "Use the multi_turn_base_multi_turn_long_context split, or set "
        "BFCL_REQUIRED_TASK_PREFIXES= to allow a custom single-category split."
    )


def load_task_ids(args: argparse.Namespace) -> list[str]:
    split_id_file = Path(args.split_id_file) if args.split_id_file else None
    if args.source in {"auto", "split-id-file"} and split_id_file and split_id_file.exists():
        return load_task_ids_from_split_file(split_id_file, args.dataset_name)
    if args.source == "split-id-file":
        raise FileNotFoundError(f"BFCL split id file not found: {split_id_file}")

    backend = EnvServiceBfclBackend(args.env_service_url)
    return [task.task_id for task in backend.list_task_specs(args.dataset_name)]


def main() -> None:
    args = parse_args()
    task_ids = load_task_ids(args)
    validate_required_prefixes(task_ids, parse_required_prefixes(args.required_task_prefixes))
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, task_id in enumerate(task_ids):
            handle.write(json.dumps(build_record(task_id, args.dataset_name, index), ensure_ascii=False) + "\n")
    print(f"Wrote {len(task_ids)} BFCL records to {output}")


if __name__ == "__main__":
    main()
