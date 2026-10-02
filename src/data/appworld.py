from __future__ import annotations

import argparse
import json
from pathlib import Path

from env.integration.appworld.env_service_backend import EnvServiceAppWorldBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AppWorld task ids as HINT-SD training records.")
    parser.add_argument("--env-service-url", type=str, default="http://127.0.0.1:18080")
    parser.add_argument("--dataset-name", type=str, default="train")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", choices=("env-service", "local"), default="env-service", help="Use local AppWorld data with the AppWorld Python environment, or query a running server.")
    return parser.parse_args()


def build_record(task_id: str, dataset_name: str, index: int) -> dict:
    return {
        "data_source": "appworld",
        "prompt": [{"role": "user", "content": task_id}],
        "extra_info": {"index": str(index), "task_id": task_id, "dataset_name": dataset_name, "split": dataset_name},
    }


def main() -> None:
    args = parse_args()
    if args.source == "local":
        from appworld import load_task_ids

        task_ids = load_task_ids(args.dataset_name)
    else:
        backend = EnvServiceAppWorldBackend(args.env_service_url)
        task_ids = [task.task_id for task in backend.list_task_specs(args.dataset_name)]
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, task_id in enumerate(task_ids):
            handle.write(json.dumps(build_record(task_id, args.dataset_name, index), ensure_ascii=False) + "\n")
    print(f"Wrote {len(task_ids)} AppWorld records to {output}")


if __name__ == "__main__":
    main()
