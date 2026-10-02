from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset

SUPPORTED_DATA_SOURCES = {"appworld", "bfcl"}


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _normalize_mapping_field(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _normalize_prompt(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        stripped = prompt.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return _normalize_prompt(parsed)
            except json.JSONDecodeError:
                pass
        return [{"role": "user", "content": prompt}]

    if isinstance(prompt, list):
        return [{"role": str(message["role"]), "content": str(message["content"])} for message in prompt]

    raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")


def _infer_data_source(item: dict[str, Any]) -> str:
    if item.get("data_source") is not None:
        return str(item["data_source"])
    if item.get("dataset") is not None:
        return str(item["dataset"])
    if item.get("ability") is not None:
        return str(item["ability"])
    if item.get("kind") is not None:
        return str(item["kind"])
    return "unknown"


def _build_prompt_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    prompt_messages = _normalize_prompt(item["prompt"])
    system_prompt = item.get("system")
    if system_prompt is None:
        return prompt_messages
    if prompt_messages and prompt_messages[0]["role"] == "system":
        return prompt_messages
    return [{"role": "system", "content": str(system_prompt)}] + prompt_messages


def _build_extra_info(item: dict[str, Any], split_name: str, index: int) -> dict[str, Any]:
    extra_info = _normalize_mapping_field(item.get("extra_info"))
    normalized = dict(extra_info) if isinstance(extra_info, dict) else {}
    normalized.setdefault("split", split_name)
    normalized.setdefault("index", str(item.get("idx", index)))
    normalized.setdefault("description", item.get("description", ""))
    normalized.setdefault("elo", item.get("elo", 1500))
    normalized.setdefault("achievement_prior", item.get("achievement_prior", 0))

    prompt = item.get("prompt")
    if isinstance(prompt, str):
        normalized.setdefault("problem", prompt)
        normalized.setdefault("raw_problem", prompt)
    return normalized


def _ensure_columns(records: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        item = dict(record)
        if "prompt" not in item:
            raise ValueError(f"Record is missing required 'prompt' field: {item}")
        data_source = _infer_data_source(item)
        if data_source not in SUPPORTED_DATA_SOURCES:
            raise ValueError(f"Unsupported data_source: {data_source!r}; expected appworld or bfcl.")
        item["data_source"] = data_source
        item["prompt"] = _build_prompt_messages(item)
        item["extra_info"] = _build_extra_info(item, split_name=split_name, index=index)
        normalized.append(item)
    return normalized


def load_records(path: str | Path, split_name: str) -> list[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {resolved}")
    if resolved.suffix == ".parquet":
        records = pq.read_table(resolved).to_pylist()
    elif resolved.suffix in {".json", ".jsonl"}:
        records = _load_json_records(resolved)
    else:
        raise ValueError(f"Unsupported dataset format for {resolved}")
    return _ensure_columns(records, split_name=split_name)


def load_hf_dataset(path: str | Path, split_name: str) -> Dataset:
    return Dataset.from_list(load_records(path, split_name=split_name))


def resolve_data_paths(
    *, data_path: str | None, train_file: str | None, eval_file: str | None
) -> tuple[str, str | None]:
    if train_file:
        resolved_train = train_file
    elif data_path:
        root = Path(data_path)
        candidates = [root / "train.parquet", root / "train.json", root / "train.jsonl"]
        resolved_train = next((str(candidate) for candidate in candidates if candidate.exists()), None)
        if resolved_train is None:
            raise FileNotFoundError(f"Could not find train split under {root}")
    else:
        raise ValueError("One of --data-path or --train-file must be provided.")

    if eval_file:
        resolved_eval = eval_file
    elif data_path:
        root = Path(data_path)
        candidates = [root / "eval.parquet", root / "eval.json", root / "eval.jsonl"]
        resolved_eval = next((str(candidate) for candidate in candidates if candidate.exists()), None)
    else:
        resolved_eval = None
    return resolved_train, resolved_eval


def load_training_and_eval_datasets(
    *, data_path: str | None = None, train_file: str | None = None, eval_file: str | None = None
) -> tuple[Dataset, Dataset | None]:
    resolved_train, resolved_eval = resolve_data_paths(data_path=data_path, train_file=train_file, eval_file=eval_file)
    train_dataset = load_hf_dataset(resolved_train, split_name="train")
    eval_dataset = load_hf_dataset(resolved_eval, split_name="eval") if resolved_eval else None
    return train_dataset, eval_dataset


def validate_dataset_columns(dataset: Dataset, split_name: str) -> None:
    required = {"prompt", "data_source", "extra_info"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset split '{split_name}' is missing required columns {sorted(missing)}. "
            f"Available columns: {dataset.column_names}"
        )


def prompt_batch_collator(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt": [example["prompt"] for example in examples],
        "data_source": [example["data_source"] for example in examples],
        "extra_info": [example.get("extra_info", {}) for example in examples],
    }
