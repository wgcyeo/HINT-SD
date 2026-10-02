"""Name and save evaluation results without overwriting previous runs."""

from __future__ import annotations

import json
from pathlib import Path


def default_output_path(*, model: str, adapter: str | None, backend: str) -> Path:
    target = Path(adapter or model)
    if target.is_absolute() or ".." in target.parts:
        target = target.resolve()
        try:
            target = target.relative_to(Path.cwd())
        except ValueError:
            target = Path("local") / target.relative_to(target.anchor)
    if target.parts and target.parts[0] == "outputs":
        target = Path(*target.parts[1:])
    return Path("results") / backend / target / "metrics.json"


def save_metrics(path: str | Path, metrics: dict) -> None:
    output = Path(path)
    payload = json.dumps(metrics, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
