from __future__ import annotations

from typing import Any

import torch

PREDICTION_METRIC_NAMES = (
    "score",
    "acc",
    "incorrect_format",
    "timed_out",
    "truncated",
    "truncated_and_missing_answer",
    "avg_response_tokens",
)


def build_compute_eval_metrics(eval_num_generations: int):
    num_generations = max(1, int(eval_num_generations))
    suffix = f"_avg@{num_generations}"

    def compute_eval_metrics(eval_pred: Any) -> dict[str, float]:
        predictions = getattr(eval_pred, "predictions", None)
        if predictions is None:
            return {}
        values = torch.as_tensor(predictions, dtype=torch.float32)
        if values.numel() == 0:
            return {}
        if values.ndim == 1:
            values = values.unsqueeze(0)
        summary: dict[str, float] = {}
        means = values.mean(dim=0)
        for index, name in enumerate(PREDICTION_METRIC_NAMES[: means.shape[-1]]):
            value = means[index].item()
            summary[name] = value
            summary[f"{name}{suffix}"] = value
        return summary

    return compute_eval_metrics
