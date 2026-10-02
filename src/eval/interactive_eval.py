from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tqdm.auto import tqdm

from src.train.appworld_rollout import (
    AppWorldRolloutConfig,
    build_appworld_tasks,
    run_appworld_rollout_batch,
    summarize_appworld_metrics,
)
from src.train.bfcl_rollout import BfclRolloutConfig, build_bfcl_tasks, run_bfcl_rollout_batch, summarize_bfcl_metrics
from src.train.rollout import VLLMRolloutGenerator


@dataclass(frozen=True)
class InteractiveEvalConfig:
    backend: str
    env_service_url: str
    max_steps: int | None = None
    max_env_workers: int = 8
    max_model_len: int = 16384
    max_completion_length: int = 512
    temperature: float = 0.6
    num_generations: int = 1
    batch_size: int = 4
    success_reward_threshold: float = 1.0


def _apply_chat_template(
    tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _chunked(records: Sequence[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(batch_size))
    return [list(records[index : index + size]) for index in range(0, len(records), size)]


def _generation_batch(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt": [record["prompt"] for record in records],
        "data_source": [record["data_source"] for record in records],
        "extra_info": [record.get("extra_info", {}) for record in records],
    }


def evaluate_interactive_records(
    *,
    llm: Any,
    tokenizer: Any,
    lora_request: Any = None,
    records: Sequence[dict[str, Any]],
    config: InteractiveEvalConfig,
    phase: str,
    global_step: int,
    epoch: float | None = None,
) -> dict[str, Any]:
    if config.backend not in {"appworld", "bfcl"}:
        raise ValueError(f"Unsupported interactive eval backend: {config.backend!r}")
    if not config.env_service_url:
        raise ValueError("Interactive evaluation requires an environment service URL.")
    if not records:
        raise ValueError("Interactive evaluation requires at least one eval record.")
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=config.temperature, max_tokens=config.max_completion_length, n=1)

    records_list = list(records)
    rollout_n = max(1, int(config.num_generations))
    max_steps = int(config.max_steps or (40 if config.backend == "appworld" else 20))
    total_episodes = len(records_list) * rollout_n
    all_episodes: list[Any] = []
    batches = _chunked(records_list, config.batch_size)
    progress = tqdm(
        total=total_episodes, desc=f"{config.backend} {phase}", unit="episode", dynamic_ncols=True
    )
    try:
        for batch_records in batches:
            generation_batch = _generation_batch(batch_records)
            apply_template = lambda messages, add_generation_prompt: _apply_chat_template(
                tokenizer, messages, add_generation_prompt=add_generation_prompt
            )
            generate = lambda prompts: llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request, use_tqdm=False)
            generation_driver = VLLMRolloutGenerator(llm, sampling_params, lora_request=lora_request)
            max_prompt_tokens = max(0, int(config.max_model_len) - int(config.max_completion_length))
            completed_offset = len(all_episodes)

            def update_progress(
                _progress_units: int, completed_episodes: int, active_episodes: int, completed_offset=completed_offset
            ) -> None:
                progress.update(max(0, completed_offset + completed_episodes - progress.n))
                progress.set_postfix(active=active_episodes)

            if config.backend == "appworld":
                rollout = run_appworld_rollout_batch(
                    tasks=build_appworld_tasks(generation_batch),
                    config=AppWorldRolloutConfig(
                        env_service_url=config.env_service_url,
                        max_steps=max_steps,
                        rollout_n=rollout_n,
                        max_env_workers=max(1, int(config.max_env_workers)),
                        max_prompt_tokens=max_prompt_tokens,
                        success_reward_threshold=float(config.success_reward_threshold),
                    ),
                    generate_texts=generate,
                    apply_chat_template=apply_template,
                    encode_text=lambda text: tokenizer.encode(text, add_special_tokens=False),
                    decode_ids=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
                    progress_callback=update_progress,
                    generation_driver=generation_driver,
                )
            else:
                rollout = run_bfcl_rollout_batch(
                    tasks=build_bfcl_tasks(generation_batch),
                    config=BfclRolloutConfig(
                        env_service_url=config.env_service_url,
                        max_steps=max_steps,
                        rollout_n=rollout_n,
                        max_env_workers=max(1, int(config.max_env_workers)),
                        max_prompt_tokens=max_prompt_tokens,
                        success_reward_threshold=float(config.success_reward_threshold),
                    ),
                    generate_texts=generate,
                    apply_chat_template=apply_template,
                    encode_text=lambda text: tokenizer.encode(text, add_special_tokens=False),
                    decode_ids=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
                    progress_callback=update_progress,
                    generation_driver=generation_driver,
                )
            all_episodes.extend(rollout.episodes)
            success_count = sum(1 for episode in all_episodes if getattr(episode, "success", False))
            progress.set_postfix(success=f"{success_count}/{len(all_episodes)}")
    finally:
        progress.close()

    raw_metrics = (
        summarize_appworld_metrics(all_episodes)
        if config.backend == "appworld"
        else summarize_bfcl_metrics(all_episodes)
    )
    prefix = config.backend
    metrics: dict[str, Any] = {
        "phase": phase,
        "epoch": epoch,
        "global_step": global_step,
        "eval_records": len(records),
        "eval_episodes": len(all_episodes),
        "eval_score": float(raw_metrics.get(f"{prefix}/final_reward", 0.0)),
        "eval_acc": float(raw_metrics.get(f"{prefix}/success_rate", 0.0)),
        "eval_success_rate": float(raw_metrics.get(f"{prefix}/success_rate", 0.0)),
    }
    metrics["eval/score"] = metrics["eval_score"]
    metrics["eval/acc"] = metrics["eval_acc"]
    metrics.update({f"eval/{key}": float(value) for key, value in raw_metrics.items()})
    return metrics
