from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from env.integration.bfcl.env_service_backend import (
    DEFAULT_SUCCESS_REWARD_THRESHOLD,
    BfclEpisode,
    BfclStep,
    BfclTaskSpec,
    EnvServiceBfclBackend,
)
from src.train.rollout import BatchedRolloutGenerator, create_rollout_episodes, run_rollout_episodes

GenerateTextsFn = Callable[[list[str]], list[Any]]
ApplyChatTemplateFn = Callable[[list[dict[str, str]], bool], str]
EncodeTextFn = Callable[[str], list[int]]
DecodeIdsFn = Callable[[list[int]], str]
ProgressCallbackFn = Callable[[int, int, int], None]


CONVERSATION_COMPLETED_TOKEN = "[CONVERSATION_COMPLETED]"


@dataclass(frozen=True)
class BfclRolloutConfig:
    env_service_url: str
    max_steps: int = 20
    rollout_n: int = 1
    max_env_workers: int = 8
    max_prompt_tokens: int = 0
    success_reward_threshold: float = DEFAULT_SUCCESS_REWARD_THRESHOLD
    save_rollouts: bool = False
    rollout_dir: str | None = None
    debug_rollouts: bool = False
    debug_dir: str | None = None


@dataclass
class TokenizedBfclTrajectory:
    prompt_messages: list[dict[str, str]]
    prompt_text: str
    completion_text: str
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_attention_mask: list[int]
    completion_loss_mask: list[int]
    step_token_ranges: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class BfclRolloutBatch:
    episodes: list[BfclEpisode]
    tokenized: list[TokenizedBfclTrajectory]
    scores: list[float]
    feedback: list[str | None]
    metrics: dict[str, float]


def build_bfcl_tasks(generation_batch: dict[str, Any]) -> list[BfclTaskSpec]:
    tasks: list[BfclTaskSpec] = []
    for index, extra_info in enumerate(generation_batch["extra_info"]):
        extra = _normalize_mapping(extra_info)
        task_id = str(extra.get("task_id") or extra.get("index") or "").strip()
        if not task_id:
            raise ValueError(f"BFCL training record at index {index} is missing extra_info.task_id.")
        dataset_name = str(extra.get("dataset_name") or extra.get("split") or "train")
        tasks.append(BfclTaskSpec(task_id=task_id, dataset_name=dataset_name))
    return tasks


def run_bfcl_rollout_batch(
    *,
    tasks: Sequence[BfclTaskSpec],
    config: BfclRolloutConfig,
    generate_texts: GenerateTextsFn,
    apply_chat_template: ApplyChatTemplateFn,
    encode_text: EncodeTextFn,
    decode_ids: DecodeIdsFn | None = None,
    backend: EnvServiceBfclBackend | None = None,
    progress_callback: ProgressCallbackFn | None = None,
    generation_driver: Any | None = None,
) -> BfclRolloutBatch:
    if config.max_steps <= 0:
        raise ValueError(f"bfcl max_steps must be positive, got {config.max_steps}.")
    if config.rollout_n <= 0:
        raise ValueError(f"bfcl rollout_n must be positive, got {config.rollout_n}.")
    resolved_backend = backend or EnvServiceBfclBackend(config.env_service_url)

    started = perf_counter()
    episodes = _create_episodes(resolved_backend, tasks, config.rollout_n, config.max_env_workers)
    initialize_seconds = perf_counter() - started
    timing_metrics = run_rollout_episodes(
        episodes=episodes, config=config, backend=resolved_backend,
        generator=generation_driver or BatchedRolloutGenerator(generate_texts),
        episode_messages=_episode_messages, apply_chat_template=apply_chat_template, encode_text=encode_text,
        extract_generation=_extract_rollout_generation, build_action=build_bfcl_action,
        step_episode=_step_episode, build_context_exceeded_step=_build_context_exceeded_step,
        progress_callback=progress_callback,
    )
    timing_metrics["rollout/initialize_seconds"] = initialize_seconds

    completed_episodes = [episode for episode in episodes if episode.steps]
    tokenized = [
        tokenize_bfcl_episode(episode, apply_chat_template=apply_chat_template, encode_text=encode_text)
        for episode in completed_episodes
    ]
    if not completed_episodes:
        raise RuntimeError("BFCL rollout produced no steps for the current batch.")

    if config.save_rollouts:
        _save_rollouts(completed_episodes, config)
    if config.debug_rollouts:
        _save_tokenization_debug(completed_episodes, tokenized, config, encode_text, decode_ids)

    return BfclRolloutBatch(
        episodes=completed_episodes,
        tokenized=tokenized,
        scores=[1.0 if episode.success else 0.0 for episode in completed_episodes],
        feedback=[summarize_bfcl_feedback(episode) for episode in completed_episodes],
        metrics={
            **timing_metrics,
            **summarize_bfcl_metrics(completed_episodes),
            **summarize_tokenization_metrics(completed_episodes, tokenized, encode_text),
        },
    )


def build_bfcl_action(completion_text: str) -> tuple[dict[str, str], dict[str, Any]]:
    text = str(completion_text)
    content = text.strip()
    if not content:
        content = ""
    return ({"role": "assistant", "content": content}, {"raw_completion": text, "action_text": content or None})


def _extract_rollout_generation(
    generated_output: Any, *, prompt_token_ids: list[int], encode_text: EncodeTextFn
) -> tuple[str, list[int], list[int], str | None]:
    request_prompt_token_ids = None
    sample_output = generated_output
    if hasattr(generated_output, "outputs"):
        sample_outputs = getattr(generated_output, "outputs", []) or []
        if len(sample_outputs) != 1:
            raise RuntimeError(f"Expected exactly one BFCL completion per prompt, got {len(sample_outputs)}.")
        sample_output = sample_outputs[0]
        request_prompt_token_ids = _coerce_token_id_list(getattr(generated_output, "prompt_token_ids", None))

    completion_text = _sample_output_to_text(sample_output)
    completion_token_ids = _coerce_token_id_list(getattr(sample_output, "token_ids", None))
    if completion_token_ids is None:
        completion_token_ids = encode_text(completion_text)

    exact_prompt_token_ids = request_prompt_token_ids or _coerce_token_id_list(
        getattr(sample_output, "prompt_token_ids", None)
    )
    if exact_prompt_token_ids is None:
        exact_prompt_token_ids = list(prompt_token_ids)

    finish_reason = getattr(sample_output, "finish_reason", None)
    if finish_reason is not None:
        finish_reason = str(finish_reason)

    return completion_text, completion_token_ids, exact_prompt_token_ids, finish_reason


def _sample_output_to_text(sample_output: Any) -> str:
    if isinstance(sample_output, str):
        return sample_output
    text = getattr(sample_output, "text", None)
    if isinstance(text, str):
        return text
    return str(sample_output)


def _coerce_token_id_list(token_ids: Any) -> list[int] | None:
    if token_ids is None:
        return None
    if isinstance(token_ids, Sequence) and not isinstance(token_ids, (str, bytes, bytearray)):
        return [int(token_id) for token_id in token_ids]
    return None


def tokenize_bfcl_episode(
    episode: BfclEpisode, *, apply_chat_template: ApplyChatTemplateFn, encode_text: EncodeTextFn
) -> TokenizedBfclTrajectory:
    prompt_messages = _stringify_messages(episode.initial_state)
    response_messages = _episode_response_messages(episode)
    prompt_text = _apply_stable_chat_template(prompt_messages, False, apply_chat_template)
    prompt_ids = encode_text(prompt_text)
    current_ids = list(prompt_ids)

    completion_ids: list[int] = []
    completion_loss_mask: list[int] = []
    step_token_ranges: list[tuple[int, int]] = []
    current_assistant_start: int | None = None
    running_messages = list(prompt_messages)
    current_text = prompt_text
    incremental_encode_validated = False
    use_incremental_encode = True
    for message in response_messages:
        running_messages.append(message)
        next_text = _apply_stable_chat_template(running_messages, False, apply_chat_template)
        increment: list[int]
        next_ids: list[int]
        if use_incremental_encode and next_text.startswith(current_text):
            suffix_text = next_text[len(current_text) :]
            increment_candidate = encode_text(suffix_text) if suffix_text else []
            candidate_ids = current_ids + increment_candidate
            if not incremental_encode_validated:
                full_ids = encode_text(next_text)
                if full_ids == candidate_ids:
                    incremental_encode_validated = True
                    next_ids = candidate_ids
                    increment = increment_candidate
                else:
                    use_incremental_encode = False
                    if full_ids[: len(current_ids)] != current_ids:
                        common_length = _common_prefix_length(current_ids, full_ids)
                        raise RuntimeError(
                            "BFCL trajectory tokenization lost chat-template prefix alignment "
                            f"(prompt_len={len(prompt_ids)}, current_len={len(current_ids)}, "
                            f"next_len={len(full_ids)}, common_prefix_len={common_length}). "
                            "Check the tokenizer chat template before training."
                        )
                    next_ids = full_ids
                    increment = full_ids[len(current_ids) :]
            else:
                next_ids = candidate_ids
                increment = increment_candidate
        else:
            next_ids = encode_text(next_text)
            if next_ids[: len(current_ids)] != current_ids:
                common_length = _common_prefix_length(current_ids, next_ids)
                raise RuntimeError(
                    "BFCL trajectory tokenization lost chat-template prefix alignment "
                    f"(prompt_len={len(prompt_ids)}, current_len={len(current_ids)}, "
                    f"next_len={len(next_ids)}, common_prefix_len={common_length}). "
                    "Check the tokenizer chat template before training."
                )
            increment = next_ids[len(current_ids) :]
        offset_before = len(completion_ids)
        completion_ids.extend(increment)
        completion_loss_mask.extend([1 if message["role"] == "assistant" else 0] * len(increment))
        current_ids = next_ids
        current_text = next_text

        if message["role"] == "assistant":
            current_assistant_start = offset_before
        elif current_assistant_start is not None:
            step_token_ranges.append((current_assistant_start, offset_before))
            current_assistant_start = None

    if current_assistant_start is not None:
        step_token_ranges.append((current_assistant_start, len(completion_ids)))

    completion_text = render_bfcl_response(response_messages)
    return TokenizedBfclTrajectory(
        prompt_messages=prompt_messages,
        prompt_text=prompt_text,
        completion_text=completion_text,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion_attention_mask=[1] * len(completion_ids),
        completion_loss_mask=completion_loss_mask,
        step_token_ranges=step_token_ranges,
    )


def render_bfcl_response(messages: list[dict[str, str]]) -> str:
    chunks = []
    for message in messages:
        role = message["role"].upper()
        chunks.append(f"{role}:\n{message['content']}")
    return "\n\n".join(chunks)


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
        if left_item != right_item:
            return index
    return min(len(left), len(right))


def _apply_stable_chat_template(
    messages: list[dict[str, str]], add_generation_prompt: bool, apply_chat_template: ApplyChatTemplateFn
) -> str:
    if add_generation_prompt or not messages or messages[-1].get("role") != "assistant":
        return apply_chat_template(messages, add_generation_prompt)

    dummy_user = {"role": "user", "content": ""}
    with_dummy = apply_chat_template([*messages, dummy_user], False)
    dummy_suffix = apply_chat_template([dummy_user], False)
    if dummy_suffix and with_dummy.endswith(dummy_suffix):
        return with_dummy[: -len(dummy_suffix)]
    return apply_chat_template(messages, add_generation_prompt)


def summarize_bfcl_feedback(episode: BfclEpisode) -> str | None:
    if episode.success or not episode.steps:
        return None
    if episode.run_metadata.get("context_exceeded") is True:
        return None
    if episode.terminated and episode.final_reward > 0.0:
        return None
    last_step = episode.steps[-1]
    state_text = "\n".join(str(message.get("content", "")) for message in last_step.state)
    info_text = json.dumps(last_step.info, ensure_ascii=False, sort_keys=True) if last_step.info else ""
    feedback = "\n".join(part for part in [state_text, info_text] if part.strip()).strip()
    return feedback or None


def summarize_bfcl_metrics(episodes: Sequence[BfclEpisode]) -> dict[str, float]:
    num_episodes = max(len(episodes), 1)
    num_steps = sum(len(episode.steps) for episode in episodes)
    num_completed_episodes = sum(
        1
        for episode in episodes
        if any(
            CONVERSATION_COMPLETED_TOKEN in str(message.get("content", ""))
            for step in episode.steps
            for message in step.state
        )
    )
    return {
        "bfcl/success_rate": sum(float(episode.success) for episode in episodes) / num_episodes,
        "bfcl/context_exceeded_rate": (
            sum(float(episode.run_metadata.get("context_exceeded") is True) for episode in episodes) / num_episodes
        ),
        "bfcl/positive_reward_rate": (sum(float(episode.final_reward > 0.0) for episode in episodes) / num_episodes),
        "bfcl/final_reward": sum(float(episode.final_reward) for episode in episodes) / num_episodes,
        "bfcl/terminated_fraction": sum(float(episode.terminated) for episode in episodes) / num_episodes,
        "bfcl/avg_steps": num_steps / num_episodes,
        "bfcl/conversation_completed_fraction": num_completed_episodes / num_episodes,
    }


def summarize_tokenization_metrics(
    episodes: Sequence[BfclEpisode], tokenized: Sequence[TokenizedBfclTrajectory], encode_text: EncodeTextFn
) -> dict[str, float]:
    records = [
        _build_tokenization_debug_record(episode, item, encode_text, None)
        for episode, item in zip(episodes, tokenized, strict=False)
    ]
    if not records:
        return {}
    count = len(records)
    return {
        "bfcl/tokenization_avg_completion_tokens": sum(record["completion_tokens"] for record in records) / count,
        "bfcl/tokenization_avg_assistant_loss_tokens": sum(record["assistant_loss_tokens"] for record in records)
        / count,
        "bfcl/tokenization_low_mask_fraction": sum(
            float("low_assistant_loss_tokens" in record["warnings"]) for record in records
        )
        / count,
    }


def _create_episodes(
    backend: EnvServiceBfclBackend, tasks: Sequence[BfclTaskSpec], rollout_n: int, max_env_workers: int = 8
) -> list[BfclEpisode]:
    return create_rollout_episodes(backend, tasks, rollout_n, max_env_workers, BfclEpisode)


def _step_episode(
    backend: EnvServiceBfclBackend,
    episode: BfclEpisode,
    episode_index: int,
    action: dict[str, str],
    agent_metadata: dict[str, Any],
) -> tuple[int, BfclStep]:
    response = backend.step(episode.instance_id, action)
    reward = float(response.get("reward", 0.0) or 0.0)
    state = _stringify_messages(response.get("state", []))
    is_terminated = bool(response.get("is_terminated", False))
    if not is_terminated:
        for message in state:
            if CONVERSATION_COMPLETED_TOKEN in message.get("content", ""):
                is_terminated = True
                break
    return (
        episode_index,
        BfclStep(
            index=len(episode.steps),
            action=action,
            state=state,
            reward=reward,
            is_terminated=is_terminated,
            info=dict(response.get("info", {})),
            agent_metadata=agent_metadata,
        ),
    )


def _build_context_exceeded_step(
    episode: BfclEpisode, *, prompt_text: str, prompt_tokens: int, max_prompt_tokens: int
) -> BfclStep:
    message = (
        "Rollout stopped before generation because the BFCL conversation exceeded "
        f"the prompt token budget ({prompt_tokens} > {max_prompt_tokens})."
    )
    return BfclStep(
        index=len(episode.steps),
        action={"role": "assistant", "content": ""},
        state=[{"role": "user", "content": message}],
        reward=0.0,
        is_terminated=False,
        info={"context_exceeded": True, "prompt_tokens": prompt_tokens, "max_prompt_tokens": max_prompt_tokens},
        agent_metadata={
            "context_exceeded": True,
            "prompt_text": prompt_text,
            "prompt_tokens": prompt_tokens,
            "max_prompt_tokens": max_prompt_tokens,
        },
    )


def _episode_messages(episode: BfclEpisode) -> list[dict[str, str]]:
    return _stringify_messages(episode.initial_state) + _episode_response_messages(episode)


def _episode_response_messages(episode: BfclEpisode) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for step in episode.steps:
        if step.agent_metadata.get("context_exceeded") is True:
            continue
        messages.append(_stringify_message(step.action))
        messages.extend(_stringify_messages(step.state))
    return messages


def _stringify_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise TypeError(f"Expected BFCL messages to be a list, got {type(messages)!r}.")
    return [_stringify_message(message) for message in messages]


def _stringify_message(message: Any) -> dict[str, str]:
    if not isinstance(message, dict):
        raise TypeError(f"Expected BFCL message to be a dict, got {type(message)!r}.")
    role = str(message.get("role", ""))
    if role not in {"system", "user", "assistant"}:
        role = "user"
    return {"role": role, "content": str(message.get("content", ""))}


def _normalize_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] == "{":
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
    return {}


def _build_tokenization_debug_record(
    episode: BfclEpisode, tokenized: TokenizedBfclTrajectory, encode_text: EncodeTextFn, decode_ids: DecodeIdsFn | None
) -> dict[str, Any]:
    response_messages = _episode_response_messages(episode)
    assistant_messages = [message for message in response_messages if message["role"] == "assistant"]
    assistant_content_tokens = sum(len(encode_text(message["content"])) for message in assistant_messages)
    assistant_loss_tokens = int(sum(tokenized.completion_loss_mask))
    spans = _mask_spans(tokenized.completion_loss_mask)
    decoded_spans: list[dict[str, Any]] = []
    if decode_ids is not None:
        for start, end in spans[:20]:
            text = decode_ids(tokenized.completion_ids[start:end])
            decoded_spans.append({"start": start, "end": end, "num_tokens": end - start, "text": text[:1000]})

    warnings: list[str] = []
    if assistant_messages and assistant_loss_tokens == 0:
        warnings.append("no_assistant_loss_tokens")
    if assistant_content_tokens and assistant_loss_tokens < max(1, assistant_content_tokens // 2):
        warnings.append("low_assistant_loss_tokens")

    return {
        "task_id": episode.task_id,
        "instance_id": episode.instance_id,
        "success": episode.success,
        "final_reward": episode.final_reward,
        "num_steps": len(episode.steps),
        "assistant_messages": len(assistant_messages),
        "prompt_tokens": len(tokenized.prompt_ids),
        "completion_tokens": len(tokenized.completion_ids),
        "assistant_content_tokens": assistant_content_tokens,
        "assistant_loss_tokens": assistant_loss_tokens,
        "mask_spans": [{"start": start, "end": end, "num_tokens": end - start} for start, end in spans],
        "decoded_mask_spans": decoded_spans,
        "warnings": warnings,
    }


def _mask_spans(mask: Sequence[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if not value else index + 1
            spans.append((start, end))
            start = None
    return spans


def _save_rollouts(episodes: Sequence[BfclEpisode], config: BfclRolloutConfig) -> None:
    rollout_dir = Path(config.rollout_dir or "eval_outputs/bfcl/training_rollouts")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    path = rollout_dir / "rollouts.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")


def _save_tokenization_debug(
    episodes: Sequence[BfclEpisode],
    tokenized: Sequence[TokenizedBfclTrajectory],
    config: BfclRolloutConfig,
    encode_text: EncodeTextFn,
    decode_ids: DecodeIdsFn | None,
) -> None:
    debug_dir = Path(config.debug_dir or config.rollout_dir or "eval_outputs/bfcl/training_rollouts")
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / "tokenization_debug.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for episode, item in zip(episodes, tokenized, strict=False):
            record = _build_tokenization_debug_record(episode, item, encode_text, decode_ids)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
