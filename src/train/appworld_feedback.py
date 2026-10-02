from __future__ import annotations

import json
from typing import Any

from env.integration.appworld.env_service_backend import AppWorldEpisode
from src.train.self_distillation import remove_thinking_trace


def format_messages_for_detection(messages: list[dict[str, Any]]) -> str:
    chunks = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = str(message.get("content", "")).strip()
        if content:
            chunks.append(f"{role}:\n{content}")
    return "\n\n".join(chunks)


def clip_text_tail(text: str, *, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return "[earlier trajectory omitted]\n" + text[-max_chars:]


def render_episode_for_detection(episode: AppWorldEpisode, env_feedback: str | None, *, max_chars: int = 12000) -> str:
    initial = format_messages_for_detection(episode.initial_state)
    step_blocks = []
    for step in episode.steps:
        action = str(step.action.get("content", "")).strip()
        observation = format_messages_for_detection(step.state)
        info_text = json.dumps(step.info, ensure_ascii=False, sort_keys=True) if step.info else ""
        step_blocks.append(
            "\n".join(
                part
                for part in [
                    f"Step {step.index + 1}",
                    f"Assistant action:\n{action}",
                    f"Environment output:\n{observation}" if observation else "",
                    f"Step reward: {step.reward}",
                    f"Terminated: {step.is_terminated}",
                    f"Info: {info_text}" if info_text else "",
                ]
                if part
            )
        )

    trajectory = "\n\n".join(step_blocks)
    trajectory = clip_text_tail(trajectory, max_chars=max_chars)
    parts = [
        f"Task id: {episode.task_id}",
        f"Dataset: {episode.dataset_name}",
        f"Final reward: {episode.final_reward}",
        f"Success: {episode.success}",
        f"Terminated: {episode.terminated}",
        f"Initial task prompt:\n{initial}",
        f"Attempt trajectory:\n{trajectory}",
    ]
    if env_feedback is not None and str(env_feedback).strip():
        parts.append(f"Last environment feedback:\n{env_feedback}")
    return "\n\n".join(parts)


def build_step_failure_detection_messages(trajectory_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You analyze failed AppWorld tool-use trajectories. "
                "Identify the FIRST step where the agent made a mistake. "
                "Write the feedback in less than three sentences. "
                "Output valid JSON only, no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review this failed AppWorld attempt and identify the first problematic step.\n\n"
                f"{trajectory_text}\n\n"
                "Output format (JSON only, no other text):\n"
                '{"failure_step": <1-indexed step number>, "feedback": "<correction sentence(s)>"}'
            ),
        },
    ]


def build_multi_step_failure_detection_messages(trajectory_text: str, *, max_steps: int = 3) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You analyze failed AppWorld tool-use trajectories. "
                f"Identify up to {max_steps} problematic steps where the agent made mistakes "
                f"(between 1 and {max_steps} steps total, ordered earliest first). "
                "For EACH problematic step, write a distinct correction in less than three sentences "
                "that targets only that step's mistake. Output valid JSON only, no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review this failed AppWorld attempt and identify the problematic steps. "
                "Return one (step, feedback) pair per problematic step.\n\n"
                f"{trajectory_text}\n\n"
                "Output format (JSON only, no other text):\n"
                '{"failures": [{"step": <1-indexed step number>, '
                '"feedback": "<correction for this step>"}, ...]}'
            ),
        },
    ]


def parse_multi_step_failure_json(raw_text: str, num_steps: int, *, max_steps: int = 3) -> list[tuple[int, str]] | None:
    cleaned = remove_thinking_trace(str(raw_text)).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None

    raw_pairs = parsed.get("failures")
    aggregate_feedback: str | None = None
    if not isinstance(raw_pairs, list):
        legacy_feedback = parsed.get("feedback")
        if isinstance(legacy_feedback, str) and legacy_feedback.strip():
            aggregate_feedback = legacy_feedback.strip()
        legacy_steps = parsed.get("failure_steps")
        if not isinstance(legacy_steps, list):
            legacy_single = parsed.get("failure_step")
            if isinstance(legacy_single, (int, float)):
                legacy_steps = [legacy_single]
        if isinstance(legacy_steps, list) and aggregate_feedback:
            raw_pairs = [
                {"step": entry, "feedback": aggregate_feedback}
                for entry in legacy_steps
                if isinstance(entry, (int, float))
            ]
        else:
            return None

    pairs: list[tuple[int, str]] = []
    seen: set[int] = set()
    for entry in raw_pairs:
        if not isinstance(entry, dict):
            continue
        raw_step = entry.get("step")
        raw_feedback = entry.get("feedback", "")
        if not isinstance(raw_step, (int, float)) or not isinstance(raw_feedback, str):
            continue
        idx = int(raw_step) - 1
        feedback = raw_feedback.strip()
        if idx < 0 or idx >= num_steps or not feedback or idx in seen:
            continue
        seen.add(idx)
        pairs.append((idx, feedback))

    if not pairs:
        return None
    pairs.sort(key=lambda item: item[0])
    if max_steps > 0:
        pairs = pairs[:max_steps]
    return pairs


def parse_step_failure_json(raw_text: str, num_steps: int) -> tuple[int, str] | None:
    cleaned = remove_thinking_trace(str(raw_text)).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    failure_step = parsed.get("failure_step")
    feedback = parsed.get("feedback", "")
    if not isinstance(failure_step, (int, float)) or not isinstance(feedback, str):
        return None
    step_index = int(failure_step) - 1
    if step_index < 0 or step_index >= num_steps:
        return None
    feedback = feedback.strip()
    if not feedback:
        return None
    return (step_index, feedback)
