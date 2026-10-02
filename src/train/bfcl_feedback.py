from __future__ import annotations

import json

from env.integration.bfcl.env_service_backend import BfclEpisode
from src.train.appworld_feedback import (
    clip_text_tail,
    format_messages_for_detection,
    parse_multi_step_failure_json,
    parse_step_failure_json,
)


def render_episode_for_detection(episode: BfclEpisode, env_feedback: str | None, *, max_chars: int = 12000) -> str:
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
                "You analyze failed BFCL function-calling trajectories. "
                "Identify the FIRST step where the agent made a mistake — wrong tool, "
                "wrong arguments, or missing required call. "
                "Write the feedback in less than three sentences. "
                "Output valid JSON only, no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review this failed BFCL multi-turn attempt and identify the first problematic step.\n\n"
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
                "You analyze failed BFCL function-calling trajectories. "
                f"Identify up to {max_steps} problematic steps where the agent made mistakes — "
                "wrong tool, wrong arguments, or missing required call "
                f"(between 1 and {max_steps} steps total, ordered earliest first). "
                "For EACH problematic step, write a distinct correction in less than three sentences "
                "that targets only that step's mistake. Output valid JSON only, no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review this failed BFCL multi-turn attempt and identify the problematic steps. "
                "Return one (step, feedback) pair per problematic step.\n\n"
                f"{trajectory_text}\n\n"
                "Output format (JSON only, no other text):\n"
                '{"failures": [{"step": <1-indexed step number>, '
                '"feedback": "<correction for this step>"}, ...]}'
            ),
        },
    ]


__all__ = [
    "build_multi_step_failure_detection_messages",
    "build_step_failure_detection_messages",
    "clip_text_tail",
    "format_messages_for_detection",
    "parse_multi_step_failure_json",
    "parse_step_failure_json",
    "render_episode_for_detection",
]
