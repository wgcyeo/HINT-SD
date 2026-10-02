from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from time import perf_counter
from typing import Any


def create_rollout_episodes(
    backend, tasks: Sequence, rollout_n: int, max_env_workers: int, episode_type: type,
) -> list:
    """Create an ordered batch, releasing every created instance if any job fails."""
    jobs = [(task, index) for task in tasks for index in range(rollout_n)]
    if not jobs:
        return []
    instance_ids: list[str | None] = [None] * len(jobs)

    def create(index):
        task, rollout_index = jobs[index]
        response = backend.create_instance(task)
        instance_id = str(response["info"]["instance_id"])
        instance_ids[index] = instance_id
        episode = episode_type(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            instance_id=instance_id,
            initial_state=list(response.get("state", [])),
            run_metadata={"rollout_index": rollout_index},
        )
        episode.tools_info = backend.get_tools_info(instance_id)
        return episode

    try:
        with ThreadPoolExecutor(max_workers=max(1, min(max_env_workers, len(jobs)))) as pool:
            futures = [pool.submit(create, index) for index in range(len(jobs))]
            try:
                return [future.result() for future in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    except BaseException as error:
        # The pool has joined: even jobs finishing after the failure are accounted for.
        for instance_id in instance_ids:
            if instance_id is None:
                continue
            try:
                if not backend.release_instance(instance_id):
                    error.add_note(f"Failed to release environment instance {instance_id}.")
            except Exception as cleanup_error:
                error.add_note(f"Failed to release environment instance {instance_id}: {cleanup_error}")
        raise


class VLLMRolloutGenerator:
    def __init__(self, llm, sampling_params, *, lora_request=None):
        from vllm.sampling_params import RequestOutputKind

        self.llm = llm
        self.lora_request = lora_request
        self.params = sampling_params.clone()
        self.params.output_kind = RequestOutputKind.FINAL_ONLY
        if llm.llm_engine.has_unfinished_requests():
            raise RuntimeError("A rollout must start with an idle generation engine.")

    def submit(self, prompt: str) -> str:
        request_id = str(next(self.llm.request_counter))
        rendered = self.llm.renderer.render_cmpl([{"prompt": prompt}])[0]
        self.llm.llm_engine.add_request(request_id, rendered, self.params, lora_request=self.lora_request)
        return request_id

    def step(self) -> list[tuple[str, Any]]:
        return [(output.request_id, output) for output in self.llm.llm_engine.step() if output.finished]

    def cancel(self, request_ids: list[str]) -> None:
        if request_ids:
            self.llm.llm_engine.abort_request(request_ids)


class BatchedRolloutGenerator:
    def __init__(self, generate_texts: Callable):
        self.generate_texts = generate_texts
        self.prompts: dict[str, str] = {}
        self.next_request_id = 0

    def submit(self, prompt: str) -> str:
        request_id = str(self.next_request_id)
        self.next_request_id += 1
        self.prompts[request_id] = prompt
        return request_id

    def step(self) -> list[tuple[str, Any]]:
        outputs = self.generate_texts(list(self.prompts.values()))
        if len(outputs) != len(self.prompts):
            raise RuntimeError(f"Expected {len(self.prompts)} rollout completions, received {len(outputs)}.")
        results = list(zip(self.prompts, outputs, strict=True))
        self.prompts.clear()
        return results

    def cancel(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self.prompts.pop(request_id, None)


def run_rollout_episodes(
    *, episodes: list, config, backend, generator, episode_messages: Callable, apply_chat_template: Callable,
    encode_text: Callable, extract_generation: Callable, build_action: Callable, step_episode: Callable,
    build_context_exceeded_step: Callable, progress_callback: Callable | None = None,
) -> dict[str, float]:
    ready = list(range(len(episodes)))
    requests: dict[str, tuple[int, str, list[int]]] = {}
    futures = {}
    completed = 0
    generation_seconds = 0.0
    env_wait_seconds = 0.0
    started = perf_counter()
    pool = ThreadPoolExecutor(max_workers=max(1, config.max_env_workers))

    def record_step(episode_index, step):
        nonlocal completed
        episode = episodes[episode_index]
        episode.steps.append(step)
        episode.terminated = step.is_terminated
        episode.final_reward = step.reward
        context_exceeded = step.agent_metadata.get("context_exceeded") is True
        if context_exceeded:
            episode.run_metadata.update(
                context_exceeded=True, context_exceeded_step_index=step.index,
                context_exceeded_prompt_tokens=step.agent_metadata.get("prompt_tokens"),
                max_prompt_tokens=config.max_prompt_tokens,
            )
        episode.success = episode.terminated and step.reward >= config.success_reward_threshold
        done = episode.terminated or context_exceeded or len(episode.steps) >= config.max_steps
        if done:
            completed += 1
        else:
            ready.append(episode_index)
        if progress_callback is not None:
            remaining = max(0, config.max_steps - len(episode.steps)) if done else 0
            progress_callback(1 + remaining, completed, len(episodes) - completed)

    try:
        while ready or requests or futures:
            for future in list(futures):
                if future.done():
                    futures.pop(future)
                    record_step(*future.result())

            for episode_index in ready:
                episode = episodes[episode_index]
                prompt = apply_chat_template(episode_messages(episode), True)
                prompt_token_ids = encode_text(prompt)
                if config.max_prompt_tokens > 0 and len(prompt_token_ids) > config.max_prompt_tokens:
                    record_step(episode_index, build_context_exceeded_step(
                        episode, prompt_text=prompt, prompt_tokens=len(prompt_token_ids),
                        max_prompt_tokens=config.max_prompt_tokens,
                    ))
                    continue
                start = perf_counter()
                request_id = generator.submit(prompt)
                generation_seconds += perf_counter() - start
                requests[request_id] = (episode_index, prompt, prompt_token_ids)
            ready.clear()

            if requests:
                start = perf_counter()
                outputs = generator.step()
                generation_seconds += perf_counter() - start
                for request_id, output in outputs:
                    episode_index, prompt, prompt_token_ids = requests.pop(request_id)
                    text, token_ids, exact_prompt_ids, finish_reason = extract_generation(
                        output, prompt_token_ids=prompt_token_ids, encode_text=encode_text,
                    )
                    action, metadata = build_action(text)
                    metadata.update(
                        prompt_text=prompt, prompt_token_ids=exact_prompt_ids, completion_token_ids=token_ids,
                        step_index=len(episodes[episode_index].steps),
                    )
                    if finish_reason is not None:
                        metadata["finish_reason"] = finish_reason
                    future = pool.submit(step_episode, backend, episodes[episode_index], episode_index, action, metadata)
                    futures[future] = episode_index
            elif futures:
                start = perf_counter()
                wait(futures, return_when=FIRST_COMPLETED)
                env_wait_seconds += perf_counter() - start
    finally:
        try:
            generator.cancel(list(requests))
        finally:
            wait(futures)
            try:
                unreleased = [episode for episode in episodes if not episode.released]
                for episode, released in zip(
                    unreleased, pool.map(lambda ep: backend.release_instance(ep.instance_id), unreleased), strict=True,
                ):
                    episode.released = released
            finally:
                pool.shutdown(wait=True)

    return {
        "rollout/seconds": perf_counter() - started,
        "rollout/generation_seconds": generation_seconds,
        "rollout/env_wait_seconds": env_wait_seconds,
    }
