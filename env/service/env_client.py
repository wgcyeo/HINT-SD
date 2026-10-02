import os
import random
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

DEFAULT_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "env_service_client" / "error.out"
LOG_PATH = os.environ.get("CLIENT_LOG_PATH", str(DEFAULT_LOG_PATH))


def safe_log(msg: str):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {msg}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass  # Avoid letting logging failures affect the RL main process.


def retry_call(
    fn: Callable,
    max_retry: int = 3,
    min_backoff: float = 3.0,
    max_backoff: float = 10.0,
    fail_return: Any = None,
    err_prefix: str = "",
    instance_id: str = "",
    action_name: str = "",
    raise_on_failure: bool = False,
):
    if max_retry < 1:
        raise ValueError("max_retry must be positive.")
    for i in range(max_retry):
        try:
            res = fn()
            if i > 0:
                safe_log(f"{err_prefix} {action_name} [instance={instance_id}] succeed at try {i + 1}/{max_retry}")
            return res
        except Exception as e:
            safe_log(f"{err_prefix} {action_name} [instance={instance_id}] retry {i + 1}/{max_retry} failed: {e}")
            if i + 1 == max_retry:
                if raise_on_failure:
                    raise
                safe_log(f"{err_prefix} {action_name} [instance={instance_id}] max retries exceeded, fallback used.")
                return fail_return
            wait = random.uniform(min_backoff, max_backoff)
            time.sleep(wait)
    return fail_return


class EnvClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.timeout = 150.0 + random.uniform(50, 200)

    def _make_request(
        self,
        endpoint: str,
        env_type: str = "default",
        task_id: str = None,
        instance_id: str = None,
        messages: dict[str, Any] = None,
        params: dict[str, Any] = None,
    ) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        data = {
            "env_type": env_type,
            "task_id": task_id,
            "instance_id": instance_id,
            "messages": messages or {},
            "params": params or {},
        }
        try:
            response = requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            safe_log(f"[{endpoint}] _make_request failed (instance={instance_id}): {e}, data: {data}")
            raise Exception(f"Request failed: {str(e)}, data: {data}")

    def get_env_profile(
        self, env_type: str, split: str = "train", params: Optional[dict] = None, max_retry: int = 3
    ) -> list[str]:
        def call():
            response = self._make_request(
                endpoint="/get_env_profile", env_type=env_type, params={"split": split, **(params or {})}
            )
            if "data" in response:
                return response["data"]
            elif "task_ids" in response:
                return response["task_ids"]
            else:
                return []

        return retry_call(
            call, max_retry=max_retry, fail_return=[], err_prefix="[get_env_profile]", action_name="get_env_profile"
        )

    def get_tools_info(self, instance_id: str, messages: dict = {}, params: dict = {}, max_retry: int = 3) -> Any:
        def call():
            response = self._make_request(
                endpoint="get_info", instance_id=instance_id, messages=messages, params=params
            )
            return response.get("data", None)

        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=None,
            err_prefix="[get_tools_info]",
            instance_id=instance_id,
            action_name="get_tools_info",
        )

    def create_instance(
        self,
        env_type: str,
        task_id: str,
        instance_id: Optional[str] = None,
        params: Optional[dict] = None,
        max_retry: int = 3,
    ) -> dict:
        fallback = {
            "state": [
                {"role": "system", "content": "create query failed, this is a empty task."},
                {"role": "user", "content": "create failed, this is a empty task,please close this task."},
            ],
            "reward": 0,
            "is_terminated": False,
            "info": {"instance_id": instance_id or "", "task_id": task_id or ""},
        }

        def call():
            r = self._make_request(
                endpoint="create", env_type=env_type, task_id=task_id, instance_id=instance_id, params=params
            )
            return r["data"]

        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=fallback,
            err_prefix="[create_instance]",
            instance_id=instance_id,
            action_name="create_instance",
        )

    def step(self, instance_id: str, action: dict = {}, params: dict = {}, max_retry: int = 3) -> dict:
        fallback = {
            "state": [{"role": "assistant", "content": "Step failed (timeout or exception),please retry"}],
            "reward": 0,
            "is_terminated": False,
            "info": {"instance_id": instance_id or "", "task_id": ""},
        }

        def call():
            resp = self._make_request(endpoint="step", instance_id=instance_id, messages=action, params=params)
            return resp["data"]

        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=fallback,
            err_prefix="[step]",
            instance_id=instance_id,
            action_name="step",
        )

    def evaluate(self, instance_id: str, messages: dict = {}, params: dict = {}, max_retry: int = 3) -> float:
        def call():
            resp = self._make_request(endpoint="evaluate", instance_id=instance_id, messages=messages, params=params)
            return resp["data"]

        return retry_call(
            call,
            max_retry=max_retry,
            err_prefix="[evaluate]",
            instance_id=instance_id,
            action_name="evaluate",
            raise_on_failure=True,
        )

    def release_instance(self, instance_id: str, max_retry: int = 3) -> bool:
        def call():
            resp = self._make_request(endpoint="release", instance_id=instance_id)
            return resp.get("success", False)

        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=False,
            err_prefix="[release_instance]",
            instance_id=instance_id,
            action_name="release_instance",
        )


# Example usage
def main():
    client = EnvClient()
    env_type = "appworld"

    # Get the task list
    task_ids = client.get_env_profile(env_type)
    print(f"Available tasks: {task_ids}")

    # Create an instance
    task_id = task_ids[0] if task_ids else None
    if not task_id:
        print("Task list is empty; cannot create an instance.")
        return
    init_response = client.create_instance(env_type, task_id)
    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response.get("state", [])
    print(f"Created instance {instance_id} with query: {query}")

    # Run one action step
    action = {"role": "assistant", "content": "print('hello appworld!!')"}
    result = client.step(instance_id, action)
    print(f"Step result: {result}")

    # Evaluate
    score = client.evaluate(instance_id)
    print(f"Evaluation score: {score}")

    # Release the instance
    success = client.release_instance(instance_id)
    print(f"Instance released: {success}")


if __name__ == "__main__":
    main()
