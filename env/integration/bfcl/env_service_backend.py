from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_SUCCESS_REWARD_THRESHOLD = 1.0
ENV_TYPE = "bfcl"


@dataclass(frozen=True)
class BfclTaskSpec:
    task_id: str
    dataset_name: str


@dataclass
class BfclStep:
    index: int
    action: dict[str, Any]
    state: list[dict[str, Any]]
    reward: float
    is_terminated: bool
    info: dict[str, Any] = field(default_factory=dict)
    agent_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BfclEpisode:
    task_id: str
    dataset_name: str
    instance_id: str
    initial_state: list[dict[str, Any]]
    tools_info: Any = None
    steps: list[BfclStep] = field(default_factory=list)
    run_metadata: dict[str, Any] = field(default_factory=dict)
    terminated: bool = False
    success: bool = False
    final_reward: float = 0.0
    released: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "dataset_name": self.dataset_name,
            "instance_id": self.instance_id,
            "initial_state": self.initial_state,
            "tools_info": self.tools_info,
            "steps": [asdict(step) for step in self.steps],
            "run_metadata": self.run_metadata,
            "terminated": self.terminated,
            "success": self.success,
            "final_reward": self.final_reward,
            "released": self.released,
        }


class EnvServiceBfclBackend:
    def __init__(self, base_url: str):
        from env.service.env_client import EnvClient

        self.client = EnvClient(base_url=base_url)

    def list_task_specs(self, dataset_name: str) -> list[BfclTaskSpec]:
        task_ids = self.client.get_env_profile(ENV_TYPE, split=dataset_name)
        if not task_ids:
            raise RuntimeError(
                f"BFCL env_service returned no task ids for split '{dataset_name}'. "
                "Check the server logs and BFCL_DATA_PATH/BFCL_SPLID_ID_PATH availability."
            )
        return [BfclTaskSpec(task_id=str(task_id), dataset_name=dataset_name) for task_id in task_ids]

    def create_instance(self, task: BfclTaskSpec, *, prompt: bool = True) -> dict[str, Any]:
        response = self.client.create_instance(ENV_TYPE, task.task_id, params={"prompt": prompt})
        instance_id = str(response.get("info", {}).get("instance_id", "")).strip()
        fallback_message = self._extract_first_content(response.get("state"))
        if not instance_id:
            raise RuntimeError(f"BFCL create_instance returned no instance_id for task '{task.task_id}'.")
        if "create failed" in fallback_message:
            raise RuntimeError(
                f"BFCL create_instance failed for task '{task.task_id}'. "
                "The env_service client returned its fallback payload instead of a live instance."
            )
        return response

    def get_tools_info(self, instance_id: str, *, prompt: bool = False) -> Any:
        return self.client.get_tools_info(instance_id, params={"prompt": prompt})

    def step(self, instance_id: str, action: dict[str, Any]) -> dict[str, Any]:
        response = self.client.step(instance_id, action)
        fallback_message = self._extract_first_content(response.get("state"))
        if "Step failed" in fallback_message:
            raise RuntimeError(
                f"BFCL step failed for instance '{instance_id}'. "
                "The env_service client returned its retry fallback instead of a real transition."
            )
        return response

    def evaluate(self, instance_id: str, *, sparse: bool = True) -> float:
        return float(self.client.evaluate(instance_id, messages={}, params={"sparse": sparse}))

    def release_instance(self, instance_id: str) -> bool:
        return bool(self.client.release_instance(instance_id))

    @staticmethod
    def _extract_first_content(messages: Any) -> str:
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                return str(first.get("content", ""))
        return ""
