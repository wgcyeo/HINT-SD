"""
This file is part of https://github.com/ShishirPatil/gorilla

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from env.service.base import BaseEnv
from env.service.environments.bfcl.env_handler import EnvHandler
from env.service.registry import Registry
from env.service.trajectory import StateMessage

os.environ.setdefault("BFCL_DATA_PATH", "./bfcl_data/multiturn_data.jsonl")
os.environ.setdefault("BFCL_ANSWER_PATH", "./bfcl_eval/possible_answer")

__all__ = ["BfclEnv"]


def parse_assistant_content_to_tool_calls(msg: dict[str, Any]) -> dict[str, Any]:
    """
    Parse tool_calls from assistant content and return a normalized message structure.
    Supports Qwen-style <tool_call>...</tool_call> tool call formatting.

    Args:
        msg (dict): Original assistant message containing the 'content' field.

    Returns:
        dict: New message structure containing 'content' and 'tool_calls'.
    """
    content = msg.get("content", "") or ""
    if not isinstance(content, str):
        content = str(content)

    tool_calls = []
    call_id_counter = 1

    pattern = r"<tool_call>\s*\n?({.*?})\s*\n?\</tool_call>"
    matches = list(re.finditer(pattern, content, re.DOTALL))

    if not matches:
        return {"role": "assistant", "content": content.strip(), "tool_calls": []}

    for match in matches:
        json_str = match.group(1).strip()
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                continue
            if "name" not in data or "arguments" not in data:
                continue

            func_name = data["name"]
            tool_call = {
                "id": f"{func_name}_{call_id_counter}",
                "type": "function",
                "function": {"name": data["name"], "arguments": data["arguments"]},
            }
            tool_calls.append(tool_call)
            call_id_counter += 1
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {json_str[:50]}... -> {e}")
            continue

    cleaned_content = re.sub(pattern, "", content, flags=re.DOTALL).strip()

    cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content).strip()

    result = {"role": "assistant", "content": cleaned_content, "tool_calls": tool_calls}

    return result


def tools_schema_to_qwen_prompt(tools_schema):
    """
    Convert tools_schema into a tool-description prompt compatible with the Qwen chat_template.

    Args:
        tools_schema (list): Tool list in the format below:
            [
                {
                    "name": "tool_name",
                    "description": "tool description",
                    "parameters": {
                        "type": "object",
                        "properties": { ... },
                        "required": [ ... ]
                    }
                }
            ]

    Returns:
        str: Complete system tool prompt containing <tools> tags.
    """
    if not tools_schema:
        return ""

    lines = []
    lines.append("\n\n# Tools\n")
    lines.append("You may call one or more functions to assist with the user query.\n")
    lines.append("You are provided with function signatures within <tools></tools> XML tags:")
    lines.append("<tools>")

    for tool in tools_schema:
        tool_json = json.dumps(tool, ensure_ascii=False, separators=(",", ":"))
        lines.append(tool_json)
    lines.append("</tools>\n")
    lines.append(
        "Important: Always use only the latest tool list provided, ignoring any functions mentioned in previous messages."
    )
    lines.append(
        "For each function call, return a json object with function name and arguments within <tool_call> and <tool_call> XML tags:"
    )
    lines.append("<tool_call>")
    lines.append('{"name": <function-name>, "arguments": <args-json-object>}')
    lines.append("</tool_call>")

    return "\n".join(lines)


def tool_message_to_qwen_text(tool_messages):
    """
    Convert messages with role='tool' into text compatible with the Qwen chat_template.
    Supports one or multiple consecutive tool messages.

    Args:
        tool_messages (list or dict): One or more tool message dictionaries.

    Returns:
        str: Text representation compatible with the Qwen template.
    """
    if isinstance(tool_messages, dict):
        tool_messages = [tool_messages]

    if not tool_messages:
        return ""

    tool_entries = []
    for msg in tool_messages:
        if msg.get("role") != "tool":
            raise ValueError("All messages must have role 'tool'")

        content = msg.get("content", "")
        tool_call_id = msg.get("tool_call_id", "")
        # BFCL returns tool call IDs rather than tool names.
        name = msg.get("name", tool_call_id)

        if not name:
            raise ValueError("Missing 'name' in tool message.")

        try:
            if isinstance(content, str):
                parsed_content = json.loads(content) if content.strip().startswith(("{", "[")) else content
            else:
                parsed_content = content
        except Exception:
            parsed_content = content

        entry = {"name": name, "content": parsed_content}
        tool_entries.append(f"<tool_call>\n{json.dumps(entry, ensure_ascii=False)}\n</tool_call>")

    inner_text = "\n".join(tool_entries) + "\n"

    return inner_text


@Registry.register("bfcl")
class BfclEnv(BaseEnv):
    """Berkeley-Function-Calling-Leaderboard multi-turn dialogue environment."""

    def __init__(
        self, task_id: str | None = None, instance_id: str | None = None, params: dict[str, Any] | None = None
    ):
        self.task_id, self.instance_id = task_id, instance_id
        self.params: dict[str, Any] = params or {}

        self.data_path = self.params.get("data_path", os.getenv("BFCL_DATA_PATH"))
        self.answer_path = self.params.get("answer_path", os.getenv("BFCL_ANSWER_PATH"))
        self.model_name = self.params.get("model_name", "env_handler")

        self.test_entry: dict[str, Any] | None = None
        self.original_test_entry: dict[str, Any] | None = None
        self.env_handler: EnvHandler | None = None
        self.conversation_history: list[dict[str, Any]] = []
        self.current_turn = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.tools_info = ""

    def get_init_state(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Load the test case and return the first user message."""
        self.test_entry = self._load_test_case(self.data_path, self.task_id)
        self.original_test_entry = self.test_entry

        self.env_handler = EnvHandler(model_name=self.model_name, answer_path=Path(self.answer_path))

        self.conversation_history = self.test_entry.get("question", [[]])[0].copy()
        self.current_turn = 0

        tools = self.test_entry.get("function", [])

        self.tools_info = "Available tools:\n" + "\n".join(
            f"- {t.get('function', {}).get('name', 'unknown')}" for t in tools
        )

        first_query = self.conversation_history[0]["content"] if self.conversation_history else ""

        tool_prompt = tools_schema_to_qwen_prompt(tools)
        return {
            "state": [{"role": "system", "content": tool_prompt}, {"role": "user", "content": first_query}],
            "info": {
                "instance_id": self.instance_id,
                "task_id": self.task_id,
                "test_id": self.test_entry.get("id", "unknown"),
                "tools_count": len(tools),
                "questions_count": len(self.original_test_entry.get("question", [])),
            },
        }

    def step(self, action: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:

        cur_turn = self.current_turn
        state_msg = self.transition(action, params=params or {})

        terminated = self._is_terminated(state_msg.simple_dict["content"])

        if cur_turn != self.current_turn and (self.params.get("is_open_query", False) == True):
            terminated = True
        reward = self.evaluate(params={"sparse": True}) if terminated else 0.0

        return {"state": [state_msg.simple_dict], "reward": reward, "is_terminated": terminated, "info": {}}

    def transition(self, assistant_entry: dict[str, Any], params: dict[str, Any]) -> StateMessage:
        """Execute one assistant action and let EnvHandler produce the response."""

        assistant_entry = parse_assistant_content_to_tool_calls(assistant_entry)

        self.conversation_history.append(assistant_entry)

        if self.env_handler is None or self.original_test_entry is None:
            raise RuntimeError("EnvHandler not initialised – call get_init_state() first.")

        env_resp = self.env_handler.interact(self.conversation_history, self.original_test_entry)

        next_msg_content = ""

        for idx, msg in enumerate(env_resp.get("messages", [])):
            self.conversation_history.append(msg)
            if msg["role"] == "tool":
                next_msg_content += tool_message_to_qwen_text(msg)
            elif msg["role"] == "user":
                next_msg_content = msg.get("content", "")
                # TODO: Include updated tool schemas when advancing to a new user turn.
                self.current_turn += 1
            elif msg["role"] == "env":
                next_msg_content = msg.get("content", "")

        return StateMessage(role="user", content=next_msg_content)

    def evaluate(self, messages: dict[str, Any] | None = None, params: dict[str, Any] | None = None):
        """Evaluate the dialogue by calling EnvHandler."""
        if self.env_handler is None:
            raise RuntimeError("EnvHandler not initialised – cannot evaluate.")

        conv_result = {
            "test_id": self.test_entry.get("id", "unknown"),
            "messages": self.conversation_history,
            "turn_count": self.current_turn,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "completed": self._is_terminated(self.conversation_history[-1]["content"]),
            "original_test_entry": self.original_test_entry,
        }
        sparse = (params or {}).get("sparse", False)
        result = self.env_handler.evaluate(conv_result)
        if not result.get("valid", False):
            raise RuntimeError(f"BFCL evaluation failed for {self.task_id}: {result.get('error', 'invalid result')}")
        return result.get("accuracy", 0.0) if sparse else result

    def get_info(self, messages: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> str:
        return self.tools_info

    def close(self):
        if self.env_handler is not None and self.test_entry is not None:
            from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils

            # The SDK caches mutable tool instances in module globals across tasks.
            for suffix in ("", "_eval", "_ground_truth_eval"):
                model_name = self.env_handler.model_name + suffix
                for class_name in self.test_entry.get("involved_classes", []):
                    instance_name = f"{model_name}_{self.test_entry['id']}_{class_name}_instance"
                    vars(multi_turn_utils).pop(re.sub(r"[-./:]", "_", instance_name), None)
        self.conversation_history.clear()

    def _is_terminated(self, env_content) -> bool:

        return env_content == "[CONVERSATION_COMPLETED]"

    @staticmethod
    def _load_test_case(data_path: str, test_id: str | None) -> dict[str, Any]:
        """Load a single JSONL test case by ID or line number. Raise if not found."""
        if not Path(data_path).exists():
            raise FileNotFoundError(f"BFCL data file '{data_path}' not found")

        if test_id is None:
            raise ValueError("task_id is required")

        with open(data_path, "r", encoding="utf-8") as f:
            if str(test_id).isdigit():
                idx = int(test_id)
                for line_no, line in enumerate(f):
                    if line_no == idx:
                        return json.loads(line)
                raise ValueError(f"Test case index {idx} not found in {data_path}")
            else:
                for line in f:
                    data = json.loads(line)
                    if data.get("id") == test_id:
                        return data
                raise ValueError(f"Test case id '{test_id}' not found in {data_path}")

    @staticmethod
    def get_query_list(split: str = "train", params={"category": ["multi_turn_base"]}):
        """
        Get query list from preprocessed dataset.

        Args:
            split: Dataset split, either 'train' or 'test'
            params: Parameters to filter dataset (currently supports 'category')

        Returns:
            List of query id
        """

        path = os.getenv("BFCL_SPLID_ID_PATH")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)[split]
