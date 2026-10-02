from __future__ import annotations

import re


def extract_python_action(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        raise RuntimeError("Model agent returned an empty completion for the current AppWorld step.")

    pattern = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(stripped)
    if matches:
        action = matches[-1].strip()
        if not action:
            raise RuntimeError("Model agent returned an empty Python code block for the current AppWorld step.")
        return action

    if stripped.startswith("```") and stripped.endswith("```"):
        fence_lines = stripped.splitlines()
        if len(fence_lines) >= 3:
            action = "\n".join(fence_lines[1:-1]).strip()
            if action:
                return action

    return None
