# Env Service API Documentation

## Overview

This document describes all interfaces provided by the Env service, including endpoint behavior, parameter descriptions, return formats, and example access patterns. The service supports environment profile queries, instance management, step execution, and related environment-interaction workflows.

## Basic Information

- Default service URL: `http://localhost:8000`
- All endpoints use the `POST` method
- Data exchange format: `JSON`
- Timeout: approximately 150-350 seconds (randomized)

## API Details

### 1. Get Environment Profile

#### Description
Retrieve configuration information for a specific environment type. The primary return value is a list of available task IDs.

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| env_type | string | Yes | Environment type, such as `"appworld"` |
| split | string | No | Dataset split, default is `"train"` |
| params | dict | No | Additional parameters |

#### Return Value
- Type: `List[str]`
- Description: List of task IDs

#### Examples

##### EnvClient
```python
from env_client import EnvClient

client = EnvClient(base_url="http://localhost:8000")
task_ids = client.get_env_profile(env_type="appworld", split="train")
print(f"Available tasks: {task_ids}")
```

##### curl
```bash
curl -X POST http://localhost:8000/get_env_profile \
  -H "Content-Type: application/json" \
  -d '{
    "env_type": "appworld",
    "params": {"split": "train"}
  }'
```

##### HTTP Request
```
POST /get_env_profile HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "env_type": "appworld",
  "params": {"split": "train"}
}
```

##### Python requests
```python
import requests

url = "http://localhost:8000/get_env_profile"
data = {"env_type": "appworld", "params": {"split": "train"}}

response = requests.post(url, json=data, timeout=350)
task_ids = response.json().get("data", [])
print(f"Available tasks: {task_ids}")
```

### 2. Get Tool Information

#### Description
Retrieve tool information for a specific instance.

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| instance_id | string | Yes | Instance ID |
| messages | dict | No | Message payload |
| params | dict | No | Additional parameters |

#### Return Value
- Type: `Any`
- Description: Tool information; exact structure depends on the environment implementation

#### Examples

##### EnvClient
```python
from env_client import EnvClient

client = EnvClient()
instance_id = "your-instance-id"
tools_info = client.get_tools_info(instance_id=instance_id)
print(f"Tools info: {tools_info}")
```

##### curl
```bash
curl -X POST http://localhost:8000/get_info \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "your-instance-id",
    "messages": {},
    "params": {}
  }'
```

##### HTTP Request
```
POST /get_info HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "instance_id": "your-instance-id",
  "messages": {},
  "params": {}
}
```

##### Python requests
```python
import requests

url = "http://localhost:8000/get_info"
data = {"instance_id": "your-instance-id", "messages": {}, "params": {}}

response = requests.post(url, json=data, timeout=350)
tools_info = response.json().get("data", None)
print(f"Tools info: {tools_info}")
```

### 3. Create Instance

#### Description
Create a new instance for the specified environment type and task ID.

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| env_type | string | Yes | Environment type |
| task_id | string | Yes | Task ID |
| instance_id | string | No | Instance ID; generated automatically if omitted |
| params | dict | No | Additional parameters |

#### Return Value
- Type: `dict`
- Structure:
  ```python
  {
      "state": [{"role": str, "content": str}, ...],  # State messages
      "reward": float,  # Reward value
      "is_terminated": bool,  # Whether the episode is terminated
      "info": {"instance_id": str, "task_id": str},  # Instance metadata
  }
  ```

#### Examples

##### EnvClient
```python
from env_client import EnvClient

client = EnvClient()
env_type = "appworld"
task_id = "task-123"
init_response = client.create_instance(env_type, task_id)
print(f"Created instance: {init_response['info']['instance_id']}")
```

##### curl
```bash
curl -X POST http://localhost:8000/create \
  -H "Content-Type: application/json" \
  -d '{
    "env_type": "appworld",
    "task_id": "task-123",
    "instance_id": "optional-instance-id",
    "params": {}
  }'
```

##### HTTP Request
```
POST /create HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "env_type": "appworld",
  "task_id": "task-123",
  "instance_id": "optional-instance-id",
  "params": {}
}
```

##### Python requests
```python
import requests

url = "http://localhost:8000/create"
data = {"env_type": "appworld", "task_id": "task-123", "instance_id": "optional-instance-id", "params": {}}

response = requests.post(url, json=data, timeout=350)
init_response = response.json().get("data", {})
print(f"Created instance: {init_response.get('info', {}).get('instance_id')}")
```

### 4. Execute Step

#### Description
Execute one action step on the specified instance.

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| instance_id | string | Yes | Instance ID |
| action | dict | No | Action payload, usually in the form `{"role": str, "content": str}` |
| params | dict | No | Additional parameters |

#### Return Value
- Type: `dict`
- Structure:
  ```python
  {
      "state": [{"role": str, "content": str}, ...],  # Updated state after execution
      "reward": float,  # Reward value
      "is_terminated": bool,  # Whether the episode is terminated
      "info": {"instance_id": str, "task_id": str},  # Instance metadata
  }
  ```

#### Examples

##### EnvClient
```python
from env_client import EnvClient

client = EnvClient()
instance_id = "your-instance-id"
action = {"role": "assistant", "content": "print('hello world')"}
result = client.step(instance_id, action)
print(f"Step result: {result}")
```

##### curl
```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "your-instance-id",
    "messages": {"role": "assistant", "content": "print('\''hello world'\'')"},
    "params": {}
  }'
```

##### HTTP Request
```
POST /step HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "instance_id": "your-instance-id",
  "messages": {"role": "assistant", "content": "print('hello world')"},
  "params": {}
}
```

##### Python requests
```python
import requests

url = "http://localhost:8000/step"
data = {
    "instance_id": "your-instance-id",
    "messages": {"role": "assistant", "content": "print('hello world')"},
    "params": {},
}

response = requests.post(url, json=data, timeout=350)
result = response.json().get("data", {})
print(f"Step result: {result}")
```

### 5. Evaluate Instance

#### Description
Evaluate the specified instance and return an evaluation score.

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| instance_id | string | Yes | Instance ID |
| messages | dict | No | Evaluation-related messages |
| params | dict | No | Additional parameters |

#### Return Value
- Type: `float`
- Description: Evaluation score

#### Examples

##### EnvClient
```python
from env_client import EnvClient

client = EnvClient()
instance_id = "your-instance-id"
score = client.evaluate(instance_id)
print(f"Evaluation score: {score}")
```

##### curl
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "your-instance-id",
    "messages": {},
    "params": {}
  }'
```

##### HTTP Request
```
POST /evaluate HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "instance_id": "your-instance-id",
  "messages": {},
  "params": {}
}
```

##### Python requests
```python
import requests

url = "http://localhost:8000/evaluate"
data = {"instance_id": "your-instance-id", "messages": {}, "params": {}}

response = requests.post(url, json=data, timeout=350)
score = response.json().get("data", 0.0)
print(f"Evaluation score: {score}")
```

### 6. Release Instance

#### Description
Release resources associated with the specified instance.

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| instance_id | string | Yes | Instance ID |

#### Return Value
- Type: `bool`
- Description: Returns `True` on successful release, otherwise `False`

#### Examples

##### EnvClient
```python
from env_client import EnvClient

client = EnvClient()
instance_id = "your-instance-id"
success = client.release_instance(instance_id)
print(f"Instance released: {success}")
```

##### curl
```bash
curl -X POST http://localhost:8000/release \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "your-instance-id"
  }'
```

##### HTTP Request
```
POST /release HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "instance_id": "your-instance-id"
}
```

##### Python requests
```python
import requests

url = "http://localhost:8000/release"
data = {"instance_id": "your-instance-id"}

response = requests.post(url, json=data, timeout=350)
success = response.json().get("success", False)
print(f"Instance released: {success}")
```

## End-to-End Example

The example below demonstrates the full Env service workflow, from retrieving task IDs to releasing the instance.

### Full EnvClient Workflow
```python
from env_client import EnvClient


def full_workflow_demo():
    # Initialize the client
    client = EnvClient(base_url="http://localhost:8000")
    env_type = "appworld"

    # 1. Get environment profile and task list
    task_ids = client.get_env_profile(env_type)
    print(f"Available tasks: {task_ids}")

    if not task_ids:
        print("No tasks available, exiting.")
        return

    # 2. Create an instance
    task_id = task_ids[0]
    init_response = client.create_instance(env_type, task_id)
    print("Initial state:", init_response)

    instance_id = init_response["info"]["instance_id"]
    print(f"Created instance: {instance_id}")

    # 3. Get tool information
    tools_info = client.get_tools_info(instance_id)
    print(f"Tools available: {tools_info}")

    # 4. Execute a step
    action = {"role": "assistant", "content": "print('Hello from workflow!')"}
    step_result = client.step(instance_id, action)
    print(f"Step result: {step_result}")

    # 5. Evaluate the instance
    score = client.evaluate(instance_id)
    print(f"Final score: {score}")

    # 6. Release the instance
    release_success = client.release_instance(instance_id)
    print(f"Instance released successfully: {release_success}")


if __name__ == "__main__":
    full_workflow_demo()
```

## Error Handling

1. All endpoints implement retry logic, with up to 3 retries by default.
2. If retries are exhausted, a predefined fallback value is returned.
3. Error messages are written to the log file. The default path is `logs/env_service_client/error.out` relative to the repository root.
4. You can customize the log path by setting the `CLIENT_LOG_PATH` environment variable.

## Notes

1. Release instances promptly after use to avoid resource leaks.
2. Some calls may take a long time, so make sure the client timeout is sufficiently large.
3. Different environment types (`env_type`) may support different features and parameters. Refer to the documentation for the specific environment.
4. When `is_terminated` is `True`, the instance has finished and `step` should not be called again.
