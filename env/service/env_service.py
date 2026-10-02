"""
Module for the EnvService class and related functionality.

This module provides an environment service that manages the creation,
execution, and release of training environment instances. It handles
the lifecycle of environment instances and exposes an API for
interacting with them.
"""

import argparse
import asyncio
import importlib
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import ray
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from .registry import Registry

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def import_and_register_env(env_name, env_file=None):
    """
    Import and register an environment class.

    Args:
        env_name (str): Name of the environment.
        env_file (str, optional): Detailed environment module name.
            Defaults to f"{env_name}_env".

    Returns:
        The registered environment class or None on failure.
    """
    try:
        if env_file is None:
            env_file = f"{env_name}_env"

        module = importlib.import_module(f"env.service.environments.{env_name}.{env_file}")

        envir_class = getattr(module, f"{env_name.capitalize()}Env")

        Registry.register(env_name)(envir_class)

        print(f"Successfully imported and registered {envir_class.__name__}")
        return envir_class
    except Exception as e:
        print(f"Error importing environment {env_name}: {e}")
        import traceback

        traceback.print_exc()
        return None


class ServiceRequest(BaseModel):
    """
    Service request class .
    """

    env_type: Optional[str] = "appworld"
    task_id: Optional[str] = None
    instance_id: Optional[str] = None
    messages: dict[str, Any] = {}
    params: dict[str, Any] = {}


class EnvService:
    """
    Manages the lifecycle of training environment instances.

    This class is responsible for creating, executing, and releasing
    environment instances. It provides methods to interact with the
    environments and handle their lifecycle.
    """

    def __init__(self):
        """
        class init
        """
        if not ray.is_initialized():
            ray.init(
                address="local",
                num_cpus=int(os.getenv("ENV_SERVICE_CPUS", "8")),
                include_dashboard=False,
                object_store_memory=512 * 1024 * 1024,
                _node_ip_address="127.0.0.1",
            )
        self.env_actors = {}
        self.env_types = {}
        self.idle_actors = {}
        self.max_idle_actors = max(0, int(os.getenv("ENV_SERVICE_IDLE_ACTORS", os.getenv("ENV_SERVICE_CPUS", "8"))))
        self.remote_env = {}
        self.last_access_time = {}
        self.cleanup_interval = 300
        self.max_idle_time = 3600

    async def cleanup_inactive_instances(self):
        """
        Periodically clean up inactive environment instances.

        Releases instances that have been idle for longer than the
        specified maximum idle time.
        """

        current_time = datetime.now()
        instances_to_release = []
        for instance_id, last_access in self.last_access_time.items():
            if (current_time - last_access) > timedelta(seconds=self.max_idle_time):
                instances_to_release.append(instance_id)

        for instance_id in instances_to_release:
            await self.release_instance(instance_id)
            print(f"Released inactive instance: {instance_id}")

    def update_access_time(self, instance_id):
        """Update the last access time for an environment instance."""
        self.last_access_time[instance_id] = datetime.now()

    def get_remote_env_cls(self, env_type: str):
        """
        Get the remote environment class for the specified environment type.

        Args:
            env_type (str): The type of environment.

        Returns:
            The remote environment class.
        """
        if env_type in self.remote_env:
            return self.remote_env[env_type]

        @ray.remote
        class RemoteEnv:
            """
            Remote environment class.
            """

            def __init__(self):
                server_dir = str(Path(__file__).resolve().parents[2])
                if server_dir not in sys.path:
                    sys.path.insert(0, server_dir)
                module = importlib.import_module(f"env.service.environments.{env_type}.{env_type}_env")
                self.env_cls = getattr(module, f"{env_type.capitalize()}Env")
                self.env = None

            def reset(self, task_id, instance_id, params):
                self.close()
                self.env = self.env_cls(task_id, instance_id, params)
                return self.env.get_init_state(params)

            def step(self, action, params):
                """remote step"""
                return self.env.step(action, params)

            def evaluate(self, messages, params):
                """remote eval"""
                return self.env.evaluate(messages, params)

            def get_info(self, messages, params):
                """remote get info"""
                return self.env.get_info(messages, params)

            def close(self):
                if self.env is not None:
                    self.env.close()
                    self.env = None

        self.remote_env[env_type] = RemoteEnv
        return RemoteEnv

    async def get_env_profile(self, env_type: str, split: str = "train", params: dict = None) -> list[str]:
        """
        Retrieve the environment profile for the specified environment type.

        Args:
            env_type (str): The type of environment.
            split (str, optional): The data split to retrieve the profile for.
                Defaults to "train".
            params (Dict, optional): Additional parameters
                for the profile retrieval.

        Returns:
            List[str]: The list of task IDs in the environment profile.
        """

        env_cls = Registry.get(env_type)
        return env_cls.get_query_list(split)

    async def get_info(self, instance_id: str, messages: dict = None, params: dict = None) -> float:
        """
        Retrieve information about an environment instance.

        Args:
            instance_id (str): The ID of the environment instance.
            messages (Dict, optional): Additional messages
                for the information request.
            params (Dict, optional): Additional parameters
                for the information request.

        Returns:
            float: The requested environment information.
        """
        self.update_access_time(instance_id)
        try:
            if instance_id not in self.env_actors:
                raise ValueError(f"Instance {instance_id} not found!")
            return await self.env_actors[instance_id].get_info.remote(messages, params)
        except Exception as e:
            print(f"Error in get_info: {str(e)}")
            raise

    async def create_instance(
        self, env_type: str, task_id: str, instance_id: Optional[str] = None, params: dict = None
    ) -> str:
        """
        Create a new instance of the specified environment.

        Args:
            env_type (str): The type of environment to create.
            task_id (str): The ID of the task associated
                with the environment.
            instance_id (Optional[str], optional):
                The ID of the environment instance.
                If not provided, a unique ID will be generated.
            params (Dict, optional): Additional parameters
                    for creating the environment instance.

        Returns:
            str: The ID of the created environment instance.
        """
        env_actor = None
        try:
            if instance_id is None:
                instance_id = f"exp_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            if instance_id in self.env_actors:
                raise ValueError(f"Instance {instance_id} already exists.")

            env_remote_cls = self.get_remote_env_cls(env_type)

            print(f"Creating instance with env_type: {env_type}, task_id: {task_id}, instance_id: {instance_id}")

            pool = self.idle_actors.setdefault(env_type, [])
            env_actor = pool.pop() if pool else env_remote_cls.remote()
            self.env_actors[instance_id] = env_actor
            self.env_types[instance_id] = env_type
            init_state = await env_actor.reset.remote(task_id, instance_id, params)

            self.update_access_time(instance_id)

            return init_state

        except Exception as e:
            if env_actor is not None:
                self.env_actors.pop(instance_id, None)
                self.env_types.pop(instance_id, None)
                self.last_access_time.pop(instance_id, None)
                ray.kill(env_actor)
            print(f"Error in create_instance: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
            print(f"sys.path: {sys.path}")
            import traceback

            traceback.print_exc()
            raise

    async def step(self, instance_id: str, action: dict, params: dict = None) -> dict:
        """
        Execute a step in the specified environment instance.

        Args:
            instance_id (str):
                The ID of the environment instance.
            action (Dict):
                The action to be executed in the environment.
            params (Dict, optional):
                Additional parameters for the step.

        Returns:
            str: The result of the step execution.
        """
        self.update_access_time(instance_id)
        try:
            if instance_id not in self.env_actors:
                raise ValueError(f"Instance {instance_id} not found!")
            data = await self.env_actors[instance_id].step.remote(action, params)

            return data

        except Exception as e:
            print(f"Error in step: {str(e)}")
            raise

    async def evaluate(self, instance_id: str, messages: dict = None, params: dict = None) -> float:
        """
        Evaluate the performance of the specified environment instance.

        Args:
            instance_id (str):
                The ID of the environment instance.
            messages (Dict, optional):
                Additional messages for the evaluation.
            params (Dict, optional):
                Additional parameters for the evaluation.

        Returns:
            float: The evaluation score.
        """
        self.update_access_time(instance_id)

        try:
            if instance_id not in self.env_actors:
                raise ValueError(f"Instance {instance_id} not found!")
            return await self.env_actors[instance_id].evaluate.remote(messages, params)
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
            raise

    async def release_instance(self, instance_id: str) -> bool:
        """
        Release the specified environment instance.

        Args:
            instance_id (str):
                The ID of the environment instance to be released.

        Returns:
            bool:
                True if the instance was successfully released,
                False otherwise.
        """
        env_actor = self.env_actors.pop(instance_id, None)
        if env_actor is None:
            return False
        env_type = self.env_types.pop(instance_id)
        self.last_access_time.pop(instance_id, None)
        try:
            await env_actor.close.remote()
        except Exception:
            ray.kill(env_actor)
            raise
        pool = self.idle_actors.setdefault(env_type, [])
        if len(pool) < self.max_idle_actors:
            pool.append(env_actor)
        else:
            ray.kill(env_actor)
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifespan of the FastAPI application.

    This context manager creates a background task
    for cleaning up inactive environment instances
    and cancels it during the shutdown process.
    """
    cleanup_task = asyncio.create_task(cleanup_loop())

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


async def cleanup_loop():
    """
    Periodically clean up inactive environment instances.

    This coroutine is run as a background task
    in the FastAPI application
    lifespan context manager.
    """
    while True:
        await asyncio.sleep(env_service.cleanup_interval)
        await env_service.cleanup_inactive_instances()


app = FastAPI(lifespan=lifespan)
env_service = EnvService()
SIM_SERVER = None


@app.get("/healthz", summary="Check the health of the API")
async def healthz():
    """
    Check the health of the API.

    Returns:
        Response: A successful response with status code 200.
    """
    return Response(content="OK", status_code=200)


@app.post("/get_env_profile")
async def handle_env_profile(request: ServiceRequest):
    """
    Retrieve the environment profile for the specified environment type.

    Args:
        request (ServiceRequest):
            The service request containing the environment type
            and optional parameters.

    Returns:
        dict: A dictionary containing the success status
        and the list of task IDs.
    """
    try:
        if request.env_type is None:
            raise ValueError("env_type is required")

        split = request.params.get("split", "train")

        task_ids = await env_service.get_env_profile(env_type=request.env_type, split=split, params=request.params)
        return {"success": True, "data": task_ids}
    except Exception as e:
        import traceback

        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=tb) from e


@app.post("/create")
async def handle_create(request: ServiceRequest):
    """
    Create a new environment instance based on the provided service request.

    This endpoint is responsible for creating a new environment instance
    with the specified environment type and task ID.

    Args:
        request (ServiceRequest):
        The service request containing environment details.

    Returns:
        dict: A dictionary with the creation status
            and initial state of the environment.

    Raises:
        HTTPException: If there's an error
            in creating the instance (400 or 500 status codes).
    """
    try:
        if not request.env_type:
            raise ValueError("env_type is required")
        if not request.task_id:
            raise ValueError("task_id is required")

        instance_id = request.instance_id

        init_state = await env_service.create_instance(
            env_type=request.env_type, task_id=request.task_id, instance_id=instance_id, params=request.params
        )
        return {"success": True, "data": init_state}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        import traceback

        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=tb) from e


@app.post("/step")
async def handle_step(request: ServiceRequest):
    """
    Execute a step in an existing environment instance.

    This endpoint allows executing an action in a specific environment instance
    identified by the instance ID.

    Args:
        request (ServiceRequest): The service request containing
            the instance ID and action details.

    Returns:
        dict: A dictionary with the step execution status and result.

    Raises:
        HTTPException: If there's an error in
            executing the step (400 or 500 status codes).
    """
    try:
        if not request.instance_id:
            raise ValueError("instance_id is required")

        result = await env_service.step(instance_id=request.instance_id, action=request.messages, params=request.params)
        return {"success": True, "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        import traceback

        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=tb) from e


@app.post("/evaluate")
async def handle_evaluate(request: ServiceRequest):
    """
    Evaluate the performance of an environment instance.

    This endpoint allows evaluating the performance of a specific environment
    instance identified by the instance ID.

    Args:
        request (ServiceRequest): The service request containing
            the instance ID and evaluation parameters.

    Returns:
        dict: A dictionary with the evaluation status and score.

    Raises:
        HTTPException: If there's an error
            in evaluating the instance (400 or 500 status codes).
    """
    try:
        if not request.instance_id:
            raise ValueError("instance_id is required")

        score = await env_service.evaluate(
            instance_id=request.instance_id, messages=request.messages, params=request.params
        )
        return {"success": True, "data": score}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        import traceback

        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=tb) from e


@app.post("/get_info")
async def handle_get_info(request: ServiceRequest):
    """
    Retrieve information about an environment instance.

    This endpoint allows fetching additional information about a specific
    environment instance identified by the instance ID.

    Args:
        request (ServiceRequest): The service request containing the
            instance ID and optional parameters for information retrieval.

    Returns:
        dict: A dictionary with the information retrieval
            status and environment info.

    Raises:
        HTTPException: If there's an error
            in retrieving the information (400 or 500 status codes).
    """
    try:
        if not request.instance_id:
            raise ValueError("instance_id is required")

        env_info = await env_service.get_info(
            instance_id=request.instance_id, messages=request.messages, params=request.params
        )
        return {"success": True, "data": env_info}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        import traceback

        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=tb) from e


@app.post("/release")
async def handle_release(request: ServiceRequest):
    """
    Release an existing environment instance.

    This endpoint allows releasing a specific environment instance
    identified by the instance ID, freeing up resources.

    Args:
        request (ServiceRequest): The service request containing
            the instance ID to be released.

    Returns:
        dict: A dictionary with the release status.

    Raises:
        HTTPException: If there's an error in
            releasing the instance (400 or 500 status codes).
    """
    try:
        if not request.instance_id:
            raise ValueError("instance_id is required")

        success = await env_service.release_instance(request.instance_id)
        return {"success": success, "data": None}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        import traceback

        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=tb) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the environment service")
    parser.add_argument("--env", type=str, default="appworld", help="Environment to use")
    parser.add_argument("--env_file_name", type=str, default=None, help="Detailed Environment name, default to the env name")

    parser.add_argument("--portal", type=str, default="0.0.0.0", help="IP address to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--debug", type=bool, default=False, help="debug mode or not")
    args = parser.parse_args()

    env_class = import_and_register_env(args.env, args.env_file_name)
    if env_class is None:
        print(f"Failed to import and register environment {args.env}")
        sys.exit(1)

    print(f"Starting server on {args.portal}:{args.port}")

    if args.debug:
        uvicorn.run(app, host=args.portal, port=args.port)
    else:
        uvicorn.run(app, host=args.portal, port=args.port, log_level="error")
