"""Evaluate a base model or a HINT-SD adapter on BFCL or AppWorld."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftConfig
from transformers import set_seed

from src.data.dataset import load_records
from src.env_utils import load_repo_dotenv
from src.eval.interactive_eval import InteractiveEvalConfig, evaluate_interactive_records
from src.eval.output import default_output_path, save_metrics

ENV_OPTIONS = {
    "BACKEND": "--backend",
    "ENV_SERVICE_URL": "--env-service-url",
    "EVAL_FILE": "--data-path",
    "EVAL_NUM_GENERATIONS": "--num-generations",
    "MAX_ENV_STEPS": "--max-env-steps",
    "MAX_MODEL_LEN": "--max-model-len",
    "MAX_COMPLETION_LENGTH": "--max-completion-length",
    "EVAL_BATCH_SIZE": "--batch-size",
    "EVAL_LIMIT": "--limit",
    "EVAL_TEMPERATURE": "--temperature",
    "SEED": "--seed",
    "VLLM_GPU_MEMORY_UTILIZATION": "--vllm-gpu-memory-utilization",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-or-path", help="Model ID, local model directory, or local LoRA adapter.")
    parser.add_argument("--model-name-or-path", help="Explicit backbone, including for remote adapters.")
    parser.add_argument("--adapter-path")
    parser.add_argument("--data-path")
    parser.add_argument("--backend", choices=("bfcl", "appworld"), default="bfcl")
    parser.add_argument("--env-service-url")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", help="Exact metrics JSON path.")
    output_group.add_argument("--output-dir", default=os.getenv("EVAL_OUTPUT_DIR"))
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without loading model weights.")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-env-steps", "--max-steps", dest="max_steps", type=int, help="Environment actions per rollout (BFCL: 20, AppWorld: 40).")
    parser.add_argument("--max-model-len", type=int, default=24576)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    env_args = [part for key, flag in ENV_OPTIONS.items() if os.getenv(key) for part in (flag, os.environ[key])]
    args = parser.parse_args(env_args + sys.argv[1:])
    if args.model_or_path:
        if (Path(args.model_or_path) / "adapter_config.json").is_file():
            if args.adapter_path:
                parser.error("Specify only one adapter path.")
            args.adapter_path = args.model_or_path
        else:
            if args.model_name_or_path or args.adapter_path:
                parser.error("Use MODEL_OR_PATH or --model-name-or-path/--adapter-path.")
            args.model_name_or_path = args.model_or_path
    if not args.model_name_or_path:
        args.model_name_or_path = (
            PeftConfig.from_pretrained(args.adapter_path).base_model_name_or_path
            if args.adapter_path
            else os.getenv("MODEL_NAME_OR_PATH", "Qwen/Qwen3-4B-Instruct-2507")
        )
    args.data_path = args.data_path or f"data/{args.backend}/test.jsonl"
    default_port = 8080 if args.backend == "bfcl" else 18080
    args.env_service_url = args.env_service_url or f"http://127.0.0.1:{default_port}"
    args.max_steps = args.max_steps if args.max_steps is not None else (20 if args.backend == "bfcl" else 40)
    if not args.output:
        args.output = str(
            Path(args.output_dir) / "metrics.json" if args.output_dir else default_output_path(
                model=args.model_name_or_path, adapter=args.adapter_path, backend=args.backend,
            )
        )
    return args


def main() -> None:
    load_repo_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return
    if Path(args.output).exists():
        raise FileExistsError(f"Evaluation output already exists: {args.output}. Choose a new --output or --output-dir.")
    if torch.cuda.device_count() != 1:
        raise ValueError("Evaluation requires one visible GPU per process.")
    set_seed(args.seed)
    records = load_records(args.data_path, "eval")
    if args.limit is not None:
        records = records[: args.limit]
    if any(record["data_source"] != args.backend for record in records):
        raise ValueError("Every record must match --backend.")
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    lora_request = None
    lora_kwargs = {}
    if args.adapter_path:
        adapter_config = PeftConfig.from_pretrained(args.adapter_path)
        rank = max([adapter_config.r, *getattr(adapter_config, "rank_pattern", {}).values()])
        lora_kwargs = {"enable_lora": True, "max_lora_rank": rank}
        lora_request = LoRARequest("student", 1, args.adapter_path)
    llm = LLM(
        model=args.model_name_or_path, dtype="bfloat16", seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization, max_model_len=args.max_model_len,
        max_num_seqs=max(1, args.batch_size * args.num_generations), **lora_kwargs,
    )
    config = InteractiveEvalConfig(
        backend=args.backend,
        env_service_url=args.env_service_url,
        max_steps=args.max_steps,
        max_model_len=args.max_model_len,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
    )
    try:
        if lora_request is not None:
            llm.llm_engine.add_lora(lora_request)
        metrics = evaluate_interactive_records(
            llm=llm, tokenizer=llm.get_tokenizer(), lora_request=lora_request,
            records=records, config=config, phase="eval", global_step=0,
        )
    finally:
        llm.llm_engine.engine_core.shutdown()
    metrics.update(generation_backend="vllm", gpu_memory_utilization=args.vllm_gpu_memory_utilization, batch_size=args.batch_size, num_generations=args.num_generations, seed=args.seed)
    metrics["config"] = vars(args)
    save_metrics(args.output, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
