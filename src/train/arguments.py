from __future__ import annotations

import argparse
from typing import Any

TRAINING_STRATEGY_CHOICES = ("no", "steps", "epoch")
ROLLOUT_BACKEND_CHOICES = ("appworld", "bfcl")


def add_base_training_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-06)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=float, default=0.0, help="Warmup step count, or a fraction below 1 of the total optimizer steps.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=-1, help="Optimizer steps; a positive value overrides --epochs.")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--save-strategy", type=str, default="steps", choices=TRAINING_STRATEGY_CHOICES)
    parser.add_argument("--eval-strategy", type=str, default=None, choices=TRAINING_STRATEGY_CHOICES)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)


def add_generation_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--feedback-temperature", type=float, default=0.6)
    parser.add_argument("--eval-temperature", type=float, default=0.6)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--eval-num-generations", type=int, default=4)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-seed", type=lambda x: None if x.lower() == "none" else int(x), default=None)
    parser.add_argument("--max-model-len", "--vllm-max-model-len", type=int, dest="max_model_len", default=24576)
    parser.add_argument("--backend", dest="rollout_backend", type=str, choices=ROLLOUT_BACKEND_CHOICES, default="bfcl")
    parser.add_argument("--appworld-env-service-url", type=str, default="http://127.0.0.1:18080")
    parser.add_argument("--appworld-max-steps", type=int, default=40)
    parser.add_argument("--appworld-success-reward-threshold", type=float, default=1.0)
    parser.add_argument("--appworld-max-env-workers", type=int, default=8)
    parser.add_argument("--appworld-save-rollouts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--appworld-rollout-dir", type=str, default=None)
    parser.add_argument("--appworld-debug-rollouts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--appworld-debug-dir", type=str, default=None)
    parser.add_argument("--appworld-hindsight-feedback-max-tokens", type=int, default=512)
    parser.add_argument("--appworld-hindsight-feedback-max-trajectory-chars", type=int, default=12000)

    parser.add_argument("--appworld-feedback-multi-step-max", type=int, default=3)
    parser.add_argument("--bfcl-env-service-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--bfcl-max-steps", type=int, default=20)
    parser.add_argument("--bfcl-success-reward-threshold", type=float, default=1.0)
    parser.add_argument("--bfcl-max-env-workers", type=int, default=8)
    parser.add_argument("--bfcl-save-rollouts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bfcl-rollout-dir", type=str, default=None)
    parser.add_argument("--bfcl-debug-rollouts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bfcl-debug-dir", type=str, default=None)
    parser.add_argument("--bfcl-hindsight-feedback-max-tokens", type=int, default=512)
    parser.add_argument("--bfcl-hindsight-feedback-max-trajectory-chars", type=int, default=12000)

    parser.add_argument("--bfcl-feedback-multi-step-max", type=int, default=3)


def add_distillation_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--distillation-topk", type=int, default=100)
    parser.add_argument("--distillation-add-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-distillation-tokens-per-batch", type=int, default=0)
    parser.add_argument("--distillation-microbatch-size", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.999)


def build_training_config_kwargs(
    *,
    args: argparse.Namespace,
    output_dir: str,
    has_eval_dataset: bool,
    extra_config_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_strategy = args.eval_strategy if has_eval_dataset else "no"
    config_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "run_name": args.run_name,
        "seed": args.seed,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "num_train_epochs": args.epochs,
        "max_steps": getattr(args, "max_steps", -1),
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "save_strategy": args.save_strategy,
        "report_to": ["wandb"] if args.report_to == "wandb" else [],
        "remove_unused_columns": False,
        "logging_first_step": True,
        "eval_strategy": eval_strategy,
        "eval_on_start": eval_strategy != "no",
    }
    if extra_config_kwargs:
        config_kwargs.update(extra_config_kwargs)
    if eval_strategy == "steps":
        config_kwargs["eval_steps"] = args.eval_steps or args.logging_steps
    if args.save_strategy == "steps":
        config_kwargs["save_steps"] = args.save_steps or args.logging_steps
    return config_kwargs


def build_hf_training_args(
    *,
    args: argparse.Namespace,
    output_dir: str,
    has_eval_dataset: bool,
    config_class: type,
    extra_config_kwargs: dict[str, Any] | None = None,
):
    config_kwargs = build_training_config_kwargs(
        args=args, output_dir=output_dir, has_eval_dataset=has_eval_dataset, extra_config_kwargs=extra_config_kwargs
    )
    return config_class(**config_kwargs)


def attach_generation_training_args(training_args: Any, args: argparse.Namespace) -> Any:
    training_args.trust_remote_code = args.trust_remote_code
    training_args.temperature = args.temperature
    training_args.feedback_temperature = args.feedback_temperature
    training_args.eval_temperature = args.eval_temperature
    training_args.max_completion_length = args.max_completion_length
    training_args.eval_num_generations = max(1, args.eval_num_generations)
    training_args.rollout_n = max(1, args.rollout_n)
    training_args.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
    training_args.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
    training_args.vllm_seed = args.vllm_seed
    training_args.max_model_len = args.max_model_len
    training_args.rollout_backend = args.rollout_backend
    training_args.appworld_env_service_url = args.appworld_env_service_url
    training_args.appworld_max_steps = args.appworld_max_steps
    training_args.appworld_success_reward_threshold = args.appworld_success_reward_threshold
    training_args.appworld_max_env_workers = args.appworld_max_env_workers
    training_args.appworld_save_rollouts = args.appworld_save_rollouts
    training_args.appworld_rollout_dir = args.appworld_rollout_dir
    training_args.appworld_debug_rollouts = args.appworld_debug_rollouts
    training_args.appworld_debug_dir = args.appworld_debug_dir
    training_args.appworld_hindsight_feedback_max_tokens = args.appworld_hindsight_feedback_max_tokens
    training_args.appworld_hindsight_feedback_max_trajectory_chars = (
        args.appworld_hindsight_feedback_max_trajectory_chars
    )
    training_args.appworld_feedback_granularity = args.appworld_feedback_granularity
    training_args.appworld_feedback_multi_step_max = args.appworld_feedback_multi_step_max
    training_args.bfcl_env_service_url = args.bfcl_env_service_url
    training_args.bfcl_max_steps = args.bfcl_max_steps
    training_args.bfcl_success_reward_threshold = args.bfcl_success_reward_threshold
    training_args.bfcl_max_env_workers = args.bfcl_max_env_workers
    training_args.bfcl_save_rollouts = args.bfcl_save_rollouts
    training_args.bfcl_rollout_dir = args.bfcl_rollout_dir
    training_args.bfcl_debug_rollouts = args.bfcl_debug_rollouts
    training_args.bfcl_debug_dir = args.bfcl_debug_dir
    training_args.bfcl_hindsight_feedback_max_tokens = args.bfcl_hindsight_feedback_max_tokens
    training_args.bfcl_hindsight_feedback_max_trajectory_chars = args.bfcl_hindsight_feedback_max_trajectory_chars
    training_args.bfcl_feedback_granularity = args.bfcl_feedback_granularity
    training_args.bfcl_feedback_multi_step_max = args.bfcl_feedback_multi_step_max
    return training_args


def attach_distillation_training_args(training_args: Any, args: argparse.Namespace) -> Any:
    training_args = attach_generation_training_args(training_args, args)
    training_args.distillation_topk = args.distillation_topk
    training_args.distillation_add_tail = args.distillation_add_tail
    training_args.max_distillation_tokens_per_batch = args.max_distillation_tokens_per_batch
    training_args.distillation_microbatch_size = args.distillation_microbatch_size
    training_args.ema_decay = args.ema_decay
    return training_args
