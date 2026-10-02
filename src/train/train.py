from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

from src.data.dataset import load_training_and_eval_datasets, prompt_batch_collator, validate_dataset_columns
from src.env_utils import init_wandb_run, load_repo_dotenv
from src.eval.metrics import build_compute_eval_metrics
from src.train.arguments import (
    add_base_training_cli_args,
    add_distillation_cli_args,
    add_generation_cli_args,
    attach_distillation_training_args,
    build_hf_training_args,
)

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

ENV_OPTIONS = {
    "MODEL_NAME_OR_PATH": "--model-name-or-path",
    "BACKEND": "--backend",
    "VARIANT": "--variant",
    "TRAIN_FILE": "--train-file",
    "EVAL_FILE": "--eval-file",
    "OUTPUT_DIR": "--output-dir",
    "RUN_NAME": "--run-name",
    "ENV_SERVICE_URL": "--env-service-url",
    "MAX_STEPS": "--max-steps",
    "MAX_ENV_STEPS": "--max-env-steps",
    "FEEDBACK_MULTI_STEP_MAX": "--feedback-multi-step-max",
    "FEEDBACK_MAX_TOKENS": "--feedback-max-tokens",
    "EMA_DECAY": "--ema-decay",
    "BATCH_SIZE": "--batch-size",
    "GRADIENT_ACCUMULATION_STEPS": "--gradient-accumulation-steps",
    "ROLLOUT_N": "--rollout-n",
    "EPOCHS": "--epochs",
    "LEARNING_RATE": "--learning-rate",
    "MAX_MODEL_LEN": "--max-model-len",
    "MAX_COMPLETION_LENGTH": "--max-completion-length",
    "DISTILLATION_MICROBATCH_SIZE": "--distillation-microbatch-size",
    "VLLM_GPU_MEMORY_UTILIZATION": "--vllm-gpu-memory-utilization",
    "EVAL_NUM_GENERATIONS": "--eval-num-generations",
    "EVAL_BATCH_SIZE": "--eval-batch-size",
    "EVAL_TEMPERATURE": "--eval-temperature",
    "REPORT_TO": "--report-to",
    "SEED": "--seed",
    "TEMPERATURE": "--temperature",
    "FEEDBACK_TEMPERATURE": "--feedback-temperature",
    "WARMUP_STEPS": "--warmup-steps",
    "LOGGING_STEPS": "--logging-steps",
    "SAVE_STEPS": "--save-steps",
    "SAVE_TOTAL_LIMIT": "--save-total-limit",
    "EVAL_STEPS": "--eval-steps",
    "RESUME_FROM_CHECKPOINT": "--resume-from-checkpoint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HINT-SD training for BFCL and AppWorld.")
    parser.add_argument("--model-name-or-path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--train-file", type=str, default=None)
    parser.add_argument("--eval-file", type=str, default=None, help="Defaults to data/<backend>/eval.jsonl unless --data-path is provided.")
    parser.add_argument("--output-dir", help="Defaults to <output-root>/<run-name>.")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--run-name", help="Defaults to <model>-<backend>-hint-sd-<variant>.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="wandb")
    parser.add_argument("--wandb-run-id", help="Resume an existing W&B run with resume='must'.")
    parser.add_argument("--variant", choices=("multi", "single"), default="multi")
    parser.add_argument("--env-service-url", help="URL for the selected benchmark server.")
    parser.add_argument("--max-env-steps", type=int, help="Environment actions per rollout (BFCL: 20, AppWorld: 40).")
    parser.add_argument("--feedback-multi-step-max", type=int, help="Maximum hindsight targets per failed rollout.")
    parser.add_argument("--feedback-max-tokens", type=int, help="Hindsight generation token budget.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without loading a model.")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default=None)

    add_base_training_cli_args(parser)
    add_generation_cli_args(parser)
    add_distillation_cli_args(parser)

    env_args = [part for key, flag in ENV_OPTIONS.items() if os.getenv(key) for part in (flag, os.environ[key])]
    args = parser.parse_args(env_args + sys.argv[1:])
    if args.train_file is None and args.data_path is None:
        args.train_file = f"data/{args.rollout_backend}/train.jsonl"
    if args.eval_file is None and args.data_path is None:
        args.eval_file = f"data/{args.rollout_backend}/eval.jsonl"
    if args.eval_strategy is None:
        args.eval_strategy = "epoch" if args.eval_file or args.data_path else "no"
    for backend in ("appworld", "bfcl"):
        setattr(args, f"{backend}_feedback_granularity", "step" if args.variant == "single" else "multi_step")
    for value, suffix in (
        (args.env_service_url, "env_service_url"),
        (args.max_env_steps, "max_steps"),
        (args.feedback_multi_step_max, "feedback_multi_step_max"),
        (args.feedback_max_tokens, "hindsight_feedback_max_tokens"),
    ):
        if value is not None:
            setattr(args, f"{args.rollout_backend}_{suffix}", value)
    if not args.run_name:
        model_name = Path(args.model_name_or_path.rstrip("/")).name
        args.run_name = f"{model_name}-{args.rollout_backend}-hint-sd-{args.variant}"
    args.output_dir = args.output_dir or str(Path(args.output_root) / args.run_name)
    return args


def parse_target_modules(value: str | None) -> list[str] | None:
    if value is None:
        return None
    modules = [module.strip() for module in value.split(",") if module.strip()]
    return modules or None


def apply_dual_adapter_lora(model: torch.nn.Module, args: argparse.Namespace):
    peft_config_kwargs: dict[str, Any] = {
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    }
    target_modules = parse_target_modules(args.lora_target_modules)
    if target_modules is None:
        model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
        if "olmo" in model_type.lower():
            target_modules = ["q_proj", "v_proj"]
    if target_modules is not None:
        peft_config_kwargs["target_modules"] = target_modules

    peft_config = LoraConfig(**peft_config_kwargs)
    model = get_peft_model(model, peft_config, adapter_name="student")
    model.add_adapter("teacher", peft_config)
    model.set_adapter("student")
    return model


def build_training_args(args: argparse.Namespace, *, has_eval_dataset: bool) -> TrainingArguments:
    training_args = build_hf_training_args(
        args=args,
        output_dir=args.output_dir,
        has_eval_dataset=has_eval_dataset,
        config_class=TrainingArguments,
        extra_config_kwargs={
            "bf16": torch.cuda.is_available(),
            "gradient_checkpointing": args.gradient_checkpointing,
            "ddp_find_unused_parameters": False,
            "load_best_model_at_end": args.load_best_model_at_end and has_eval_dataset,
            "metric_for_best_model": "eval_score",
            "greater_is_better": True,
        },
    )
    return attach_distillation_training_args(training_args, args)


def main() -> None:
    load_repo_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return
    if torch.cuda.device_count() != 1:
        raise ValueError("Colocated training requires one visible CUDA GPU. Set CUDA_VISIBLE_DEVICES to a single GPU.")
    from src.train.trainer import Trainer

    set_seed(args.seed)

    train_dataset, eval_dataset = load_training_and_eval_datasets(
        data_path=args.data_path, train_file=args.train_file, eval_file=args.eval_file
    )
    validate_dataset_columns(train_dataset, "train")
    if eval_dataset is not None:
        validate_dataset_columns(eval_dataset, "eval")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, dtype=dtype, trust_remote_code=args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False

    model = apply_dual_adapter_lora(model, args)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    training_args = build_training_args(args, has_eval_dataset=eval_dataset is not None and args.eval_strategy != "no")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=prompt_batch_collator,
        compute_metrics=build_compute_eval_metrics(getattr(training_args, "eval_num_generations", 1)),
        distillation_topk=args.distillation_topk,
        distillation_add_tail=args.distillation_add_tail,
        max_distillation_tokens_per_batch=args.max_distillation_tokens_per_batch,
        distillation_microbatch_size=args.distillation_microbatch_size,
        ema_decay=args.ema_decay,
    )

    wandb_run = None
    try:
        wandb_run = init_wandb_run(args, is_main_process=trainer.accelerator.is_main_process)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(args.output_dir)
        trainer.save_state()
        if eval_dataset is not None:
            trainer.evaluate()
    finally:
        trainer.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
