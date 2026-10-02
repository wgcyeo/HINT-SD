from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
from torch.nn.functional import log_softmax
from transformers import PreTrainedModel, TrainerCallback
from transformers import Trainer as HFTrainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils import logging

from env.integration.appworld.env_service_backend import AppWorldEpisode
from env.integration.bfcl.env_service_backend import BfclEpisode
from src.eval.metrics import PREDICTION_METRIC_NAMES
from src.train.appworld_feedback import (
    build_multi_step_failure_detection_messages,
    build_step_failure_detection_messages,
    parse_multi_step_failure_json,
    parse_step_failure_json,
    render_episode_for_detection,
)
from src.train.appworld_rollout import (
    AppWorldRolloutBatch,
    AppWorldRolloutConfig,
    _stringify_message,
    _stringify_messages,
    build_appworld_tasks,
    run_appworld_rollout_batch,
)
from src.train.bfcl_feedback import (
    build_multi_step_failure_detection_messages as build_bfcl_multi_step_failure_detection_messages,
)
from src.train.bfcl_feedback import build_step_failure_detection_messages as build_bfcl_step_failure_detection_messages
from src.train.bfcl_feedback import parse_multi_step_failure_json as parse_bfcl_multi_step_failure_json
from src.train.bfcl_feedback import render_episode_for_detection as render_bfcl_episode_for_detection
from src.train.bfcl_rollout import BfclRolloutBatch, BfclRolloutConfig, build_bfcl_tasks, run_bfcl_rollout_batch
from src.train.bfcl_rollout import _stringify_message as _bfcl_stringify_message
from src.train.bfcl_rollout import _stringify_messages as _bfcl_stringify_messages
from src.train.rollout import VLLMRolloutGenerator
from src.train.self_distillation import compute_distillation_loss

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
logger = logging.get_logger(__name__)


class TeacherEMACallback(TrainerCallback):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self._last_updated_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.global_step <= 0 or state.global_step == self._last_updated_step:
            return control
        if self.trainer._update_teacher_adapter_ema(self.trainer.ema_decay):
            self._last_updated_step = state.global_step
        return control


class Trainer(HFTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args,
        train_dataset,
        processing_class,
        *,
        student_adapter_name: str = "student",
        teacher_adapter_name: str = "teacher",
        distillation_topk: int | None = 100,
        distillation_add_tail: bool = True,
        max_distillation_tokens_per_batch: int = 0,
        distillation_microbatch_size: int = 0,
        ema_decay: float = 0.999,
        **kwargs,
    ):
        super().__init__(
            model=model, args=args, train_dataset=train_dataset, processing_class=processing_class, **kwargs
        )
        self.student_adapter_name = student_adapter_name
        self.teacher_adapter_name = teacher_adapter_name
        self.teacher_adapter_subdir = "teacher_adapter"
        self.model_accepts_loss_kwargs = False
        self.distillation_topk = distillation_topk
        self.distillation_add_tail = distillation_add_tail
        self.max_distillation_tokens_per_batch = max(0, int(max_distillation_tokens_per_batch or 0))
        self.distillation_microbatch_size = max(0, int(distillation_microbatch_size or 0))
        self.ema_decay = ema_decay
        self._mark_vllm_weights_stale()
        self._latest_rollout_metrics: dict[str, float] = {}

        self.processing_class.padding_side = "left"
        self._copy_adapter_state(self.student_adapter_name, self.teacher_adapter_name)
        self._freeze_adapter_parameters(self.teacher_adapter_name)
        self._set_active_adapter(self.student_adapter_name)
        self.add_callback(TeacherEMACallback(self))

        self.llm = self._init_vllm_engine()

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self._latest_rollout_metrics:
            logs = {**logs, **self._latest_rollout_metrics}
        return super().log(logs, *args, **kwargs)

    def _distillation_microbatch_size(self) -> int:
        return max(
            0,
            int(
                getattr(self, "distillation_microbatch_size", getattr(self.args, "distillation_microbatch_size", 0))
                or 0
            ),
        )

    @staticmethod
    def _trim_left_padding_columns(ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.dim() != 2 or mask.dim() != 2 or ids.size(1) == 0:
            return ids, mask
        active_columns = torch.nonzero(mask.to(dtype=torch.bool).any(dim=0), as_tuple=False).flatten()
        if active_columns.numel() == 0:
            return ids[:, -1:], mask[:, -1:]
        first_active = int(active_columns[0].item())
        if first_active <= 0:
            return ids, mask
        return ids[:, first_active:], mask[:, first_active:]

    def _init_vllm_engine(self):
        if self.accelerator.num_processes != 1 or getattr(self.args, "vllm_tensor_parallel_size", 1) != 1:
            raise ValueError("Colocated training supports one process and vLLM tensor_parallel_size=1.")
        dist_env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_PORT", "MASTER_ADDR"]
        backup_env = {key: os.environ[key] for key in dist_env_vars if key in os.environ}
        backup_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        local_rank_str = str(self.accelerator.local_process_index)

        for key in dist_env_vars:
            os.environ.pop(key, None)

        if backup_cvd is not None:
            visible_devices = backup_cvd.split(",")
            if int(local_rank_str) < len(visible_devices):
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices[int(local_rank_str)]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = local_rank_str

        try:
            from vllm import LLM

            llm_kwargs: dict[str, Any] = {
                "model": self.model.config._name_or_path,
                "seed": self.args.seed if getattr(self.args, "vllm_seed", None) is None else self.args.vllm_seed,
                "trust_remote_code": getattr(self.args, "trust_remote_code", False),
                "tensor_parallel_size": getattr(self.args, "vllm_tensor_parallel_size", 1),
                "gpu_memory_utilization": getattr(self.args, "vllm_gpu_memory_utilization", 0.5),
            }
            max_model_len = getattr(self.args, "max_model_len", None)
            if max_model_len is not None:
                llm_kwargs["max_model_len"] = max_model_len
            return LLM(**llm_kwargs)
        finally:
            for key, value in backup_env.items():
                os.environ[key] = value
            if backup_cvd is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = backup_cvd
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def close(self) -> None:
        """Release the colocated inference engine and its process groups."""
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        self.llm.llm_engine.engine_core.shutdown()
        cleanup_dist_env_and_memory()

    def _unwrap_adapter_model(self) -> PreTrainedModel:
        return self.accelerator.unwrap_model(self.model)

    def _get_active_adapter_name(self, model: PreTrainedModel | None = None) -> str:
        resolved_model = model or self._unwrap_adapter_model()
        active_adapters = getattr(resolved_model, "active_adapters", None)
        if active_adapters:
            return active_adapters[0]
        active_adapter = getattr(resolved_model, "active_adapter", None)
        if isinstance(active_adapter, str) and active_adapter:
            return active_adapter
        raise ValueError("Expected a PEFT model with an active adapter.")

    def _set_active_adapter(self, adapter_name: str) -> None:
        model = self._unwrap_adapter_model()
        if self._get_active_adapter_name(model) != adapter_name:
            model.set_adapter(adapter_name)

    @contextmanager
    def _use_adapter(self, adapter_name: str):
        model = self._unwrap_adapter_model()
        previous_adapter = self._get_active_adapter_name(model)
        if previous_adapter != adapter_name:
            model.set_adapter(adapter_name)
        try:
            yield model
        finally:
            if previous_adapter != adapter_name:
                model.set_adapter(previous_adapter)

    def _copy_adapter_state(self, source_adapter_name: str, target_adapter_name: str) -> None:
        model = self._unwrap_adapter_model()
        adapter_state = get_peft_model_state_dict(model, adapter_name=source_adapter_name)
        set_peft_model_state_dict(model, adapter_state, adapter_name=target_adapter_name)

    def _freeze_adapter_parameters(self, adapter_name: str) -> None:
        model = self._unwrap_adapter_model()
        marker = f".{adapter_name}."
        for name, parameter in model.named_parameters():
            if marker in name:
                parameter.requires_grad_(False)

    def _iter_student_teacher_parameter_pairs(self):
        model = self._unwrap_adapter_model()
        parameter_map = dict(model.named_parameters())
        student_marker = f".{self.student_adapter_name}."
        teacher_marker = f".{self.teacher_adapter_name}."
        matched = 0

        for name, student_param in parameter_map.items():
            if student_marker not in name:
                continue
            teacher_name = name.replace(student_marker, teacher_marker, 1)
            teacher_param = parameter_map.get(teacher_name)
            if teacher_param is None:
                raise KeyError(f"Teacher parameter '{teacher_name}' is missing from the self-distill model.")
            matched += 1
            yield name, student_param, teacher_param

        if matched == 0:
            raise ValueError("Did not find any student adapter parameters for self-distillation.")

    def _update_teacher_adapter_ema(self, decay: float) -> bool:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA decay must be in the interval [0, 1].")
        if getattr(self.accelerator, "optimizer_step_was_skipped", False):
            return False

        with torch.no_grad():
            for _, student_param, teacher_param in self._iter_student_teacher_parameter_pairs():
                teacher_param.lerp_(
                    student_param.detach().to(device=teacher_param.device, dtype=teacher_param.dtype), 1.0 - decay
                )
        self._mark_vllm_weights_stale()
        return True

    def _copy_adapter_artifacts(self, source_dir: Path, target_dir: Path, adapter_name: str) -> None:
        adapter_dir = source_dir / adapter_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "README.md"):
            target_path = target_dir / filename
            if target_path.exists():
                target_path.unlink()
        readme_path = source_dir / "README.md"
        if readme_path.exists():
            shutil.copy2(readme_path, target_dir / "README.md")
        if not adapter_dir.is_dir():
            raise ValueError(f"Expected saved adapter directory '{adapter_dir}', but it was not created.")
        for child in adapter_dir.iterdir():
            if child.is_file():
                shutil.copy2(child, target_dir / child.name)

    def _save_adapter_to_dir(self, adapter_name: str, target_dir: str) -> None:
        model = self._unwrap_adapter_model()
        output_dir = Path(target_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{adapter_name}_save_", dir=str(output_dir.parent)) as temp_dir:
            temp_path = Path(temp_dir)
            model.save_pretrained(
                str(temp_path),
                safe_serialization=getattr(self.args, "save_safetensors", True),
                selected_adapters=[adapter_name],
                save_embedding_layers="auto",
            )
            self._copy_adapter_artifacts(temp_path, output_dir, adapter_name)

    def _load_adapter_from_dir(self, adapter_name: str, source_dir: str, *, strict: bool = False) -> str | None:
        adapter_dir = Path(source_dir)
        if not adapter_dir.is_dir():
            if strict:
                raise ValueError(f"Expected adapter checkpoint at '{adapter_dir}', but it does not exist.")
            return None
        model = self._unwrap_adapter_model()
        adapter_state_dict = load_peft_weights(str(adapter_dir), device="cpu")
        set_peft_model_state_dict(model, adapter_state_dict, adapter_name=adapter_name)
        return str(adapter_dir)

    def _save(self, output_dir: str | None = None, state_dict=None):
        del state_dict
        resolved_output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(resolved_output_dir, exist_ok=True)
        logger.info("Saving self-distill adapter checkpoint to %s", resolved_output_dir)

        self._save_adapter_to_dir(self.student_adapter_name, resolved_output_dir)
        self._save_adapter_to_dir(
            self.teacher_adapter_name, str(Path(resolved_output_dir) / self.teacher_adapter_subdir)
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(resolved_output_dir)
        torch.save(self.args, os.path.join(resolved_output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        del model
        checkpoint_dir = Path(resume_from_checkpoint)
        logger.info("Loading self-distill adapters from %s", checkpoint_dir)
        self._load_adapter_from_dir(self.student_adapter_name, str(checkpoint_dir), strict=True)
        self._load_adapter_from_dir(
            self.teacher_adapter_name, str(checkpoint_dir / self.teacher_adapter_subdir), strict=True
        )
        self._set_active_adapter(self.student_adapter_name)
        self._mark_vllm_weights_stale()

    def _load_best_model(self) -> None:
        self._load_from_checkpoint(self.state.best_model_checkpoint)

    def _apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
        kwargs["enable_thinking"] = False
        try:
            return self.processing_class.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.processing_class.apply_chat_template(messages, **kwargs)

    def _encode_text(
        self,
        text: str,
        *,
        max_length: int | None = None,
        truncation_side: str = "right",
        error_on_overflow: bool = False,
    ) -> list[int]:
        previous_truncation_side = getattr(self.processing_class, "truncation_side", "right")
        if truncation_side in {"left", "right"}:
            self.processing_class.truncation_side = truncation_side
        try:
            encoded = self.processing_class(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=max_length is not None and not error_on_overflow,
                max_length=max_length,
                return_attention_mask=False,
            )
        finally:
            self.processing_class.truncation_side = previous_truncation_side

        input_ids = encoded.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if error_on_overflow and max_length is not None and len(input_ids) > max_length:
            raise ValueError(
                f"Reprompt exceeded max length {max_length} with truncation disabled: {len(input_ids)} tokens."
            )
        return list(input_ids)

    def _encode_texts_batched_with_truncation(
        self, texts: list[str], *, max_length: int | None, truncation_side: str, error_on_overflow: bool
    ) -> list[list[int]]:
        if not texts:
            return []
        previous_truncation_side = getattr(self.processing_class, "truncation_side", "right")
        if truncation_side in {"left", "right"}:
            self.processing_class.truncation_side = truncation_side
        try:
            encoded = self.processing_class(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=max_length is not None and not error_on_overflow,
                max_length=max_length,
                return_attention_mask=False,
            )
        finally:
            self.processing_class.truncation_side = previous_truncation_side
        input_ids = encoded.get("input_ids", [])
        results: list[list[int]] = []
        for ids in input_ids:
            ids_list = list(ids)
            if error_on_overflow and max_length is not None and len(ids_list) > max_length:
                raise ValueError(
                    f"Reprompt exceeded max length {max_length} with truncation disabled: {len(ids_list)} tokens."
                )
            results.append(ids_list)
        return results

    def _decode_ids(self, token_ids: list[int]) -> str:
        return self.processing_class.decode(token_ids, skip_special_tokens=False)

    def _pad_sequences(self, sequences: list[list[int]], padding_side: str) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.accelerator.device
        pad_token_id = self.processing_class.pad_token_id or self.processing_class.eos_token_id
        max_len = max((len(sequence) for sequence in sequences), default=0)
        padded_seqs: list[list[int]] = []
        masks: list[list[int]] = []

        for sequence in sequences:
            pad_len = max_len - len(sequence)
            if padding_side == "right":
                padded_seqs.append(sequence + [pad_token_id] * pad_len)
                masks.append([1] * len(sequence) + [0] * pad_len)
            else:
                padded_seqs.append([pad_token_id] * pad_len + sequence)
                masks.append([0] * pad_len + [1] * len(sequence))

        return (
            torch.tensor(padded_seqs, device=device, dtype=torch.long),
            torch.tensor(masks, device=device, dtype=torch.long),
        )

    def _mark_vllm_weights_stale(self) -> None:
        self._last_loaded_step = -1
        self._last_loaded_adapter_name: str | None = None

    def _sync_adapter_weights_to_vllm(self, adapter_name: str) -> None:
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_context = deepspeed.zero.GatheredParameters
        else:
            gather_context = nullcontext

        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        model = self._unwrap_adapter_model()
        is_peft = hasattr(model, "peft_config")

        if is_peft:
            from peft.tuners.lora.layer import LoraLayer

            if adapter_name not in model.peft_config:
                raise ValueError(f"Adapter '{adapter_name}' is not attached to the self-distill model.")

            with self._use_adapter(adapter_name):
                for name, module in model.named_modules():
                    if not isinstance(module, LoraLayer):
                        continue
                    if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
                        continue

                    vllm_name = name.replace("base_model.model.", "") + ".weight"
                    base_param = module.base_layer.weight
                    lora_a_param = module.lora_A[adapter_name].weight
                    lora_b_param = module.lora_B[adapter_name].weight
                    scaling = module.scaling[adapter_name]

                    with gather_context([base_param, lora_a_param, lora_b_param]):
                        merged_weight = base_param.data + (lora_b_param.data @ lora_a_param.data) * scaling
                        llm_model.load_weights([(vllm_name, merged_weight)])
        else:
            for name, param in model.named_parameters():
                vllm_name = name.replace("_checkpoint_wrapped_module.", "")
                with gather_context([param]):
                    llm_model.load_weights([(vllm_name, param.data)])

        if hasattr(self.llm, "reset_prefix_cache"):
            self.llm.reset_prefix_cache()

    def _ensure_vllm_adapter_weights(self, adapter_name: str) -> None:
        if self.state.global_step != self._last_loaded_step or adapter_name != self._last_loaded_adapter_name:
            self._sync_adapter_weights_to_vllm(adapter_name)
            self._last_loaded_step = self.state.global_step
            self._last_loaded_adapter_name = adapter_name

    def _sample_model_outputs(self, prompt_texts: list[str], num_generations: int, *, temperature: float | None = None):
        self._ensure_vllm_adapter_weights(self.student_adapter_name)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else getattr(self.args, "temperature", 1.0),
            max_tokens=getattr(self.args, "max_completion_length", 2048),
            n=max(1, int(num_generations)),
            seed=getattr(self.args, "vllm_seed", None),
        )
        return self.llm.generate(prompt_texts, sampling_params=sampling_params, use_tqdm=False)

    def _rollout_generator(self, temperature: float | None = None) -> VLLMRolloutGenerator:
        from vllm import SamplingParams

        self._ensure_vllm_adapter_weights(self.student_adapter_name)
        return VLLMRolloutGenerator(self.llm, SamplingParams(
            temperature=temperature if temperature is not None else self.args.temperature,
            max_tokens=self.args.max_completion_length, n=1, seed=self.args.vllm_seed,
        ))

    def _sample_teacher_model_texts(self, prompt_texts: list[str], *, max_new_tokens: int) -> list[str]:
        if not prompt_texts:
            return []
        max_new_tokens = max(1, int(max_new_tokens))
        max_model_len = int(getattr(self.args, "max_model_len", 0) or 0)
        max_prompt_tokens = max_model_len - max_new_tokens if max_model_len > max_new_tokens else None
        if max_prompt_tokens is not None:
            encoded_prompts = self._encode_texts_batched_with_truncation(
                prompt_texts,
                max_length=max_prompt_tokens,
                truncation_side=getattr(self.processing_class, "truncation_side", "right"),
                error_on_overflow=False,
            )
            prompt_texts = [
                self.processing_class.decode(token_ids, skip_special_tokens=False) for token_ids in encoded_prompts
            ]

        active_adapter_name = self.teacher_adapter_name
        self._ensure_vllm_adapter_weights(active_adapter_name)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=getattr(self.args, "feedback_temperature", 0.6),
            top_p=0.95,
            max_tokens=max_new_tokens,
            n=1,
            seed=getattr(self.args, "vllm_seed", None),
        )
        outputs = self.llm.generate(prompt_texts, sampling_params=sampling_params, use_tqdm=False)
        texts: list[str] = []
        for request_output in outputs:
            sample_outputs = getattr(request_output, "outputs", []) or []
            if len(sample_outputs) != 1:
                raise RuntimeError(f"Expected exactly one feedback completion per prompt, got {len(sample_outputs)}.")
            texts.append(self._sample_output_to_text(sample_outputs[0]).strip())
        return texts

    def _sample_output_to_text(self, sample_output: Any) -> str:
        if hasattr(sample_output, "text") and isinstance(sample_output.text, str):
            return sample_output.text
        token_ids = getattr(sample_output, "token_ids", None)
        if token_ids is not None:
            return self.processing_class.decode(token_ids, skip_special_tokens=False)
        return str(sample_output)

    def _render_appworld_episode_for_hindsight(self, episode: AppWorldEpisode, env_feedback: str | None) -> str:
        max_chars = int(getattr(self.args, "appworld_hindsight_feedback_max_trajectory_chars", 12000) or 12000)
        return render_episode_for_detection(episode, env_feedback, max_chars=max_chars)

    def _detect_appworld_failure_step(
        self, episodes: list[AppWorldEpisode], env_feedback: list[str | None]
    ) -> list[tuple[int, str] | None]:
        prompts: list[str] = []
        prompt_indices: list[int] = []
        for index, episode in enumerate(episodes):
            if episode.success or not episode.steps:
                continue
            messages = self._build_appworld_step_failure_detection_messages(
                episode, env_feedback[index] if index < len(env_feedback) else None
            )
            prompts.append(self._apply_chat_template(messages, add_generation_prompt=True))
            prompt_indices.append(index)

        results: list[tuple[int, str] | None] = [None] * len(episodes)
        if not prompts:
            return results

        max_new_tokens = int(getattr(self.args, "appworld_hindsight_feedback_max_tokens", 192) or 192)
        generated = self._sample_teacher_model_texts(prompts, max_new_tokens=max_new_tokens)
        for index, raw_text in zip(prompt_indices, generated, strict=False):
            parsed = self._parse_step_failure_json(raw_text, len(episodes[index].steps))
            results[index] = parsed
        return results

    def _build_appworld_step_failure_detection_messages(
        self, episode: AppWorldEpisode, env_feedback: str | None
    ) -> list[dict[str, str]]:
        trajectory = self._render_appworld_episode_for_hindsight(episode, env_feedback)
        return build_step_failure_detection_messages(trajectory)

    @staticmethod
    def _parse_step_failure_json(raw_text: str, num_steps: int) -> tuple[int, str] | None:
        return parse_step_failure_json(raw_text, num_steps)

    def _detect_appworld_failure_steps_multi(
        self, episodes: list[AppWorldEpisode], env_feedback: list[str | None], *, max_steps: int = 3
    ) -> list[list[tuple[int, str]] | None]:
        prompts: list[str] = []
        prompt_indices: list[int] = []
        for index, episode in enumerate(episodes):
            if episode.success or not episode.steps:
                continue
            messages = self._build_appworld_multi_step_failure_detection_messages(
                episode, env_feedback[index] if index < len(env_feedback) else None, max_steps=max_steps
            )
            prompts.append(self._apply_chat_template(messages, add_generation_prompt=True))
            prompt_indices.append(index)

        results: list[list[tuple[int, str]] | None] = [None] * len(episodes)
        if not prompts:
            return results

        max_new_tokens = int(getattr(self.args, "appworld_hindsight_feedback_max_tokens", 192) or 192)
        generated = self._sample_teacher_model_texts(prompts, max_new_tokens=max_new_tokens)
        for index, raw_text in zip(prompt_indices, generated, strict=False):
            parsed = self._parse_multi_step_failure_json(raw_text, len(episodes[index].steps), max_steps=max_steps)
            results[index] = parsed
        return results

    def _build_appworld_multi_step_failure_detection_messages(
        self, episode: AppWorldEpisode, env_feedback: str | None, *, max_steps: int = 3
    ) -> list[dict[str, str]]:
        trajectory = self._render_appworld_episode_for_hindsight(episode, env_feedback)
        return build_multi_step_failure_detection_messages(trajectory, max_steps=max_steps)

    @staticmethod
    def _parse_multi_step_failure_json(
        raw_text: str, num_steps: int, *, max_steps: int = 3
    ) -> list[tuple[int, str]] | None:
        return parse_multi_step_failure_json(raw_text, num_steps, max_steps=max_steps)

    def _build_per_step_teacher_prompt(
        self, episode: AppWorldEpisode, failure_step_index: int, hint: str
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = list(_stringify_messages(episode.initial_state))
        for step in episode.steps[:failure_step_index]:
            if step.agent_metadata.get("context_exceeded") is True:
                continue
            messages.append(_stringify_message(step.action))
            messages.extend(_stringify_messages(step.state))
        messages.append({"role": "user", "content": f"Consider the following feedback to solve this step:\n\n{hint}"})
        return messages

    @staticmethod
    def _coerce_rollout_token_ids(token_ids: Any) -> list[int] | None:
        if token_ids is None:
            return None
        if isinstance(token_ids, Sequence) and not isinstance(token_ids, (str, bytes, bytearray)):
            return [int(token_id) for token_id in token_ids]
        return None

    def _compute_completion_logits_at_positions(
        self,
        model: PreTrainedModel,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
        completion_positions: torch.Tensor,
    ) -> torch.Tensor:
        if completion_positions.numel() == 0:
            raise ValueError("completion_positions must not be empty.")
        max_completion_position = int(completion_positions.max().item())
        completion_prefix_len = max_completion_position
        input_ids = torch.cat([prompt_ids, completion_ids[:, :completion_prefix_len]], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_attention_mask[:, :completion_prefix_len]], dim=1)
        logit_positions = prompt_ids.size(1) + completion_positions - 1
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logit_positions)
            return outputs.logits
        except TypeError:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits.index_select(dim=1, index=logit_positions)

    def _compute_completion_log_probs_at_positions(
        self,
        model: PreTrainedModel,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
        completion_positions: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._compute_completion_logits_at_positions(
            model, prompt_ids, prompt_mask, completion_ids, completion_attention_mask, completion_positions
        )
        return log_softmax(logits, dim=-1)

    def _zero_loss(self, model: PreTrainedModel) -> torch.Tensor:
        for parameter in model.parameters():
            if parameter.requires_grad:
                return parameter.sum() * 0.0
        return torch.zeros((), device=self.accelerator.device, requires_grad=True)

    def _compute_distillation_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        *,
        backward_fn: Callable[[torch.Tensor], None] | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, float], bool]:
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        teacher_prompt_ids = inputs["teacher_prompt_ids"]
        teacher_prompt_mask = inputs["teacher_prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_attention_mask = inputs["completion_attention_mask"]
        completion_loss_mask = inputs["completion_loss_mask"]
        self_distillation_mask = inputs["self_distillation_mask"]
        active_fraction_value = float(
            inputs.get(
                "self_distillation_active_fraction",
                float(self_distillation_mask.float().mean().item()) if self_distillation_mask.numel() > 0 else 0.0,
            )
        )
        active_completion_mask = completion_loss_mask * self_distillation_mask.to(
            dtype=completion_loss_mask.dtype
        ).unsqueeze(1)
        active_completion_positions, selected_completion_mask, selection_metrics = self._select_distillation_positions(
            active_completion_mask
        )

        if active_completion_positions.numel() == 0:
            metrics = {
                "self_distillation/active_fraction": float(active_fraction_value),
                "self_distillation/empty_target_batch": 1.0,
                **selection_metrics,
            }
            return None, metrics, False

        selected_row_mask = selected_completion_mask.any(dim=1)
        if not selected_row_mask.any().item():
            metrics = {
                "self_distillation/active_fraction": float(active_fraction_value),
                "self_distillation/empty_target_batch": 1.0,
                **selection_metrics,
            }
            return None, metrics, False

        prompt_ids = prompt_ids[selected_row_mask]
        prompt_mask = prompt_mask[selected_row_mask]
        teacher_prompt_ids = teacher_prompt_ids[selected_row_mask]
        teacher_prompt_mask = teacher_prompt_mask[selected_row_mask]
        completion_ids = completion_ids[selected_row_mask]
        completion_attention_mask = completion_attention_mask[selected_row_mask]
        filtered_self_distillation_mask = self_distillation_mask[selected_row_mask]
        filtered_completion_mask = selected_completion_mask[selected_row_mask][:, active_completion_positions]

        if backward_fn is not None:
            row_count = int(prompt_ids.size(0))
            row_microbatch_size = max(1, self._distillation_microbatch_size() or row_count)
            valid_tokens = filtered_completion_mask.to(dtype=torch.float32).sum().clamp(min=1.0)
            loss_value = torch.zeros((), device=completion_ids.device, dtype=torch.float32)
            processed_microbatch_count = 0
            max_microbatch_position_count = 0

            for start in range(0, row_count, row_microbatch_size):
                end = min(start + row_microbatch_size, row_count)
                row_completion_mask = filtered_completion_mask[start:end]
                microbatch_position_offsets = torch.nonzero(
                    row_completion_mask.sum(dim=0) > 0, as_tuple=False
                ).flatten()
                if microbatch_position_offsets.numel() == 0:
                    continue

                microbatch_positions = active_completion_positions.index_select(
                    dim=0, index=microbatch_position_offsets
                )
                microbatch_completion_mask = row_completion_mask.index_select(dim=1, index=microbatch_position_offsets)
                microbatch_token_count = microbatch_completion_mask.to(dtype=torch.float32).sum()
                if float(microbatch_token_count.item()) <= 0.0:
                    continue

                microbatch_prompt_ids, microbatch_prompt_mask = self._trim_left_padding_columns(
                    prompt_ids[start:end], prompt_mask[start:end]
                )
                microbatch_teacher_prompt_ids, microbatch_teacher_prompt_mask = self._trim_left_padding_columns(
                    teacher_prompt_ids[start:end], teacher_prompt_mask[start:end]
                )

                with self._use_adapter(self.teacher_adapter_name), torch.no_grad():
                    teacher_log_probs = self._compute_completion_log_probs_at_positions(
                        model,
                        microbatch_teacher_prompt_ids,
                        microbatch_teacher_prompt_mask,
                        completion_ids[start:end],
                        completion_attention_mask[start:end],
                        microbatch_positions,
                    )

                with self._use_adapter(self.student_adapter_name):
                    student_log_probs = self._compute_completion_log_probs_at_positions(
                        model,
                        microbatch_prompt_ids,
                        microbatch_prompt_mask,
                        completion_ids[start:end],
                        completion_attention_mask[start:end],
                        microbatch_positions,
                    )

                microbatch_loss, _ = compute_distillation_loss(
                    student_log_probs=student_log_probs,
                    teacher_log_probs=teacher_log_probs,
                    completion_mask=microbatch_completion_mask,
                    distillation_topk=self.distillation_topk,
                    distillation_add_tail=self.distillation_add_tail,
                    self_distillation_mask=filtered_self_distillation_mask[start:end],
                )
                weighted_microbatch_loss = microbatch_loss * (
                    microbatch_token_count.to(dtype=microbatch_loss.dtype)
                    / valid_tokens.to(dtype=microbatch_loss.dtype)
                )
                backward_fn(weighted_microbatch_loss)
                loss_value = loss_value + weighted_microbatch_loss.detach().to(dtype=torch.float32)
                processed_microbatch_count += 1
                max_microbatch_position_count = max(max_microbatch_position_count, int(microbatch_positions.numel()))

            if processed_microbatch_count == 0:
                metrics = {
                    "self_distillation/active_fraction": float(active_fraction_value),
                    "self_distillation/empty_target_batch": 1.0,
                    **selection_metrics,
                }
                return None, metrics, False

            metrics = {
                "self_distillation/active_fraction": float(self_distillation_mask.float().mean().item()),
                "self_distillation/empty_target_batch": 0.0,
                "self_distillation/microbatch_size": float(row_microbatch_size),
                "self_distillation/microbatch_count": float(processed_microbatch_count),
                "self_distillation/max_microbatch_position_count": float(max_microbatch_position_count),
                "self_distillation/chunk_backward": 1.0,
                **selection_metrics,
            }
            return loss_value, metrics, True

        with self._use_adapter(self.student_adapter_name):
            student_log_probs = self._compute_completion_log_probs_at_positions(
                model, prompt_ids, prompt_mask, completion_ids, completion_attention_mask, active_completion_positions
            )

        with self._use_adapter(self.teacher_adapter_name), torch.no_grad():
            teacher_log_probs = self._compute_completion_log_probs_at_positions(
                model,
                teacher_prompt_ids,
                teacher_prompt_mask,
                completion_ids,
                completion_attention_mask,
                active_completion_positions,
            )

        loss, metrics = compute_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            completion_mask=filtered_completion_mask,
            distillation_topk=self.distillation_topk,
            distillation_add_tail=self.distillation_add_tail,
            self_distillation_mask=filtered_self_distillation_mask,
        )
        metrics["self_distillation/active_fraction"] = float(self_distillation_mask.float().mean().item())
        metrics["self_distillation/empty_target_batch"] = 0.0
        metrics["self_distillation/chunk_backward"] = 0.0
        metrics.update(selection_metrics)
        return loss, metrics, True

    def _select_distillation_positions(
        self, active_completion_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        active_completion_positions = torch.nonzero(active_completion_mask.sum(dim=0) > 0, as_tuple=False).flatten()
        active_loss_tokens = float(active_completion_mask.sum().item())
        max_tokens = int(
            getattr(
                self, "max_distillation_tokens_per_batch", getattr(self.args, "max_distillation_tokens_per_batch", 0)
            )
            or 0
        )
        metrics = {
            "self_distillation/active_loss_tokens": active_loss_tokens,
            "self_distillation/selected_loss_tokens": active_loss_tokens,
            "self_distillation/selected_position_count": float(active_completion_positions.numel()),
            "self_distillation/loss_token_keep_fraction": 1.0,
        }
        if max_tokens <= 0 or active_loss_tokens <= max_tokens or active_completion_positions.numel() == 0:
            return active_completion_positions, active_completion_mask, metrics

        selected_completion_mask = torch.zeros_like(active_completion_mask)
        selected_position_values: list[torch.Tensor] = []
        remaining_tokens = max_tokens
        for position in active_completion_positions:
            active_rows = torch.nonzero(active_completion_mask[:, position] > 0, as_tuple=False).flatten()
            if active_rows.numel() == 0:
                continue
            selected_position_values.append(position)
            if active_rows.numel() <= remaining_tokens:
                selected_completion_mask[:, position] = active_completion_mask[:, position]
                remaining_tokens -= int(active_rows.numel())
                if remaining_tokens == 0:
                    break
                continue

            selected_rows = active_rows[:remaining_tokens]
            selected_completion_mask[selected_rows, position] = active_completion_mask[selected_rows, position]
            break

        selected_positions = torch.stack(selected_position_values).to(active_completion_positions.device)
        selected_loss_tokens = float(selected_completion_mask.sum().item())
        metrics.update(
            {
                "self_distillation/selected_loss_tokens": selected_loss_tokens,
                "self_distillation/selected_position_count": float(selected_positions.numel()),
                "self_distillation/loss_token_keep_fraction": selected_loss_tokens / max(active_loss_tokens, 1.0),
            }
        )
        return selected_positions, selected_completion_mask, metrics

    def _resolve_rollout_backend(self, data_sources: list[Any]) -> str:
        sources = set(map(str, data_sources))
        if len(sources) != 1 or not sources <= {"appworld", "bfcl"}:
            raise ValueError("Each batch must contain only AppWorld or only BFCL records.")
        backend = next(iter(sources))
        configured = str(getattr(self.args, "rollout_backend", "auto"))
        if configured not in {"auto", backend}:
            raise ValueError(f"rollout_backend={configured} does not match data_source={backend}.")
        return backend

    def _pad_mask_sequences(self, sequences: list[list[int]], padding_side: str) -> torch.Tensor:
        device = self.accelerator.device
        max_len = max((len(sequence) for sequence in sequences), default=0)
        padded: list[list[int]] = []
        for sequence in sequences:
            pad_len = max_len - len(sequence)
            if padding_side == "right":
                padded.append(sequence + [0] * pad_len)
            else:
                padded.append([0] * pad_len + sequence)
        return torch.tensor(padded, device=device, dtype=torch.long)

    def _prepare_inputs(self, generation_batch: dict[str, Any]) -> dict[str, Any]:
        backend = self._resolve_rollout_backend(generation_batch["data_source"])
        if backend == "appworld":
            return self._prepare_appworld_inputs(generation_batch)
        return self._prepare_bfcl_inputs(generation_batch)

    def _run_appworld_rollout(
        self, generation_batch: dict[str, Any], *, rollout_n_override: int | None = None, temperature: float | None = None
    ) -> AppWorldRolloutBatch:
        config = AppWorldRolloutConfig(
            env_service_url=getattr(self.args, "appworld_env_service_url", ""),
            max_steps=int(getattr(self.args, "appworld_max_steps", 8)),
            rollout_n=max(
                1, int(rollout_n_override if rollout_n_override is not None else getattr(self.args, "rollout_n", 1))
            ),
            max_env_workers=int(getattr(self.args, "appworld_max_env_workers", 8)),
            max_prompt_tokens=max(
                0,
                int(getattr(self.args, "max_model_len", 0) or 0)
                - int(getattr(self.args, "max_completion_length", 0) or 0),
            ),
            success_reward_threshold=float(getattr(self.args, "appworld_success_reward_threshold", 1.0)),
            save_rollouts=bool(getattr(self.args, "appworld_save_rollouts", False)),
            rollout_dir=getattr(self.args, "appworld_rollout_dir", None),
            debug_rollouts=bool(getattr(self.args, "appworld_debug_rollouts", False)),
            debug_dir=getattr(self.args, "appworld_debug_dir", None),
        )
        if not config.env_service_url:
            raise ValueError("--appworld-env-service-url is required for AppWorld training rollouts.")
        return run_appworld_rollout_batch(
            tasks=build_appworld_tasks(generation_batch),
            config=config,
            generate_texts=lambda prompts: self._sample_model_outputs(prompts, num_generations=1, temperature=temperature),
            apply_chat_template=lambda messages, add_generation_prompt: self._apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt
            ),
            encode_text=self._encode_text,
            decode_ids=self._decode_ids,
            generation_driver=self._rollout_generator(temperature),
        )

    def _prepare_appworld_inputs(self, generation_batch: dict[str, Any]) -> dict[str, Any]:
        rollout_batch = self._run_appworld_rollout(generation_batch)

        base_rollout_metrics = {
            **rollout_batch.metrics,
            "rollout/mean_score": sum(rollout_batch.scores) / max(len(rollout_batch.scores), 1),
        }
        granularity = str(getattr(self.args, "appworld_feedback_granularity", "multi_step") or "multi_step")
        if granularity == "step":
            distillation_inputs = self._prepare_appworld_per_step_inputs(
                rollout_batch, rollout_metrics=base_rollout_metrics
            )
        elif granularity == "multi_step":
            distillation_inputs = self._prepare_appworld_multi_step_inputs(
                rollout_batch, rollout_metrics=base_rollout_metrics
            )
        else:
            raise ValueError(f"Unsupported feedback granularity: {granularity}")
        return distillation_inputs

    def _prepare_appworld_multi_step_inputs(
        self, rollout_batch: AppWorldRolloutBatch, *, rollout_metrics: dict[str, float] | None = None
    ) -> dict[str, Any]:
        max_steps = int(getattr(self.args, "appworld_feedback_multi_step_max", 3) or 3)
        max_steps = max(max_steps, 1)

        env_feedback = list(rollout_batch.feedback)
        detections = self._detect_appworld_failure_steps_multi(
            rollout_batch.episodes, env_feedback, max_steps=max_steps
        )

        student_prompt_ids_list: list[list[int]] = []
        teacher_prompt_ids_list: list[list[int]] = []
        completion_ids_list: list[list[int]] = []
        completion_attention_mask_list: list[list[int]] = []
        completion_loss_mask_list: list[list[int]] = []
        completion_texts: list[str] = []
        distillation_mask: list[bool] = []

        rollout_hit_count = 0
        rollout_miss_count = 0
        expanded_step_count = 0
        steps_per_hit_total = 0
        success_count = 0

        def append_passthrough(tokenized) -> None:
            prompt_ids = list(tokenized.prompt_ids)
            student_prompt_ids_list.append(prompt_ids)
            teacher_prompt_ids_list.append(prompt_ids)
            completion_ids_list.append(tokenized.completion_ids)
            completion_attention_mask_list.append(tokenized.completion_attention_mask)
            completion_loss_mask_list.append([0] * len(tokenized.completion_ids))
            completion_texts.append(tokenized.completion_text)
            distillation_mask.append(False)

        for idx, (episode, tokenized) in enumerate(zip(rollout_batch.episodes, rollout_batch.tokenized, strict=False)):
            if episode.success:
                success_count += 1
                append_passthrough(tokenized)
                continue

            detection = detections[idx]
            if detection is None or not detection:
                rollout_miss_count += 1
                append_passthrough(tokenized)
                continue

            covered_any = False
            for step_index, step_feedback in detection:
                if step_index < 0 or step_index >= len(episode.steps):
                    continue

                target_step = episode.steps[step_index]
                student_prompt_ids = self._coerce_rollout_token_ids(target_step.agent_metadata.get("prompt_token_ids"))
                if student_prompt_ids is None:
                    student_prompt_messages = list(_stringify_messages(episode.initial_state))
                    for prior in episode.steps[:step_index]:
                        if prior.agent_metadata.get("context_exceeded") is True:
                            continue
                        student_prompt_messages.append(_stringify_message(prior.action))
                        student_prompt_messages.extend(_stringify_messages(prior.state))
                    student_prompt_text = self._apply_chat_template(student_prompt_messages, add_generation_prompt=True)
                    student_prompt_ids = self._encode_text(student_prompt_text)

                step_completion_ids = self._coerce_rollout_token_ids(
                    target_step.agent_metadata.get("completion_token_ids")
                )
                if step_completion_ids is None:
                    if step_index >= len(tokenized.step_token_ranges):
                        continue
                    start, end = tokenized.step_token_ranges[step_index]
                    step_completion_ids = tokenized.completion_ids[start:end]
                if not step_completion_ids:
                    continue

                teacher_prompt_messages = self._build_per_step_teacher_prompt(episode, step_index, step_feedback)
                teacher_prompt_text = self._apply_chat_template(teacher_prompt_messages, add_generation_prompt=True)

                student_prompt_ids_list.append(student_prompt_ids)
                teacher_prompt_ids_list.append(self._encode_text(teacher_prompt_text))
                completion_ids_list.append(step_completion_ids)
                completion_attention_mask_list.append([1] * len(step_completion_ids))
                completion_loss_mask_list.append([1] * len(step_completion_ids))
                completion_texts.append("")
                distillation_mask.append(True)
                expanded_step_count += 1
                covered_any = True

            if covered_any:
                rollout_hit_count += 1
                steps_per_hit_total += sum(1 for step_index, _ in detection if 0 <= step_index < len(episode.steps))
            else:
                rollout_miss_count += 1
                append_passthrough(tokenized)

        num_episodes = max(len(rollout_batch.episodes), 1)
        multi_step_metrics = {
            "appworld/multi_step_detection_hit_fraction": rollout_hit_count / num_episodes,
            "appworld/multi_step_detection_miss_fraction": rollout_miss_count / num_episodes,
            "appworld/multi_step_skipped_success_fraction": success_count / num_episodes,
            "appworld/multi_step_steps_per_hit_mean": (
                steps_per_hit_total / rollout_hit_count if rollout_hit_count else 0.0
            ),
            "appworld/multi_step_expanded_entries": float(expanded_step_count),
        }
        merged_rollout_metrics = {
            **rollout_batch.metrics,
            **multi_step_metrics,
            **(rollout_metrics or {}),
            "rollout/mean_score": sum(rollout_batch.scores) / max(len(rollout_batch.scores), 1),
        }
        self._latest_rollout_metrics = merged_rollout_metrics

        active_indices = [i for i, active in enumerate(distillation_mask) if active]
        active_fraction = sum(distillation_mask) / max(len(distillation_mask), 1)
        kept_indices = active_indices if active_indices else list(range(len(distillation_mask)))

        kept_student_prompt_ids = [student_prompt_ids_list[i] for i in kept_indices]
        kept_teacher_prompt_ids = [teacher_prompt_ids_list[i] for i in kept_indices]
        kept_completion_ids = [completion_ids_list[i] for i in kept_indices]
        kept_completion_attention_mask = [completion_attention_mask_list[i] for i in kept_indices]
        kept_completion_loss_mask = [completion_loss_mask_list[i] for i in kept_indices]
        kept_distillation_mask = [distillation_mask[i] for i in kept_indices]
        kept_completion_texts = [completion_texts[i] for i in kept_indices]

        prompt_ids, prompt_mask = self._pad_sequences(kept_student_prompt_ids, padding_side="left")
        teacher_prompt_ids, teacher_prompt_mask = self._pad_sequences(kept_teacher_prompt_ids, padding_side="left")
        completion_ids, _ = self._pad_sequences(kept_completion_ids, padding_side="right")
        completion_attention_mask = self._pad_mask_sequences(kept_completion_attention_mask, padding_side="right")
        completion_loss_mask = self._pad_mask_sequences(kept_completion_loss_mask, padding_side="right")
        self_distillation_mask = torch.tensor(
            kept_distillation_mask, device=self.accelerator.device, dtype=torch.float32
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "completion_ids": completion_ids,
            "completion_attention_mask": completion_attention_mask,
            "completion_loss_mask": completion_loss_mask,
            "self_distillation_mask": self_distillation_mask,
            "completion_texts": kept_completion_texts,
            "self_distillation_active_fraction": float(active_fraction),
        }

    def _prepare_appworld_per_step_inputs(
        self, rollout_batch: AppWorldRolloutBatch, *, rollout_metrics: dict[str, float] | None = None
    ) -> dict[str, Any]:
        env_feedback = list(rollout_batch.feedback)
        detections = self._detect_appworld_failure_step(rollout_batch.episodes, env_feedback)

        student_prompt_ids_list: list[list[int]] = []
        teacher_prompt_ids_list: list[list[int]] = []
        completion_ids_list: list[list[int]] = []
        completion_attention_mask_list: list[list[int]] = []
        completion_loss_mask_list: list[list[int]] = []
        completion_texts: list[str] = []
        distillation_mask: list[bool] = []
        detection_hit_count = 0
        detection_miss_count = 0

        for idx, (episode, tokenized) in enumerate(zip(rollout_batch.episodes, rollout_batch.tokenized, strict=False)):
            if episode.success:
                prompt_ids = list(tokenized.prompt_ids)
                student_prompt_ids_list.append(prompt_ids)
                teacher_prompt_ids_list.append(prompt_ids)
                completion_ids_list.append(tokenized.completion_ids)
                completion_attention_mask_list.append(tokenized.completion_attention_mask)
                completion_loss_mask_list.append([0] * len(tokenized.completion_ids))
                completion_texts.append(tokenized.completion_text)
                distillation_mask.append(False)
                continue

            detection = detections[idx]
            if detection is None:
                detection_miss_count += 1
                prompt_ids = list(tokenized.prompt_ids)
                student_prompt_ids_list.append(prompt_ids)
                teacher_prompt_ids_list.append(prompt_ids)
                completion_ids_list.append(tokenized.completion_ids)
                completion_attention_mask_list.append(tokenized.completion_attention_mask)
                completion_loss_mask_list.append([0] * len(tokenized.completion_ids))
                completion_texts.append(tokenized.completion_text)
                distillation_mask.append(False)
                continue

            failure_step_index, detection_feedback = detection
            if failure_step_index >= len(episode.steps):
                failure_step_index = len(episode.steps) - 1
            detection_hit_count += 1

            teacher_prompt_messages = self._build_per_step_teacher_prompt(
                episode, failure_step_index, detection_feedback
            )
            teacher_prompt_text = self._apply_chat_template(teacher_prompt_messages, add_generation_prompt=True)

            target_step = episode.steps[failure_step_index]
            student_prompt_ids = self._coerce_rollout_token_ids(target_step.agent_metadata.get("prompt_token_ids"))
            if student_prompt_ids is None:
                student_prompt_messages = list(_stringify_messages(episode.initial_state))
                for step in episode.steps[:failure_step_index]:
                    if step.agent_metadata.get("context_exceeded") is True:
                        continue
                    student_prompt_messages.append(_stringify_message(step.action))
                    student_prompt_messages.extend(_stringify_messages(step.state))
                student_prompt_text = self._apply_chat_template(student_prompt_messages, add_generation_prompt=True)
                student_prompt_ids = self._encode_text(student_prompt_text)

            step_completion_ids = self._coerce_rollout_token_ids(target_step.agent_metadata.get("completion_token_ids"))
            if step_completion_ids is None:
                if failure_step_index >= len(tokenized.step_token_ranges):
                    detection_miss_count += 1
                    student_prompt_ids_list.append(student_prompt_ids)
                    teacher_prompt_ids_list.append(student_prompt_ids)
                    completion_ids_list.append(tokenized.completion_ids)
                    completion_attention_mask_list.append(tokenized.completion_attention_mask)
                    completion_loss_mask_list.append([0] * len(tokenized.completion_ids))
                    completion_texts.append(tokenized.completion_text)
                    distillation_mask.append(False)
                    detection_hit_count -= 1
                    continue
                start, end = tokenized.step_token_ranges[failure_step_index]
                step_completion_ids = tokenized.completion_ids[start:end]

            student_prompt_ids_list.append(student_prompt_ids)
            teacher_prompt_ids_list.append(self._encode_text(teacher_prompt_text))
            completion_ids_list.append(step_completion_ids)
            completion_attention_mask_list.append([1] * len(step_completion_ids))
            completion_loss_mask_list.append([1] * len(step_completion_ids))
            completion_texts.append("")
            distillation_mask.append(True)

        num_episodes = max(len(rollout_batch.episodes), 1)
        per_step_metrics = {
            "appworld/per_step_detection_hit_fraction": detection_hit_count / num_episodes,
            "appworld/per_step_detection_miss_fraction": detection_miss_count / num_episodes,
            "appworld/per_step_skipped_success_fraction": (
                sum(float(ep.success) for ep in rollout_batch.episodes) / num_episodes
            ),
        }
        merged_rollout_metrics = {
            **rollout_batch.metrics,
            **per_step_metrics,
            **(rollout_metrics or {}),
            "rollout/mean_score": sum(rollout_batch.scores) / max(len(rollout_batch.scores), 1),
        }
        self._latest_rollout_metrics = merged_rollout_metrics

        active_indices = [i for i, active in enumerate(distillation_mask) if active]
        active_fraction = sum(distillation_mask) / max(len(distillation_mask), 1)
        if active_indices:
            kept_indices = active_indices
        else:
            kept_indices = list(range(len(distillation_mask)))

        kept_student_prompt_ids = [student_prompt_ids_list[i] for i in kept_indices]
        kept_teacher_prompt_ids = [teacher_prompt_ids_list[i] for i in kept_indices]
        kept_completion_ids = [completion_ids_list[i] for i in kept_indices]
        kept_completion_attention_mask = [completion_attention_mask_list[i] for i in kept_indices]
        kept_completion_loss_mask = [completion_loss_mask_list[i] for i in kept_indices]
        kept_distillation_mask = [distillation_mask[i] for i in kept_indices]
        kept_completion_texts = [completion_texts[i] for i in kept_indices]

        prompt_ids, prompt_mask = self._pad_sequences(kept_student_prompt_ids, padding_side="left")
        teacher_prompt_ids, teacher_prompt_mask = self._pad_sequences(kept_teacher_prompt_ids, padding_side="left")
        completion_ids, _ = self._pad_sequences(kept_completion_ids, padding_side="right")
        completion_attention_mask = self._pad_mask_sequences(kept_completion_attention_mask, padding_side="right")
        completion_loss_mask = self._pad_mask_sequences(kept_completion_loss_mask, padding_side="right")
        self_distillation_mask = torch.tensor(
            kept_distillation_mask, device=self.accelerator.device, dtype=torch.float32
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "completion_ids": completion_ids,
            "completion_attention_mask": completion_attention_mask,
            "completion_loss_mask": completion_loss_mask,
            "self_distillation_mask": self_distillation_mask,
            "completion_texts": kept_completion_texts,
            "self_distillation_active_fraction": float(active_fraction),
        }

    def _compute_avg_eval_metric_tensor(self, generation_batch: dict[str, Any]) -> torch.Tensor:
        backend = self._resolve_rollout_backend(generation_batch["data_source"])
        if backend == "appworld":
            rollout_batch = self._run_appworld_rollout(
                generation_batch,
                rollout_n_override=max(1, int(getattr(self.args, "eval_num_generations", 1))),
                temperature=getattr(self.args, "eval_temperature", 0.6),
            )
            return self._build_appworld_rollout_eval_metric_tensor(generation_batch, rollout_batch)
        if backend == "bfcl":
            rollout_batch_bfcl = self._run_bfcl_rollout(
                generation_batch,
                rollout_n_override=max(1, int(getattr(self.args, "eval_num_generations", 1))),
                temperature=getattr(self.args, "eval_temperature", 0.6),
            )
            return self._build_bfcl_rollout_eval_metric_tensor(generation_batch, rollout_batch_bfcl)

    def _build_appworld_rollout_eval_metric_tensor(
        self, generation_batch: dict[str, Any], rollout_batch: AppWorldRolloutBatch
    ) -> torch.Tensor:
        self._latest_rollout_metrics = dict(rollout_batch.metrics)
        metric_values = {
            "score": float(rollout_batch.metrics.get("appworld/final_reward", 0.0)),
            "acc": float(rollout_batch.metrics.get("appworld/success_rate", 0.0)),
            "incorrect_format": 0.0,
            "timed_out": 0.0,
            "truncated": 0.0,
            "truncated_and_missing_answer": 0.0,
        }
        token_count = sum(float(sum(item.completion_loss_mask)) for item in rollout_batch.tokenized) / max(
            len(rollout_batch.tokenized), 1
        )
        row = [metric_values.get(name, 0.0) for name in PREDICTION_METRIC_NAMES[:-1]] + [token_count]
        return torch.tensor(
            [row for _ in generation_batch["data_source"]], device=self.accelerator.device, dtype=torch.float32
        )

    def _render_bfcl_episode_for_hindsight(self, episode: BfclEpisode, env_feedback: str | None) -> str:
        max_chars = int(getattr(self.args, "bfcl_hindsight_feedback_max_trajectory_chars", 12000) or 12000)
        return render_bfcl_episode_for_detection(episode, env_feedback, max_chars=max_chars)

    def _detect_bfcl_failure_step(
        self, episodes: list[BfclEpisode], env_feedback: list[str | None]
    ) -> list[tuple[int, str] | None]:
        prompts: list[str] = []
        prompt_indices: list[int] = []
        for index, episode in enumerate(episodes):
            if episode.success or not episode.steps:
                continue
            trajectory = self._render_bfcl_episode_for_hindsight(
                episode, env_feedback[index] if index < len(env_feedback) else None
            )
            messages = build_bfcl_step_failure_detection_messages(trajectory)
            prompts.append(self._apply_chat_template(messages, add_generation_prompt=True))
            prompt_indices.append(index)

        results: list[tuple[int, str] | None] = [None] * len(episodes)
        if not prompts:
            return results

        max_new_tokens = int(getattr(self.args, "bfcl_hindsight_feedback_max_tokens", 192) or 192)
        generated = self._sample_teacher_model_texts(prompts, max_new_tokens=max_new_tokens)
        for index, raw_text in zip(prompt_indices, generated, strict=False):
            parsed = parse_step_failure_json(raw_text, len(episodes[index].steps))
            results[index] = parsed
        return results

    def _detect_bfcl_failure_steps_multi(
        self, episodes: list[BfclEpisode], env_feedback: list[str | None], *, max_steps: int = 3
    ) -> list[list[tuple[int, str]] | None]:
        prompts: list[str] = []
        prompt_indices: list[int] = []
        for index, episode in enumerate(episodes):
            if episode.success or not episode.steps:
                continue
            trajectory = self._render_bfcl_episode_for_hindsight(
                episode, env_feedback[index] if index < len(env_feedback) else None
            )
            messages = build_bfcl_multi_step_failure_detection_messages(trajectory, max_steps=max_steps)
            prompts.append(self._apply_chat_template(messages, add_generation_prompt=True))
            prompt_indices.append(index)

        results: list[list[tuple[int, str]] | None] = [None] * len(episodes)
        if not prompts:
            return results

        max_new_tokens = int(getattr(self.args, "bfcl_hindsight_feedback_max_tokens", 192) or 192)
        generated = self._sample_teacher_model_texts(prompts, max_new_tokens=max_new_tokens)
        for index, raw_text in zip(prompt_indices, generated, strict=False):
            parsed = parse_bfcl_multi_step_failure_json(raw_text, len(episodes[index].steps), max_steps=max_steps)
            results[index] = parsed
        return results

    def _build_bfcl_per_step_teacher_prompt(
        self, episode: BfclEpisode, failure_step_index: int, hint: str
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = list(_bfcl_stringify_messages(episode.initial_state))
        for step in episode.steps[:failure_step_index]:
            if step.agent_metadata.get("context_exceeded") is True:
                continue
            messages.append(_bfcl_stringify_message(step.action))
            messages.extend(_bfcl_stringify_messages(step.state))
        messages.append({"role": "user", "content": f"Consider the following feedback to solve this step:\n\n{hint}"})
        return messages

    def _run_bfcl_rollout(
        self, generation_batch: dict[str, Any], *, rollout_n_override: int | None = None, temperature: float | None = None
    ) -> BfclRolloutBatch:
        config = BfclRolloutConfig(
            env_service_url=getattr(self.args, "bfcl_env_service_url", ""),
            max_steps=int(getattr(self.args, "bfcl_max_steps", 20)),
            rollout_n=max(
                1, int(rollout_n_override if rollout_n_override is not None else getattr(self.args, "rollout_n", 1))
            ),
            max_env_workers=int(getattr(self.args, "bfcl_max_env_workers", 8)),
            max_prompt_tokens=max(
                0,
                int(getattr(self.args, "max_model_len", 0) or 0)
                - int(getattr(self.args, "max_completion_length", 0) or 0),
            ),
            success_reward_threshold=float(getattr(self.args, "bfcl_success_reward_threshold", 1.0)),
            save_rollouts=bool(getattr(self.args, "bfcl_save_rollouts", False)),
            rollout_dir=getattr(self.args, "bfcl_rollout_dir", None),
            debug_rollouts=bool(getattr(self.args, "bfcl_debug_rollouts", False)),
            debug_dir=getattr(self.args, "bfcl_debug_dir", None),
        )
        if not config.env_service_url:
            raise ValueError("--bfcl-env-service-url is required for BFCL training rollouts.")
        return run_bfcl_rollout_batch(
            tasks=build_bfcl_tasks(generation_batch),
            config=config,
            generate_texts=lambda prompts: self._sample_model_outputs(prompts, num_generations=1, temperature=temperature),
            apply_chat_template=lambda messages, add_generation_prompt: self._apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt
            ),
            encode_text=self._encode_text,
            decode_ids=self._decode_ids,
            generation_driver=self._rollout_generator(temperature),
        )

    def _prepare_bfcl_inputs(self, generation_batch: dict[str, Any]) -> dict[str, Any]:
        rollout_batch = self._run_bfcl_rollout(generation_batch)

        base_rollout_metrics = {
            **rollout_batch.metrics,
            "rollout/mean_score": sum(rollout_batch.scores) / max(len(rollout_batch.scores), 1),
        }
        granularity = str(getattr(self.args, "bfcl_feedback_granularity", "step") or "step")
        if granularity == "step":
            distillation_inputs = self._prepare_bfcl_per_step_inputs(
                rollout_batch, rollout_metrics=base_rollout_metrics
            )
        elif granularity == "multi_step":
            distillation_inputs = self._prepare_bfcl_multi_step_inputs(
                rollout_batch, rollout_metrics=base_rollout_metrics
            )
        else:
            raise ValueError(f"Unsupported feedback granularity: {granularity}")
        return distillation_inputs

    def _prepare_bfcl_multi_step_inputs(
        self, rollout_batch: BfclRolloutBatch, *, rollout_metrics: dict[str, float] | None = None
    ) -> dict[str, Any]:
        max_steps = int(getattr(self.args, "bfcl_feedback_multi_step_max", 3) or 3)
        max_steps = max(max_steps, 1)

        env_feedback = list(rollout_batch.feedback)
        detections = self._detect_bfcl_failure_steps_multi(rollout_batch.episodes, env_feedback, max_steps=max_steps)

        student_prompt_ids_list: list[list[int]] = []
        teacher_prompt_ids_list: list[list[int]] = []
        completion_ids_list: list[list[int]] = []
        completion_attention_mask_list: list[list[int]] = []
        completion_loss_mask_list: list[list[int]] = []
        completion_texts: list[str] = []
        distillation_mask: list[bool] = []

        rollout_hit_count = 0
        rollout_miss_count = 0
        expanded_step_count = 0
        steps_per_hit_total = 0
        success_count = 0

        def append_passthrough(tokenized) -> None:
            prompt_ids = list(tokenized.prompt_ids)
            student_prompt_ids_list.append(prompt_ids)
            teacher_prompt_ids_list.append(prompt_ids)
            completion_ids_list.append(tokenized.completion_ids)
            completion_attention_mask_list.append(tokenized.completion_attention_mask)
            completion_loss_mask_list.append([0] * len(tokenized.completion_ids))
            completion_texts.append(tokenized.completion_text)
            distillation_mask.append(False)

        for idx, (episode, tokenized) in enumerate(zip(rollout_batch.episodes, rollout_batch.tokenized, strict=False)):
            if episode.success:
                success_count += 1
                append_passthrough(tokenized)
                continue

            detection = detections[idx]
            if detection is None or not detection:
                rollout_miss_count += 1
                append_passthrough(tokenized)
                continue

            covered_any = False
            for step_index, step_feedback in detection:
                if step_index < 0 or step_index >= len(episode.steps):
                    continue

                target_step = episode.steps[step_index]
                student_prompt_ids = self._coerce_rollout_token_ids(target_step.agent_metadata.get("prompt_token_ids"))
                if student_prompt_ids is None:
                    student_prompt_messages = list(_bfcl_stringify_messages(episode.initial_state))
                    for prior in episode.steps[:step_index]:
                        if prior.agent_metadata.get("context_exceeded") is True:
                            continue
                        student_prompt_messages.append(_bfcl_stringify_message(prior.action))
                        student_prompt_messages.extend(_bfcl_stringify_messages(prior.state))
                    student_prompt_text = self._apply_chat_template(student_prompt_messages, add_generation_prompt=True)
                    student_prompt_ids = self._encode_text(student_prompt_text)

                step_completion_ids = self._coerce_rollout_token_ids(
                    target_step.agent_metadata.get("completion_token_ids")
                )
                if step_completion_ids is None:
                    if step_index >= len(tokenized.step_token_ranges):
                        continue
                    start, end = tokenized.step_token_ranges[step_index]
                    step_completion_ids = tokenized.completion_ids[start:end]
                if not step_completion_ids:
                    continue

                teacher_prompt_messages = self._build_bfcl_per_step_teacher_prompt(episode, step_index, step_feedback)
                teacher_prompt_text = self._apply_chat_template(teacher_prompt_messages, add_generation_prompt=True)

                student_prompt_ids_list.append(student_prompt_ids)
                teacher_prompt_ids_list.append(self._encode_text(teacher_prompt_text))
                completion_ids_list.append(step_completion_ids)
                completion_attention_mask_list.append([1] * len(step_completion_ids))
                completion_loss_mask_list.append([1] * len(step_completion_ids))
                completion_texts.append("")
                distillation_mask.append(True)
                expanded_step_count += 1
                covered_any = True

            if covered_any:
                rollout_hit_count += 1
                steps_per_hit_total += sum(1 for step_index, _ in detection if 0 <= step_index < len(episode.steps))
            else:
                rollout_miss_count += 1
                append_passthrough(tokenized)

        num_episodes = max(len(rollout_batch.episodes), 1)
        multi_step_metrics = {
            "bfcl/multi_step_detection_hit_fraction": rollout_hit_count / num_episodes,
            "bfcl/multi_step_detection_miss_fraction": rollout_miss_count / num_episodes,
            "bfcl/multi_step_skipped_success_fraction": success_count / num_episodes,
            "bfcl/multi_step_steps_per_hit_mean": (
                steps_per_hit_total / rollout_hit_count if rollout_hit_count else 0.0
            ),
            "bfcl/multi_step_expanded_entries": float(expanded_step_count),
        }
        merged_rollout_metrics = {
            **rollout_batch.metrics,
            **multi_step_metrics,
            **(rollout_metrics or {}),
            "rollout/mean_score": sum(rollout_batch.scores) / max(len(rollout_batch.scores), 1),
        }
        self._latest_rollout_metrics = merged_rollout_metrics

        active_indices = [i for i, active in enumerate(distillation_mask) if active]
        active_fraction = sum(distillation_mask) / max(len(distillation_mask), 1)
        kept_indices = active_indices if active_indices else list(range(len(distillation_mask)))

        kept_student_prompt_ids = [student_prompt_ids_list[i] for i in kept_indices]
        kept_teacher_prompt_ids = [teacher_prompt_ids_list[i] for i in kept_indices]
        kept_completion_ids = [completion_ids_list[i] for i in kept_indices]
        kept_completion_attention_mask = [completion_attention_mask_list[i] for i in kept_indices]
        kept_completion_loss_mask = [completion_loss_mask_list[i] for i in kept_indices]
        kept_distillation_mask = [distillation_mask[i] for i in kept_indices]
        kept_completion_texts = [completion_texts[i] for i in kept_indices]

        prompt_ids, prompt_mask = self._pad_sequences(kept_student_prompt_ids, padding_side="left")
        teacher_prompt_ids, teacher_prompt_mask = self._pad_sequences(kept_teacher_prompt_ids, padding_side="left")
        completion_ids, _ = self._pad_sequences(kept_completion_ids, padding_side="right")
        completion_attention_mask = self._pad_mask_sequences(kept_completion_attention_mask, padding_side="right")
        completion_loss_mask = self._pad_mask_sequences(kept_completion_loss_mask, padding_side="right")
        self_distillation_mask = torch.tensor(
            kept_distillation_mask, device=self.accelerator.device, dtype=torch.float32
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "completion_ids": completion_ids,
            "completion_attention_mask": completion_attention_mask,
            "completion_loss_mask": completion_loss_mask,
            "self_distillation_mask": self_distillation_mask,
            "completion_texts": kept_completion_texts,
            "self_distillation_active_fraction": float(active_fraction),
        }

    def _prepare_bfcl_per_step_inputs(
        self, rollout_batch: BfclRolloutBatch, *, rollout_metrics: dict[str, float] | None = None
    ) -> dict[str, Any]:
        env_feedback = list(rollout_batch.feedback)
        detections = self._detect_bfcl_failure_step(rollout_batch.episodes, env_feedback)

        student_prompt_ids_list: list[list[int]] = []
        teacher_prompt_ids_list: list[list[int]] = []
        completion_ids_list: list[list[int]] = []
        completion_attention_mask_list: list[list[int]] = []
        completion_loss_mask_list: list[list[int]] = []
        completion_texts: list[str] = []
        distillation_mask: list[bool] = []
        detection_hit_count = 0
        detection_miss_count = 0

        for idx, (episode, tokenized) in enumerate(zip(rollout_batch.episodes, rollout_batch.tokenized, strict=False)):
            if episode.success:
                prompt_ids = list(tokenized.prompt_ids)
                student_prompt_ids_list.append(prompt_ids)
                teacher_prompt_ids_list.append(prompt_ids)
                completion_ids_list.append(tokenized.completion_ids)
                completion_attention_mask_list.append(tokenized.completion_attention_mask)
                completion_loss_mask_list.append([0] * len(tokenized.completion_ids))
                completion_texts.append(tokenized.completion_text)
                distillation_mask.append(False)
                continue

            detection = detections[idx]
            if detection is None:
                detection_miss_count += 1
                prompt_ids = list(tokenized.prompt_ids)
                student_prompt_ids_list.append(prompt_ids)
                teacher_prompt_ids_list.append(prompt_ids)
                completion_ids_list.append(tokenized.completion_ids)
                completion_attention_mask_list.append(tokenized.completion_attention_mask)
                completion_loss_mask_list.append([0] * len(tokenized.completion_ids))
                completion_texts.append(tokenized.completion_text)
                distillation_mask.append(False)
                continue

            failure_step_index, detection_feedback = detection
            if failure_step_index >= len(episode.steps):
                failure_step_index = len(episode.steps) - 1
            detection_hit_count += 1

            teacher_prompt_messages = self._build_bfcl_per_step_teacher_prompt(
                episode, failure_step_index, detection_feedback
            )
            teacher_prompt_text = self._apply_chat_template(teacher_prompt_messages, add_generation_prompt=True)

            target_step = episode.steps[failure_step_index]
            student_prompt_ids = self._coerce_rollout_token_ids(target_step.agent_metadata.get("prompt_token_ids"))
            if student_prompt_ids is None:
                student_prompt_messages = list(_bfcl_stringify_messages(episode.initial_state))
                for step in episode.steps[:failure_step_index]:
                    if step.agent_metadata.get("context_exceeded") is True:
                        continue
                    student_prompt_messages.append(_bfcl_stringify_message(step.action))
                    student_prompt_messages.extend(_bfcl_stringify_messages(step.state))
                student_prompt_text = self._apply_chat_template(student_prompt_messages, add_generation_prompt=True)
                student_prompt_ids = self._encode_text(student_prompt_text)

            step_completion_ids = self._coerce_rollout_token_ids(target_step.agent_metadata.get("completion_token_ids"))
            if step_completion_ids is None:
                if failure_step_index >= len(tokenized.step_token_ranges):
                    detection_miss_count += 1
                    student_prompt_ids_list.append(student_prompt_ids)
                    teacher_prompt_ids_list.append(student_prompt_ids)
                    completion_ids_list.append(tokenized.completion_ids)
                    completion_attention_mask_list.append(tokenized.completion_attention_mask)
                    completion_loss_mask_list.append([0] * len(tokenized.completion_ids))
                    completion_texts.append(tokenized.completion_text)
                    distillation_mask.append(False)
                    detection_hit_count -= 1
                    continue
                start, end = tokenized.step_token_ranges[failure_step_index]
                step_completion_ids = tokenized.completion_ids[start:end]

            student_prompt_ids_list.append(student_prompt_ids)
            teacher_prompt_ids_list.append(self._encode_text(teacher_prompt_text))
            completion_ids_list.append(step_completion_ids)
            completion_attention_mask_list.append([1] * len(step_completion_ids))
            completion_loss_mask_list.append([1] * len(step_completion_ids))
            completion_texts.append("")
            distillation_mask.append(True)

        num_episodes = max(len(rollout_batch.episodes), 1)
        per_step_metrics = {
            "bfcl/per_step_detection_hit_fraction": detection_hit_count / num_episodes,
            "bfcl/per_step_detection_miss_fraction": detection_miss_count / num_episodes,
            "bfcl/per_step_skipped_success_fraction": (
                sum(float(ep.success) for ep in rollout_batch.episodes) / num_episodes
            ),
        }
        merged_rollout_metrics = {
            **rollout_batch.metrics,
            **per_step_metrics,
            **(rollout_metrics or {}),
            "rollout/mean_score": sum(rollout_batch.scores) / max(len(rollout_batch.scores), 1),
        }
        self._latest_rollout_metrics = merged_rollout_metrics

        active_indices = [i for i, active in enumerate(distillation_mask) if active]
        active_fraction = sum(distillation_mask) / max(len(distillation_mask), 1)
        if active_indices:
            kept_indices = active_indices
        else:
            kept_indices = list(range(len(distillation_mask)))

        kept_student_prompt_ids = [student_prompt_ids_list[i] for i in kept_indices]
        kept_teacher_prompt_ids = [teacher_prompt_ids_list[i] for i in kept_indices]
        kept_completion_ids = [completion_ids_list[i] for i in kept_indices]
        kept_completion_attention_mask = [completion_attention_mask_list[i] for i in kept_indices]
        kept_completion_loss_mask = [completion_loss_mask_list[i] for i in kept_indices]
        kept_distillation_mask = [distillation_mask[i] for i in kept_indices]
        kept_completion_texts = [completion_texts[i] for i in kept_indices]

        prompt_ids, prompt_mask = self._pad_sequences(kept_student_prompt_ids, padding_side="left")
        teacher_prompt_ids, teacher_prompt_mask = self._pad_sequences(kept_teacher_prompt_ids, padding_side="left")
        completion_ids, _ = self._pad_sequences(kept_completion_ids, padding_side="right")
        completion_attention_mask = self._pad_mask_sequences(kept_completion_attention_mask, padding_side="right")
        completion_loss_mask = self._pad_mask_sequences(kept_completion_loss_mask, padding_side="right")
        self_distillation_mask = torch.tensor(
            kept_distillation_mask, device=self.accelerator.device, dtype=torch.float32
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "completion_ids": completion_ids,
            "completion_attention_mask": completion_attention_mask,
            "completion_loss_mask": completion_loss_mask,
            "self_distillation_mask": self_distillation_mask,
            "completion_texts": kept_completion_texts,
            "self_distillation_active_fraction": float(active_fraction),
        }

    def _build_bfcl_rollout_eval_metric_tensor(
        self, generation_batch: dict[str, Any], rollout_batch: BfclRolloutBatch
    ) -> torch.Tensor:
        self._latest_rollout_metrics = dict(rollout_batch.metrics)
        metric_values = {
            "score": float(rollout_batch.metrics.get("bfcl/final_reward", 0.0)),
            "acc": float(rollout_batch.metrics.get("bfcl/success_rate", 0.0)),
            "incorrect_format": 0.0,
            "timed_out": 0.0,
            "truncated": 0.0,
            "truncated_and_missing_answer": 0.0,
        }
        token_count = sum(float(sum(item.completion_loss_mask)) for item in rollout_batch.tokenized) / max(
            len(rollout_batch.tokenized), 1
        )
        row = [metric_values.get(name, 0.0) for name in PREDICTION_METRIC_NAMES[:-1]] + [token_count]
        return torch.tensor(
            [row for _ in generation_batch["data_source"]], device=self.accelerator.device, dtype=torch.float32
        )

    def _uses_distillation_chunk_backward(self, inputs: dict[str, Any]) -> bool:
        del inputs
        return self._distillation_microbatch_size() > 0

    def _training_step_distillation_chunk_backward(
        self, model: torch.nn.Module, inputs: dict[str, Any], num_items_in_batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        del num_items_in_batch
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            gradient_accumulation_steps = max(
                1,
                int(
                    getattr(
                        self,
                        "current_gradient_accumulation_steps",
                        getattr(self.args, "gradient_accumulation_steps", 1),
                    )
                    or 1
                ),
            )
            backward_kwargs: dict[str, Any] = {}
            if getattr(getattr(self, "accelerator", None), "distributed_type", None) is not None:
                distributed_type = self.accelerator.distributed_type
                if getattr(distributed_type, "name", str(distributed_type)) == "DEEPSPEED":
                    backward_kwargs["scale_wrt_gas"] = False

            def backward_chunk(chunk_loss: torch.Tensor) -> None:
                self.accelerator.backward(chunk_loss / gradient_accumulation_steps, **backward_kwargs)

            with self.compute_loss_context_manager():
                distillation_loss, distillation_metrics, distillation_active = self._compute_distillation_loss(
                    model, inputs, backward_fn=backward_chunk
                )

            metrics = dict(self._latest_rollout_metrics)
            metrics.update(distillation_metrics)

            if not distillation_active or distillation_loss is None:
                metrics["objective/inactive"] = 1.0
                self._latest_rollout_metrics = metrics
                loss = self._zero_loss(model) / gradient_accumulation_steps
                self.accelerator.backward(loss, **backward_kwargs)
                return loss.detach()

            metrics["objective/inactive"] = 0.0
            self._latest_rollout_metrics = metrics
            return (distillation_loss / gradient_accumulation_steps).detach()

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self._uses_distillation_chunk_backward(inputs):
            return self._training_step_distillation_chunk_backward(model, inputs, num_items_in_batch=num_items_in_batch)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        metrics = dict(self._latest_rollout_metrics)

        loss, distillation_metrics, active = self._compute_distillation_loss(model, inputs)
        metrics.update(distillation_metrics)
        active = active and loss is not None
        metrics["objective/inactive"] = float(not active)
        if not active:
            loss = self._zero_loss(model)
        self._latest_rollout_metrics = metrics
        if return_outputs:
            return loss, {"metrics": self._latest_rollout_metrics}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        del ignore_keys
        del model

        with torch.no_grad():
            metric_tensor = self._compute_avg_eval_metric_tensor(inputs)

        if prediction_loss_only:
            return (None, None, None)
        return (None, metric_tensor, metric_tensor)
