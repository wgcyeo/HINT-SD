from __future__ import annotations

import re

import torch
import torch.nn.functional as F


def remove_thinking_trace(text: str) -> str:
    result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    result = re.sub(r"^.*?</think>\s*", "", result, flags=re.DOTALL)
    return result


def _append_tail_log_prob(log_probs: torch.Tensor) -> torch.Tensor:
    log_sum = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_sum = torch.clamp(log_sum, max=-1e-7)
    tail_log_prob = torch.log(-torch.expm1(log_sum))
    return torch.cat([log_probs, tail_log_prob], dim=-1)


def _renormalize_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def select_student_topk_log_probs(
    student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor, topk: int | None, add_tail: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if topk is None:
        return student_log_probs, teacher_log_probs
    topk = max(1, min(int(topk), student_log_probs.size(-1)))
    student_topk_log_probs, topk_indices = torch.topk(student_log_probs, k=topk, dim=-1)
    teacher_topk_log_probs = torch.gather(teacher_log_probs, dim=-1, index=topk_indices)
    if add_tail:
        return _append_tail_log_prob(student_topk_log_probs), _append_tail_log_prob(teacher_topk_log_probs)
    return _renormalize_log_probs(student_topk_log_probs), _renormalize_log_probs(teacher_topk_log_probs)


def compute_distillation_loss(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    distillation_topk: int | None,
    distillation_add_tail: bool,
    self_distillation_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute HINT-SD reverse KL on the selected action tokens."""
    student_distill_log_probs, teacher_distill_log_probs = select_student_topk_log_probs(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        topk=distillation_topk,
        add_tail=distillation_add_tail,
    )

    tokenwise_loss = F.kl_div(
        teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True
    )

    per_token_loss = tokenwise_loss.sum(dim=-1)
    loss_mask = completion_mask.to(dtype=per_token_loss.dtype)
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.to(dtype=loss_mask.dtype).unsqueeze(1)

    valid_tokens = loss_mask.sum().clamp(min=1.0)
    loss = (per_token_loss * loss_mask).sum() / valid_tokens
    metrics = {
        "self_distillation/active_fraction": (
            float(self_distillation_mask.float().mean().item()) if self_distillation_mask is not None else 1.0
        ),
        "self_distillation/empty_target_batch": (
            float(bool(self_distillation_mask is not None and self_distillation_mask.sum().item() == 0))
        ),
    }
    return loss, metrics
