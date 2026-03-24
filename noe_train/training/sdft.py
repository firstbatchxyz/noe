"""Self-Distillation Fine-Tuning (SDFT).

Shenfeld et al., "Self-Distillation Enables Continual Learning" (2026).
arXiv:2601.19897

Uses the frozen base model as teacher and the LoRA-adapted model as student.
The KL divergence term prevents the LoRA adapter from drifting too far from
the base distribution, preserving general capabilities while learning new
role-specific skills.

    L_total = L_SFT + α * T² * KL(softmax(z_teacher/T) || softmax(z_student/T))

With PEFT/LoRA, the teacher is just the base model with adapters disabled —
no separate model copy needed. Extra cost: one forward pass per batch
(no gradients), ~1.5x total compute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import Trainer

logger = logging.getLogger(__name__)


@dataclass
class SDFTConfig:
    enabled: bool = True
    alpha: float = 0.5         # weight of KL term (0 = pure SFT, 1 = equal weight)
    temperature: float = 2.0   # distillation temperature (higher = softer targets)


class SDFTTrainer(Trainer):
    """HF Trainer with SDFT distillation loss for PEFT models.

    Toggles LoRA adapters off to get teacher (base model) logits,
    then computes KL divergence against student (base + LoRA) logits.
    """

    def __init__(self, *args, sdft_config: SDFTConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdft_config = sdft_config or SDFTConfig()
        if self.sdft_config.enabled:
            logger.info(
                f"SDFT enabled: alpha={self.sdft_config.alpha}, "
                f"temperature={self.sdft_config.temperature}"
            )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute SFT loss + SDFT KL divergence."""
        # Student forward (LoRA enabled — default state)
        outputs = model(**inputs)
        student_logits = outputs.logits

        # Standard causal LM loss (shifted, ignoring -100 labels)
        labels = inputs.get("labels")
        shift_student = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        sft_loss = F.cross_entropy(
            shift_student.view(-1, shift_student.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        if not self.sdft_config.enabled:
            return (sft_loss, outputs) if return_outputs else sft_loss

        # Teacher forward (LoRA disabled — base model only, no gradients)
        model.disable_adapter_layers()
        with torch.no_grad():
            teacher_outputs = model(**inputs)
            teacher_logits = teacher_outputs.logits
        model.enable_adapter_layers()

        # KL divergence: teacher || student (only on non-masked positions)
        T = self.sdft_config.temperature
        shift_teacher = teacher_logits[..., :-1, :].contiguous()

        # Compute KL only where we have real labels (not padding/-100)
        mask = (shift_labels != -100)  # (batch, seq-1)

        # Temperature-scaled softmax — stay in bf16 (model dtype), cast to
        # float32 only for the KL computation to avoid bf16 precision loss
        # in log/exp. The logits themselves remain bf16 in GPU memory.
        with torch.amp.autocast("cuda", enabled=False):
            teacher_f32 = (shift_teacher.float() / T)
            student_f32 = (shift_student.float() / T)
            teacher_probs = F.softmax(teacher_f32, dim=-1)
            student_log_probs = F.log_softmax(student_f32, dim=-1)
            del teacher_f32, student_f32

            kl_per_token = F.kl_div(
                student_log_probs, teacher_probs,
                reduction="none",
            ).sum(dim=-1)  # (batch, seq-1)
            del teacher_probs, student_log_probs

        num_tokens = mask.sum().clamp(min=1)
        kl_loss = (kl_per_token * mask.float()).sum() / num_tokens

        # Scale by T² (standard distillation convention — gradients scale as 1/T²,
        # so multiply by T² to keep gradient magnitude independent of temperature)
        kl_loss = kl_loss * (T ** 2)

        # Combined loss
        alpha = self.sdft_config.alpha
        loss = sft_loss + alpha * kl_loss

        # Log both components
        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(
                f"Step {self.state.global_step}: "
                f"sft_loss={sft_loss.item():.4f}, "
                f"kl_loss={kl_loss.item():.4f}, "
                f"total={loss.item():.4f}, "
                f"alpha={alpha}"
            )

        return (loss, outputs) if return_outputs else loss
