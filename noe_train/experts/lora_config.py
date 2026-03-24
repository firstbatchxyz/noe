"""LoRA configuration for expert adapters on Qwen2.5-Coder-3B-Instruct."""

from __future__ import annotations

from noe_train.schema.messages import ExpertRole

# Standard transformer targets for Qwen2.5
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Rank per role: planner/debugger get r=32, coder/tester start at r=16
ROLE_RANKS: dict[ExpertRole, int] = {
    ExpertRole.PLANNER: 32,
    ExpertRole.CODER: 16,
    ExpertRole.TESTER: 16,
    ExpertRole.DEBUGGER: 32,
}


def get_lora_config(role: ExpertRole, rank_override: int | None = None):
    from peft import LoraConfig, TaskType

    r = rank_override if rank_override is not None else ROLE_RANKS[role]
    return LoraConfig(
        r=r,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
