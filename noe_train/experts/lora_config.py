"""LoRA configuration for expert adapters.

Supports both Qwen2.5 (standard transformer) and Qwen3.5 (DeltaNet hybrid).
"""

from __future__ import annotations

from noe_train.schema.messages import ExpertRole

# Qwen3.5 DeltaNet hybrid targets:
# - DeltaNet layers (24/32): in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
# - Full attention layers (8/32): q_proj, k_proj, v_proj, o_proj
# - MLP (all 32): gate_proj, up_proj, down_proj
# NOT targeted: conv1d (nn.Conv1d), A_log, dt_bias (nn.Parameter — not nn.Linear)
LORA_TARGET_MODULES = [
    # DeltaNet (linear_attn) projections
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    # Full attention (self_attn) projections
    "q_proj", "k_proj", "v_proj", "o_proj",
    # MLP projections (all layers)
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
