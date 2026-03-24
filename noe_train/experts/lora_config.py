"""LoRA configuration for expert adapters.

Supports both Qwen2.5 (standard transformer) and Qwen3.5 (DeltaNet hybrid).
"""

from __future__ import annotations

from noe_train.schema.messages import ExpertRole

# Qwen3.5 DeltaNet hybrid — LoRA targets follow Unsloth's recommendation:
# - Full attention layers (8/32): q_proj, k_proj, v_proj, o_proj
# - MLP (all 32 layers): gate_proj, up_proj, down_proj
# DeltaNet projections (in_proj_qkv/z/b/a, out_proj) are NOT targeted —
# DeltaNet layers have carefully tuned recurrence dynamics (A_log, dt_bias)
# that LoRA on input projections can destabilize. MLP LoRA on all 32 layers
# provides sufficient adaptation capacity.
LORA_TARGET_MODULES = [
    # Full attention (self_attn) projections — 8 layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    # MLP projections — all 32 layers
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
