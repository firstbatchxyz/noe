"""Ablation study configurations.

8 ablation experiments:
1. Fixed workflow vs learned router
2. No role split (single LoRA)
3. Free text vs typed messages
4. No tester
5. No debugger
6. No retrieval
7. No critic
8. No cost penalty
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AblationType(str, Enum):
    FIXED_WORKFLOW = "fixed_workflow"
    NO_ROLE_SPLIT = "no_role_split"
    FREE_TEXT_MESSAGES = "free_text_messages"
    NO_TESTER = "no_tester"
    NO_DEBUGGER = "no_debugger"
    NO_RETRIEVAL = "no_retrieval"
    NO_CRITIC = "no_critic"
    NO_COST_PENALTY = "no_cost_penalty"


@dataclass()
class AblationConfig:
    ablation_type: AblationType
    description: str
    overrides: dict[str, Any]


# Canonical ablation configs
ABLATION_CONFIGS: dict[AblationType, AblationConfig] = {
    AblationType.FIXED_WORKFLOW: AblationConfig(
        ablation_type=AblationType.FIXED_WORKFLOW,
        description="Fixed planner→coder→tester workflow, no learned routing",
        overrides={"router.enabled": False, "fixed_workflow": ["planner", "coder", "tester"]},
    ),
    AblationType.NO_ROLE_SPLIT: AblationConfig(
        ablation_type=AblationType.NO_ROLE_SPLIT,
        description="Single LoRA adapter for all roles",
        overrides={"experts.shared_lora": True, "experts.num_adapters": 1},
    ),
    AblationType.FREE_TEXT_MESSAGES: AblationConfig(
        ablation_type=AblationType.FREE_TEXT_MESSAGES,
        description="Free-text messages instead of typed JSON schemas",
        overrides={"schema.typed_messages": False},
    ),
    AblationType.NO_TESTER: AblationConfig(
        ablation_type=AblationType.NO_TESTER,
        description="Remove tester expert from the team",
        overrides={"experts.enabled": ["planner", "coder", "debugger"]},
    ),
    AblationType.NO_DEBUGGER: AblationConfig(
        ablation_type=AblationType.NO_DEBUGGER,
        description="Remove debugger expert from the team",
        overrides={"experts.enabled": ["planner", "coder", "tester"]},
    ),
    AblationType.NO_RETRIEVAL: AblationConfig(
        ablation_type=AblationType.NO_RETRIEVAL,
        description="No BM25/symbol retrieval, only issue + repo_map",
        overrides={"retrieval.enabled": False},
    ),
    AblationType.NO_CRITIC: AblationConfig(
        ablation_type=AblationType.NO_CRITIC,
        description="No critic model (GRPO only, no value baseline)",
        overrides={"critic.enabled": False},
    ),
    AblationType.NO_COST_PENALTY: AblationConfig(
        ablation_type=AblationType.NO_COST_PENALTY,
        description="Remove cost penalties from reward",
        overrides={
            "reward.lambda_call": 0.0,
            "reward.lambda_round": 0.0,
            "reward.lambda_msg": 0.0,
        },
    ),
}


def get_ablation_config(ablation_type: AblationType) -> AblationConfig:
    return ABLATION_CONFIGS[ablation_type]


def list_ablations() -> list[dict[str, str]]:
    return [
        {"type": cfg.ablation_type.value, "description": cfg.description}
        for cfg in ABLATION_CONFIGS.values()
    ]
