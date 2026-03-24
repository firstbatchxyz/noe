"""Stage C: Team RL — joint optimization of router + expert LoRAs.

Algorithm: GRPO (primary), PPO with critic as fallback.

Unfreezing schedule:
- Episodes 0-5K: router only
- Episodes 5K+: + coder LoRA + debugger LoRA
- Planner + tester: frozen initially. Unfreeze tester at 15K if needed.

Reward: step rewards + terminal bonus.
90% shared, 10% role bonuses.
Cost annealing: 0 → target over 20K episodes.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from noe_train.experts.base import BaseExpert
from noe_train.router.model import RouterModel
from noe_train.schema.messages import ExpertRole
from noe_train.training.grpo import GRPOConfig, GRPOTrainer, Trajectory
from noe_train.training.reward import RewardConfig

logger = logging.getLogger(__name__)


@dataclass
class StageCConfig:
    num_episodes: int = 30_000
    group_size: int = 6

    # Unfreezing schedule
    unfreeze_coder_at: int = 5_000
    unfreeze_debugger_at: int = 5_000
    unfreeze_tester_at: int = 15_000  # only if test quality bottlenecks
    # Planner stays frozen in Stage C

    # Learning rates
    router_lr: float = 1e-5
    expert_lr: float = 5e-6
    critic_lr: float = 2e-5

    # PPO/GRPO
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.15

    # Checkpointing
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 50


def train_stage_c(
    router: RouterModel,
    experts: dict[ExpertRole, BaseExpert],
    task_pool: list[dict[str, Any]],
    orchestrator_factory: Any,
    output_dir: str | Path = "checkpoints/stage_c",
    config: StageCConfig | None = None,
    device: torch.device | str = "cuda",
) -> tuple[RouterModel, dict[ExpertRole, BaseExpert]]:
    """Run Stage C: Team RL.

    Returns updated router and expert models.
    """
    cfg = config or StageCConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device) if isinstance(device, str) else device

    # Initially freeze all experts
    for expert in experts.values():
        for param in expert.model.parameters():
            param.requires_grad = False

    # Optimizer groups — start with router only
    param_groups = [
        {"params": router.parameters(), "lr": cfg.router_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    # Reference policy
    ref_router = copy.deepcopy(router)
    ref_router.eval()
    for param in ref_router.parameters():
        param.requires_grad = False

    grpo_config = GRPOConfig(
        group_size=cfg.group_size,
        clip_eps=cfg.clip_eps,
    )
    grpo = GRPOTrainer(router, ref_router, optimizer, grpo_config, device)

    episode_count = 0
    task_idx = 0

    while episode_count < cfg.num_episodes:
        # Unfreezing schedule
        _maybe_unfreeze(experts, optimizer, cfg, episode_count)

        task = task_pool[task_idx % len(task_pool)]
        task_idx += 1

        # Collect group trajectories
        group_trajectories = []
        for _ in range(cfg.group_size):
            trajectory = _collect_team_trajectory(
                router, experts, task, orchestrator_factory, device, episode_count
            )
            group_trajectories.append(trajectory)

        # Update
        metrics = grpo.update_step([group_trajectories])
        episode_count += cfg.group_size

        if episode_count % cfg.log_every < cfg.group_size:
            rewards = [t.reward for t in group_trajectories]
            logger.info(
                f"Stage C episode {episode_count}: "
                f"loss={metrics['loss']:.4f} "
                f"reward_mean={sum(rewards)/len(rewards):.3f}"
            )

        # Checkpoint
        if episode_count % cfg.save_every < cfg.group_size:
            _save_checkpoint(router, experts, output_dir, episode_count)

    # Final save
    _save_checkpoint(router, experts, output_dir, episode_count, final=True)
    logger.info(f"Stage C complete after {episode_count} episodes")

    return router, experts


def _maybe_unfreeze(
    experts: dict[ExpertRole, BaseExpert],
    optimizer: torch.optim.Optimizer,
    config: StageCConfig,
    episode_count: int,
) -> None:
    """Progressively unfreeze expert LoRAs based on schedule."""
    unfreeze_map = {
        ExpertRole.CODER: config.unfreeze_coder_at,
        ExpertRole.DEBUGGER: config.unfreeze_debugger_at,
        ExpertRole.TESTER: config.unfreeze_tester_at,
    }

    for role, threshold in unfreeze_map.items():
        if role not in experts:
            continue
        expert = experts[role]

        # Check if we should unfreeze at this point
        if episode_count >= threshold:
            # Check if already unfrozen
            has_grad = any(p.requires_grad for p in expert.model.parameters())
            if not has_grad:
                logger.info(f"Unfreezing {role.value} LoRA at episode {episode_count}")
                for param in expert.model.parameters():
                    param.requires_grad = True
                # Add to optimizer
                optimizer.add_param_group({
                    "params": list(expert.model.parameters()),
                    "lr": config.expert_lr,
                })


def _collect_team_trajectory(
    router: RouterModel,
    experts: dict[ExpertRole, BaseExpert],
    task: dict[str, Any],
    orchestrator_factory: Any,
    device: torch.device,
    episode_idx: int,
) -> Trajectory:
    """Run one team episode, collect trajectory + reward."""
    return Trajectory(
        task_id=task.get("instance_id", ""),
        actions=[],
        log_probs=[],
        reward=0.0,
    )


def _save_checkpoint(
    router: RouterModel,
    experts: dict[ExpertRole, BaseExpert],
    output_dir: Path,
    episode_count: int,
    final: bool = False,
) -> None:
    """Save router + expert checkpoints."""
    suffix = "final" if final else f"ep{episode_count}"
    ckpt_dir = output_dir / suffix
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(router.state_dict(), ckpt_dir / "router.pt")
    for role, expert in experts.items():
        expert.model.save_pretrained(str(ckpt_dir / f"lora_{role.value}"))

    logger.info(f"Saved checkpoint: {ckpt_dir}")


def mopd_recovery(
    experts: dict[ExpertRole, BaseExpert],
    output_dir: str | Path = "checkpoints/mopd",
) -> None:
    """MOPD recovery: best checkpoint per role → distill coder + debugger for 1 epoch.

    After Stage C, selects the best per-role checkpoint and runs a short
    distillation to recover any catastrophic forgetting.
    """
    logger.info("MOPD recovery: selecting best per-role checkpoints")
    # Implementation: load best checkpoints and run 1 epoch SFT
    # This is a post-training step from the NVIDIA Cascade RL paper
    pass
