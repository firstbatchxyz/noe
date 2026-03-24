"""Stage B: Router RL via GRPO.

- Group size 6: same task, different routing trajectories
- Experts frozen (Stage A). Router only updated.
- lr=2e-5, clip=0.2, KL to supervised router=0.02, max rounds=4
- ~80-100K episodes, ~2 weeks
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from noe_train.experts.base import BaseExpert
from noe_train.orchestrator.loop import Orchestrator
from noe_train.router.model import RouterModel
from noe_train.training.grpo import GRPOConfig, GRPOTrainer, Trajectory
from noe_train.training.reward import RewardConfig, compute_phi, compute_terminal_reward

logger = logging.getLogger(__name__)


@dataclass
class StageBConfig:
    num_episodes: int = 100_000
    group_size: int = 6
    lr: float = 2e-5
    clip_eps: float = 0.2
    kl_coeff: float = 0.02
    max_rounds: int = 4
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 50


def train_stage_b(
    router: RouterModel,
    experts: dict[str, BaseExpert],
    task_pool: list[dict[str, Any]],
    orchestrator_factory: Any,  # callable that creates Orchestrator
    output_dir: str | Path = "checkpoints/stage_b",
    config: StageBConfig | None = None,
    device: torch.device | str = "cuda",
) -> RouterModel:
    """Run Stage B: Router GRPO training.

    Experts are frozen. Only router is updated.
    """
    cfg = config or StageBConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device) if isinstance(device, str) else device

    # Freeze all experts
    for expert in experts.values():
        for param in expert.model.parameters():
            param.requires_grad = False

    # Reference policy (copy of initial router)
    ref_router = copy.deepcopy(router)
    ref_router.eval()
    for param in ref_router.parameters():
        param.requires_grad = False

    # GRPO trainer
    grpo_config = GRPOConfig(
        group_size=cfg.group_size,
        clip_eps=cfg.clip_eps,
        kl_coeff=cfg.kl_coeff,
        max_rounds=cfg.max_rounds,
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=cfg.lr)
    grpo = GRPOTrainer(router, ref_router, optimizer, grpo_config, device)

    # Training loop
    episode_count = 0
    task_idx = 0

    while episode_count < cfg.num_episodes:
        # Sample a task
        task = task_pool[task_idx % len(task_pool)]
        task_idx += 1

        # Run group of trajectories
        group_trajectories = []
        for _ in range(cfg.group_size):
            # Create orchestrator and run episode
            # The orchestrator_factory handles episode creation, sandbox, etc.
            trajectory = _collect_trajectory(
                router, experts, task, orchestrator_factory, device
            )
            group_trajectories.append(trajectory)

        # GRPO update
        metrics = grpo.update_step([group_trajectories])
        episode_count += cfg.group_size

        if episode_count % cfg.log_every < cfg.group_size:
            rewards = [t.reward for t in group_trajectories]
            logger.info(
                f"Stage B episode {episode_count}: "
                f"loss={metrics['loss']:.4f} "
                f"reward_mean={sum(rewards)/len(rewards):.3f} "
                f"reward_max={max(rewards):.3f}"
            )

        # Save checkpoint
        if episode_count % cfg.save_every < cfg.group_size:
            ckpt_path = output_dir / f"router_ep{episode_count}"
            torch.save(router.state_dict(), ckpt_path / "model.pt")
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final
    final_path = output_dir / "router_final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(router.state_dict(), final_path / "model.pt")
    logger.info(f"Stage B complete. Final model: {final_path}")

    return router


def _collect_trajectory(
    router: RouterModel,
    experts: dict[str, BaseExpert],
    task: dict[str, Any],
    orchestrator_factory: Any,
    device: torch.device,
) -> Trajectory:
    """Run one episode, collect routing trajectory + reward."""
    # This is a placeholder — actual implementation requires running
    # the full orchestrator loop and collecting log probs from router
    return Trajectory(
        task_id=task.get("instance_id", ""),
        actions=[],
        log_probs=[],
        reward=0.0,
    )
