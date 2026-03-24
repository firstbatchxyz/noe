"""GRPO (Group Relative Policy Optimization) for router and team RL.

Group size 6: same task, different routing trajectories.
Score whole episodes, normalize within group, update policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    group_size: int = 6
    clip_eps: float = 0.2
    kl_coeff: float = 0.02  # KL penalty to reference policy
    max_rounds: int = 4
    normalize_rewards: bool = True


@dataclass()
class Trajectory:
    """A single trajectory in a GRPO group."""

    task_id: str
    actions: list[dict[str, Any]]  # router decisions per step
    log_probs: list[torch.Tensor]  # log probs of actions under current policy
    reward: float  # episode-level reward


class GRPOTrainer:
    """GRPO training loop for router or combined team."""

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        config: GRPOConfig | None = None,
        device: torch.device | str = "cuda",
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.config = config or GRPOConfig()
        self.device = torch.device(device) if isinstance(device, str) else device

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-5)
        else:
            self.optimizer = optimizer

    def compute_group_advantages(
        self,
        trajectories: list[Trajectory],
    ) -> list[float]:
        """Normalize rewards within group → advantages."""
        rewards = [t.reward for t in trajectories]
        if self.config.normalize_rewards and len(rewards) > 1:
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            std_r = max(std_r, 1e-8)
            advantages = [(r - mean_r) / std_r for r in rewards]
        else:
            advantages = rewards
        return advantages

    def update_step(
        self,
        groups: list[list[Trajectory]],
    ) -> dict[str, float]:
        """Run one GRPO update over a batch of groups.

        Each group = group_size trajectories on the same task.
        """
        total_loss = 0.0
        total_pg_loss = 0.0
        total_kl_loss = 0.0
        n_groups = 0

        self.optimizer.zero_grad()

        for trajectories in groups:
            if len(trajectories) < 2:
                continue

            advantages = self.compute_group_advantages(trajectories)

            for traj, advantage in zip(trajectories, advantages):
                if not traj.log_probs:
                    continue

                # Policy gradient loss
                log_prob_sum = torch.stack(traj.log_probs).sum()

                # Clipped surrogate
                # For GRPO, we use the group-normalized advantage directly
                pg_loss = -advantage * log_prob_sum

                # KL penalty (if reference policy available)
                kl_loss = torch.tensor(0.0, device=self.device)
                if self.ref_policy is not None:
                    # KL divergence estimated from log prob differences
                    # Approximation: KL ≈ log_pi - log_pi_ref
                    kl_loss = self.config.kl_coeff * log_prob_sum.detach()

                loss = pg_loss + kl_loss
                loss.backward()

                total_pg_loss += pg_loss.item()
                total_kl_loss += kl_loss.item()
                total_loss += loss.item()

            n_groups += 1

        if n_groups > 0:
            # Normalize gradients
            for p in self.policy.parameters():
                if p.grad is not None:
                    p.grad /= n_groups * self.config.group_size

            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        return {
            "loss": total_loss / max(n_groups, 1),
            "pg_loss": total_pg_loss / max(n_groups, 1),
            "kl_loss": total_kl_loss / max(n_groups, 1),
            "n_groups": n_groups,
        }
