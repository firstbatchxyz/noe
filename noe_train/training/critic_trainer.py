"""Critic pre-training and update: MSE on (state, return) pairs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from noe_train.critic.model import CriticModel

logger = logging.getLogger(__name__)


@dataclass
class CriticTrainerConfig:
    learning_rate: float = 2e-5
    num_epochs: int = 5
    batch_size: int = 32
    max_seq_len: int = 512


class CriticTrainer:
    """Trains critic model on (state_summary, return) pairs using MSE loss."""

    def __init__(
        self,
        critic: CriticModel,
        config: CriticTrainerConfig | None = None,
        device: torch.device | str = "cuda",
    ):
        self.critic = critic
        self.config = config or CriticTrainerConfig()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.optimizer = torch.optim.AdamW(
            critic.parameters(), lr=self.config.learning_rate
        )
        self.critic.to(self.device)

    def train_epoch(
        self,
        state_texts: list[str],
        returns: list[float],
    ) -> dict[str, float]:
        """Train one epoch on (state, return) pairs."""
        self.critic.train()
        total_loss = 0.0
        n_batches = 0

        indices = list(range(len(state_texts)))
        # Shuffle
        import random
        random.shuffle(indices)

        for start in range(0, len(indices), self.config.batch_size):
            batch_idx = indices[start : start + self.config.batch_size]
            batch_texts = [state_texts[i] for i in batch_idx]
            batch_returns = torch.tensor(
                [returns[i] for i in batch_idx],
                dtype=torch.float32,
                device=self.device,
            )

            # Tokenize batch
            tokens = self.critic.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_len,
                padding="max_length",
            ).to(self.device)

            predicted = self.critic(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

            loss = nn.functional.mse_loss(predicted, batch_returns)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return {"mse_loss": avg_loss, "n_batches": n_batches}

    def train(
        self,
        state_texts: list[str],
        returns: list[float],
    ) -> list[dict[str, float]]:
        """Full training loop."""
        history = []
        for epoch in range(self.config.num_epochs):
            metrics = self.train_epoch(state_texts, returns)
            metrics["epoch"] = epoch
            history.append(metrics)
            logger.info(f"Critic epoch {epoch}: MSE={metrics['mse_loss']:.4f}")
        return history
