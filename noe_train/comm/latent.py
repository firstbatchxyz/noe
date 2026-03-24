"""Latent communication primitives.

Architecture:
  Expert A generates → hidden states → LatentProjector → latent vector (256-dim)
  latent vector → LatentReceiver → virtual token(s) prepended to Expert B's input

The receiver gate starts nearly closed (sigmoid(-5) ≈ 0.007). Stage C RL
learns to open it when latent signals improve reward. If latent doesn't help,
the gate stays closed and the system uses text messages only.

Bandwidth: 256 × 2 bytes (bf16) = 512 bytes per latent message.
Compare to text: ~100 tokens × 4 bytes = 400 bytes, but text requires
full autoregressive decoding while latent is a single matmul.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LatentConfig:
    hidden_dim: int = 2048     # Qwen2.5-Coder-3B hidden size
    latent_dim: int = 256      # bottleneck dimension
    num_virtual_tokens: int = 4  # prepended to receiver input
    gate_init: float = -5.0    # sigmoid(-5) ≈ 0.007, nearly closed
    pooling: str = "mean"      # "mean" or "last"
    dropout: float = 0.0


class LatentProjector(nn.Module):
    """Maps expert hidden states to a compact latent vector.

    Input:  hidden_states (batch, seq_len, hidden_dim)  — last transformer layer
    Output: latent vector (batch, latent_dim)

    Tiny: Linear(2048→256) + LayerNorm ≈ 525K params.
    """

    def __init__(self, config: LatentConfig):
        super().__init__()
        self.config = config
        self.down_proj = nn.Linear(config.hidden_dim, config.latent_dim)
        self.norm = nn.LayerNorm(config.latent_dim)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim) from last transformer layer
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            (batch, latent_dim)
        """
        if self.config.pooling == "last":
            if attention_mask is not None:
                # Get last non-padding position per batch element
                lengths = attention_mask.sum(dim=1).long() - 1  # (batch,)
                pooled = hidden_states[torch.arange(hidden_states.size(0)), lengths]
            else:
                pooled = hidden_states[:, -1, :]
        else:
            # Mean pool over non-padding positions
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden_states.mean(dim=1)

        return self.norm(self.dropout(self.down_proj(pooled)))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LatentReceiver(nn.Module):
    """Injects latent vector into an expert's input as virtual token(s).

    Projects latent (256-dim) → num_virtual_tokens × hidden_dim, then prepends
    to input embeddings. A learned gate controls how much latent signal passes
    through — initialized nearly closed so the system works with text only
    until RL opens the gate.

    For decentralized inference: the virtual tokens are computed with a single
    matmul (no autoregressive decoding), adding negligible latency.
    """

    def __init__(self, config: LatentConfig):
        super().__init__()
        self.config = config
        total_dim = config.hidden_dim * config.num_virtual_tokens

        self.up_proj = nn.Linear(config.latent_dim, total_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.gate = nn.Parameter(torch.tensor(config.gate_init))
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    @property
    def gate_value(self) -> float:
        """Current gate activation (0=closed, 1=open)."""
        return torch.sigmoid(self.gate).item()

    def forward(
        self,
        latent: torch.Tensor,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            latent: (batch, latent_dim)
            input_embeds: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len) or None
        Returns:
            augmented_embeds: (batch, num_virtual + seq_len, hidden_dim)
            augmented_mask:   (batch, num_virtual + seq_len) or None
        """
        batch_size = input_embeds.size(0)
        g = torch.sigmoid(self.gate)

        # Project latent → virtual tokens
        projected = self.up_proj(latent)  # (batch, hidden_dim * num_virtual)
        virtual_tokens = projected.view(
            batch_size, self.config.num_virtual_tokens, self.config.hidden_dim
        )
        virtual_tokens = self.norm(self.dropout(virtual_tokens)) * g

        # Prepend virtual tokens to input embeddings
        augmented = torch.cat([virtual_tokens, input_embeds], dim=1)

        # Extend attention mask for virtual tokens (always attended to)
        augmented_mask = None
        if attention_mask is not None:
            virtual_mask = torch.ones(
                batch_size, self.config.num_virtual_tokens,
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            augmented_mask = torch.cat([virtual_mask, attention_mask], dim=1)

        return augmented, augmented_mask

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LatentChannel(nn.Module):
    """Complete bidirectional latent channel for one expert.

    Each expert owns:
    - A projector: encodes its hidden states → latent vector (sender side)
    - A receiver: decodes incoming latent → virtual tokens (receiver side)

    Usage during episode:
        # Expert A finishes generating
        latent_a = channel_a.project(hidden_states_a, mask_a)

        # Expert B receives A's latent
        augmented_embeds, augmented_mask = channel_b.receive(
            latent_a, input_embeds_b, mask_b
        )
    """

    def __init__(self, config: LatentConfig | None = None):
        super().__init__()
        self.config = config or LatentConfig()
        self.projector = LatentProjector(self.config)
        self.receiver = LatentReceiver(self.config)

    def project(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode hidden states → latent vector. (sender side)"""
        return self.projector(hidden_states, attention_mask)

    def receive(
        self,
        latent: torch.Tensor,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Inject latent into input embeddings. (receiver side)"""
        return self.receiver(latent, input_embeds, attention_mask)

    @property
    def gate_value(self) -> float:
        return self.receiver.gate_value

    def param_count(self) -> int:
        return self.projector.param_count() + self.receiver.param_count()

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.config.latent_dim}, "
            f"virtual_tokens={self.config.num_virtual_tokens}, "
            f"gate={self.gate_value:.4f}, "
            f"params={self.param_count():,}"
        )
