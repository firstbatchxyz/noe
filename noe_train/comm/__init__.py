"""Latent communication between experts.

Experts communicate via two channels:
1. Text messages (typed JSON) — human-readable, used in v1 routing
2. Latent vectors — compact hidden-state projections, learned in Stage C RL

The latent channel is gated: starts closed (gate ≈ 0), RL learns to open it
when latent signals improve episode return. This means the system gracefully
degrades to text-only communication if latent doesn't help.

For decentralized inference, latent vectors are 256-dim bf16 = 512 bytes,
far cheaper to transmit between nodes than full text generation.
"""

from noe_train.comm.latent import (
    LatentChannel,
    LatentConfig,
    LatentProjector,
    LatentReceiver,
)

__all__ = [
    "LatentChannel",
    "LatentConfig",
    "LatentProjector",
    "LatentReceiver",
]
