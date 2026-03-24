"""Router model: UniXcoder-base + 4 heads for expert selection, stop, budget, chunks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass()
class RouterOutput:
    expert_logits: torch.Tensor   # (4,) sigmoid → multi-label expert selection
    stop_logit: torch.Tensor      # (1,) sigmoid → stop/continue
    budget_logits: torch.Tensor   # (5,) softmax → budget class B0-B4
    chunk_logits: torch.Tensor    # (N,) sigmoid → chunk selection scores
    hidden: torch.Tensor          # (D,) hidden state for critic/logging


class RouterModel(nn.Module):
    """UniXcoder-based router with 4 decision heads."""

    def __init__(
        self,
        encoder_name: str = "microsoft/unixcoder-base",
        max_chunks: int = 64,
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.hidden_dim = hidden_dim
        self.max_chunks = max_chunks

        # Head 1: Expert selection (4 experts)
        self.expert_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4),
        )

        # Head 2: Stop/continue
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Head 3: Budget class (B0-B4)
        self.budget_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 5),
        )

        # Head 4: Chunk scoring
        self.chunk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, max_chunks),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        n_chunks: int | None = None,
    ) -> RouterOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        hidden = outputs.last_hidden_state[:, 0, :]  # (B, D)

        expert_logits = self.expert_head(hidden).squeeze(0)    # (4,)
        stop_logit = self.stop_head(hidden).squeeze(0)         # (1,)
        budget_logits = self.budget_head(hidden).squeeze(0)    # (5,)
        chunk_logits = self.chunk_head(hidden).squeeze(0)      # (max_chunks,)

        # Mask unused chunk slots
        if n_chunks is not None and n_chunks < self.max_chunks:
            chunk_logits[n_chunks:] = float("-inf")

        return RouterOutput(
            expert_logits=expert_logits,
            stop_logit=stop_logit,
            budget_logits=budget_logits,
            chunk_logits=chunk_logits,
            hidden=hidden.squeeze(0),
        )

    def encode_state(self, state_text: str, device: torch.device | None = None) -> dict:
        """Tokenize state text for forward pass."""
        tokens = self.tokenizer(
            state_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        return tokens

    @torch.inference_mode()
    def decide(
        self,
        state_text: str,
        n_chunks: int = 0,
        device: torch.device | None = None,
    ) -> RouterOutput:
        """Full inference: tokenize state → forward → return decisions."""
        tokens = self.encode_state(state_text, device)
        return self.forward(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            n_chunks=n_chunks,
        )
