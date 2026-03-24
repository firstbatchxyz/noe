"""Critic model: UniXcoder-base + scalar value head for V(s)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CriticModel(nn.Module):
    """Critic: encodes state packet → scalar value V(s)."""

    def __init__(
        self,
        encoder_name: str = "microsoft/unixcoder-base",
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.hidden_dim = hidden_dim

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        value = self.value_head(hidden)  # (B, 1)
        return value.squeeze(-1)  # (B,)

    def encode_state(self, state_text: str, device: torch.device | None = None) -> dict:
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
    def predict_value(self, state_text: str, device: torch.device | None = None) -> float:
        tokens = self.encode_state(state_text, device)
        value = self.forward(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        return value.item()
