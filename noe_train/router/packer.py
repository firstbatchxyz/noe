"""Greedy knapsack packer: select chunks that fit within token budget."""

from __future__ import annotations

import torch

from noe_train.experts.base import Chunk
from noe_train.schema.budget import BudgetClass


def greedy_knapsack(
    chunks: list[Chunk],
    scores: torch.Tensor,
    budget: BudgetClass,
    reserved_tokens: int = 512,
) -> list[Chunk]:
    """Select chunks by score/size ratio within budget.

    Args:
        chunks: available chunks
        scores: router chunk scores (sigmoid output), same length as chunks
        budget: budget class with max_input_tokens
        reserved_tokens: tokens reserved for system prompt + generation overhead
    """
    capacity = budget.max_input_tokens - reserved_tokens
    if capacity <= 0 or not chunks:
        return []

    # Compute score/size ratio
    items = []
    for i, chunk in enumerate(chunks):
        if chunk.token_count <= 0:
            continue
        score = scores[i].item() if i < len(scores) else 0.0
        ratio = score / chunk.token_count
        items.append((ratio, score, chunk))

    # Sort by ratio descending
    items.sort(key=lambda x: x[0], reverse=True)

    selected = []
    total_tokens = 0
    for _ratio, _score, chunk in items:
        if total_tokens + chunk.token_count > capacity:
            continue
        selected.append(chunk)
        total_tokens += chunk.token_count

    return selected
