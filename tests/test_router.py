"""Tests for router model and packer."""

import torch

from noe_train.experts.base import Chunk
from noe_train.router.packer import greedy_knapsack
from noe_train.schema.budget import get_budget, BudgetLevel


def test_greedy_knapsack_basic():
    chunks = [
        Chunk(id="c0", chunk_type="issue", content="x", token_count=100),
        Chunk(id="c1", chunk_type="code", content="y", token_count=200),
        Chunk(id="c2", chunk_type="test", content="z", token_count=50),
    ]
    scores = torch.tensor([0.9, 0.5, 0.8])
    budget = get_budget(BudgetLevel.B2)

    selected = greedy_knapsack(chunks, scores, budget)
    assert len(selected) > 0
    # All selected chunks should fit in budget
    total = sum(c.token_count for c in selected)
    assert total <= budget.max_input_tokens


def test_greedy_knapsack_empty():
    selected = greedy_knapsack([], torch.tensor([]), get_budget(BudgetLevel.B0))
    assert selected == []


def test_greedy_knapsack_respects_budget():
    # Create chunks that won't all fit in B0 (768 input tokens - 512 reserved = 256)
    chunks = [
        Chunk(id="c0", chunk_type="a", content="x" * 100, token_count=100),
        Chunk(id="c1", chunk_type="b", content="y" * 100, token_count=100),
        Chunk(id="c2", chunk_type="c", content="z" * 100, token_count=100),
    ]
    scores = torch.tensor([0.9, 0.8, 0.7])
    budget = get_budget(BudgetLevel.B0)

    selected = greedy_knapsack(chunks, scores, budget)
    total = sum(c.token_count for c in selected)
    assert total <= budget.max_input_tokens - 512  # reserved
