"""Budget classes B0-B4 controlling expert resource allocation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class BudgetLevel(IntEnum):
    B0 = 0
    B1 = 1
    B2 = 2
    B3 = 3
    B4 = 4


@dataclass(frozen=True)
class BudgetClass:
    level: BudgetLevel
    max_input_tokens: int
    max_gen_tokens: int
    max_msg_tokens: int
    max_tools: int

    def allows_tools(self) -> bool:
        return self.max_tools > 0


# Canonical budget table
BUDGET_TABLE: dict[BudgetLevel, BudgetClass] = {
    BudgetLevel.B0: BudgetClass(BudgetLevel.B0, 768, 192, 64, 0),
    BudgetLevel.B1: BudgetClass(BudgetLevel.B1, 1536, 384, 96, 1),
    BudgetLevel.B2: BudgetClass(BudgetLevel.B2, 3072, 768, 128, 2),
    BudgetLevel.B3: BudgetClass(BudgetLevel.B3, 4608, 1024, 192, 3),
    BudgetLevel.B4: BudgetClass(BudgetLevel.B4, 6144, 2048, 256, 3),
}


def get_budget(level: BudgetLevel | int) -> BudgetClass:
    return BUDGET_TABLE[BudgetLevel(level)]


@dataclass()
class BudgetTracker:
    """Tracks token/tool usage against a budget class."""

    budget: BudgetClass
    input_tokens_used: int = 0
    gen_tokens_used: int = 0
    msg_tokens_used: int = 0
    tools_used: int = 0

    def can_generate(self, n_tokens: int = 1) -> bool:
        return self.gen_tokens_used + n_tokens <= self.budget.max_gen_tokens

    def can_use_tool(self) -> bool:
        return self.tools_used < self.budget.max_tools

    @property
    def soft_exceeded(self) -> bool:
        return (
            self.gen_tokens_used >= self.budget.max_gen_tokens
            or self.input_tokens_used >= self.budget.max_input_tokens
        )

    @property
    def hard_exceeded(self) -> bool:
        return (
            self.gen_tokens_used >= 2 * self.budget.max_gen_tokens
            or self.input_tokens_used >= 2 * self.budget.max_input_tokens
        )

    @property
    def pct_used(self) -> float:
        gen_pct = self.gen_tokens_used / self.budget.max_gen_tokens if self.budget.max_gen_tokens else 0
        inp_pct = self.input_tokens_used / self.budget.max_input_tokens if self.budget.max_input_tokens else 0
        return max(gen_pct, inp_pct)
