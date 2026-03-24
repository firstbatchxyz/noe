"""Episode state tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from noe_train.schema.messages import ExpertRole, MessageHistory, TypedMessage


class EpisodeStatus(str, Enum):
    RUNNING = "running"
    ACCEPTED = "accepted"
    ROLLED_BACK = "rolled_back"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass()
class EnvState:
    """Observable environment state at a given point in the episode."""

    compile_ok: bool = False
    lint_ok: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    failing_tests: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    patch_applied: bool = False
    current_patch_diff: str = ""


@dataclass()
class EpisodeState:
    episode_id: str
    task_id: str
    repo: str
    instance_id: str
    issue_text: str
    repo_map: str  # compact repo structure

    # Mutable state
    round_idx: int = 0
    status: EpisodeStatus = EpisodeStatus.RUNNING
    history: MessageHistory = field(default_factory=MessageHistory)
    env: EnvState = field(default_factory=EnvState)
    repair_count: int = 0
    max_repairs: int = 2
    max_rounds: int = 5

    # Tracking
    expert_calls: list[ExpertRole] = field(default_factory=list)
    total_input_tokens: int = 0
    total_gen_tokens: int = 0
    total_tool_calls: int = 0

    def is_terminal(self) -> bool:
        return self.status != EpisodeStatus.RUNNING

    def can_repair(self) -> bool:
        return self.repair_count < self.max_repairs

    def advance_round(self) -> int:
        self.round_idx += 1
        return self.round_idx

    def record_call(self, expert: ExpertRole, input_tokens: int, gen_tokens: int) -> None:
        self.expert_calls.append(expert)
        self.total_input_tokens += input_tokens
        self.total_gen_tokens += gen_tokens

    def to_reward_info(self) -> dict[str, Any]:
        """Extract info needed by reward function."""
        return {
            "compile_ok": self.env.compile_ok,
            "lint_ok": self.env.lint_ok,
            "tests_passed": self.env.tests_passed,
            "tests_failed": self.env.tests_failed,
            "files_changed": self.env.files_changed,
            "expert_calls": [e.value for e in self.expert_calls],
            "total_input_tokens": self.total_input_tokens,
            "total_gen_tokens": self.total_gen_tokens,
            "total_tool_calls": self.total_tool_calls,
            "repair_count": self.repair_count,
            "rounds": self.round_idx,
        }
