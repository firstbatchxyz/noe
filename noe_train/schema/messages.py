"""Typed message definitions for inter-expert communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    PLAN = "PLAN"
    PATCH_PLAN = "PATCH_PLAN"
    PATCH_HUNK = "PATCH_HUNK"
    PATCH_DONE = "PATCH_DONE"
    EXEC_REPORT = "EXEC_REPORT"
    BUG_REPORT = "BUG_REPORT"
    ROUTE_HINT = "ROUTE_HINT"
    FINAL = "FINAL"


class ExpertRole(str, Enum):
    PLANNER = "planner"
    CODER = "coder"
    TESTER = "tester"
    DEBUGGER = "debugger"
    ROUTER = "router"


class FinalVerdict(str, Enum):
    ACCEPT = "accept"
    ROLLBACK = "rollback"


@dataclass()
class TypedMessage:
    msg_type: MessageType
    sender: ExpertRole
    round_idx: int
    content: dict[str, Any]
    summary: str  # compact summary for state compiler (64-96 tokens)
    token_count: int = 0  # token count of full content
    artifact_ref: str | None = None  # SHA-256 ref in artifact store
    latent_vector: Any = None  # torch.Tensor (latent_dim,) or None — latent comm channel


@dataclass()
class PlanContent:
    files_to_touch: list[str]
    invariants: list[str]
    risks: list[str]
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_to_touch": self.files_to_touch,
            "invariants": self.invariants,
            "risks": self.risks,
            "strategy": self.strategy,
        }


@dataclass()
class PatchPlanContent:
    files: list[str]
    description: str
    estimated_hunks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "description": self.description,
            "estimated_hunks": self.estimated_hunks,
        }


@dataclass()
class PatchHunkContent:
    file_path: str
    hunk_idx: int
    diff: str  # unified diff format

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "hunk_idx": self.hunk_idx,
            "diff": self.diff,
        }


@dataclass()
class PatchDoneContent:
    files_patched: list[str]
    total_hunks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_patched": self.files_patched,
            "total_hunks": self.total_hunks,
        }


@dataclass()
class ExecReportContent:
    compile_ok: bool
    lint_ok: bool
    tests_passed: int
    tests_failed: int
    failing_tests: list[str]
    trace_ref: str | None = None  # artifact store reference

    def to_dict(self) -> dict[str, Any]:
        return {
            "compile_ok": self.compile_ok,
            "lint_ok": self.lint_ok,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "failing_tests": self.failing_tests,
            "trace_ref": self.trace_ref,
        }


@dataclass()
class BugReportContent:
    root_cause: str
    suspect_files: list[str]
    suspect_symbols: list[str]
    minimal_fix: str  # natural language description

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "suspect_files": self.suspect_files,
            "suspect_symbols": self.suspect_symbols,
            "minimal_fix": self.minimal_fix,
        }


@dataclass()
class RouteHintContent:
    next_expert: ExpertRole
    budget_level: int
    chunk_ids: list[str]
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_expert": self.next_expert.value,
            "budget_level": self.budget_level,
            "chunk_ids": self.chunk_ids,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass()
class FinalContent:
    verdict: FinalVerdict
    patch_artifact_ref: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "patch_artifact_ref": self.patch_artifact_ref,
            "reason": self.reason,
        }


# Message history for an episode
@dataclass()
class MessageHistory:
    messages: list[TypedMessage] = field(default_factory=list)

    def add(self, msg: TypedMessage) -> None:
        self.messages.append(msg)

    def by_type(self, msg_type: MessageType) -> list[TypedMessage]:
        return [m for m in self.messages if m.msg_type == msg_type]

    def by_sender(self, sender: ExpertRole) -> list[TypedMessage]:
        return [m for m in self.messages if m.sender == sender]

    def by_round(self, round_idx: int) -> list[TypedMessage]:
        return [m for m in self.messages if m.round_idx == round_idx]

    def last(self) -> TypedMessage | None:
        return self.messages[-1] if self.messages else None

    def latent_vectors(self, sender: ExpertRole | None = None) -> list[Any]:
        """Get all latent vectors, optionally filtered by sender."""
        vectors = []
        for m in self.messages:
            if m.latent_vector is not None:
                if sender is None or m.sender == sender:
                    vectors.append(m.latent_vector)
        return vectors

    def latest_latent(self, sender: ExpertRole | None = None) -> Any:
        """Get the most recent latent vector."""
        for m in reversed(self.messages):
            if m.latent_vector is not None:
                if sender is None or m.sender == sender:
                    return m.latent_vector
        return None

    def summaries(self, max_tokens: int = 512) -> str:
        parts = []
        total = 0
        for msg in self.messages:
            if total + len(msg.summary.split()) > max_tokens:
                break
            parts.append(f"[R{msg.round_idx}] {msg.sender.value}: {msg.summary}")
            total += len(msg.summary.split())
        return "\n".join(parts)
