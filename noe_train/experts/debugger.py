"""Debugger expert: (issue, plan, patch_diff, failing_tests, trace) → BUG_REPORT."""

from __future__ import annotations

import json
from typing import Any

from noe_train.schema.artifacts import Artifact, ArtifactType
from noe_train.schema.budget import BudgetLevel
from noe_train.schema.episode import EnvState
from noe_train.schema.messages import (
    BugReportContent,
    ExpertRole,
    MessageType,
    TypedMessage,
)
from noe_train.experts.base import BaseExpert, Chunk

DEBUGGER_SYSTEM = """You are the Debugger expert in a software engineering team.
Given an issue, a plan, a patch, failing tests, and execution traces, diagnose the root cause.

Output JSON with these fields:
- root_cause: description of the root cause
- suspect_files: list of file paths most likely containing the bug
- suspect_symbols: list of function/class names involved
- minimal_fix: natural language description of the minimal fix

Output ONLY valid JSON, no other text."""

DEFAULT_BUDGET = BudgetLevel.B2


class DebuggerExpert(BaseExpert):
    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(ExpertRole.DEBUGGER, model, tokenizer, device)

    def build_prompt(
        self,
        chunks: list[Chunk],
        messages: list[TypedMessage],
        env: EnvState,
        task_context: dict[str, Any],
    ) -> str:
        parts = [DEBUGGER_SYSTEM, ""]

        issue = task_context.get("issue_text", "")
        parts.append(f"## Issue\n{issue}\n")

        # Plan
        plan_msgs = [m for m in messages if m.msg_type == MessageType.PLAN]
        if plan_msgs:
            parts.append(f"## Plan\n{json.dumps(plan_msgs[-1].content, indent=2)}\n")

        # Current patch
        if env.current_patch_diff:
            parts.append(f"## Current Patch\n{env.current_patch_diff}\n")

        # Failing tests
        if env.failing_tests:
            parts.append(
                f"## Failing Tests ({env.tests_failed})\n"
                + "\n".join(f"- {t}" for t in env.failing_tests[:10])
                + "\n"
            )

        # Execution traces and other chunks
        for chunk in self._truncate_chunks(chunks, 2048):
            parts.append(f"## {chunk.chunk_type}\n{chunk.content}\n")

        # Prior messages
        if messages:
            parts.append(f"## History\n{self._format_messages(messages, 128)}\n")

        parts.append("## Diagnosis (JSON)")
        return "\n".join(parts)

    def parse_output(self, raw: str) -> tuple[Artifact, TypedMessage]:
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
            content = BugReportContent(
                root_cause=data.get("root_cause", ""),
                suspect_files=data.get("suspect_files", []),
                suspect_symbols=data.get("suspect_symbols", []),
                minimal_fix=data.get("minimal_fix", ""),
            )
        except (json.JSONDecodeError, KeyError):
            content = BugReportContent(
                root_cause=text[:300],
                suspect_files=[],
                suspect_symbols=[],
                minimal_fix="",
            )

        artifact = Artifact(
            artifact_type=ArtifactType.BUG_REPORT,
            content=json.dumps(content.to_dict(), indent=2),
        )

        suspects = ", ".join(content.suspect_files[:3])
        summary = f"Bug: {content.root_cause[:60]}. Suspects: {suspects}"

        message = TypedMessage(
            msg_type=MessageType.BUG_REPORT,
            sender=ExpertRole.DEBUGGER,
            round_idx=0,
            content=content.to_dict(),
            summary=summary,
        )

        return artifact, message
