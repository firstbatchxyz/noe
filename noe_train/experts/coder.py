"""Coder expert: (issue, plan, file_slices, error_summary) → PATCH via PATCH_PLAN→HUNK→DONE."""

from __future__ import annotations

import json
import re
from typing import Any

from noe_train.schema.artifacts import Artifact, ArtifactType
from noe_train.schema.budget import BudgetLevel
from noe_train.schema.episode import EnvState
from noe_train.schema.messages import (
    ExpertRole,
    MessageType,
    PatchDoneContent,
    TypedMessage,
)
from noe_train.experts.base import BaseExpert, Chunk

CODER_SYSTEM = """You are the Coder expert in a software engineering team.
Given an issue, a plan, and relevant code, produce a patch in unified diff format.

Output a unified diff that can be applied with `git apply`. Use standard --- a/ and +++ b/ prefixes.
Include only the necessary changes. Each hunk should have correct line numbers.

If multiple files need changes, include all of them in a single diff output."""

DEFAULT_BUDGET = BudgetLevel.B4


class CoderExpert(BaseExpert):
    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(ExpertRole.CODER, model, tokenizer, device)

    def build_prompt(
        self,
        chunks: list[Chunk],
        messages: list[TypedMessage],
        env: EnvState,
        task_context: dict[str, Any],
    ) -> str:
        parts = [CODER_SYSTEM, ""]

        # Issue text
        issue = task_context.get("issue_text", "")
        parts.append(f"## Issue\n{issue}\n")

        # Plan from planner
        plan_msgs = [m for m in messages if m.msg_type == MessageType.PLAN]
        if plan_msgs:
            plan = plan_msgs[-1]
            parts.append(f"## Plan\n{json.dumps(plan.content, indent=2)}\n")

        # Bug report from debugger (if repair loop)
        bug_msgs = [m for m in messages if m.msg_type == MessageType.BUG_REPORT]
        if bug_msgs:
            bug = bug_msgs[-1]
            parts.append(f"## Bug Report\n{json.dumps(bug.content, indent=2)}\n")

        # Error summary from previous attempt
        if env.tests_failed > 0:
            parts.append(f"## Previous Errors\nFailing tests: {', '.join(env.failing_tests[:5])}\n")

        # Code chunks
        for chunk in self._truncate_chunks(chunks, 4096):
            parts.append(f"## {chunk.chunk_type}\n{chunk.content}\n")

        # Previous patch (if repair)
        if env.current_patch_diff:
            parts.append(f"## Previous Patch (needs fixing)\n{env.current_patch_diff}\n")

        parts.append("## Unified Diff")
        return "\n".join(parts)

    def parse_output(self, raw: str) -> tuple[Artifact, TypedMessage]:
        text = raw.strip()
        # Extract diff content — handle markdown code blocks
        if "```diff" in text:
            text = text.split("```diff")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Extract file paths from diff
        files_patched = []
        for line in text.split("\n"):
            if line.startswith("+++ b/"):
                filepath = line[6:].strip()
                if filepath not in files_patched:
                    files_patched.append(filepath)

        # Count hunks
        total_hunks = text.count("@@")

        artifact = Artifact(
            artifact_type=ArtifactType.PATCH,
            content=text,
            metadata={"files": files_patched, "hunks": total_hunks},
        )

        files_summary = ", ".join(files_patched[:3])
        if len(files_patched) > 3:
            files_summary += f" (+{len(files_patched) - 3})"
        summary = f"Patch: {len(files_patched)} file(s) [{files_summary}], {total_hunks} hunk(s)"

        content = PatchDoneContent(
            files_patched=files_patched,
            total_hunks=total_hunks,
        )

        message = TypedMessage(
            msg_type=MessageType.PATCH_DONE,
            sender=ExpertRole.CODER,
            round_idx=0,
            content=content.to_dict(),
            summary=summary,
            artifact_ref=None,  # set by orchestrator after store
        )

        return artifact, message
