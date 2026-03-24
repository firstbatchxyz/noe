"""Tester expert: (issue, patch_diff, touched_symbols, harness_summary) → EXEC_REPORT."""

from __future__ import annotations

import json
from typing import Any

from noe_train.schema.artifacts import Artifact, ArtifactType
from noe_train.schema.budget import BudgetLevel
from noe_train.schema.episode import EnvState
from noe_train.schema.messages import (
    ExecReportContent,
    ExpertRole,
    MessageType,
    TypedMessage,
)
from noe_train.experts.base import BaseExpert, Chunk

TESTER_SYSTEM = """You are the Tester expert in a software engineering team.
Given an issue, a patch diff, and repository context, generate test cases.

Output JSON with these fields:
- test_code: the test code to write (Python pytest format)
- expected_behavior: what the tests verify
- edge_cases: list of edge cases considered

Output ONLY valid JSON, no other text."""

DEFAULT_BUDGET = BudgetLevel.B2


class TesterExpert(BaseExpert):
    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(ExpertRole.TESTER, model, tokenizer, device)

    def build_prompt(
        self,
        chunks: list[Chunk],
        messages: list[TypedMessage],
        env: EnvState,
        task_context: dict[str, Any],
    ) -> str:
        parts = [TESTER_SYSTEM, ""]

        issue = task_context.get("issue_text", "")
        parts.append(f"## Issue\n{issue}\n")

        # Patch diff
        patch_msgs = [m for m in messages if m.msg_type == MessageType.PATCH_DONE]
        if patch_msgs:
            parts.append(f"## Patch\n{patch_msgs[-1].summary}\n")
        if env.current_patch_diff:
            parts.append(f"## Diff\n{env.current_patch_diff}\n")

        # Harness summary
        if env.tests_passed or env.tests_failed:
            parts.append(
                f"## Current Test Status\n"
                f"Passed: {env.tests_passed}, Failed: {env.tests_failed}\n"
                f"Failing: {', '.join(env.failing_tests[:5])}\n"
            )

        for chunk in self._truncate_chunks(chunks, 2048):
            parts.append(f"## {chunk.chunk_type}\n{chunk.content}\n")

        parts.append("## Test Cases (JSON)")
        return "\n".join(parts)

    def parse_output(self, raw: str) -> tuple[Artifact, TypedMessage]:
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
            test_code = data.get("test_code", "")
            expected = data.get("expected_behavior", "")
        except (json.JSONDecodeError, KeyError):
            test_code = text[:1000]
            expected = "parse_error"

        artifact = Artifact(
            artifact_type=ArtifactType.TEST_SUITE,
            content=test_code,
            metadata={"expected_behavior": expected},
        )

        # The message carries the exec report — actual execution done by sandbox
        # At generation time, we report a placeholder
        content = ExecReportContent(
            compile_ok=True,  # placeholder — sandbox fills real values
            lint_ok=True,
            tests_passed=0,
            tests_failed=0,
            failing_tests=[],
        )

        summary = f"Test: generated test cases. {expected[:60]}"

        message = TypedMessage(
            msg_type=MessageType.EXEC_REPORT,
            sender=ExpertRole.TESTER,
            round_idx=0,
            content=content.to_dict(),
            summary=summary,
        )

        return artifact, message
