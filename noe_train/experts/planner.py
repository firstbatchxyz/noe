"""Planner expert: (issue, repo_map, retrieved_files) → PLAN message."""

from __future__ import annotations

import json
from typing import Any

from noe_train.schema.artifacts import Artifact, ArtifactType
from noe_train.schema.budget import BudgetLevel
from noe_train.schema.episode import EnvState
from noe_train.schema.messages import (
    ExpertRole,
    MessageType,
    PlanContent,
    TypedMessage,
)
from noe_train.experts.base import BaseExpert, Chunk

PLANNER_SYSTEM = """You are the Planner expert in a software engineering team.
Given an issue description and repository context, produce a structured plan.

Output JSON with these fields:
- files_to_touch: list of file paths that need modification
- invariants: list of things that must remain true after the fix
- risks: list of potential issues with the approach
- strategy: step-by-step description of the fix approach

Output ONLY valid JSON, no other text."""

DEFAULT_BUDGET = BudgetLevel.B2


class PlannerExpert(BaseExpert):
    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__(ExpertRole.PLANNER, model, tokenizer, device)

    def build_prompt(
        self,
        chunks: list[Chunk],
        messages: list[TypedMessage],
        env: EnvState,
        task_context: dict[str, Any],
    ) -> str:
        parts = [PLANNER_SYSTEM, ""]

        # Issue text
        issue = task_context.get("issue_text", "")
        parts.append(f"## Issue\n{issue}\n")

        # Repo map
        repo_map = task_context.get("repo_map", "")
        if repo_map:
            parts.append(f"## Repository Structure\n{repo_map}\n")

        # Retrieved chunks
        for chunk in self._truncate_chunks(chunks, 2048):
            parts.append(f"## {chunk.chunk_type}\n{chunk.content}\n")

        # Prior messages (if any repair loop)
        if messages:
            parts.append(f"## Prior Context\n{self._format_messages(messages, 128)}\n")

        parts.append("## Plan (JSON)")
        return "\n".join(parts)

    def parse_output(self, raw: str) -> tuple[Artifact, TypedMessage]:
        # Try to extract JSON from output
        text = raw.strip()
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
            content = PlanContent(
                files_to_touch=data.get("files_to_touch", []),
                invariants=data.get("invariants", []),
                risks=data.get("risks", []),
                strategy=data.get("strategy", ""),
            )
        except (json.JSONDecodeError, KeyError):
            content = PlanContent(
                files_to_touch=[],
                invariants=[],
                risks=["Failed to parse plan output"],
                strategy=text[:500],
            )

        artifact = Artifact(
            artifact_type=ArtifactType.PLAN,
            content=json.dumps(content.to_dict(), indent=2),
        )

        files_summary = ", ".join(content.files_to_touch[:3])
        if len(content.files_to_touch) > 3:
            files_summary += f" (+{len(content.files_to_touch) - 3})"
        summary = f"Plan: touch {files_summary}. {content.strategy[:80]}"

        message = TypedMessage(
            msg_type=MessageType.PLAN,
            sender=ExpertRole.PLANNER,
            round_idx=0,  # set by orchestrator
            content=content.to_dict(),
            summary=summary,
        )

        return artifact, message
