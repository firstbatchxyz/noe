"""Teacher-generated planner traces using Qwen2.5-Coder-32B-Instruct.

Generates structured plans from (issue, repo_map) pairs.
~5-8K samples to fill schema gaps (Nemotron SWE has localization but not structured plans).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

TEACHER_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

TEACHER_PROMPT_TEMPLATE = """You are a senior software engineer analyzing a bug report.

Given the issue description and repository map, produce a structured plan for fixing the bug.

## Issue
{issue}

## Repository Structure
{repo_map}

## Output Format (JSON)
{{
    "files_to_touch": ["path/to/file1.py", "path/to/file2.py"],
    "invariants": ["existing tests must still pass", "..."],
    "risks": ["might break X if Y", "..."],
    "strategy": "Step-by-step description of the fix approach"
}}

Output ONLY valid JSON:"""


def build_teacher_prompt(issue: str, repo_map: str) -> str:
    """Build teacher prompt for plan generation."""
    return TEACHER_PROMPT_TEMPLATE.format(issue=issue, repo_map=repo_map)


def validate_plan(plan_text: str, valid_files: list[str] | None = None) -> tuple[bool, dict]:
    """Validate a generated plan.

    Checks:
    - Parseable JSON
    - Required fields present
    - files_to_touch are real files (if valid_files provided)
    """
    try:
        data = json.loads(plan_text)
    except json.JSONDecodeError:
        return False, {"error": "invalid_json"}

    required = ["files_to_touch", "invariants", "risks", "strategy"]
    missing = [f for f in required if f not in data]
    if missing:
        return False, {"error": "missing_fields", "missing": missing}

    if not isinstance(data["files_to_touch"], list):
        return False, {"error": "files_to_touch_not_list"}

    if not isinstance(data["strategy"], str) or len(data["strategy"]) < 10:
        return False, {"error": "strategy_too_short"}

    # Validate file paths against actual repo
    if valid_files is not None:
        valid_set = set(valid_files)
        invalid = [f for f in data["files_to_touch"] if f not in valid_set]
        if invalid:
            return False, {"error": "invalid_files", "invalid": invalid}

    return True, data


def format_as_training_sample(
    issue: str,
    repo_map: str,
    plan_json: dict,
    instance_id: str = "",
) -> dict[str, Any]:
    """Format a validated plan into a training sample."""
    input_text = f"## Issue\n{issue}\n\n## Repository Structure\n{repo_map}"
    output_text = json.dumps(plan_json, indent=2)

    return {
        "role": "planner",
        "input_text": input_text,
        "output_text": output_text,
        "source": "teacher_generated",
        "metadata": {
            "teacher_model": TEACHER_MODEL,
            "instance_id": instance_id,
        },
    }
