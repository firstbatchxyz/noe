"""Process Nemotron-Cascade-SFT-SWE into role-aligned format.

Primary Stage A corpus:
- ~92K localization samples → debugger + planner SFT
- ~87K repair samples → coder SFT
- ~32K test generation samples → tester SFT
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

DATASET_NAME = "nvidia/Nemotron-Cascade-SFT-SWE"

# Expected subset sizes (approximate)
EXPECTED_COUNTS = {
    "localization": 92_000,
    "repair": 87_000,
    "test_generation": 32_000,
}


@dataclass()
class RoleSample:
    role: str
    input_text: str
    output_text: str
    source: str
    metadata: dict[str, Any]


def load_nemotron_swe(
    split: str = "train",
    cache_dir: str | None = None,
) -> dict[str, Dataset]:
    """Load and split Nemotron-Cascade-SFT-SWE by task type."""
    logger.info(f"Loading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split=split, cache_dir=cache_dir)

    # Identify subsets by task_type column or similar
    subsets = {}
    if "task_type" in ds.column_names:
        for task_type in ds.unique("task_type"):
            subsets[task_type] = ds.filter(lambda x: x["task_type"] == task_type)
    else:
        # If no explicit task_type, use heuristics on content
        subsets["all"] = ds

    logger.info(f"Loaded subsets: {[(k, len(v)) for k, v in subsets.items()]}")
    return subsets


def process_localization(ds: Dataset) -> list[dict[str, Any]]:
    """Process localization samples for debugger SFT.

    Input: issue + repo context
    Output: localized files + diagnosis
    """
    samples = []
    for row in ds:
        input_text = _build_input(row, task="localization")
        output_text = _extract_output(row, task="localization")
        samples.append({
            "role": "debugger",
            "input_text": input_text,
            "output_text": output_text,
            "source": "nemotron_swe_localization",
            "metadata": _extract_metadata(row),
        })
    return samples


def process_repair(ds: Dataset) -> list[dict[str, Any]]:
    """Process repair samples for coder SFT.

    Input: issue + localized files + context
    Output: patch
    """
    samples = []
    for row in ds:
        input_text = _build_input(row, task="repair")
        output_text = _extract_output(row, task="repair")
        samples.append({
            "role": "coder",
            "input_text": input_text,
            "output_text": output_text,
            "source": "nemotron_swe_repair",
            "metadata": _extract_metadata(row),
        })
    return samples


def process_test_generation(ds: Dataset) -> list[dict[str, Any]]:
    """Process test generation samples for tester SFT.

    Input: issue + patch
    Output: test cases + expected results
    """
    samples = []
    for row in ds:
        input_text = _build_input(row, task="test_generation")
        output_text = _extract_output(row, task="test_generation")
        samples.append({
            "role": "tester",
            "input_text": input_text,
            "output_text": output_text,
            "source": "nemotron_swe_test_gen",
            "metadata": _extract_metadata(row),
        })
    return samples


def _build_input(row: dict[str, Any], task: str) -> str:
    """Build input text from dataset row, adapted per task."""
    parts = []

    # Issue / problem statement
    issue = row.get("problem_statement") or row.get("issue") or row.get("input", "")
    if issue:
        parts.append(f"## Issue\n{issue}")

    # Repository context
    repo = row.get("repo") or row.get("repository", "")
    if repo:
        parts.append(f"## Repository\n{repo}")

    # File context (for repair/localization)
    files = row.get("file_context") or row.get("relevant_files", "")
    if files:
        parts.append(f"## Relevant Files\n{files}")

    # For repair: include localization
    if task == "repair":
        loc = row.get("localization") or row.get("localized_files", "")
        if loc:
            parts.append(f"## Localized Files\n{loc}")

    # For test_generation: include patch
    if task == "test_generation":
        patch = row.get("patch") or row.get("gold_patch", "")
        if patch:
            parts.append(f"## Patch\n{patch}")

    return "\n\n".join(parts)


def _extract_output(row: dict[str, Any], task: str) -> str:
    """Extract target output from dataset row."""
    if task == "localization":
        return row.get("output") or row.get("localization") or row.get("response", "")
    elif task == "repair":
        return row.get("output") or row.get("patch") or row.get("gold_patch", "")
    elif task == "test_generation":
        return row.get("output") or row.get("test_code") or row.get("response", "")
    return row.get("output", "")


def _extract_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata fields from a dataset row."""
    meta = {}
    for key in ["instance_id", "repo", "base_commit", "version"]:
        if key in row:
            meta[key] = row[key]
    return meta
