"""Process Nemotron-Cascade-SFT-SWE into role-aligned format.

Actual schema (discovered Phase 0):
- Columns: category, messages, generator, thinking
- category: "SWE Localization" (92K), "SWE Repair" (87K), "SWE TestGen" (32K)
- messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
- Assistant content contains <think>...</think> blocks (DeepSeek-R1) that we strip.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

DATASET_NAME = "nvidia/Nemotron-Cascade-SFT-SWE"

# Category names as they appear in the dataset
CATEGORY_LOCALIZATION = "SWE Localization"
CATEGORY_REPAIR = "SWE Repair"
CATEGORY_TESTGEN = "SWE TestGen"

# Category → role mapping
CATEGORY_TO_ROLE = {
    CATEGORY_LOCALIZATION: "debugger",
    CATEGORY_REPAIR: "coder",
    CATEGORY_TESTGEN: "tester",
}

# Expected subset sizes (approximate)
EXPECTED_COUNTS = {
    CATEGORY_LOCALIZATION: 92_000,
    CATEGORY_REPAIR: 87_000,
    CATEGORY_TESTGEN: 32_000,
}

# Regex to strip <think>...</think> blocks from assistant output
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass()
class RoleSample:
    role: str
    input_text: str
    output_text: str
    source: str
    metadata: dict[str, Any]


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return _THINK_RE.sub("", text).strip()


def load_nemotron_swe(
    split: str = "train",
    cache_dir: str | None = None,
) -> dict[str, Dataset]:
    """Load and split Nemotron-Cascade-SFT-SWE by category.

    Returns: {category_name: dataset} mapping.
    """
    logger.info(f"Loading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split=split, cache_dir=cache_dir)
    logger.info(f"Total rows: {len(ds)}")
    logger.info(f"Columns: {ds.column_names}")

    if "category" not in ds.column_names:
        raise ValueError(
            f"Expected 'category' column in {DATASET_NAME}, "
            f"got: {ds.column_names}"
        )

    subsets = {}
    for cat in ds.unique("category"):
        subset = ds.filter(lambda x, c=cat: x["category"] == c)
        subsets[cat] = subset
        logger.info(f"  {cat}: {len(subset)} rows")

    return subsets


def _extract_messages(row: dict[str, Any]) -> tuple[str, str]:
    """Extract (user_content, assistant_content) from messages column.

    Messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    messages = row.get("messages", [])
    user_content = ""
    assistant_content = ""

    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
        elif msg["role"] == "assistant":
            assistant_content = msg["content"]

    return user_content, assistant_content


def _extract_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata fields from a dataset row."""
    meta = {}
    if "generator" in row and row["generator"]:
        meta["generator"] = row["generator"]
    if "category" in row:
        meta["category"] = row["category"]
    return meta


def process_category(ds: Dataset, category: str) -> list[dict[str, Any]]:
    """Process a category subset into role-aligned samples.

    Extracts user message as input, assistant message (with think blocks stripped)
    as output. Maps category to role via CATEGORY_TO_ROLE.
    """
    role = CATEGORY_TO_ROLE.get(category)
    if role is None:
        logger.warning(f"Unknown category: {category}, skipping")
        return []

    source_tag = f"nemotron_swe_{category.lower().replace(' ', '_')}"
    samples = []

    for row in ds:
        user_content, assistant_content = _extract_messages(row)

        if not user_content or not assistant_content:
            continue

        # Strip <think>...</think> blocks from assistant output
        output_text = strip_think_blocks(assistant_content)
        if not output_text:
            continue

        samples.append({
            "role": role,
            "input_text": user_content,
            "output_text": output_text,
            "source": source_tag,
            "metadata": _extract_metadata(row),
        })

    return samples


# Keep backward-compatible aliases that process_nemotron.py no longer needs,
# but other code might reference.
def process_localization(ds: Dataset) -> list[dict[str, Any]]:
    """Process localization samples for debugger SFT."""
    return process_category(ds, CATEGORY_LOCALIZATION)


def process_repair(ds: Dataset) -> list[dict[str, Any]]:
    """Process repair samples for coder SFT."""
    return process_category(ds, CATEGORY_REPAIR)


def process_test_generation(ds: Dataset) -> list[dict[str, Any]]:
    """Process test generation samples for tester SFT."""
    return process_category(ds, CATEGORY_TESTGEN)
