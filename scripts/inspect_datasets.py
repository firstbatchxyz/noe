#!/usr/bin/env python3
"""Inspect Nemotron datasets: verify counts, schemas, and sample data.

- Load nvidia/Nemotron-Cascade-SFT-SWE — verify category counts
- Load nvidia/Nemotron-Cascade-RL-SWE — verify ~110K rows
- Print column schemas, sample rows, message structure
"""

import logging
import sys

from datasets import load_dataset

from noe_train.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def inspect_sft_swe():
    """Inspect Nemotron-Cascade-SFT-SWE."""
    logger.info("=" * 60)
    logger.info("Inspecting nvidia/Nemotron-Cascade-SFT-SWE")
    logger.info("=" * 60)

    try:
        ds = load_dataset("nvidia/Nemotron-Cascade-SFT-SWE", split="train")
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        return False

    logger.info(f"Total rows: {len(ds)}")
    logger.info(f"Columns: {ds.column_names}")
    logger.info(f"Features: {ds.features}")

    # Check category column
    if "category" in ds.column_names:
        logger.info("\nCategory breakdown:")
        for cat in sorted(ds.unique("category")):
            subset = ds.filter(lambda x, c=cat: x["category"] == c)
            logger.info(f"  {cat}: {len(subset)} rows")
    else:
        logger.warning("No 'category' column found!")

    # Inspect message structure
    if "messages" in ds.column_names:
        logger.info("\nMessage structure (first row):")
        messages = ds[0]["messages"]
        for i, msg in enumerate(messages):
            logger.info(f"  messages[{i}]: role={msg.get('role')}, "
                       f"content_len={len(msg.get('content', ''))}")
            content_preview = msg.get("content", "")[:200]
            logger.info(f"    content preview: {content_preview}...")

        # Check for <think> blocks
        has_think = False
        for msg in messages:
            if "<think>" in msg.get("content", ""):
                has_think = True
                break
        logger.info(f"\n  Contains <think> blocks: {has_think}")

    # Print 3 sample rows (abbreviated)
    logger.info("\nSample rows:")
    for i in range(min(3, len(ds))):
        row = ds[i]
        logger.info(f"\n--- Row {i} ---")
        logger.info(f"  category: {row.get('category', 'N/A')}")
        logger.info(f"  generator: {row.get('generator', 'N/A')}")
        if "messages" in row:
            for j, msg in enumerate(row["messages"]):
                content = msg.get("content", "")
                logger.info(f"  messages[{j}].role: {msg.get('role')}")
                logger.info(f"  messages[{j}].content_len: {len(content)}")
                logger.info(f"  messages[{j}].content[:150]: {content[:150]}")
        if "thinking" in row and row["thinking"]:
            logger.info(f"  thinking[:150]: {str(row['thinking'])[:150]}")

    return True


def inspect_rl_swe():
    """Inspect Nemotron-Cascade-RL-SWE."""
    logger.info("=" * 60)
    logger.info("Inspecting nvidia/Nemotron-Cascade-RL-SWE")
    logger.info("=" * 60)

    try:
        ds = load_dataset("nvidia/Nemotron-Cascade-RL-SWE", split="train")
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        return False

    logger.info(f"Total rows: {len(ds)}")
    logger.info(f"Columns: {ds.column_names}")
    logger.info(f"Features: {ds.features}")

    # Inspect relevant_file_contents structure
    if "relevant_file_contents" in ds.column_names:
        rfc = ds[0].get("relevant_file_contents", [])
        logger.info(f"\nrelevant_file_contents structure:")
        logger.info(f"  type: {type(rfc)}, length: {len(rfc) if isinstance(rfc, list) else 'N/A'}")
        if isinstance(rfc, list) and len(rfc) > 0:
            first = rfc[0]
            logger.info(f"  first entry type: {type(first)}")
            if isinstance(first, dict):
                logger.info(f"  first entry keys: {list(first.keys())}")
                for k, v in first.items():
                    logger.info(f"    {k}: {str(v)[:100]}")

    # Print 3 sample rows
    logger.info("\nSample rows:")
    for i in range(min(3, len(ds))):
        row = ds[i]
        logger.info(f"\n--- Row {i} ---")
        logger.info(f"  instance_id: {row.get('instance_id', 'N/A')}")
        logger.info(f"  source: {row.get('source', 'N/A')}")
        prompt = row.get("prompt", "")
        logger.info(f"  prompt_len: {len(prompt)}")
        logger.info(f"  prompt[:200]: {prompt[:200]}")
        patch = row.get("golden_patch", "")
        logger.info(f"  golden_patch_len: {len(patch)}")
        logger.info(f"  golden_patch[:200]: {patch[:200]}")
        rfc = row.get("relevant_file_contents", [])
        logger.info(f"  relevant_file_contents: {len(rfc) if isinstance(rfc, list) else 'N/A'} files")

    return True


def main():
    ok = True
    ok = inspect_sft_swe() and ok
    ok = inspect_rl_swe() and ok

    if ok:
        logger.info("\nAll datasets inspected successfully")
    else:
        logger.error("\nSome datasets failed to load")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
