#!/usr/bin/env python3
"""Inspect Nemotron datasets: verify counts, schemas, and sample data.

- Load nvidia/Nemotron-Cascade-SFT-SWE — verify ~92K/87K/32K split
- Load nvidia/Nemotron-Cascade-RL-SWE — verify ~110K rows
- Print column schemas, 5 example rows per subset
"""

import json
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

    # Check for task_type column to identify subsets
    if "task_type" in ds.column_names:
        for task_type in ds.unique("task_type"):
            subset = ds.filter(lambda x: x["task_type"] == task_type)
            logger.info(f"  {task_type}: {len(subset)} rows")
    else:
        logger.info("No 'task_type' column — inspecting raw structure")

    # Print 5 sample rows
    logger.info("\nSample rows:")
    for i in range(min(5, len(ds))):
        row = ds[i]
        logger.info(f"\n--- Row {i} ---")
        for key, value in row.items():
            val_str = str(value)[:200]
            logger.info(f"  {key}: {val_str}")

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

    # Print 5 sample rows
    logger.info("\nSample rows:")
    for i in range(min(5, len(ds))):
        row = ds[i]
        logger.info(f"\n--- Row {i} ---")
        for key, value in row.items():
            val_str = str(value)[:200]
            logger.info(f"  {key}: {val_str}")

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
