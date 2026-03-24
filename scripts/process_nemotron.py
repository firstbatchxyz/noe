#!/usr/bin/env python3
"""Process Nemotron-Cascade-SFT-SWE into role-aligned training data.

Uses the actual dataset schema: category column with values
"SWE Localization", "SWE Repair", "SWE TestGen".
"""

import argparse
import logging
import sys

from noe_train.data.nemotron_swe import (
    CATEGORY_LOCALIZATION,
    CATEGORY_TO_ROLE,
    derive_planner_from_localization,
    load_nemotron_swe,
    process_category,
)
from noe_train.data.role_dataset import build_role_datasets, verify_counts
from noe_train.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process Nemotron SWE data")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    logger.info("Loading Nemotron-Cascade-SFT-SWE...")
    subsets = load_nemotron_swe(cache_dir=args.cache_dir)

    samples_by_role = {
        "planner": [],
        "coder": [],
        "tester": [],
        "debugger": [],
    }

    # Process each category subset
    for category, ds in subsets.items():
        role = CATEGORY_TO_ROLE.get(category)
        if role is None:
            logger.warning(f"Unknown category: {category}, skipping {len(ds)} rows")
            continue

        logger.info(f"Processing '{category}' → {role} ({len(ds)} samples)")
        samples = process_category(ds, category)
        samples_by_role[role].extend(samples)
        logger.info(f"  Extracted {len(samples)} valid samples")

    # Derive planner data from localization subset
    if CATEGORY_LOCALIZATION in subsets:
        logger.info(f"\nDeriving planner data from localization samples...")
        planner_samples = derive_planner_from_localization(
            subsets[CATEGORY_LOCALIZATION],
            max_samples=8000,
        )
        samples_by_role["planner"].extend(planner_samples)
    else:
        logger.warning("No localization subset found — cannot derive planner data")

    # Report counts
    logger.info("\nFinal counts:")
    for role, samples in samples_by_role.items():
        logger.info(f"  {role}: {len(samples)} samples")

    # Build unified datasets
    dd = build_role_datasets(samples_by_role, output_dir=args.output_dir)
    verify_counts(dd)

    logger.info(f"Done. Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
