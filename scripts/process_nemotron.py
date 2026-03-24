#!/usr/bin/env python3
"""Process Nemotron-Cascade-SFT-SWE into role-aligned training data."""

import argparse
import logging
import sys

from noe_train.data.nemotron_swe import (
    load_nemotron_swe,
    process_localization,
    process_repair,
    process_test_generation,
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

    # Process subsets based on available keys
    for key, ds in subsets.items():
        key_lower = key.lower()
        if "local" in key_lower or "loc" in key_lower:
            logger.info(f"Processing {key} → debugger ({len(ds)} samples)")
            samples_by_role["debugger"].extend(process_localization(ds))
        elif "repair" in key_lower or "fix" in key_lower or "patch" in key_lower:
            logger.info(f"Processing {key} → coder ({len(ds)} samples)")
            samples_by_role["coder"].extend(process_repair(ds))
        elif "test" in key_lower or "gen" in key_lower:
            logger.info(f"Processing {key} → tester ({len(ds)} samples)")
            samples_by_role["tester"].extend(process_test_generation(ds))
        else:
            logger.warning(f"Unknown subset key: {key}, processing as debugger")
            samples_by_role["debugger"].extend(process_localization(ds))

    # Report counts
    for role, samples in samples_by_role.items():
        logger.info(f"  {role}: {len(samples)} samples")

    # Build unified datasets
    dd = build_role_datasets(samples_by_role, output_dir=args.output_dir)
    verify_counts(dd)

    logger.info(f"Done. Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
