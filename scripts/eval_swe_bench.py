#!/usr/bin/env python3
"""Evaluate on SWE-bench Verified (500 instances)."""

import argparse
import json
import logging
import sys
from pathlib import Path

from noe_train.data.rl_tasks import load_swebench_verified
from noe_train.eval.swe_bench import evaluate_swe_bench, save_results
from noe_train.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on SWE-bench Verified")
    parser.add_argument("--predictions", type=str, required=True, help="JSON file with predictions")
    parser.add_argument("--output", type=str, default="eval_results/swe_bench.json")
    args = parser.parse_args()

    # Load predictions
    with open(args.predictions) as f:
        predictions = json.load(f)
    logger.info(f"Loaded {len(predictions)} predictions")

    # Load SWE-bench Verified
    verified = load_swebench_verified()
    verified_list = [verified[i] for i in range(len(verified))]

    # Evaluate
    summary = evaluate_swe_bench(predictions, verified_list)

    logger.info(f"Results: {summary.resolved}/{summary.total} resolved ({summary.resolve_rate:.1%})")
    logger.info(f"Pass-to-pass rate: {summary.pass_to_pass_rate:.1%}")
    logger.info(f"Avg patch size: {summary.avg_patch_size:.1f} lines")

    # Save
    save_results(summary, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
