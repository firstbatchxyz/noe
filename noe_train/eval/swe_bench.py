"""SWE-bench Verified evaluation — primary evaluation metric.

500 instances, NEVER train on this.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass()
class SWEBenchResult:
    instance_id: str
    resolved: bool
    patch: str
    fail_to_pass: list[str]
    pass_to_pass_maintained: bool
    error: str = ""


@dataclass()
class SWEBenchSummary:
    total: int
    resolved: int
    resolve_rate: float
    pass_to_pass_rate: float
    avg_patch_size: float
    results: list[SWEBenchResult]


def evaluate_swe_bench(
    predictions: list[dict[str, Any]],
    verified_instances: list[dict[str, Any]],
    sandbox_factory: Any = None,
) -> SWEBenchSummary:
    """Evaluate predictions against SWE-bench Verified.

    Args:
        predictions: list of {"instance_id": str, "patch": str}
        verified_instances: SWE-bench Verified dataset rows
        sandbox_factory: callable to create sandbox for test execution
    """
    instance_map = {inst["instance_id"]: inst for inst in verified_instances}

    results = []
    resolved_count = 0
    p2p_count = 0
    total_patch_lines = 0

    for pred in predictions:
        instance_id = pred["instance_id"]
        patch = pred.get("patch", "")
        instance = instance_map.get(instance_id)

        if instance is None:
            results.append(SWEBenchResult(
                instance_id=instance_id,
                resolved=False,
                patch=patch,
                fail_to_pass=[],
                pass_to_pass_maintained=False,
                error="instance not found in verified set",
            ))
            continue

        # Count patch lines
        total_patch_lines += len(patch.split("\n"))

        # Execute evaluation (requires sandbox)
        if sandbox_factory is not None:
            result = _evaluate_instance(instance, patch, sandbox_factory)
        else:
            result = SWEBenchResult(
                instance_id=instance_id,
                resolved=False,
                patch=patch,
                fail_to_pass=instance.get("FAIL_TO_PASS", []),
                pass_to_pass_maintained=False,
                error="no sandbox available for evaluation",
            )

        results.append(result)
        if result.resolved:
            resolved_count += 1
        if result.pass_to_pass_maintained:
            p2p_count += 1

    total = len(predictions)
    return SWEBenchSummary(
        total=total,
        resolved=resolved_count,
        resolve_rate=resolved_count / max(total, 1),
        pass_to_pass_rate=p2p_count / max(total, 1),
        avg_patch_size=total_patch_lines / max(total, 1),
        results=results,
    )


def _evaluate_instance(
    instance: dict[str, Any],
    patch: str,
    sandbox_factory: Any,
) -> SWEBenchResult:
    """Evaluate a single instance in sandbox."""
    instance_id = instance["instance_id"]
    fail_to_pass = instance.get("FAIL_TO_PASS", [])

    try:
        # Create sandbox, apply patch, run tests
        sandbox = sandbox_factory(instance)
        success, msg = sandbox.apply_and_test(patch)

        return SWEBenchResult(
            instance_id=instance_id,
            resolved=success,
            patch=patch,
            fail_to_pass=fail_to_pass,
            pass_to_pass_maintained=success,  # simplified
        )
    except Exception as e:
        return SWEBenchResult(
            instance_id=instance_id,
            resolved=False,
            patch=patch,
            fail_to_pass=fail_to_pass,
            pass_to_pass_maintained=False,
            error=str(e),
        )


def save_results(summary: SWEBenchSummary, output_path: str | Path) -> None:
    """Save evaluation results to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total": summary.total,
        "resolved": summary.resolved,
        "resolve_rate": summary.resolve_rate,
        "pass_to_pass_rate": summary.pass_to_pass_rate,
        "avg_patch_size": summary.avg_patch_size,
        "results": [
            {
                "instance_id": r.instance_id,
                "resolved": r.resolved,
                "pass_to_pass_maintained": r.pass_to_pass_maintained,
                "error": r.error,
            }
            for r in summary.results
        ],
    }

    path.write_text(json.dumps(data, indent=2))
    logger.info(f"Results saved to {path}")
