"""Per-role evaluation metrics.

Planner: file recall@k vs gold changed files
Coder: syntax-valid rate, compile rate on sandbox
Tester: generated tests that run without error
Debugger: correct file in top-3 (localization F1)
Router: recall@2, stop accuracy, ECE
Efficiency: solved-per-1k-tokens, solved-per-tool-call
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass()
class RoleMetrics:
    role: str
    total: int
    metrics: dict[str, float]


def evaluate_planner(
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    k: int = 3,
) -> RoleMetrics:
    """Planner: file recall@k vs gold changed files."""
    hits = 0
    total = 0

    for pred, g in zip(predictions, gold):
        gold_files = set(g.get("files_changed", []))
        pred_files = pred.get("files_to_touch", [])[:k]
        if gold_files:
            recall = len(set(pred_files) & gold_files) / len(gold_files)
            hits += recall
            total += 1

    return RoleMetrics(
        role="planner",
        total=total,
        metrics={f"file_recall@{k}": hits / max(total, 1)},
    )


def evaluate_coder(
    predictions: list[dict[str, Any]],
    sandbox_results: list[dict[str, Any]] | None = None,
) -> RoleMetrics:
    """Coder: syntax-valid rate, compile rate."""
    total = len(predictions)
    syntax_valid = 0
    compile_ok = 0

    for i, pred in enumerate(predictions):
        patch = pred.get("patch", "")
        # Syntax check: does it look like a valid diff?
        if patch.strip() and ("---" in patch or "+++" in patch or "@@" in patch):
            syntax_valid += 1

        if sandbox_results and i < len(sandbox_results):
            if sandbox_results[i].get("compile_ok", False):
                compile_ok += 1

    return RoleMetrics(
        role="coder",
        total=total,
        metrics={
            "syntax_valid_rate": syntax_valid / max(total, 1),
            "compile_rate": compile_ok / max(total, 1) if sandbox_results else 0.0,
        },
    )


def evaluate_tester(
    predictions: list[dict[str, Any]],
    execution_results: list[dict[str, Any]] | None = None,
) -> RoleMetrics:
    """Tester: generated tests that run without error."""
    total = len(predictions)
    runnable = 0

    if execution_results:
        for result in execution_results:
            if result.get("tests_passed", 0) > 0 or result.get("no_error", False):
                runnable += 1

    return RoleMetrics(
        role="tester",
        total=total,
        metrics={
            "runnable_rate": runnable / max(total, 1) if execution_results else 0.0,
        },
    )


def evaluate_debugger(
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    k: int = 3,
) -> RoleMetrics:
    """Debugger: correct file in top-k (localization)."""
    hits = 0
    total = 0

    for pred, g in zip(predictions, gold):
        gold_files = set(g.get("files_changed", []))
        pred_files = pred.get("suspect_files", [])[:k]
        if gold_files:
            hit = len(set(pred_files) & gold_files) > 0
            hits += int(hit)
            total += 1

    return RoleMetrics(
        role="debugger",
        total=total,
        metrics={f"file_hit@{k}": hits / max(total, 1)},
    )


def evaluate_router(
    decisions: list[dict[str, Any]],
    gold_labels: list[dict[str, Any]],
) -> RoleMetrics:
    """Router: expert recall@2, budget accuracy, stop accuracy."""
    expert_hits = 0
    budget_correct = 0
    stop_correct = 0
    total = len(decisions)

    for dec, gold in zip(decisions, gold_labels):
        # Expert recall@2
        pred_experts = dec.get("selected_experts", [])[:2]
        gold_experts = set(gold.get("useful_experts", []))
        if gold_experts:
            recall = len(set(pred_experts) & gold_experts) / len(gold_experts)
            expert_hits += recall

        # Budget accuracy
        if dec.get("budget_level") == gold.get("optimal_budget"):
            budget_correct += 1

        # Stop accuracy
        if dec.get("should_stop") == gold.get("should_stop"):
            stop_correct += 1

    return RoleMetrics(
        role="router",
        total=total,
        metrics={
            "expert_recall@2": expert_hits / max(total, 1),
            "budget_accuracy": budget_correct / max(total, 1),
            "stop_accuracy": stop_correct / max(total, 1),
        },
    )


def compute_efficiency(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Efficiency metrics: solved-per-1k-tokens, solved-per-tool-call."""
    total_solved = sum(1 for r in results if r.get("resolved", False))
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    total_calls = sum(r.get("total_tool_calls", 0) for r in results)

    return {
        "solved_per_1k_tokens": total_solved / max(total_tokens / 1000, 1),
        "solved_per_tool_call": total_solved / max(total_calls, 1),
        "total_solved": total_solved,
        "total_tokens": total_tokens,
        "total_calls": total_calls,
    }
