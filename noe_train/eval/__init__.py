"""Evaluation module."""

from noe_train.eval.ablations import ABLATION_CONFIGS, AblationConfig, AblationType
from noe_train.eval.role_metrics import (
    compute_efficiency,
    evaluate_coder,
    evaluate_debugger,
    evaluate_planner,
    evaluate_router,
    evaluate_tester,
)


def __getattr__(name):
    if name in ("SWEBenchResult", "SWEBenchSummary", "evaluate_swe_bench"):
        from noe_train.eval import swe_bench
        return getattr(swe_bench, name)
    raise AttributeError(f"module 'noe_train.eval' has no attribute {name!r}")


__all__ = [
    "ABLATION_CONFIGS",
    "AblationConfig",
    "AblationType",
    "SWEBenchResult",
    "SWEBenchSummary",
    "compute_efficiency",
    "evaluate_coder",
    "evaluate_debugger",
    "evaluate_planner",
    "evaluate_router",
    "evaluate_swe_bench",
    "evaluate_tester",
]
