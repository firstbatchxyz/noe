"""Utilities module."""

from noe_train.utils.logging import init_wandb, log_metrics, setup_logging


def __getattr__(name):
    if name == "CheckpointManager":
        from noe_train.utils.checkpoint import CheckpointManager
        return CheckpointManager
    if name in ("get_device", "get_gpu_assignments", "log_gpu_status"):
        import importlib
        mod = importlib.import_module("noe_train.utils.gpu_plan")
        return getattr(mod, name)
    raise AttributeError(f"module 'noe_train.utils' has no attribute {name!r}")


__all__ = [
    "CheckpointManager",
    "get_device",
    "get_gpu_assignments",
    "init_wandb",
    "log_gpu_status",
    "log_metrics",
    "setup_logging",
]
