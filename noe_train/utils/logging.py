"""Logging configuration with wandb integration."""

from __future__ import annotations

import logging
import sys
from typing import Any


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
) -> None:
    """Configure logging for noe_train."""
    root = logging.getLogger("noe_train")
    root.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root.addHandler(fh)


def init_wandb(
    project: str = "noe-train",
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    """Initialize wandb run."""
    try:
        import wandb
        run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
        )
        return run
    except ImportError:
        logging.getLogger("noe_train").warning("wandb not installed, skipping init")
        return None


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to wandb if available."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass
