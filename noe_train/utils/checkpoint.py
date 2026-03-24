"""Checkpoint management.

Structure:
checkpoint/
├── base_model/           # Qwen2.5-Coder-3B-Instruct (ref only)
├── lora_{planner,coder,tester,debugger}/
├── router/
├── critic/
├── optimizer_states/
└── metadata.json         # step, stage, metrics

Save every 1K episodes. Keep last 5 + best by eval metric.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

MAX_KEPT = 5


@dataclass()
class CheckpointMeta:
    step: int
    stage: str
    metrics: dict[str, float]
    path: str


class CheckpointManager:
    """Manages checkpoint saving, loading, and pruning."""

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._history: list[CheckpointMeta] = []
        self._best_metric: float = -float("inf")
        self._best_path: str | None = None

    def save(
        self,
        step: int,
        stage: str,
        models: dict[str, Any],
        optimizer_states: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        eval_metric_key: str = "resolve_rate",
    ) -> Path:
        """Save a checkpoint."""
        ckpt_dir = self.root / f"step_{step:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        for name, model in models.items():
            model_dir = ckpt_dir / name
            model_dir.mkdir(exist_ok=True)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(str(model_dir))
            elif isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), model_dir / "model.pt")

        # Save optimizer states
        if optimizer_states:
            opt_dir = ckpt_dir / "optimizer_states"
            opt_dir.mkdir(exist_ok=True)
            for name, state in optimizer_states.items():
                torch.save(state, opt_dir / f"{name}.pt")

        # Save metadata
        meta = {
            "step": step,
            "stage": stage,
            "metrics": metrics or {},
        }
        (ckpt_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        entry = CheckpointMeta(
            step=step, stage=stage, metrics=metrics or {}, path=str(ckpt_dir)
        )
        self._history.append(entry)

        # Track best
        if metrics and eval_metric_key in metrics:
            val = metrics[eval_metric_key]
            if val > self._best_metric:
                self._best_metric = val
                self._best_path = str(ckpt_dir)
                # Symlink best
                best_link = self.root / "best"
                if best_link.exists() or best_link.is_symlink():
                    best_link.unlink()
                best_link.symlink_to(ckpt_dir.name)

        # Prune old checkpoints (keep last MAX_KEPT + best)
        self._prune()

        logger.info(f"Checkpoint saved: {ckpt_dir}")
        return ckpt_dir

    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        """Load checkpoint metadata. Returns metadata dict."""
        if path is None:
            # Load latest
            ckpts = sorted(self.root.glob("step_*"), key=lambda p: p.name)
            if not ckpts:
                raise FileNotFoundError("No checkpoints found")
            path = ckpts[-1]

        path = Path(path)
        meta_path = path / "metadata.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}

    def load_model(self, path: str | Path, name: str) -> Path:
        """Get path to a model within a checkpoint."""
        return Path(path) / name

    @property
    def best_path(self) -> str | None:
        return self._best_path

    def _prune(self) -> None:
        """Keep only last MAX_KEPT checkpoints + best."""
        if len(self._history) <= MAX_KEPT:
            return

        to_keep = set()
        # Keep last MAX_KEPT
        for entry in self._history[-MAX_KEPT:]:
            to_keep.add(entry.path)
        # Keep best
        if self._best_path:
            to_keep.add(self._best_path)

        for entry in self._history:
            if entry.path not in to_keep:
                p = Path(entry.path)
                if p.exists():
                    shutil.rmtree(p)
                    logger.info(f"Pruned checkpoint: {p}")

        self._history = [e for e in self._history if e.path in to_keep]
